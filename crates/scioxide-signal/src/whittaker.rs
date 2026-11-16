//! Whittaker–Henderson smoothing with a difference penalty.
use crate::SignalExecutionOptions;
use crate::util::{
    axis_mismatch, capacity, exact_len, finite, inconsistent, invalid,
    scalar_from_usize_rounded as scalar,
};
use crate::util::{check, inputs, size, zeros};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::Triangle;
use scioxide_core::{
    Capacity, ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_linalg::{
    BandedCholeskyDecomposition, HermitianBandMatrixView, LinalgExecutionOptions,
    LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut, StructuredOperation,
    StructuredWorkspaceRequest,
};

/// Conditioning of the linear system `(W + λ DᵀD) z = W y` of a
/// [`WhittakerHendersonPlan`].
///
/// It measures the sensitivity of the solution to perturbations. It does not
/// depend on the data and is computed once, when the plan is prepared.
#[derive(Clone, Copy, Debug)]
pub enum SmoothingConditioning<S: RealScalar> {
    /// `λ = 0`: the smoother returns the input unchanged and solves no system.
    Identity,
    /// Lower bound on the reciprocal spectral condition number when every weight
    /// is positive:
    ///
    /// ```text
    /// min(w) / (max(w) + λ (Σⱼ |δⱼ|)²)
    /// ```
    ///
    /// where `δⱼ` are the coefficients of the order-`d` difference. The bound holds
    /// in exact arithmetic and ignores the rounding of its own evaluation.
    ReciprocalSpectralBound(S),
    /// Hager–Higham estimate of the reciprocal one-norm condition number, used
    /// when some weights are zero.
    OneNormEstimate(scioxide_linalg::ConditionEstimate<S>),
}

/// Diagnostics of one smoothing solve.
///
/// The values describe the computed solution. They are not error bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothingDiagnostics<S: RealScalar> {
    data_fidelity: S,
    penalty: S,
    scaled_optimality_residual: S,
}
impl<S: RealScalar> SmoothingDiagnostics<S> {
    /// Returns the data term `Σᵢ wᵢ |zᵢ − yᵢ|²`, in the units of the given weights.
    pub const fn data_fidelity(self) -> S {
        self.data_fidelity
    }
    /// Returns the penalty term `λ ‖Dᵈ z‖²`, in the units of the given weights.
    pub const fn penalty(self) -> S {
        self.penalty
    }
    /// Returns the componentwise backward error `maxᵢ |Az − b|ᵢ / (|A||z| + |b|)ᵢ`
    /// of the solution `z` of `A z = b`, with `A = W + λ DᵀD` and `b = W y`.
    ///
    /// The magnitude of a complex value is the sum of the absolute values of its
    /// real and imaginary parts.
    pub const fn scaled_optimality_residual(self) -> S {
        self.scaled_optimality_residual
    }
}
/// Storage that a [`WhittakerHendersonPlan`] needs to smooth a number of records.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct WhittakerRequirements {
    values: usize,
    solver: LinalgRequirements,
}
impl WhittakerRequirements {
    /// Returns the number of samples in each of the two staging buffers.
    pub const fn values_len(&self) -> usize {
        self.values
    }
}
/// Workspace of a [`WhittakerHendersonPlan`].
///
/// One workspace can serve several plans: reserve it for the largest
/// requirement. Smoothing does not grow it.
#[derive(Clone, Debug)]
pub struct WhittakerWorkspace<T: NumericScalar> {
    rhs: Vec<T>,
    solution: Vec<T>,
    solver: LinalgWorkspace<'static, T>,
}
impl<T: NumericScalar> Default for WhittakerWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> WhittakerWorkspace<T> {
    /// Returns an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            rhs: Vec::new(),
            solution: Vec::new(),
            solver: LinalgWorkspace::new(),
        }
    }
    /// Grows the workspace to meet `requirements`, keeping any larger storage.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the solver storage exceeds its byte limit.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &WhittakerRequirements,
    ) -> Result<(), SciOxideError> {
        for buffer in [&mut self.rhs, &mut self.solution] {
            if buffer.len() < requirements.values {
                scioxide_core::backend::try_reserve(
                    buffer,
                    requirements.values - buffer.len(),
                    scioxide_core::quantity!("whittaker_workspace"),
                )?;
                buffer.resize(requirements.values, T::ZERO);
            }
        }
        self.solver.try_reserve(&requirements.solver)
    }
    /// Reports whether the workspace meets `requirements`, without growing it.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the solver storage size overflows `usize`.
    /// - `ResourceLimit` if the solver storage exceeds its byte limit.
    pub fn check(&self, requirements: &WhittakerRequirements) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.rhs.len() >= requirements.values
                && self.solution.len() >= requirements.values
                && self.solver.check(&requirements.solver)? == Capacity::Sufficient
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
}
/// Whittaker–Henderson smoother of uniformly spaced samples.
///
/// The smoothed record `z` minimizes
///
/// ```text
/// S(z) = Σᵢ wᵢ |zᵢ − yᵢ|² + λ ‖Dᵈ z‖²
/// ```
///
/// over `n` samples `y` with weights `wᵢ ≥ 0`, penalty `λ ≥ 0` and the
/// `(n − d) × n` forward-difference matrix `Dᵈ` of order `d`. The minimizer
/// solves `(W + λ DᵀD) z = W y` with `W = diag(w)` and `D = Dᵈ`. A zero weight marks a
/// missing observation, whose value is interpolated by the penalty.
///
/// The plan factors the band matrix `W + λ DᵀD` of half-bandwidth `d` once by
/// banded Cholesky factorization, in `O(n d²)` operations and `O(n d)` storage.
/// Each record then costs `O(n d)` operations, including its
/// [`SmoothingDiagnostics`]. The data may be real or complex. The weights and `λ`
/// are real. [`WhittakerHendersonPlan::conditioning`] reports the conditioning
/// of the system.
///
/// # References
///
/// E. T. Whittaker, "On a new method of graduation", *Proc. Edinburgh Math.
/// Soc.* 41, 63–75, 1923.
///
/// P. H. C. Eilers, "A perfect smoother", *Anal. Chem.* 75(14), 3631–3636, 2003.
#[derive(Clone, Debug)]
pub struct WhittakerHendersonPlan<T: NumericScalar> {
    n: usize,
    order: usize,
    weights: Vec<T::Real>,
    uniform_weight: T::Real,
    lambda: T::Real,
    scale: T::Real,
    differences: Vec<T::Real>,
    band: Vec<T>,
    factor: Option<BandedCholeskyDecomposition<T>>,
    conditioning: SmoothingConditioning<T::Real>,
    diagnostic_range: Option<T::Real>,
    options: SignalExecutionOptions,
}
impl<T: NumericScalar> WhittakerHendersonPlan<T> {
    /// Prepares a smoother of `n` samples with difference order `order` and
    /// penalty `lambda`.
    ///
    /// `weights` of `None` means unit weights. A unique minimizer needs at least
    /// `order` positive weights, and every weight positive when `lambda` is zero.
    /// The execution policy of `options` applies to the factorization and to
    /// every solve. Runs [`WhittakerHendersonPlan::build`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// ```
    /// use scioxide_signal::{SignalExecutionOptions, SmoothingConditioning, WhittakerHendersonPlan};
    /// let options = SignalExecutionOptions::new();
    /// let plan = WhittakerHendersonPlan::<f64>::new(5, 2, 10., None, options)?;
    /// let (smoothed, diagnostics) = plan.smooth(&[0., 1., -1., 2., 0.])?;
    /// assert_eq!(smoothed.len(), 5);
    /// // The banded Cholesky solve is backward stable.
    /// assert!(diagnostics.scaled_optimality_residual() < 1e-12);
    /// assert!(matches!(
    ///     plan.conditioning(),
    ///     SmoothingConditioning::ReciprocalSpectralBound(_)
    /// ));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`WhittakerHendersonPlan::build`], except `ExecutionStopped`.
    pub fn new(
        n: usize,
        order: usize,
        lambda: T::Real,
        weights: Option<&[T::Real]>,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(n, order, lambda, weights, options, None)
    }
    /// Prepares a smoother like [`WhittakerHendersonPlan::new`], factoring under
    /// an execution context.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n < 2`, `order` is not in `1..n`, `lambda` or a
    ///   weight is negative, or the weights do not determine a unique minimizer.
    /// - `DimensionMismatch` if `weights` does not have `n` entries.
    /// - `NonFinite` if `lambda` or a weight is NaN or infinite.
    /// - `NumericalInstability` if a positive weight or `lambda` underflows when
    ///   the weights are scaled by their maximum, or a positive weight `wᵢ` is at
    ///   most `ε λ (DᵀD)ᵢᵢ` and so is lost in rounding (`IllConditioned`).
    /// - `SizeOverflow` if the band storage overflows `usize`.
    /// - `AllocationFailure` if the band or its factor cannot be allocated.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    ///
    /// Errors of the banded Cholesky factorization propagate.
    pub fn build(
        n: usize,
        order: usize,
        lambda: T::Real,
        weights: Option<&[T::Real]>,
        options: SignalExecutionOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(n, order, lambda, weights, options, Some(context))
    }
    /// Validates and factors the smoothing system; `None` factors with the
    /// sequential native context.
    fn prepare(
        n: usize,
        order: usize,
        lambda: T::Real,
        weights: Option<&[T::Real]>,
        options: SignalExecutionOptions,
        context: Option<ExecutionContext<'_>>,
    ) -> Result<Self, SciOxideError> {
        if n < 2 {
            return Err(invalid(
                scioxide_core::argument!("n"),
                "record length must be at least two",
            ));
        }
        if order == 0 || order >= n {
            return Err(invalid(
                scioxide_core::argument!("order"),
                "difference order must satisfy 1 <= order < n",
            ));
        }
        crate::util::check_real_slice_finite(&[lambda])?;
        if lambda < T::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("lambda"),
                "lambda must be nonnegative",
            ));
        }
        let band_len = size(n, order + 1)?;
        size(band_len, core::mem::size_of::<T>())?;
        if let Some(w) = weights {
            exact_len(n, w.len())?;
            inputs(w)?;
        }
        let weight = |i| weights.map_or(T::Real::ONE, |w| w[i]);
        let (positive, scale, minimum_weight, maximum_weight) = match weights {
            Some(weights) => weights.iter().try_fold(
                (0, lambda, T::Real::MAX, T::Real::ZERO),
                |(positive, scale, minimum, maximum), &w| {
                    if w < T::Real::ZERO {
                        return Err(invalid(
                            scioxide_core::argument!("weights"),
                            "weights must be nonnegative",
                        ));
                    }
                    Ok((
                        positive + usize::from(w > T::Real::ZERO),
                        scale.max(w),
                        minimum.min(w),
                        maximum.max(w),
                    ))
                },
            )?,
            None => (n, lambda.max(T::Real::ONE), T::Real::ONE, T::Real::ONE),
        };
        if (lambda == T::Real::ZERO && positive != n) || positive < order {
            return Err(inconsistent(
                scioxide_core::argument!("weights"),
                "weights and lambda do not determine a unique smoothing",
            ));
        }
        let execution = context.unwrap_or(ExecutionContext::sequential_native());
        execution.check()?;
        if lambda == T::Real::ZERO {
            return Ok(Self {
                n,
                order,
                weights: Vec::new(),
                uniform_weight: T::Real::ONE,
                lambda,
                scale: T::Real::ONE,
                differences: Vec::new(),
                band: Vec::new(),
                factor: None,
                conditioning: SmoothingConditioning::Identity,
                diagnostic_range: None,
                options,
            });
        }
        let uniform_weight = T::Real::ONE / scale;
        let mut normalized = if weights.is_some() {
            zeros::<T::Real>(n)?
        } else {
            Vec::new()
        };
        for (i, w) in normalized.iter_mut().enumerate() {
            *w = weight(i) / scale;
            if *w == T::Real::ZERO && weight(i) > T::Real::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "smoothing weights lost to scaling",
                ));
            }
        }
        let normalized_weight = |i| {
            if weights.is_none() {
                uniform_weight
            } else {
                normalized[i]
            }
        };
        let normalized_lambda = lambda / scale;
        if normalized_lambda == T::Real::ZERO && lambda > T::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "smoothing penalty lost to scaling",
            ));
        }
        let differences = scioxide_interp::difference_coefficients::<T::Real>(order)?;
        let mut band = zeros::<T>(band_len)?;
        // Interior Gram entries are identical shifts of one filter autocorrelation.
        // Only the at-most-order boundary columns need truncated reductions.
        for offset in 0..=order {
            execution.check()?;
            let last = order - offset;
            let entry = |first: usize, end: usize| {
                (first..=end).rev().try_fold(T::Real::ZERO, |sum, j| {
                    finite(
                        sum + finite(normalized_lambda * differences[j] * differences[j + offset])?,
                    )
                })
            };
            let interior = entry(0, last)?;
            for i in 0..n - offset {
                if i.is_multiple_of(256) {
                    execution.check()?;
                }
                let first = i.saturating_sub(n - order - 1);
                let end = i.min(last);
                let value = if first == 0 && end == last {
                    interior
                } else {
                    entry(first, end)?
                };
                band[i * (order + 1) + offset] = T::from_real(value);
            }
        }
        for i in 0..n {
            let w = normalized_weight(i);
            let diagonal = &mut band[i * (order + 1)];
            let penalty = diagonal.real_component();
            if w > T::Real::ZERO && (w <= T::Real::EPSILON * penalty || penalty + w == penalty) {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::IllConditioned,
                    "smoothing fidelity is lost at the scalar precision",
                ));
            }
            *diagonal = check(*diagonal + T::from_real(w))?;
        }
        let view = HermitianBandMatrixView::new(n, order, Triangle::Lower, &band)?;
        let mut factor = BandedCholeskyDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        let linalg = linalg_options(options);
        if let Some(context) = context {
            BandedCholeskyDecomposition::factor_into(
                view,
                &linalg,
                &mut factor,
                &mut workspace,
                context,
            )?;
        } else {
            factor = BandedCholeskyDecomposition::factor_with_options(view, &linalg)?;
        }
        let minimum = minimum_weight / scale;
        let maximum = maximum_weight / scale;
        let magnitude = differences
            .iter()
            .copied()
            .fold(T::Real::ZERO, |a, c| a + c.abs());
        let upper = maximum + normalized_lambda * magnitude * magnitude;
        let conditioning = if positive == n {
            let reciprocal = if upper.is_finite() {
                minimum / upper
            } else {
                Scaled::from_value(minimum)?
                    .div(
                        Scaled::from_value(maximum)?.add(
                            Scaled::from_value(normalized_lambda)?
                                .mul(Scaled::from_value(magnitude)?)?
                                .mul(Scaled::from_value(magnitude)?)?,
                        )?,
                    )?
                    .round()?
            };
            SmoothingConditioning::ReciprocalSpectralBound(reciprocal)
        } else {
            SmoothingConditioning::OneNormEstimate(factor.condition_estimate_scoped(
                &linalg,
                &mut workspace,
                execution,
            )?)
        };
        // U^16 <= MAX. With counts, nonzero components and coefficients in
        // [1/U,U], even squared stencil sums and original-unit objectives stay
        // inside range with several powers of headroom. Nonzero rounded short
        // differences have an epsilon-scaled grid far above sqrt(MIN_POSITIVE).
        let bound = T::Real::MAX.sqrt().sqrt().sqrt().sqrt() / scalar::<T::Real>(2)?;
        let lower = T::Real::ONE / bound;
        // Binomial difference coefficients have magnitude >= 1. Each nonzero
        // off-diagonal Gram sum has one sign, so lambda also bounds it below.
        let diagnostic_range = (scalar::<T::Real>(n)? <= bound
            && scale >= lower
            && scale <= bound
            && normalized_lambda >= lower
            && minimum >= lower
            && magnitude <= bound / scalar::<T::Real>(4)?
            && upper <= bound / scalar::<T::Real>(4)?)
        .then_some(bound);
        Ok(Self {
            n,
            order,
            weights: normalized,
            uniform_weight,
            lambda: normalized_lambda,
            scale,
            differences,
            band,
            factor: Some(factor),
            conditioning,
            diagnostic_range,
            options,
        })
    }
    /// Returns the execution options.
    pub const fn options(&self) -> SignalExecutionOptions {
        self.options
    }
    /// Returns the conditioning of the system, computed at preparation.
    pub const fn conditioning(&self) -> SmoothingConditioning<T::Real> {
        self.conditioning
    }
    /// Returns the record length `n`.
    pub fn input_len(&self) -> usize {
        self.n
    }
    /// Returns the difference order `d`.
    pub fn difference_order(&self) -> usize {
        self.order
    }
    /// Returns the storage needed to smooth `lanes` records at once.
    ///
    /// Use one lane for [`WhittakerHendersonPlan::smooth_into`] and the number of
    /// lanes along the axis for [`WhittakerHendersonPlan::smooth_axis_into`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the solver storage exceeds its byte limit.
    pub fn workspace_requirements(
        &self,
        lanes: usize,
    ) -> Result<WhittakerRequirements, SciOxideError> {
        Ok(WhittakerRequirements {
            values: size(self.n, lanes)?,
            solver: BandedCholeskyDecomposition::<T>::workspace_requirements(
                &StructuredWorkspaceRequest::solve_matrix(
                    self.n,
                    lanes,
                    &linalg_options(self.options),
                ),
            )?,
        })
    }
    /// Allocates a workspace for `lanes` records.
    ///
    /// # Errors
    ///
    /// The errors of [`WhittakerHendersonPlan::workspace_requirements`], and
    /// `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, lanes: usize) -> Result<WhittakerWorkspace<T>, SciOxideError> {
        let mut workspace = WhittakerWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(lanes)?)?;
        Ok(workspace)
    }
    /// Smooths one record and returns the result with its diagnostics.
    ///
    /// # Errors
    ///
    /// The errors of [`WhittakerHendersonPlan::smooth_into`], and
    /// `AllocationFailure` if the result or workspace cannot be allocated.
    pub fn smooth(
        &self,
        input: &[T],
    ) -> Result<(Vec<T>, SmoothingDiagnostics<T::Real>), SciOxideError> {
        let mut output = zeros(self.n)?;
        let diagnostics = self.smooth_into(
            input,
            &mut output,
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )?;
        Ok((output, diagnostics))
    }
    /// Smooths one record into the first `n` entries of `output` and returns
    /// the diagnostics of the solve.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `n` samples, `output` is
    ///   shorter than `n`, or the workspace is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a weighted sample overflows or underflows to
    ///   zero, or the solution or a diagnostic overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn smooth_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut WhittakerWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<SmoothingDiagnostics<T::Real>, SciOxideError> {
        exact_len(self.n, input.len())?;
        capacity(self.n, output.len())?;
        self.capacity(workspace, 1)?;
        inputs(input)?;
        for (i, (rhs, &x)) in workspace.rhs.iter_mut().zip(input).enumerate() {
            *rhs = self.rhs_value(x, i)?;
        }
        self.solve(workspace, 1, execution)?;
        let diagnostics = self.diagnostics(input, &workspace.solution[..self.n])?;
        execution.check()?;
        output[..self.n].copy_from_slice(&workspace.solution[..self.n]);
        Ok(diagnostics)
    }
    #[inline]
    fn weight(&self, i: usize) -> T::Real {
        if self.weights.is_empty() {
            self.uniform_weight
        } else {
            self.weights[i]
        }
    }
    fn rhs_value(&self, x: T, i: usize) -> Result<T, SciOxideError> {
        if self.factor.is_none() {
            Ok(x)
        } else {
            let weight = self.weight(i);
            let rhs = check(x * T::from_real(weight))?;
            if weight > T::Real::ZERO
                && [
                    (x.real_component(), rhs.real_component()),
                    (x.imaginary_component(), rhs.imaginary_component()),
                ]
                .into_iter()
                .any(|(input, output)| input != T::Real::ZERO && output == T::Real::ZERO)
            {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "a nonzero weighted sample is not representable",
                ));
            }
            Ok(rhs)
        }
    }
    /// Reject staging or (for penalized plans with work) solver storage below
    /// the requirements of `lanes` right-hand sides.
    fn capacity(
        &self,
        workspace: &WhittakerWorkspace<T>,
        lanes: usize,
    ) -> Result<(), SciOxideError> {
        let requirements = self.workspace_requirements(lanes)?;
        capacity(requirements.values, workspace.rhs.len())?;
        capacity(requirements.values, workspace.solution.len())?;
        if self.factor.is_some()
            && lanes != 0
            && workspace.solver.check(&requirements.solver)? != Capacity::Sufficient
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(requirements.solver.scalar_len()?),
                scioxide_core::Dimensions::one(
                    workspace.solver.retained_bytes()? / core::mem::size_of::<T>().max(1),
                ),
            ));
        }
        Ok(())
    }
    fn solve(
        &self,
        workspace: &mut WhittakerWorkspace<T>,
        lanes: usize,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let count = size(self.n, lanes)?;
        execution.check()?;
        let Some(factor) = &self.factor else {
            workspace.solution[..count].copy_from_slice(&workspace.rhs[..count]);
            return Ok(());
        };
        let rhs =
            MatrixView::from_slice_with_strides(self.n, lanes, &workspace.rhs[..count], 1, self.n)?;
        let output = MatrixViewMut::from_slice_with_strides(
            self.n,
            lanes,
            &mut workspace.solution[..count],
            1,
            self.n,
        )?;
        factor
            .solve_matrix_into(
                StructuredOperation::Normal,
                rhs,
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
                    .with_execution_options(linalg_options(self.options)),
                output,
                &mut workspace.solver,
                execution,
            )
            .and_then(|report| {
                if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                    Ok(())
                } else {
                    Err(scioxide_core::SciOxideError::singular(None))
                }
            })
    }
    fn diagnostics(
        &self,
        input: &[T],
        solution: &[T],
    ) -> Result<SmoothingDiagnostics<T::Real>, SciOxideError> {
        if self.factor.is_none() {
            return Ok(SmoothingDiagnostics {
                data_fidelity: T::Real::ZERO,
                penalty: T::Real::ZERO,
                scaled_optimality_residual: T::Real::ZERO,
            });
        }
        self.working_diagnostics(input, solution)
            .map_or_else(|| self.extended_diagnostics(input, solution), Ok)
    }

    // One row traversal; real jobs instantiate no imaginary reductions. None
    // requests the same independent diagnostics in an extended exponent range.
    fn working_diagnostics(
        &self,
        input: &[T],
        solution: &[T],
    ) -> Option<SmoothingDiagnostics<T::Real>> {
        let bounded = self.diagnostic_range.is_some_and(|bound| {
            let lower = T::Real::ONE / bound;
            let ordinary =
                |v: T::Real| v == T::Real::ZERO || (v.abs() >= lower && v.abs() <= bound);
            input.iter().chain(solution).all(|&v| {
                ordinary(v.real_component()) && (T::IS_REAL || ordinary(v.imaginary_component()))
            })
        });
        match (self.order, bounded) {
            (2, true) => self.working_diagnostics_for_order::<2, true>(input, solution),
            (2, false) => self.working_diagnostics_for_order::<2, false>(input, solution),
            (_, true) => self.working_diagnostics_for_order::<0, true>(input, solution),
            (_, false) => self.working_diagnostics_for_order::<0, false>(input, solution),
        }
    }

    fn working_diagnostics_for_order<const ORDER: usize, const BOUNDED: bool>(
        &self,
        input: &[T],
        solution: &[T],
    ) -> Option<SmoothingDiagnostics<T::Real>> {
        use scioxide_core::CompensatedSum;
        let order = if ORDER == 0 { self.order } else { ORDER };
        let components = if T::IS_REAL { 1 } else { 2 };
        let component = |x: T, c| {
            if c == 0 {
                x.real_component()
            } else {
                x.imaginary_component()
            }
        };
        let mut fidelity = CompensatedSum::ZERO;
        let mut penalty = CompensatedSum::ZERO;
        let mut residual = T::Real::ZERO;
        for (i, (&x, &z)) in input.iter().zip(solution).enumerate() {
            let mut numerator = T::Real::ZERO;
            let mut denominator = T::Real::ZERO;
            for c in 0..components {
                let delta = component(x, c) - component(z, c);
                fidelity
                    .add(diagnostic_product::<_, BOUNDED>(
                        diagnostic_product::<_, BOUNDED>(delta, delta)?,
                        self.weight(i),
                    )?)
                    .ok()?;
                if i < self.n - order {
                    let value = self.differences[..order + 1]
                        .iter()
                        .zip(&solution[i..])
                        .fold(T::Real::ZERO, |sum, (&coefficient, &value)| {
                            sum + coefficient * component(value, c)
                        });
                    penalty
                        .add(diagnostic_product::<_, BOUNDED>(
                            diagnostic_product::<_, BOUNDED>(value, value)?,
                            self.lambda,
                        )?)
                        .ok()?;
                }
                let mut sum = T::Real::ZERO;
                for (j, &value) in solution
                    .iter()
                    .enumerate()
                    .take((i + order + 1).min(self.n))
                    .skip(i.saturating_sub(order))
                {
                    let a = self.band[i.min(j) * (order + 1) + i.abs_diff(j)].real_component();
                    let product = diagnostic_product::<_, BOUNDED>(a, component(value, c))?;
                    sum += product;
                    denominator += product.abs();
                }
                let rhs = diagnostic_product::<_, BOUNDED>(self.weight(i), component(x, c))?;
                numerator += (sum - rhs).abs();
                denominator += rhs.abs();
            }
            let den = denominator;
            if !numerator.is_finite() || !den.is_finite() {
                return None;
            }
            if den > T::Real::ZERO && numerator > residual * den {
                residual = residual.max(numerator / den);
            }
        }
        Some(SmoothingDiagnostics {
            data_fidelity: diagnostic_product::<_, BOUNDED>(fidelity.total().ok()?, self.scale)?,
            penalty: diagnostic_product::<_, BOUNDED>(penalty.total().ok()?, self.scale)?,
            scaled_optimality_residual: residual,
        })
    }

    fn extended_diagnostics(
        &self,
        input: &[T],
        solution: &[T],
    ) -> Result<SmoothingDiagnostics<T::Real>, SciOxideError> {
        if self.factor.is_none() {
            return Ok(SmoothingDiagnostics {
                data_fidelity: T::Real::ZERO,
                penalty: T::Real::ZERO,
                scaled_optimality_residual: T::Real::ZERO,
            });
        }
        let mut fidelity = ScaledAccumulator::ZERO;
        for ((&x, &z), w) in input
            .iter()
            .zip(solution)
            .zip((0..self.n).map(|i| self.weight(i)))
        {
            for (a, b) in [
                (x.real_component(), z.real_component()),
                (x.imaginary_component(), z.imaginary_component()),
            ] {
                let difference = Scaled::from_value(a)?.sub(Scaled::from_value(b)?)?;
                fidelity.add_scaled(difference.mul(difference)?.mul(Scaled::from_value(w)?)?)?;
            }
        }
        let mut penalty = ScaledAccumulator::ZERO;
        for row in 0..self.n - self.order {
            for imaginary in [false, true] {
                let mut sum = ScaledAccumulator::ZERO;
                for (&coefficient, &z) in self.differences.iter().zip(&solution[row..]) {
                    sum.add_product(
                        Scaled::from_value(coefficient)?,
                        Scaled::from_value(if imaginary {
                            z.imaginary_component()
                        } else {
                            z.real_component()
                        })?,
                    )?;
                }
                let difference = sum.total()?;
                penalty.add_scaled(
                    difference
                        .mul(difference)?
                        .mul(Scaled::from_value(self.lambda)?)?,
                )?;
            }
        }
        let mut residual = T::Real::ZERO;
        // Independent compact-matrix residual; scaled accumulation protects large samples.
        for (i, &x) in input.iter().enumerate() {
            let mut real = ScaledAccumulator::ZERO;
            let mut imag = ScaledAccumulator::ZERO;
            let mut denominator = ScaledAccumulator::ZERO;
            for (j, &z) in solution
                .iter()
                .enumerate()
                .take((i + self.order + 1).min(self.n))
                .skip(i.saturating_sub(self.order))
            {
                let a = self.band[i.min(j) * (self.order + 1) + i.abs_diff(j)].real_component();
                real.add_product(
                    Scaled::from_value(a)?,
                    Scaled::from_value(z.real_component())?,
                )?;
                imag.add_product(
                    Scaled::from_value(a)?,
                    Scaled::from_value(z.imaginary_component())?,
                )?;
                for component in [z.real_component(), z.imaginary_component()] {
                    denominator.add_product(
                        Scaled::from_value(a.abs())?,
                        Scaled::from_value(component.abs())?,
                    )?;
                }
            }
            let mut numerator = ScaledAccumulator::ZERO;
            for (sum, component) in [(real, x.real_component()), (imag, x.imaginary_component())] {
                let rhs =
                    Scaled::from_value(self.weight(i))?.mul(Scaled::from_value(component)?)?;
                numerator.add_scaled(sum.total()?.sub(rhs)?.abs())?;
                denominator.add_scaled(rhs.abs())?;
            }
            let den = denominator.total()?;
            if !den.is_zero() {
                residual = residual.max(numerator.total()?.div(den)?.round()?);
            }
        }
        Ok(SmoothingDiagnostics {
            data_fidelity: fidelity
                .total()?
                .mul(Scaled::from_value(self.scale)?)?
                .round()?,
            penalty: penalty
                .total()?
                .mul(Scaled::from_value(self.scale)?)?
                .round()?,
            scaled_optimality_residual: residual,
        })
    }
    /// Smooths every lane along `axis` of an N-D array.
    ///
    /// Each lane is smoothed on its own with the plan's weights and order. This
    /// is not a smoother with a multidimensional penalty. The workspace must meet
    /// [`WhittakerHendersonPlan::workspace_requirements`] for the number of
    /// lanes, and the solves run under the execution policy of the options. No
    /// diagnostics are computed.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis` is out of range.
    /// - `DimensionMismatch` if the input and output shapes differ, the axis
    ///   extent is not `n`, or the workspace is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a weighted sample or the solution overflows,
    ///   or a weighted sample underflows to zero.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn smooth_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, T>,
        workspace: &mut WhittakerWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, total) = crate::axis::shape(&input, &output, axis)?;
        if n != self.n {
            let shape = input.layout().shape().extents();
            return Err(axis_mismatch(shape, axis, self.n, shape));
        }
        self.capacity(workspace, total / n)?;
        execution.check()?;
        if total == 0 {
            return Ok(());
        }
        crate::axis::gather_blocks(&input, axis, &mut workspace.rhs[..total], n, |x| {
            inputs(core::slice::from_ref(&x))
        })?;
        for lane in workspace.rhs[..total].chunks_exact_mut(n) {
            for (position, x) in lane.iter_mut().enumerate() {
                *x = self.rhs_value(*x, position)?;
            }
        }
        self.solve(workspace, total / n, execution)?;
        execution.check()?;
        crate::axis::publish_blocks(&mut output, axis, &workspace.solution, n, 0)?;
        Ok(())
    }
}
/// Linear-algebra options bounded by the plan's execution policy.
fn linalg_options(options: SignalExecutionOptions) -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(options.execution_policy())
}
// Retry products that overflow or lose a nonzero term before original-unit scaling.
#[inline]
fn working_product<S: RealScalar>(a: S, b: S) -> Option<S> {
    let value = a * b;
    (value.is_finite() && (value != S::ZERO || a == S::ZERO || b == S::ZERO)).then_some(value)
}

#[inline]
fn diagnostic_product<S: RealScalar, const BOUNDED: bool>(a: S, b: S) -> Option<S> {
    if BOUNDED {
        Some(a * b)
    } else {
        working_product(a, b)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    fn range_guard<T: NumericScalar>() {
        let s = |x| T::Real::try_from_f64_lossy(x).unwrap();
        for order in [1, 2, 4] {
            let plan = WhittakerHendersonPlan::<T>::new(
                17,
                order,
                s(0.1),
                None,
                SignalExecutionOptions::new(),
            )
            .unwrap();
            assert!(plan.diagnostic_range.is_some());
            let input: Vec<T> = (0..17)
                .map(|i| {
                    T::from_components(
                        s(1. + ((i * 7 % 19) as f64 - 9.) / 64.),
                        s(1. + ((i * 3 % 11) as f64 - 5.) / 64.),
                    )
                })
                .collect();
            let (solution, observed) = plan.smooth(&input).unwrap();
            let checked = plan
                .working_diagnostics_for_order::<0, false>(&input, &solution)
                .unwrap();
            let extended = plan.extended_diagnostics(&input, &solution).unwrap();
            for diagnostics in [checked, extended] {
                for (a, b) in [
                    (observed.data_fidelity, diagnostics.data_fidelity),
                    (observed.penalty, diagnostics.penalty),
                    (
                        observed.scaled_optimality_residual,
                        diagnostics.scaled_optimality_residual,
                    ),
                ] {
                    assert!((a - b).abs() <= s(64.) * T::Real::EPSILON * b.abs().max(T::Real::ONE));
                }
            }
        }
    }
    #[test]
    fn bounded_checked_and_extended_diagnostics_agree_in_all_scalar_domains() {
        range_guard::<f32>();
        range_guard::<f64>();
        range_guard::<scioxide_core::Complex<f32>>();
        range_guard::<scioxide_core::Complex<f64>>();
    }
}
