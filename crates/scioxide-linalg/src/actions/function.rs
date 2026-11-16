//! Restarted Arnoldi approximation of `f(A) b` with an accumulated Hessenberg matrix.
use super::workspace::{FunctionActionRequirements, FunctionLayout};
use super::{ActionSlot, SpectralTermination, Work, run_action};
use crate::workspace::SolveSlot;
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};
use crate::{
    LinalgWorkspace, Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionWorkspaceRequest,
    SchurDecomposition, SchurOptions, SchurWorkspaceRequest, VectorView,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::checked_mul;
use scioxide_core::{
    AttemptRecord, BlockVectorView, BlockVectorViewMut, Budget, Complex, LinearOperator,
    NumericScalar, RealScalar, SciOxideError, Tolerance,
};
/// Action `f(A) B` of a matrix function on vectors, by restarted Arnoldi.
///
/// `A` is an `n × n` operator, used only through products `A x`. `B` holds
/// `columns` vectors of length `n`, stored one after another as in
/// [`BlockVectorView`]. `f` is any [`MatrixFunction`], including an analytic
/// function that you supply. Use [`crate::ExponentialAction`] for the
/// exponential when a norm bound is available.
///
/// For each column, the method builds an Arnoldi basis of length `m` (the
/// restart length) for `b/‖b‖₂` and then restarts from the last basis vector.
/// The Hessenberg matrices of all cycles are kept together as one Hessenberg
/// matrix `H` of order at most the projection size `p`. Each cycle adds the
/// correction `‖b‖₂ Vₖ (f(H) e₁)ₖ`, where `Vₖ` is the new basis and `(·)ₖ` the
/// matching rows (Eiermann and Ernst, 2006). `f(H)` is computed from a complex
/// Schur decomposition. Memory is O(n m) for the basis plus O(p²) for `H`. A
/// column stops when its error indicator meets the tolerance or the Krylov
/// space becomes invariant. Reaching the projection size first gives
/// `IterationLimit`.
///
/// The value holds only the settings; `n` and `columns` are read from the
/// operator and the input of each call. Defaults: restart length 16,
/// projection size 64, at most 100 000 operator applications, tolerance
/// `atol = rtol = √ε`, and default
/// [`MatrixFunctionSettings`](crate::MatrixFunctionSettings).
///
/// # Examples
///
/// ```
/// use scioxide_core::BlockVectorView;
/// use scioxide_linalg::{FunctionAction, Matrix, MatrixFunction};
/// let a = Matrix::<f64>::identity(2)?;
/// let input = BlockVectorView::new(&[1.0, 2.0], 2, 1)?;
/// let (output, _report): (Vec<f64>, _) =
///     FunctionAction::default().apply(&a.view().try_finite()?, MatrixFunction::Sqrt, input)?;
/// // sqrt(I) b = b.
/// assert!((output[1] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Eiermann and O. G. Ernst, "A restarted Krylov subspace method for the
/// evaluation of matrix functions", *SIAM J. Numer. Anal.* 44(6), 2006.
#[derive(Debug, Clone, Copy)]
pub struct FunctionAction<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Arnoldi basis length per restart cycle.
    restart: Budget,
    /// Maximum accumulated projected dimension across restarts.
    projection: Budget,
    /// Total operator applications across all RHS columns.
    applications: Budget,
    /// Stopping target `absolute * ||b||₂ + relative * ||f(A)b||₂`; indicators are not certificates.
    tolerance: Tolerance<R>,
    /// Bounded small dense function evaluation policy.
    function: crate::MatrixFunctionSettings<R>,
}
impl<R: RealScalar> FunctionAction<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    ///
    /// Every Arnoldi step counts as one iteration toward the iteration limit.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the Arnoldi basis length `m` of one restart cycle (default 16).
    pub const fn with_restart(mut self, restart: Budget) -> Self {
        self.restart = restart;
        self
    }

    /// Sets the maximum order `p` of the accumulated Hessenberg matrix (default 64).
    ///
    /// `p` bounds the total Krylov dimension over all restart cycles and must be
    /// at least the restart length. Each cycle evaluates `f` on a matrix of
    /// order up to `p` with a dense Schur method, at O(p³) cost.
    pub const fn with_projection(mut self, projection: Budget) -> Self {
        self.projection = projection;
        self
    }

    /// Sets the maximum number of operator applications over all columns (default 100 000).
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the stopping tolerance (default `atol = rtol = √ε`).
    ///
    /// A column stops when its error indicator is at most `atol ‖b‖₂ + rtol ‖y‖₂`,
    /// where `y` is the current approximation. The indicator is described at
    /// [`FunctionActionAccepted::estimated_errors`].
    pub const fn with_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the tolerance and series limit of the dense evaluations of `f` on
    /// the projected matrix.
    pub const fn with_function_settings(
        mut self,
        function: crate::MatrixFunctionSettings<R>,
    ) -> Self {
        self.function = function;
        self
    }

    /// Returns the Arnoldi basis length of one restart cycle.
    pub const fn restart(&self) -> Budget {
        self.restart
    }

    /// Returns the maximum order of the accumulated Hessenberg matrix.
    pub const fn projection(&self) -> Budget {
        self.projection
    }

    /// Returns the maximum number of operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the stopping tolerance.
    pub const fn tolerance(&self) -> Tolerance<R> {
        self.tolerance
    }

    /// Returns the settings of the dense evaluations of `f`.
    pub const fn function_settings(&self) -> crate::MatrixFunctionSettings<R> {
        self.function
    }
}
impl<R: RealScalar> Default for FunctionAction<R> {
    fn default() -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            restart: Budget::ONE.saturating_add(15),
            projection: Budget::ONE.saturating_add(63),
            applications: Budget::ONE.saturating_add(99_999),
            tolerance: Tolerance::from_sqrt_epsilon(),
            function: crate::MatrixFunctionSettings::default(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct FunctionCall<R: RealScalar> {
    n: usize,
    columns: usize,
    options: FunctionAction<R>,
}
/// Workspace of [`FunctionAction`].
///
/// It holds O(n m + p² + n c) scalars for restart length `m`, projection size
/// `p` and `c` columns. After a call that returns `Ok`, it also holds
/// per-column diagnostics; see [`Self::last_accepted`].
#[derive(Debug)]
pub struct FunctionActionWorkspace<S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    accepted_columns: Option<usize>,
    slot: SolveSlot<()>,
    n: usize,
    columns: usize,
    restart: usize,
    projection: usize,
    basis: Vec<S>,
    image: Vec<S>,
    coefficients: Vec<S>,
    provider: Vec<S>,
    h: Matrix<Complex<S::Real>>,
    factor: SchurDecomposition<Complex<S::Real>>,
    schur: LinalgWorkspace<'static, Complex<S::Real>>,
    function: LinalgWorkspace<'static, Complex<S::Real>>,
    value: Matrix<Complex<S::Real>>,
    candidate: Vec<Complex<S::Real>>,
    correction: Vec<Complex<S::Real>>,
    errors: Vec<S::Real>,
    completed: Vec<bool>,
}
/// Termination status and work counts of a function action.
///
/// Per-column diagnostics are in [`FunctionActionWorkspace::last_accepted`].
/// After a limit or cancellation, each output column holds the approximation
/// after its completed cycles. A column with no completed cycle is zero.
#[derive(Debug, Clone, Copy)]
pub struct FunctionActionReport {
    termination: SpectralTermination,
    applications: usize,
    cycles: usize,
}

impl FunctionActionReport {
    /// Returns the termination status, common to all columns.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }
    /// Returns the number of operator applications.
    pub const fn applications(&self) -> usize {
        self.applications
    }
    /// Returns the number of completed restart cycles over all columns.
    pub const fn cycles(&self) -> usize {
        self.cycles
    }
}

/// Per-column diagnostics of the latest function action that returned `Ok`.
///
/// Obtained from [`FunctionActionWorkspace::last_accepted`]; the slices borrow
/// the workspace.
#[derive(Debug, Clone, Copy)]
pub struct FunctionActionAccepted<'a, R: RealScalar> {
    estimated_errors: &'a [R],
    completed: &'a [bool],
}
impl<R: RealScalar> FunctionActionAccepted<'_, R> {
    /// Returns, per column, the error indicator of the last completed cycle.
    ///
    /// The indicator is `‖b‖₂ β |eₗᵀ f(H) e₁|`, where `β` is the last
    /// subdiagonal entry of the Arnoldi recurrence and `l` the order of `H`,
    /// but at least `ε ‖y‖₂` for the approximation `y`. It is zero when the
    /// Krylov space became invariant. It estimates the error and is not a
    /// bound on it.
    pub const fn estimated_errors(&self) -> &[R] {
        self.estimated_errors
    }
    /// Returns, per column, whether the stopping test was met.
    pub const fn completed(&self) -> &[bool] {
        self.completed
    }
}

impl<S: NumericScalar> ActionSlot for FunctionActionWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> FunctionActionWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Returns the attempt record of the most recent call, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
    /// Returns the per-column diagnostics of the latest call.
    ///
    /// Returns `None` before the first call, after [`Self::try_reserve`], and
    /// after a call that returned `Err`.
    pub fn last_accepted(&self) -> Option<FunctionActionAccepted<'_, S::Real>> {
        let columns = self.accepted_columns?;
        Some(FunctionActionAccepted {
            estimated_errors: &self.errors[..columns],
            completed: &self.completed[..columns],
        })
    }
}
impl<R: RealScalar> FunctionCall<R> {
    fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
    ) -> Result<FunctionActionRequirements<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        crate::checked::expect_shape([self.n, self.n], [a.rows(), a.cols()])?;
        let n = self.n;
        let m = self.options.restart.get();
        let p = self.options.projection.get();
        let columns = self.columns;
        let schur =
            SchurDecomposition::<Complex<R>>::workspace_requirements(&SchurWorkspaceRequest::new(
                p,
                &SchurOptions::<R>::default().with_execution_options(
                    super::projected_execution_options(self.options.execution),
                ),
            ))?;
        let function = crate::functions::requirements::<R>(
            p,
            MatrixFunctionWorkspaceRequest::Evaluate,
            self.options.execution.byte_limit(),
        )?;
        let payload = sum([
            bytes::<S>(sum([
                m,
                product(&[n, sum([m, 1])?])?,
                n,
                a.apply_workspace_len()?,
            ])?)?,
            bytes::<Complex<R>>(sum([product(&[4, p, p])?, p, product(&[n, columns])?, n])?)?,
            bytes::<crate::SchurBlock>(p)?,
            bytes::<R>(columns)?,
            bytes::<bool>(columns)?,
            schur.required_bytes::<Complex<R>>()?,
            function.required_bytes::<Complex<R>>()?,
        ])?;
        FunctionActionRequirements::new(
            FunctionLayout {
                n: self.n,
                columns: self.columns,
                restart: self.options.restart.get(),
                projection: self.options.projection.get(),
                provider: a.apply_workspace_len()?,
            },
            self.options.execution,
            payload,
            Default::default(),
            [schur, function],
        )
    }
    fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
    ) -> Result<FunctionActionWorkspace<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let requirements = self.workspace_requirements(a)?;
        let mut workspace = FunctionActionWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    fn apply<S: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        function: MatrixFunction<'_, R>,
        input: BlockVectorView<'_, S>,
    ) -> Result<(Vec<O>, FunctionActionReport), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let mut workspace = self.workspace(a)?;
        let mut output = scioxide_core::backend::try_filled(
            checked_mul(
                self.n,
                self.columns,
                scioxide_core::quantity!("block_elements"),
            )?,
            O::ZERO,
            scioxide_core::quantity!("output"),
        )?;
        let report = self.apply_into(
            a,
            function,
            input,
            BlockVectorViewMut::new(&mut output, self.n, self.columns)?,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn apply_into<'s, 'c, S: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        function: MatrixFunction<'_, R>,
        input: BlockVectorView<'_, S>,
        output: BlockVectorViewMut<'_, O>,
        w: &mut FunctionActionWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<FunctionActionReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        w.accepted_columns = None;
        w.slot.reject();
        let function = crate::functions::ValidatedFunction::new(function)?;
        let requirements = self.workspace_requirements(a)?;
        let shape = [self.n, self.columns];
        crate::checked::expect_shape(shape, [input.rows(), input.right_hand_sides()])?;
        crate::checked::expect_shape(shape, [output.rows(), output.right_hand_sides()])?;
        crate::checked::validate_values(
            input.as_slice(),
            scioxide_core::NonFiniteOrigin::Input,
            "action input must be finite",
        )?;
        w.prepare(&requirements)?;
        let context = context.into();
        let (termination, applications, cycles) = run_action(
            w,
            context,
            scioxide_core::algorithm_id!("linear", "FunctionAction", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| self.run(a, function, FunctionBlock { input, output }, w, work),
            |(termination, _, _)| *termination == SpectralTermination::Converged,
        )?;
        w.accepted_columns = Some(w.columns);
        Ok(FunctionActionReport {
            termination,
            applications,
            cycles,
        })
    }

    fn run<S: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        function: crate::functions::ValidatedFunction<'_, R>,
        block: FunctionBlock<'_, S, O>,
        w: &mut FunctionActionWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<(SpectralTermination, usize, usize), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let FunctionBlock { input, mut output } = block;

        let execution = work.execution;

        let n = self.n;
        let columns = self.columns;
        let p = self.options.projection.get();
        if w.provider.len() != a.apply_workspace_len()? {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("a"),
                "operator changed its scratch requirement after preparation",
            ));
        }
        let mut cycles = 0;
        let mut termination = SpectralTermination::Converged;
        w.candidate.fill(Complex::ZERO);
        w.errors.fill(R::ZERO);
        w.completed.fill(false);
        let outcome = (|| {
            execution.check()?;
            for column in 0..columns {
                let b = input.column(column)?;
                let bnorm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(b),
                    work.kernels,
                )?;
                if bnorm == R::ZERO {
                    w.completed[column] = true;
                    continue;
                }
                for (v, &x) in w.basis[..n].iter_mut().zip(b) {
                    *v = x / S::from_real(bnorm);
                }
                // Retain the coupled Hessenberg projection across restarts (Eiermann--Ernst).
                // The current basis contributes the new correction; the final coupling
                // times the last entry of f(H)e1 estimates the remaining Arnoldi tail.
                w.h.as_mut_slice().fill(Complex::ZERO);
                let mut offset = 0;
                while offset < p {
                    let core::ops::ControlFlow::Continue((dim, happy, beta)) =
                        self.expand_projection(a, w, &mut work, offset)?
                    else {
                        termination = SpectralTermination::EvaluationLimit;
                        break;
                    };
                    let total = offset + dim;
                    self.evaluate_projection(function, w, &mut work, total)?;
                    for i in 0..n {
                        let mut correction = Complex::ZERO;
                        for j in 0..dim {
                            correction += crate::eigen::into_complex(w.basis[j * n + i])
                                * w.value.get_element(offset + j, 0);
                        }
                        correction *= Complex::from_real(bnorm);
                        w.correction[i] = correction;
                    }
                    let error = if happy {
                        R::ZERO
                    } else {
                        scioxide_core::Scaled::from_value(bnorm)?
                            .mul(scioxide_core::Scaled::from_value(beta)?)?
                            .mul(
                                scioxide_core::ScaledValue::from_value(
                                    w.value.get_element(total - 1, 0),
                                )?
                                .magnitude()?,
                            )?
                            .round()?
                    };
                    for (next, &previous) in w
                        .correction
                        .iter_mut()
                        .zip(&w.candidate[column * n..(column + 1) * n])
                    {
                        *next = crate::checked::finite_candidate(previous + *next)?;
                    }
                    let norm = crate::reduction::l2_norm_with_context(
                        VectorView::from_slice(&w.correction),
                        work.kernels,
                    )?;
                    let threshold =
                        scioxide_core::Scaled::from_value(self.options.tolerance.absolute())?
                            .mul(scioxide_core::Scaled::from_value(bnorm)?)?
                            .add(
                                scioxide_core::Scaled::from_value(
                                    self.options.tolerance.relative(),
                                )?
                                .mul(scioxide_core::Scaled::from_value(norm)?)?,
                            )?;
                    let error = if happy {
                        R::ZERO
                    } else {
                        error.max(R::EPSILON * norm)
                    };
                    if O::IS_REAL {
                        // `f(H)` comes from a complex Schur form, so a real
                        // result carries imaginary rounding and truncation
                        // errors. Only an imaginary part above both the
                        // target and the error indicator shows a complex
                        // result.
                        let error = scioxide_core::Scaled::from_value(error)?;
                        let allowance = if error.abs_le(threshold) {
                            threshold
                        } else {
                            error
                        };
                        for z in &w.correction {
                            if !scioxide_core::Scaled::from_value(z.im.abs())?.abs_le(allowance) {
                                return Err(SciOxideError::domain_error(
                                    scioxide_core::argument!("output"),
                                    scioxide_core::DomainReason::OutsideDomain,
                                    "function has a complex result; provide complex output",
                                ));
                            }
                        }
                    }
                    w.candidate[column * n..(column + 1) * n].copy_from_slice(&w.correction);
                    cycles += 1;
                    w.errors[column] = error;
                    if happy || scioxide_core::Scaled::from_value(error)?.abs_le(threshold) {
                        w.completed[column] = true;
                        break;
                    }
                    offset = total;
                    if offset < p {
                        // `expand_projection` already stored the coupling `beta`.
                        w.basis.copy_within(dim * n..(dim + 1) * n, 0);
                    }
                }
                if !w.completed[column] && termination != SpectralTermination::EvaluationLimit {
                    termination = SpectralTermination::IterationLimit;
                }
                if termination == SpectralTermination::EvaluationLimit {
                    break;
                }
            }
            execution.check()
        })();
        termination = super::action_termination(outcome, termination)?;
        for (out, &z) in output.as_mut_slice().iter_mut().zip(&w.candidate) {
            *out = O::from_components(z.re, z.im);
        }
        Ok((termination, work.count, cycles))
    }
    fn expand_projection<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        w: &mut FunctionActionWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        offset: usize,
    ) -> Result<core::ops::ControlFlow<(), (usize, bool, R)>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let n = self.n;
        let m = self.options.restart.get();
        let p = self.options.projection.get();

        let capacity = m.min(p - offset);
        let mut dim = 0;
        let mut happy = false;
        let mut beta = R::ZERO;
        for j in 0..capacity {
            if !work.iteration()? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            if !work.apply(
                a,
                &w.basis[j * n..(j + 1) * n],
                &mut w.image,
                &mut w.provider,
            )? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            let reference = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.image),
                work.kernels,
            )?;
            beta = crate::kernel::subspace::project(
                &mut w.image,
                &w.basis[..(j + 1) * n],
                &mut w.coefficients,
                |i, c| {
                    let index = (offset + i) * p + offset + j;
                    w.h.as_mut_slice()[index] =
                        w.h.as_slice()[index] + crate::eigen::into_complex(c);
                },
                work.kernels,
            )?;
            dim = j + 1;
            if beta <= crate::checked::dimension_epsilon::<R>(n.max(1))? * reference {
                happy = true;
                break;
            }
            if offset + j + 1 < p {
                w.h.as_mut_slice()[(offset + j + 1) * p + offset + j] = Complex::from_real(beta);
            }
            for (v, &x) in w.basis[(j + 1) * n..(j + 2) * n].iter_mut().zip(&w.image) {
                *v = x / S::from_real(beta);
            }
        }
        Ok(core::ops::ControlFlow::Continue((dim, happy, beta)))
    }
    fn evaluate_projection<S: NumericScalar<Real = R>>(
        &self,
        function: crate::functions::ValidatedFunction<'_, R>,
        w: &mut FunctionActionWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        total: usize,
    ) -> Result<(), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let projected = w.h.view().block(0, 0, total, total)?;
        SchurDecomposition::decompose_into(
            projected,
            &SchurOptions::default()
                .with_execution_options(super::projected_execution_options(self.options.execution)),
            &mut w.factor,
            &mut w.schur,
            work.kernels.scoped(&mut *work.solve),
        )?;
        w.value.try_resize_flat(total, total)?;
        let options = MatrixFunctionOptions::new()
            .with_function_settings(self.options.function)
            .with_execution_options(super::projected_execution_options(self.options.execution));
        let context = work
            .kernels
            .scoped(&mut *work.solve)
            .with_options(&options.execution_options())?;
        w.function.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linear", "ProjectedFunction", 1),
                total as u128,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |scratch, kernels, solve| {
                crate::functions::evaluate_schur_into(
                    &w.factor,
                    function,
                    w.value.view_mut(),
                    scratch,
                    &options,
                    kernels,
                    Some(solve),
                )
            },
        )
    }
}
impl<R: RealScalar> FunctionAction<R> {
    /// Checks the settings against `n × n` operators and blocks of `columns`
    /// vectors and returns the per-call value.
    fn call(&self, n: usize, columns: usize) -> Result<FunctionCall<R>, SciOxideError> {
        if self.projection.get() < self.restart.get() {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("self.projection"),
                "function action projection must cover its restart basis",
            ));
        }
        checked_mul(n, columns, scioxide_core::quantity!("block_elements"))?;
        Ok(FunctionCall {
            n,
            columns,
            options: *self,
        })
    }

    /// Returns the order of the square operator `op`.
    fn order<S: NumericScalar<Real = R>>(
        op: &dyn LinearOperator<S>,
    ) -> Result<usize, SciOxideError> {
        crate::checked::square(op.rows(), op.cols())?;
        Ok(op.rows())
    }
    /// Returns the workspace storage needed to apply a function of the `n × n`
    /// operator `a` to blocks of `columns` vectors.
    ///
    /// The operator is not applied; only its shape and scratch length are read.
    /// The storage includes the dense Schur and function evaluations. The
    /// settings are checked first, so the execution methods report the same
    /// errors before any work.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square.
    /// - `InvalidInput` (`Inconsistent`, argument `self.projection`) if the
    ///   projection size is smaller than the restart length.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length query of `a`.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        columns: usize,
    ) -> Result<FunctionActionRequirements<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        self.call(Self::order(a)?, columns)?
            .workspace_requirements(a)
    }

    /// Allocates a workspace for `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        columns: usize,
    ) -> Result<FunctionActionWorkspace<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        self.call(Self::order(a)?, columns)?.workspace(a)
    }

    /// Computes `f(A) B` with a new workspace and returns the output block.
    ///
    /// The returned vector stores the `n` entries of each column one column
    /// after another, as in `input`. The output scalar `O` may be real or
    /// complex, as in [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::apply_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::BlockVectorView;
    /// use scioxide_linalg::{FunctionAction, Matrix, MatrixFunction};
    /// let a = Matrix::<f64>::identity(2)?;
    /// let method = FunctionAction::default();
    /// let (output, report) = method.apply::<f64, f64>(
    ///     &a.view().try_finite()?,
    ///     MatrixFunction::Sqrt,
    ///     BlockVectorView::new(&[1., 2.], 2, 1)?,
    /// )?;
    /// // sqrt(I) b = b.
    /// assert!((output[1] - 2.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply<S: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        function: MatrixFunction<'_, R>,
        input: BlockVectorView<'_, S>,
    ) -> Result<(Vec<O>, FunctionActionReport), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        self.call(Self::order(a)?, input.right_hand_sides())?
            .apply(a, function, input)
    }

    /// Computes `f(A) B` and writes it to `output`.
    ///
    /// With a real output scalar the result must be real. The output receives
    /// the real part of each approximation; its imaginary part, which for a
    /// real result is an error of the complex Schur evaluation of `f(H)`, may
    /// not exceed the larger of the stopping threshold and the error
    /// indicator. A real operator may
    /// write complex output, which is needed when `f(A) b` is complex, for
    /// example the square root of a matrix with negative eigenvalues. For
    /// [`MatrixFunction::Analytic`], the assumptions of your function must hold
    /// on the eigenvalues of every projected matrix `H`, not only on those of `A`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `input` or `output` is not `n × columns`.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `DomainError` if `function` is a power with a non-finite exponent, or
    ///   if `output` is real and an approximation has an imaginary part above
    ///   both the stopping threshold and its error indicator.
    /// - `NonFinite` if an entry of `input` is not finite, or if `a` returns a
    ///   non-finite value.
    /// - `InvalidInput` (`Inconsistent`) if the scratch length of `a` changed
    ///   after the workspace was reserved.
    /// - `NumericalInstability` (`Overflow`) if the approximation overflows.
    /// - Errors of the dense Schur decomposition and of evaluating `f` on `H`,
    ///   for example when an eigenvalue of `H` lies on a branch cut of `f`.
    /// - Errors returned by `a` or by an analytic `function`.
    pub fn apply_into<'s, 'c, S: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        function: MatrixFunction<'_, R>,
        input: BlockVectorView<'_, S>,
        output: BlockVectorViewMut<'_, O>,
        w: &mut FunctionActionWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<FunctionActionReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        self.call(Self::order(a)?, input.right_hand_sides())?
            .apply_into(a, function, input, output, w, context)
    }
}

struct FunctionBlock<'a, S: NumericScalar, O: NumericScalar> {
    input: BlockVectorView<'a, S>,
    output: BlockVectorViewMut<'a, O>,
}

impl<S: NumericScalar> Default for FunctionActionWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> FunctionActionWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            accepted_columns: None,
            slot: SolveSlot::new(),
            n: 0,
            columns: 0,
            restart: 0,
            projection: 0,
            basis: Vec::new(),
            image: Vec::new(),
            coefficients: Vec::new(),
            provider: Vec::new(),
            h: Matrix::empty(),
            factor: SchurDecomposition::empty(),
            schur: LinalgWorkspace::new(),
            function: LinalgWorkspace::new(),
            value: Matrix::empty(),
            candidate: Vec::new(),
            correction: Vec::new(),
            errors: Vec::new(),
            completed: Vec::new(),
        }
    }
    /// Returns whether the workspace already holds the storage that `requirements` describes.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn check(
        &self,
        requirements: &FunctionActionRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        PlannedStorage::check(self, requirements)
    }
    /// Grows the workspace to the storage that `requirements` describes.
    ///
    /// Buffers that are already large enough keep their capacity, and the
    /// attempt record is kept. If growth fails, some buffers may already have
    /// grown; the workspace stays usable.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `AllocationFailure` if an allocation fails.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &FunctionActionRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for FunctionActionWorkspace<S>
where
    scioxide_core::Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    type Requirements = FunctionActionRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &FunctionActionRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let FunctionLayout {
            n,
            columns,
            restart: m,
            projection: p,
            provider,
        } = requirements.layout;

        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.basis, product(&[n, sum([m, 1])?])?)?;
        plan.buffer(&self.coefficients, m)?;
        plan.buffer(&self.image, n)?;
        plan.buffer(&self.provider, provider)?;
        plan.buffer(&self.candidate, product(&[n, columns])?)?;
        plan.buffer(&self.correction, n)?;
        plan.buffer(&self.errors, columns)?;
        plan.buffer(&self.completed, columns)?;
        plan.matrix(&self.h, p, p)?;
        plan.matrix(&self.value, p, p)?;
        self.factor.action_storage(&mut plan, p)?;
        plan.dense(&self.schur, &requirements.storage.complex[0])?;
        plan.dense(&self.function, &requirements.storage.complex[1])?;
        Ok(plan)
    }
    fn configure(
        &mut self,
        requirements: &FunctionActionRequirements<S>,
    ) -> Result<(), SciOxideError> {
        self.accepted_columns = None;
        let FunctionLayout {
            n,
            columns,
            restart: m,
            projection: p,
            provider,
        } = requirements.layout;

        resize(&mut self.basis, product(&[n, sum([m, 1])?])?, S::ZERO)?;
        resize(&mut self.coefficients, m, S::ZERO)?;
        resize(&mut self.image, n, S::ZERO)?;
        resize(&mut self.provider, provider, S::ZERO)?;
        resize(&mut self.candidate, product(&[n, columns])?, Complex::ZERO)?;
        resize(&mut self.correction, n, Complex::ZERO)?;
        resize(&mut self.errors, columns, S::Real::ZERO)?;
        resize(&mut self.completed, columns, false)?;
        self.h.try_resize_flat(p, p)?;
        self.value.try_resize_flat(p, p)?;
        self.factor.reserve(p)?;
        self.schur.try_reserve(&requirements.storage.complex[0])?;
        self.function
            .try_reserve(&requirements.storage.complex[1])?;
        self.n = n;
        self.columns = columns;
        self.restart = m;
        self.projection = p;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for FunctionActionWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.basis.storage_bytes()?,
            self.image.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.provider.storage_bytes()?,
            self.h.storage_bytes()?,
            self.factor.storage_bytes()?,
            self.schur.storage_bytes()?,
            self.function.storage_bytes()?,
            self.value.storage_bytes()?,
            self.candidate.storage_bytes()?,
            self.correction.storage_bytes()?,
            self.errors.storage_bytes()?,
            self.completed.storage_bytes()?,
        ])
    }
}
