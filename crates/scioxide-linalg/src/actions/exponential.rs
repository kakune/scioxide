//! Al-Mohy--Higham adaptive Taylor exponential and implicit augmented phi actions.
use super::workspace::{ExponentialActionRequirements, ExponentialLayout};
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};
mod parameters;
use super::{ActionSlot, SpectralTermination, Work, run_action};
use crate::workspace::SolveSlot;
use alloc::vec::Vec;
use parameters::Parameters;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    AdjointOperator, AttemptRecord, BlockVectorView, BlockVectorViewMut, Budget, Complex,
    NonFiniteOrigin, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError, Tolerance,
};
/// Action `φₖ(tA) B` of the matrix exponential or a φ-function on vectors.
///
/// `A` is an `n × n` operator, used only through products `A x` and `Aᴴ x`.
/// `B` holds `columns` vectors of length `n`, stored one after another as in
/// [`BlockVectorView`]. See [`PhiFunction`] for the definition of `φₖ`.
///
/// The method is the truncated Taylor method of Al-Mohy and Higham (2011).
/// With `μ` the shift, `exp(tA) b ≈ e^(tμ) (T_m(t(A − μI)/s))ˢ b`, where `T_m`
/// is the Taylor polynomial of degree `m`. The method chooses `(m, s)` to
/// minimize the number of products `m s`, using the `θₘ` tables of that paper
/// with the norm bound. When the bound is large, it also uses one-norm
/// estimates of `‖(t(A − μI))ᵖ‖₁^(1/p)`, which need adjoint products. Each
/// step stops adding terms once the last two are below the tolerance. For `k ≥ 1`,
/// `φₖ(tA) b` is read from the first `n` entries of the exponential of an
/// augmented matrix of order `n + k`, which is applied without being formed.
/// Its trailing `k × k` block is nilpotent up to the shift and is advanced
/// exactly; the Taylor series and its stopping test cover the first `n`
/// entries, which continue until their own tolerance is met within the degree
/// limit. For the exponential one pair `(m, s)` serves all columns;
/// φ-functions choose it per column.
///
/// The value holds only the settings; `n` and `columns` are read from the
/// operator and the input of each call. The norm bound is required: it must
/// bound `‖A − μI‖₁`, where `μ` is the shift (zero by default). The method uses
/// it to choose the Taylor degree and the number of scaling steps but cannot
/// verify it; an underestimate can make the result inaccurate. Defaults: shift
/// 0, tolerance `atol = 0` and `rtol = u = ε/2`, at most 100 000 operator
/// applications, Taylor degree at most 55, and exponential only
/// (`max_phi = 0`).
///
/// # Accuracy
///
/// In exact arithmetic and with accurate norm estimates, `(m, s)` keeps the
/// relative backward error of the truncated series below `u = ε/2`, or below
/// `rtol` if that is smaller. The per-column error indicators in
/// [`ExponentialActionAccepted`] are truncation estimates, not error bounds.
///
/// # Examples
///
/// ```
/// use scioxide_core::BlockVectorView;
/// use scioxide_linalg::{ExponentialAction, Matrix, PhiFunction, SpectralTermination};
/// let a = Matrix::from_rows(&[[-2.0_f64, 1.0], [0.0, -2.0]])?;
/// // μ = trace(A)/2 = −2, and ‖A − μI‖₁ = 1.
/// let action = ExponentialAction::new(1.0).with_shift(-2.0);
/// let input = BlockVectorView::new(&[1.0, 2.0], 2, 1)?;
/// let (output, report) = action.apply(&a.view().try_finite()?, PhiFunction::exp(1.0), input)?;
/// assert_eq!(report.termination(), SpectralTermination::Converged);
/// // A = −2I + N with N² = 0, so exp(A) b = e⁻² (b + N b) = e⁻² (3, 2).
/// assert!((output[0] - 3.0 * (-2.0_f64).exp()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. H. Al-Mohy and N. J. Higham, "Computing the action of the matrix
/// exponential, with an application to exponential integrators",
/// *SIAM J. Sci. Comput.* 33(2), 2011.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialAction<S: NumericScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Caller-supplied finite upper bound on the shifted operator's induced one-norm.
    operator_norm_bound: S::Real,
    shift: S,
    /// Local truncation target `absolute * ||b||₁ + relative * ||action||₁`, distributed over steps.
    tolerance: Tolerance<S::Real>,
    /// Maximum forward plus adjoint applications, including power estimates and all RHS columns.
    applications: Budget,
    /// Maximum Taylor degree per scaling step; degrees through 55 are supported.
    terms: Budget,
    /// Largest supported phi index; zero reserves exponential-only storage.
    max_phi: usize,
}
impl<S: NumericScalar> ExponentialAction<S> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    ///
    /// Every Taylor term and every product of a norm estimate counts as one
    /// iteration toward the iteration limit.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the truncation tolerance (default `atol = 0`, `rtol = u = ε/2`).
    ///
    /// Within each of the `s` scaling steps, Taylor terms are added until the
    /// 1-norms of the last two terms sum to at most `(atol ‖b‖₁ + rtol ‖y‖₁) / s`,
    /// where `‖y‖₁` is the largest 1-norm of the partial sums of the step, the
    /// scale of their rounding errors; for φ-functions both refer to the first
    /// `n` entries of the augmented vector. A relative tolerance below `u` also
    /// lowers the degree thresholds `θₘ`, so that more steps are taken.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<S::Real>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of operator applications (default 100 000).
    ///
    /// Forward and adjoint products count, over all columns and including the
    /// norm estimates.
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the maximum Taylor degree per scaling step, at most 55 (default 55).
    pub const fn with_terms(mut self, terms: Budget) -> Self {
        self.terms = terms;
        self
    }

    /// Sets the largest φ index the method accepts (default 0, exponential only).
    ///
    /// The index may not exceed the Taylor degree limit. Workspace vectors have
    /// length `n + max_phi`.
    pub const fn with_max_phi(mut self, max_phi: usize) -> Self {
        self.max_phi = max_phi;
        self
    }

    /// Sets the shift `μ` in `exp(tA) = e^(tμ) exp(t(A − μI))` (default 0).
    ///
    /// A shift can reduce `‖A − μI‖₁` and with it the number of scaling steps.
    /// A common choice is `μ = trace(A)/n`, which the method cannot compute from a
    /// general operator. The norm bound must then bound `‖A − μI‖₁`. For
    /// φ-functions the shift applies to the whole augmented matrix, whose
    /// trailing block then decays like `e^(−tμ)`; the steps are chosen for the
    /// one-norm `max(|t| ‖A − μI‖₁, 1 + |tμ|)`. The factor `e^(tμ)` is applied
    /// with an extended exponent range, so it need not be representable on
    /// its own. A large shift makes the φ series of each step cancel: for
    /// `A − μI` of norm one and `tμ` from 100 to 800, `φ₁` to `φ₃` have
    /// relative errors from `2·10⁻¹³` to `4·10⁻¹¹` in double precision.
    /// [`ExponentialAction::new`] rejects a non-finite shift.
    pub const fn with_shift(mut self, mu: S) -> Self {
        self.shift = mu;
        self
    }

    /// Returns the bound on `‖A − μI‖₁`.
    pub const fn operator_norm_bound(&self) -> S::Real {
        self.operator_norm_bound
    }

    /// Returns the shift `μ`.
    pub const fn shift(&self) -> S {
        self.shift
    }

    /// Returns the truncation tolerance.
    pub const fn tolerance(&self) -> Tolerance<S::Real> {
        self.tolerance
    }

    /// Returns the maximum number of operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the maximum Taylor degree per scaling step.
    pub const fn terms(&self) -> Budget {
        self.terms
    }

    /// Returns the largest φ index the method accepts.
    pub const fn max_phi(&self) -> usize {
        self.max_phi
    }
}
impl<S: NumericScalar> ExponentialAction<S> {
    /// Returns default options with the given bound on `‖A − μI‖₁`.
    ///
    /// [`ExponentialAction::new`] checks that the bound is finite and
    /// nonnegative.
    pub fn new(operator_norm_bound: S::Real) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            operator_norm_bound,
            shift: S::ZERO,
            tolerance: Tolerance::new(S::Real::ZERO, S::Real::EPSILON * S::Real::HALF)
                .unwrap_or_else(|_| unreachable!("unit roundoff is finite and positive")),
            applications: Budget::ONE.saturating_add(99_999),
            terms: Budget::ONE.saturating_add(54),
            max_phi: 0,
        }
    }
}
/// The function `φₖ(tA)` applied by [`ExponentialAction`], with `φ₀ = exp`.
///
/// ```text
/// φ₀(z) = eᶻ,   φₖ(z) = Σ_{j ≥ 0} zʲ / (j + k)!   for k ≥ 1.
/// ```
///
/// The φ-functions appear in exponential integrators. `t` is a real time that
/// scales the operator.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::PhiFunction;
/// let exp = PhiFunction::exp(0.5_f64);
/// assert_eq!((exp.index(), exp.time()), (0, 0.5));
/// assert_eq!(PhiFunction::phi(2, 0.5_f64).index(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PhiFunction<R: RealScalar> {
    index: usize,
    time: R,
}
impl<R: RealScalar> PhiFunction<R> {
    /// Returns the exponential `exp(time·A)`.
    pub const fn exp(time: R) -> Self {
        Self { index: 0, time }
    }
    /// Returns `φ_index(time·A)`; index zero is the exponential.
    pub const fn phi(index: usize, time: R) -> Self {
        Self { index, time }
    }
    /// Returns the index `k`, zero for the exponential.
    pub const fn index(self) -> usize {
        self.index
    }
    /// Returns the time `t` that scales the operator.
    pub const fn time(self) -> R {
        self.time
    }
}
#[derive(Debug, Clone, Copy)]
struct ExponentialCall<S: NumericScalar> {
    n: usize,
    columns: usize,
    options: ExponentialAction<S>,
}
/// Workspace of [`ExponentialAction`].
///
/// After a call that returns `Ok`, it also holds per-column diagnostics; see
/// [`Self::last_accepted`].
#[derive(Debug)]
pub struct ExponentialActionWorkspace<S: NumericScalar> {
    accepted_columns: Option<usize>,
    slot: SolveSlot<()>,
    n: usize,
    columns: usize,
    max_phi: usize,
    scaled: Vec<ScaledValue<Complex<S::Real>>>,
    current: Vec<S>,
    sum: Vec<S>,
    term: Vec<S>,
    next: Vec<S>,
    provider: Vec<S>,
    candidate: Vec<S>,
    progress: Vec<S::Real>,
    errors: Vec<S::Real>,
    completed: Vec<bool>,
}
/// Termination status and operator-application count of an exponential action.
///
/// Per-column diagnostics are in [`ExponentialActionWorkspace::last_accepted`].
/// After a limit or cancellation, each output column holds the result of the
/// scaling steps it completed. For the exponential this is `exp(f t A) b`,
/// where `f` is the completed fraction given by
/// [`ExponentialActionAccepted::progress`]. A column with no completed step
/// holds `b` for the exponential and zero for a φ-function.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialActionReport {
    termination: SpectralTermination,
    applications: usize,
}

impl ExponentialActionReport {
    /// Returns the termination status, common to all columns.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }
    /// Returns the number of operator applications, forward and adjoint,
    /// including those of the norm estimates.
    pub const fn applications(&self) -> usize {
        self.applications
    }
}

/// Per-column diagnostics of the latest exponential action that returned `Ok`.
///
/// Obtained from [`ExponentialActionWorkspace::last_accepted`]; the slices
/// borrow the workspace.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialActionAccepted<'a, R: RealScalar> {
    progress: &'a [R],
    estimated_errors: &'a [R],
    completed: &'a [bool],
}
impl<R: RealScalar> ExponentialActionAccepted<'_, R> {
    /// Returns, per column, the fraction of `[0, t]` covered by completed scaling steps.
    pub const fn progress(&self) -> &[R] {
        self.progress
    }
    /// Returns, per column, the truncation indicator summed over completed steps.
    ///
    /// For each step the indicator is the 1-norm of the last two Taylor terms,
    /// scaled to the output. It estimates the truncation error and is not a
    /// bound on the forward error.
    pub const fn estimated_errors(&self) -> &[R] {
        self.estimated_errors
    }
    /// Returns, per column, whether every scaling step completed.
    pub const fn completed(&self) -> &[bool] {
        self.completed
    }
}

impl<S: NumericScalar> ActionSlot for ExponentialActionWorkspace<S> {
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> ExponentialActionWorkspace<S> {
    /// Returns the attempt record of the most recent call, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
    /// Returns the per-column diagnostics of the latest call.
    ///
    /// Returns `None` before the first call, after [`Self::try_reserve`], and
    /// after a call that returned `Err`.
    pub fn last_accepted(&self) -> Option<ExponentialActionAccepted<'_, S::Real>> {
        let columns = self.accepted_columns?;
        Some(ExponentialActionAccepted {
            progress: &self.progress[..columns],
            estimated_errors: &self.errors[..columns],
            completed: &self.completed[..columns],
        })
    }
}
impl<S: NumericScalar> ExponentialCall<S> {
    fn workspace_requirements(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<ExponentialActionRequirements<S>, SciOxideError> {
        crate::checked::expect_shape([self.n, self.n], [op.rows(), op.cols()])?;
        let dim = sum([self.n, self.options.max_phi])?;
        let provider = op
            .apply_workspace_len()?
            .max(op.apply_adjoint_workspace_len()?);
        let payload = sum([
            bytes::<S>(sum([
                product(&[4, dim])?,
                provider,
                product(&[self.n, self.columns])?,
            ])?)?,
            bytes::<ScaledValue<Complex<S::Real>>>(dim)?,
            bytes::<S::Real>(product(&[2, self.columns])?)?,
            bytes::<bool>(self.columns)?,
        ])?;
        ExponentialActionRequirements::new(
            ExponentialLayout {
                n: self.n,
                columns: self.columns,
                phi: self.options.max_phi,
                provider: op
                    .apply_workspace_len()?
                    .max(op.apply_adjoint_workspace_len()?),
            },
            self.options.execution,
            payload,
            Default::default(),
            Default::default(),
        )
    }
    fn workspace(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<ExponentialActionWorkspace<S>, SciOxideError> {
        let requirements = self.workspace_requirements(op)?;
        let mut workspace = ExponentialActionWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    fn apply(
        &self,
        op: &dyn AdjointOperator<S>,
        function: PhiFunction<S::Real>,
        input: BlockVectorView<'_, S>,
    ) -> Result<(Vec<S>, ExponentialActionReport), SciOxideError> {
        let mut workspace = self.workspace(op)?;
        let mut output = scioxide_core::backend::try_filled(
            checked_mul(
                self.n,
                self.columns,
                scioxide_core::quantity!("block_elements"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("output"),
        )?;
        let report = self.apply_into(
            op,
            function,
            input,
            BlockVectorViewMut::new(&mut output, self.n, self.columns)?,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn apply_into<'s, 'c>(
        &self,
        op: &dyn AdjointOperator<S>,
        function: PhiFunction<S::Real>,
        input: BlockVectorView<'_, S>,
        output: BlockVectorViewMut<'_, S>,
        w: &mut ExponentialActionWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<ExponentialActionReport, SciOxideError> {
        w.accepted_columns = None;
        w.slot.reject();
        let requirements = self.workspace_requirements(op)?;
        let bound = self.preflight(function, input, &output)?;
        w.prepare(&requirements)?;
        let (termination, applications) = run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "ExponentialAction", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| {
                self.run(
                    op,
                    function,
                    bound,
                    ExponentialBlock { input, output },
                    w,
                    work,
                )
            },
            |(termination, _)| *termination == SpectralTermination::Converged,
        )?;
        w.accepted_columns = Some(w.columns);
        Ok(ExponentialActionReport {
            termination,
            applications,
        })
    }

    /// Validate the call arguments before an epoch begins and return the
    /// augmented one-norm bound `max(|t| ||A||, 1 + |t mu|)` (the second term
    /// only for phi functions); an unrepresentable bound overflowed from finite operands.
    fn preflight(
        &self,
        function: PhiFunction<S::Real>,
        input: BlockVectorView<'_, S>,
        output: &BlockVectorViewMut<'_, S>,
    ) -> Result<S::Real, SciOxideError> {
        let PhiFunction { time, index: p } = function;
        let shape = [self.n, self.columns];
        crate::checked::expect_shape(shape, [input.rows(), input.right_hand_sides()])?;
        crate::checked::expect_shape(shape, [output.rows(), output.right_hand_sides()])?;
        crate::checked::finite(time, NonFiniteOrigin::Input, "time must be finite")?;
        if p > self.options.max_phi {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("function"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phi index exceeds the reserved maximum",
            ));
        }
        crate::checked::validate_values(
            input.as_slice(),
            NonFiniteOrigin::Input,
            "action input must be finite",
        )?;
        let shift = crate::checked::finite_candidate(self.options.shift * S::from_real(time))?;
        let bound = (time.abs() * self.options.operator_norm_bound).max(if p > 0 {
            S::Real::ONE + shift.abs()
        } else {
            S::Real::ZERO
        });
        crate::checked::finite_candidate(bound)
    }

    fn run(
        &self,
        op: &dyn AdjointOperator<S>,
        function: PhiFunction<S::Real>,
        bound: S::Real,
        block: ExponentialBlock<'_, S>,
        w: &mut ExponentialActionWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<(SpectralTermination, usize), SciOxideError> {
        let PhiFunction { time, index: p } = function;
        let ExponentialBlock { input, mut output } = block;

        let execution = work.execution;

        let n = self.n;
        let columns = self.columns;
        if w.provider.len()
            != op
                .apply_workspace_len()?
                .max(op.apply_adjoint_workspace_len()?)
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("op"),
                "operator changed its scratch requirement after preparation",
            ));
        }
        let dim = n + p;
        if p == 0 {
            w.candidate.copy_from_slice(input.as_slice());
        } else {
            w.candidate.fill(S::ZERO);
        }
        w.progress.fill(S::Real::ZERO);
        w.errors.fill(S::Real::ZERO);
        w.completed.fill(false);
        let mut termination = SpectralTermination::Converged;
        let outcome = (|| {
            execution.check()?;
            let mut exponential_parameters = None;
            for column in 0..columns {
                let b = input.column(column)?;
                if time == S::Real::ZERO {
                    let column_output = &mut w.candidate[column * n..(column + 1) * n];
                    column_output.copy_from_slice(b);
                    for order in 1..=p {
                        let divisor =
                            S::from_real(crate::checked::rounded_count::<S::Real>(order)?);
                        for out in column_output.iter_mut() {
                            *out = crate::checked::finite_candidate(*out / divisor)?;
                        }
                        if column_output.iter().all(|&x| x == S::ZERO) {
                            break;
                        }
                    }
                    w.completed[column] = true;
                    w.progress[column] = S::Real::ONE;
                    continue;
                }
                let bnorm = crate::estimate::vector_one_norm(b)?;
                if bnorm.is_zero() || n == 0 {
                    w.completed[column] = true;
                    w.progress[column] = S::Real::ONE;
                    continue;
                }
                // `phi_p(tA) b` (with `phi_0 = exp`) is computed as a block of
                // `exp(Ã) e_{n+p}` for the implicit augmented matrix
                // `Ã = [tA, b_hat e_1^T; 0, J]`, J the p x p upward shift
                // (Al-Mohy & Higham, 2011). Ã is never formed: a product with it
                // is one application of A plus a shift of the p extra entries.
                // b is normalized by its 1-norm so the augmented entries are O(1),
                // and the result is rescaled at the end.
                let augmented = Augmented {
                    op,
                    time,
                    shift: self.options.shift,
                    p,
                    b,
                    bnorm,
                };
                let selected = if p == 0 && exponential_parameters.is_some() {
                    exponential_parameters
                } else {
                    parameters::select(
                        bound,
                        self.options.terms.get(),
                        p.max(1),
                        if p == 0 { columns } else { 1 },
                        self.options.tolerance.relative(),
                        |power| augmented.power_norm(power, w, &mut work),
                    )?
                };
                let Some(selected) = selected else {
                    termination = SpectralTermination::EvaluationLimit;
                    break;
                };
                if selected.degree == 0 {
                    termination = SpectralTermination::EvaluationLimit;
                    break;
                }
                if p == 0 {
                    exponential_parameters = Some(selected);
                }
                let Parameters { degree, steps } = selected;
                let step_scale = S::Real::ONE / crate::checked::rounded_count::<S::Real>(steps)?;
                w.current[..dim].fill(S::ZERO);
                if p == 0 {
                    for (out, &value) in w.current[..n].iter_mut().zip(b) {
                        *out = if self.options.shift == S::ZERO {
                            value
                        } else {
                            normalized(value, bnorm)?
                        };
                    }
                } else {
                    w.current[dim - 1] = S::ONE;
                }
                if let core::ops::ControlFlow::Break(reason) = self.advance_column(
                    op,
                    w,
                    &mut work,
                    ExponentialColumn {
                        time,
                        p,
                        b,
                        bnorm,
                        steps,
                        step_scale,
                        degree,
                        column,
                    },
                )? {
                    termination = reason;
                }
                if matches!(
                    termination,
                    SpectralTermination::EvaluationLimit
                        | SpectralTermination::IterationLimit
                        | SpectralTermination::ExecutionStopped(_)
                ) {
                    break;
                }
            }
            execution.check()
        })();
        termination = super::action_termination(outcome, termination)?;
        output.as_mut_slice().copy_from_slice(&w.candidate);
        Ok((termination, work.count))
    }
    fn advance_column(
        &self,
        op: &dyn AdjointOperator<S>,
        w: &mut ExponentialActionWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        problem: ExponentialColumn<'_, S>,
    ) -> Result<core::ops::ControlFlow<SpectralTermination>, SciOxideError> {
        let ExponentialColumn {
            time,
            p,
            b,
            bnorm,
            steps,
            step_scale,
            degree,
            column,
        } = problem;
        let n = self.n;
        let dim = n + p;
        // A shifted recurrence carries its magnitude outside the scalar buffers.
        // This prevents either exp(tμ) or exp(t(A−μI)) from exhausting IEEE range
        // before their product is restored with the original RHS magnitude.
        let shifted = self.options.shift != S::ZERO;
        let mut frame = Scaled::ONE;
        let rhs_scale = if p > 0 || shifted { bnorm } else { Scaled::ONE };
        let interval_shift =
            crate::checked::finite_candidate(self.options.shift * S::from_real(time))?;
        let mut termination = core::ops::ControlFlow::Continue(());
        let mut accepted = 0;
        let outcome = (|| {
            'steps: for step in 0..steps {
                let fraction = crate::checked::rounded_count::<S::Real>(step + 1)?
                    / crate::checked::rounded_count::<S::Real>(steps)?;
                let restoration = shift_magnitude(interval_shift, fraction)?;
                let output_scale = rhs_scale.mul(frame)?.mul(restoration)?;
                let absolute_target = Scaled::from_value(self.options.tolerance.absolute())?
                    .mul(bnorm)?
                    .div(output_scale)?;
                w.sum[..dim].copy_from_slice(&w.current[..dim]);
                w.term[..dim].copy_from_slice(&w.current[..dim]);
                let mut tail = Scaled::ZERO;
                // An augmented exponential can be accurate in norm while its
                // tiny phi block has poor relative accuracy. Advance that block
                // until its own truncation target is met, within the term cap.
                // Only that block is measured: the trailing p entries evolve on
                // their own and are advanced exactly after the series.
                let measured = if p > 0 { n } else { dim };
                let mut previous = crate::estimate::vector_one_norm(&w.term[..measured])?;
                // The partial sums carry rounding errors of about `u` times the
                // largest of them, which exceeds the final sum when the series
                // cancels (`exp` of a strongly decaying operator, or the phi
                // block driven by a shifted trailing block); the relative
                // target refers to that largest partial sum.
                let mut peak = crate::estimate::vector_one_norm(&w.sum[..n])?;
                let term_limit = if p > 0 {
                    self.options.terms.get()
                } else {
                    degree
                };
                let mut converged = p == 0;
                for k in 1..=term_limit {
                    if !work.iteration()? {
                        termination =
                            core::ops::ControlFlow::Break(SpectralTermination::EvaluationLimit);
                        break 'steps;
                    }
                    if !(Augmented {
                        op,
                        time,
                        shift: self.options.shift,
                        p,
                        b,
                        bnorm,
                    })
                    .apply(
                        &w.term[..dim],
                        &mut w.next[..dim],
                        &mut w.provider,
                        work,
                        false,
                    )? {
                        termination =
                            core::ops::ControlFlow::Break(SpectralTermination::EvaluationLimit);
                        break 'steps;
                    }
                    let scale = step_scale / crate::checked::rounded_count::<S::Real>(k)?;
                    for i in 0..dim {
                        w.term[i] =
                            crate::checked::finite_candidate(w.next[i] * S::from_real(scale))?;
                        if p > 0
                            && !bnorm.abs_le(Scaled::ONE)
                            && ((w.next[i].real_component() != S::Real::ZERO
                                && w.term[i].real_component() == S::Real::ZERO)
                                || (w.next[i].imaginary_component() != S::Real::ZERO
                                    && w.term[i].imaginary_component() == S::Real::ZERO))
                        {
                            return Err(SciOxideError::numerical_instability(
                                scioxide_core::NumericalIssue::Underflow,
                                "normalized phi term lost a component before restoring RHS scale",
                            ));
                        }
                        w.sum[i] = crate::checked::add(w.sum[i], w.term[i])?;
                    }
                    let termnorm = crate::estimate::vector_one_norm(&w.term[..measured])?;
                    let sum_norm = crate::estimate::vector_one_norm(&w.sum[..n])?;
                    if !sum_norm.abs_le(peak) {
                        peak = sum_norm;
                    }
                    tail = previous.add(termnorm)?;
                    let threshold = absolute_target
                        .add(Scaled::from_value(self.options.tolerance.relative())?.mul(peak)?)?
                        .mul(Scaled::from_value(step_scale)?)?;
                    if k >= p.max(2) && tail.abs_le(threshold) {
                        converged = true;
                        break;
                    }
                    previous = termnorm;
                }
                if !converged {
                    termination =
                        core::ops::ControlFlow::Break(SpectralTermination::IterationLimit);
                    break;
                }
                if p > 0 {
                    // The trailing block of the shifted augmented matrix is
                    // `J − tμI` with J nilpotent, so one step maps it exactly
                    // by `e^(−tμ/s) exp(J/s)`, an upper triangular Toeplitz
                    // matrix with entries `s⁻ʲ/j!`. Its Taylor sum would
                    // suffer the cancellation of the series of `e^(−tμ/s)`.
                    let decay = shift_factor(interval_shift, -step_scale)?;
                    for i in 0..p {
                        let mut value = S::ZERO;
                        let mut coefficient = S::Real::ONE;
                        for (j, &entry) in w.current[n + i..dim].iter().enumerate() {
                            if j > 0 {
                                coefficient = coefficient * step_scale
                                    / crate::checked::rounded_count::<S::Real>(j)?;
                            }
                            value = value + S::from_real(coefficient) * entry;
                        }
                        w.sum[n + i] = ScaledValue::from_value(value)?.mul(decay)?.round()?;
                    }
                }
                // Completing the selected degree satisfies the backward-error
                // selection. Early truncation merely saves remaining products.
                // Only completed steps advance the range frame or the candidate.
                // Renormalization is homogeneous across the entire augmentation.
                let magnitude = if shifted {
                    crate::estimate::vector_one_norm(&w.sum[..dim])?
                } else {
                    Scaled::ONE
                };
                if shifted && !magnitude.is_zero() {
                    for (out, &value) in w.current[..dim].iter_mut().zip(&w.sum[..dim]) {
                        *out = normalized(value, magnitude)?;
                    }
                } else {
                    w.current[..dim].copy_from_slice(&w.sum[..dim]);
                }
                accepted = step + 1;
                w.errors[column] = crate::checked::finite_candidate(
                    w.errors[column] + tail.mul(output_scale)?.round()?,
                )?;
                if shifted && !magnitude.is_zero() {
                    frame = frame.mul(magnitude)?;
                }
            }
            Ok(())
        })();
        let status = super::action_termination(
            outcome,
            match termination {
                core::ops::ControlFlow::Break(reason) => reason,
                core::ops::ControlFlow::Continue(()) => SpectralTermination::Converged,
            },
        )?;
        if status != SpectralTermination::Converged {
            termination = core::ops::ControlFlow::Break(status);
        }
        w.progress[column] = crate::checked::rounded_count::<S::Real>(accepted)?
            / crate::checked::rounded_count::<S::Real>(steps)?;
        w.completed[column] = accepted == steps;
        let progress = w.progress[column];
        let scale = shift_factor(interval_shift, progress)?.mul(ScaledValue::from_components(
            rhs_scale.mul(frame)?,
            Scaled::ZERO,
        ))?;
        for i in 0..n {
            w.candidate[column * n + i] =
                ScaledValue::from_value(w.current[i])?.mul(scale)?.round()?;
        }

        Ok(termination)
    }
}
impl<S: NumericScalar> ExponentialAction<S> {
    /// Checks the settings against `n × n` operators and blocks of `columns`
    /// vectors and returns the per-call value.
    fn call(&self, n: usize, columns: usize) -> Result<ExponentialCall<S>, SciOxideError> {
        let options = *self;
        crate::checked::finite(
            options.operator_norm_bound,
            NonFiniteOrigin::Input,
            "operator norm bound must be finite",
        )?;
        if options.operator_norm_bound < S::Real::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self.operator_norm_bound"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "operator norm bound must be nonnegative",
            ));
        }
        crate::checked::finite(
            options.shift,
            NonFiniteOrigin::Input,
            "operator shift must be finite",
        )?;
        if options.terms.get() > 55 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self.terms"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Taylor degree must be at most 55",
            ));
        }
        if options.max_phi > options.terms.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self.max_phi"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "maximum phi index must not exceed the Taylor degree",
            ));
        }
        checked_mul(n, columns, scioxide_core::quantity!("block_elements"))?;
        checked_add(
            n,
            options.max_phi,
            scioxide_core::quantity!("augmented_dimension"),
        )?;
        Ok(ExponentialCall {
            n,
            columns,
            options,
        })
    }

    /// Returns the order of the square operator `op`.
    fn order(op: &dyn AdjointOperator<S>) -> Result<usize, SciOxideError> {
        crate::checked::square(op.rows(), op.cols())?;
        Ok(op.rows())
    }
    /// Returns the workspace storage needed to apply the action of the `n × n`
    /// operator `op` to blocks of `columns` vectors.
    ///
    /// The operator is not applied; only its shape and scratch lengths are read.
    /// The settings are checked first, so the execution methods report the
    /// same errors before any work.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `op` is not square.
    /// - `NonFinite` if the norm bound or the shift is not finite.
    /// - `InvalidInput` (`OutOfRange`, argument `self.operator_norm_bound`) if
    ///   the norm bound is negative.
    /// - `InvalidInput` (`OutOfRange`, argument `self.terms`) if the Taylor
    ///   degree exceeds 55, or (argument `self.max_phi`) if the largest φ
    ///   index exceeds it.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of `op`.
    pub fn workspace_requirements(
        &self,
        op: &dyn AdjointOperator<S>,
        columns: usize,
    ) -> Result<ExponentialActionRequirements<S>, SciOxideError> {
        self.call(Self::order(op)?, columns)?
            .workspace_requirements(op)
    }

    /// Allocates a workspace for `op`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace(
        &self,
        op: &dyn AdjointOperator<S>,
        columns: usize,
    ) -> Result<ExponentialActionWorkspace<S>, SciOxideError> {
        self.call(Self::order(op)?, columns)?.workspace(op)
    }

    /// Computes `φₖ(tA) B` with a new workspace and returns the output block.
    ///
    /// The returned vector stores the `n` entries of each column one column
    /// after another, as in `input`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::apply_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::BlockVectorView;
    /// use scioxide_linalg::{ExponentialAction, Matrix, PhiFunction};
    /// let a = Matrix::<f64>::identity(2)?;
    /// let method = ExponentialAction::new(1.);
    /// let (output, report) = method.apply(
    ///     &a.view().try_finite()?,
    ///     PhiFunction::exp(0.),
    ///     BlockVectorView::new(&[1., 2.], 2, 1)?,
    /// )?;
    /// // exp(0·A) b = b.
    /// assert_eq!(output, [1., 2.]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply(
        &self,
        op: &dyn AdjointOperator<S>,
        function: PhiFunction<S::Real>,
        input: BlockVectorView<'_, S>,
    ) -> Result<(Vec<S>, ExponentialActionReport), SciOxideError> {
        self.call(Self::order(op)?, input.right_hand_sides())?
            .apply(op, function, input)
    }

    /// Computes `function` applied to `input`, `φₖ(tA) B`, and writes it to `output`.
    ///
    /// The one-norm estimates are lower estimates used only to choose the
    /// degree and step count; the norm bound of the options still has to hold.
    /// Their adjoint products count toward the application limit together with
    /// the forward products. For a φ-function, Taylor terms are added up to the
    /// degree limit until the tolerance is met; otherwise the status is
    /// `IterationLimit`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `input` or `output` is not `n × columns`.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `NonFinite` if the time or an entry of `input` is not finite, or if
    ///   `op` returns a non-finite value.
    /// - `InvalidInput` (`OutOfRange`) if the φ index exceeds `max_phi`.
    /// - `InvalidInput` (`Inconsistent`) if the scratch length of `op` changed
    ///   after the workspace was reserved.
    /// - `NumericalInstability` (`Overflow`) if `|t| · bound`, `tμ` or an
    ///   intermediate value overflows, and (`Underflow`) if a scaled φ term
    ///   underflows before the norm of `b` is restored.
    /// - Errors returned by `op`.
    pub fn apply_into<'s, 'c>(
        &self,
        op: &dyn AdjointOperator<S>,
        function: PhiFunction<S::Real>,
        input: BlockVectorView<'_, S>,
        output: BlockVectorViewMut<'_, S>,
        w: &mut ExponentialActionWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<ExponentialActionReport, SciOxideError> {
        self.call(Self::order(op)?, input.right_hand_sides())?
            .apply_into(op, function, input, output, w, context)
    }
}

/// Magnitude and phase are kept separate until the caller's range frame is known.
fn shift_magnitude<S: NumericScalar>(
    shift: S,
    fraction: S::Real,
) -> Result<Scaled<S::Real>, SciOxideError> {
    Scaled::from_log(S::Real::ONE, shift.real_component() * fraction)
}

fn shift_factor<S: NumericScalar>(
    shift: S,
    fraction: S::Real,
) -> Result<ScaledValue<S>, SciOxideError> {
    let angle = shift.imaginary_component() * fraction;
    let phase = S::from_components(angle.cos(), angle.sin());
    ScaledValue::from_value(phase)?.mul(ScaledValue::from_components(
        shift_magnitude(shift, fraction)?,
        Scaled::ZERO,
    ))
}

fn normalized<S: NumericScalar>(value: S, norm: Scaled<S::Real>) -> Result<S, SciOxideError> {
    ScaledValue::from_value(value)?
        .div(ScaledValue::from_components(norm, Scaled::ZERO))?
        .try_value()
}

struct ExponentialColumn<'a, S: NumericScalar> {
    time: S::Real,
    p: usize,
    b: &'a [S],
    bnorm: Scaled<S::Real>,
    steps: usize,
    step_scale: S::Real,
    degree: usize,
    column: usize,
}

struct ExponentialBlock<'a, S: NumericScalar> {
    input: BlockVectorView<'a, S>,
    output: BlockVectorViewMut<'a, S>,
}

/// The augmented operator is implicit for both forward and adjoint products.
/// For phi_p, its top-right block is b/||b||_1 in column zero of the shift.
struct Augmented<'a, S: NumericScalar> {
    op: &'a dyn AdjointOperator<S>,
    time: S::Real,
    shift: S,
    p: usize,
    b: &'a [S],
    bnorm: Scaled<S::Real>,
}
impl<S: NumericScalar> Augmented<'_, S> {
    fn apply(
        &self,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
        work: &mut Work<'_, '_, '_>,
        adjoint: bool,
    ) -> Result<bool, SciOxideError> {
        let n = self.b.len();
        let admitted = if adjoint {
            work.adjoint(
                self.op,
                &x[..n],
                &mut y[..n],
                &mut scratch[..self.op.apply_adjoint_workspace_len()?],
            )?
        } else {
            work.apply(
                self.op,
                &x[..n],
                &mut y[..n],
                &mut scratch[..self.op.apply_workspace_len()?],
            )?
        };
        if !admitted {
            return Ok(false);
        }
        let mu = if adjoint {
            self.shift.conj()
        } else {
            self.shift
        };
        for (v, &x) in y[..n].iter_mut().zip(x) {
            *v = crate::checked::finite_candidate((*v - mu * x) * S::from_real(self.time))?;
        }
        if self.p > 0 {
            if adjoint {
                let mut sum = ScaledValue::ZERO;
                for (&b, &x) in self.b.iter().zip(x) {
                    sum = sum.add(
                        ScaledValue::from_value(normalized(b, self.bnorm)?.conj())?
                            .mul(ScaledValue::from_value(x)?)?,
                    )?;
                }
                y[n] = sum.round()?;
                y[n + 1..].copy_from_slice(&x[n..x.len() - 1]);
            } else {
                for (y, &b) in y[..n].iter_mut().zip(self.b) {
                    *y = crate::checked::finite_candidate(*y + normalized(b, self.bnorm)? * x[n])?;
                }
                y[n..x.len() - 1].copy_from_slice(&x[n + 1..]);
                y[x.len() - 1] = S::ZERO;
            }
        }
        let shift = mu * S::from_real(self.time);
        for (v, &x) in y[n..].iter_mut().zip(&x[n..]) {
            *v = crate::checked::finite_candidate(*v - shift * x)?;
        }
        Ok(true)
    }

    fn power_norm(
        &self,
        power: usize,
        w: &mut ExponentialActionWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Option<S::Real>, SciOxideError> {
        let dim = self.b.len() + self.p;
        let estimate =
            crate::estimate::operator_one_norm(&mut w.scaled[..dim], |values, adjoint| {
                for _ in 0..power {
                    if !work.iteration()? {
                        return Ok(false);
                    }
                    let mut maximum = Scaled::ZERO;
                    for &v in values.iter() {
                        let magnitude = v.magnitude()?;
                        if !magnitude.abs_le(maximum) {
                            maximum = magnitude;
                        }
                    }
                    if maximum.is_zero() {
                        return Ok(true);
                    }
                    // Normalize each product, tracking the removed scale in extended
                    // range. The dimension factor also protects adjoint row sums.
                    let scale =
                        maximum.mul(Scaled::from_value(crate::checked::rounded_count::<
                            S::Real,
                        >(dim)?)?)?;
                    let factor = ScaledValue::from_components(scale, Scaled::ZERO);
                    for (x, &v) in w.term[..dim].iter_mut().zip(values.iter()) {
                        let z = v.div(factor)?.round()?;
                        *x = S::from_components(z.re, z.im);
                    }
                    if !self.apply(
                        &w.term[..dim],
                        &mut w.next[..dim],
                        &mut w.provider,
                        work,
                        adjoint,
                    )? {
                        return Ok(false);
                    }
                    for (v, &y) in values.iter_mut().zip(&w.next[..dim]) {
                        *v = ScaledValue::from_value(Complex::new(
                            y.real_component(),
                            y.imaginary_component(),
                        ))?
                        .mul(factor)?;
                    }
                }
                Ok(true)
            })?;
        estimate
            .complete
            .then_some(estimate.value)
            .map(|v| {
                if v.is_zero() {
                    Ok(S::Real::ZERO)
                } else {
                    let root =
                        (v.ln_abs()? / crate::checked::rounded_count::<S::Real>(power)?).exp();
                    if root.is_finite() {
                        Ok(root)
                    } else {
                        Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Overflow,
                            "power norm is not representable",
                        ))
                    }
                }
            })
            .transpose()
    }
}

impl<S: NumericScalar> Default for ExponentialActionWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> ExponentialActionWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            accepted_columns: None,
            slot: SolveSlot::new(),
            n: 0,
            columns: 0,
            max_phi: 0,
            scaled: Vec::new(),
            current: Vec::new(),
            sum: Vec::new(),
            term: Vec::new(),
            next: Vec::new(),
            provider: Vec::new(),
            candidate: Vec::new(),
            progress: Vec::new(),
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
        requirements: &ExponentialActionRequirements<S>,
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
        requirements: &ExponentialActionRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for ExponentialActionWorkspace<S> {
    type Requirements = ExponentialActionRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &ExponentialActionRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let ExponentialLayout {
            n,
            columns,
            phi,
            provider,
        } = requirements.layout;
        let dim = sum([n, phi])?;
        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.scaled, dim)?;
        plan.buffer(&self.current, dim)?;
        plan.buffer(&self.sum, dim)?;
        plan.buffer(&self.term, dim)?;
        plan.buffer(&self.next, dim)?;
        plan.buffer(&self.provider, provider)?;
        plan.buffer(&self.candidate, product(&[n, columns])?)?;
        plan.buffer(&self.progress, columns)?;
        plan.buffer(&self.errors, columns)?;
        plan.buffer(&self.completed, columns)?;
        Ok(plan)
    }
    fn configure(
        &mut self,
        requirements: &ExponentialActionRequirements<S>,
    ) -> Result<(), SciOxideError> {
        self.accepted_columns = None;
        let ExponentialLayout {
            n,
            columns,
            phi,
            provider,
        } = requirements.layout;
        let dim = sum([n, phi])?;
        resize(&mut self.scaled, dim, ScaledValue::ZERO)?;
        resize(&mut self.current, dim, S::ZERO)?;
        resize(&mut self.sum, dim, S::ZERO)?;
        resize(&mut self.term, dim, S::ZERO)?;
        resize(&mut self.next, dim, S::ZERO)?;
        resize(&mut self.provider, provider, S::ZERO)?;
        resize(&mut self.candidate, product(&[n, columns])?, S::ZERO)?;
        resize(&mut self.progress, columns, S::Real::ZERO)?;
        resize(&mut self.errors, columns, S::Real::ZERO)?;
        resize(&mut self.completed, columns, false)?;
        self.n = n;
        self.columns = columns;
        self.max_phi = phi;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for ExponentialActionWorkspace<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.scaled.storage_bytes()?,
            self.current.storage_bytes()?,
            self.sum.storage_bytes()?,
            self.term.storage_bytes()?,
            self.next.storage_bytes()?,
            self.provider.storage_bytes()?,
            self.candidate.storage_bytes()?,
            self.progress.storage_bytes()?,
            self.errors.storage_bytes()?,
            self.completed.storage_bytes()?,
        ])
    }
}
