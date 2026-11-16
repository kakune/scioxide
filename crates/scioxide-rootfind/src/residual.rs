//! Residual maps for the nonlinear-system and fixed-point solvers.
//!
//! Every solver accepts the core [`ResidualOperator`] on slices. The dense
//! Newton, Broyden and trust-region solvers also use the optional
//! [`DenseJacobian`] trait, and [`ResidualFn`] wraps closures. One checked
//! evaluation, [`evaluate`], checks shapes, charges the call to the solve,
//! passes the solve context and checks the output of every residual call. Its
//! variants differ only in how the call is charged.

use core::marker::PhantomData;

use scioxide_core::{
    Dimensions, EvaluationRole, InvalidInputReason, JacobianOperator, NonFiniteOrigin,
    NumericScalar, RealScalar, ResidualOperator, SciOxideError,
};
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy, FiniteDifferenceWorkspace};

use crate::context::{KernelResult, Session};
use scioxide_core::SolveContext;

/// Supplies the dense Jacobian `J(x) = ∂F/∂x` of a square residual map.
///
/// The dense solvers ([`crate::NewtonSystem`], [`crate::Broyden`] and
/// [`crate::TrustRegionSystem`]) take problems through this trait. Every method
/// has a default, so an empty implementation is enough when the Jacobian comes
/// from finite differences (see [`DerivativeStrategy`]). The solvers try the
/// sources of their strategy in order, and `Ok(false)` marks a source as
/// unavailable. The default strategy, `AnalyticOrForward`, uses
/// [`jacobian_into`](Self::jacobian_into) and falls back to forward
/// differences. With `Analytic`, a problem without an analytic Jacobian is an
/// `InvalidInput` error with reason `UnsupportedConfiguration`.
///
/// The Jacobian is stored row-major, `output_dimension × input_dimension`, so
/// entry `(i, j)` is `∂Fᵢ/∂xⱼ`. A call receives exactly
/// [`Self::jacobian_workspace_len`] scratch values and must write every entry
/// when it returns `Ok(true)`. The solvers reject NaN and infinite entries.
/// For complex scalars the Jacobian is the complex derivative of a holomorphic
/// `F`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{SciOxideError, ResidualOperator};
/// use scioxide_rootfind::{DenseJacobian, NewtonSystem, NewtonSystemOptions, SystemRootSolver};
///
/// /// `F(x) = x² − 2` componentwise, with its diagonal Jacobian.
/// struct Squares(usize);
/// impl ResidualOperator<f64> for Squares {
///     fn input_dimension(&self) -> usize { self.0 }
///     fn output_dimension(&self) -> usize { self.0 }
///     fn residual_workspace_len(&self) -> Result<usize, SciOxideError> { Ok(0) }
///     fn residual_into(&self, x: &[f64], out: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
///         out.iter_mut().zip(x).for_each(|(out, x)| *out = x * x - 2.0);
///         Ok(())
///     }
/// }
/// impl DenseJacobian<f64> for Squares {
///     fn jacobian_into(&self, x: &[f64], out: &mut [f64], _: &mut [f64]) -> Result<bool, SciOxideError> {
///         out.fill(0.0);
///         (0..self.0).for_each(|i| out[i * self.0 + i] = 2.0 * x[i]);
///         Ok(true)
///     }
/// }
///
/// let system = Squares(3);
/// let result = NewtonSystem::new(NewtonSystemOptions::default())?.solve(&system, &[1.0; 3])?;
/// assert!(result.is_success());
/// assert!(result.point().iter().all(|x| (x - 2.0_f64.sqrt()).abs() < 1e-8));
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait DenseJacobian<T: NumericScalar>: ResidualOperator<T> {
    /// Returns the number of scratch values one Jacobian call needs (default 0).
    ///
    /// # Errors
    ///
    /// An implementation returns `SizeOverflow` if the count overflows `usize`,
    /// or an error that describes an invalid configuration.
    fn jacobian_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }

    /// Writes the analytic Jacobian at `point` to `output` and returns
    /// `Ok(true)`, or returns `Ok(false)` if there is none (the default).
    ///
    /// # Errors
    ///
    /// An implementation returns `DimensionMismatch` for slices of the wrong
    /// length, `DomainError` if `point` is outside the domain of `F`, or its own
    /// numerical errors. The solver returns them unchanged.
    fn jacobian_into(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<bool, SciOxideError> {
        let _ = (point, output, scratch);
        Ok(false)
    }

    /// Writes a Jacobian computed by automatic differentiation at `point` to
    /// `output` and returns `Ok(true)`, or returns `Ok(false)` if there is none
    /// (the default).
    ///
    /// # Errors
    ///
    /// As for [`Self::jacobian_into`].
    fn automatic_jacobian_into(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<bool, SciOxideError> {
        let _ = (point, output, scratch);
        Ok(false)
    }
}

/// Wraps closures as a residual map, with optional Jacobians and
/// Jacobian–vector products.
///
/// Two shapes are supported, for real or complex scalars `T`:
///
/// - [`Self::new`] wraps a square map `F: Tⁿ → Tⁿ`, the shape of the dense,
///   fixed-point and Newton–Krylov solvers.
/// - [`Self::with_parameter`] wraps `F: Tⁿ⁺¹ → Tⁿ`, whose last input is the
///   parameter `λ`, the shape of [`crate::ContinuationProblem`].
///
/// The residual closure writes `F(x)` to its output slice. The Jacobian
/// closures of [`Self::with_jacobian`] and [`Self::with_automatic_jacobian`]
/// write the `m × p` Jacobian in row-major order, where `m` is the number of
/// outputs and `p` the number of inputs. The closure of [`Self::with_jvp`]
/// receives `(x, v, out)` and writes the Jacobian–vector product `J(x)·v`.
///
/// The wrapper needs no scratch and checks only slice lengths
/// (`DimensionMismatch`); the solvers reject NaN and infinite outputs. It
/// implements [`ResidualOperator`] and [`DenseJacobian`], so a square map works
/// with every dense and fixed-point solver in this crate and, through
/// [`crate::NewtonKrylovProblem::numerical`], with [`crate::NewtonKrylovSolver`].
/// The dense solvers reject the parameter shape with `DimensionMismatch`. Only
/// a wrapper with a closure from [`Self::with_jvp`] implements
/// [`JacobianOperator`], which continuation, deflation, [`crate::LinearHomotopy`]
/// and [`crate::NewtonKrylovProblem::with_jacobian`] take.
///
/// # Examples
///
/// ```
/// use scioxide_core::ResidualOperator;
/// use scioxide_rootfind::{DenseJacobian, ResidualFn};
///
/// // F(x, y) = (x² + y² − 1, x − y) at (1, 0) is (0, 1).
/// let circle = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
///     out[1] = x[0] - x[1];
///     Ok(())
/// })
/// .with_jacobian(|x: &[f64], out: &mut [f64]| {
///     out.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 1.0, -1.0]);
///     Ok(())
/// });
/// let mut residual = [0.0; 2];
/// circle.residual_into(&[1.0, 0.0], &mut residual, &mut [])?;
/// assert_eq!(residual, [0.0, 1.0]);
/// let mut jacobian = [0.0; 4];
/// assert!(circle.jacobian_into(&[1.0, 0.0], &mut jacobian, &mut [])?);
/// assert!(!circle.automatic_jacobian_into(&[1.0, 0.0], &mut jacobian, &mut [])?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// A map without [`Self::with_jvp`] is not a [`JacobianOperator`], so
/// [`crate::ContinuationProblem::new`] does not accept it:
///
/// ```compile_fail,E0277
/// use scioxide_rootfind::{ContinuationProblem, ResidualFn};
///
/// let line = ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
///     out[0] = y[0] - y[1];
///     Ok(())
/// });
/// let problem = ContinuationProblem::new(&line);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ResidualFn<
    T,
    F,
    J = fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    A = fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    P = (),
> {
    inputs: usize,
    outputs: usize,
    residual: F,
    jacobian: Option<J>,
    automatic: Option<A>,
    jvp: P,
    scalar: PhantomData<fn() -> T>,
}

impl<T, F> ResidualFn<T, F>
where
    T: NumericScalar,
    F: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Wraps a residual closure `F: Tⁿ → Tⁿ` with `dimension = n` unknowns and
    /// equations.
    ///
    /// A zero dimension is rejected by the solvers, not here.
    pub const fn new(dimension: usize, residual: F) -> Self {
        Self::shaped(dimension, dimension, residual)
    }

    /// Wraps a residual closure `F: Tⁿ⁺¹ → Tⁿ` with `dimension = n` equations,
    /// whose last input is the parameter `λ`.
    ///
    /// The input dimension saturates at `usize::MAX`. A zero dimension is
    /// rejected by [`crate::ContinuationProblem::new`], not here.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{JacobianOperator, ResidualOperator};
    /// use scioxide_rootfind::ResidualFn;
    ///
    /// // F(x, λ) = x² − λ and its Jacobian–vector product 2x·v₀ − v₁.
    /// let fold = ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
    ///     out[0] = y[0] * y[0] - y[1];
    ///     Ok(())
    /// })
    /// .with_jvp(|y: &[f64], v: &[f64], out: &mut [f64]| {
    ///     out[0] = 2.0 * y[0] * v[0] - v[1];
    ///     Ok(())
    /// });
    /// assert_eq!((fold.input_dimension(), fold.output_dimension()), (2, 1));
    /// let mut product = [0.0];
    /// fold.jvp_into(&[3.0, 9.0], &[1.0, 1.0], &mut product, &mut [])?;
    /// assert!((product[0] - 5.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn with_parameter(dimension: usize, residual: F) -> Self {
        Self::shaped(dimension.saturating_add(1), dimension, residual)
    }

    const fn shaped(inputs: usize, outputs: usize, residual: F) -> Self {
        Self {
            inputs,
            outputs,
            residual,
            jacobian: None,
            automatic: None,
            jvp: (),
            scalar: PhantomData,
        }
    }
}

impl<T: NumericScalar, F, J, A, P> ResidualFn<T, F, J, A, P> {
    /// Adds a closure that writes the analytic Jacobian in row-major order.
    #[must_use]
    pub fn with_jacobian<J2>(self, jacobian: J2) -> ResidualFn<T, F, J2, A, P>
    where
        J2: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    {
        ResidualFn {
            inputs: self.inputs,
            outputs: self.outputs,
            residual: self.residual,
            jacobian: Some(jacobian),
            automatic: self.automatic,
            jvp: self.jvp,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes a Jacobian computed by automatic
    /// differentiation, in row-major order.
    #[must_use]
    pub fn with_automatic_jacobian<A2>(self, automatic: A2) -> ResidualFn<T, F, J, A2, P>
    where
        A2: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    {
        ResidualFn {
            inputs: self.inputs,
            outputs: self.outputs,
            residual: self.residual,
            jacobian: self.jacobian,
            automatic: Some(automatic),
            jvp: self.jvp,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that receives `(x, v, out)` and writes the
    /// Jacobian–vector product `J(x)·v`, so that the wrapper implements
    /// [`JacobianOperator`].
    ///
    /// The product is not checked for finiteness here; the solvers reject NaN
    /// and infinite products.
    #[must_use]
    pub fn with_jvp<P2>(self, jvp: P2) -> ResidualFn<T, F, J, A, P2>
    where
        P2: Fn(&[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
    {
        ResidualFn {
            inputs: self.inputs,
            outputs: self.outputs,
            residual: self.residual,
            jacobian: self.jacobian,
            automatic: self.automatic,
            jvp,
            scalar: PhantomData,
        }
    }

    /// Runs a Jacobian closure after checking the slice lengths.
    fn assemble(
        &self,
        closure: Option<&impl Fn(&[T], &mut [T]) -> Result<(), SciOxideError>>,
        point: &[T],
        output: &mut [T],
    ) -> Result<bool, SciOxideError> {
        let Some(closure) = closure else {
            return Ok(false);
        };
        expect_len(self.inputs, point.len())?;
        expect_len(
            scioxide_core::backend::checked_mul(
                self.outputs,
                self.inputs,
                scioxide_core::quantity!("jacobian_entries"),
            )?,
            output.len(),
        )?;
        closure(point, output)?;
        Ok(true)
    }
}

impl<T, F, J, A, P> ResidualOperator<T> for ResidualFn<T, F, J, A, P>
where
    T: NumericScalar,
    F: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn input_dimension(&self) -> usize {
        self.inputs
    }
    fn output_dimension(&self) -> usize {
        self.outputs
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        point: &[T],
        output: &mut [T],
        _scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        expect_len(self.inputs, point.len())?;
        expect_len(self.outputs, output.len())?;
        (self.residual)(point, output)
    }
}

impl<T, F, J, A, P> DenseJacobian<T> for ResidualFn<T, F, J, A, P>
where
    T: NumericScalar,
    F: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    J: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    A: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn jacobian_into(
        &self,
        point: &[T],
        output: &mut [T],
        _scratch: &mut [T],
    ) -> Result<bool, SciOxideError> {
        self.assemble(self.jacobian.as_ref(), point, output)
    }
    fn automatic_jacobian_into(
        &self,
        point: &[T],
        output: &mut [T],
        _scratch: &mut [T],
    ) -> Result<bool, SciOxideError> {
        self.assemble(self.automatic.as_ref(), point, output)
    }
}

impl<T, F, J, A, P> JacobianOperator<T> for ResidualFn<T, F, J, A, P>
where
    T: NumericScalar,
    F: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    P: Fn(&[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        _scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        expect_len(self.inputs, point.len())?;
        expect_len(self.inputs, direction.len())?;
        expect_len(self.outputs, output.len())?;
        (self.jvp)(point, direction, output)
    }
}

/// Require `actual == expected` for a vector length.
pub(crate) fn expect_len(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(actual),
        ))
    }
}

/// The leading `len` scalars of borrowed scratch.
pub(crate) fn scratch_slice<T>(scratch: &mut [T], len: usize) -> Result<&mut [T], SciOxideError> {
    let available = scratch.len();
    scratch.get_mut(..len).ok_or_else(|| {
        SciOxideError::dimension_mismatch(Dimensions::one(len), Dimensions::one(available))
    })
}

/// Reject non-finite components, classified by `origin`.
pub(crate) fn finite<T: NumericScalar>(
    values: &[T],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values.iter().all(|value| value.has_finite_components()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "every component must be finite",
        ))
    }
}

/// The positive, square dimension of `function`; `argument` names it.
pub(crate) fn square_dimension<T: NumericScalar, F: ResidualOperator<T> + ?Sized>(
    function: &F,
    argument: scioxide_core::Argument,
) -> Result<usize, SciOxideError> {
    let dimension = function.input_dimension();
    expect_len(dimension, function.output_dimension())?;
    if dimension == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "system dimension must be at least one",
        ));
    }
    Ok(dimension)
}

/// Scratch serving every residual and Jacobian call of `function`.
pub(crate) fn scratch_len<T: NumericScalar, F: DenseJacobian<T> + ?Sized>(
    function: &F,
) -> Result<usize, SciOxideError> {
    Ok(function
        .residual_workspace_len()?
        .max(function.jacobian_workspace_len()?))
}

/// Check the point, output and scratch shapes of one residual call.
///
/// Returns exactly the `residual_workspace_len()` leading scratch scalars.
fn checked<'s, T, F>(
    function: &F,
    point: &[T],
    output: &[T],
    scratch: &'s mut [T],
) -> Result<&'s mut [T], SciOxideError>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    expect_len(function.input_dimension(), point.len())?;
    expect_len(function.output_dimension(), output.len())?;
    scratch_slice(scratch, function.residual_workspace_len()?)
}

/// Poison `output` so components a callback leaves unwritten fail the finite check.
fn poison<T: NumericScalar>(output: &mut [T]) {
    output.fill(T::from_real(T::Real::NAN));
}

/// The one checked residual evaluation of the nonlinear solvers.
///
/// Every variant performs the same steps: check the point, output and scratch
/// shapes (the callback receives exactly `residual_workspace_len()` scratch
/// values, never a panicking slice); charge the call under `role`; invoke the
/// callback with the solve's context; and require every component to be
/// finite. The variants differ only in how the call is charged:
///
/// - [`evaluate`] charges through a kernel [`Session`], whose stop is an error.
///   The dense and fixed-point solvers use it on cheap maps, where poisoning
///   would be a measurable extra pass, so a callback must write every component.
/// - [`evaluate_scoped`] charges a raw [`SolveContext`] and reports a stop as
///   [`crate::context::KernelFault::Stopped`];
/// - [`evaluate_unadmitted`] is the uncounted path for callers that hold no
///   solve context.
///
/// The last two serve Newton–Krylov and continuation and first poison the
/// output with NaN, so components a callback leaves unwritten fail the check.
///
/// On error `output` holds unspecified values, so callers that must keep their
/// output unchanged evaluate into a separate buffer.
pub(crate) fn evaluate<T, F>(
    function: &F,
    role: EvaluationRole,
    point: &[T],
    output: &mut [T],
    scratch: &mut [T],
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    evaluate_raw(function, role, point, output, scratch, session)?;
    finite(output, NonFiniteOrigin::Callback)
}

/// A trial may leave the residual's mathematical domain without failing the solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrialStatus {
    Finite,
    Rejected,
}

/// Classifies only the callback result, never input validation or budget checks.
pub(crate) fn trial_status<T: NumericScalar>(
    result: Result<(), SciOxideError>,
    output: &[T],
) -> Result<TrialStatus, SciOxideError> {
    match result {
        Ok(()) => Ok(if output.iter().all(|v| v.has_finite_components()) {
            TrialStatus::Finite
        } else {
            TrialStatus::Rejected
        }),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
                    | scioxide_core::SciOxideErrorView::DomainError {
                        reason: scioxide_core::DomainReason::OutsideDomain,
                        ..
                    }
            ) =>
        {
            Ok(TrialStatus::Rejected)
        }
        Err(error) => Err(error),
    }
}

/// [`evaluate`] without imposing the accepted-point finite policy.
pub(crate) fn evaluate_raw<T, F>(
    function: &F,
    role: EvaluationRole,
    point: &[T],
    output: &mut [T],
    scratch: &mut [T],
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    let scratch = checked(function, point, output, scratch)?;
    session.invoke_context(role, |context| {
        function.residual_into_scoped(point, output, scratch, context)
    })
}

/// [`evaluate_scoped`] without the finite policy.
///
/// Shape and budget errors are faults; a stop or exhausted budget is
/// [`crate::context::KernelFault::Stopped`]. The callback's own result is returned unchanged so
/// a caller can classify it, for example as a rejected trial.
pub(crate) fn evaluate_scoped_raw<T, F>(
    function: &F,
    role: EvaluationRole,
    point: &[T],
    output: &mut [T],
    scratch: &mut [T],
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Result<(), SciOxideError>>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    let scratch = checked(function, point, output, scratch)?;
    crate::context::admit(context, role)?;
    poison(output);
    Ok(function.residual_into_scoped(point, output, scratch, context))
}

/// [`evaluate`] charged to a raw solve context, with a stop as a value.
///
/// A callback error converts like any kernel error, so a nested stop or work
/// limit it reports also becomes [`crate::context::KernelFault::Stopped`].
pub(crate) fn evaluate_scoped<T, F>(
    function: &F,
    role: EvaluationRole,
    point: &[T],
    output: &mut [T],
    scratch: &mut [T],
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<()>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    evaluate_scoped_raw(function, role, point, output, scratch, context)??;
    Ok(finite(output, NonFiniteOrigin::Callback)?)
}

/// [`evaluate`] for a caller without a solve context: not counted and without
/// a context for nested work.
pub(crate) fn evaluate_unadmitted<T, F>(
    function: &F,
    point: &[T],
    output: &mut [T],
    scratch: &mut [T],
) -> Result<(), SciOxideError>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    let scratch = checked(function, point, output, scratch)?;
    poison(output);
    function.residual_into(point, output, scratch)?;
    finite(output, NonFiniteOrigin::Callback)
}

/// Assemble the Jacobian at `point`, where `F(point) = base`, into `output`
/// from the first available source of `strategy`.
///
/// Every attempted source counts as one Jacobian call. Finite differences
/// reuse `base`, evaluate through [`evaluate`] and need reserved storage. The
/// assembled entries must be finite; on error `output` holds unspecified values.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn jacobian_into<S, F>(
    function: &F,
    point: &[S],
    base: &[S],
    strategy: DerivativeStrategy,
    output: &mut [S],
    scratch: &mut [S],
    finite_difference: Option<&mut FiniteDifferenceWorkspace<S>>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    let dimension = function.input_dimension();
    let mut finite_difference = finite_difference;
    let _ = strategy.select(|source| {
        session.admit(EvaluationRole::Jacobian)?;
        let available = match source {
            DerivativeCandidate::Analytic => function.jacobian_into(
                point,
                output,
                scratch_slice(scratch, function.jacobian_workspace_len()?)?,
            )?,
            DerivativeCandidate::AutomaticDifferentiation => function.automatic_jacobian_into(
                point,
                output,
                scratch_slice(scratch, function.jacobian_workspace_len()?)?,
            )?,
            DerivativeCandidate::FiniteDifference(formula) => {
                let buffers = finite_difference.as_deref_mut().ok_or_else(|| {
                    SciOxideError::invalid_input(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "finite differences were not reserved",
                    )
                })?;
                scioxide_differentiate::FiniteDifference::default()
                    .with_scheme(formula)
                    .jacobian_into(
                        |perturbed: &[S], values: &mut [S]| {
                            evaluate(
                                function,
                                EvaluationRole::Residual,
                                perturbed,
                                values,
                                scratch,
                                session,
                            )
                        },
                        point,
                        dimension,
                        None,
                        Some(base),
                        output,
                        buffers,
                        scioxide_core::ExecutionContext::sequential(),
                    )?
                    .output_available()
            }
        };
        Ok(available.then_some(()))
    })?;
    finite(output, NonFiniteOrigin::Callback)
}

/// The solver's view of a residual handed to a line search: every call goes
/// through [`evaluate_raw`], so it is counted and receives the solve context. The search
/// rejects non-finite trial output and recoverable callback errors.
pub(crate) struct Admitted<'a, 'context, 'state, 'capability, F: ?Sized> {
    function: &'a F,
    session: &'a Session<'context, 'state, 'capability>,
    pending: core::cell::Cell<bool>,
}

impl<'a, 'context, 'state, 'capability, F: ?Sized> Admitted<'a, 'context, 'state, 'capability, F> {
    pub(crate) const fn new(
        function: &'a F,
        session: &'a Session<'context, 'state, 'capability>,
    ) -> Self {
        Self {
            function,
            session,
            pending: core::cell::Cell::new(false),
        }
    }

    /// The last callback was accepted only when the search returned a step.
    pub(crate) fn finish(&self, accepted: bool) -> Result<(), SciOxideError> {
        if self.pending.replace(false) && !accepted {
            self.session.reject()?;
        }
        Ok(())
    }
}

impl<T, F> ResidualOperator<T> for Admitted<'_, '_, '_, '_, F>
where
    T: NumericScalar,
    F: ResidualOperator<T> + ?Sized,
{
    fn input_dimension(&self) -> usize {
        self.function.input_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.function.output_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.function.residual_workspace_len()
    }
    fn residual_into(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        if self.pending.replace(true) {
            self.session.reject()?;
        }
        evaluate_raw(
            self.function,
            EvaluationRole::Residual,
            point,
            output,
            scratch,
            self.session,
        )
    }
}

/// Operator dimensions, used only to size workspaces without allocating.
pub(crate) struct LinearShape<T> {
    dimension: usize,
    scratch: usize,
    scalar: core::marker::PhantomData<T>,
}
impl<T> LinearShape<T> {
    pub(crate) const fn new(dimension: usize, scratch: usize) -> Self {
        Self {
            dimension,
            scratch,
            scalar: core::marker::PhantomData,
        }
    }
}
impl<T: scioxide_core::NumericScalar> scioxide_core::LinearOperator<T> for LinearShape<T> {
    fn rows(&self) -> usize {
        self.dimension
    }
    fn cols(&self) -> usize {
        self.dimension
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_into(
        &self,
        _: &[T],
        _: &mut [T],
        _: &mut [T],
    ) -> Result<(), scioxide_core::SciOxideError> {
        Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("operator"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "operator geometry cannot execute",
        ))
    }
}
