//! [`OdeFn`]: an ODE system built from closures.

use core::marker::PhantomData;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::MatrixViewMut;

use crate::traits::OdeSystem;

/// Marks an [`OdeFn`] that has no closure for the corresponding Jacobian.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoJacobian;

/// Crate-private evaluation of an optional Jacobian callback.
pub(crate) trait CallJacobian<S: RealScalar, T: NumericScalar<Real = S>> {
    fn call(
        &self,
        t: S,
        y: &[T],
        jacobian: &mut MatrixViewMut<'_, T>,
    ) -> Result<Option<()>, SciOxideError>;
}

/// The Jacobian slot of an [`OdeFn`]: either [`NoJacobian`] or a closure
/// `Fn(S, &[T], &mut MatrixViewMut<'_, T>) -> Result<(), SciOxideError>` that writes
/// `∂f/∂y` at `(t, y)`.
///
/// The trait is sealed and implemented only for these two cases.
#[allow(private_bounds)]
pub trait JacobianCallback<S: RealScalar, T: NumericScalar<Real = S> = S>:
    CallJacobian<S, T>
{
}

impl<S: RealScalar, T: NumericScalar<Real = S>, C: CallJacobian<S, T>> JacobianCallback<S, T>
    for C
{
}

impl<S: RealScalar, T: NumericScalar<Real = S>> CallJacobian<S, T> for NoJacobian {
    fn call(
        &self,
        _t: S,
        _y: &[T],
        _jacobian: &mut MatrixViewMut<'_, T>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }
}

impl<S, T, F> CallJacobian<S, T> for F
where
    S: RealScalar,
    T: NumericScalar<Real = S>,
    F: Fn(S, &[T], &mut MatrixViewMut<'_, T>) -> Result<(), SciOxideError>,
{
    fn call(
        &self,
        t: S,
        y: &[T],
        jacobian: &mut MatrixViewMut<'_, T>,
    ) -> Result<Option<()>, SciOxideError> {
        self(t, y, jacobian).map(|()| Some(()))
    }
}

/// An [`OdeSystem`] `y′ = f(t, y)` defined by a closure for `f`, with optional
/// closures for the analytic and the automatic Jacobian.
///
/// The right-hand side closure has the form
/// `Fn(S, &[T], &mut [T]) -> Result<(), SciOxideError>` and writes `f(t, y)` into its
/// last argument. The closure types are part of the `OdeFn` type, so calls are
/// statically dispatched. The mass matrix is the identity.
///
/// Calling [`OdeSystem::rhs`] directly checks that `t` is finite and that both slices
/// have the system dimension, and returns `NonFinite` or `DimensionMismatch`
/// otherwise. The Jacobian methods also check that `y` is finite and that the
/// matrix is `n × n`, and reject a non-finite result. A closure that fails may leave
/// its output partly written; the integrators never commit such output. See
/// [Callbacks](crate#callbacks).
pub struct OdeFn<
    S: RealScalar + NumericScalar<Real = S>,
    R,
    J = NoJacobian,
    AJ = NoJacobian,
    T: NumericScalar<Real = S> = S,
> {
    dimension: usize,
    rhs: R,
    jacobian: J,
    automatic_jacobian: AJ,
    scalar: PhantomData<(S, T)>,
}

impl<S, R, T> OdeFn<S, R, NoJacobian, NoJacobian, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    R: Fn(S, &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Creates a system of state dimension `dimension` with right-hand side `rhs`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` (`Empty`) if `dimension` is zero.
    pub fn new(dimension: usize, rhs: R) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::Empty,
                "ODE dimension must be positive",
            ));
        }
        Ok(Self {
            dimension,
            rhs,
            jacobian: NoJacobian,
            automatic_jacobian: NoJacobian,
            scalar: PhantomData,
        })
    }
}

impl<S, R, J, AJ, T> OdeFn<S, R, J, AJ, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    /// Returns the system with `jacobian` as its analytic Jacobian
    /// ([`OdeSystem::jacobian`]).
    #[must_use]
    pub fn with_jacobian<J2>(self, jacobian: J2) -> OdeFn<S, R, J2, AJ, T> {
        OdeFn {
            dimension: self.dimension,
            rhs: self.rhs,
            jacobian,
            automatic_jacobian: self.automatic_jacobian,
            scalar: self.scalar,
        }
    }

    /// Returns the system with `automatic_jacobian` as its automatic Jacobian
    /// ([`OdeSystem::automatic_jacobian`]).
    #[must_use]
    pub fn with_automatic_jacobian<AJ2>(self, automatic_jacobian: AJ2) -> OdeFn<S, R, J, AJ2, T> {
        OdeFn {
            dimension: self.dimension,
            rhs: self.rhs,
            jacobian: self.jacobian,
            automatic_jacobian,
            scalar: self.scalar,
        }
    }
}

impl<S, R, J, AJ, T> OdeSystem<S, T> for OdeFn<S, R, J, AJ, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    R: Fn(S, &[T], &mut [T]) -> Result<(), SciOxideError>,
    J: JacobianCallback<S, T>,
    AJ: JacobianCallback<S, T>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn rhs(
        &self,
        t: S,
        y: &[T],
        dydt: &mut [T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        validate_time(t)?;
        if y.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(y.len()),
            ));
        }
        if dydt.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(dydt.len()),
            ));
        }
        (self.rhs)(t, y, dydt)
    }

    fn jacobian(
        &self,
        t: S,
        y: &[T],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        validate_time(t)?;
        crate::safety::validate_vector(y, self.dimension, scioxide_core::NonFiniteOrigin::Input)?;
        crate::safety::validate_matrix_shape(self.dimension, jacobian.as_view())?;
        let available = self.jacobian.call(t, y, jacobian)?;
        if available.is_some() {
            crate::safety::validate_matrix(self.dimension, jacobian.as_view())?;
        }
        Ok(available)
    }

    fn automatic_jacobian(
        &self,
        t: S,
        y: &[T],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        validate_time(t)?;
        crate::safety::validate_vector(y, self.dimension, scioxide_core::NonFiniteOrigin::Input)?;
        crate::safety::validate_matrix_shape(self.dimension, jacobian.as_view())?;
        let available = self.automatic_jacobian.call(t, y, jacobian)?;
        if available.is_some() {
            crate::safety::validate_matrix(self.dimension, jacobian.as_view())?;
        }
        Ok(available)
    }
}

fn validate_time<S: RealScalar>(t: S) -> Result<(), SciOxideError> {
    crate::safety::validate_finite_time(t, scioxide_core::NonFiniteOrigin::Input)
}

/// Adapts `g(t, y)` to an event function without a total derivative.
/// Evaluation forwards to the closure without allocation and preserves its error.
pub struct EventFn<F> {
    function: F,
    direction: crate::EventDirection,
    terminal: bool,
}
impl<F> EventFn<F> {
    /// Detects crossings in either direction without stopping integration.
    pub fn new(function: F) -> Self {
        Self {
            function,
            direction: crate::EventDirection::Either,
            terminal: false,
        }
    }
    /// Sets the crossing direction, measured in integration order.
    #[must_use]
    pub fn with_direction(mut self, direction: crate::EventDirection) -> Self {
        self.direction = direction;
        self
    }
    /// Sets whether a crossing stops integration.
    #[must_use]
    pub fn with_terminal(mut self, terminal: bool) -> Self {
        self.terminal = terminal;
        self
    }
}
impl<S: RealScalar, T: NumericScalar<Real = S>, F: Fn(S, &[T]) -> Result<S, SciOxideError>>
    crate::EventFunction<S, T> for EventFn<F>
{
    fn eval(&self, t: S, y: &[T]) -> Result<S, SciOxideError> {
        (self.function)(t, y)
    }
    fn direction(&self) -> crate::EventDirection {
        self.direction
    }
    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

/// Adapts closures for the residual `F(t, y, y′)`.
/// Calls forward into solver-owned slices without allocation; callback errors are
/// preserved. Dimensions are checked by the consuming solver in preflight.
pub struct DaeFn<T, F> {
    dimension: usize,
    residual: F,
    marker: PhantomData<fn() -> T>,
}
impl<T: NumericScalar, F> DaeFn<T, F>
where
    F: Fn(T::Real, &[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Stores the dimension and callbacks without allocating or evaluating them.
    pub fn new(dimension: usize, residual: F) -> Self {
        Self {
            dimension,
            residual,
            marker: PhantomData,
        }
    }
}
impl<T: NumericScalar, F> crate::DaeSystem<T> for DaeFn<T, F>
where
    F: Fn(T::Real, &[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn residual(
        &self,
        time: T::Real,
        state: &[T],
        derivative: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.residual)(time, state, derivative, output)
    }
}

/// Adapts closures for the drift `a(t, x)` and diffusion product `B(t, x) dw`.
/// Calls forward into solver-owned slices without allocation; callback errors are
/// preserved. Dimensions are checked by the consuming solver in preflight.
pub struct SdeFn<T, A, B> {
    dimension: usize,
    noise_dimension: usize,
    drift: A,
    diffusion_product: B,
    marker: PhantomData<fn() -> T>,
}
impl<T: NumericScalar, A, B> SdeFn<T, A, B>
where
    A: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    B: Fn(T::Real, &[T], &[T::Real], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Stores the dimension and callbacks without allocating or evaluating them.
    pub fn new(dimension: usize, noise_dimension: usize, drift: A, diffusion_product: B) -> Self {
        Self {
            dimension,
            noise_dimension,
            drift,
            diffusion_product,
            marker: PhantomData,
        }
    }
}
impl<T: NumericScalar, A, B> crate::SdeSystem<T> for SdeFn<T, A, B>
where
    A: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    B: Fn(T::Real, &[T], &[T::Real], &mut [T]) -> Result<(), SciOxideError>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn noise_dimension(&self) -> usize {
        self.noise_dimension
    }
    fn drift(
        &self,
        time: T::Real,
        state: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.drift)(time, state, output)
    }
    fn diffusion_product(
        &self,
        time: T::Real,
        state: &[T],
        increment: &[T::Real],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.diffusion_product)(time, state, increment, output)
    }
}

/// Adapts closures for a boundary-value equation and boundary residual with no unknown parameters.
/// Calls forward into solver-owned slices without allocation; callback errors are
/// preserved. Dimensions are checked by the consuming solver in preflight.
pub struct BvpFn<T, F, G> {
    dimension: usize,
    rhs: F,
    boundary: G,
    marker: PhantomData<fn() -> T>,
}
impl<T: NumericScalar, F, G> BvpFn<T, F, G>
where
    F: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Stores the dimension and callbacks without allocating or evaluating them.
    pub fn new(dimension: usize, rhs: F, boundary: G) -> Self {
        Self {
            dimension,
            rhs,
            boundary,
            marker: PhantomData,
        }
    }
}
impl<T: NumericScalar, F, G> crate::BvpSystem<T> for BvpFn<T, F, G>
where
    F: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn rhs(
        &self,
        x: T::Real,
        y: &[T],
        _p: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.rhs)(x, y, output)
    }
    fn boundary(
        &self,
        left: &[T],
        right: &[T],
        _p: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.boundary)(left, right, output)
    }
}

/// Adapts closures for the velocity `∇T(p)` and potential gradient `∇V(q)`.
/// Calls forward into solver-owned slices without allocation; callback errors are
/// preserved. Dimensions are checked by the consuming solver in preflight.
pub struct HamiltonianFn<T, V, G> {
    dimension: usize,
    velocity: V,
    gradient: G,
    marker: PhantomData<fn() -> T>,
}
impl<T: NumericScalar, V, G> HamiltonianFn<T, V, G>
where
    V: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    /// Stores the dimension and callbacks without allocating or evaluating them.
    pub fn new(dimension: usize, velocity: V, gradient: G) -> Self {
        Self {
            dimension,
            velocity,
            gradient,
            marker: PhantomData,
        }
    }
}
impl<T: NumericScalar, V, G> crate::SeparableHamiltonian<T> for HamiltonianFn<T, V, G>
where
    V: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn velocity(
        &self,
        momentum: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.velocity)(momentum, output)
    }
    fn gradient(
        &self,
        position: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        (self.gradient)(position, output)
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::BackwardEuler;
    use crate::stepper::testing::with_context;
    use crate::stepper::{StepRequest, StepTrace, Stepper};
    use scioxide_linalg::Vector;

    #[test]
    fn borrowed_callback_failure_does_not_commit_a_step() {
        let system = OdeFn::new(1, |_: f64, _: &[f64], output: &mut [f64]| {
            output[0] = 99.0;
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "partial write",
            ))
        })
        .unwrap();
        let solver = BackwardEuler::new();
        let mut workspace = Default::default();
        Stepper::<f64, f64>::reserve(&solver, 1, false, &mut workspace).unwrap();
        Stepper::<f64, f64>::begin(&solver, 1, false, &mut workspace).unwrap();
        let mut state = Vector::from_vec(vec![7.0]);
        let trace = StepTrace::default();
        with_context(|cx| {
            assert!(
                solver
                    .step(
                        &system,
                        0.0,
                        &mut state,
                        0.1,
                        StepRequest::new(&trace),
                        &mut workspace,
                        cx,
                    )
                    .is_err()
            );
            Ok(())
        })
        .unwrap();
        assert_eq!(state.as_slice(), &[7.0]);
    }

    #[test]
    fn callbacks_borrow_stack_storage() {
        let coefficient = -2.0;
        let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
            output[0] = coefficient * state[0];
            Ok(())
        })
        .unwrap()
        .with_jacobian(|_: f64, _: &[f64], output: &mut MatrixViewMut<'_, f64>| {
            output.set(0, 0, coefficient)
        });
        let mut output = [0.0];
        let mut matrix = [0.0];
        with_context(|cx| {
            system.rhs(0.0, &[3.0], &mut output, cx)?;
            system.jacobian(
                0.0,
                &[3.0],
                &mut MatrixViewMut::from_slice(1, 1, &mut matrix)?,
                cx,
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(output, [-6.0]);
        assert_eq!(matrix, [-2.0]);
    }
}
