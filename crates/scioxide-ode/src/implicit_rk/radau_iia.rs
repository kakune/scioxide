//! Adaptive three-stage Radau IIA (order 5, L-stable), with cubic collocation output.

use core::fmt;

use scioxide_core::{
    Budget, ErrorTolerance, NumericScalar, RealScalar, Scaled, SciOxideError, SolveContext,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Vector;

use super::{CoupledMethod, ImplicitConfig, collocation_step};
use crate::dense_output::{DenseExtension, DenseOutputStorage};
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::stiff::kernel::StageLayout;
use crate::stiff::{accuracy::StageAccuracy, kernel::StageKernel};
use crate::traits::OdeSystem;
use crate::traits::evaluate_rhs_staged;
use crate::workspace::ImplicitOdeWorkspace;

/// Only Radau prepares estimator and collocation history storage.
#[derive(Debug)]
pub(crate) struct RadauWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
> {
    error: Vector<T>,
    defect: Vector<T>,
    forcing: Vector<T>,
    perturbed: Vector<T>,
    dense: DenseOutputStorage<S, T>,
    accepted: DenseOutputStorage<S, T>,
    first: bool,
    rejected: bool,
    pending: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> RadauWorkspace<S, T> {
    pub(crate) fn new(n: usize) -> Result<Self, SciOxideError> {
        let mut result = Self {
            error: Vector::empty(),
            defect: Vector::empty(),
            forcing: Vector::empty(),
            perturbed: Vector::empty(),
            dense: DenseOutputStorage::new(),
            accepted: DenseOutputStorage::new(),
            first: true,
            rejected: false,
            pending: false,
        };
        for vector in [
            &mut result.error,
            &mut result.defect,
            &mut result.forcing,
            &mut result.perturbed,
        ] {
            vector.try_resize(n, T::ZERO)?;
        }
        result.dense.try_resize(n, 3)?;
        result.accepted.try_resize(n, 3)?;
        Ok(result)
    }

    pub(crate) fn reset(&mut self) {
        self.first = true;
        self.rejected = false;
        self.pending = false;
    }

    pub(crate) fn accept(&mut self) -> Result<(), SciOxideError> {
        if self.pending {
            self.accepted.copy_from(&self.dense)?;
            self.first = false;
            self.rejected = false;
            self.pending = false;
        }
        Ok(())
    }

    pub(crate) fn discard(&mut self) {
        self.pending = false;
        self.rejected = true;
    }

    pub(crate) fn predict(
        &mut self,
        t: S,
        y: &[T],
        dt: S,
        nodes: &[S],
        stages: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.pending = false;
        let (left, right) = self.accepted.interval();
        let continues = !self.first && self.accepted.continues(t, y) && {
            let ratio = dt / (right - left);
            ratio > S::ZERO && ratio <= S::try_from_f64_lossy(8.0)?
        };
        for (&node, stage) in nodes.iter().zip(stages.chunks_mut(y.len().max(1))) {
            if continues {
                self.accepted
                    .extrapolate_into(crate::safety::checked_stage_time(t, dt, node)?, stage)?;
            } else {
                stage.copy_from_slice(y);
            }
        }
        Ok(())
    }

    /// The embedded error estimate in the kernel's M − hγJ normalization, with state-unit defects.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        kernel: &mut StageKernel<S, T>,
        system: &P,
        t: S,
        dt: S,
        old: &Vector<T>,
        trial: &Vector<T>,
        stages: &[T],
        first_rhs: &Vector<T>,
        nodes: &[S],
        request: StepRequest<'_, S, T>,
        rhs: &mut Vector<T>,
        candidate: &mut Vector<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = old.len();
        let three = S::try_from_f64_lossy(3.0)?;
        let seven = S::try_from_f64_lossy(7.0)?;
        let thirteen = S::try_from_f64_lossy(13.0)?;
        let root = S::try_from_f64_lossy(6.0)?.sqrt();
        let e = [
            (-thirteen - seven * root) / three,
            (-thirteen + seven * root) / three,
            -S::ONE / three,
        ];
        let gamma = kernel.radau_shift()?;
        // Keep M·ZE separate: the second filter changes only f(t,y+e).
        estimator_forcing(
            &e,
            gamma,
            old.as_slice(),
            stages,
            kernel.mass(),
            self.defect.as_mut_slice(),
            self.forcing.as_mut_slice(),
        )?;
        self.filter_rhs(gamma, dt, first_rhs)?;
        let accuracy = StageAccuracy::from_request(old.as_slice(), request);
        kernel.radau_filter(
            system,
            request,
            accuracy,
            trial.as_slice(),
            self.defect.as_slice(),
            self.error.as_mut_slice(),
            cx,
        )?;
        let default = ErrorTolerance::default();
        let tolerance = request.error_tolerance.unwrap_or(&default);
        let norm = tolerance.normalized_norm_by(n, |i| {
            (
                self.error.as_slice()[i].abs(),
                old.as_slice()[i].abs().max(trial.as_slice()[i].abs()),
            )
        })?;
        if (self.first || self.rejected) && norm > S::ONE {
            for ((out, &old), &error) in self
                .perturbed
                .as_mut_slice()
                .iter_mut()
                .zip(old.as_slice())
                .zip(self.error.as_slice())
            {
                *out = crate::safety::stage_affine(old, S::ONE, [(S::ONE, error)])?;
            }
            evaluate_rhs_staged(system, t, &self.perturbed, rhs, candidate, request, cx)?;

            self.filter_rhs(gamma, dt, rhs)?;
            kernel.radau_filter(
                system,
                request,
                accuracy,
                trial.as_slice(),
                self.defect.as_slice(),
                self.error.as_mut_slice(),
                cx,
            )?;
        }
        self.prepare_dense(t, dt, old, trial, stages, nodes)?;
        self.pending = true;
        Ok(())
    }

    fn filter_rhs(&mut self, gamma: S, dt: S, rhs: &Vector<T>) -> Result<(), SciOxideError> {
        for ((out, &forcing), &rhs) in self
            .defect
            .as_mut_slice()
            .iter_mut()
            .zip(self.forcing.as_slice())
            .zip(rhs.as_slice())
        {
            *out = crate::safety::stage_affine(forcing, dt, [(gamma, rhs)])?;
        }
        Ok(())
    }

    fn prepare_dense(
        &mut self,
        t: S,
        dt: S,
        old: &Vector<T>,
        trial: &Vector<T>,
        stages: &[T],
        nodes: &[S],
    ) -> Result<(), SciOxideError> {
        use crate::dense_output::{component, native_product};
        let n = old.len();
        // Collocation geometry is shared by every real and imaginary component.
        let products: [S; 3] = core::array::from_fn(|j| nodes[(j + 1) % 3] * nodes[(j + 2) % 3]);
        let denominators: [S; 3] = core::array::from_fn(|j| {
            let c = nodes[j];
            c * (c - nodes[(j + 1) % 3]) * (c - nodes[(j + 2) % 3])
        });
        let linear_weights: [S; 3] = core::array::from_fn(|j| products[j] / denominators[j]);
        let cubic_weights: [S; 3] = core::array::from_fn(|j| S::ONE / denominators[j]);
        self.dense.prepare_coefficients(
            DenseExtension::RadauIia,
            dt,
            old,
            |i, imaginary| {
                let initial = component(old.as_slice()[i], imaginary);
                let mut linear = S::ZERO;
                let mut cubic = S::ZERO;
                for j in 0..3 {
                    let delta = component(stages[j * n + i], imaginary) - initial;
                    native_product(linear_weights[j], delta)?;
                    native_product(cubic_weights[j], delta)?;
                    linear = scioxide_core::fast_mul_add(linear_weights[j], delta, linear);
                    cubic = scioxide_core::fast_mul_add(cubic_weights[j], delta, cubic);
                }
                let delta = component(trial.as_slice()[i], imaginary) - initial;
                let mut values = [S::ZERO; 7];
                values[..3].copy_from_slice(&[delta, linear - delta, -cubic]);
                Some(values)
            },
            |i, imaginary| {
                let initial = Scaled::from_value(component(old.as_slice()[i], imaginary))?;
                let mut linear = Scaled::ZERO;
                let mut cubic = Scaled::ZERO;
                for j in 0..3 {
                    let delta = Scaled::from_value(component(stages[j * n + i], imaginary))?
                        .sub(initial)?;
                    let weight = delta.div(Scaled::from_value(denominators[j])?)?;
                    linear = linear.add(weight.mul(Scaled::from_value(products[j])?)?)?;
                    cubic = cubic.add(weight)?;
                }
                let delta =
                    Scaled::from_value(component(trial.as_slice()[i], imaginary))?.sub(initial)?;
                let mut values = [Scaled::ZERO; 7];
                values[..3].copy_from_slice(&[delta, linear.sub(delta)?, -cubic]);
                Ok(values)
            },
        )?;
        self.dense
            .finish(DenseExtension::RadauIia, t, dt, old, trial)?;
        Ok(())
    }

    pub(crate) fn result<'a>(
        &'a self,
        request: StepRequest<'_, S, T>,
        result: &mut StepResult<'a, S, T>,
    ) {
        result.raw_error = Some(self.error.as_slice());
        result.error_order = Some(4);
        if request.requires_dense_output() {
            result.dense_output = Some(&self.dense);
        }
    }
}

/// The estimator's stage defect `zᵢ = Σⱼ eⱼ(Yⱼᵢ − yᵢ)` and forcing `γ·(M z)ᵢ`, identity `M`
/// when `mass` is `None`. Rows are fused native sums. A row whose difference, product
/// or sum is non-finite, or whose nonzero product leaves the normal range, replays its
/// original operands in extended-range arithmetic before one rounding.
fn estimator_forcing<S, T>(
    e: &[S; 3],
    gamma: S,
    old: &[T],
    stages: &[T],
    mass: Option<&scioxide_linalg::Matrix<T>>,
    defect: &mut [T],
    forcing: &mut [T],
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    use crate::dense_output::{component, native_product};
    let n = old.len();
    for (i, out) in defect.iter_mut().enumerate() {
        let native = native_parts::<S, T>(|imaginary| {
            let initial = component(old[i], imaginary);
            let mut sum = S::ZERO;
            for (j, &weight) in e.iter().enumerate() {
                let delta = component(stages[j * n + i], imaginary) - initial;
                native_product(weight, delta)?;
                sum = scioxide_core::fast_mul_add(weight, delta, sum);
            }
            Some(sum)
        });
        *out = match native {
            Some(value) => value,
            None => defect_replay(e, old[i], (0..3).map(|j| stages[j * n + i]))?,
        };
    }
    for (row, out) in forcing.iter_mut().enumerate() {
        let native = match mass {
            Some(mass) => {
                let coefficients = &mass.as_slice()[row * n..(row + 1) * n];
                native_parts::<S, T>(|imaginary| {
                    let mut sum = S::ZERO;
                    for (&m, &z) in coefficients.iter().zip(defect.iter()) {
                        // Dense mass coefficients can be complex.
                        let (mr, zr) = (m.real_component(), component(z, imaginary));
                        native_product(mr, zr)?;
                        sum = scioxide_core::fast_mul_add(mr, zr, sum);
                        if !T::IS_REAL {
                            let (mi, zi) = (m.imaginary_component(), component(z, !imaginary));
                            let mi = if imaginary { mi } else { -mi };
                            native_product(mi, zi)?;
                            sum = scioxide_core::fast_mul_add(mi, zi, sum);
                        }
                    }
                    native_product(gamma, sum)
                })
            }
            None => native_parts::<S, T>(|imaginary| {
                native_product(gamma, component(defect[row], imaginary))
            }),
        };
        *out = match native {
            Some(value) => value,
            None => forcing_replay(
                gamma,
                mass.map(|mass| &mass.as_slice()[row * n..(row + 1) * n]),
                defect,
                defect[row],
            )?,
        };
    }
    Ok(())
}

/// Assemble a native value from its real and imaginary planes; `None` if either plane
/// is exceptional or non-finite.
#[inline(always)]
fn native_parts<S, T>(native: impl Fn(bool) -> Option<S>) -> Option<T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let real = native(false)?;
    let imaginary = if T::IS_REAL { S::ZERO } else { native(true)? };
    (real.is_finite() && imaginary.is_finite()).then(|| T::from_components(real, imaginary))
}

#[cold]
#[inline(never)]
fn defect_replay<S, T>(
    e: &[S; 3],
    old: T,
    stages: impl Iterator<Item = T>,
) -> Result<T, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    use crate::safety::StageValue;
    let mut sum = StageValue::<T>::from_value(T::ZERO)?;
    for (&weight, stage) in e.iter().zip(stages) {
        let increment = StageValue::<T>::from_value(stage)?.sub(StageValue::from_value(old)?)?;
        sum = sum.add(increment.mul(StageValue::from_value(weight)?)?)?;
    }
    sum.round()
}

#[cold]
#[inline(never)]
fn forcing_replay<S, T>(
    gamma: S,
    mass: Option<&[T]>,
    defect: &[T],
    diagonal: T,
) -> Result<T, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    use crate::safety::StageValue;
    let mass_defect = match mass {
        Some(row) => row
            .iter()
            .zip(defect)
            .try_fold(StageValue::<T>::from_value(T::ZERO)?, |sum, (&a, &z)| {
                sum.add(StageValue::from_value(a)?.mul(StageValue::from_value(z)?)?)
            })?,
        None => StageValue::from_value(diagonal)?,
    };
    mass_defect.mul(StageValue::from_value(gamma)?)?.round()
}

type RadauCoefficients<S> = ([S; 9], [S; 3], [S; 3]);

/// Integrates stiff problems with the three-stage Radau IIA method, of order 5.
///
/// Radau IIA is the collocation method at the Radau nodes
/// `c = (4 − √6)/10, (4 + √6)/10, 1`. It is L-stable and stiffly accurate (the
/// last stage is the step result), with stage order 3. It suits stiff problems
/// at moderate to tight tolerances, and it accepts a constant mass matrix `M` on
/// the dense backend.
///
/// The stage equations are solved by simplified Newton iteration with one
/// Jacobian `J` for all three stages. A change of variables diagonalizes the
/// stage coupling, so each iteration solves one real and one complex `n×n`
/// system with the shifted matrices `M − hγJ`, where `γ` runs over the
/// eigenvalues of the Runge–Kutta coefficient matrix. Complex states use three
/// complex `n×n` systems. Each Newton iteration evaluates `f` at the three stages,
/// and a step takes at most seven iterations.
///
/// The local error estimate has order 3. It is the difference to an embedded
/// formula, filtered through the real shifted matrix so that it stays bounded for
/// stiff components. On the first step and after a rejection, an estimate above
/// the tolerance is filtered once more at the cost of one extra evaluation of
/// `f`. By default the step size follows Gustafsson's predictive controller,
/// which also takes the number of Newton iterations into account; a
/// [`OdeStepControl`](crate::OdeStepControl) other than `MethodDefault` replaces
/// it. Dense output is the cubic collocation polynomial of the step, of
/// continuous order 3, and supports events.
///
/// A failed Newton iteration or a singular shifted matrix rejects the step,
/// which is retried at half the size; see [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ErrorTolerance, NormSelection};
/// use scioxide_linalg::Vector;
/// use scioxide_ode::{OdeFn, OdeOptions, OdeMethod, OdeProblem, OdeState, OdeTermination, RadauIia};
/// // y' = −10 y, y(0) = 1, so y(1) = e⁻¹⁰.
/// let system = OdeFn::new(1, |_: f64, y: &[f64], f: &mut [f64]| {
///     f[0] = -10.0 * y[0];
///     Ok(())
/// })?;
/// let tolerance = ErrorTolerance::scalar(1e-10, 1e-7, NormSelection::RootMeanSquare)?;
/// let options = OdeOptions::new().with_error_tolerance(tolerance);
/// let plan = (RadauIia::<f64>::new(), options);
/// let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
/// let solution = plan.0.solve(&OdeProblem::new(&system, 1.0), &plan.1, initial)?;
/// assert_eq!(solution.report().termination(), Some(OdeTermination::ReachedEnd));
/// assert!((solution.state().values()[0] - (-10.0_f64).exp()).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*,
///   2nd ed., Springer, 1996, §IV.5 (construction) and §IV.8 (implementation,
///   error estimate and step-size control).
/// - K. Gustafsson, "Control-theoretic techniques for stepsize selection in
///   implicit Runge–Kutta methods", *ACM Trans. Math. Softw.* 20(4), 496–517,
///   1994.
#[derive(Clone)]
pub struct RadauIia<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    config: ImplicitConfig<S, T>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> fmt::Debug
    for RadauIia<S, T>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.config.debug("RadauIia", formatter)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> RadauIia<S, T> {
    /// Creates the method with dense LU stages and the default Jacobian sources.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: ImplicitConfig::new(),
        }
    }

    /// Selects the linear solver of the stage systems: dense LU (the default),
    /// sparse LU or restarted GMRES.
    ///
    /// See [`OdeStageBackend`](crate::OdeStageBackend).
    ///
    /// Sparse and Krylov stages need the identity mass matrix. With any other mass
    /// matrix the solve fails with `InvalidInput` when the first step loads it.
    /// Sparse limits apply to each `n×n` factor separately. GMRES uses
    /// [`OdeSystem::stage_preconditioner`] at the real part of each shift, applied
    /// separately to the real and imaginary parts;
    /// [`OdeSystem::coupled_preconditioner`] is not used.
    pub fn with_backend(mut self, backend: crate::OdeStageBackend) -> Self {
        self.config.backend(backend);
        self
    }

    /// Sets the order in which Jacobian sources are tried.
    ///
    /// The default tries the analytic Jacobian of the system, then automatic
    /// differentiation, then forward differences. See
    /// [Implicit stages](crate#implicit-stages).
    #[must_use]
    pub fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.config.strategy(strategy);
        self
    }

    fn coefficients() -> Result<RadauCoefficients<S>, SciOxideError> {
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let six = three + three;
        let ten = two * (three + two);
        let sqrt_six = six.sqrt();
        let sqrt_six_f64 = <f64 as scioxide_core::RealScalar>::sqrt(6.0);
        let coefficients = [
            S::try_from_f64_lossy((88.0 - 7.0 * sqrt_six_f64) / 360.0)?,
            S::try_from_f64_lossy((296.0 - 169.0 * sqrt_six_f64) / 1800.0)?,
            S::try_from_f64_lossy((-2.0 + 3.0 * sqrt_six_f64) / 225.0)?,
            S::try_from_f64_lossy((296.0 + 169.0 * sqrt_six_f64) / 1800.0)?,
            S::try_from_f64_lossy((88.0 + 7.0 * sqrt_six_f64) / 360.0)?,
            S::try_from_f64_lossy((-2.0 - 3.0 * sqrt_six_f64) / 225.0)?,
            S::try_from_f64_lossy((16.0 - sqrt_six_f64) / 36.0)?,
            S::try_from_f64_lossy((16.0 + sqrt_six_f64) / 36.0)?,
            S::try_from_f64_lossy(1.0 / 9.0)?,
        ];
        let weights = [coefficients[6], coefficients[7], coefficients[8]];
        let nodes = [(four - sqrt_six) / ten, (four + sqrt_six) / ten, one];
        Ok((coefficients, weights, nodes))
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for RadauIia<S, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for RadauIia<S, T>
{
    type Workspace = ImplicitOdeWorkspace<S, T>;
    fn stage_kernel<'w>(
        &self,
        workspace: &'w Self::Workspace,
    ) -> Option<&'w crate::stiff::kernel::StageKernel<S, T>> {
        workspace.kernel.as_ref()
    }

    /// A-stable: stable on both half-axes for every step.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::unconditional()
    }

    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "RadauIia",
            algorithm: scioxide_core::algorithm_id!("ode", "RadauIia", 1),
            order: 5,
            initial_step_order: 4,
            controller: DefaultController::Radau,
            dense_output: true,
            events: true,
            adaptive: true,
            mass_matrices: true,
            injectable: true,
        }
    }

    fn fits(&self, dimension: usize, provided: bool, workspace: &Self::Workspace) -> bool {
        workspace.fits(dimension, self.config.stage.radau_shape(provided))
    }

    fn reserve(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        workspace.prepare(dimension, self.config.stage.radau_shape(provided))
    }

    fn begin(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.reserve(dimension, provided, workspace)?;
        workspace.reset_epoch()
    }
    fn accept(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        workspace.accept_step()
    }
    fn discard(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if let Some(radau) = &mut workspace.radau {
            radau.discard();
        }
        workspace
            .kernel
            .as_mut()
            .map_or(Ok(()), StageKernel::reject)
    }

    fn step<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        system: &System,
        t: S,
        y: &mut Vector<T>,
        dt: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError> {
        let (coefficients, weights, nodes) = Self::coefficients()?;
        collocation_step(
            CoupledMethod {
                stage: &self.config.stage,
                coefficients: &coefficients,
                weights: &weights,
                nodes: &nodes,
                stiffly_accurate: true,
                max_iterations: Budget::ONE.saturating_add(6),
            },
            StageLayout::Radau,
            system,
            t,
            y,
            dt,
            request,
            workspace,
            cx,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use scioxide_core::Complex;
    use scioxide_linalg::Matrix;

    fn weights() -> [f64; 3] {
        let root = 6_f64.sqrt();
        [(-13. - 7. * root) / 3., (-13. + 7. * root) / 3., -1. / 3.]
    }

    /// Reference rows computed only by the extended-exponent replay.
    fn replayed<T: NumericScalar<Real = f64>>(
        gamma: f64,
        old: &[T],
        stages: &[T],
        mass: Option<&Matrix<T>>,
    ) -> (Vec<T>, Vec<T>) {
        let (e, n) = (weights(), old.len());
        let defect: Vec<T> = (0..n)
            .map(|i| defect_replay(&e, old[i], (0..3).map(|j| stages[j * n + i])).unwrap())
            .collect();
        let forcing = (0..n)
            .map(|row| {
                let coefficients = mass.map(|mass| &mass.as_slice()[row * n..(row + 1) * n]);
                forcing_replay(gamma, coefficients, &defect, defect[row]).unwrap()
            })
            .collect();
        (defect, forcing)
    }

    fn fused<T: NumericScalar<Real = f64>>(
        gamma: f64,
        old: &[T],
        stages: &[T],
        mass: Option<&Matrix<T>>,
    ) -> Result<(Vec<T>, Vec<T>), SciOxideError> {
        let mut defect = vec![T::ZERO; old.len()];
        let mut forcing = vec![T::ZERO; old.len()];
        estimator_forcing(
            &weights(),
            gamma,
            old,
            stages,
            mass,
            &mut defect,
            &mut forcing,
        )?;
        Ok((defect, forcing))
    }

    fn agree<T: NumericScalar<Real = f64>>(values: &[T], expected: &[T], ulps: f64) {
        for (&value, &expected) in values.iter().zip(expected) {
            assert!(
                (value - expected).abs() <= ulps * f64::EPSILON * expected.abs(),
                "{value:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn ordinary_estimator_rows_agree_with_the_extended_replay() {
        let gamma = 3.637_834_252_744_496;
        let old = [1.0, -0.5, 2e-3];
        let stages = [
            1.01, -0.49, 2.1e-3, 1.02, -0.52, 2.3e-3, 1.03, -0.51, 2.2e-3,
        ];
        let mass =
            Matrix::from_vec(3, 3, vec![2., 0.5, 0., 0.5, 1., -0.25, 0., -0.25, 3.]).unwrap();
        for mass in [None, Some(&mass)] {
            let (defect, forcing) = fused(gamma, &old, &stages, mass).unwrap();
            let (expected_defect, expected_forcing) = replayed(gamma, &old, &stages, mass);
            agree(&defect, &expected_defect, 8.);
            agree(&forcing, &expected_forcing, 8.);
        }
        let old = [Complex::new(1.0, -2.0), Complex::new(0.25, 0.5)];
        let stages: Vec<_> = (1..=6)
            .map(|k| {
                let k = f64::from(k);
                Complex::new(1.0 + 0.01 * k, -2.0 + 0.02 * k * k)
            })
            .collect();
        let mass = Matrix::from_vec(
            2,
            2,
            vec![
                Complex::new(1., 0.5),
                Complex::new(0., 0.25),
                Complex::new(0., -0.25),
                Complex::new(2., 0.),
            ],
        )
        .unwrap();
        let (defect, forcing) = fused(gamma, &old, &stages, Some(&mass)).unwrap();
        let (expected_defect, expected_forcing) = replayed(gamma, &old, &stages, Some(&mass));
        agree(&defect, &expected_defect, 16.);
        agree(&forcing, &expected_forcing, 16.);
    }

    #[test]
    fn extreme_estimator_rows_return_the_replayed_result() {
        let gamma = 0.25;
        // Y₃ − y overflows natively, but e₃(Y₃ − y) ≈ −1.08e308 is representable.
        let big = 0.9 * f64::MAX;
        let tiny = f64::from_bits(3);
        let old = [-big, 0.0, 1.0];
        let stages = [-big, 0.0, 1.0, -big, tiny, 1.0, big, 2. * tiny, 1.5];
        assert!(!(stages[6] - old[0]).is_finite());
        let (defect, forcing) = fused(gamma, &old, &stages, None).unwrap();
        let (expected_defect, expected_forcing) = replayed(gamma, &old, &stages, None);
        assert_eq!(defect, expected_defect);
        assert_eq!(forcing, expected_forcing);
        assert!((defect[0] / (big / 3. * -2.) - 1.).abs() < 1e-15);
        // Subnormal stage differences take the replay too.
        assert!(defect[1] != 0.0 && !defect[1].is_normal());
        // γ·z₀ overflows in every arithmetic, and the replay reports it.
        assert!(fused(8.0, &old, &stages, None).is_err());
        // A mass row whose native products overflow but cancel replays to zero.
        let old = [0.0, 0.0];
        let stages = [0.0, 0.0, 0.0, 0.0, -3e10, -3e10];
        let mass = Matrix::from_vec(2, 2, vec![1e300, -1e300, 0.0, 1.0]).unwrap();
        let (defect, forcing) = fused(gamma, &old, &stages, Some(&mass)).unwrap();
        assert!(!(1e300 * defect[0]).is_finite());
        assert_eq!(forcing, replayed(gamma, &old, &stages, Some(&mass)).1);
        assert_eq!(forcing, [0.0, gamma * defect[1]]);
    }
}
