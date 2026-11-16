//! Consecutive-order Coulomb sequences, returned as values or written all-or-nothing to caller
//! storage.
use super::*;
use crate::util::scoped;
use alloc::vec::Vec;
use scioxide_core::algorithm_component;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{EvaluationCounts, ExecutionContext, SolveScope};

/// Selects the Coulomb wave component that [`coulomb_sequence`] computes.
///
/// Each variant corresponds to a scalar function with the same domain at every order.
#[derive(Debug, Clone, Copy)]
pub enum CoulombComponent {
    /// The regular wave `F_l(η,z)`, as in [`crate::coulomb_f`].
    F,
    /// The irregular wave `G_l(η,z)`, as in [`crate::coulomb_g`].
    G,
    /// The derivative `dF_l(η,z)/dz`, as in [`crate::coulomb_f_prime`].
    FPrime,
    /// The derivative `dG_l(η,z)/dz`, as in [`crate::coulomb_g_prime`].
    GPrime,
    /// The spherical wave `F_l(η,z)/z`, as in [`crate::coulomb_spherical_f`].
    SphericalF,
}

/// Returns the first `len` members of `staging`.
fn prefix<M>(staging: &mut [M], len: usize) -> Result<&mut [M], SciOxideError> {
    let available = staging.len();
    staging
        .get_mut(..len)
        .ok_or_else(|| SciOxideError::dimension_mismatch(len, available))
}

/// Collects the ordinary values of staged members into a vector of the same length.
fn values<M: Copy, V>(
    staging: &[M],
    value: impl Fn(M) -> Result<V, SciOxideError>,
) -> Result<Vec<V>, SciOxideError> {
    let mut values = try_with_capacity(staging.len(), scioxide_core::quantity!("count"))?;
    for &member in staging {
        values.push(value(member)?);
    }
    Ok(values)
}

/// Computes one Coulomb wave component at the consecutive orders `l, l+1, …, l+count−1`.
///
/// Element `k` of the result is the component at order `l + k`. The component, its domain and
/// its branch cut are those of the scalar function named by `component` (see
/// [`CoulombComponent`]), and every order must lie in that domain. `F`, `F′` and `F/z` use the
/// downward order recurrence from the highest order, and `G`, `G′` the upward recurrence from
/// `l` (DLMF 33.4); both directions are numerically stable. The cost is one scalar evaluation
/// plus `O(count)` recurrence steps.
///
/// All members share the default work budget of [`CoulombOptions`], 65,536 units. The function
/// allocates two vectors of `count` elements; [`coulomb_sequence_into`] writes detailed results
/// to caller storage instead. The parameters are checked even when `count` is zero.
///
/// # Accuracy
///
/// Each member has the accuracy of the scalar function at its order; see
/// [`crate::coulomb_wave`].
///
/// # Errors
///
/// - The domain errors of the scalar function for `component` at any order `l + k`.
/// - `DomainError` (`OutsideDomain`) with argument `count` if an order `l + k` is not finite or
///   rounds to the previous order in the scalar type.
/// - `NumericalInstability` (`Overflow`) if a member exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) if the sequence needs more work than the default
///   budget.
/// - `AllocationFailure` if the vectors cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_special::{CoulombComponent, coulomb_sequence};
///
/// let slopes = coulomb_sequence(CoulombComponent::FPrime, 0.0, 0.0, 1.0_f64, 4)?;
/// // F₀′(0, z) = cos z
/// assert!((slopes[0] - 1.0_f64.cos()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §33.4. <https://dlmf.nist.gov/33.4>
pub fn coulomb_sequence<T: NumericScalar>(
    component: CoulombComponent,
    l: T::Real,
    eta: T::Real,
    z: T,
    count: usize,
) -> Result<Vec<T>, SciOxideError> {
    let mut staging = try_filled(
        count,
        SpecialFunctionResult::zero(),
        scioxide_core::quantity!("count"),
    )?;
    fill_sequence(
        component,
        l,
        eta,
        z,
        CoulombOptions::default(),
        &mut staging,
        scioxide_core::argument!("count"),
        ExecutionContext::sequential_native(),
    )?;
    values(&staging, SpecialFunctionResult::try_value)
}

/// Computes one Coulomb wave component at the consecutive orders `l, l+1, …, l+N−1` into
/// `output`.
///
/// Writes the detailed result at order `l + k` to `output[k]`, where `N = output.len()`. The
/// component, the recurrences and the cost are those of [`coulomb_sequence`]; a member outside
/// the range of the scalar type keeps its scaled and logarithmic forms.
///
/// All members share the work budget of `options`, and each reports the counts of the whole
/// sequence. The sequence runs in `scope`; see
/// [scoped evaluation](crate#scoped-evaluation). Members are first written to `staging`, which
/// must be at least as long as `output`, and copied to `output` only when all of them succeed:
/// [on error `output` is unchanged](scioxide_core#output-on-error). Nothing is allocated. The
/// parameters are checked even when `output` is empty. The accuracy is that of the scalar
/// functions; see [`crate::coulomb_wave`].
///
/// # Errors
///
/// - `DimensionMismatch` if `staging` is shorter than `output`.
/// - The domain errors of the scalar function for `component` at any order `l + k`.
/// - `DomainError` (`OutsideDomain`) with argument `output` if an order `l + k` is not finite
///   or rounds to the previous order in the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) with the attempted counts if the sequence needs
///   more work than the budget.
/// - `ExecutionStopped` if `scope` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_special::{
///     CoulombComponent, CoulombOptions, SpecialFunctionResult, coulomb_sequence_into,
/// };
///
/// let mut output = [SpecialFunctionResult::<f64>::zero(); 4];
/// let mut staging = output;
/// let (component, options) = (CoulombComponent::FPrime, CoulombOptions::default());
/// let scope = ExecutionContext::sequential();
/// coulomb_sequence_into(component, 0.0, 0.0, 1.0, options, &mut output, &mut staging, scope)?;
/// // F₀′(0, z) = cos z
/// assert!((output[0].try_value()? - 1.0_f64.cos()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §33.4. <https://dlmf.nist.gov/33.4>
// Keep the selected component separate from the shared positional wave parameters.
#[allow(clippy::too_many_arguments)]
pub fn coulomb_sequence_into<'scope, 'cap, T: NumericScalar>(
    component: CoulombComponent,
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    output: &mut [SpecialFunctionResult<T>],
    staging: &mut [SpecialFunctionResult<T>],
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let staging = prefix(staging, output.len())?;
    fill_sequence(
        component,
        l,
        eta,
        z,
        options,
        staging,
        scioxide_core::argument!("output"),
        scope,
    )?;
    output.copy_from_slice(staging);
    Ok(())
}

/// Fills `staging` with the component at the orders `l + k`; `length` names the argument that
/// sets the sequence length.
#[allow(clippy::too_many_arguments)]
fn fill_sequence<'scope, 'cap, T: NumericScalar>(
    component: CoulombComponent,
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    staging: &mut [SpecialFunctionResult<T>],
    length: Argument,
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let len = staging.len();
    let request = Request::Component(component);
    let (l, eta, z) = parameters::<T>(request, l, eta, Some(z), len, length)?;
    scoped(scope, options.iteration_budget(), |context| {
        context.execution().check()?;
        if len == 0 {
            return Ok(());
        }
        let mut w = Work::with_context(options.iteration_budget().get(), Some(context));
        let outcome = sequence_run(component, l, eta, z, staging, &mut w);
        let counts = completed(outcome, &w)?;
        for result in staging.iter_mut() {
            *result = result.with_counts(counts);
        }
        context.execution().check()
    })
}

/// Returns the attempted counts of a sequence run.
///
/// Work exhaustion becomes an iteration-limit convergence failure carrying the attempted counts,
/// whichever kernel observed it.
fn completed(
    outcome: Result<(), SciOxideError>,
    w: &Work<'_, '_, '_>,
) -> Result<EvaluationCounts, SciOxideError> {
    let counts = EvaluationCounts::from_iterations(w.used);
    match outcome {
        Ok(()) => Ok(counts),
        Err(_) if w.exhausted() => crate::result::require_completion(
            crate::SpecialFunctionTermination::IterationLimit,
            scioxide_core::algorithm_id!("coulomb", "order_recurrence", 1),
            counts,
        )
        .map(|()| counts),
        Err(error) => Err(error),
    }
}

fn sequence_run<T: NumericScalar>(
    kind: Component,
    l: f64,
    eta: f64,
    z: C,
    staging: &mut [SpecialFunctionResult<T>],
    w: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if z == ZERO {
        for (k, slot) in staging.iter_mut().enumerate() {
            let order = l + k as f64;
            let derivative = match kind {
                Component::F => order >= 0.0,
                Component::G => eta == 0.0,
                Component::FPrime | Component::GPrime => false,
                Component::SphericalF => order == 0.0 || order >= 1.0,
            };
            let estimate = origin(kind, order, eta, w)?;
            *slot = crate::complex_support::report(
                Ok(estimate),
                derivative,
                w,
                algorithm_component!("coulomb"),
            )?
            .publish()?;
            slot.require_completion()?;
        }
        return Ok(());
    }
    let up = matches!(kind, Component::G | Component::GPrime);
    let mut order = if up {
        l
    } else {
        l + (staging.len() - 1) as f64
    };
    let mut s = if up {
        kernel::waves(order, eta, z, w)?.1
    } else {
        kernel::regular(order, eta, z, w)?
    };
    for step in 0..staging.len() {
        if step > 0 {
            s = s.advance(order, eta, z, up, w)?;
            order += if up { 1.0 } else { -1.0 };
        }
        let k = if up { step } else { staging.len() - 1 - step };
        let mut value = s;
        if matches!(kind, Component::SphericalF) {
            value.d = (value.d - value.y / z) / z;
            value.y /= z;
            value.error *= (1.0 + 1.0 / z.norm()) / z.norm();
        }
        let derivative = matches!(kind, Component::FPrime | Component::GPrime);
        staging[k] = crate::complex_support::report(
            Ok(value.estimate(derivative, algorithm_component!("coulomb_order_recurrence"))),
            !derivative,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        staging[k].require_completion()?;
    }
    Ok(())
}

/// Computes `[F, G, F′, G′]` at the consecutive orders `l, l+1, …, l+count−1`.
///
/// Element `k` of the result is `[F, G, F′, G′]` at order `l + k`, in the order of
/// [`crate::coulomb_wave`], whose definitions, domain and branch cut apply to every order. `F`
/// is seeded at the highest order and recurs downward; `G` follows from the Wronskian at order
/// `l` and recurs upward (DLMF 33.4). The cost is about one scalar evaluation plus `O(count)`
/// recurrence steps.
///
/// All members share the default work budget of [`CoulombOptions`], 65,536 units. The function
/// allocates two vectors of `count` elements; [`coulomb_wave_sequence_into`] writes detailed
/// results to caller storage instead. The parameters are checked even when `count` is zero.
///
/// # Accuracy
///
/// Each member has the accuracy of [`crate::coulomb_wave`] at its order.
///
/// # Errors
///
/// - The domain errors of [`crate::coulomb_wave`] at any order `l + k`; in particular `z = 0`
///   accepts only one order, `l = η = 0`.
/// - `DomainError` (`OutsideDomain`) with argument `count` if an order `l + k` is not finite or
///   rounds to the previous order in the scalar type.
/// - `NumericalInstability` (`Overflow`) if a component exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) if the sequence needs more work than the default
///   budget.
/// - `AllocationFailure` if the vectors cannot be allocated.
///
/// # Examples
///
/// ```
/// // F₀(0, z) = sin z and G₀(0, z) = cos z
/// let waves = scioxide_special::coulomb_wave_sequence(0.0, 0.0, 1.0_f64, 2)?;
/// let [f, g, _, _] = waves[0];
/// assert!((f - 1.0_f64.sin()).abs() < 1e-12);
/// assert!((g - 1.0_f64.cos()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §33.4. <https://dlmf.nist.gov/33.4>
pub fn coulomb_wave_sequence<T: NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
    count: usize,
) -> Result<Vec<[T; 4]>, SciOxideError> {
    let mut staging = try_filled(
        count,
        CoulombWaveResult::zero(),
        scioxide_core::quantity!("count"),
    )?;
    fill_wave_sequence(
        l,
        eta,
        z,
        CoulombOptions::default(),
        &mut staging,
        scioxide_core::argument!("count"),
        ExecutionContext::sequential_native(),
    )?;
    values(&staging, |wave| {
        Ok([
            wave.f.try_value()?,
            wave.g.try_value()?,
            wave.f_prime.try_value()?,
            wave.g_prime.try_value()?,
        ])
    })
}

/// Computes `F`, `G`, `F′` and `G′` at the consecutive orders `l, l+1, …, l+N−1` into `output`.
///
/// Writes the detailed results at order `l + k` to `output[k]`, where `N = output.len()`. The
/// definitions, the recurrences and the cost are those of [`coulomb_wave_sequence`]; a
/// component outside the range of the scalar type keeps its scaled and logarithmic forms.
///
/// All members share the work budget of `options`, and each reports the counts of the whole
/// sequence. The sequence runs in `scope`; see
/// [scoped evaluation](crate#scoped-evaluation). Results are first written to `staging`, which
/// must be at least as long as `output`, and copied to `output` only when all of them succeed:
/// [on error `output` is unchanged](scioxide_core#output-on-error). Nothing is allocated. The
/// accuracy is that of [`crate::coulomb_wave`].
///
/// # Errors
///
/// - `DimensionMismatch` if `staging` is shorter than `output`.
/// - The domain errors of [`crate::coulomb_wave`] at any order `l + k`; in particular `z = 0`
///   accepts only one order, `l = η = 0`.
/// - `DomainError` (`OutsideDomain`) with argument `output` if an order `l + k` is not finite
///   or rounds to the previous order in the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) with the attempted counts if the sequence needs
///   more work than the budget.
/// - `ExecutionStopped` if `scope` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_special::{CoulombOptions, CoulombWaveResult, coulomb_wave_sequence_into};
///
/// let mut output = [CoulombWaveResult::<f64>::zero(); 2];
/// let mut staging = output;
/// let (options, scope) = (CoulombOptions::default(), ExecutionContext::sequential());
/// coulomb_wave_sequence_into(0.0, 0.0, 1.0, options, &mut output, &mut staging, scope)?;
/// // F₀(0, z) = sin z
/// assert!((output[0].f().try_value()? - 1.0_f64.sin()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §33.4. <https://dlmf.nist.gov/33.4>
pub fn coulomb_wave_sequence_into<'scope, 'cap, T: NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    output: &mut [CoulombWaveResult<T>],
    staging: &mut [CoulombWaveResult<T>],
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let staging = prefix(staging, output.len())?;
    fill_wave_sequence(
        l,
        eta,
        z,
        options,
        staging,
        scioxide_core::argument!("output"),
        scope,
    )?;
    output.copy_from_slice(staging);
    Ok(())
}

/// Fills `staging` with the waves at the orders `l + k`; `length` names the argument that sets
/// the sequence length.
fn fill_wave_sequence<'scope, 'cap, T: NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    staging: &mut [CoulombWaveResult<T>],
    length: Argument,
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let len = staging.len();
    let (l, eta, z) = parameters::<T>(Request::Wave, l, eta, Some(z), len, length)?;
    scoped(scope, options.iteration_budget(), |context| {
        context.execution().check()?;
        if len == 0 {
            return Ok(());
        }
        wave_sequence_run(l, eta, z, staging, options, context)?;
        context.execution().check()
    })
}

fn stage_waves<T: NumericScalar>(
    l: f64,
    eta: f64,
    z: C,
    staging: &mut [CoulombWaveResult<T>],
    w: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let top = l + (staging.len() - 1) as f64;
    if z == ZERO {
        let [f, g, f_prime, g_prime] = wave_estimates(l, eta, z, w)?;
        let publish = |estimate, derivative| {
            crate::complex_support::report(
                Ok(estimate),
                derivative,
                w,
                algorithm_component!("coulomb"),
            )?
            .publish()
        };
        staging[0] = CoulombWaveResult {
            f: publish(f, true)?,
            g: publish(g, true)?,
            f_prime: publish(f_prime, false)?,
            g_prime: publish(g_prime, false)?,
        };
        return Ok(());
    }
    let mut f = kernel::regular(top, eta, z, w)?;
    let mut order = top;
    for slot in staging.iter_mut().rev() {
        slot.f = crate::complex_support::report(
            Ok(f.estimate(false, algorithm_component!("coulomb_order_recurrence"))),
            true,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        slot.f_prime = crate::complex_support::report(
            Ok(f.estimate(true, algorithm_component!("coulomb_order_recurrence"))),
            false,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        slot.f.require_completion()?;
        slot.f_prime.require_completion()?;
        if order > l {
            f = f.advance(order, eta, z, false, w)?;
            order -= 1.0;
        }
    }
    let mut g = kernel::irregular(l, eta, z, f, w)?;
    for (k, slot) in staging.iter_mut().enumerate() {
        if k > 0 {
            g = g.advance(l + (k - 1) as f64, eta, z, true, w)?;
        }
        slot.g = crate::complex_support::report(
            Ok(g.estimate(false, algorithm_component!("coulomb_order_recurrence"))),
            true,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        slot.g_prime = crate::complex_support::report(
            Ok(g.estimate(true, algorithm_component!("coulomb_order_recurrence"))),
            false,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        slot.g.require_completion()?;
        slot.g_prime.require_completion()?;
    }
    Ok(())
}

/// Stages F, G, F' and G'.
fn wave_sequence_run<T: NumericScalar>(
    l: f64,
    eta: f64,
    z: C,
    staging: &mut [CoulombWaveResult<T>],
    options: CoulombOptions,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let mut w = Work::with_context(options.iteration_budget().get(), Some(context));
    let outcome = stage_waves(l, eta, z, staging, &mut w);
    let counts = completed(outcome, &w)?;
    for r in staging.iter_mut() {
        r.f = r.f.with_counts(counts);
        r.g = r.g.with_counts(counts);
        r.f_prime = r.f_prime.with_counts(counts);
        r.g_prime = r.g_prime.with_counts(counts);
    }
    Ok(())
}

/// Computes the Coulomb normalization constants `C_l(η), C_(l+1)(η), …, C_(l+count−1)(η)`.
///
/// Element `k` of the result is `C_(l+k)(η)`, defined in [`crate::coulomb_normalization`]. The
/// first member is evaluated directly; each next member follows from the ratio
/// `C_(l+1)/C_l = √((l+1)² + η²) / ((l+1)(2l+3))`, with the logarithms accumulated in
/// double-word arithmetic.
///
/// All members share the default work budget of [`CoulombOptions`], 65,536 units. The function
/// allocates two vectors of `count` elements; [`coulomb_normalization_sequence_into`] writes
/// detailed results to caller storage instead. The parameters are checked even when `count`
/// is zero.
///
/// # Accuracy
///
/// Each member has the accuracy of [`crate::coulomb_normalization`] at its order.
///
/// # Errors
///
/// - `DomainError` (`OutsideDomain`) if `l ≤ −1`, or, with argument `count`, if an order
///   `l + k` is not finite or rounds to the previous order in the scalar type.
/// - `NumericalInstability` (`Overflow`) if a member exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) if the sequence needs more work than the default
///   budget.
/// - `AllocationFailure` if the vectors cannot be allocated.
///
/// # Examples
///
/// ```
/// // C₀(0) = 1 and C₁(0) = 1/3
/// let constants = scioxide_special::coulomb_normalization_sequence(0.0_f64, 0.0, 3)?;
/// assert!((constants[0] - 1.0).abs() < 1e-14);
/// assert!((constants[1] - 1.0 / 3.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn coulomb_normalization_sequence<R: RealScalar + NumericScalar<Real = R>>(
    l: R,
    eta: R,
    count: usize,
) -> Result<Vec<R>, SciOxideError> {
    let mut staging = try_filled(
        count,
        SpecialFunctionResult::zero(),
        scioxide_core::quantity!("count"),
    )?;
    fill_normalization_sequence(
        l,
        eta,
        CoulombOptions::default(),
        &mut staging,
        scioxide_core::argument!("count"),
        ExecutionContext::sequential_native(),
    )?;
    values(&staging, SpecialFunctionResult::try_value)
}

/// Computes the Coulomb normalization constants `C_l(η), C_(l+1)(η), …, C_(l+N−1)(η)` into
/// `output`.
///
/// Writes the detailed result for `C_(l+k)(η)` to `output[k]`, where `N = output.len()`. The
/// recurrence is that of [`coulomb_normalization_sequence`]; a member outside the range of the
/// scalar type keeps its scaled and logarithmic forms.
///
/// All members share the work budget of `options`, and each reports the counts of the whole
/// sequence. The sequence runs in `scope`; see
/// [scoped evaluation](crate#scoped-evaluation). Results are first written to `staging`, which
/// must be at least as long as `output`, and copied to `output` only when all of them succeed:
/// [on error `output` is unchanged](scioxide_core#output-on-error). Nothing is allocated. The
/// parameters are checked even when `output` is empty.
///
/// # Errors
///
/// - `DimensionMismatch` if `staging` is shorter than `output`.
/// - `DomainError` (`OutsideDomain`) if `l ≤ −1`, or, with argument `output`, if an order
///   `l + k` is not finite or rounds to the previous order in the scalar type.
/// - `ConvergenceFailure` (`IterationLimit`) with the attempted counts if the sequence needs
///   more work than the budget.
/// - `ExecutionStopped` if `scope` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_special::{CoulombOptions, SpecialFunctionResult, coulomb_normalization_sequence_into};
///
/// let mut output = [SpecialFunctionResult::<f64>::zero(); 3];
/// let mut staging = output;
/// let (options, scope) = (CoulombOptions::default(), ExecutionContext::sequential());
/// coulomb_normalization_sequence_into(0.0, 0.0, options, &mut output, &mut staging, scope)?;
/// // C₀(0) = 1 and C₁(0) = 1/3
/// assert!((output[0].try_value()? - 1.0).abs() < 1e-14);
/// assert!((output[1].try_value()? - 1.0 / 3.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn coulomb_normalization_sequence_into<
    'scope,
    'cap,
    R: RealScalar + NumericScalar<Real = R>,
>(
    l: R,
    eta: R,
    options: CoulombOptions,
    output: &mut [SpecialFunctionResult<R>],
    staging: &mut [SpecialFunctionResult<R>],
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let staging = prefix(staging, output.len())?;
    fill_normalization_sequence(
        l,
        eta,
        options,
        staging,
        scioxide_core::argument!("output"),
        scope,
    )?;
    output.copy_from_slice(staging);
    Ok(())
}

/// Fills `staging` with `C_(l+k)(η)`; `length` names the argument that sets the sequence length.
fn fill_normalization_sequence<'scope, 'cap, R: RealScalar + NumericScalar<Real = R>>(
    l: R,
    eta: R,
    options: CoulombOptions,
    staging: &mut [SpecialFunctionResult<R>],
    length: Argument,
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<(), SciOxideError> {
    let len = staging.len();
    let (l, eta, _) = parameters::<R>(Request::Normalization, l, eta, None, len, length)?;
    scoped(scope, options.iteration_budget(), |context| {
        context.execution().check()?;
        if len == 0 {
            return Ok(());
        }
        let mut w = Work::with_context(options.iteration_budget().get(), Some(context));
        let outcome = normalization_sequence_run(l, eta, staging, &mut w);
        let counts = completed(outcome, &w)?;
        for result in staging.iter_mut() {
            *result = result.with_counts(counts);
        }
        context.execution().check()
    })
}

fn normalization_sequence_run<R: RealScalar + NumericScalar<Real = R>>(
    l: f64,
    eta: f64,
    staging: &mut [SpecialFunctionResult<R>],
    w: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    use crate::support::wide::Wide as D;
    let mut norm = kernel::normalization(l, eta, w)?;
    for (k, slot) in staging.iter_mut().enumerate() {
        if k > 0 {
            // Four bounded compensated logs, at most 24 terms each.
            for _ in 0..97 {
                w.try_step()?;
            }
            let a = D::from(l) + D::from(k as f64);
            let scale = a.hi.abs().max(eta.abs());
            let x = a.div_f64(scale);
            let y = D::from(eta).div_f64(scale);
            let log_hypot = D::from(scale).ln() + (x * x + y * y).ln().mul_f64(0.5);
            let log_a = a.ln();
            let log_b = (a.mul_f64(2.0) + D::from(1.0)).ln();
            norm.log = norm.log + log_hypot - log_a - log_b;
            norm.error += 512e-30 * (1.0 + log_hypot.hi.abs() + log_a.hi.abs() + log_b.hi.abs());
        }
        *slot = crate::complex_support::report(
            Ok(norm.estimate()),
            false,
            w,
            algorithm_component!("coulomb"),
        )?
        .publish()?;
        slot.require_completion()?;
    }
    Ok(())
}
