//! Real evaluation adapters and family accuracy envelopes.

use crate::result::{Bound, Evaluation, Ulps, enveloped};
use crate::{SpecialFunctionResult, SpecialFunctionTermination, ZetaOptions};
use scioxide_core::{AlgorithmId, RealScalar, SciOxideError, algorithm_component, algorithm_id};

const ZETA_ULPS: Ulps = Ulps::new(64.0, 64.0);
pub(crate) fn riemann_zeta<S: RealScalar>(s: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <ZetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    riemann_zeta_evaluate(s, options, None)?.value()
}

pub(crate) fn dirichlet_eta<S: RealScalar>(s: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <ZetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    dirichlet_eta_evaluate(s, options, None)?.value()
}

pub(crate) fn hurwitz_zeta<S: RealScalar>(s: S, a: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <ZetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    hurwitz_zeta_evaluate(s, a, options, None)?.value()
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the natural logarithm of the Hurwitz zeta function, `ln ζ(s, q)`, for real `s > 1` and
`q > 0`.

On this domain every term of `ζ(s, q) = Σₙ₌₀^∞ (n + q)^(−s)` is positive, so the logarithm is real.
It stays representable when `ζ(s, q)` itself underflows, for large `s` or large `q`. Use
[`crate::hurwitz_zeta`] for other arguments.

The sum is computed by Euler–Maclaurin summation normalized by `q^(−s)`, and the logarithm is formed
without evaluating `ζ(s, q)`.

# Examples

```
// ζ(1000, 2) = 2^(−1000) (1 + (2/3)^1000 + …) underflows, but its logarithm does not
let l = scioxide_special::hurwitz_zeta_log(1000.0_f64, 2.0)?;
assert!((l + 1000.0 * core::f64::consts::LN_2).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no separate ulp target. The detailed result reports an absolute error bound of
`ε (64 + |ln ζ(s, q)|)` in natural-log units, where `ε` is the machine epsilon of the scalar type.
See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `s = 1` or `q = 0`.
- `DomainError` (`OutsideDomain`) if `s < 1` or `q < 0`.

# References

- NIST DLMF, §25.2–§25.4 and §25.11. <https://dlmf.nist.gov/25>"###)]

    pub hurwitz_zeta_log [S: RealScalar] (s: S, q: S) -> S
    where [];
    options options: ZetaOptions;
    detailed hurwitz_zeta_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        hurwitz_zeta_log_evaluate(s, q, options, context)
    }
    scoped pub hurwitz_zeta_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Returns only completed zeta estimates; interrupted kernels keep their work counts.
// Plain projections must be able to discard unused diagnostic forms.
#[inline(always)]
fn zeta_result<S: RealScalar>(
    result: Result<(S, AlgorithmId), SciOxideError>,
    work: &crate::support::Work,
) -> Result<Evaluation<S>, SciOxideError> {
    match result {
        Ok((value, algorithm)) => enveloped(
            value,
            ZETA_ULPS,
            algorithm,
            work.used,
            SpecialFunctionTermination::Converged,
        ),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::IterationLimit,
                    ..
                }
            ) =>
        {
            Err(crate::result::exhausted_work(work.used))
        }
        Err(error) => Err(error),
    }
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn riemann_zeta_evaluate<S: RealScalar>(
    s: S,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let family = algorithm_component!("riemann_zeta");
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = crate::zeta_polylog::zeta::riemann_zeta_evaluate(s, &mut work)
        .map(|(value, method, ..)| (value, AlgorithmId::from_components(family, method, 1)));
    work.check()?;
    zeta_result(result, &work)
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn dirichlet_eta_evaluate<S: RealScalar>(
    s: S,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let family = algorithm_component!("dirichlet_eta");
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = crate::zeta_polylog::zeta::dirichlet_eta_evaluate(s, &mut work)
        .map(|(value, method, ..)| (value, AlgorithmId::from_components(family, method, 1)));
    work.check()?;
    zeta_result(result, &work)
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn hurwitz_zeta_evaluate<S: RealScalar>(
    s: S,
    a: S,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let family = algorithm_component!("hurwitz_zeta");
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = crate::zeta_polylog::zeta::hurwitz_zeta_evaluate(s, a, &mut work);
    work.check()?;
    match result {
        Ok(e) => {
            let algorithm = AlgorithmId::from_components(family, e.method, 1);
            if let Some(error) = e.error {
                Ok(Evaluation::finite(
                    e.value,
                    Bound::derived(error),
                    None,
                    algorithm,
                    work.used,
                    SpecialFunctionTermination::Converged,
                ))
            } else {
                zeta_result(Ok((e.value, algorithm)), &work)
            }
        }
        Err(error) => zeta_result(Err(error), &work),
    }
}

/// Evaluate before projecting the value or forming the detailed result.
fn hurwitz_zeta_log_evaluate<S: RealScalar>(
    s: S,
    q: S,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !s.is_finite() || !q.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "parameters must be finite",
        ));
    }
    if s <= S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("s"),
            if s == S::ONE {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "requires s>1 and q>0",
        ));
    }
    if q <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("q"),
            if q == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "requires s>1 and q>0",
        ));
    }
    let mut allowance =
        crate::support::Work::with_context(options.iteration_budget().get(), context);
    let (value, work) = crate::zeta_polylog::zeta::hurwitz_positive_log(s, q, &mut allowance)?;
    allowance.check()?;
    // The scaled series stops at one epsilon of its sum. The compensated
    // logarithmic projection adds rounding in natural-log units.
    let error = S::EPSILON * (S::try_from_f64_lossy(64.0)? + value.abs());
    Ok(Evaluation::finite(
        value,
        Bound::derived(error),
        None,
        algorithm_id!("hurwitz_zeta_log", "hurwitz_scaled_euler_maclaurin", 1),
        work,
        SpecialFunctionTermination::Converged,
    ))
}
