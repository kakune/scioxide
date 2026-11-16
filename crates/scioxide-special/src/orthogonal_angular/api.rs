//! Legendre entry points and their accuracy envelopes.

use crate::result::{Evaluation, Ulps, enveloped};
use crate::support::Work;
use crate::{LegendreOptions, SpecialFunctionResult, SpecialFunctionTermination};
use scioxide_core::{RealScalar, SciOxideError, algorithm_id};

const LEGENDRE_ULPS: Ulps = Ulps::new(64.0, 64.0);
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Legendre polynomial `P_l(x)`.

```text
P_l(x) = (1/(2^l l!)) dˡ/dxˡ (x² − 1)^l,
```

for integer degree `l ≥ 0` and any real `x`, including `|x| > 1` (DLMF §18.5). The
normalization is `P_l(1) = 1`, and `P_l(−1) = (−1)^l`.

The three-term recurrence `(k+1) P_(k+1)(x) = (2k+1) x P_k(x) − k P_(k−1)(x)` runs in `f64`
with an FMA-based correction of its rounding error. Near a zero of `P_l`, and for `|x| > 1`
with rescaling, it runs in double-word arithmetic instead. The cost is `l − 1` steps, one
work unit each; `x = ±1` costs nothing.

# Examples

```
// P₂(x) = (3x² − 1)/2 and P₃(x) = (5x³ − 3x)/2
let p2 = scioxide_special::legendre_p(2, 0.5_f64)?;
let p3 = scioxide_special::legendre_p(3, -2.0_f64)?;
assert!((p2 + 0.125).abs() < 1e-14);
assert!((p3 + 17.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range the error is within 64 ulps for `f64` and `f32`, with absolute bounds
near zeros; the measured maximum on the `f64` test set is 0.5 ulps. Tests cover degrees up to
20,000. See the [crate accuracy table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `l > 2^52` and `x ≠ ±1`, because the recurrence
  coefficients would not be exact `f64` integers.
- `NumericalInstability` (`Overflow`) if `|P_l(x)|` exceeds the range of the scalar type,
  which can happen for `|x| > 1`.
- `ConvergenceFailure` (`IterationLimit`) if the recurrence needs more steps than the work
  budget (65,536 by default).

# References

NIST DLMF, §18.5 and §18.9. <https://dlmf.nist.gov/18>"###)]

    pub legendre_p [S: RealScalar] (degree: usize, x: S) -> S
    where [];
    options options: LegendreOptions;
    detailed legendre_p_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        legendre_p_evaluate(degree, x, options, context)
    }
    scoped pub legendre_p_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Legendre function of the second kind `Q_l(x)` on `−1 ≤ x ≤ 1`.

`Q_l` is the Ferrers function of the second kind of integer degree `l ≥ 0` (DLMF §14.5):

```text
Q₀(x) = atanh x = ½ ln((1 + x)/(1 − x)),   Q₁(x) = x atanh x − 1,
```

and higher degrees follow `(k+1) Q_(k+1)(x) = (2k+1) x Q_k(x) − k Q_(k−1)(x)`. At `x = 1` the
result is `+∞` for every degree. At `x = −1` it is `−∞` for even `l` and `+∞` for odd `l`.
These limits are returned as values, not errors.

`Q₀` is evaluated as a compensated logarithm and the recurrence runs in double-word
arithmetic. The cost is `l − 1` steps, one work unit each.

# Examples

```
// Q₀(x) = atanh x and Q₁(x) = x atanh x − 1
let q0 = scioxide_special::legendre_q(0, 0.5_f64)?;
let q1 = scioxide_special::legendre_q(1, 0.5_f64)?;
assert!((q0 - 0.5_f64.atanh()).abs() < 1e-14);
assert!((q1 - (0.5 * 0.5_f64.atanh() - 1.0)).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range the error is within 64 ulps for `f64` and `f32`, with absolute bounds
near zeros. Tests cover degrees up to 20,000. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `|x| > 1`.
- `InvalidInput` (`OutOfRange`) if `l > 2^52` and `|x| < 1`, because the recurrence
  coefficients would not be exact `f64` integers.
- `ConvergenceFailure` (`IterationLimit`) if the recurrence needs more steps than the work
  budget (65,536 by default).

# References

NIST DLMF, §14.5 and §14.10. <https://dlmf.nist.gov/14>"###)]

    pub legendre_q [S: RealScalar] (degree: usize, x: S) -> S
    where [];
    options options: LegendreOptions;
    detailed legendre_q_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        legendre_q_evaluate(degree, x, options, context)
    }
    scoped pub legendre_q_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the associated Legendre function `P_l^m(x)` on `−1 ≤ x ≤ 1`.

`P_l^m` is the Ferrers function of the first kind of integer degree `l ≥ 0` and integer order
`|m| ≤ l` (DLMF 14.6.1 and §14.9):

```text
P_l^m(x)  = (−1)^m (1 − x²)^(m/2) dᵐ P_l(x)/dxᵐ,   0 ≤ m ≤ l,
P_l^(−m)(x) = (−1)^m (l − m)!/(l + m)! P_l^m(x).
```

The factor `(−1)^m` is the Condon–Shortley phase, and `P_l` is [`crate::legendre_p`]. For
`m ≠ 0` the function vanishes at `x = ±1`.

`P_m^m` is formed from the double-factorial product in compensated arithmetic with exact
binary rescaling, and the recurrence then runs upward in `l`. Away from the endpoints,
nonnegative orders cost `l` steps and negative orders cost `l + 2|m|` steps, one work unit each.
Endpoints `x = ±1` require no recurrence steps.
Values outside the range of the scalar type keep a scaled form in the detailed result.

# Examples

```
// P₁¹(x) = −√(1 − x²) and P₂²(x) = 3(1 − x²)
let p11 = scioxide_special::legendre_p_assoc(1, 1, 0.0_f64)?;
let p22 = scioxide_special::legendre_p_assoc(2, 2, 0.5_f64)?;
assert!((p11 + 1.0).abs() < 1e-14);
assert!((p22 - 2.25).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range the error is within 64 ulps for `f64` and `f32`, with absolute bounds
near zeros. See the [crate accuracy table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `|m| > l`.
- `DomainError` (`OutsideDomain`) if `|x| > 1`.
- `SizeOverflow` if `l + |m| > 2^53`.
- `NumericalInstability` (`Overflow`) if `|P_l^m(x)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the recurrence needs more steps than the work
  budget (65,536 by default).

# References

NIST DLMF, §14.6–14.10. <https://dlmf.nist.gov/14>"###)]

    pub legendre_p_assoc [S: RealScalar] (degree: usize, order: i32, x: S) -> S
    where [];
    options options: LegendreOptions;
    detailed legendre_p_assoc_with_options -> SpecialFunctionResult<S>;
    value {
        crate::orthogonal_angular::harmonics::associated_report(
            degree,
            order,
            x,
            LegendreOptions::default(),
            None,
        )?
        .value()
    }
    report {
        crate::orthogonal_angular::harmonics::associated_report(degree, order, x, options, None)?
            .publish()
    }
    scoped pub legendre_p_assoc_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            crate::orthogonal_angular::harmonics::associated_report(
                degree,
                order,
                x,
                options,
                Some(context),
            )?
            .publish()
        })
    }
}

/// Evaluates before projecting the result.
fn legendre_p_evaluate<S: RealScalar>(
    degree: usize,
    x: S,
    options: LegendreOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let value = crate::orthogonal_angular::legendre::legendre_p(degree, x, &mut work)?;
    work.check()?;
    enveloped(
        value,
        LEGENDRE_ULPS,
        algorithm_id!("legendre_p", "residual_compensated_recurrence", 1),
        work.used,
        SpecialFunctionTermination::Converged,
    )
}

/// Evaluates before projecting the result.
fn legendre_q_evaluate<S: RealScalar>(
    degree: usize,
    x: S,
    options: LegendreOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let value = crate::orthogonal_angular::legendre::legendre_q(degree, x, &mut work)?;
    work.check()?;
    enveloped(
        value,
        LEGENDRE_ULPS,
        algorithm_id!("legendre_q", "compensated_forward_recurrence", 1),
        work.used,
        SpecialFunctionTermination::Converged,
    )
}
