//! Checked symmetric elliptic integrals, principal branches and real principal values.
pub(crate) use crate::elliptic_family::elliptic_kernel::Family;
use crate::elliptic_family::elliptic_kernel::{self as kernel, Companion, companion, duplication};
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use crate::support::widen_numeric as widen;
use crate::support::{C, ZERO};
use crate::util::scoped;
use crate::{EllipticOptions, SpecialFunctionResult};
use core::ops::Neg;
#[allow(unused_imports)] // A sibling may enable core/std through feature unification.
use scioxide_core::RealScalar;
use scioxide_core::algorithm_component;
use scioxide_core::{DomainReason, NumericScalar, SciOxideError, SolveContext, SolveScope};

/// Carlson argument names, in slot order.
const ARGUMENTS: [scioxide_core::Argument; 4] = [
    scioxide_core::argument!("x"),
    scioxide_core::argument!("y"),
    scioxide_core::argument!("z"),
    scioxide_core::argument!("p"),
];
/// Validate Carlson arguments.
pub(crate) fn validate(family: Family, a: [C; 4], real_output: bool) -> Result<(), SciOxideError> {
    for z in a {
        crate::support::require_finite_param(z, "elliptic parameter z must be finite")?;
    }
    let count = if matches!(family, Family::Rc) { 2 } else { 3 };
    for (i, z) in a[..count].iter().enumerate() {
        if real_output && z.im == 0.0 && z.re < 0.0 && !(matches!(family, Family::Rc) && i == 1) {
            return Err(ERRORS.domain(ARGUMENTS[i], DomainReason::OutsideDomain));
        }
    }
    let zeros = a[..count].iter().filter(|&&z| z == ZERO).count();
    if match family {
        Family::Rg => false,
        Family::Rc => a[1] == ZERO,
        Family::Rd => a[2] == ZERO || zeros > 1,
        Family::Rj => a[3] == ZERO || zeros > 1,
        Family::Rf => zeros > 1,
    } {
        let index = match family {
            Family::Rc => 1,
            Family::Rd if a[2] == ZERO => 2,
            Family::Rj if a[3] == ZERO => 3,
            _ => a[..count].iter().position(|&z| z == ZERO).unwrap_or(0),
        };
        return Err(ERRORS.domain(ARGUMENTS[index], DomainReason::Pole));
    }
    if !real_output && matches!(family, Family::Rj) {
        let on_cut = |z: &C| z.re < 0.0 && z.im == 0.0;
        if let Some(first) = a.iter().find(|z| on_cut(z)) {
            // Name the first cut argument whose signed zero conflicts with an earlier one.
            if let Some(index) = a
                .iter()
                .position(|z| on_cut(z) && z.im.is_sign_negative() != first.im.is_sign_negative())
            {
                return Err(SciOxideError::domain_error(
                    ARGUMENTS[index],
                    DomainReason::OutsideDomain,
                    "incompatible RJ signed cut limits",
                ));
            }
        }
    }
    Ok(())
}
pub(crate) fn evaluate(
    family: Family,
    mut a: [C; 4],
    real_output: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    validate(family, a, real_output)?;
    work.check()?;
    if !real_output && matches!(family, Family::Rj) {
        return complex_rj(a, work);
    }
    if matches!(family, Family::Rg) {
        // The median nonnegative argument avoids subtraction in the RD term.
        // For complex inputs put the largest component in z to avoid division by zero.
        a[..3].sort_by(|x, y| x.norm().total_cmp(&y.norm()));
        if a[..3].iter().all(|z| z.im == 0.0) && a[1] != ZERO {
            a.swap(1, 2);
        }
    }
    let [x, y, z, p] = a.map(Z::from_value);
    let [x, y, z, p] = [x?, y?, z?, p?];
    match family {
        Family::Rf => Ok(duplication(x, y, z, Companion::None, work)?.rf),
        Family::Rd => companion(x, y, z, Companion::Rd, work),
        Family::Rj if a[3].re < 0.0 && a[3].im == 0.0 => principal_rj(a, work),
        Family::Rj => companion(x, y, z, Companion::Rj(p), work),
        Family::Rc if real_output && a[1].re < 0.0 && a[1].im == 0.0 => {
            work.try_step()?;
            if a[0] == ZERO {
                return Estimate::exact(ZERO);
            }
            let t = x.sub(y)?;
            let mut e = duplication(t, y.neg(), y.neg(), Companion::None, work)?.rf;
            e.value = e.value.mul(x.div(t)?.sqrt()?)?;
            e.relative += 8.0 * f64::EPSILON;
            Ok(e)
        }
        Family::Rc => Ok(duplication(x, y, y, Companion::None, work)?.rf),
        Family::Rg => {
            if a[2] == ZERO {
                return Estimate::exact(ZERO);
            }
            if a[0] == ZERO && a[1] == ZERO {
                return Ok(Estimate::new(
                    z.sqrt()?
                        .mul(Z::from_value(scioxide_core::Complex::new(0.5, 0.0))?)?,
                    f64::EPSILON,
                ));
            }
            let pair = duplication(x, y, z, Companion::Rd, work)?;
            let first = z.mul(pair.rf.value)?;
            let second = x
                .sub(z)?
                .mul(y.sub(z)?)?
                .mul(pair.companion.value)?
                .mul(Z::from_value(scioxide_core::Complex::new(-1.0 / 3.0, 0.0))?)?;
            let third = x.sqrt()?.mul(y.sqrt()?)?.div(z.sqrt()?)?;
            combine(
                [first, second, third],
                Z::from_value(scioxide_core::Complex::new(0.5, 0.0))?,
                pair.companion.relative + 8.0 * f64::EPSILON,
            )
        }
    }
}
/// Combine three extended-range terms;
pub(crate) fn combine(
    terms: [Z; 3],
    multiplier: Z,
    relative: f64,
) -> Result<Estimate, SciOxideError> {
    let value = terms[0].add(terms[1])?.add(terms[2])?;
    let norm = terms
        .into_iter()
        .try_fold(scioxide_core::Scaled::ZERO, |sum, t| {
            sum.add(t.magnitude()?)
        })?;
    if value.is_zero() {
        let error = norm
            .mul(multiplier.magnitude()?)?
            .mul(scioxide_core::Scaled::from_value(relative)?)?;
        let mut e = Estimate::new(value, 0.0);
        e.absolute = if error.is_zero() {
            0.0
        } else {
            error.round()?.max(f64::from_bits(1))
        };
        return Ok(e);
    }
    let condition = norm.div(value.magnitude()?)?.round()?;
    Ok(Estimate::new(value.mul(multiplier)?, relative * condition))
}
fn principal_rj(mut a: [C; 4], work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    a[..3].sort_by(|x, y| x.re.total_cmp(&y.re));
    let [x, y, z, n] = a.map(Z::from_value);
    let [x, y, z, n] = [x?, y?, z?, n?];
    let q = n.neg();
    let denominator = z.add(q)?;
    let w = z.sub(x)?.div(denominator)?;
    let p = w.mul(y)?.add(
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?
            .sub(w)?
            .mul(z)?,
    )?;
    let pair = duplication(x, y, z, Companion::Rj(p), work)?;
    let pq = p.mul(q)?;
    let s = x.mul(y)?.add(pq)?;
    let rc = duplication(s, pq, pq, Companion::None, work)?.rf;
    let first = p.sub(z)?.mul(pair.companion.value)?;
    let second = pair
        .rf
        .value
        .mul(Z::from_value(scioxide_core::Complex::new(-3.0, 0.0))?)?;
    let third = x
        .mul(y)?
        .mul(z)?
        .div(s)?
        .sqrt()?
        .mul(rc.value)?
        .mul(Z::from_value(scioxide_core::Complex::new(3.0, 0.0))?)?;
    combine(
        [first, second, third],
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?.div(denominator)?,
        pair.companion.relative + rc.relative + 16.0 * f64::EPSILON,
    )
}
pub(crate) fn run<T: NumericScalar>(
    family: Family,
    a: [T; 4],
    options: EllipticOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<kernel::EllipticEvaluation<T>, SciOxideError> {
    let a = [
        widen(a[0], "elliptic parameter value must be finite")?,
        widen(a[1], "elliptic parameter value must be finite")?,
        widen(a[2], "elliptic parameter value must be finite")?,
        widen(a[3], "elliptic parameter value must be finite")?,
    ];
    let mut work = Work::elliptic(
        options.iteration_budget().get(),
        kernel::family_target::<T::Real>(),
        context,
    );
    if crate::support::is_real::<T>()
        && crate::elliptic_family::elliptic_real::ordinary(&a.map(|x| x.re))
    {
        validate(family, a, true)?;
        work.check()?;
        let e = crate::elliptic_family::elliptic_real::evaluate(family, a.map(|x| x.re), &mut work);
        return kernel::report_real(
            e,
            &work,
            match family {
                Family::Rf => algorithm_component!("real_carlson_rf"),
                Family::Rd => algorithm_component!("real_carlson_rd"),
                Family::Rj => algorithm_component!("real_carlson_rj"),
                Family::Rc => algorithm_component!("real_carlson_rc"),
                Family::Rg => algorithm_component!("real_carlson_rg"),
            },
            matches!(family, Family::Rg) && a[..3].iter().all(|x| *x == ZERO)
                || matches!(family, Family::Rc) && a[0] == ZERO && a[1].re < 0.0,
        );
    }
    let e = evaluate(family, a, crate::support::is_real::<T>(), &mut work);
    kernel::report(
        e,
        &work,
        match family {
            Family::Rf => algorithm_component!("carlson_rf"),
            Family::Rd => algorithm_component!("carlson_rd"),
            Family::Rj => algorithm_component!("carlson_rj"),
            Family::Rc => algorithm_component!("carlson_rc"),
            Family::Rg => algorithm_component!("carlson_rg"),
        },
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Carlson's symmetric elliptic integral of the first kind `R_F(x, y, z)`.

```text
R_F(x, y, z) = (1/2) ∫₀^∞ dt / √((t + x)(t + y)(t + z))
```

`R_F` is symmetric in its arguments and homogeneous of degree `−1/2`. Real arguments must be
nonnegative, with at most one of them zero. Legendre's integrals follow from it, for example
`K(m) = R_F(0, 1 − m, 1)`.

Evaluation uses Carlson's duplication algorithm (DLMF §19.36(i)): each step divides the relative
spread of the arguments by about four, and a fifth-degree series finishes the sum once the spread is
below 0.001. Each step counts toward the iteration budget of
[`EllipticOptions`](crate::EllipticOptions). Complex arguments use the principal square root of each
factor. A negative real argument lies on the branch cut, and the sign of its zero imaginary part
selects the side.

# Examples

```
// R_F(x, x, x) = x^(−1/2)
let r = scioxide_special::carlson_rf(4.0_f64, 4.0, 4.0)?;
assert!((r - 0.5).abs() < 1e-14);
// R_F(0, 1 − m, 1) = K(m)
let r = scioxide_special::carlson_rf(0.0_f64, 0.5, 1.0)?;
assert!((r - scioxide_special::elliptic_k(0.5_f64)?).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; the measured maximum for `f64` is 4.8 ulps.
Complex arguments have no uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if an argument is negative, for a real type.
- `DomainError` (`Pole`) if two or more arguments are zero.

# References

- NIST DLMF, §19.16 and §19.36. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]

    pub carlson_rf [T: NumericScalar] (x: T, y: T, z: T) -> T
    where [];
    options options: EllipticOptions;
    detailed carlson_rf_with_options -> SpecialFunctionResult<T>;
    value {
        run(
            Family::Rf,
            [x, y, z, T::ZERO],
            EllipticOptions::default(),
            None,
        )?
        .value()
    }
    report {
        run(Family::Rf, [x, y, z, T::ZERO], options, None)?.publish()
    }
    scoped pub carlson_rf_scoped ['scope, 'cap] (scope: impl Into<SolveScope<'scope, 'cap>>) {
        scoped(scope, options.iteration_budget(), |context| {
            run(Family::Rf, [x, y, z, T::ZERO], options, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Carlson's symmetric elliptic integral of the second kind `R_D(x, y, z)`.

```text
R_D(x, y, z) = (3/2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))
```

`R_D` is symmetric in `x` and `y`, homogeneous of degree `−3/2`, and equals `R_J(x, y, z, z)`. Real
arguments must satisfy `x, y ≥ 0` with at most one of them zero, and `z > 0`.

Evaluation uses Carlson's duplication algorithm (DLMF §19.36(i)): each step divides the relative
spread of the arguments by about four, and a fifth-degree series finishes the sum once the spread is
below 0.001. Each step counts toward the iteration budget of
[`EllipticOptions`](crate::EllipticOptions). Complex arguments use the principal square root of each
factor. A negative real argument lies on the branch cut, and the sign of its zero imaginary part
selects the side.

# Examples

```
// R_D(x, x, x) = x^(−3/2)
let r = scioxide_special::carlson_rd(4.0_f64, 4.0, 4.0)?;
assert!((r - 0.125).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; the measured maximum for `f64` is 4.8 ulps.
Complex arguments have no uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if an argument is negative, for a real type.
- `DomainError` (`Pole`) if `z = 0` or `x = y = 0`.

# References

- NIST DLMF, §19.16 and §19.36. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]

    pub carlson_rd [T: NumericScalar] (x: T, y: T, z: T) -> T
    where [];
    options options: EllipticOptions;
    detailed carlson_rd_with_options -> SpecialFunctionResult<T>;
    value {
        run(
            Family::Rd,
            [x, y, z, T::ZERO],
            EllipticOptions::default(),
            None,
        )?
        .value()
    }
    report {
        run(Family::Rd, [x, y, z, T::ZERO], options, None)?.publish()
    }
    scoped pub carlson_rd_scoped ['scope, 'cap] (scope: impl Into<SolveScope<'scope, 'cap>>) {
        scoped(scope, options.iteration_budget(), |context| {
            run(Family::Rd, [x, y, z, T::ZERO], options, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Carlson's symmetric elliptic integral of the third kind `R_J(x, y, z, p)`.

```text
R_J(x, y, z, p) = (3/2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))
```

`R_J` is symmetric in `x`, `y` and `z` and homogeneous of degree `−3/2`. Real arguments must satisfy
`x, y, z ≥ 0` with at most one of them zero, and `p ≠ 0`. For `p < 0` the integrand has a pole on
the path of integration, and the result is the Cauchy principal value (DLMF §19.20). Complex
arguments on the negative real axis must all lie on the same side of the branch cut, as given by the
signs of their zero imaginary parts.

Evaluation uses Carlson's duplication algorithm (DLMF §19.36(i)): each step divides the relative
spread of the arguments by about four, and a fifth-degree series finishes the sum once the spread is
below 0.001. Each step counts toward the iteration budget of
[`EllipticOptions`](crate::EllipticOptions). Complex arguments use the principal square root of each
factor. A negative real argument lies on the branch cut, and the sign of its zero imaginary part
selects the side.

# Examples

```
// R_J(x, x, x, x) = x^(−3/2)
let r = scioxide_special::carlson_rj(4.0_f64, 4.0, 4.0, 4.0)?;
assert!((r - 0.125).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; the measured maximum for `f64` is 4.8 ulps.
Complex arguments have no uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x`, `y` or `z` is negative, for a real type.
- `DomainError` (`Pole`) if `p = 0` or two or more of `x`, `y`, `z` are zero.
- `DomainError` (`OutsideDomain`) for a complex type if arguments on the negative real axis have
  zero imaginary parts of opposite sign.

# References

- NIST DLMF, §19.16 and §19.36. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]

    pub carlson_rj [T: NumericScalar] (x: T, y: T, z: T, p: T) -> T
    where [];
    options options: EllipticOptions;
    detailed carlson_rj_with_options -> SpecialFunctionResult<T>;
    value {
        run(Family::Rj, [x, y, z, p], EllipticOptions::default(), None)?.value()
    }
    report {
        run(Family::Rj, [x, y, z, p], options, None)?.publish()
    }
    scoped pub carlson_rj_scoped ['scope, 'cap] (scope: impl Into<SolveScope<'scope, 'cap>>) {
        scoped(scope, options.iteration_budget(), |context| {
            run(Family::Rj, [x, y, z, p], options, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Carlson's degenerate elliptic integral `R_C(x, y) = R_F(x, y, y)`.

```text
R_C(x, y) = (1/2) ∫₀^∞ dt / ((t + y) √(t + x))
```

`R_C` is an elementary function: for `0 ≤ x < y`, `R_C(x, y) = arccos(√(x/y)) / √(y − x)`. Real
arguments must satisfy `x ≥ 0` and `y ≠ 0`. For `y < 0` the result is the Cauchy principal value,
and `R_C(0, y) = 0`.

Evaluation uses Carlson's duplication algorithm (DLMF §19.36(i)): each step divides the relative
spread of the arguments by about four, and a fifth-degree series finishes the sum once the spread is
below 0.001. Each step counts toward the iteration budget of
[`EllipticOptions`](crate::EllipticOptions). Complex arguments use the principal square root of each
factor. A negative real argument lies on the branch cut, and the sign of its zero imaginary part
selects the side.

# Examples

```
// R_C(0, 1) = π/2 and R_C(x, x) = x^(−1/2)
let r = scioxide_special::carlson_rc(0.0_f64, 1.0)?;
assert!((r - core::f64::consts::FRAC_PI_2).abs() < 1e-14);
assert!((scioxide_special::carlson_rc(4.0_f64, 4.0)? - 0.5).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0` for a real type.
- `DomainError` (`Pole`) if `y = 0`.

# References

- NIST DLMF, §19.16 and §19.36. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]

    pub carlson_rc [T: NumericScalar] (x: T, y: T) -> T
    where [];
    options options: EllipticOptions;
    detailed carlson_rc_with_options -> SpecialFunctionResult<T>;
    value {
        run(
            Family::Rc,
            [x, y, T::ZERO, T::ZERO],
            EllipticOptions::default(),
            None,
        )?
        .value()
    }
    report {
        run(Family::Rc, [x, y, T::ZERO, T::ZERO], options, None)?.publish()
    }
    scoped pub carlson_rc_scoped ['scope, 'cap] (scope: impl Into<SolveScope<'scope, 'cap>>) {
        scoped(scope, options.iteration_budget(), |context| {
            run(Family::Rc, [x, y, T::ZERO, T::ZERO], options, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Carlson's completely symmetric elliptic integral of the second kind `R_G(x, y, z)`.

```text
R_G(x, y, z) = (1/4) ∫₀^∞ (x/(t + x) + y/(t + y) + z/(t + z)) t dt / √((t + x)(t + y)(t + z))
```

`R_G` is symmetric in its arguments and homogeneous of degree `1/2`. Real arguments must be
nonnegative; any of them may be zero, with `R_G(0, 0, 0) = 0` and `R_G(0, 0, z) = √z / 2`.

`R_G` is computed from `R_F` and `R_D` (DLMF §19.21), with the median argument in the `R_D` term to
avoid cancellation. Evaluation uses Carlson's duplication algorithm (DLMF §19.36(i)): each step
divides the relative spread of the arguments by about four, and a fifth-degree series finishes the
sum once the spread is below 0.001. Each step counts toward the iteration budget of
[`EllipticOptions`](crate::EllipticOptions). Complex arguments use the principal square root of each
factor. A negative real argument lies on the branch cut, and the sign of its zero imaginary part
selects the side.

# Examples

```
// R_G(x, x, x) = √x
let r = scioxide_special::carlson_rg(4.0_f64, 4.0, 4.0)?;
assert!((r - 2.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if an argument is negative, for a real type.

# References

- NIST DLMF, §19.16 and §19.36. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]

    pub carlson_rg [T: NumericScalar] (x: T, y: T, z: T) -> T
    where [];
    options options: EllipticOptions;
    detailed carlson_rg_with_options -> SpecialFunctionResult<T>;
    value {
        run(
            Family::Rg,
            [x, y, z, T::ZERO],
            EllipticOptions::default(),
            None,
        )?
        .value()
    }
    report {
        run(Family::Rg, [x, y, z, T::ZERO], options, None)?.publish()
    }
    scoped pub carlson_rg_scoped ['scope, 'cap] (scope: impl Into<SolveScope<'scope, 'cap>>) {
        scoped(scope, options.iteration_budget(), |context| {
            run(Family::Rg, [x, y, z, T::ZERO], options, Some(context))
        })?
        .publish()
    }
}

/// Shift the defining contour into the right half-plane before duplication. Carlson (1995) permits
/// direct duplication there. The contour stays on each input square root's principal sheet and on
/// the chosen side of the p pole.
pub(crate) fn complex_rj(a: [C; 4], work: &mut Work) -> Result<Estimate, SciOxideError> {
    work.check()?;
    for z in a {
        crate::support::require_finite_param(z, "elliptic parameter z must be finite")?;
    }
    if a[3] == ZERO || a[..3].iter().filter(|&&z| z == ZERO).count() > 1 {
        let index = if a[3] == ZERO {
            3
        } else {
            a[..3].iter().position(|&z| z == ZERO).unwrap_or(0)
        };
        return Err(ERRORS.domain(ARGUMENTS[index], DomainReason::Pole));
    }
    if a[..3].iter().all(|z| z.re >= 0.0) && a[3].re > 0.0 {
        let [x, y, z, p] = a.map(Z::from_value);
        let [x, y, z, p] = [x?, y?, z?, p?];
        return companion(x, y, z, Companion::Rj(p), work);
    }
    work.try_step()?;
    let scale = a
        .iter()
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0, f64::max);
    let normalized = a.map(|z| z.unscale(scale));
    if normalized
        .iter()
        .zip(a)
        .any(|(&q, p)| q == ZERO && p != ZERO)
    {
        return Err(crate::support::ELLIPTIC_ERRORS.unstable());
    }
    let cuts = normalized.iter().filter(|z| z.re <= 0.0 && **z != ZERO);
    let lower = cuts.clone().any(|z| z.im == 0.0 && z.im.is_sign_negative())
        || cuts.clone().all(|z| z.im < 0.0);
    let mut margin = 1.0_f64;
    for z in cuts {
        if z.im.is_sign_negative() != lower {
            margin = margin.min(z.im.abs() * 0.5);
        }
    }
    if margin == 0.0 {
        // Name the cut argument whose signed zero leaves no admissible contour margin.
        let index = normalized
            .iter()
            .position(|z| {
                z.re <= 0.0
                    && *z != ZERO
                    && z.im.is_sign_negative() != lower
                    && z.im.abs() * 0.5 == 0.0
            })
            .unwrap_or(3);
        return Err(SciOxideError::domain_error(
            ARGUMENTS[index],
            DomainReason::OutsideDomain,
            "incompatible RJ signed cut limits",
        ));
    }
    let shift = C::new(
        1.0 - normalized.iter().map(|z| z.re).fold(0.0, f64::min),
        if lower { -margin } else { margin },
    );
    // Logarithmic radial spacing resolves widely separated branch/pole scales;
    // squaring the coordinate removes the integrable zero-endpoint radical.
    let smallest = normalized
        .iter()
        .filter(|z| **z != ZERO)
        .map(|z| z.norm())
        .fold(1.0, f64::min);
    let stretch = (-smallest.ln()).max(1.0);
    let denominator = -(-stretch).exp_m1();
    let prefix = crate::elliptic_family::elliptic_quadrature::integrate(
        |u, _work| {
            let exponential = (stretch * (u * u - 1.0)).exp();
            let radius = exponential * (-(-stretch * u * u).exp_m1()) / denominator;
            let derivative = 2.0 * stretch * u * exponential / denominator;
            let t = Z::from_value(shift)?
                .mul(Z::from_value(scioxide_core::Complex::new(radius, 0.0))?)?;
            let [x, y, z, p] = normalized.map(Z::from_value);
            let [x, y, z, p] = [x?, y?, z?, p?];
            let denominator = t
                .add(x)?
                .sqrt()?
                .mul(t.add(y)?.sqrt()?)?
                .mul(t.add(z)?.sqrt()?)?
                .mul(t.add(p)?)?;
            Ok(Estimate::new(
                Z::from_value(shift)?
                    .mul(Z::from_value(scioxide_core::Complex::new(
                        1.5 * derivative,
                        0.0,
                    ))?)?
                    .div(denominator)?,
                32.0 * f64::EPSILON,
            ))
        },
        1.0,
        work,
    )?;
    let [x, y, z, p] = normalized.map(|z| Z::from_value(z + shift));
    let [x, y, z, p] = [x?, y?, z?, p?];
    let tail = companion(x, y, z, Companion::Rj(p), work)?;
    let factor = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?.div(
        Z::from_value(scioxide_core::Complex::new(scale, 0.0))?
            .mul(Z::from_value(scioxide_core::Complex::new(scale, 0.0))?.sqrt()?)?,
    )?;
    combine(
        [prefix.value, tail.value, Z::ZERO],
        factor,
        prefix.relative.max(tail.relative) + 16.0 * f64::EPSILON,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncertain_cancellation_retains_an_absolute_bound() {
        let e = combine(
            [
                Z::from_value(scioxide_core::Complex::new(1.0, 0.0)).unwrap(),
                Z::from_value(scioxide_core::Complex::new(-1.0, 0.0)).unwrap(),
                Z::ZERO,
            ],
            Z::from_value(scioxide_core::Complex::new(2.0, 0.0)).unwrap(),
            1e-12,
        )
        .unwrap();
        assert!(e.value.is_zero());
        assert_eq!(e.relative, 0.0);
        assert_eq!(e.absolute, 4e-12);
        let work = Work::elliptic(1, 1e-20, None);
        let r = kernel::report::<f64>(
            Ok(e),
            &work,
            scioxide_core::algorithm_component!("cancellation"),
        )
        .unwrap()
        .publish()
        .unwrap();
        assert_eq!(
            r.termination(),
            crate::SpecialFunctionTermination::Converged
        );
        assert!(r.value().unwrap().absolute_error() >= 4e-12);
    }
}
