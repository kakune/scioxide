//! Jacobi functions with the parameter m, AGM descent and real-parameter continuation.
//!
//! Every component carries a first-order absolute error: the amplitude error
//! of the period reduction and descent, propagated through the addition
//! formulas and the parameter transformations, plus the rounding of each
//! formed value. The errors vanish with the argument and scale with the
//! components, so subnormal values keep representable estimates.
pub(crate) use crate::elliptic_family::elliptic_kernel::Component;
use crate::elliptic_family::elliptic_kernel::{self as kernel, principal_amplitude};
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO, real};
use crate::{EllipticOptions, SpecialFunctionResult};
use core::f64::consts::{PI, TAU};
#[allow(unused_imports)] // A sibling may enable core/std through feature unification.
use scioxide_core::RealScalar;
use scioxide_core::algorithm_component;
use scioxide_core::{DomainReason, NumericScalar, SciOxideError, SolveContext};

const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy)]
struct Values<V = C> {
    sn: V,
    cn: V,
    dn: V,
    amplitude: f64,
    /// Absolute errors of sn, cn, dn and the amplitude.
    errors: [f64; 4],
}

/// First-order absolute error of a product of `(value, error)` factors, including one rounding per
/// factor.
fn product_error(factors: &[(f64, f64)]) -> f64 {
    let magnitude: f64 = factors.iter().map(|(value, _)| value.abs()).product();
    factors
        .iter()
        .enumerate()
        .map(|(index, &(_, error))| {
            error
                * factors
                    .iter()
                    .enumerate()
                    .filter(|&(other, _)| other != index)
                    .map(|(_, (value, _))| value.abs())
                    .product::<f64>()
        })
        .sum::<f64>()
        + factors.len() as f64 * EPS * magnitude
}

/// AGM and inverse-sine amplitude descent (DLMF 22.20), keeping the
/// complementary parameter independently. `argument_error` is the absolute error already present in
/// `u`; the amplitude derivative `dn` is at most one.
fn real_base(
    u: f64,
    m: f64,
    mc: f64,
    argument_error: f64,
    need_amplitude: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Values<f64>, SciOxideError> {
    // The cubic correction is below half a binary64 quantum, including the
    // subnormal lattice. Avoid losing u in the initial multiplication by AGM.
    if u.abs() < 1e-154 {
        work.try_step()?;
        return Ok(Values {
            sn: u,
            cn: 1.0,
            dn: 1.0,
            amplitude: u,
            errors: [
                2.0 * EPS * u.abs() + argument_error,
                0.0,
                0.0,
                argument_error,
            ],
        });
    }
    if m == 0.0 {
        work.try_step()?;
        let (sn, cn) = u.sin_cos();
        return Ok(Values {
            sn,
            cn,
            dn: 1.0,
            amplitude: u,
            errors: [
                2.0 * EPS * sn.abs() + cn.abs() * argument_error,
                2.0 * EPS * cn.abs() + sn.abs() * argument_error,
                0.0,
                argument_error,
            ],
        });
    }
    if mc == 0.0 {
        work.try_step()?;
        let e = (-u.abs()).exp();
        let cn = 2.0 * e / (1.0 + e * e);
        let sn = u.tanh();
        let amplitude = if need_amplitude { sn.atan2(cn) } else { 0.0 };
        // d tanh/du = sech², d sech/du = -sech tanh and d am/du = sech.
        let sech_error = 4.0 * EPS * cn + (cn * sn).abs() * argument_error;
        return Ok(Values {
            sn,
            cn,
            dn: cn,
            amplitude,
            errors: [
                2.0 * EPS * sn.abs() + cn * cn * argument_error,
                sech_error,
                sech_error,
                2.0 * EPS * amplitude.abs() + cn * argument_error,
            ],
        });
    }
    // DLMF 22.20.1--5: ascend the arithmetic-geometric mean and descend
    // the amplitude. Store c_n/a_n, the coefficient in the inverse sine step.
    work.try_step()?;
    let mut ratios = [0.0; 32];
    let mut count = 0;
    let (mut a, mut b) = (1.0, mc.sqrt());
    while (a - b).abs() > 2.0 * EPS * a {
        work.try_step()?;
        if count == ratios.len() {
            return Err(ERRORS.unstable());
        }
        let mean = 0.5 * (a + b);
        ratios[count] = (0.5 * (a - b)) / mean;
        count += 1;
        b = (a * b).sqrt();
        a = mean;
    }
    let period = PI / a;
    let periods = (u / period).round();
    let product = a * u;
    let angle = (-periods).mul_add(PI, product);
    let tail = a.mul_add(u, -product) - periods * crate::support::constants::PI.lo;
    let reduced = (angle + tail) / a;
    if !reduced.is_finite() {
        return Err(ERRORS.unstable());
    }
    // Reflect around K/2 so the inverse-sine descent never reconstructs a
    // tiny cn by subtracting an amplitude from pi/2.
    let quarter = 0.5 * period;
    let reflected = reduced.abs() > 0.5 * quarter;
    let distance = if reflected {
        quarter - reduced.abs()
    } else {
        reduced.abs()
    };
    let mut amplitude = (a * distance) * (2.0_f64).powi(count as i32);
    for &ratio in ratios[..count].iter().rev() {
        work.try_step()?;
        let sine = ratio * amplitude.sin();
        if sine.abs() > 1.0 {
            return Err(ERRORS.unstable());
        }
        amplitude = 0.5 * (amplitude + sine.asin());
    }
    let (s, c) = amplitude.sin_cos();
    let d = (mc + m * c * c).sqrt();
    let (mut sn, mut cn, dn) = if reflected {
        (c / d, mc.sqrt() * s / d, mc.sqrt() / d)
    } else {
        (s, c, d)
    };
    sn = sn.copysign(reduced);
    // For 0 <= m <= 1, |am(v)| <= |v|; this also bounds phase error
    // when a component-only call does not need the amplitude itself.
    let phi = if need_amplitude {
        sn.atan2(cn)
    } else {
        reduced
    };
    if periods % 2.0 != 0.0 {
        sn = -sn;
        cn = -cn;
    }
    let amplitude = periods.mul_add(PI, phi);
    // Phase reduction and AGM roundoff propagate through d am/du = dn.
    // Each amplitude descent adds at most eight first-order rounding terms.
    let steps = (count + 2) as f64;
    let phase =
        EPS * (steps * (periods * period).abs() + 3.0 * steps * phi.abs()) + dn * argument_error;
    Ok(Values {
        sn,
        cn,
        dn,
        amplitude,
        errors: [
            cn.abs() * phase + 8.0 * steps * EPS * sn.abs(),
            sn.abs() * phase + 8.0 * steps * EPS * cn.abs(),
            (m * sn * cn).abs() / dn * phase
                + EPS * (8.0 * steps * dn + (mc + m.abs() * cn * cn) / dn),
            phase + EPS * amplitude.abs(),
        ],
    })
}
fn complex_base(
    u: C,
    m: f64,
    mc: f64,
    argument_error: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<Values, SciOxideError> {
    if u.im == 0.0 {
        return real_base(u.re, m, mc, argument_error, false, work).map(|v| Values {
            sn: real(v.sn),
            cn: real(v.cn),
            dn: real(v.dn),
            amplitude: v.amplitude,
            errors: v.errors,
        });
    }
    let x = real_base(u.re, m, mc, argument_error, false, work)?;
    let y = real_base(u.im, mc, m, argument_error, false, work)?;
    work.try_step()?;
    let [xs, xc, xd] = [
        (x.sn, x.errors[0]),
        (x.cn, x.errors[1]),
        (x.dn, x.errors[2]),
    ];
    let [ys, yc, yd] = [
        (y.sn, y.errors[0]),
        (y.cn, y.errors[1]),
        (y.dn, y.errors[2]),
    ];
    let denominator = yc.0 * yc.0 + m * xs.0 * xs.0 * ys.0 * ys.0;
    if denominator == 0.0 {
        return Err(ERRORS.domain(scioxide_core::argument!("u"), DomainReason::Pole));
    }
    let denominator_error =
        product_error(&[yc, yc]) + m.abs() * product_error(&[xs, xs, ys, ys]) + EPS * denominator;
    // First-order error of N/D, including its rounding.
    let quotient = |numerator: &[(f64, f64)], scale: f64| {
        let value: f64 = scale * numerator.iter().map(|(value, _)| value).product::<f64>();
        let error = scale.abs() * product_error(numerator);
        (
            value / denominator,
            (error + (value / denominator).abs() * denominator_error) / denominator.abs()
                + EPS * (value / denominator).abs(),
        )
    };
    let (sn_re, sn_re_error) = quotient(&[xs, yd], 1.0);
    let (sn_im, sn_im_error) = quotient(&[xc, xd, ys, yc], 1.0);
    let (cn_re, cn_re_error) = quotient(&[xc, yc], 1.0);
    let (cn_im, cn_im_error) = quotient(&[xs, xd, ys, yd], -1.0);
    let (dn_re, dn_re_error) = quotient(&[xd, yc, yd], 1.0);
    let (dn_im, dn_im_error) = quotient(&[xs, xc, ys], -m);
    let (sn, cn, dn) = (
        C::new(sn_re, sn_im),
        C::new(cn_re, cn_im),
        C::new(dn_re, dn_im),
    );
    if !sn.is_finite() || !cn.is_finite() || !dn.is_finite() {
        return Err(ERRORS.unstable());
    }
    let sn_error = sn_re_error + sn_im_error;
    let cn_error = cn_re_error + cn_im_error;
    Ok(Values {
        sn,
        cn,
        dn,
        amplitude: 0.0,
        // The principal amplitude is a logarithm of the larger of cn ± i sn,
        // whose modulus is at least one.
        errors: [
            sn_error,
            cn_error,
            dn_re_error + dn_im_error,
            sn_error + cn_error,
        ],
    })
}
/// Real parameter transformations keep the complementary parameter separately.
fn transformed<V: NumericScalar<Real = f64>>(
    u: V,
    m: f64,
    need_amplitude: bool,
    work: &mut Work<'_, '_, '_>,
    base: impl Fn(V, f64, f64, f64, bool, &mut Work<'_, '_, '_>) -> Result<Values<V>, SciOxideError>,
) -> Result<Values<V>, SciOxideError> {
    if (0.0..=1.0).contains(&m) {
        return base(u, m, 1.0 - m, 0.0, need_amplitude, work);
    }
    work.try_step()?;
    let negative = m < 0.0;
    let scale = if negative { (1.0 - m).sqrt() } else { m.sqrt() };
    let mc = if negative {
        1.0 / (1.0 - m)
    } else {
        (m - 1.0) / m
    };
    let mu = if negative {
        if m < -1.0 { 1.0 - mc } else { m / (m - 1.0) }
    } else {
        1.0 / m
    };
    let v = u * V::from_real(scale);
    if !v.has_finite_components() {
        return Err(ERRORS.unstable());
    }
    let t = base(v, mu, mc, 1.5 * EPS * v.abs(), need_amplitude, work)?;
    if negative {
        if t.dn == V::ZERO {
            return Err(ERRORS.domain(scioxide_core::argument!("u"), DomainReason::Pole));
        }
        let sn = t.sn / t.dn / V::from_real(scale);
        let cn = t.cn / t.dn;
        let dn = V::ONE / t.dn;
        let amplitude = if need_amplitude {
            let arg = sn.real_component().atan2(cn.real_component());
            arg + TAU * ((t.amplitude - arg) / TAU).round()
        } else {
            0.0
        };
        let sn_error = (t.errors[0] + t.sn.abs() / t.dn.abs() * t.errors[2]) / t.dn.abs() / scale
            + 3.0 * EPS * sn.abs();
        let cn_error = (t.errors[1] + cn.abs() * t.errors[2]) / t.dn.abs() + 2.0 * EPS * cn.abs();
        Ok(Values {
            sn,
            cn,
            dn,
            amplitude,
            errors: [
                sn_error,
                cn_error,
                t.errors[2] / (t.dn.abs() * t.dn.abs()) + EPS * dn.abs(),
                sn_error + cn_error + EPS * amplitude.abs(),
            ],
        })
    } else {
        let sn = t.sn / V::from_real(scale);
        let error = t.errors[0] / scale + 2.0 * EPS * sn.abs();
        let amplitude = if need_amplitude {
            sn.real_component().clamp(-1.0, 1.0).asin()
        } else {
            0.0
        };
        let slope = (1.0 - sn.real_component() * sn.real_component())
            .max(0.0)
            .sqrt();
        let amplitude_error = if slope > 0.0 {
            (error / slope).min((2.0 * error).sqrt())
        } else {
            (2.0 * error).sqrt()
        };
        Ok(Values {
            sn,
            cn: t.dn,
            dn: t.cn,
            amplitude,
            errors: [
                error,
                t.errors[2],
                t.errors[1],
                amplitude_error + EPS * amplitude.abs(),
            ],
        })
    }
}

fn evaluate_real(
    component: Component,
    u: f64,
    m: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<(f64, f64), SciOxideError> {
    work.check()?;
    if u == 0.0 {
        return Ok((
            match component {
                Component::Sn | Component::Amplitude => u,
                _ => 1.0,
            },
            0.0,
        ));
    }
    let t = transformed(
        u,
        m,
        matches!(component, Component::Amplitude),
        work,
        real_base,
    )?;
    Ok(match component {
        Component::Sn => (t.sn, t.errors[0]),
        Component::Cn => (t.cn, t.errors[1]),
        Component::Dn => (t.dn, t.errors[2]),
        Component::Amplitude => (t.amplitude, t.errors[3]),
    })
}

fn values(u: C, m: f64, work: &mut Work<'_, '_, '_>) -> Result<Values, SciOxideError> {
    transformed(u, m, false, work, |u, m, mc, error, _, work| {
        complex_base(u, m, mc, error, work)
    })
}
/// Stable hyperbolic values, keeping exponentially small cn/dn in scaled form.
fn hyperbolic(
    component: Component,
    u: C,
    real_output: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    if u.im == 0.0 && matches!(component, Component::Sn) {
        return Ok(Estimate::new(
            Z::from_value(scioxide_core::Complex::new(u.re.tanh(), 0.0))?,
            4.0 * f64::EPSILON,
        ));
    }
    if u.re.abs() > 700.0 && matches!(component, Component::Cn | Component::Dn) {
        let value = kernel::scaled_exp(C::new(-u.re.abs(), -u.im * u.re.signum()))?
            .mul(Z::from_value(scioxide_core::Complex::new(2.0, 0.0))?)?;
        return Ok(Estimate::new(value, 16.0 * f64::EPSILON));
    }
    // tanh and sech from exp(-|Re u|) avoid overflow in cosh.
    let sign = if u.re < 0.0 { -1.0 } else { 1.0 };
    let e = (u * real(-sign)).exp();
    let denominator = ONE + e * e;
    if denominator == ZERO {
        return Err(ERRORS.domain(scioxide_core::argument!("u"), DomainReason::Pole));
    }
    let sn = (ONE - e * e) / denominator * real(sign);
    let cn = e * real(2.0) / denominator;
    let v = match component {
        Component::Sn => sn,
        Component::Cn | Component::Dn => cn,
        Component::Amplitude => {
            if real_output {
                real(sn.re.atan2(cn.re))
            } else {
                principal_amplitude(sn, cn)?
            }
        }
    };
    if !v.is_finite() {
        return Err(ERRORS.unstable());
    }
    Ok(Estimate::new(
        Z::from_value(v)?,
        32.0 * f64::EPSILON / denominator.norm(),
    ))
}
/// Validate Jacobi arguments.
pub(crate) fn validate(u: C, m: f64) -> Result<(), SciOxideError> {
    crate::support::require_finite_param(u, "elliptic parameter u must be finite")?;
    crate::support::require_finite_param(real(m), "elliptic parameter value must be finite")?;
    Ok(())
}
pub(crate) fn evaluate(
    component: Component,
    u: C,
    m: f64,
    real_output: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    validate(u, m)?;
    work.check()?;
    if u == ZERO {
        return Estimate::exact(match component {
            Component::Sn | Component::Amplitude => u,
            _ => ONE,
        });
    }
    if m == 0.0 {
        work.try_step()?;
        if u.im.abs() > 700.0 && matches!(component, Component::Sn | Component::Cn) {
            let e = kernel::scaled_exp(C::new(-u.im, u.re))?;
            let inverse = kernel::scaled_exp(C::new(u.im, -u.re))?;
            let value = if matches!(component, Component::Sn) {
                e.sub(inverse)?.mul(Z::from_value(C::new(0.0, -0.5))?)?
            } else {
                e.add(inverse)?
                    .mul(Z::from_value(scioxide_core::Complex::new(0.5, 0.0))?)?
            };
            return Ok(Estimate::new(value, 16.0 * f64::EPSILON));
        }
        let value = match component {
            Component::Sn => u.sin(),
            Component::Cn => u.cos(),
            Component::Dn => ONE,
            Component::Amplitude => {
                if real_output {
                    u
                } else {
                    C::new(u.re.sin().atan2(u.re.cos()), u.im)
                }
            }
        };
        if !value.is_finite() {
            return Err(ERRORS.unstable());
        }
        return Ok(
            if matches!(component, Component::Dn)
                || real_output && matches!(component, Component::Amplitude)
            {
                Estimate::exact(value)?
            } else {
                Estimate::new(Z::from_value(value)?, 8.0 * f64::EPSILON)
            },
        );
    }
    if m == 1.0 {
        return hyperbolic(component, u, real_output, work);
    }
    let t = values(u, m, work)?;
    let (value, error) = match component {
        Component::Sn => (t.sn, t.errors[0]),
        Component::Cn => (t.cn, t.errors[1]),
        Component::Dn => (t.dn, t.errors[2]),
        Component::Amplitude => (
            if real_output {
                real(t.amplitude)
            } else {
                principal_amplitude(t.sn, t.cn)?
            },
            t.errors[3],
        ),
    };
    if !value.is_finite() {
        return Err(ERRORS.unstable());
    }
    let mut e = Estimate::new(Z::from_value(value)?, 0.0);
    e.absolute = error;
    Ok(e)
}
pub(crate) fn run<T: NumericScalar>(
    component: Component,
    u: T,
    m: T,
    options: EllipticOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<kernel::EllipticEvaluation<T>, SciOxideError> {
    let u = widen(u, "elliptic parameter u must be finite")?;
    if !m.has_finite_components() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "m must be finite",
        ));
    }
    let m = widen(m, "elliptic parameter m must be finite")?;
    let mut work = Work::elliptic(
        options.iteration_budget().get(),
        kernel::family_target::<T::Real>(),
        context,
    );
    if crate::support::is_real::<T>()
        && !(m.re == 1.0
            && u.re.abs() > 700.0
            && matches!(component, Component::Cn | Component::Dn))
    {
        let e = evaluate_real(component, u.re, m.re, &mut work);
        return kernel::report_real(
            e,
            &work,
            match component {
                Component::Sn => algorithm_component!("real_jacobi_sn"),
                Component::Cn => algorithm_component!("real_jacobi_cn"),
                Component::Dn => algorithm_component!("real_jacobi_dn"),
                Component::Amplitude => algorithm_component!("real_jacobi_amplitude"),
            },
            u.re == 0.0 || m.re == 0.0 && matches!(component, Component::Dn | Component::Amplitude),
        );
    }
    let e = if m.im == 0.0 {
        evaluate(
            component,
            u,
            m.re,
            crate::support::is_real::<T>(),
            &mut work,
        )
    } else {
        crate::elliptic_family::complex_jacobi::evaluate(component, u, m, &mut work)
    };
    kernel::report(
        e,
        &work,
        match component {
            Component::Sn => algorithm_component!("jacobi_sn"),
            Component::Cn => algorithm_component!("jacobi_cn"),
            Component::Dn => algorithm_component!("jacobi_dn"),
            Component::Amplitude => algorithm_component!("jacobi_amplitude"),
        },
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Jacobi elliptic function `sn(u|m)`.

```text
sn(u|m) = sin(am(u|m)),   u = ∫₀^am(u|m) (1 − m sin²θ)^(−1/2) dθ
```

`m = k²` is the parameter; DLMF §22.2 writes the same function with the modulus `k`. For real `u`
and `m` every finite pair is accepted. Parameters `m < 0` and `m > 1` are handled by the real
transformations of DLMF §22.17. As functions of complex `u`, `sn`, `cn` and `dn` are meromorphic
with poles at `u = 2jK + (2l + 1)iK'`, where `K = K(m)`, `K' = K(1 − m)` and `j, l` are integers.
`m` may be real while `u` is complex.

Real `m` uses the arithmetic-geometric mean and inverse-sine amplitude descent (DLMF §22.20) after reduction by the real period. Complex `m` uses quotients of Jacobi theta functions after
modular reduction (DLMF §20.7, §22.2).

# Examples

```
// sn(u|0) = sin u and sn(u|1) = tanh u
let s = scioxide_special::jacobi_sn(0.5_f64, 0.0)?;
assert!((s - 0.5_f64.sin()).abs() < 1e-14);
let s = scioxide_special::jacobi_sn(0.5_f64, 1.0)?;
assert!((s - 0.5_f64.tanh()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; for `f64` the measured median error is at most
1.2 ulps (a median, not a maximum). Complex arguments have no uniform ulp bound. See the [crate
accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) or `NumericalInstability` (`LossOfSignificance`) at or next to a pole of
  complex `u`, when the reduction cannot resolve the value.

# References

- NIST DLMF, §22.2, §22.16 and §22.17. <https://dlmf.nist.gov/22>
- NIST DLMF, §22.20(ii), Arithmetic-Geometric Mean. <https://dlmf.nist.gov/22.20>"###)]

    pub jacobi_sn [T: NumericScalar] (u: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: EllipticOptions;
    detailed jacobi_sn_with_options -> SpecialFunctionResult<T>;
    value {
        let options: EllipticOptions = EllipticOptions::default();
        run(Component::Sn, u, m, options, None)?.value()
    }
    report {
        run(Component::Sn, u, m, options, None)?.publish()
    }
    scoped pub jacobi_sn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            run(Component::Sn, u, m, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Jacobi elliptic function `cn(u|m)`.

```text
cn(u|m) = cos(am(u|m)),   u = ∫₀^am(u|m) (1 − m sin²θ)^(−1/2) dθ
```

`m = k²` is the parameter; DLMF §22.2 writes the same function with the modulus `k`. For real `u`
and `m` every finite pair is accepted. Parameters `m < 0` and `m > 1` are handled by the real
transformations of DLMF §22.17. As functions of complex `u`, `sn`, `cn` and `dn` are meromorphic
with poles at `u = 2jK + (2l + 1)iK'`, where `K = K(m)`, `K' = K(1 − m)` and `j, l` are integers.
`m` may be real while `u` is complex.

Real `m` uses the arithmetic-geometric mean and inverse-sine amplitude descent (DLMF §22.20) after reduction by the real period. Complex `m` uses quotients of Jacobi theta functions after
modular reduction (DLMF §20.7, §22.2).

# Examples

```
// cn(u|0) = cos u and cn(u|1) = sech u
let c = scioxide_special::jacobi_cn(0.5_f64, 0.0)?;
assert!((c - 0.5_f64.cos()).abs() < 1e-14);
let c = scioxide_special::jacobi_cn(0.5_f64, 1.0)?;
assert!((c - 1.0 / 0.5_f64.cosh()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) or `NumericalInstability` (`LossOfSignificance`) at or next to a pole of
  complex `u`, when the reduction cannot resolve the value.

# References

- NIST DLMF, §22.2, §22.16 and §22.17. <https://dlmf.nist.gov/22>
- NIST DLMF, §22.20(ii), Arithmetic-Geometric Mean. <https://dlmf.nist.gov/22.20>"###)]

    pub jacobi_cn [T: NumericScalar] (u: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: EllipticOptions;
    detailed jacobi_cn_with_options -> SpecialFunctionResult<T>;
    value {
        let options: EllipticOptions = EllipticOptions::default();
        run(Component::Cn, u, m, options, None)?.value()
    }
    report {
        run(Component::Cn, u, m, options, None)?.publish()
    }
    scoped pub jacobi_cn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            run(Component::Cn, u, m, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Jacobi elliptic function `dn(u|m)`.

```text
dn(u|m) = (1 − m sn²(u|m))^(1/2),   dn(0|m) = 1
```

`m = k²` is the parameter; DLMF §22.2 writes the same function with the modulus `k`. For real `u`
and `m` every finite pair is accepted. Parameters `m < 0` and `m > 1` are handled by the real
transformations of DLMF §22.17. As functions of complex `u`, `sn`, `cn` and `dn` are meromorphic
with poles at `u = 2jK + (2l + 1)iK'`, where `K = K(m)`, `K' = K(1 − m)` and `j, l` are integers.
`m` may be real while `u` is complex.

Real `m` uses the arithmetic-geometric mean and inverse-sine amplitude descent (DLMF §22.20) after reduction by the real period. Complex `m` uses quotients of Jacobi theta functions after
modular reduction (DLMF §20.7, §22.2).

# Examples

```
// dn(u|0) = 1 and dn(u|1) = sech u
assert!((scioxide_special::jacobi_dn(0.5_f64, 0.0)? - 1.0).abs() < 1e-14);
let d = scioxide_special::jacobi_dn(0.5_f64, 1.0)?;
assert!((d - 1.0 / 0.5_f64.cosh()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) or `NumericalInstability` (`LossOfSignificance`) at or next to a pole of
  complex `u`, when the reduction cannot resolve the value.

# References

- NIST DLMF, §22.2, §22.16 and §22.17. <https://dlmf.nist.gov/22>
- NIST DLMF, §22.20(ii), Arithmetic-Geometric Mean. <https://dlmf.nist.gov/22.20>"###)]

    pub jacobi_dn [T: NumericScalar] (u: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: EllipticOptions;
    detailed jacobi_dn_with_options -> SpecialFunctionResult<T>;
    value {
        let options: EllipticOptions = EllipticOptions::default();
        run(Component::Dn, u, m, options, None)?.value()
    }
    report {
        run(Component::Dn, u, m, options, None)?.publish()
    }
    scoped pub jacobi_dn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            run(Component::Dn, u, m, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Jacobi amplitude `am(u|m)`, the inverse of `F(φ|m)` in `φ`.

```text
u = ∫₀^am(u|m) (1 − m sin²θ)^(−1/2) dθ
```

`m = k²` is the parameter (DLMF §22.16). For real `u` and `m ≤ 1`, `am` is the continuous increasing
real branch with `am(u + 2K|m) = am(u|m) + π`, `K = K(m)`. For `m > 1` it follows the real branch
given by the reciprocal-parameter transformation (DLMF §22.17). For complex arguments it is the
principal value `−i ln(cn(u|m) + i sn(u|m))`, which has logarithmic branch cuts. `m` may be real
while `u` is complex.

Real `m` uses the arithmetic-geometric mean and inverse-sine amplitude descent (DLMF §22.20) after reduction by the real period. Complex `m` uses quotients of Jacobi theta functions after
modular reduction (DLMF §20.7, §22.2).

# Examples

```
// am(u|0) = u and am(u|1) = gd(u) = atan(sinh u)
let a = scioxide_special::jacobi_amplitude(0.5_f64, 0.0)?;
assert!((a - 0.5).abs() < 1e-14);
let a = scioxide_special::jacobi_amplitude(0.5_f64, 1.0)?;
assert!((a - 0.5_f64.sinh().atan()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) or `NumericalInstability` (`LossOfSignificance`) at or next to a pole of
  complex `u`, when the reduction cannot resolve the value.

# References

- NIST DLMF, §22.2, §22.16 and §22.17. <https://dlmf.nist.gov/22>
- NIST DLMF, §22.20(ii), Arithmetic-Geometric Mean. <https://dlmf.nist.gov/22.20>"###)]

    pub jacobi_amplitude [T: NumericScalar] (u: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: EllipticOptions;
    detailed jacobi_amplitude_with_options -> SpecialFunctionResult<T>;
    value {
        let options: EllipticOptions = EllipticOptions::default();
        run(Component::Amplitude, u, m, options, None)?.value()
    }
    report {
        run(Component::Amplitude, u, m, options, None)?.publish()
    }
    scoped pub jacobi_amplitude_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            run(Component::Amplitude, u, m, options, Some(context))?.publish()
        })
    }
}
