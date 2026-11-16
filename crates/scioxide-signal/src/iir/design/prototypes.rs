use super::*;

/// Designs a Butterworth, Chebyshev, Bessel or elliptic IIR filter.
///
/// The design takes the analog lowpass prototype of `order` and `family`
/// (see [`iir_prototype`]) and maps it to `band` with
/// [`PoleZeroGain::transform`]. A digital design then applies the bilinear
/// transform `s = 2(z − 1)/(z + 1)`, after prewarping each band edge `f`, in
/// multiples of Nyquist, to `ω = 2 tan(πf/2)`. The digital response at each
/// band edge therefore equals the prototype response at `ω = 1`. The filter
/// has `order` poles for a lowpass or highpass band and `2·order` poles for a
/// bandpass or bandstop band. Elliptic designs and the Bessel magnitude
/// normalization solve scalar equations with Brent's method in a fresh root
/// scope; [`design_iir_scoped`] runs them inside the caller's solve.
///
/// # Errors
///
/// - `InvalidInput` if `order` is zero, if a band edge is not positive, if
///   band edges are not increasing, if a digital band edge is not below one,
///   if ripple or attenuation is not positive, or if an elliptic
///   `attenuation` does not exceed `ripple`.
/// - `NonFinite` if a band edge, ripple or attenuation is NaN or infinite.
/// - `NumericalInstability` (`Overflow` or `Underflow`) if the gain is above
///   the finite range or nonzero and below the normal range, as for high
///   orders at low cutoffs.
/// - `NumericalInstability` (`LossOfSignificance`) if a pole is not strictly
///   stable after rounding.
/// - `ConvergenceFailure` if a design equation does not converge.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
/// // Fourth-order Butterworth lowpass with half power at a quarter of Nyquist.
/// let band = IirBand::Lowpass(0.25_f64);
/// let design = design_iir(4, IirFamily::Butterworth, band, IirDomain::Digital)?;
/// assert!((design.zpk().response(0.0)?.norm() - 1.0).abs() < 1e-12);
/// assert!((design.zpk().response(0.25)?.norm_sqr() - 0.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
///   3rd ed., Prentice Hall, 2010, ch. 7.
/// - T. W. Parks and C. S. Burrus, *Digital Filter Design*, Wiley, 1987.
pub fn design_iir<S>(
    order: usize,
    family: IirFamily<S>,
    band: IirBand<S>,
    domain: IirDomain,
) -> Result<IirDesign<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    design_iir_scoped(
        order,
        family,
        band,
        domain,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Designs an IIR filter as [`design_iir`] does, as one epoch of `scope`.
///
/// The Brent solves of the elliptic and Bessel design equations count toward
/// the limits of `scope` and observe its cancellation and deadline, as
/// described in [scoped designs](crate#scoped-designs).
///
/// # Errors
///
/// - The errors of [`design_iir`].
/// - `ConvergenceFailure` if the solves exhaust the limits of `scope`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
pub fn design_iir_scoped<'scope, 'capability, S>(
    order: usize,
    family: IirFamily<S>,
    band: IirBand<S>,
    domain: IirDomain,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<IirDesign<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    run_design(
        scope,
        scioxide_core::algorithm_id!("signal", "iir_design", 1),
        |context| design_impl(order, family, band, domain, context),
    )
}
/// Design a filter.
fn design_impl<S>(
    order: usize,
    family: IirFamily<S>,
    band: IirBand<S>,
    domain: IirDomain,
    context: &mut SolveContext<'_, '_>,
) -> Result<IirDesign<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let mut ctx = Some(context);
    let band_argument = scioxide_core::argument!("band");
    let band = map_band(
        band,
        |x| {
            positive_parameter(x, band_argument)?;
            if domain == IirDomain::Digital {
                if x >= S::ONE {
                    return Err(invalid(
                        band_argument,
                        "digital band edges must lie below Nyquist",
                    ));
                }
                check(number::<S>(2.)? * (S::PI * x * S::HALF).tan())
            } else {
                Ok(x)
            }
        },
        band_argument,
    )?;
    let prototype = prototype_impl(order, family, ctx.as_deref_mut())?;
    step(&mut ctx)?;
    let mut zpk = prototype.transform(band)?;
    if domain == IirDomain::Digital {
        zpk = zpk.bilinear(S::ONE)?;
    }
    if !zpk.is_stable() {
        return Err(failure("design poles are not strictly stable"));
    }
    Ok(IirDesign { zpk, order })
}
/// Computes the analog lowpass prototype of `order` and `family`.
///
/// The prototype is normalized as [`IirFamily`] describes. Butterworth and
/// Chebyshev poles and zeros have closed forms. Elliptic poles and zeros come
/// from Jacobi elliptic functions, with the modulus found from the degree
/// equation by Brent's method. Bessel poles are the zeros of the reverse
/// Bessel polynomial `θₙ`, the reciprocals of the zeros of the ordinary Bessel
/// polynomial `yₙ`. They are found in `f64` by Aberth–Ehrlich iteration on an
/// evaluation of `yₙ` through modified Bessel functions, which avoids the
/// cancellation of the polynomial coefficients, and are accurate to a few
/// units of `ε` at every order. The iterations run in a fresh root scope;
/// [`iir_prototype_scoped`] runs them inside the caller's solve.
///
/// # Errors
///
/// - `InvalidInput` if `order` is zero, if ripple or attenuation is not
///   positive, or if an elliptic `attenuation` does not exceed `ripple`.
/// - `NonFinite` if ripple or attenuation is NaN or infinite.
/// - `NumericalInstability` if the gain is above the finite range or nonzero
///   and below the normal range, if a pole is not strictly stable, or if the
///   Bessel zeros fail their accuracy check.
/// - `ConvergenceFailure` if an iteration does not converge.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{IirFamily, iir_prototype};
/// let prototype = iir_prototype(3, IirFamily::<f64>::Butterworth)?;
/// assert_eq!(prototype.poles().len(), 3);
/// // Butterworth poles lie on the unit circle.
/// for p in prototype.poles() {
///     assert!((p.norm() - 1.0).abs() < 1e-14);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - T. W. Parks and C. S. Burrus, *Digital Filter Design*, Wiley, 1987.
/// - O. Aberth, "Iteration methods for finding all zeros of a polynomial
///   simultaneously", *Math. Comp.* 27 (1973), 339–344.
pub fn iir_prototype<S>(
    order: usize,
    family: IirFamily<S>,
) -> Result<PoleZeroGain<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    iir_prototype_scoped(
        order,
        family,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes the analog lowpass prototype as [`iir_prototype`] does, as one
/// epoch of `scope`.
///
/// The elliptic modulus solve, the Jacobi elliptic functions and the Bessel
/// iterations count toward the limits of `scope` and observe its cancellation
/// and deadline, as described in [scoped designs](crate#scoped-designs).
///
/// # Errors
///
/// - The errors of [`iir_prototype`].
/// - `ConvergenceFailure` if the solves exhaust the limits of `scope`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
pub fn iir_prototype_scoped<'scope, 'capability, S>(
    order: usize,
    family: IirFamily<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<PoleZeroGain<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    run_design(
        scope,
        scioxide_core::algorithm_id!("signal", "iir_prototype", 1),
        |context| prototype_impl(order, family, Some(context)),
    )
}
/// Normalized analog prototype.
fn prototype_impl<S>(
    order: usize,
    family: IirFamily<S>,
    mut ctx: Option<&mut SolveContext<'_, '_>>,
) -> Result<PoleZeroGain<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    if order == 0 {
        return Err(invalid(
            scioxide_core::argument!("order"),
            "prototype order must be positive",
        ));
    }
    let family_argument = scioxide_core::argument!("family");
    let n = scalar::<S>(order)?;
    let one = S::ONE;
    let two = S::TWO;
    let mut poles = zeros(order)?;
    let mut z = Vec::new();
    let mut dc = one;
    match family {
        IirFamily::Butterworth | IirFamily::ChebyshevI { .. } | IirFamily::ChebyshevII { .. } => {
            let mu = match family {
                IirFamily::ChebyshevI { ripple } => {
                    let e = ripple_power(ripple, family_argument)?.sqrt();
                    if order.is_multiple_of(2) {
                        dc = one / (one + e * e).sqrt();
                    }
                    (one / e).asinh() / n
                }
                IirFamily::ChebyshevII { attenuation } => {
                    ripple_power(attenuation, family_argument)?.sqrt().asinh() / n
                }
                _ => S::ZERO,
            };
            z.try_reserve_exact(order).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("prototype_zeros"),
                    order,
                )
            })?;
            for (i, p) in poles.iter_mut().enumerate() {
                step(&mut ctx)?;
                let angle = S::PI * (scalar::<S>(2 * i + 1)?) / (two * n);
                let (sin, cos) = (angle.sin(), angle.cos());
                *p = match family {
                    IirFamily::Butterworth => Complex::new(-sin, cos),
                    IirFamily::ChebyshevI { .. } => Complex::new(-mu.sinh() * sin, mu.cosh() * cos),
                    _ => {
                        if 2 * i + 1 != order {
                            z.push(Complex::new(S::ZERO, one / cos));
                        }
                        Complex::new(-mu.sinh() * sin, mu.cosh() * cos).inv()
                    }
                };
                if 2 * i + 1 == order {
                    p.im = S::ZERO;
                }
            }
            for i in 0..order / 2 {
                poles[order - 1 - i] = poles[i].conj();
            }
            for i in 0..z.len() / 2 {
                let j = z.len() - 1 - i;
                z[j] = z[i].conj();
            }
        }
        IirFamily::Bessel(norm) => {
            // Binary64 poles for both widths; see `bessel` for the certificate.
            let delay = bessel::delay_poles(order, &mut ctx)?;
            // Phase normalization gives a unit geometric-mean pole radius.
            let radius = if norm == BesselNormalization::Phase {
                let logarithm = delay.iter().map(|p| p.norm().ln()).sum::<f64>();
                (logarithm / scalar::<f64>(order)?).exp()
            } else {
                1.0
            };
            for (pole, p) in poles.iter_mut().zip(&delay) {
                let p = p.unscale(radius);
                *pole = Complex::new(S::try_from_f64_lossy(p.re)?, S::try_from_f64_lossy(p.im)?);
            }
            let scale = match norm {
                BesselNormalization::Delay | BesselNormalization::Phase => one,
                BesselNormalization::Magnitude => {
                    let temporary = normalized(&[], &poles, one)?;
                    let residual = ScalarFn::new(|frequency| {
                        Ok(temporary.response(frequency)?.norm_sqr() - S::HALF)
                    });
                    let root = design_root(
                        &residual,
                        scioxide_core::Interval::new(S::ZERO, n + one)?,
                        &mut ctx,
                    )?;
                    one / root.point()
                }
            };
            for p in &mut poles {
                *p = check(p.scale(scale))?;
            }
        }
        IirFamily::Elliptic {
            ripple,
            attenuation,
        } => {
            let eps2 = ripple_power(ripple, family_argument)?;
            let m1 = eps2 / ripple_power(attenuation, family_argument)?;
            if m1 <= S::ZERO || m1 >= one {
                return Err(inconsistent(
                    family_argument,
                    "elliptic attenuation must exceed the ripple",
                ));
            }
            if order == 1 {
                poles[0] = c(-one / eps2.sqrt());
                return normalized(&z, &poles, one);
            }
            let k1 = k(m1, &mut ctx)?;
            let target = n * k1 / rf(S::ZERO, m1, one, &mut ctx)?;
            let residual = EllipticResidual {
                fraction: target / (one + target),
            };
            let m = design_root(
                &residual,
                scioxide_core::Interval::new(S::ZERO, one)?,
                &mut ctx,
            )?
            .point();
            if m <= S::ZERO || m >= one {
                return Err(failure("elliptic modulus lost precision"));
            }
            let km = k(m, &mut ctx)?;
            // inverse sc(v,1-m1)=1/epsilon, using incomplete F via Carlson RF.
            let sin = one / (one + eps2).sqrt();
            let cos2 = eps2 / (one + eps2);
            let r = sin * rf(cos2, cos2 + m1 * sin * sin, one, &mut ctx)?;
            let (sv, cv, dv) = jacobi(km * r / (n * k1), one - m, &mut ctx)?;
            z.try_reserve_exact(order).map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("elliptic_zeros"), order)
            })?;
            let mut index = 0;
            for j in (1 - order % 2..order).step_by(2) {
                step(&mut ctx)?;
                let (s, cn, d) = jacobi(scalar::<S>(j)? * km / n, m, &mut ctx)?;
                let den = one - (d * sv) * (d * sv);
                let p = Complex::new(-cn * d * sv * cv / den, -s * dv / den);
                poles[index] = p;
                index += 1;
                if j != 0 {
                    poles[index] = p.conj();
                    index += 1;
                    let zero = Complex::new(S::ZERO, one / (m.sqrt() * s));
                    z.push(zero);
                    z.push(zero.conj());
                }
            }
            if order.is_multiple_of(2) {
                dc = one / (one + eps2).sqrt();
            }
        }
    }
    normalized(&z, &poles, dc)
}
fn normalized<S: RealScalar>(
    z: &[Complex<S>],
    p: &[Complex<S>],
    dc: S,
) -> Result<PoleZeroGain<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut product = ZpkGain::new(c(dc))?;
    for (i, p) in p.iter().enumerate() {
        product = product.times(-*p)?;
        if let Some(z) = z.get(i) {
            product = product.over(-*z)?;
        }
    }
    let mut gain = product.finish()?;
    // Named prototypes are real. Roundoff in conjugate products is not a complex design.
    gain.im = S::ZERO;
    let out = PoleZeroGain::new(IirDomain::Analog, z, p, gain)?;
    if !out.is_stable() {
        return Err(failure("unstable prototype"));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod normalization_tests {
    use super::*;
    fn normalization_counts<S: RealScalar + NumericScalar<Real = S>>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        // A first-order delay-normalized Bessel prototype has pole -1.
        let prototype = normalized(&[], &[Complex::new(-S::ONE, S::ZERO)], S::ONE).unwrap();
        let function = ScalarFn::new(|w| Ok(prototype.response(w)?.norm_sqr() - S::HALF));
        let result = design_root(
            &function,
            scioxide_core::Interval::new(S::ZERO, S::TWO).unwrap(),
            &mut None,
        )
        .unwrap();
        let tolerance = S::EPSILON * (S::TWO + S::TWO);
        assert!((result.point() - S::ONE).abs() <= tolerance);
        assert!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Function)
                < (S::MANTISSA_DIGITS + 8) as usize
        );
        for fraction in [S::HALF, S::ONE / (S::TWO + S::TWO)] {
            let function = EllipticResidual { fraction };
            let result = design_root(
                &function,
                scioxide_core::Interval::new(S::ZERO, S::ONE).unwrap(),
                &mut None,
            )
            .unwrap();
            assert!(function.eval(result.point()).unwrap().abs() <= tolerance);
            assert!(
                result
                    .counts()
                    .provider(scioxide_core::EvaluationRole::Function)
                    < (S::MANTISSA_DIGITS + 8) as usize
            );
        }
    }
    #[test]
    fn normalization_uses_fewer_evaluations_than_fixed_bisection() {
        normalization_counts::<f32>();
        normalization_counts::<f64>();
    }
}
