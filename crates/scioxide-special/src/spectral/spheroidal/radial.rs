//! Spherical product expansions and polynomial-ODE continuation.
use super::*;
use crate::bessel_airy::complex_bessel::{Kind, spherical_pair};
use crate::support::Work;
use core::ops::Neg;

fn from_bessel(
    v: crate::bessel_airy::complex_bessel::ExponentialEstimate,
) -> Result<(Z, Scaled<f64>), SciOxideError> {
    let exponent = (v.scale / core::f64::consts::LN_2).floor();
    if !(i64::MIN as f64..i64::MAX as f64).contains(&exponent) {
        return Err(instability("spherical logarithmic exponent overflow"));
    }
    let factor =
        scalar((v.scale - exponent * core::f64::consts::LN_2).exp())?.shifted(exponent as i64)?;
    Ok((
        zscale(Z::from_value(v.value)?, factor)?,
        factor.abs().mul(scalar(v.error)?)?,
    ))
}
fn quadratic(x: C, s: f64) -> Result<Z, SciOxideError> {
    let z = Z::from_value(x)?;
    let result = z
        .mul(z)?
        .add(Z::from_value(scioxide_core::Complex::new(s, 0.0))?)?;
    if x.re == 0.0 || x.im == 0.0 {
        Ok(Z::from_components(
            result.real(),
            scalar((x.re * x.im) * 2.0)?,
        ))
    } else {
        Ok(result)
    }
}
fn root(x: C, s: f64) -> Result<Z, SciOxideError> {
    if s < 0.0 {
        let a = (-s).sqrt();
        Z::from_value(C::new(x.re - a, x.im))?
            .sqrt()?
            .mul(Z::from_value(C::new(x.re + a, x.im))?.sqrt()?)
    } else {
        quadratic(x, s)?.sqrt()
    }
}
impl<S: RealScalar> SpheroidalPlan<S> {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn accumulate_radial(
        &self,
        basis: &mut Basis,
        bessel: Kind,
        argument: C,
        center_derivative: bool,
        wp: Z,
        rp: Z,
        bessel_work: &mut Work<'_, '_, '_>,
    ) -> Result<
        (
            Z,
            Z,
            Scaled<f64>,
            Scaled<f64>,
            [Scaled<f64>; 4],
            [Scaled<f64>; 4],
        ),
        SciOxideError,
    > {
        let mut value = Z::ZERO;
        let mut derivative = Z::ZERO;
        let mut error = Scaled::ZERO;
        let mut derivative_error = Scaled::ZERO;
        let mut tails = [Scaled::ZERO; 4];
        let mut dtails = tails;
        for k in 0..self.coefficient_count() {
            let degree = u64::from(self.m) + u64::from((self.n - self.m) % 2) + 2 * k as u64;
            // Basis advancement uses the same parent via a short work callback
            // below; no public special-function call resets the allowance.
            while (basis.l as u64) < degree {
                basis.advance_with(&mut || bessel_work.try_step())?;
            }
            bessel_work.try_step()?;
            let order = u32::try_from(degree)
                .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("degree")))?;
            let (j, jp) = spherical_pair(bessel, order, argument, bessel_work)?;
            let (j, je) = from_bessel(j)?;
            let (jp, jpe) = from_bessel(jp)?;
            let q = if center_derivative {
                basis.derivative
            } else {
                basis.current
            };
            let qprime = if center_derivative {
                Z::ZERO
            } else {
                basis.derivative.mul(wp)?
            };
            let v = q.mul(j)?;
            let d = qprime.mul(j)?.add(
                q.mul(jp)?
                    .mul(zscale(rp, scalar(self.c.try_to_f64_exact()?)?)?)?,
            )?;
            let coefficient = if (k as i64 - i64::from((self.n - self.m) / 2)).rem_euclid(2) == 0 {
                self.spectrum.coefficients[k]
            } else {
                self.spectrum.coefficients[k].neg()
            };
            value = value.add(zscale(v, coefficient)?)?;
            derivative = derivative.add(zscale(d, coefficient)?)?;
            let rounding = scalar(64.0 * f64::EPSILON * (degree as f64 + 1.0).powi(2))?;
            let coefficient_error =
                self.spectrum.errors[k].add(coefficient.abs().mul(rounding)?)?;
            let ve = norm(q)?.mul(je)?;
            let de = norm(qprime)?.mul(je)?.add(
                norm(q)?
                    .mul(jpe)?
                    .mul(norm(zscale(rp, scalar(self.c.try_to_f64_exact()?)?)?)?)?,
            )?;
            error = error
                .add(coefficient_error.mul(norm(v)?)?)?
                .add(coefficient.abs().mul(ve)?)?;
            derivative_error = derivative_error
                .add(coefficient_error.mul(norm(d)?)?)?
                .add(coefficient.abs().mul(de)?)?;
            tails[k % 4] = coefficient.abs().mul(norm(v)?)?;
            dtails[k % 4] = coefficient.abs().mul(norm(d)?)?;
        }
        Ok((value, derivative, error, derivative_error, tails, dtails))
    }

    fn radial_seed(
        &self,
        x: C,
        bessel: Kind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        let second = !matches!(bessel, Kind::J);
        let s = if self.kind == SpheroidalKind::Prolate {
            -1.0
        } else {
            1.0
        };
        // The product coordinate is free. Change it near a vanishing product
        // radius so the regular first kind does not need singular ODE transport.
        let selected_radius = root(x, s * (1.0 - self.eta * self.eta))?;
        let (eta, center_derivative, den, den_error) =
            if !second && norm(selected_radius)?.abs_le(scalar(0.2)?) {
                let eta = if x.norm() > 0.5 { 1.0 } else { 0.0 };
                let derivative = eta == 0.0 && (self.n - self.m) % 2 == 1;
                let (den, error) = self.normalizer(eta, derivative, context)?;
                (eta, derivative, den, error)
            } else {
                (
                    self.eta,
                    self.center_derivative,
                    self.denominator,
                    self.denominator_error,
                )
            };
        let a = s * (1.0 - eta * eta);
        let r = root(x, a)?;
        if r.is_zero() {
            return Err(instability("product radius requires a limiting expansion"));
        }
        let radius = r.round()?;
        let argument = radius * self.c.try_to_f64_exact()?;
        if !argument.is_finite() {
            return Err(instability(
                "spherical Bessel argument exceeds working range",
            ));
        }
        let w = zscale(Z::from_value(x)?.div(r)?, scalar(eta)?)?.round()?;
        let rp = Z::from_value(x)?.div(r)?;
        let wp =
            Z::from_value(scioxide_core::Complex::new(eta * a, 0.0))?.div(r.mul(r)?.mul(r)?)?;
        let geometric = root(x, s)?;
        let pref = power(geometric, self.m, context)?.div(power(r, self.m, context)?)?;
        let pref_derivative = if self.m == 0 {
            Some(Z::ZERO)
        } else if geometric.is_zero() {
            if self.m == 1 {
                None
            } else if self.m == 2 {
                Some(Z::from_value(x * 2.0)?.div(r.mul(r)?)?)
            } else {
                Some(Z::ZERO)
            }
        } else {
            Some(zscale(
                pref.mul(
                    Z::from_value(x)?
                        .div(quadratic(x, s)?)?
                        .sub(Z::from_value(x)?.div(r.mul(r)?)?)?,
                )?,
                scalar(f64::from(self.m))?,
            )?)
        };
        let extra = if center_derivative {
            Z::from_value(x)?.div(r)?
        } else {
            Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?
        };
        let extra_derivative = if center_derivative {
            Z::from_value(scioxide_core::Complex::new(s, 0.0))?.div(r.mul(r)?.mul(r)?)?
        } else {
            Z::ZERO
        };
        let base = pref.mul(extra)?;
        let base_derivative = pref_derivative
            .map(|d| d.mul(extra)?.add(pref.mul(extra_derivative)?))
            .transpose()?;
        let mut basis = Basis::new(self.m, w, self.diagonal, &self.steps)?;
        let mut bessel_work =
            Work::with_context(self.options.iteration_budget.get(), Some(&mut *context));
        let result = self.accumulate_radial(
            &mut basis,
            bessel,
            argument,
            center_derivative,
            wp,
            rp,
            &mut bessel_work,
        );
        bessel_work.check()?;
        let (mut value, derivative, mut error, mut derivative_error, tails, dtails) = result?;
        let rnorm = radius.norm();
        if second && rnorm <= 1.0 {
            return Err(instability(
                "spherical Y expansion requires an exterior product radius",
            ));
        }
        let growth = if second {
            1.0 / (rnorm * rnorm)
        } else {
            self.spectrum.tail_ratio
        };
        let ratio = growth.max(self.spectrum.tail_ratio);
        if !ratio.is_finite() || ratio >= 1.0 {
            return Err(instability("radial tail is unresolved"));
        }
        let largest = |v: &[Scaled<f64>]| {
            v.iter()
                .copied()
                .fold(Scaled::ZERO, |a, b| if a.abs_le(b) { b } else { a })
        };
        error = error.add(largest(&tails).mul(scalar(ratio / (1.0 - ratio))?)?)?;
        derivative_error =
            derivative_error.add(largest(&dtails).mul(scalar(ratio / (1.0 - ratio))?)?)?;
        let den_error = den_error.div(den.abs())?.round()?;
        if den_error >= 0.5 {
            return Err(instability("spheroidal radial normalization is unresolved"));
        }
        let before = value;
        value = zscale(base.mul(value)?, scalar(1.0)?.div(den)?)?;
        let derivative = base_derivative
            .map(|d| {
                zscale(
                    d.mul(before)?.add(base.mul(derivative)?)?,
                    scalar(1.0)?.div(den)?,
                )
            })
            .transpose()?;
        let de = if let Some(d) = base_derivative {
            error
                .mul(norm(d)?)?
                .add(derivative_error.mul(norm(base)?)?)?
                .div(den.abs())?
        } else {
            Scaled::ZERO
        };
        error = error
            .mul(norm(base)?)?
            .div(den.abs())?
            .add(norm(value)?.mul(scalar(den_error / (1.0 - den_error))?)?)?;
        let derivative_error = de.add(
            derivative
                .map(norm)
                .transpose()?
                .unwrap_or(Scaled::ZERO)
                .mul(scalar(den_error / (1.0 - den_error))?)?,
        )?;
        Ok(Evaluation {
            value,
            derivative,
            error,
            derivative_error,
        })
    }
}

struct Transfer {
    matrix: [[Z; 2]; 2],
    error: Scaled<f64>,
    next_scale: Scaled<f64>,
}
impl<S: RealScalar> SpheroidalPlan<S> {
    fn state_scale(&self, x: C, s: f64, signed_q: f64) -> Result<Scaled<f64>, SciOxideError> {
        let p = quadratic(x, s)?;
        let c = scalar(self.c.try_to_f64_exact()?)?;
        let q = zscale(
            Z::from_value(x)?.mul(Z::from_value(x)?)?,
            c.mul(c)?.mul(scalar(signed_q)?)?,
        )?
        .sub(Z::from_value(scioxide_core::Complex::new(
            self.spectrum.characteristic,
            0.0,
        ))?)?
        .add(
            Z::from_value(scioxide_core::Complex::new(
                s * f64::from(self.m).powi(2),
                0.0,
            ))?
            .div(p)?,
        )?
        .div(p)?;
        let damping = Z::from_value(x * 2.0)?.div(p)?;
        let floor = 1.0 / (1.0 + x.re.abs().max(x.im.abs()));
        norm(q)?.sqrt()?.add(norm(damping)?)?.add(scalar(floor)?)
    }
    #[allow(clippy::too_many_arguments)]
    fn transfer(
        &self,
        x: C,
        h: C,
        s: f64,
        signed_q: f64,
        old_scale: Scaled<f64>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Transfer, SciOxideError> {
        let xz = Z::from_value(x)?;
        let hz = Z::from_value(h)?;
        let p = [
            quadratic(x, s)?,
            zscale(xz, scalar(2.0)?)?,
            Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
        ];
        let c = scalar(self.c.try_to_f64_exact()?)?;
        let q = c.mul(c)?.mul(scalar(signed_q)?)?;
        let potential = [
            zscale(xz.mul(xz)?, q)?.sub(Z::from_value(scioxide_core::Complex::new(
                self.spectrum.characteristic,
                0.0,
            ))?)?,
            zscale(xz, q.mul(scalar(2.0)?)?)?,
            Z::from_components(q, Scaled::ZERO),
        ];
        let mut a = [Z::ZERO; 5];
        let mut b = a;
        let mut d = a;
        for i in 0..3 {
            for j in 0..3 {
                a[i + j] = a[i + j].add(p[i].mul(p[j])?)?;
                d[i + j] = d[i + j].add(potential[i].mul(p[j])?)?;
            }
            b[i] = b[i].add(zscale(xz.mul(p[i])?, scalar(2.0)?)?)?;
            b[i + 1] = b[i + 1].add(zscale(p[i], scalar(2.0)?)?)?;
        }
        d[0] = d[0].add(Z::from_value(scioxide_core::Complex::new(
            s * f64::from(self.m).powi(2),
            0.0,
        ))?)?;
        let denominator = a[0];
        if denominator.is_zero() {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::Pole,
                "continuation reached a spheroidal singularity",
            ));
        }
        let mut hp = [Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?; 7];
        for k in 1..7 {
            hp[k] = hp[k - 1].mul(hz)?;
        }
        for k in 0..5 {
            a[k] = a[k].mul(hp[k])?.div(denominator)?;
            b[k] = b[k].mul(hp[k + 1])?.div(denominator)?;
            d[k] = d[k].mul(hp[k + 2])?.div(denominator)?;
        }
        let next_scale = self.state_scale(x + h, s, signed_q)?;
        let derivative_divisor = zscale(hz, next_scale)?;
        let mut coefficients = [[Z::ZERO; 6]; 2];
        coefficients[0][0] = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?;
        coefficients[1][1] = zscale(hz, old_scale)?;
        let mut values = [coefficients[0][0], coefficients[1][1]];
        let mut derivatives = [
            Z::ZERO,
            Z::from_components(old_scale.div(next_scale)?, Scaled::ZERO),
        ];
        let mut magnitude = norm(values[0])?
            .add(norm(values[1])?)?
            .add(norm(derivatives[1])?)?;
        let mut recent = [Scaled::ZERO; 4];
        let mut previous_tail = Scaled::from_value(f64::MAX)?;
        for j in 2_usize.. {
            step(context)?;
            let mut tail = Scaled::ZERO;
            for column in 0..2 {
                let mut value = Z::ZERO;
                for (k, &ak) in a.iter().enumerate().skip(1) {
                    if j >= k + 2 {
                        let index = j - k;
                        value = value.add(zscale(
                            ak.mul(coefficients[column][index % 6])?,
                            scalar((index as f64) * (index as f64 - 1.0))?,
                        )?)?;
                    }
                }
                for (k, &bk) in b.iter().enumerate().take(4) {
                    if j >= k + 2 {
                        let index = j - k - 1;
                        value = value.add(zscale(
                            bk.mul(coefficients[column][index % 6])?,
                            scalar(index as f64)?,
                        )?)?;
                    }
                }
                for (k, &dk) in d.iter().enumerate() {
                    if j >= k + 2 {
                        let index = j - k - 2;
                        value = value.add(dk.mul(coefficients[column][index % 6])?)?;
                    }
                }
                value = zscale(value, scalar(-1.0 / (j as f64 * (j as f64 - 1.0)))?)?;
                coefficients[column][j % 6] = value;
                let derivative = zscale(value, scalar(j as f64)?)?.div(derivative_divisor)?;
                values[column] = values[column].add(value)?;
                derivatives[column] = derivatives[column].add(derivative)?;
                tail = tail.add(norm(value)?)?.add(norm(derivative)?)?;
            }
            magnitude = magnitude.add(tail)?;
            recent[j % 4] = tail;
            if j >= 12 && j.is_multiple_of(4) {
                let maximum = recent
                    .iter()
                    .copied()
                    .fold(Scaled::ZERO, |a, b| if a.abs_le(b) { b } else { a });
                let size = norm(values[0])?
                    .add(norm(values[1])?)?
                    .add(norm(derivatives[0])?)?
                    .add(norm(derivatives[1])?)?;
                if maximum.abs_le(size.mul(scalar(f64::EPSILON * 0.125)?)?)
                    && maximum.abs_le(previous_tail.mul(scalar(0.5)?)?)
                {
                    let error = maximum
                        .mul(scalar(8.0)?)?
                        .add(magnitude.mul(scalar(16.0 * f64::EPSILON * (j + 1) as f64)?)?)?;
                    return Ok(Transfer {
                        matrix: [[values[0], values[1]], [derivatives[0], derivatives[1]]],
                        error,
                        next_scale,
                    });
                }
                previous_tail = maximum;
            }
        }
        Err(instability("Taylor coefficient index exhausted"))
    }
    #[allow(clippy::too_many_arguments)]
    fn continue_segment(
        &self,
        mut point: C,
        target: C,
        mut value: Evaluation,
        s: f64,
        signed_q: f64,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        let poles = if s < 0.0 {
            [real(-1.0), real(1.0)]
        } else {
            [C::new(0.0, -1.0), C::new(0.0, 1.0)]
        };
        let mut scale = self.state_scale(point, s, signed_q)?;
        // Carry one Euclidean state-error bound through all transfer steps.
        let mut state_error = value
            .error
            .mul(value.error)?
            .add(
                value
                    .derivative_error
                    .div(scale)?
                    .mul(value.derivative_error.div(scale)?)?,
            )?
            .sqrt()?;
        while point != target {
            step(context)?;
            let remaining = target - point;
            let distance = poles
                .iter()
                .map(|&p| (point - p).norm())
                .fold(f64::INFINITY, f64::min);
            let limit = (scalar(0.5)?.div(scale)?.round()?).min(0.2 * distance);
            let mut h = remaining * (limit / remaining.norm()).min(1.0);
            if point + h == point || h == real(0.0) {
                if remaining.norm() < 0.8 * distance {
                    h = remaining;
                } else {
                    return Err(SciOxideError::convergence_failure(
                        scioxide_core::ConvergenceReason::StepSizeLimit,
                        scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                        "continuation cannot make representable progress",
                    ));
                }
            }
            let next = if remaining.norm() <= limit {
                target
            } else {
                point + h
            };
            h = next - point;
            let transfer = self.transfer(point, h, s, signed_q, scale, context)?;
            let derivative = value
                .derivative
                .ok_or_else(|| instability("continuation seed derivative unavailable"))?;
            let scaled_derivative = zscale(derivative, scalar(1.0)?.div(scale)?)?;
            let new_value = transfer.matrix[0][0]
                .mul(value.value)?
                .add(transfer.matrix[0][1].mul(scaled_derivative)?)?;
            let new_derivative = zscale(
                transfer.matrix[1][0]
                    .mul(value.value)?
                    .add(transfer.matrix[1][1].mul(scaled_derivative)?)?,
                transfer.next_scale,
            )?;
            let a = transfer.matrix[0][0].round()?;
            let b = transfer.matrix[0][1].round()?;
            let c = transfer.matrix[1][0].round()?;
            let d = transfer.matrix[1][1].round()?;
            let aa = a.norm_sqr() + c.norm_sqr();
            let bb = b.norm_sqr() + d.norm_sqr();
            let cross = a.conj() * b + c.conj() * d;
            let amplification =
                (0.5 * (aa + bb + ((aa - bb).powi(2) + 4.0 * cross.norm_sqr()).sqrt())).sqrt();
            if !amplification.is_finite() {
                return Err(instability("continuation transfer is too ill-conditioned"));
            }
            let input_size = norm(value.value)?.add(norm(scaled_derivative)?)?;
            let error = state_error
                .mul(scalar(amplification)?)?
                .add(transfer.error.mul(input_size)?)?;
            value = Evaluation {
                value: new_value,
                derivative: Some(new_derivative),
                error,
                derivative_error: error.mul(transfer.next_scale)?,
            };
            state_error = error;
            point = next;
            scale = transfer.next_scale;
        }
        Ok(value)
    }
    fn continue_radial(
        &self,
        target: C,
        bessel: Kind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        let mut point = real(3.0);
        let mut value = self.radial_seed(point, bessel, context)?;
        let s = if self.kind == SpheroidalKind::Prolate {
            -1.0
        } else {
            1.0
        };
        if self.kind == SpheroidalKind::Prolate && target.re <= 1.0 {
            let height = target.im.abs().max(0.5).copysign(target.im);
            for waypoint in [C::new(point.re, height), C::new(target.re, height), target] {
                value = self.continue_segment(point, waypoint, value, s, 1.0, context)?;
                point = waypoint;
            }
        } else {
            let real_part = if target.re == 0.0 && target.im.abs() >= 1.0 {
                0.25_f64.copysign(target.re)
            } else {
                target.re
            };
            for waypoint in [real(real_part), C::new(real_part, target.im), target] {
                value = self.continue_segment(point, waypoint, value, s, 1.0, context)?;
                point = waypoint;
            }
        }
        Ok(value)
    }
    fn radial_evaluation(
        &self,
        x: C,
        kind: RadialKind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        if self.c == S::ZERO {
            step(context)?;
            if kind != RadialKind::First {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("c"),
                    scioxide_core::DomainReason::Pole,
                    "higher radial kinds diverge at c=0",
                ));
            }
            return Ok(Evaluation {
                value: Z::from_value(scioxide_core::Complex::new(
                    if self.n == 0 { 1.0 } else { 0.0 },
                    0.0,
                ))?,
                derivative: Some(Z::ZERO),
                error: Scaled::ZERO,
                derivative_error: Scaled::ZERO,
            });
        }
        let endpoint = match self.kind {
            SpheroidalKind::Prolate => x.im == 0.0 && x.re.abs() == 1.0,
            SpheroidalKind::Oblate => x.re == 0.0 && x.im.abs() == 1.0,
        };
        if endpoint && kind != RadialKind::First {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::Pole,
                "higher radial kinds diverge at a focal endpoint",
            ));
        }
        let bessel = kind.bessel_kind();
        let s = if self.kind == SpheroidalKind::Prolate {
            -1.0
        } else {
            1.0
        };
        let radius = norm(root(x, s * (1.0 - self.eta * self.eta))?)?.round()?;
        if kind != RadialKind::First
            && (radius < 1.5 || self.kind == SpheroidalKind::Oblate && x.re < 0.0)
        {
            self.continue_radial(x, bessel, context)
        } else {
            self.radial_seed(x, bessel, context)
        }
    }
}
impl<S: RealScalar> SpheroidalPlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Evaluates the radial spheroidal function `R_mn^(j)(x)` and its derivative with respect to
    /// `x`.
    ///
    /// The kind `j` is selected by `kind`. The physical range is `x ≥ 1` for prolate and `x ≥ 0`
    /// for oblate functions; other finite real or complex `x` give the analytic continuation,
    /// with principal square roots for the focal factors. The functions are expansions in
    /// products of spherical Bessel functions of the selected kind (DLMF §30.11), normalized at
    /// [`Self::radial_normalization_point`]; the angular normalization does not affect them.
    /// Near the focal points the second, third and fourth kinds are continued by Taylor-series
    /// integration of the radial equation. The third and fourth kinds are first ± i · second
    /// kind. At `c = 0` the first kind is `1` for `n = 0` and `0` otherwise.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` has a NaN or infinite component.
    /// - `DomainError` (`Pole`) for the second, third and fourth kinds at `c = 0`, and at the
    ///   focal points `x = ±1` (prolate) or `x = ±i` (oblate).
    /// - `NumericalInstability` (`LossOfSignificance`) if the expansion or the continuation
    ///   cannot resolve `x`.
    /// - `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the
    ///   budget.
    pub fn radial<T: NumericScalar<Real = S>>(
        &self,
        x: T,
        kind: RadialKind,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        self.radial_scoped(x, kind, ExecutionContext::sequential_native())
    }
    /// Evaluates [`Self::radial`] inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::radial`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn radial_scoped<'scope, 'cap, T: NumericScalar<Real = S>>(
        &self,
        x: T,
        kind: RadialKind,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        let x = spectral::coordinate(x, "radial coordinate must be finite")?;
        scoped(scope, self.options.iteration_budget, |context| {
            self.radial_run(x, kind, context)
        })
    }

    /// Evaluates the radial function at a validated coordinate inside an open epoch.
    fn radial_run(
        &self,
        x: C,
        kind: RadialKind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        let value = self.radial_evaluation(x, kind, context);
        report(value, algorithm_component!("spheroidal_radial"), context)
    }
}

/// Computes the radial spheroidal wave function `R_mn^(j)(x)`.
///
/// `R_mn^(j)` solves the spheroidal wave equation of [`SpheroidalPlan`] in the radial coordinate
/// `x`, with order `m ≥ 0`, degree `n ≥ m`, real size parameter `c`, prolate or oblate by
/// `kind`; `radial_kind` selects the kind `j`. The physical range is `x ≥ 1` for prolate and
/// `x ≥ 0` for oblate functions; other finite real or complex `x` give the analytic
/// continuation, with principal square roots for the focal factors. The value is the expansion
/// in products of spherical Bessel functions of DLMF §30.11, normalized as described for
/// [`SpheroidalPlan::radial`] and independent of the angular normalization; the third and
/// fourth kinds are first ± i · second kind. At `c = 0` the first kind is `1` for `n = 0` and
/// `0` otherwise. Each call prepares one [`SpheroidalPlan`] with default [`SpectralOptions`]
/// and evaluates its expansion once; preparation and evaluation share one budget of 2,097,152
/// work units. Prepare a [`SpheroidalPlan`] to evaluate many points.
///
/// # Errors
///
/// - `NonFinite` if `c` is NaN or infinite, or if `x` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `n` if `n < m`, or if the mode lies outside
///   the default truncation bounds: `n − m ≥ 496`, or `n − m ≥ 512` at `c = 0`.
/// - `DomainError` (`Pole`) for the second, third and fourth kinds: with argument `c` at
///   `c = 0`, and with argument `x` at the focal points `x = ±1` (prolate) or `x = ±i`
///   (oblate).
/// - `NumericalInstability` (`LossOfSignificance`) if `c²` or the matrix entries leave the
///   `f64` range, if no normalization point of the radial expansion is resolved, or if the
///   expansion or its continuation cannot resolve `x`; (`Overflow`) if the value exceeds the
///   range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`, and the
/// error estimate is derived from the coefficient and truncation errors. See
/// [`SpheroidalPlan`] and the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::{RadialKind, SpheroidalKind, spheroidal_radial};
///
/// // R₀₀⁽¹⁾(x) = 1 at c = 0
/// let kind = SpheroidalKind::Prolate;
/// let r = spheroidal_radial(kind, 0, 0, 0.0, 1.5_f64, RadialKind::First)?;
/// assert!((r.re - 1.0).abs() < 1e-12 && r.im.abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §30.11. <https://dlmf.nist.gov/30>
/// - C. Flammer, *Spheroidal Wave Functions*, Stanford University Press, 1957.
pub fn spheroidal_radial<T: NumericScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: T::Real,
    x: T,
    radial_kind: RadialKind,
) -> Result<Complex<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let x = spectral::coordinate(x, "radial coordinate must be finite")?;
    prepared_call(
        kind,
        m,
        n,
        c,
        SpectralOptions::default(),
        |plan, context| plan.radial_run(x, radial_kind, context),
    )?
    .try_value()
}
