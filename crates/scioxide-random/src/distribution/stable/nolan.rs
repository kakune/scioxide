//! Positive Nolan integrals in endpoint-distance coordinates.
use super::IntegralKind;
use crate::distribution::quadrature;
use crate::distribution::quadrature::positive_integral;
pub(super) use crate::law::charge;
use crate::law::{log1mexp, logadd};
use core::f64::consts::{FRAC_PI_2, PI};
#[cfg(not(any(feature = "std", test)))]
#[allow(
    unused_imports,
    reason = "dependency feature unification can enable inherent f64 math"
)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

/// S1-to-S0 shift, keeping the residual lost when alpha approaches one.
pub(super) fn shift(alpha: f64, beta: f64) -> (f64, f64) {
    let d = alpha - 1.;
    if alpha == 1. || beta == 0. {
        return (0., 0.);
    }
    if d.abs() < 0.001 {
        let n = -beta * core::f64::consts::FRAC_2_PI;
        // Low part of 2/pi, rounded from 100-digit arithmetic.
        let low = (-beta).mul_add(core::f64::consts::FRAC_2_PI, -n) + beta * 3.935735335036497e-17;
        let hi = n / d;
        let z = FRAC_PI_2 * d;
        // cot(z) = 1/z - z/3 - z^3/45 - 2z^5/945 - z^7/4725.
        let correction = (-hi).mul_add(d, n) / d
            + low / d
            + beta * z * (1. / 3. + z * z * (1. / 45. + z * z * (2. / 945. + z * z / 4725.)));
        let q = hi + correction;
        (q, (hi - q) + correction)
    } else if d.abs() < 0.5 {
        (-beta / (FRAC_PI_2 * d).tan(), 0.)
    } else if alpha > 1.5 {
        (-beta * (FRAC_PI_2 * (2. - alpha)).tan(), 0.)
    } else {
        (beta * (FRAC_PI_2 * alpha).tan(), 0.)
    }
}

pub(super) struct Nolan {
    pub(super) alpha: f64,
    beta: f64,
    x: f64,
    pub(super) xi: f64,
    pub(super) zeta: f64,
    width: f64,
    angle_gap: f64,
    left_gap: f64,
    offset: f64,
    log_x: f64,
    log_r: f64,
    saddle: f64,
}

impl Nolan {
    pub(super) fn new(alpha: f64, beta: f64, x: f64, centered: Option<f64>) -> Self {
        let delta = alpha - 1.;
        let (q, shift_low) = shift(alpha, beta);
        let phi = q.atan();
        let xi = if alpha == 1. {
            FRAC_PI_2
        } else if alpha < 1e-8 {
            beta * FRAC_PI_2 + beta * (1. - beta * beta) * FRAC_PI_2.powi(3) * alpha * alpha / 3.
        } else {
            phi / alpha
        };
        // Retain the small angular interval when phi is nearly -pi/2.
        let tangent = if alpha < 0.5 {
            (FRAC_PI_2 * alpha).tan()
        } else {
            1. / (FRAC_PI_2 * (1. - alpha).abs()).tan()
        };
        let width = if alpha < 1. && beta < 0. {
            if alpha < 1e-8 {
                (1. + beta) * FRAC_PI_2
                    - beta * (1. - beta) * (1. + beta) * FRAC_PI_2.powi(3) * alpha * alpha / 3.
            } else {
                ((1. + beta) / (1. / tangent - beta * tangent)).atan() / alpha
            }
        } else if alpha == 1. || (alpha < 1. && beta == 1.) {
            PI
        } else if alpha > 1. && beta == -1. {
            PI / alpha
        } else if q < -1. {
            (FRAC_PI_2 * delta + (-1. / q).atan()) / alpha
        } else {
            FRAC_PI_2 + xi
        };
        let angle_gap = if alpha < 1. && beta == 1. {
            PI * (1. - alpha)
        } else if alpha > 1. && beta == -1. {
            0.
        } else if alpha > 1. && beta < 0. {
            ((1. + beta) / (1. / tangent - beta * tangent)).atan()
        } else if q > 1. {
            -FRAC_PI_2 * delta + (1. / q).atan()
        } else {
            FRAC_PI_2 * (2. - alpha) - phi
        };
        let mut result = Self {
            alpha,
            beta,
            x,
            xi,
            zeta: -q,
            width,
            angle_gap,
            left_gap: 0.,
            offset: centered.unwrap_or((x - q) - shift_low),
            log_x: x.ln(),
            log_r: q.hypot(1.).ln(),
            saddle: 0.,
        };
        result.left_gap = if alpha == 1. {
            0.
        } else {
            PI * result.origin_lower()
        };
        if result.light_tail() {
            result.saddle = result.saddle_log_g();
        }
        result
    }

    pub(super) fn origin_lower(&self) -> f64 {
        if self.alpha < 1. && self.beta == 1. {
            return 0.;
        }
        if self.alpha > 1. && self.beta == -1. {
            return (self.alpha - 1.) / self.alpha;
        }
        let q = -self.zeta;
        if self.alpha < 1. && self.beta > 0. {
            if self.alpha < 1e-8 {
                return (1. - self.beta) * 0.5
                    - self.beta
                        * (1. - self.beta)
                        * (1. + self.beta)
                        * FRAC_PI_2.powi(3)
                        * self.alpha
                        * self.alpha
                        / (3. * PI);
            }
            let t = q / self.beta;
            return ((1. - self.beta) / (1. / t + self.beta * t)).atan() / (PI * self.alpha);
        }
        if q > 1. {
            (FRAC_PI_2 * (self.alpha - 1.) + (1. / q).atan()) / (PI * self.alpha)
        } else {
            0.5 - self.xi / PI
        }
    }

    pub(super) fn origin_upper(&self) -> f64 {
        if self.alpha < 1. && self.beta == -1. {
            return 0.;
        }
        if self.alpha > 1. && self.beta == 1. {
            return (self.alpha - 1.) / self.alpha;
        }
        if self.alpha < 1. && self.beta < 0. {
            return self.width / PI;
        }
        if self.zeta > 1. {
            (FRAC_PI_2 * (self.alpha - 1.) + (1. / self.zeta).atan()) / (PI * self.alpha)
        } else {
            0.5 + self.xi / PI
        }
    }

    /// Integrated origin series with the absolute next-term bound from the
    /// characteristic-function expansion (without its trigonometric factor).
    pub(super) fn origin(
        &self,
        kind: IntegralKind,
        tol: f64,
        left: &mut usize,
    ) -> Result<Option<f64>, SciOxideError> {
        if self.alpha < 0.1
            || self.alpha == 1.
            || (self.alpha < 1. && self.beta.abs() == 1.)
            || self.x <= 0.
        {
            return Ok(None);
        }
        let scale = self.log_r / self.alpha;
        let lx = self.log_x;
        if lx - scale > -4.6 {
            return Ok(None);
        }
        let f0 = self.origin_lower();
        let s0 = self.origin_upper();
        let mut sum = 0.;
        for k in 1..=12 {
            charge(left)?;
            let kf = f64::from(k);
            let log_coefficient = scioxide_special::ln_gamma(kf / self.alpha)?
                - (PI * self.alpha).ln()
                - kf * scale
                - scioxide_special::ln_gamma(kf + 1.)?;
            let log_power = if matches!(kind, IntegralKind::Density) {
                (kf - 1.) * lx + kf.ln()
            } else {
                kf * lx
            };
            sum += (log_coefficient + log_power).exp() * scioxide_core::sin_pi(kf * f0)?;
            let n = kf + 1.;
            let next = scioxide_special::ln_gamma(n / self.alpha)?
                - (PI * self.alpha).ln()
                - n * scale
                - scioxide_special::ln_gamma(n + 1.)?
                + if matches!(kind, IntegralKind::Density) {
                    kf * lx + n.ln()
                } else {
                    n * lx
                };
            let value = match kind {
                IntegralKind::Density => sum,
                IntegralKind::Lower => f0 + sum,
                IntegralKind::Upper => s0 - sum,
            };
            let small = if matches!(kind, IntegralKind::Density) {
                value
            } else {
                (f0 + sum).min(s0 - sum)
            };
            if small > 0. && next < (tol * small / 16.).ln() {
                return Ok(Some(value.ln()));
            }
        }
        Ok(None)
    }

    fn light_tail(&self) -> bool {
        (self.alpha <= 1. && self.beta == 1.) || (self.alpha > 1. && self.beta == -1.)
    }

    pub(super) fn log_tail_overflows(&self, max: f64) -> bool {
        self.light_tail() && self.x != 0. && self.saddle > max.ln()
    }

    fn saddle_log_g(&self) -> f64 {
        if self.alpha == 1. {
            return (2. / PI).ln() - 1. - self.x * FRAC_PI_2;
        }
        let d = self.alpha - 1.;
        let q = -self.zeta;
        if d.abs() < 0.05 {
            let r = self.offset / q;
            if r > -1. {
                return (d.abs() * q).ln() + self.alpha / d * (r.ln_1p() - d.ln_1p())
                    - (1. / q / q).ln_1p() / (2. * d);
            }
        }
        d.abs().ln() + (self.alpha * (self.log_x - self.alpha.ln()) - q.hypot(1.).ln()) / d
    }

    fn saddle_delta(&self, u: f64) -> f64 {
        let a = self.alpha;
        let d = a - 1.;
        if u < 0.05 {
            let mut total = 0.;
            let mut power = u * u;
            for (k, c) in [1. / 6., 1. / 180., 1. / 2835., 1. / 37800., 1. / 467775.]
                .iter()
                .enumerate()
            {
                let n = 2 * (k + 1);
                let geometric = (0..n).fold(0., |sum, _| sum * a + 1.);
                total += c * (a * geometric + 1. - d.powi(n as i32)) * power;
                power *= u * u;
            }
            return total;
        }
        if a == 1. {
            return 1. - u * u.cos() / u.sin() - log_sinc(u);
        }
        let du = d * u;
        let ratio = (-2. * (du * 0.5).sin().powi(2) + u.cos() / u.sin() * du.sin() - d) / a;
        if ratio > -1. {
            -a / d * ratio.ln_1p() + log_sinc(du) - log_sinc(u)
        } else {
            a / d * (log_sinc(u) - log_sinc(a * u)) + log_sinc(du) - log_sinc(u)
        }
    }

    fn log_g(&self, distance: f64, right: bool) -> f64 {
        if self.light_tail() {
            let u = if right == (self.alpha > 1.) {
                distance
            } else {
                self.width - distance
            };
            return self.saddle + self.saddle_delta(u);
        }
        let (v, w) = if right {
            (self.width - distance, distance)
        } else {
            (distance, self.width - distance)
        };
        let ct = if w < FRAC_PI_2 {
            w.sin()
        } else {
            (self.left_gap + v).sin()
        };
        let theta = if right {
            FRAC_PI_2 - distance
        } else {
            -self.xi + distance
        };
        if self.alpha == 1. {
            let t = 2. * self.beta / PI;
            let h = if right {
                1. + self.beta - t * distance
            } else {
                1. - self.beta + t * distance
            };
            let tangent = if right {
                distance.cos() / distance.sin()
            } else {
                -distance.cos() / distance.sin()
            };
            return h.ln() + (h * tangent - self.x) / t - ct.ln();
        }
        let delta = self.alpha - 1.;
        let angle = self.alpha * v;
        let sa = if angle < FRAC_PI_2 {
            angle.sin()
        } else {
            (self.angle_gap + self.alpha * w).sin()
        };
        let last_angle = if right {
            self.angle_gap + delta * distance
        } else {
            self.left_gap - delta * distance
        };
        let last = if last_angle < FRAC_PI_2 {
            last_angle.sin()
        } else {
            (self.alpha * v + w).sin()
        };
        // In the continuous S0 coordinate the logarithmic ratio tends to one.
        // Form its small difference before division by alpha-1.
        let q = -self.zeta;
        if delta.abs() < 0.05 && q > 1. {
            let s = delta * theta;
            // cos(delta*theta)-q*sin(delta*theta) = hypot(1,q)*last.
            // The endpoint-distance form retains this small difference for
            // skewness immediately adjacent to +/-1.
            let a = q.hypot(1.) * last;
            let tangent = theta.sin() / ct;
            let d = tangent * a + s.sin() - 2. * q * (s * 0.5).sin().powi(2);
            let r = (self.offset - d) / (q + d);
            if r > -1. && a > 0. {
                return self.alpha / delta * r.ln_1p() + a.ln() - ct.ln();
            }
        }
        (self.alpha
            * (self.log_x + ct.ln()
                - if angle < 0.01 {
                    self.alpha.ln() + v.ln() + log_sinc(angle)
                } else {
                    sa.ln()
                })
            - self.log_r)
            / delta
            + last.ln()
            - ct.ln()
    }

    fn partition(
        &self,
        threshold: f64,
        right: bool,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        let mut lo = 0.;
        let mut hi = self.width * 0.5;
        let increasing = (self.alpha <= 1.) != right;
        charge(left)?;
        let at_half = self.log_g(hi, right);
        if (at_half < threshold) == increasing {
            return Ok(hi);
        }
        if self.light_tail() && right == (self.alpha > 1.) && threshold <= self.saddle {
            return Ok(0.);
        }
        for _ in 0..1076 {
            let mid = lo * 0.5 + hi * 0.5;
            if mid == lo || mid == hi {
                break;
            }
            charge(left)?;
            let y = self.log_g(mid, right);
            // Partition accuracy only locates the transition; quadrature
            // controls the error of the completed integral.
            if (y - threshold).abs() < 0.001 {
                return Ok(mid);
            }
            if (y < threshold) == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(lo * 0.5 + hi * 0.5)
    }

    pub(super) fn log_integral(
        &self,
        kind: IntegralKind,
        tol: f64,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        if self.light_tail() {
            let lg = self.saddle;
            if lg > 0. {
                return self.saddle_integral(kind, lg, tol, left);
            }
        }
        let mut cuts = [[0.; 8]; 2];
        for (side, points) in cuts.iter_mut().enumerate() {
            points[7] = self.width * 0.5;
            // Resolve both sides of the transition: exp(-g) is almost one
            // for small g, but its narrow deficit must not fall between all
            // nodes of a wide panel. Bound that deficit by the caller target.
            for (point, g) in points[1..7].iter_mut().zip([
                tol * 0.001,
                0.01_f64,
                1.,
                5.,
                10.,
                (-(tol * 0.001).ln()).max(40.),
            ]) {
                *point = self.partition(g.ln(), side == 1, left)?;
            }
            points.sort_by(f64::total_cmp);
        }
        // The eighth-power mapping makes the weighted endpoint integrand
        // smooth through seven derivatives, even as alpha tends to zero.
        let power = if self.alpha < 0.5 { 8. } else { 1. };
        let half = self.width * 0.5;
        if power != 1. {
            for points in &mut cuts {
                for point in points {
                    *point = (*point / half).powf(1. / power);
                }
            }
        }
        let integrand = |d: f64, right: bool| -> Result<f64, SciOxideError> {
            let (d, jacobian) = if power == 1. {
                (d, 0.)
            } else {
                (
                    half * d.powf(power),
                    (half * power).ln() + (power - 1.) * d.ln(),
                )
            };
            let lg = self.log_g(d, right);
            let value = match kind {
                IntegralKind::Density => {
                    if lg.is_infinite() {
                        f64::NEG_INFINITY
                    } else {
                        lg - lg.exp()
                    }
                }
                IntegralKind::Lower => -lg.exp(),
                IntegralKind::Upper => {
                    if lg < -36. {
                        lg
                    } else {
                        log1mexp(-lg.exp())?
                    }
                }
            };
            if value.is_nan() {
                Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "stable integrand lost its positive domain",
                ))
            } else {
                Ok(value + jacobian)
            }
        };
        // One error scale for the complete integral, including all partitions.
        let mut estimate = f64::NEG_INFINITY;
        for (side, points) in cuts.iter().enumerate() {
            for p in points.windows(2).filter(|p| p[0] < p[1]) {
                let (k, _) = quadrature::rule(p[0], p[1], |d| {
                    charge(left)?;
                    integrand(d, side == 1)
                })?;
                estimate = logadd(estimate, k + ((p[1] - p[0]) * 0.5).ln())?;
            }
        }
        let floor = estimate + (tol / 28.).ln();
        let mut sum = f64::NEG_INFINITY;
        for (side, points) in cuts.iter().enumerate() {
            for p in points.windows(2).filter(|p| p[0] < p[1]) {
                sum = logadd(
                    sum,
                    positive_integral(
                        p[0],
                        p[1],
                        tol,
                        left,
                        &|d| integrand(d, side == 1),
                        Some(floor),
                    )?,
                )?;
            }
        }
        Ok(sum)
    }

    fn saddle_integral(
        &self,
        kind: IntegralKind,
        lg: f64,
        tol: f64,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        charge(left)?;
        let g = lg.exp();
        if !g.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "stable log tail is outside the finite scalar range",
            ));
        }
        if matches!(kind, IntegralKind::Upper) {
            let small = self.saddle_integral(IntegralKind::Lower, lg, tol, left)?;
            return Ok(self.width.ln() + log1mexp(small - self.width.ln())?);
        }
        let log_scale = -0.5 * (self.alpha.ln() + lg);
        // The first Laplace correction is O(1/g); use the leading term only
        // when it is beneath the requested relative integral accuracy.
        if g > 32. / tol {
            return Ok(-g
                + log_scale
                + (PI / 2.).ln() * 0.5
                + if matches!(kind, IntegralKind::Density) {
                    lg
                } else {
                    0.
                });
        }
        let scale = log_scale.exp();
        let end = (self.width / scale).min(2. * (-2. * (tol * 0.001).ln()).sqrt());
        let integrand = |z: f64| {
            let delta = self.saddle_delta(z * scale);
            Ok(-g * delta.exp_m1()
                + if matches!(kind, IntegralKind::Density) {
                    lg + delta
                } else {
                    0.
                })
        };
        Ok(-g + log_scale + positive_integral(0., end, tol, left, &integrand, None)?)
    }
}

fn log_sinc(x: f64) -> f64 {
    if x.abs() < 0.01 {
        let u = x * x;
        -u * (1. / 6. + u * (1. / 180. + u * (1. / 2835. + u / 37800.)))
    } else {
        (x.sin() / x).ln()
    }
}
