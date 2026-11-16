//! Prepared integer-order Mathieu functions. DLMF 28.4 and 28.24.

use crate::result::Evaluation;
use crate::spectral::{self, Spectrum, instability, invalid, step};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::{C, real};
use crate::util::scoped;
use crate::{
    RadialKind, SpecialFunctionResult, SpectralDiagnostics, SpectralNormalization, SpectralOptions,
};
use core::ops::Neg;
use scioxide_core::AlgorithmComponent;
use scioxide_core::algorithm_component;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::{
    Argument, Complex, ExecutionContext, NumericScalar, RealScalar, ScaledValue, SciOxideError,
    SolveContext, SolveScope,
};
use scioxide_core::{Scaled, ScaledAccumulator};
/// Selects the even or odd Mathieu functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathieuParity {
    /// The even functions `ce_n` and the radial functions `Mc_n^(j)`, for `n ≥ 0`.
    Even,
    /// The odd functions `se_n` and the radial functions `Ms_n^(j)`, for `n ≥ 1`.
    Odd,
}
/// Holds the Fourier coefficients and the characteristic value of one Mathieu function.
///
/// Mathieu's equation (DLMF 28.2.1) is
///
/// ```text
/// w″ + (a − 2q cos 2z) w = 0,
/// ```
///
/// with real parameter `q`. For the characteristic values `a = a_n(q)` and `a = b_n(q)` it has
/// the periodic solutions `ce_n(z,q)` (even) and `se_n(z,q)` (odd), which reduce to `cos nz` and
/// `sin nz` at `q = 0`, with `ce₀(z,0) = 1/√2`. The radial (modified) Mathieu functions
/// `Mc_n^(j)` and `Ms_n^(j)` solve `w″ − (a − 2q cosh 2z) w = 0` (DLMF §28.20).
///
/// [`Self::new`] computes the coefficients once; [`Self::angular`] and [`Self::radial`] then
/// evaluate the functions at many points. The plan stores the characteristic value, the `N`
/// Fourier coefficients of the requested function with their error bounds, and an estimate of
/// the omitted tail, where `N ≤ 256` is the coefficient count chosen by [`SpectralOptions`].
/// Preparation solves the eigenproblem of each doubled truncation by Sturm bisection and
/// inverse iteration for the one requested eigenpair, in `O(N)` operations and memory; an
/// angular evaluation then takes `O(N)` operations and a radial one `O(N)` Bessel-function
/// pairs. Preparation and evaluation compute in `f64` for both scalar types. The one-shot
/// functions [`mathieu_a`], [`mathieu_ce`] and [`mathieu_mc`] and their odd counterparts prepare
/// one plan per call. See [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The tolerance of [`SpectralOptions`] steers the truncation,
/// [`Self::diagnostics`] reports the eigenvalue residual and the omitted tail, and each
/// evaluation returns an error estimate that includes the truncation error. The estimates are
/// not guaranteed bounds. For complex `z` the omitted tail grows like `e^(ν|Im z|)`, where `ν`
/// is the first omitted Fourier frequency. See the [crate accuracy table](crate#accuracy).
///
/// # References
///
/// NIST DLMF, Chapter 28. <https://dlmf.nist.gov/28>
#[derive(Debug, Clone)]
pub struct MathieuPlan<S: RealScalar> {
    order: u32,
    parity: MathieuParity,
    q: S,
    pub(crate) options: SpectralOptions<S>,
    normalization: SpectralNormalization,
    spectrum: Spectrum,
}
struct Sum {
    re: ScaledAccumulator<f64>,
    im: ScaledAccumulator<f64>,
    error: Scaled<f64>,
}
impl Sum {
    fn new() -> Self {
        Self {
            re: ScaledAccumulator::ZERO,
            im: ScaledAccumulator::ZERO,
            error: Scaled::ZERO,
        }
    }
    fn add(
        &mut self,
        z: C,
        scale: f64,
        coefficient: Scaled<f64>,
        error: f64,
        coefficient_error: Scaled<f64>,
    ) -> Result<(), SciOxideError> {
        let exponential = Scaled::from_log(1.0, scale)?;
        let factor = coefficient.mul(exponential)?;
        self.re.add_product(factor, Scaled::from_value(z.re)?)?;
        self.im.add_product(factor, Scaled::from_value(z.im)?)?;
        self.error = self
            .error
            .add(factor.abs().mul(Scaled::from_value(error)?)?)?
            .add(
                coefficient_error
                    .mul(exponential)?
                    .mul(scioxide_core::ScaledValue::from_value(z)?.magnitude()?)?,
            )?;
        Ok(())
    }
    fn add_real(
        &mut self,
        value: f64,
        coefficient: Scaled<f64>,
        error: f64,
        coefficient_error: Scaled<f64>,
    ) -> Result<(), SciOxideError> {
        let value = Scaled::from_value(value)?;
        self.re.add_product(coefficient, value)?;
        self.error = self
            .error
            .add(coefficient.abs().mul(Scaled::from_value(error)?)?)?
            .add(coefficient_error.mul(value.abs())?)?;
        Ok(())
    }
}
fn trig(frequency: u32, z: C, phase: C, sine: bool) -> Result<(C, C, f64), SciOxideError> {
    let imaginary = z.im * f64::from(frequency);
    if !imaginary.is_finite() {
        return Err(instability(
            "complex Mathieu coordinate exceeds logarithmic range",
        ));
    }
    let scale = imaginary.abs();
    let small = (-2.0 * scale).exp();
    let ch = 0.5 * (1.0 + small);
    let sh = 0.5 * (-(-2.0 * scale).exp_m1()).copysign(imaginary);
    let cosine = C::new(phase.re * ch, -phase.im * sh);
    let sine_value = C::new(phase.im * ch, phase.re * sh);
    Ok(if sine {
        (sine_value, cosine * f64::from(frequency), scale)
    } else {
        (cosine, -sine_value * f64::from(frequency), scale)
    })
}
impl<S: RealScalar> MathieuPlan<S> {
    /// Prepares the Mathieu function of order `n = order` with the given `parity` and parameter `q`.
    ///
    /// The Fourier coefficients satisfy the three-term recurrences of DLMF §28.4, a symmetric
    /// tridiagonal eigenproblem truncated as described in [`SpectralOptions`]. For each
    /// truncation only the requested eigenpair is computed, and the coefficients are rebuilt
    /// from the eigenvalue by scaled recurrences from both ends. The sign follows the standard
    /// convention of DLMF §28.2. At `q = 0` no eigenproblem is solved. `normalization`
    /// selects the angular normalization.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `q` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `parity` is odd and `order = 0`, or if the mode index
    ///   does not fit the truncation bounds of `options` (see [`SpectralOptions`]).
    /// - `ResourceLimit` if preparation needs more memory than `options.preparation_bytes()`.
    /// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients
    ///   leave the `f64` range.
    /// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree
    ///   within the tolerance before `options.terms().maximum()` is reached; the error carries
    ///   the residual and the counts.
    /// - `ConvergenceFailure` (`IterationLimit`) if preparation needs more work than the
    ///   budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_special::{MathieuParity, MathieuPlan, SpectralNormalization};
    ///
    /// // At q = 0: a₂ = 4 and ce₂(z) = cos 2z
    /// let normalization = SpectralNormalization::Standard;
    /// let plan = MathieuPlan::new(2, MathieuParity::Even, 0.0_f64, normalization, Default::default())?;
    /// assert!((plan.characteristic_value()? - 4.0).abs() < 1e-12);
    /// assert!((plan.angular(0.5_f64)?.try_value()? - 1.0_f64.cos()).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        order: u32,
        parity: MathieuParity,
        q: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            order,
            parity,
            q,
            normalization,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan as [`Self::new`] does, inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn build<'scope, 'cap>(
        order: u32,
        parity: MathieuParity,
        q: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        validate(order, parity, q)?;
        scoped(scope, options.iteration_budget, |context| {
            Self::prepare(
                order,
                parity,
                q,
                normalization,
                options,
                scioxide_core::argument!("options.terms.maximum"),
                context,
            )
        })
    }

    /// Computes the coefficients of validated parameters inside an open epoch.
    ///
    /// `mode` names the argument reported when the mode lies outside the truncation bounds.
    fn prepare(
        order: u32,
        parity: MathieuParity,
        q: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
        mode: Argument,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let index = if parity == MathieuParity::Even {
            order / 2
        } else {
            (order - 1) / 2
        } as usize;
        let offset = frequency_offset(order, parity);
        let parameter = q.try_to_f64_exact()?;
        let mut spectrum = spectral::prepare(
            index,
            parameter == 0.0,
            options,
            mode,
            context,
            |k| {
                let frequency = (k as f64) * 2.0 + f64::from(offset);
                let mut diagonal = frequency * frequency;
                if k == 0 && offset == 1 {
                    diagonal += if parity == MathieuParity::Even {
                        parameter
                    } else {
                        -parameter
                    };
                }
                let off = if k == 0 && offset == 0 {
                    core::f64::consts::SQRT_2 * parameter
                } else {
                    parameter
                };
                Ok((diagonal, off))
            },
            |k| {
                let f = k as f64 * 2.0 + f64::from(offset);
                Ok((f * f, parameter.abs()))
            },
        )?;
        if parameter < 0.0 && index % 2 == 1 {
            spectrum.coefficients.iter_mut().for_each(|c| *c = c.neg());
        }
        Ok(Self {
            order,
            parity,
            q,
            options,
            normalization,
            spectrum,
        })
    }

    /// Returns the angular normalization chosen at preparation.
    pub const fn normalization(&self) -> SpectralNormalization {
        self.normalization
    }

    /// Returns the order `n`.
    pub const fn order(&self) -> u32 {
        self.order
    }
    /// Returns the parity.
    pub const fn parity(&self) -> MathieuParity {
        self.parity
    }
    /// Returns the parameter `q`.
    pub const fn parameter(&self) -> S {
        self.q
    }
    /// Returns the characteristic value `a_n(q)` (even) or `b_n(q)` (odd).
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value exceeds the range of the scalar type.
    pub fn characteristic_value(&self) -> Result<S, SciOxideError> {
        let v = S::try_from_f64_lossy(self.spectrum.characteristic)?;
        if !v.is_finite() {
            return Err(instability("characteristic value exceeds scalar range"));
        }
        Ok(v)
    }
    /// Returns the convergence diagnostics of preparation.
    pub const fn diagnostics(&self) -> SpectralDiagnostics {
        self.spectrum.diagnostics
    }
    /// Returns the number of Fourier coefficients kept.
    #[inline]
    pub fn coefficient_count(&self) -> usize {
        self.spectrum.coefficients.len()
    }
    /// Returns the Fourier coefficient `A_k`, `k = index`, in the chosen angular normalization,
    /// as a scaled value.
    ///
    /// The angular function is `Σ_k A_k cos((2k+p)z)` for even parity and
    /// `Σ_k A_k sin((2k+p)z)` for odd parity, summed over `0 ≤ k < coefficient_count()`. The
    /// offset is `p = n mod 2`, except `p = 2` for odd parity and even `n`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index ≥ coefficient_count()`.
    pub fn coefficient(&self, index: usize) -> Result<ScaledValue<S>, SciOxideError> {
        if index >= self.coefficient_count() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("coefficients"),
                index,
                self.coefficient_count(),
            ));
        }
        let value = self.coefficient_scaled(index)?;
        ScaledValue::from_parts(
            S::try_from_f64_lossy(value.significand())?,
            value.exponent(),
        )
    }
    #[inline]
    fn frequency(&self, k: usize) -> Result<u32, SciOxideError> {
        u32::try_from(k)
            .ok()
            .and_then(|k| k.checked_mul(2))
            .and_then(|k| k.checked_add(frequency_offset(self.order, self.parity)))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("mathieu_frequency"))
            })
    }
    #[inline]
    fn coefficient_scale(&self, k: usize) -> f64 {
        let endpoint =
            if self.parity == MathieuParity::Even && self.order.is_multiple_of(2) && k == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
        endpoint
            * if self.normalization == SpectralNormalization::UnitL2 {
                1.0 / core::f64::consts::PI.sqrt()
            } else {
                1.0
            }
    }
    fn coefficient_scaled(&self, k: usize) -> Result<Scaled<f64>, SciOxideError> {
        self.spectrum.coefficients[k].mul(Scaled::from_value(self.coefficient_scale(k))?)
    }
    /// Evaluates `ce_n(z,q)` or `se_n(z,q)` and its derivative with respect to `z`.
    ///
    /// `z` is any finite real or complex number, in radians; the angular functions are entire.
    /// The normalization is the one chosen in [`Self::new`]. The Fourier series is summed with
    /// scaled accumulation, and the phases advance by multiplication with `e^(2iz)`. The error
    /// estimate includes the rounding of that recurrence, the coefficient errors and the
    /// omitted tail. The cost is one work unit per coefficient.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `z` has a NaN or infinite component.
    /// - `NumericalInstability` (`LossOfSignificance`) if `|Im z|` is so large that the omitted
    ///   tail of the Fourier series does not decay.
    /// - `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the
    ///   budget.
    pub fn angular<T: NumericScalar<Real = S>>(
        &self,
        z: T,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        self.angular_scoped(z, ExecutionContext::sequential_native())
    }
    /// Evaluates [`Self::angular`] inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::angular`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn angular_scoped<'scope, 'cap, T: NumericScalar<Real = S>>(
        &self,
        z: T,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        let z = spectral::coordinate(z, "coordinate must be finite")?;
        scoped(scope, self.options.iteration_budget, |context| {
            self.angular_run(z, context)
        })
    }

    /// Evaluates the angular function at a validated coordinate inside an open epoch.
    fn angular_run<T: NumericScalar<Real = S>>(
        &self,
        z: C,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        let raw = self.angular_estimate(z, context);
        spectral::report(raw, true, algorithm_component!("mathieu_angular"), context)
    }

    fn angular_estimate(
        &self,
        z: C,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let (mut value, mut derivative) = (Sum::new(), Sum::new());
        // Adjacent retained frequencies differ by two. Form the
        // increment from sin/cos(z), never from 2*z (which can overflow
        // or lose argument-reduction bits for a large finite input).
        let (sin, cos) = RealScalar::sin_cos(z.re);
        let first = C::new(cos, sin);
        let increment = first * first;
        let increment = increment / increment.norm();
        let mut phase = match self.frequency(0)? {
            0 => real(1.0),
            1 => first,
            _ => increment,
        };
        let sine = self.parity == MathieuParity::Odd;
        let normalization = Scaled::from_value(self.coefficient_scale(1))?;
        for k in 0..self.coefficient_count() {
            step(context)?;
            let frequency = self.frequency(k)?;
            let scale = if k == 0 {
                Scaled::from_value(self.coefficient_scale(0))?
            } else {
                normalization
            };
            let coefficient = self.spectrum.coefficients[k].mul(scale)?;
            let error = self.spectrum.errors[k].mul(scale)?;
            // Absolute phase error, including recurrence drift, remains
            // meaningful at a sine/cosine zero. Derivatives amplify it
            // by the frequency; scaled complex trig has magnitude <= 1.
            let rounding = 32.0 * f64::EPSILON * (1.0 + f64::from(frequency));
            if z.im == 0.0 {
                let (v, d) = if sine {
                    (phase.im, phase.re)
                } else {
                    (phase.re, -phase.im)
                };
                value.add_real(v, coefficient, rounding, error)?;
                derivative.add_real(
                    d * f64::from(frequency),
                    coefficient,
                    rounding * f64::from(frequency),
                    error,
                )?;
            } else {
                let (v, d, scale) = trig(frequency, z, phase, sine)?;
                value.add(v, scale, coefficient, rounding, error)?;
                derivative.add(
                    d,
                    scale,
                    coefficient,
                    rounding * f64::from(frequency),
                    error,
                )?;
            }
            phase *= increment;
            phase = phase / phase.norm();
        }
        let ratio = self.spectrum.tail_ratio;
        if ratio > 0.0 {
            let log_ratio = ratio.ln() + 2.0 * z.im.abs();
            if log_ratio >= 0.0 {
                return Err(instability(
                    "retained Fourier tail cannot resolve this complex coordinate",
                ));
            }
            let frequency = self.frequency(self.coefficient_count())?;
            let denominator = 1.0 - log_ratio.exp();
            let tail_factor = (1.0 - ratio) / denominator;
            let scale = f64::from(frequency) * z.im.abs();
            let tail = self
                .spectrum
                .tail
                .mul(Scaled::from_value(tail_factor)?)?
                .mul(if scale == 0.0 {
                    Scaled::ONE
                } else {
                    exp_scaled(scale)?
                })?;
            value.error = value.error.add(tail)?;
            derivative.error = derivative.error.add(tail.mul(Scaled::from_value(
                f64::from(frequency) + 2.0 / denominator,
            )?)?)?;
        }
        to_estimate(value, derivative, algorithm_component!("mathieu_fourier"))
    }
}

/// Rejects parameters that no preparation can serve, before a preparation epoch opens.
fn validate<S: RealScalar>(order: u32, parity: MathieuParity, q: S) -> Result<(), SciOxideError> {
    if !q.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "q must be finite",
        ));
    }
    if parity == MathieuParity::Odd && order == 0 {
        return Err(invalid(
            scioxide_core::argument!("order"),
            "se_0 is not an eigenfunction family",
        ));
    }
    Ok(())
}

/// Returns the lowest Fourier frequency `p` of the expansion: `n mod 2`, or `2` for odd parity
/// and even `n`.
const fn frequency_offset(order: u32, parity: MathieuParity) -> u32 {
    if order % 2 == 1 {
        1
    } else if matches!(parity, MathieuParity::Even) {
        0
    } else {
        2
    }
}
fn exp_scaled(log: f64) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::from_log(1.0, log)
}
fn to_estimate(
    value: Sum,
    derivative: Sum,
    method: AlgorithmComponent,
) -> Result<Estimate, SciOxideError> {
    let (parts, exponent) = factor_common_scale([
        value.re.total()?,
        value.im.total()?,
        derivative.re.total()?,
        derivative.im.total()?,
        value.error,
        derivative.error,
    ])?;
    let error = parts[4].round()?;
    Ok(Estimate::from(crate::support::JetData {
        derivative_error: Some(parts[5].round()?),
        y: Jet {
            v: C::new(parts[0].round()?, parts[1].round()?),
            d: C::new(parts[2].round()?, parts[3].round()?),
        },
        scale: Jet::constant(real(exponent as f64 * core::f64::consts::LN_2)),
        error,
        method,
    }))
}
type BesselValue = crate::bessel_airy::complex_bessel::ExponentialEstimate;
#[derive(Clone, Copy)]
struct Product {
    value: BesselValue,
    derivative: BesselValue,
    bound: BesselValue,
    derivative_bound: BesselValue,
}
fn normalized(v: BesselValue) -> BesselValue {
    let size = v.value.re.abs().max(v.value.im.abs());
    if size == 0.0 {
        v
    } else {
        BesselValue {
            value: v.value / size,
            scale: v.scale + size.ln(),
            error: v.error / size,
        }
    }
}
fn product(a: BesselValue, b: BesselValue) -> Result<BesselValue, SciOxideError> {
    let a = normalized(a);
    let b = normalized(b);
    let value = a.value * b.value;
    let error = a.error * b.value.norm()
        + b.error * a.value.norm()
        + a.error * b.error
        + 32.0 * f64::EPSILON * value.norm();
    let scale = a.scale + b.scale;
    if !value.is_finite() || !error.is_finite() || !scale.is_finite() {
        return Err(instability(
            "radial Bessel product lost its working representation",
        ));
    }
    Ok(BesselValue {
        value,
        error,
        scale,
    })
}
fn sum(a: BesselValue, b: BesselValue, sign: f64) -> Result<BesselValue, SciOxideError> {
    let a = normalized(a);
    let b = normalized(b);
    let scale = a.scale.max(b.scale);
    let pa = (a.scale - scale).exp();
    let pb = (b.scale - scale).exp();
    let value = a.value * pa + b.value * (sign * pb);
    let error = a.error * pa
        + b.error * pb
        + 16.0 * f64::EPSILON * (a.value.norm() * pa + b.value.norm() * pb);
    if !value.is_finite() || !error.is_finite() {
        return Err(instability("radial Bessel combination lost significance"));
    }
    Ok(BesselValue {
        value,
        error,
        scale,
    })
}
fn coordinate_factor(z: C) -> BesselValue {
    BesselValue {
        value: z,
        scale: 0.0,
        error: 32.0 * f64::EPSILON * z.re.abs().max(z.im.abs()),
    }
}
fn term(
    a: f64,
    b: f64,
    u: C,
    v: C,
    kind: crate::bessel_airy::complex_bessel_common::Kind,
    work: &mut crate::support::Work,
) -> Result<Product, SciOxideError> {
    use crate::bessel_airy::complex_bessel::{Kind, cylindrical_pair};

    let (j, jp) = cylindrical_pair(Kind::J, a, u, work)?;
    let (c, cp) = cylindrical_pair(kind, b, v, work)?;
    let jp = jp.ok_or_else(|| instability("radial J derivative unavailable"))?;
    let cp = cp.ok_or_else(|| instability("radial cylinder derivative unavailable"))?;
    let value = product(j, c)?;
    let left = product(product(jp, c)?, coordinate_factor(-u))?;
    let right = product(product(j, cp)?, coordinate_factor(v))?;
    let derivative = sum(left, right, 1.0)?;
    let bound = BesselValue {
        value: real(value.value.norm() + value.error),
        scale: value.scale,
        error: 0.0,
    };
    let derivative_bound = sum(
        BesselValue {
            value: real(left.value.norm() + left.error),
            scale: left.scale,
            error: 0.0,
        },
        BesselValue {
            value: real(right.value.norm() + right.error),
            scale: right.scale,
            error: 0.0,
        },
        1.0,
    )?;
    Ok(Product {
        value,
        derivative,
        bound,
        derivative_bound,
    })
}
fn absolute(v: BesselValue) -> Result<Scaled<f64>, SciOxideError> {
    exp_scaled(v.scale)?.mul(Scaled::from_value(v.value.norm() + v.error)?)
}
impl<S: RealScalar> MathieuPlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Evaluates the radial Mathieu function `Mc_n^(j)(z,q)` or `Ms_n^(j)(z,q)` and its derivative
    /// with respect to `z`.
    ///
    /// The function solves `w″ − (a − 2q cosh 2z) w = 0` and has the kind `j` selected by `kind`.
    /// It is the series of products of Bessel functions `J(√q e^(−z))` and `C(√q e^z)` of
    /// DLMF §28.24, where `C` is the Bessel function of the selected kind. That series fixes the
    /// normalization, which does not depend on the angular normalization. The third and fourth
    /// kinds are first ± i · second kind. For `q < 0`, `√q` is the principal square root, so the
    /// result is complex even for real `z`. `z` is any finite real or complex number. At `q = 0`
    /// the first kind is `1` for `n = 0` and `0` otherwise. The cost is two Bessel-function pairs
    /// per coefficient.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `z` has a NaN or infinite component.
    /// - `DomainError` (`Pole`) at `q = 0` for the second, third and fourth kinds.
    /// - `NumericalInstability` (`LossOfSignificance`) if `√q e^(±z)` overflows or vanishes, or
    ///   if the kept coefficients cannot resolve `z`; raise the truncation bounds in that case.
    /// - `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the
    ///   budget.
    pub fn radial<T: NumericScalar<Real = S>>(
        &self,
        z: T,
        kind: RadialKind,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        self.radial_scoped(z, kind, ExecutionContext::sequential_native())
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
        z: T,
        kind: RadialKind,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        let z = spectral::coordinate(z, "radial coordinate must be finite")?;
        scoped(scope, self.options.iteration_budget, |context| {
            self.radial_run(z, kind, context)
        })
    }

    /// Evaluates the radial function at a validated coordinate inside an open epoch.
    fn radial_run(
        &self,
        z: C,
        kind: RadialKind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SpecialFunctionResult<Complex<S>>, SciOxideError> {
        if self.q == S::ZERO {
            step(context)?;
            if kind != RadialKind::First {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("q"),
                    scioxide_core::DomainReason::Pole,
                    "the higher radial kinds diverge at q=0",
                ));
            }
            let value = Complex::new(if self.order == 0 { S::ONE } else { S::ZERO }, S::ZERO);
            return Evaluation::exact(
                value,
                Some(Complex::new(S::ZERO, S::ZERO)),
                scioxide_core::algorithm_id!("mathieu_radial", "zero_parameter_limit", 1),
                context.counts().own().iterations(),
            )
            .publish();
        }
        let raw = self.radial_estimate(z, kind, context);
        spectral::report(raw, true, algorithm_component!("mathieu_radial"), context)
    }

    fn radial_estimate(
        &self,
        z: C,
        kind: RadialKind,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let mut work = crate::support::Work::with_context(
            self.options.iteration_budget.get(),
            Some(&mut *context),
        );
        let calculation =
            self.radial_sums(z, kind.bessel_kind(), &mut work)
                .and_then(|(value, derivative)| {
                    to_estimate(
                        value,
                        derivative,
                        algorithm_component!("mathieu_bessel_cross_products"),
                    )
                });
        work.check()?;
        calculation
    }

    fn radial_sums(
        &self,
        z: C,
        kind: crate::bessel_airy::complex_bessel_common::Kind,
        work: &mut crate::support::Work,
    ) -> Result<(Sum, Sum), SciOxideError> {
        let parameter = self.q.try_to_f64_exact()?;
        let h = C::new(parameter, 0.0).sqrt();
        let u = h * (-z).exp();
        let v = h * z.exp();
        if !u.is_finite() || !v.is_finite() || u == real(0.0) || v == real(0.0) {
            return Err(instability(
                "radial coordinate cannot form finite nonzero Bessel arguments",
            ));
        }
        let mut pivot = 0;
        for k in 1..self.coefficient_count() {
            work.try_step()?;
            if self
                .coefficient_scaled(pivot)?
                .abs_le(self.coefficient_scaled(k)?.abs())
            {
                pivot = k;
            }
        }
        let denominator = self.coefficient_scaled(pivot)?;
        let denominator_error =
            self.spectrum.errors[pivot].mul(Scaled::from_value(self.coefficient_scale(pivot))?)?;
        let relative_denominator_error = denominator_error.div(denominator.abs())?.round()?;
        if relative_denominator_error >= 0.5 {
            return Err(instability(
                "radial normalization coefficient is unresolved",
            ));
        }
        let mode = if self.parity == MathieuParity::Even {
            self.order / 2
        } else {
            (self.order - 1) / 2
        };
        let offset = self.frequency(0)?;
        let epsilon = if offset == 0 && pivot == 0 { 2.0 } else { 1.0 };
        let (mut value, mut derivative) = (Sum::new(), Sum::new());
        let mut tails = [Scaled::ZERO; 4];
        let mut dtails = tails;
        for k in 0..self.coefficient_count() {
            work.try_step()?;
            let a = k as f64 - pivot as f64;
            let b = (k + pivot) as f64 + f64::from(offset);
            let first = term(a, b, u, v, kind, work)?;
            let swap = if a == b {
                first
            } else {
                term(b, a, u, v, kind, work)?
            };
            let sign = if self.parity == MathieuParity::Even {
                1.0
            } else {
                -1.0
            };
            let combined = sum(first.value, swap.value, sign)?;
            let differentiated = sum(first.derivative, swap.derivative, sign)?;
            let weight = self
                .coefficient_scaled(k)?
                .div(denominator)?
                .mul(Scaled::from_value(
                    if (u64::from(mode) + k as u64).is_multiple_of(2) {
                        1.0 / epsilon
                    } else {
                        -1.0 / epsilon
                    },
                )?)?;
            let error = self.spectrum.errors[k]
                .mul(Scaled::from_value(self.coefficient_scale(k))?)?
                .div(denominator.abs())?
                .mul(Scaled::from_value(1.0 / epsilon)?)?
                .add(weight.abs().mul(Scaled::from_value(
                    relative_denominator_error / (1.0 - relative_denominator_error),
                )?)?)?;
            value.add(
                combined.value,
                combined.scale,
                weight,
                combined.error,
                error,
            )?;
            derivative.add(
                differentiated.value,
                differentiated.scale,
                weight,
                differentiated.error,
                error,
            )?;
            tails[k % 4] = absolute(sum(first.bound, swap.bound, 1.0)?)?.mul(weight.abs())?;
            dtails[k % 4] = absolute(sum(first.derivative_bound, swap.derivative_bound, 1.0)?)?
                .mul(weight.abs())?;
        }
        // Coefficient majorant combined with coordinate and order growth.
        // This is a tail estimate, not a rigorous enclosure of cylinder functions.
        let n = self.coefficient_count() as f64;
        let p = pivot as f64;
        let order_growth = ((n + p + f64::from(offset) + 2.0) / (n - p)).powi(2);
        let coordinate_growth =
            if !matches!(kind, crate::bessel_airy::complex_bessel_common::Kind::J) {
                (-2.0 * z.re).max(0.0).exp()
            } else {
                1.0
            };
        let growth = coordinate_growth
            .max(parameter.abs() / (4.0 * (n - p).powi(2)))
            .max(1.0)
            * order_growth;
        let ratio = self.spectrum.tail_ratio * growth;
        if !ratio.is_finite() || ratio >= 1.0 {
            return Err(instability(
                "retained radial expansion cannot resolve the coordinate; increase SpectralTerms::minimum and maximum",
            ));
        }
        let largest = |values: &[Scaled<f64>]| {
            values
                .iter()
                .copied()
                .fold(Scaled::ZERO, |a, b| if a.abs_le(b) { b } else { a })
        };
        let multiplier = Scaled::from_value(ratio / (1.0 - ratio))?;
        value.error = value.error.add(largest(&tails).mul(multiplier)?)?;
        derivative.error = derivative.error.add(largest(&dtails).mul(multiplier)?)?;
        Ok((value, derivative))
    }
}

/// Computes the characteristic value `a_n(q)` of the even Mathieu functions.
///
/// `a_n(q)` is the value of `a` for which Mathieu's equation `w″ + (a − 2q cos 2z) w = 0`
/// (DLMF 28.2.1), with order `n ≥ 0` and real parameter `q`, has the even periodic solution
/// `ce_n(z, q)`; `a_n(0) = n²`. It is the characteristic value of a [`MathieuPlan`] with
/// [`MathieuParity::Even`] and default [`SpectralOptions`]. Each call prepares one plan, as
/// described for [`MathieuPlan`]; prepare the plan to also evaluate `ce_n` and `Mc_n^(j)`.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if the mode lies outside the default
///   truncation bounds: `order ≥ 496`, or `order ≥ 512` at `q = 0`.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, or if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation needs more than 2,097,152 work
///   units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. Preparation refines the truncation until two successive
/// characteristic values agree to `1e-13`, absolutely or relatively; see [`MathieuPlan`] and
/// the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::mathieu_a;
///
/// // a₂(0) = 4
/// assert!((mathieu_a(2, 0.0_f64)? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.2 and §28.4. <https://dlmf.nist.gov/28>
pub fn mathieu_a<S: RealScalar>(order: u32, q: S) -> Result<S, SciOxideError> {
    prepared_call(
        order,
        MathieuParity::Even,
        q,
        SpectralOptions::default(),
        |plan, _| plan.characteristic_value(),
    )
}

/// Computes the characteristic value `b_n(q)` of the odd Mathieu functions.
///
/// `b_n(q)` is the value of `a` for which Mathieu's equation `w″ + (a − 2q cos 2z) w = 0`
/// (DLMF 28.2.1), with order `n ≥ 1` and real parameter `q`, has the odd periodic solution
/// `se_n(z, q)`; `b_n(0) = n²`. It is the characteristic value of a [`MathieuPlan`] with
/// [`MathieuParity::Odd`] and default [`SpectralOptions`]. Each call prepares one plan, as
/// described for [`MathieuPlan`]; prepare the plan to also evaluate `se_n` and `Ms_n^(j)`.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if `order = 0`, or if the mode lies
///   outside the default truncation bounds: `order ≥ 497`, or `order ≥ 513` at `q = 0`.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, or if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation needs more than 2,097,152 work
///   units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. Preparation refines the truncation until two successive
/// characteristic values agree to `1e-13`, absolutely or relatively; see [`MathieuPlan`] and
/// the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::mathieu_b;
///
/// // b₂(0) = 4
/// assert!((mathieu_b(2, 0.0_f64)? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.2 and §28.4. <https://dlmf.nist.gov/28>
pub fn mathieu_b<S: RealScalar>(order: u32, q: S) -> Result<S, SciOxideError> {
    prepared_call(
        order,
        MathieuParity::Odd,
        q,
        SpectralOptions::default(),
        |plan, _| plan.characteristic_value(),
    )
}

/// Computes the even angular Mathieu function `ce_n(z, q)`.
///
/// `ce_n(z, q)` is the even periodic solution of Mathieu's equation
/// `w″ + (a_n(q) − 2q cos 2z) w = 0` (DLMF 28.2.1), with order `n ≥ 0`, real parameter `q` and
/// real or complex `z` in radians; `ce_n(z, 0) = cos nz` for `n ≥ 1` and `ce₀(z, 0) = 1/√2`. The
/// sign and the normalization `∫₀^2π ce_n(z, q)² dz = π` are those of DLMF §28.2
/// ([`SpectralNormalization::Standard`]). Each call prepares one [`MathieuPlan`] with default
/// [`SpectralOptions`] and sums its Fourier series once, at one work unit per coefficient;
/// preparation and evaluation share one budget of 2,097,152 work units. Prepare a
/// [`MathieuPlan`] to evaluate many points or to choose another normalization.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite, or if `z` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if the mode lies outside the default
///   truncation bounds: `order ≥ 496`, or `order ≥ 512` at `q = 0`.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, or if `|Im z|` is so large that the omitted tail of the Fourier series
///   does not decay; (`Overflow`) if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`; for
/// complex `z` the omitted tail grows like `e^(ν|Im z|)`, where `ν` is the first omitted Fourier
/// frequency. See [`MathieuPlan`] and the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::mathieu_ce;
///
/// // ce₂(z, 0) = cos 2z
/// let z = 0.5_f64;
/// assert!((mathieu_ce(2, 0.0, z)? - (2.0 * z).cos()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.2 and §28.4. <https://dlmf.nist.gov/28>
pub fn mathieu_ce<T: NumericScalar>(order: u32, q: T::Real, z: T) -> Result<T, SciOxideError> {
    angular_value(order, MathieuParity::Even, q, z)
}

/// Computes the odd angular Mathieu function `se_n(z, q)`.
///
/// `se_n(z, q)` is the odd periodic solution of Mathieu's equation
/// `w″ + (b_n(q) − 2q cos 2z) w = 0` (DLMF 28.2.1), with order `n ≥ 1`, real parameter `q` and
/// real or complex `z` in radians; `se_n(z, 0) = sin nz`. The sign and the normalization
/// `∫₀^2π se_n(z, q)² dz = π` are those of DLMF §28.2 ([`SpectralNormalization::Standard`]).
/// Each call prepares one [`MathieuPlan`] with default [`SpectralOptions`] and sums its Fourier
/// series once, at one work unit per coefficient; preparation and evaluation share one budget
/// of 2,097,152 work units. Prepare a [`MathieuPlan`] to evaluate many points or to choose
/// another normalization.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite, or if `z` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if `order = 0`, or if the mode lies
///   outside the default truncation bounds: `order ≥ 497`, or `order ≥ 513` at `q = 0`.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, or if `|Im z|` is so large that the omitted tail of the Fourier series
///   does not decay; (`Overflow`) if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`; for
/// complex `z` the omitted tail grows like `e^(ν|Im z|)`, where `ν` is the first omitted Fourier
/// frequency. See [`MathieuPlan`] and the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::mathieu_se;
///
/// // se₂(z, 0) = sin 2z
/// let z = 0.5_f64;
/// assert!((mathieu_se(2, 0.0, z)? - (2.0 * z).sin()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.2 and §28.4. <https://dlmf.nist.gov/28>
pub fn mathieu_se<T: NumericScalar>(order: u32, q: T::Real, z: T) -> Result<T, SciOxideError> {
    angular_value(order, MathieuParity::Odd, q, z)
}

/// Computes the radial Mathieu function `Mc_n^(j)(z, q)` of the even family.
///
/// `Mc_n^(j)` solves the modified Mathieu equation `w″ − (a_n(q) − 2q cosh 2z) w = 0`
/// (DLMF 28.20.1), with order `n ≥ 0`, real parameter `q` and real or complex `z`; `kind`
/// selects the kind `j`. The value is the series of products of Bessel functions of
/// DLMF §28.24, whose normalization does not depend on the angular one; the third and fourth
/// kinds are first ± i · second kind. For `q < 0`, `√q` is the principal square root, so the
/// value is complex even for real `z`. At `q = 0` the first kind is `1` for `n = 0` and `0`
/// otherwise. Each call prepares one [`MathieuPlan`] with default [`SpectralOptions`] and sums
/// its series once, at two Bessel-function pairs per coefficient; preparation and evaluation
/// share one budget of 2,097,152 work units. Prepare a [`MathieuPlan`] to evaluate many points.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite, or if `z` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if the mode lies outside the default
///   truncation bounds: `order ≥ 496`, or `order ≥ 512` at `q = 0`.
/// - `DomainError` (`Pole`) with argument `q` at `q = 0` for the second, third and fourth kinds.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, if `√q e^(±z)` overflows or vanishes, or if the kept coefficients cannot
///   resolve `z`; (`Overflow`) if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`, and the
/// omitted tail of the product series is estimated, not bounded. See [`MathieuPlan`] and the
/// [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::{RadialKind, mathieu_mc};
///
/// // Mc₀⁽¹⁾(z, 0) = 1
/// let mc = mathieu_mc(0, 0.0, 0.5_f64, RadialKind::First)?;
/// assert!((mc.re - 1.0).abs() < 1e-12 && mc.im.abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.20 and §28.24. <https://dlmf.nist.gov/28>
pub fn mathieu_mc<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    kind: RadialKind,
) -> Result<Complex<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    radial_value(order, MathieuParity::Even, q, z, kind)
}

/// Computes the radial Mathieu function `Ms_n^(j)(z, q)` of the odd family.
///
/// `Ms_n^(j)` solves the modified Mathieu equation `w″ − (b_n(q) − 2q cosh 2z) w = 0`
/// (DLMF 28.20.1), with order `n ≥ 1`, real parameter `q` and real or complex `z`; `kind`
/// selects the kind `j`. The value is the series of products of Bessel functions of
/// DLMF §28.24, whose normalization does not depend on the angular one; the third and fourth
/// kinds are first ± i · second kind. For `q < 0`, `√q` is the principal square root, so the
/// value is complex even for real `z`. At `q = 0` the first kind is `0`. Each call prepares one
/// [`MathieuPlan`] with default [`SpectralOptions`] and sums its series once, at two
/// Bessel-function pairs per coefficient; preparation and evaluation share one budget of
/// 2,097,152 work units. Prepare a [`MathieuPlan`] to evaluate many points.
///
/// # Errors
///
/// - `NonFinite` if `q` is NaN or infinite, or if `z` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `order` if `order = 0`, or if the mode lies
///   outside the default truncation bounds: `order ≥ 497`, or `order ≥ 513` at `q = 0`.
/// - `DomainError` (`Pole`) with argument `q` at `q = 0` for the second, third and fourth kinds.
/// - `NumericalInstability` (`LossOfSignificance`) if the matrix entries or coefficients leave
///   the `f64` range, if `√q e^(±z)` overflows or vanishes, or if the kept coefficients cannot
///   resolve `z`; (`Overflow`) if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`, and the
/// omitted tail of the product series is estimated, not bounded. See [`MathieuPlan`] and the
/// [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_special::{RadialKind, mathieu_ms};
///
/// // Ms₁⁽³⁾ = Ms₁⁽¹⁾ + i Ms₁⁽²⁾
/// let ms = |kind| mathieu_ms(1, 1.0, 0.5_f64, kind);
/// let (first, second) = (ms(RadialKind::First)?, ms(RadialKind::Second)?);
/// let third = ms(RadialKind::Third)?;
/// assert!((third - (first + second * Complex::i())).norm() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §28.20 and §28.24. <https://dlmf.nist.gov/28>
pub fn mathieu_ms<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    kind: RadialKind,
) -> Result<Complex<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    radial_value(order, MathieuParity::Odd, q, z, kind)
}

fn angular_value<T: NumericScalar>(
    order: u32,
    parity: MathieuParity,
    q: T::Real,
    z: T,
) -> Result<T, SciOxideError> {
    let z = spectral::coordinate(z, "coordinate must be finite")?;
    prepared_call(
        order,
        parity,
        q,
        SpectralOptions::default(),
        |plan, context| plan.angular_run(z, context),
    )?
    .try_value()
}

fn radial_value<T: NumericScalar>(
    order: u32,
    parity: MathieuParity,
    q: T::Real,
    z: T,
    kind: RadialKind,
) -> Result<Complex<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let z = spectral::coordinate(z, "radial coordinate must be finite")?;
    prepared_call(
        order,
        parity,
        q,
        SpectralOptions::default(),
        |plan, context| plan.radial_run(z, kind, context),
    )?
    .try_value()
}

// One allowance covers preparation, its nested eigensolver, and evaluation.
fn prepared_call<S: RealScalar, V>(
    order: u32,
    parity: MathieuParity,
    q: S,
    options: SpectralOptions<S>,
    run: impl FnOnce(&MathieuPlan<S>, &mut SolveContext<'_, '_>) -> Result<V, SciOxideError>,
) -> Result<V, SciOxideError> {
    validate(order, parity, q)?;
    scoped(
        ExecutionContext::sequential_native(),
        options.iteration_budget,
        |context| {
            let plan = MathieuPlan::prepare(
                order,
                parity,
                q,
                SpectralNormalization::Standard,
                options,
                scioxide_core::argument!("order"),
                context,
            )?;
            run(&plan, context)
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Budget;

    #[test]
    fn one_shot_allowance_covers_preparation_and_evaluation() {
        let plan = |options| {
            MathieuPlan::new(
                2,
                MathieuParity::Even,
                1.0_f64,
                SpectralNormalization::Standard,
                options,
            )
        };
        let reference = plan(SpectralOptions::default()).unwrap();
        let work = reference.diagnostics().work();
        let options =
            SpectralOptions::default().with_iteration_budget(Budget::try_from(work + 1).unwrap());
        // The budget suffices for preparation alone, with one unit left for evaluation.
        assert_eq!(plan(options).unwrap().diagnostics().work(), work);
        let call = |options, radial: bool| {
            prepared_call(2, MathieuParity::Even, 1.0_f64, options, |plan, context| {
                if radial {
                    plan.radial_run(real(0.5), RadialKind::First, context)
                        .and_then(SpecialFunctionResult::try_value)
                        .map(|value| value.re)
                } else {
                    plan.angular_run::<f64>(real(0.5), context)
                        .and_then(SpecialFunctionResult::try_value)
                }
            })
        };
        for radial in [false, true] {
            let _ = crate::test_support::exhausted_counts(&call(options, radial).unwrap_err());
        }
        assert_eq!(
            call(SpectralOptions::default(), false).unwrap().to_bits(),
            reference
                .angular(0.5_f64)
                .unwrap()
                .try_value()
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            call(SpectralOptions::default(), true).unwrap().to_bits(),
            reference
                .radial(0.5_f64, RadialKind::First)
                .unwrap()
                .try_value()
                .unwrap()
                .re
                .to_bits()
        );
    }
}
