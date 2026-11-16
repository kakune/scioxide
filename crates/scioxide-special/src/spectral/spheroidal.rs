//! Prepared prolate and oblate spheroidal functions; DLMF 30.8, 30.11 and product expansions.

use crate::spectral::{self, Spectrum, instability, invalid, step};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::{C, real};
use crate::util::scoped;
use crate::{
    RadialKind, SpecialFunctionResult, SpectralDiagnostics, SpectralNormalization, SpectralOptions,
};
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{
    Argument, Complex, ExecutionContext, NumericScalar, RealScalar, ScaledValue, SciOxideError,
    SolveContext, SolveScope,
};
use scioxide_core::{Scaled, ScaledAccumulator};
type Z = ScaledValue<scioxide_core::Complex<f64>>;
/// Selects prolate or oblate spheroidal functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpheroidalKind {
    /// Prolate: `s = +1` in the equation of [`SpheroidalPlan`]; the physical radial coordinate
    /// is `x ≥ 1`.
    Prolate,
    /// Oblate: `s = −1` in the equation of [`SpheroidalPlan`]; the physical radial coordinate
    /// is `x ≥ 0`.
    Oblate,
}
pub(crate) struct Evaluation<V: NumericScalar<Real = f64> = C> {
    pub value: ScaledValue<V>,
    pub derivative: Option<ScaledValue<V>>,
    pub error: Scaled<f64>,
    pub derivative_error: Scaled<f64>,
}
fn zscale<V: NumericScalar<Real = f64>>(
    z: ScaledValue<V>,
    s: Scaled<f64>,
) -> Result<ScaledValue<V>, SciOxideError> {
    z.mul_real(s)
}
fn scalar(x: f64) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::from_value(x)
}
fn norm<V: NumericScalar<Real = f64>>(z: ScaledValue<V>) -> Result<Scaled<f64>, SciOxideError> {
    if V::IS_REAL {
        Ok(z.real().abs())
    } else {
        z.magnitude()
    }
}
// Keep real recurrence products in the real scaled domain; complex products
// still retain independently scaled components and compound cancellation.
fn product<V: NumericScalar<Real = f64>>(
    a: ScaledValue<V>,
    b: ScaledValue<V>,
) -> Result<ScaledValue<V>, SciOxideError> {
    if V::IS_REAL {
        Ok(ScaledValue::from_components(
            a.real().mul(b.real())?,
            Scaled::ZERO,
        ))
    } else {
        a.mul(b)
    }
}
fn power<V: NumericScalar<Real = f64>>(
    mut z: ScaledValue<V>,
    mut n: u32,
    context: &mut SolveContext<'_, '_>,
) -> Result<ScaledValue<V>, SciOxideError> {
    let mut out = ScaledValue::ONE;
    while n > 0 {
        step(context)?;
        if n & 1 != 0 {
            out = product(out, z)?;
        }
        n >>= 1;
        if n > 0 {
            z = product(z, z)?;
        }
    }
    Ok(out)
}
struct Basis<'a, V: NumericScalar<Real = f64> = C> {
    m: usize,
    l: usize,
    x: ScaledValue<V>,
    previous: ScaledValue<V>,
    current: ScaledValue<V>,
    previous_derivative: ScaledValue<V>,
    derivative: ScaledValue<V>,
    steps: &'a [(Scaled<f64>, Scaled<f64>)],
}
impl<'a, V: NumericScalar<Real = f64>> Basis<'a, V> {
    fn new(
        m: u32,
        x: V,
        diagonal: Scaled<f64>,
        steps: &'a [(Scaled<f64>, Scaled<f64>)],
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            m: m as usize,
            l: m as usize,
            x: ScaledValue::from_value(x)?,
            previous: ScaledValue::ZERO,
            current: ScaledValue::from_components(diagonal, Scaled::ZERO),
            previous_derivative: ScaledValue::ZERO,
            derivative: ScaledValue::ZERO,
            steps,
        })
    }
    fn advance(&mut self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        self.advance_with(&mut || step(context))
    }
    fn advance_with(
        &mut self,
        admit: &mut impl FnMut() -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        admit()?;
        let (a, b) = self.steps[self.l - self.m];
        self.l = self
            .l
            .checked_add(1)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("degree")))?;
        let next = zscale(product(self.x, self.current)?, a)?.sub(zscale(self.previous, b)?)?;
        let derivative = zscale(self.current.add(product(self.x, self.derivative)?)?, a)?
            .sub(zscale(self.previous_derivative, b)?)?;
        self.previous = self.current;
        self.current = next;
        self.previous_derivative = self.derivative;
        self.derivative = derivative;
        Ok(())
    }
}
/// Holds the expansion coefficients and the characteristic value of one spheroidal wave function.
///
/// The spheroidal wave equation is, in the convention of Flammer (1957),
///
/// ```text
/// (1 − x²) w″ − 2x w′ + (λ_mn − s c² x² − m²/(1 − x²)) w = 0,
/// ```
///
/// where `m ≥ 0` is the order, `n ≥ m` the degree, `c` the real size parameter, `s = +1` for
/// prolate and `s = −1` for oblate functions, and `λ_mn` the characteristic value. In the
/// notation of DLMF 30.2.1, `γ² = s c²` and `λ_mn = λ_n^m(γ²) + γ²`. The angular functions
/// `S_mn(x)` on `[−1, 1]` reduce to the Ferrers functions `P_n^m(x)` at `c = 0`, including the
/// Condon–Shortley phase. The radial functions `R_mn^(j)(x)` solve the same equation in the
/// radial coordinate.
///
/// [`Self::new`] computes the coefficients once; [`Self::angular`] and [`Self::radial`] then
/// evaluate the functions at many points. The plan stores the characteristic value, the `N`
/// expansion coefficients with their error bounds, an estimate of the omitted tail, the `2N`
/// Legendre recurrence factors that evaluation uses and the normalization of the radial
/// product expansion, where `N ≤ 256` is the coefficient count chosen by [`SpectralOptions`].
/// Preparation solves the eigenproblem of each doubled truncation by Sturm bisection and
/// inverse iteration for the one requested eigenpair, then builds the recurrence factors and
/// the radial normalization, in `O(N + m)` operations and `O(N)` memory. Preparation and
/// evaluation compute in `f64` for both scalar types. The one-shot functions
/// [`spheroidal_characteristic_value`], [`spheroidal_angular`] and [`spheroidal_radial`]
/// prepare one plan per call. See
/// [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The tolerance of [`SpectralOptions`] steers the truncation,
/// [`Self::diagnostics`] reports the eigenvalue residual and the omitted tail, and each
/// evaluation returns an error estimate derived from the coefficient and truncation errors.
/// The estimates are not guaranteed bounds, and complex coordinates can amplify the error. See
/// the [crate accuracy table](crate#accuracy).
///
/// # References
///
/// - NIST DLMF, Chapter 30. <https://dlmf.nist.gov/30>
/// - C. Flammer, *Spheroidal Wave Functions*, Stanford University Press, 1957.
#[derive(Debug, Clone)]
pub struct SpheroidalPlan<S: RealScalar> {
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: S,
    pub(crate) options: SpectralOptions<S>,
    normalization: SpectralNormalization,
    spectrum: Spectrum,
    steps: Vec<(Scaled<f64>, Scaled<f64>)>,
    diagonal: Scaled<f64>,
    angular_scale: Scaled<f64>,
    eta: f64,
    center_derivative: bool,
    denominator: Scaled<f64>,
    denominator_error: Scaled<f64>,
}
fn x_coefficient(l: u64, m: u32) -> f64 {
    let l = l as f64;
    let m = f64::from(m);
    (((l + 1.0 - m) * (l + 1.0 + m)) / ((2.0 * l + 1.0) * (2.0 * l + 3.0))).sqrt()
}
fn square_diagonal(l: u64, m: u32) -> f64 {
    x_coefficient(l, m).powi(2)
        + if l == u64::from(m) {
            0.0
        } else {
            x_coefficient(l - 1, m).powi(2)
        }
}
impl<S: RealScalar> SpheroidalPlan<S> {
    /// Prepares the spheroidal function of order `m` and degree `n` for the size parameter `c`.
    ///
    /// The angular function is expanded in the Ferrers functions `P_(m+2k+p)^m`, `k ≥ 0`, with
    /// `p = (n − m) mod 2` (DLMF §30.8). The coefficients solve a symmetric tridiagonal
    /// eigenproblem truncated as described in [`SpectralOptions`], and for each truncation only
    /// the requested eigenpair is computed. At `c = 0` no eigenproblem is solved. Preparation
    /// also stores the Legendre recurrence factors that evaluation uses, within the memory limit
    /// of `options`. `normalization` selects the angular normalization.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `n < m`, or if the mode index `⌊(n − m)/2⌋` does not
    ///   fit the truncation bounds of `options` (see [`SpectralOptions`]).
    /// - `ResourceLimit` if preparation needs more memory than `options.preparation_bytes()`.
    /// - `NumericalInstability` (`LossOfSignificance`) if `c²` or the matrix entries leave the
    ///   `f64` range, or if no normalization point of the radial expansion is resolved.
    /// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree
    ///   within the tolerance before `options.terms().maximum()` is reached; the error carries
    ///   the residual and the counts.
    /// - `ConvergenceFailure` (`IterationLimit`) if preparation needs more work than the
    ///   budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_special::{SpectralNormalization, SpheroidalKind, SpheroidalPlan};
    ///
    /// // At c = 0: λ₀₂ = n(n+1) = 6 and S₀₂(x) = P₂(x) = (3x² − 1)/2
    /// let (kind, normalization) = (SpheroidalKind::Prolate, SpectralNormalization::Standard);
    /// let plan = SpheroidalPlan::new(kind, 0, 2, 0.0_f64, normalization, Default::default())?;
    /// assert!((plan.characteristic_value()? - 6.0).abs() < 1e-12);
    /// assert!((plan.angular(0.5_f64)?.try_value()? + 0.125).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        kind: SpheroidalKind,
        m: u32,
        n: u32,
        c: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            kind,
            m,
            n,
            c,
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
    #[allow(clippy::too_many_arguments)]
    pub fn build<'scope, 'cap>(
        kind: SpheroidalKind,
        m: u32,
        n: u32,
        c: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        validate(kind, m, n, c)?;
        scoped(scope, options.iteration_budget, |context| {
            Self::prepare(
                kind,
                m,
                n,
                c,
                normalization,
                options,
                scioxide_core::argument!("options.terms.maximum"),
                context,
            )
        })
    }

    /// Computes the coefficients and recurrence factors of validated parameters inside an open
    /// epoch.
    ///
    /// `mode` names the argument reported when the mode lies outside the truncation bounds.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        kind: SpheroidalKind,
        m: u32,
        n: u32,
        c: S,
        normalization: SpectralNormalization,
        options: SpectralOptions<S>,
        mode: Argument,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let q = squared_parameter(kind, c)?;
        let offset = (n - m) % 2;
        let index = ((n - m) / 2) as usize;
        let mut spectrum = spectral::prepare(
            index,
            q == 0.0,
            options,
            mode,
            context,
            |k| {
                let l = u64::from(m) + u64::from(offset) + 2 * k as u64;
                let lf = l as f64;
                Ok((
                    lf * (lf + 1.0) + q * square_diagonal(l, m),
                    q * x_coefficient(l, m) * x_coefficient(l + 1, m),
                ))
            },
            |k| {
                let l = u64::from(m) + u64::from(offset) + 2 * k as u64;
                let x2 = square_diagonal(l, m);
                let xbound = if (q >= 0.0) == (m == 0) { 0.5 } else { x2 };
                let off = q.abs() * 0.25_f64.max(x_coefficient(l, m) * x_coefficient(l + 1, m));
                Ok((l as f64 * (l as f64 + 1.0) + q * xbound, off))
            },
        )?;
        if q < 0.0 && index % 2 == 1 {
            spectrum.coefficients.iter_mut().for_each(|v| *v = v.neg());
        }
        // The shared preparation cap already covers these at most 2*N
        // scaled pairs. Build them once for normalization, angular and
        // radial recurrences, charging every prepared step.
        let count = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_mul(
                spectrum.coefficients.len() - 1,
                2,
                scioxide_core::quantity!("degree"),
            )?,
            offset as usize,
            scioxide_core::quantity!("degree"),
        )?;
        let mut steps = scioxide_core::backend::try_filled(
            count,
            (Scaled::ZERO, Scaled::ZERO),
            scioxide_core::quantity!("legendre_steps"),
        )?;
        for (i, pair) in steps.iter_mut().enumerate() {
            step(context)?;
            let degree = scioxide_core::backend::checked_add(
                m as usize,
                i + 1,
                scioxide_core::quantity!("degree"),
            )?;
            let (a, b) = crate::orthogonal_angular::harmonics::normalized_step(degree, m as usize)?;
            *pair = (scalar(a)?, scalar(b)?);
        }
        let diagonal = crate::orthogonal_angular::harmonics::normalized_associated(
            m as usize,
            m as usize,
            0.0,
            1.0,
            &mut || step(context),
        )?
        .value
        .mul(scalar((2.0 * core::f64::consts::PI).sqrt())?)?;
        let angular_scale = if normalization == SpectralNormalization::UnitL2 {
            scalar(1.0)?
        } else {
            let mut value = scalar(2.0 / (2.0 * f64::from(n) + 1.0))?;
            for k in (u64::from(n - m) + 1)..=(u64::from(n) + u64::from(m)) {
                step(context)?;
                value = value.mul(scalar(k as f64)?)?;
            }
            value.sqrt()?
        };
        let mut result = Self {
            kind,
            m,
            n,
            c,
            options,
            normalization,
            spectrum,
            steps,
            diagonal,
            angular_scale,
            eta: 0.0,
            center_derivative: offset == 1,
            denominator: Scaled::ZERO,
            denominator_error: Scaled::ZERO,
        };
        let mut condition = f64::INFINITY;
        for eta in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let derivative = eta == 0.0 && offset == 1;
            let (value, error) = result.normalizer(eta, derivative, context)?;
            if value.is_zero() {
                continue;
            }
            let ratio = error.div(value.abs())?.round()?;
            if ratio < condition {
                result.eta = eta;
                result.center_derivative = derivative;
                result.denominator = value;
                result.denominator_error = error;
                condition = ratio;
            }
        }
        if result.denominator.is_zero() {
            return Err(instability("no resolved spheroidal product normalization"));
        }
        result.spectrum.diagnostics.work = spectral::used(context)?;
        Ok(result)
    }

    fn normalizer(
        &self,
        x: f64,
        derivative: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(Scaled<f64>, Scaled<f64>), SciOxideError> {
        let mut basis = Basis::new(self.m, x, self.diagonal, &self.steps)?;
        let mut sum = ScaledAccumulator::ZERO;
        let mut error = Scaled::ZERO;
        for k in 0..self.spectrum.coefficients.len() {
            let degree = u64::from(self.m) + u64::from((self.n - self.m) % 2) + 2 * k as u64;
            while (basis.l as u64) < degree {
                basis.advance(context)?;
            }
            step(context)?;
            let v = if derivative {
                basis.derivative.real()
            } else {
                basis.current.real()
            };
            sum.add_product(self.spectrum.coefficients[k], v)?;
            error = error.add(self.spectrum.errors[k].mul(v.abs())?)?;
            error = error.add(
                self.spectrum.coefficients[k]
                    .abs()
                    .mul(v.abs())?
                    .mul(scalar(16.0 * f64::EPSILON * (1.0 + degree as f64))?)?,
            )?;
        }
        Ok((sum.total()?, error))
    }
    /// Returns the angular normalization chosen at preparation.
    pub const fn normalization(&self) -> SpectralNormalization {
        self.normalization
    }

    /// Returns prolate or oblate.
    pub const fn kind(&self) -> SpheroidalKind {
        self.kind
    }
    /// Returns the order `m`.
    pub const fn order(&self) -> u32 {
        self.m
    }
    /// Returns the degree `n`.
    pub const fn degree(&self) -> u32 {
        self.n
    }
    /// Returns the size parameter `c`.
    pub const fn parameter(&self) -> S {
        self.c
    }
    /// Returns the characteristic value `λ_mn = λ_n^m(γ²) + γ²` of the equation in
    /// [`SpheroidalPlan`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value exceeds the range of the scalar type.
    pub fn characteristic_value(&self) -> Result<S, SciOxideError> {
        let value = S::try_from_f64_lossy(self.spectrum.characteristic)?;
        if !value.is_finite() {
            return Err(instability(
                "spheroidal characteristic exceeds scalar range",
            ));
        }
        Ok(value)
    }
    /// Returns the convergence diagnostics of preparation.
    pub const fn diagnostics(&self) -> SpectralDiagnostics {
        self.spectrum.diagnostics
    }
    /// Returns the number of expansion coefficients kept.
    pub fn coefficient_count(&self) -> usize {
        self.spectrum.coefficients.len()
    }
    /// Returns the angular coordinate `η` at which the radial product expansion is normalized.
    ///
    /// Preparation picks the best-conditioned of `0, 0.25, 0.5, 0.75, 0.9` and `1`.
    pub const fn radial_normalization_point(&self) -> f64 {
        self.eta
    }
    /// Returns the coefficient `d_k` of the angular function in orthonormal Ferrers functions,
    /// as a scaled value.
    ///
    /// The basis function of index `k` is the Ferrers function `P_(m+2k+p)^m` normalized to unit
    /// `L²` norm on `[−1, 1]`, with `p = (n − m) mod 2`. The coefficients have unit Euclidean
    /// norm; the factor of the chosen [`SpectralNormalization`] is applied on evaluation.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `k ≥ coefficient_count()`.
    pub fn coefficient(&self, k: usize) -> Result<ScaledValue<S>, SciOxideError> {
        let value = *self.spectrum.coefficients.get(k).ok_or_else(|| {
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("coefficients"),
                k,
                self.coefficient_count(),
            )
        })?;
        ScaledValue::from_parts(
            S::try_from_f64_lossy(value.significand())?,
            value.exponent(),
        )
    }
    fn angular_evaluation<V: NumericScalar<Real = f64>>(
        &self,
        x: V,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation<V>, SciOxideError> {
        let left = ScaledValue::<V>::from_value(V::from_components(
            1.0 - x.real_component(),
            -x.imaginary_component(),
        ))?;
        let right = ScaledValue::<V>::from_value(V::from_components(
            1.0 + x.real_component(),
            x.imaginary_component(),
        ))?;
        let root = product(left.sqrt()?, right.sqrt()?)?;
        let base = power(root, self.m, context)?;
        let base_derivative = if self.m == 0 {
            Some(ScaledValue::<V>::ZERO)
        } else if root.is_zero() {
            if self.m == 1 {
                None
            } else if self.m == 2 {
                Some(ScaledValue::<V>::from_value(x * V::from_real(-2.0))?)
            } else {
                Some(ScaledValue::<V>::ZERO)
            }
        } else {
            Some(zscale(
                product(
                    ScaledValue::<V>::from_value(-x)?,
                    power(root, self.m - 1, context)?,
                )?
                .div(root)?,
                scalar(f64::from(self.m))?,
            )?)
        };
        let mut basis = Basis::new(self.m, x, self.diagonal, &self.steps)?;
        let (mut value, mut derivative) = (ScaledValue::<V>::ZERO, ScaledValue::<V>::ZERO);
        let (mut error, mut derivative_error) = (Scaled::ZERO, Scaled::ZERO);
        let mut tails = [Scaled::ZERO; 4];
        let mut dtails = tails;
        for k in 0..self.coefficient_count() {
            let degree = u64::from(self.m) + u64::from((self.n - self.m) % 2) + 2 * k as u64;
            while (basis.l as u64) < degree {
                basis.advance(context)?;
            }
            step(context)?;
            let coefficient = self.spectrum.coefficients[k];
            let term = zscale(basis.current, coefficient)?;
            let dterm = zscale(basis.derivative, coefficient)?;
            value = value.add(term)?;
            derivative = derivative.add(dterm)?;
            let envelope = norm(basis.current)?.add(norm(basis.previous)?)?;
            let denvelope = norm(basis.derivative)?.add(norm(basis.previous_derivative)?)?;
            let rounding = scalar(32.0 * f64::EPSILON * (1.0 + degree as f64).powi(2))?;
            error = error
                .add(self.spectrum.errors[k].mul(norm(basis.current)?)?)?
                .add(coefficient.abs().mul(envelope)?.mul(rounding)?)?;
            derivative_error = derivative_error
                .add(self.spectrum.errors[k].mul(norm(basis.derivative)?)?)?
                .add(coefficient.abs().mul(denvelope)?.mul(rounding)?)?;
            tails[k % 4] = norm(term)?;
            dtails[k % 4] = norm(dterm)?;
        }
        if self.spectrum.tail_ratio > 0.0 {
            let rho = if x.imaginary_component() == 0.0 && x.real_component().abs() <= 1.0 {
                1.0
            } else {
                let x = C::new(x.real_component(), x.imaginary_component());
                let root = (x * x - real(1.0)).sqrt();
                (x + root).norm().max((x - root).norm())
            };
            let ratio = self.spectrum.tail_ratio * rho * rho;
            if !ratio.is_finite() || ratio >= 1.0 {
                return Err(instability(
                    "angular coordinate requires more spheroidal coefficients",
                ));
            }
            let largest = |v: &[Scaled<f64>]| {
                v.iter()
                    .copied()
                    .fold(Scaled::ZERO, |a, b| if a.abs_le(b) { b } else { a })
            };
            error = error.add(largest(&tails).mul(scalar(ratio / (1.0 - ratio))?)?)?;
            derivative_error =
                derivative_error.add(largest(&dtails).mul(scalar(ratio / (1.0 - ratio))?)?)?;
        }
        let value_before = value;
        value = zscale(product(base, value)?, self.angular_scale)?;
        let derivative = base_derivative
            .map(|d| {
                zscale(
                    product(d, value_before)?.add(product(base, derivative)?)?,
                    self.angular_scale,
                )
            })
            .transpose()?;
        let derivative_error = if let Some(d) = base_derivative {
            error
                .mul(norm(d)?)?
                .add(derivative_error.mul(norm(base)?)?)?
                .mul(self.angular_scale)?
        } else {
            Scaled::ZERO
        };
        error = error.mul(norm(base)?)?.mul(self.angular_scale)?;
        Ok(Evaluation {
            value,
            derivative,
            error,
            derivative_error,
        })
    }
    /// Evaluates the angular spheroidal function `S_mn(x)` and its derivative with respect to `x`.
    ///
    /// Real `x` must lie in `[−1, 1]`. Complex `x` gives the analytic continuation, with the
    /// principal square root in the factor `(1 − x²)^(m/2)` of the Ferrers functions. At
    /// `x = ±1` the value is finite; where the derivative is singular there, the result omits
    /// it. The normalization and the sign are those chosen in [`Self::new`]. The expansion is
    /// summed in scaled arithmetic with the stored recurrence factors, at one work unit per
    /// coefficient and degree step.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` has a NaN or infinite component.
    /// - `DomainError` (`OutsideDomain`) if `x` is real and `|x| > 1`.
    /// - `NumericalInstability` if the scaled sum breaks down.
    /// - `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the
    ///   budget.
    pub fn angular<T: NumericScalar<Real = S>>(
        &self,
        x: T,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        self.angular_scoped(x, ExecutionContext::sequential_native())
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
        x: T,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        let point = angular_point(x)?;
        scoped(scope, self.options.iteration_budget, |context| {
            self.angular_run(point, context)
        })
    }

    /// Evaluates the angular function at a validated coordinate inside an open epoch.
    fn angular_run<T: NumericScalar<Real = S>>(
        &self,
        point: C,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        if T::IS_REAL {
            report(
                self.angular_evaluation(point.re, context),
                algorithm_component!("spheroidal_angular"),
                context,
            )
        } else {
            report(
                self.angular_evaluation(point, context),
                algorithm_component!("spheroidal_angular"),
                context,
            )
        }
    }
}

/// Rejects parameters that no preparation can serve, before a preparation epoch opens.
fn validate<S: RealScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: S,
) -> Result<(), SciOxideError> {
    if n < m {
        return Err(invalid(
            scioxide_core::argument!("n"),
            "spheroidal degree must not be smaller than order",
        ));
    }
    if !c.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "c must be finite",
        ));
    }
    squared_parameter(kind, c).map(|_| ())
}

/// Returns `γ² = s c²` of the spheroidal equation in `f64`.
fn squared_parameter<S: RealScalar>(kind: SpheroidalKind, c: S) -> Result<f64, SciOxideError> {
    let parameter = c.try_to_f64_exact()?;
    let sign = match kind {
        SpheroidalKind::Prolate => 1.0,
        SpheroidalKind::Oblate => -1.0,
    };
    crate::support::finite(parameter * parameter * sign, || {
        instability("spheroidal squared parameter exceeds working range")
    })
}

/// Widens a finite angular coordinate and rejects a real one outside `[−1, 1]`, before an
/// evaluation epoch opens.
fn angular_point<T: NumericScalar>(x: T) -> Result<C, SciOxideError> {
    let point = spectral::coordinate(x, "coordinate must be finite")?;
    if T::IS_REAL && point.re.abs() > 1.0 {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "real angular coordinate must be in [-1,1]",
        ));
    }
    Ok(point)
}

/// Computes the characteristic value `λ_mn` of the prolate or oblate spheroidal wave equation.
///
/// `λ_mn` is the value for which the spheroidal wave equation
/// `(1 − x²) w″ − 2x w′ + (λ_mn − s c² x² − m²/(1 − x²)) w = 0` has a solution bounded at
/// `x = ±1`, with order `m ≥ 0`, degree `n ≥ m`, real size parameter `c`, and `s = +1` for
/// [`SpheroidalKind::Prolate`] or `s = −1` for [`SpheroidalKind::Oblate`]. In the notation of
/// DLMF 30.2.1, `γ² = s c²` and `λ_mn = λ_n^m(γ²) + γ²`; `λ_mn = n(n + 1)` at `c = 0`. It is the
/// characteristic value of a [`SpheroidalPlan`] with default [`SpectralOptions`]. Each call
/// prepares one plan, as described for [`SpheroidalPlan`]; prepare the plan to also evaluate
/// `S_mn` and `R_mn^(j)`.
///
/// # Errors
///
/// - `NonFinite` if `c` is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) with argument `n` if `n < m`, or if the mode lies outside
///   the default truncation bounds: `n − m ≥ 496`, or `n − m ≥ 512` at `c = 0`.
/// - `NumericalInstability` (`LossOfSignificance`) if `c²` or the matrix entries leave the
///   `f64` range, if no normalization point of the radial expansion is resolved, or if the
///   value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation needs more than 2,097,152 work
///   units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. Preparation refines the truncation until two successive
/// characteristic values agree to `1e-13`, absolutely or relatively; see [`SpheroidalPlan`]
/// and the [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::{SpheroidalKind, spheroidal_characteristic_value};
///
/// // λ₀₂ = n(n + 1) = 6 at c = 0
/// let lambda = spheroidal_characteristic_value(SpheroidalKind::Prolate, 0, 2, 0.0_f64)?;
/// assert!((lambda - 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §30.2 and §30.8. <https://dlmf.nist.gov/30>
/// - C. Flammer, *Spheroidal Wave Functions*, Stanford University Press, 1957.
pub fn spheroidal_characteristic_value<S: RealScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: S,
) -> Result<S, SciOxideError> {
    prepared_call(kind, m, n, c, SpectralOptions::default(), |plan, _| {
        plan.characteristic_value()
    })
}

/// Computes the angular spheroidal wave function `S_mn(x)`.
///
/// `S_mn` is the solution of the spheroidal wave equation of [`SpheroidalPlan`] that is bounded
/// on `[−1, 1]`, with order `m ≥ 0`, degree `n ≥ m`, real size parameter `c`, prolate or oblate
/// by `kind`. Real `x` lies in `[−1, 1]`; complex `x` gives the analytic continuation, with the
/// principal square root in the factor `(1 − x²)^(m/2)`. The sign and the normalization
/// `∫₋₁¹ S_mn(x)² dx = 2(n + m)!/((2n + 1)(n − m)!)` are those of the Ferrers function
/// `P_n^m(x)` (DLMF §30.4, [`SpectralNormalization::Standard`]), to which `S_mn` reduces at
/// `c = 0`, including the Condon–Shortley phase. Each call prepares one [`SpheroidalPlan`] with
/// default [`SpectralOptions`] and sums its expansion once, at one work unit per coefficient
/// and degree step; preparation and evaluation share one budget of 2,097,152 work units.
/// Prepare a [`SpheroidalPlan`] to evaluate many points or to choose another normalization.
///
/// # Errors
///
/// - `NonFinite` if `c` is NaN or infinite, or if `x` has a NaN or infinite component.
/// - `InvalidInput` (`OutOfRange`) with argument `n` if `n < m`, or if the mode lies outside
///   the default truncation bounds: `n − m ≥ 496`, or `n − m ≥ 512` at `c = 0`.
/// - `DomainError` (`OutsideDomain`) with argument `x` if `x` is real and `|x| > 1`.
/// - `NumericalInstability` (`LossOfSignificance`) if `c²` or the matrix entries leave the
///   `f64` range, if no normalization point of the radial expansion is resolved, or if the
///   scaled sum breaks down; (`Overflow`) if the value exceeds the range of the scalar type.
/// - `ConvergenceFailure` (`TruncationLimit`) if two successive truncations do not agree before
///   256 coefficients, and (`IterationLimit`) if preparation and evaluation together need more
///   than 2,097,152 work units.
///
/// # Accuracy
///
/// No fixed ulp bound is stated. The truncation is refined to the tolerance `1e-13`, and
/// complex `x` can amplify the error. See [`SpheroidalPlan`] and the
/// [crate accuracy table](crate#accuracy).
///
/// # Examples
///
/// ```
/// use scioxide_special::{SpheroidalKind, spheroidal_angular};
///
/// // S₀₂(x) = P₂(x) = (3x² − 1)/2 at c = 0
/// let s = spheroidal_angular(SpheroidalKind::Prolate, 0, 2, 0.0, 0.5_f64)?;
/// assert!((s + 0.125).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §30.4 and §30.8. <https://dlmf.nist.gov/30>
/// - C. Flammer, *Spheroidal Wave Functions*, Stanford University Press, 1957.
pub fn spheroidal_angular<T: NumericScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: T::Real,
    x: T,
) -> Result<T, SciOxideError> {
    let point = angular_point(x)?;
    prepared_call(
        kind,
        m,
        n,
        c,
        SpectralOptions::default(),
        |plan, context| plan.angular_run(point, context),
    )?
    .try_value()
}

// One allowance covers preparation, its nested eigensolver, and evaluation.
fn prepared_call<S: RealScalar, V>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: S,
    options: SpectralOptions<S>,
    run: impl FnOnce(&SpheroidalPlan<S>, &mut SolveContext<'_, '_>) -> Result<V, SciOxideError>,
) -> Result<V, SciOxideError> {
    validate(kind, m, n, c)?;
    scoped(
        ExecutionContext::sequential_native(),
        options.iteration_budget,
        |context| {
            let plan = SpheroidalPlan::prepare(
                kind,
                m,
                n,
                c,
                SpectralNormalization::Standard,
                options,
                scioxide_core::argument!("n"),
                context,
            )?;
            run(&plan, context)
        },
    )
}
pub(crate) fn report<T: NumericScalar, V: NumericScalar<Real = f64>>(
    result: Result<Evaluation<V>, SciOxideError>,
    family: AlgorithmComponent,
    context: &mut SolveContext<'_, '_>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    let mut derivative = true;
    let estimate = result.and_then(|r| {
        derivative = r.derivative.is_some();
        let d = r.derivative.unwrap_or(ScaledValue::ZERO);
        let (v, e) = factor_common_scale([
            r.value.real(),
            r.value.imaginary(),
            d.real(),
            d.imaginary(),
            r.error,
            r.derivative_error,
        ])?;
        Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: C::new(v[0].round()?, v[1].round()?),
                d: C::new(v[2].round()?, v[3].round()?),
            },
            scale: Jet::constant(real(e as f64 * core::f64::consts::LN_2)),
            error: v[4].round()?,
            derivative_error: Some(v[5].round()?),
            method: algorithm_component!("scaled_spectral_expansion"),
        }))
    });
    spectral::report(estimate, derivative, family, context)
}

#[path = "spheroidal/radial.rs"]
mod radial;
pub use radial::spheroidal_radial;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Budget;

    #[test]
    fn one_shot_allowance_covers_preparation_and_evaluation() {
        let kind = SpheroidalKind::Prolate;
        let plan = |options| {
            SpheroidalPlan::new(
                kind,
                0,
                2,
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
        let call = |options| {
            prepared_call(kind, 0, 2, 1.0_f64, options, |plan, context| {
                plan.angular_run::<f64>(real(0.5), context)
                    .and_then(SpecialFunctionResult::try_value)
            })
        };
        let _ = crate::test_support::exhausted_counts(&call(options).unwrap_err());
        assert_eq!(
            call(SpectralOptions::default()).unwrap().to_bits(),
            reference
                .angular(0.5_f64)
                .unwrap()
                .try_value()
                .unwrap()
                .to_bits()
        );
    }
}
