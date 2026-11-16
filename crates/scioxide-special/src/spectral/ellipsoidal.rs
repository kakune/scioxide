//! Prepared Lamé functions of the first and second kind and their normalization.
use crate::elliptic_family::elliptic_kernel::{self as kernel};
use crate::result::{Evaluation, Forms, Publish};
use crate::support::ELLIPSOIDAL_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use crate::support::widen_numeric as widen;
use crate::support::{C, real};
use crate::util::scoped;
use crate::{EllipsoidalOptions, SpecialFunctionResult};
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::AlgorithmComponent;
use scioxide_core::Scaled;
use scioxide_core::algorithm_component;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    ExecutionContext, NumericScalar, RealScalar, SciOxideError, SolveContext, SolveScope, Tolerance,
};

mod quadrature;
#[cfg(test)]
mod tests;

type Report<T> = SpecialFunctionResult<T>;

/// Lists the four classes of Lamé functions of degree `n`, in the order of the index `p`.
///
/// With `r = ⌊n/2⌋` the classes have `r + 1`, `n − r`, `n − r` and `r` members, `2n + 1` in
/// all. `h` and `k` are the semi-focal distances of the ellipsoidal coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LameClass {
    /// A polynomial in `s` of degree `n`.
    K,
    /// `√|s² − h²|` times a polynomial of degree `n − 1`.
    L,
    /// `√|s² − k²|` times a polynomial of degree `n − 1`.
    M,
    /// `√|s² − h²| √|s² − k²|` times a polynomial of degree `n − 2`.
    N,
}
impl LameClass {
    fn factors(self) -> (bool, bool) {
        (
            matches!(self, Self::L | Self::N),
            matches!(self, Self::M | Self::N),
        )
    }
}

/// Selects the signs of the focal factors `√|s² − h²|` and `√|s² − k²|` of a Lamé function.
///
/// The default takes both factors positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EllipsoidalSigns {
    negative_h: bool,
    negative_k: bool,
}
impl EllipsoidalSigns {
    /// Creates sign choices for the focal factors.
    ///
    /// `negative_h` and `negative_k` multiply the factor `√|s² − h²|` or `√|s² − k²|` by `−1`
    /// when the [`LameClass`] of the function contains it. For complex `s` the signs multiply the
    /// analytic continuation of the factor.
    pub const fn new(negative_h: bool, negative_k: bool) -> Self {
        Self {
            negative_h,
            negative_k,
        }
    }
    fn multiplier(self, class: LameClass) -> f64 {
        let (h, k) = class.factors();
        if (h && self.negative_h) ^ (k && self.negative_k) {
            -1.0
        } else {
            1.0
        }
    }
}

/// Holds the coefficients of one Lamé function `E_n^p` for a triaxial ellipsoidal coordinate system.
///
/// Ellipsoidal harmonics separate Laplace's equation in ellipsoidal coordinates with semi-focal
/// distances `0 < h < k`. The Lamé function of the first kind `E_n^p(s)`, of degree `n ≥ 0`
/// and index `1 ≤ p ≤ 2n + 1`, is the `p`-th solution of Lamé's equation that is a polynomial
/// times focal factors; `p` runs through the classes `K`, `L`, `M`, `N` of [`LameClass`] in
/// that order, and through increasing separation constants within a class. The functions are
/// monic: the leading coefficient for large `s` is one. The function of the second kind
/// `F_n^p` and the normalization constant `γ_n^p` follow Dassios (2012).
///
/// Preparation computes in `f64` for both scalar types. It solves a symmetric tridiagonal
/// eigenproblem for the Chebyshev coefficients of `E_n^p` in `s/h`, then recomputes the
/// eigenvector from its three-term recurrence toward the largest component. This takes
/// `O(n²)` work and temporary memory; the plan keeps `O(n)` coefficients. For `n > 0` it also
/// prepares a second expansion in `√(k² − s²)`, used for real `s` with `h < |s| < k`. The plan
/// keeps its [`EllipsoidalOptions`]: preparation and each evaluation have their own work
/// budget, and the relative tolerance steers the quadrature of [`Self::second_kind`] and
/// [`Self::normalization`]. See
/// [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Accuracy
///
/// No fixed ulp bound is stated. [`Self::spectral_residual`] reports the residual of the
/// eigenproblem, and each evaluation returns an error estimate derived from the coefficient
/// error, the condition of the Chebyshev sum and, for quadrature, the tolerance. The estimates
/// are not guaranteed bounds. See the [crate accuracy table](crate#accuracy).
///
/// # References
///
/// - G. Dassios, *Ellipsoidal Harmonics: Theory and Applications*, Cambridge University
///   Press, 2012.
/// - NIST DLMF, Chapter 29. <https://dlmf.nist.gov/29>
#[derive(Debug, Clone)]
pub struct EllipsoidalHarmonicPlan<R: RealScalar> {
    h2: R,
    k2: R,
    options: EllipsoidalOptions<R>,
    degree: usize,
    order: usize,
    class: LameClass,
    k: f64,
    ratio: f64,
    coefficients: Vec<f64>,
    scale: Z,
    eigenvalue: f64,
    residual: f64,
    coefficient_error: f64,
    counts: scioxide_core::SolveCounts,
    complementary: Option<AngularExpansion>,
}

#[derive(Debug, Clone)]
struct AngularExpansion {
    class: LameClass,
    ratio: f64,
    coefficients: Vec<f64>,
    scale: Z,
    coefficient_error: f64,
}

fn power(mut x: Z, mut n: usize) -> Result<Z, SciOxideError> {
    let mut y = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?;
    while n > 0 {
        if n & 1 != 0 {
            y = y.mul(x)?;
        }
        n >>= 1;
        if n > 0 {
            x = x.mul(x)?;
        }
    }
    Ok(y)
}

impl<R: RealScalar> EllipsoidalHarmonicPlan<R> {
    /// Prepares the Lamé function `E_n^p` for the squared semi-focal distances `h2 = h²` and
    /// `k2 = k²`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `h2` or `k2` is NaN or infinite.
    /// - `DomainError` (`OutsideDomain`) unless `0 < h² < k²`.
    /// - `InvalidInput` (`OutOfRange`) unless `1 ≤ p ≤ 2n + 1`.
    /// - `SizeOverflow` if `n` is so large that the eigenproblem size overflows.
    /// - `NumericalInstability` (`LossOfSignificance`) if the eigenvector or its coefficients are
    ///   not resolved in `f64`.
    /// - `ConvergenceFailure` (`IterationLimit`) if preparation needs more work than the budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_special::EllipsoidalHarmonicPlan;
    ///
    /// // E₀¹(s) = 1 and E₁¹(s) = s
    /// let constant = EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 0, 1, Default::default())?;
    /// let linear = EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 1, 1, Default::default())?;
    /// assert!((constant.first_kind(1.5_f64, Default::default())?.try_value()? - 1.0).abs() < 1e-12);
    /// assert!((linear.first_kind(1.5_f64, Default::default())?.try_value()? - 1.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        h2: R,
        k2: R,
        n: usize,
        p: usize,
        options: EllipsoidalOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::build(h2, k2, n, p, options, ExecutionContext::sequential_native())
    }

    /// Prepares a plan as [`Self::new`] does, inside `scope`.
    ///
    /// The work budget covers the eigenproblem, including the eigenvalue solver. See
    /// [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn build<'scope, 'cap>(
        h2: R,
        k2: R,
        n: usize,
        p: usize,
        options: EllipsoidalOptions<R>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        scoped(scope, options.iteration_budget(), |child| {
            Self::prepare(h2, k2, n, p, options, child)
        })
    }

    /// Prepares the plan and, for `n > 0`, its complementary expansion.
    fn prepare(
        h2: R,
        k2: R,
        n: usize,
        p: usize,
        options: EllipsoidalOptions<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let mut plan = Self::prepare_one(h2, k2, n, p, false, options, context)?;
        if n > 0 {
            let (beta, gamma) = plan.class.factors();
            let alpha = (n - usize::from(beta) - usize::from(gamma)) % 2;
            let complement_class = match (beta, alpha != 0) {
                (false, false) => LameClass::K,
                (true, false) => LameClass::L,
                (false, true) => LameClass::M,
                (true, true) => LameClass::N,
            };
            let sizes = [n / 2 + 1, n - n / 2, n - n / 2, n / 2];
            let ordinal = |class| match class {
                LameClass::K => 0,
                LameClass::L => 1,
                LameClass::M => 2,
                LameClass::N => 3,
            };
            let offset = |class| sizes[..ordinal(class)].iter().sum::<usize>();
            let index = p - 1 - offset(plan.class);
            let complement_order =
                offset(complement_class) + sizes[ordinal(complement_class)] - index;
            let second = Self::prepare_one(h2, k2, n, complement_order, true, options, context)?;
            let sum = (n as f64) * (n as f64 + 1.0);
            if (plan.eigenvalue + second.eigenvalue - sum).abs() > 1e-12 * sum {
                return Err(ERRORS.unstable());
            }
            plan.residual = plan.residual.max(second.residual);
            plan.complementary = Some(AngularExpansion {
                class: second.class,
                ratio: second.ratio,
                coefficients: second.coefficients,
                scale: second.scale,
                coefficient_error: second.coefficient_error,
            });
            plan.counts = context.counts();
        }
        Ok(plan)
    }

    fn prepare_one(
        h2: R,
        k2: R,
        n: usize,
        p: usize,
        complementary: bool,
        options: EllipsoidalOptions<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        context.execution().check()?;
        let h = widen(h2, "spectral parameter h2 must be finite")?.re;
        let k2f = widen(k2, "spectral parameter k2 must be finite")?.re;
        if !(h > 0.0 && h < k2f) {
            return Err(ERRORS.outside(scioxide_core::argument!("h2")));
        }
        let h = if complementary { k2f - h } else { h };
        let total = n
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("lame_orders")))?;
        if p == 0 || p > total {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("p"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Lamé order must be in 1..=2*n+1",
            ));
        }
        let nf = f64::try_from_usize_exact(n)?;
        let ratio = h / k2f;
        if !(ratio > 0.0 && ratio < 1.0) {
            return Err(ERRORS.unstable());
        }
        let r = n / 2;
        let side = n - r;
        let (class, index) = if p <= r + 1 {
            (LameClass::K, p - 1)
        } else if p <= r + 1 + side {
            (LameClass::L, p - r - 2)
        } else if p <= r + 1 + 2 * side {
            (LameClass::M, p - r - 2 - side)
        } else {
            (LameClass::N, p - r - 2 - 2 * side)
        };
        let (sine, k_factor) = class.factors();
        let gamma = usize::from(k_factor);
        let maximum = n - gamma;
        let minimum = if sine { 2 - maximum % 2 } else { maximum % 2 };
        let size = (maximum - minimum) / 2 + 1;
        size.checked_mul(size)
            .and_then(|s| s.checked_mul(core::mem::size_of::<f64>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("eigensystem_bytes"))
            })?;
        // Admit the O(size²) initialization before allocating the eigensystem.
        // Even enormous valid degrees therefore fail promptly under a small
        // caller allowance instead of attempting an enormous allocation first.
        for _ in 0..size * size {
            crate::spectral::step(context)?;
        }
        let mut d = try_filled(size, 0.0, scioxide_core::quantity!("diagonal"))?;
        let mut upper = try_filled(
            size.saturating_sub(1),
            0.0,
            scioxide_core::quantity!("upper_diagonal"),
        )?;
        let mut lower = try_filled(
            size.saturating_sub(1),
            0.0,
            scioxide_core::quantity!("lower_diagonal"),
        )?;
        // Lamé's equation after s=h cos(phi), y=(1-s²/k²)^(gamma/2) v.
        // cos(j phi), or sin(j phi), gives a terminating tridiagonal problem.
        let c = 1.0 + 2.0 * gamma as f64;
        let q = nf * (nf + 1.0);
        for i in 0..size {
            crate::spectral::step(context)?;
            let j = (minimum + 2 * i) as f64;
            d[i] = j * j * (1.0 - ratio * 0.5) + q * ratio * 0.5;
            let up = ((nf - gamma as f64 - j) * (nf + gamma as f64 + j + 1.0)) * (ratio * 0.25);
            let down = (q - 2.0 * gamma as f64 - j * j + c * j) * (ratio * 0.25);
            if j == 1.0 {
                d[i] += if sine { -down } else { down };
            }
            if i + 1 < size {
                lower[i] = if j == 0.0 { 2.0 * up } else { up };
            }
            if i > 0 {
                upper[i - 1] = down;
            }
        }
        let mut off = try_filled(
            size.saturating_sub(1),
            0.0,
            scioxide_core::quantity!("off_diagonal"),
        )?;
        for ((out, a), b) in off.iter_mut().zip(&upper).zip(&lower) {
            *out = a.sqrt() * b.sqrt();
        }
        // The eigensolver's child epoch is bounded by the remaining shared allowance.
        let allowance = match crate::spectral::remaining(context)? {
            Some(0) => {
                return Err(crate::result::exhausted(
                    scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                ));
            }
            Some(left) => scioxide_core::Budget::try_from(left)?,
            None => scioxide_core::Budget::ONE.saturating_add(usize::MAX - 1),
        };
        let eigen_options = scioxide_linalg::HermitianEigenOptions::default()
            .with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(allowance),
            )
            .with_deflation_tolerance(Tolerance::from_machine_epsilon());
        let mut eig = scioxide_linalg::HermitianEigenDecomposition::empty();
        scioxide_linalg::HermitianEigenDecomposition::tridiagonal_into(
            &d,
            &off,
            &eigen_options,
            &mut eig,
            &mut scioxide_linalg::LinalgWorkspace::new(),
            &mut *context,
        )?;
        let eigenvalue = eig.eigenvalues()[index];
        // QL gives an accurate eigenvalue and peak location, but its tiny endpoint
        // eigenvector components need not have good relative accuracy. Recover
        // each tail towards the largest component with the defining recurrence.
        let peak = (0..size)
            .max_by(|&a, &b| {
                eig.eigenvectors().as_slice()[a * size + index]
                    .abs()
                    .total_cmp(&eig.eigenvectors().as_slice()[b * size + index].abs())
            })
            .ok_or_else(|| ERRORS.unstable())?;
        let mut vector = try_filled(size, 0.0, scioxide_core::quantity!("eigenvector"))?;
        vector[0] = 1.0;
        for i in 0..peak {
            crate::spectral::step(context)?;
            let previous = if i == 0 {
                0.0
            } else {
                off[i - 1] * vector[i - 1]
            };
            vector[i + 1] = ((eigenvalue - d[i]) * vector[i] - previous) / off[i];
        }
        let left_peak = vector[peak];
        vector[size - 1] = 1.0;
        for i in (peak + 1..size).rev() {
            crate::spectral::step(context)?;
            let next = if i + 1 == size {
                0.0
            } else {
                off[i] * vector[i + 1]
            };
            vector[i - 1] = ((eigenvalue - d[i]) * vector[i] - next) / off[i - 1];
        }
        let matching = vector[peak] / left_peak;
        for value in &mut vector[..peak] {
            *value *= matching;
        }
        let vector_norm = vector.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
        if !vector_norm.is_finite() || vector_norm == 0.0 {
            return Err(ERRORS.unstable());
        }
        for value in &mut vector {
            *value /= vector_norm;
        }
        let mut coefficients =
            try_filled(maximum + 1, 0.0, scioxide_core::quantity!("coefficients"))?;
        let mut balance = 1.0;
        let mut residual = 0.0_f64;
        let mut smallest_gap = f64::INFINITY;
        for (i, &lambda) in eig.eigenvalues().iter().enumerate() {
            if i != index {
                smallest_gap = smallest_gap.min((lambda - eigenvalue).abs());
            }
        }
        for i in 0..size {
            crate::spectral::step(context)?;
            let v = vector[i];
            let mut defect = (d[i] - eigenvalue) * v;
            if i > 0 {
                defect += off[i - 1] * vector[i - 1];
            }
            if i + 1 < size {
                defect += off[i] * vector[i + 1];
            }
            residual = residual.hypot(defect);
            if i > 0 {
                balance *= (lower[i - 1] / upper[i - 1]).sqrt();
            }
            let j = minimum + 2 * i;
            coefficients[j - usize::from(sine)] = v * balance;
        }
        if sine {
            coefficients.pop();
        }
        let leading = *coefficients.last().ok_or_else(|| ERRORS.unstable())?;
        if leading == 0.0 || !leading.is_finite() || coefficients.iter().any(|x| !x.is_finite()) {
            return Err(ERRORS.unstable());
        }
        let lead_power = maximum.saturating_sub(1);
        let leading_scale =
            Scaled::from_value(leading)?.shifted(i64::try_from(lead_power).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("leading_exponent"))
            })?)?;
        let k = k2f.sqrt();
        let scale = power(
            Z::from_value(scioxide_core::Complex::new(h.sqrt(), 0.0))?,
            n,
        )?
        .div(Z::from_components(leading_scale, Scaled::ZERO))?;
        let relative_residual = residual / q.max(1.0);
        let coefficient_error = 16.0 * f64::EPSILON * (n + 1) as f64 + residual / smallest_gap;
        if !coefficient_error.is_finite() || relative_residual > 1e-12 {
            return Err(ERRORS.unstable());
        }
        Ok(Self {
            options,
            h2,
            k2,
            degree: n,
            order: p,
            class,
            k,
            ratio,
            coefficients,
            scale,
            eigenvalue,
            residual: relative_residual,
            coefficient_error,
            counts: context.counts(),
            complementary: None,
        })
    }

    /// Returns the squared semi-focal distances `(h², k²)`.
    pub const fn geometry(&self) -> (R, R) {
        (self.h2, self.k2)
    }
    /// Returns the degree `n`.
    pub const fn degree(&self) -> usize {
        self.degree
    }
    /// Returns the index `p`, counted from one through the classes `K`, `L`, `M`, `N`.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the class of the function.
    pub const fn class(&self) -> LameClass {
        self.class
    }
    /// Returns the separation constant of Lamé's equation divided by `k²`, in `f64`.
    pub const fn eigenvalue_over_k_squared(&self) -> f64 {
        self.eigenvalue
    }
    /// Returns the relative residual of the symmetrized eigenproblem.
    ///
    /// It is a diagnostic, not an error bound.
    pub const fn spectral_residual(&self) -> f64 {
        self.residual
    }
    /// Returns the work counts of preparation, including the eigenvalue solver.
    pub const fn preparation_accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }

    /// Evaluates the Lamé function of the first kind `E_n^p(s)` and its derivative with respect
    /// to `s`.
    ///
    /// `s` is any finite real or complex number. For real `s` the focal factors are
    /// `√|s² − h²|` and `√|s² − k²|`, with the signs chosen by `signs`. For complex `s` they are
    /// continued analytically, and the sign of a zero imaginary part selects the side of the
    /// cut. At `s = ±h` or `s = ±k` the value is finite; where the derivative is singular there,
    /// the result omits it. The Chebyshev series is summed by Clenshaw's recurrence, at one work
    /// unit per coefficient.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `s` has a NaN or infinite component.
    /// - `NumericalInstability` if the scaled evaluation breaks down.
    /// - `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the
    ///   budget.
    pub fn first_kind<T: NumericScalar<Real = R>>(
        &self,
        s: T,
        signs: EllipsoidalSigns,
    ) -> Result<Report<T>, SciOxideError> {
        self.first_run(s, signs, None)?.publish()
    }
    /// Evaluates [`Self::first_kind`] inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::first_kind`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn first_kind_scoped<'scope, 'cap, T: NumericScalar<Real = R>>(
        &self,
        s: T,
        signs: EllipsoidalSigns,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Report<T>, SciOxideError> {
        scoped(scope, self.options.iteration_budget(), |cx| {
            self.first_run(s, signs, Some(cx))
        })?
        .publish()
    }
    fn first_run<T: NumericScalar<Real = R>>(
        &self,
        s: T,
        signs: EllipsoidalSigns,
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<LameEvaluation<T>, SciOxideError> {
        let s = widen(s, "spectral parameter s must be finite")?;
        let mut work = work(self.options, context)?;
        let result = self
            .evaluate(s, crate::support::is_real::<T>(), &mut work)
            .and_then(|mut e| {
                let sign = Z::from_value(scioxide_core::Complex::new(
                    signs.multiplier(self.class),
                    0.0,
                ))?;
                e.value.value = e.value.value.mul(sign)?;
                if let Some(d) = &mut e.derivative {
                    d.value = d.value.mul(sign)?;
                }
                Ok(e)
            });
        report(result, &work, algorithm_component!("lame_chebyshev"))
    }

    /// Evaluates the Lamé function of the second kind `F_n^p(s)` and its derivative with respect
    /// to `s`.
    ///
    /// ```text
    /// F_n^p(s) = (2n+1) E_n^p(s) ∫ₛ^∞ dt / (E_n^p(t)² √(t² − h²) √(t² − k²)),
    /// ```
    ///
    /// which decays like `s^(−n−1)` and gives the exterior ellipsoidal harmonics. `s` is real with
    /// `s > k`, or complex with `|s| > k`; the square roots are principal, continued from the
    /// exterior. The integral is computed by adaptive quadrature to the relative tolerance of the
    /// plan's options.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `s` has a NaN or infinite component.
    /// - `DomainError` (`OutsideDomain`) if `s` is real and `s ≤ k`, or complex and `|s| ≤ k`.
    /// - `NumericalInstability` if the evaluation breaks down.
    /// - `ConvergenceFailure` (`IterationLimit`) if the quadrature needs more work than the
    ///   budget.
    pub fn second_kind<T: NumericScalar<Real = R>>(
        &self,
        s: T,
    ) -> Result<Report<T>, SciOxideError> {
        self.second_run(s, None)?.publish()
    }
    /// Evaluates [`Self::second_kind`] inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::second_kind`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn second_kind_scoped<'scope, 'cap, T: NumericScalar<Real = R>>(
        &self,
        s: T,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Report<T>, SciOxideError> {
        scoped(scope, self.options.iteration_budget(), |cx| {
            self.second_run(s, Some(cx))
        })?
        .publish()
    }
    fn second_run<T: NumericScalar<Real = R>>(
        &self,
        s: T,
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<LameEvaluation<T>, SciOxideError> {
        let s = widen(s, "spectral parameter s must be finite")?;
        if (crate::support::is_real::<T>() && s.re <= self.k) || s.norm() <= self.k {
            return Err(ERRORS.outside(scioxide_core::argument!("s")));
        }
        let mut work = work(self.options, context)?;
        let result = quadrature::second(self, s, &mut work);
        report(
            result,
            &work,
            algorithm_component!("lame_exterior_integral"),
        )
    }

    /// Computes the normalization constant `γ_n^p` of the surface ellipsoidal harmonic.
    ///
    /// ```text
    /// γ_n^p = 8 ∫₀^h ∫ₕ^k E(ν)² E(μ)² (μ² − ν²) / √((h² − ν²)(k² − ν²)(μ² − h²)(k² − μ²)) dμ dν,
    /// ```
    ///
    /// where `E = E_n^p` in the monic normalization of [`Self::new`]. The factor 8 covers the
    /// eight octants of the ellipsoid. The endpoint singularities are integrable; the double
    /// integral is computed by adaptive quadrature after a substitution with Jacobi elliptic
    /// functions, to the relative tolerance of the plan's options. For `n = 0`, `γ₀¹ = 4π`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the evaluation breaks down.
    /// - `ConvergenceFailure` (`IterationLimit`) if the quadrature needs more work than the
    ///   budget.
    pub fn normalization(&self) -> Result<Report<R>, SciOxideError> {
        self.normalization_run(None)?.publish()
    }
    /// Evaluates [`Self::normalization`] inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::normalization`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn normalization_scoped<'scope, 'cap>(
        &self,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Report<R>, SciOxideError> {
        scoped(scope, self.options.iteration_budget(), |cx| {
            self.normalization_run(Some(cx))
        })?
        .publish()
    }
    fn normalization_run(
        &self,
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<LameEvaluation<R>, SciOxideError> {
        let mut work = work(self.options, context)?;
        let result = quadrature::normalization(self, &mut work).map(|value| LameEstimate {
            value,
            derivative: None,
        });
        report(
            result,
            &work,
            algorithm_component!("lame_angular_normalization"),
        )
    }

    // Scaled Clenshaw for T_j or U_j, including its differentiated recurrence.
    // The returned derivative is with respect to s, not to s/k.
    fn evaluate(
        &self,
        s: C,
        real_output: bool,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<LameEstimate, SciOxideError> {
        if real_output
            && s.re.abs() > self.k * self.ratio.sqrt()
            && s.re.abs() < self.k
            && let Some(expansion) = &self.complementary
        {
            let t = ((self.k - s.re.abs()) * (self.k + s.re.abs())).sqrt();
            let mut result = self.evaluate_expansion(
                Z::from_value(scioxide_core::Complex::new(t, 0.0))?,
                true,
                Some(expansion),
                work,
            )?;
            let (beta, gamma) = self.class.factors();
            let alpha = (self.degree - usize::from(beta) - usize::from(gamma)) % 2;
            let polynomial_degree =
                (self.degree - alpha - usize::from(beta) - usize::from(gamma)) / 2;
            let sign =
                if !polynomial_degree.is_multiple_of(2) ^ (alpha != 0 && s.re.is_sign_negative()) {
                    -1.0
                } else {
                    1.0
                };
            result.value.value = result
                .value
                .value
                .mul(Z::from_value(scioxide_core::Complex::new(sign, 0.0))?)?;
            if let Some(d) = &mut result.derivative {
                d.value = d.value.mul(Z::from_value(scioxide_core::Complex::new(
                    -sign * s.re / t,
                    0.0,
                ))?)?;
            }
            return Ok(result);
        }
        self.evaluate_scaled(Z::from_value(s)?, real_output, work)
    }
    fn evaluate_scaled(
        &self,
        s: Z,
        real_output: bool,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<LameEstimate, SciOxideError> {
        self.evaluate_expansion(s, real_output, None, work)
    }
    fn evaluate_expansion(
        &self,
        s: Z,
        real_output: bool,
        complementary: Option<&AngularExpansion>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<LameEstimate, SciOxideError> {
        work.check()?;
        let (coefficients, class, ratio, scale, coefficient_error) = complementary.map_or(
            (
                &self.coefficients,
                self.class,
                self.ratio,
                self.scale,
                self.coefficient_error,
            ),
            |e| {
                (
                    &e.coefficients,
                    e.class,
                    e.ratio,
                    e.scale,
                    e.coefficient_error,
                )
            },
        );
        let real_argument = s.imaginary().is_zero();
        let lip = if s.imaginary().is_sign_negative() {
            -0.0
        } else {
            0.0
        };
        let coordinate_scale = self.k * ratio.sqrt();
        let x = s.div(Z::from_value(scioxide_core::Complex::new(
            coordinate_scale,
            0.0,
        ))?)?;
        let two_x = x.mul(Z::from_value(scioxide_core::Complex::new(2.0, 0.0))?)?;
        let mut magnitude = Scaled::ZERO;
        let ((b1, d1), (b2, d2)) =
            crate::support::polynomial::clenshaw(
                coefficients.iter(),
                (Z::ZERO, Z::ZERO),
                |(b1, d1), (b2, d2), &coefficient| {
                    work.try_step()?;
                    let b = two_x.mul(b1)?.sub(b2)?.add(Z::from_value(
                        scioxide_core::Complex::new(coefficient, 0.0),
                    )?)?;
                    let d = b1
                        .mul(Z::from_value(scioxide_core::Complex::new(2.0, 0.0))?)?
                        .add(two_x.mul(d1)?)?
                        .sub(d2)?;
                    magnitude = magnitude.add(b.magnitude()?)?;
                    Ok::<_, SciOxideError>((b, d))
                },
            )?;
        let (h_factor, k_factor) = class.factors();
        let mut value = if h_factor { b1 } else { b1.sub(x.mul(b2)?)? };
        let mut derivative = if h_factor {
            d1
        } else {
            d1.sub(b2)?.sub(x.mul(d2)?)?
        };
        let norm = value.magnitude()?;
        let condition = if norm.is_zero() {
            1.0
        } else {
            magnitude.div(norm)?.round()?.max(1.0)
        };
        let mut singular = false;
        let k_squared = self.k2.try_to_f64_exact()?;
        let h_squared = self.h2.try_to_f64_exact()?;
        let h_squared = if complementary.is_some() {
            k_squared - h_squared
        } else {
            h_squared
        };
        for (active, focal_squared) in [(h_factor, h_squared), (k_factor, k_squared)] {
            if !active {
                continue;
            }
            work.try_step()?;
            // Preserve the exact IEEE input geometry near a focal point. Forming
            // (x-sqrt(H))*(x+sqrt(H)) would erase the square-root rounding residual.
            let difference = s.difference_of_products(
                s,
                Z::from_value(scioxide_core::Complex::new(focal_squared, 0.0))?,
                Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
            )?;
            let root = if real_output {
                Z::from_components(difference.real().abs().sqrt()?, Scaled::ZERO)
            } else if real_argument {
                let magnitude = difference.real().abs().sqrt()?;
                if difference.real().is_sign_negative() {
                    Z::from_components(
                        Scaled::ZERO,
                        if lip.is_sign_negative() {
                            magnitude.neg()
                        } else {
                            magnitude
                        },
                    )
                } else {
                    Z::from_components(
                        if s.real().is_sign_negative() {
                            magnitude.neg()
                        } else {
                            magnitude
                        },
                        Scaled::from_value(lip)?,
                    )
                }
            } else if s.real().is_zero() {
                let magnitude = difference.real().abs().sqrt()?;
                Z::from_components(
                    Scaled::ZERO,
                    if s.imaginary().is_sign_negative() {
                        magnitude.neg()
                    } else {
                        magnitude
                    },
                )
            } else {
                let root = difference.sqrt()?;
                if s.real().is_sign_negative() {
                    root.neg()
                } else {
                    root
                }
            }
            .div(Z::from_value(scioxide_core::Complex::new(
                coordinate_scale,
                0.0,
            ))?)?;
            if root.is_zero() {
                singular = true;
            } else {
                let mut dr = x.div(root)?;
                if real_output && difference.real().is_sign_negative() {
                    dr = dr.neg();
                }
                derivative = derivative.mul(root)?.add(value.mul(dr)?)?;
            }
            value = value.mul(root)?;
            magnitude = magnitude.mul(root.magnitude()?)?;
        }
        let relative = coefficient_error * condition;
        let mut value = Estimate::new(value.mul(scale)?, relative);
        if norm.is_zero() {
            value.absolute = magnitude
                .mul(scale.magnitude()?)?
                .mul(Scaled::from_value(coefficient_error)?)?
                .round()
                .unwrap_or(f64::MAX);
        }
        let derivative = if singular {
            None
        } else {
            Some(Estimate::new(
                derivative
                    .mul(scale)?
                    .div(Z::from_value(scioxide_core::Complex::new(
                        coordinate_scale,
                        0.0,
                    ))?)?,
                relative * 2.0,
            ))
        };
        Ok(LameEstimate { value, derivative })
    }
}

fn work<'a, 's, 'c, R: RealScalar>(
    options: EllipsoidalOptions<R>,
    context: Option<&'a mut SolveContext<'s, 'c>>,
) -> Result<Work<'a, 's, 'c>, SciOxideError> {
    let budget = options.iteration_budget().get();
    // A facade's preparation and its nested eigensolver epoch share the scope
    // allowance, so evaluation gets only what own and nested work left.
    let limit = match context.as_deref() {
        Some(context) => {
            crate::spectral::remaining(context)?.map_or(budget, |left| left.min(budget))
        }
        None => budget,
    };
    Ok(Work::for_family(
        limit,
        options
            .relative_tolerance()
            .relative()
            .try_to_f64_exact()?
            .max(8.0 * f64::EPSILON),
        context,
        algorithm_component!("ellipsoidal"),
    ))
}
/// A Lamé value estimate with its derivative estimate.
pub(crate) struct LameEstimate {
    value: Estimate,
    derivative: Option<Estimate>,
}

/// Evaluation of an ellipsoidal harmonic or normalization.
type LameEvaluation<T> = Evaluation<T, Lame<T>>;

/// The value forms and the derivative estimate of a Lamé evaluation.
#[derive(Clone, Copy)]
pub(crate) struct Lame<T> {
    value: kernel::EllipticForm<T>,
    derivative: Option<(Option<T>, kernel::EllipticForm<T>)>,
}

impl<T: NumericScalar> Publish<T> for Lame<T> {
    fn publish(self, value: Option<T>) -> Result<Forms<T>, SciOxideError> {
        let derivative = self
            .derivative
            .map(|(value, form)| form.publish(value))
            .transpose()?
            .and_then(|forms| forms.value);
        Ok(Forms {
            derivative,
            ..self.value.publish(value)?
        })
    }
}

fn report<T: NumericScalar>(
    result: Result<LameEstimate, SciOxideError>,
    work: &Work<'_, '_, '_>,
    method: AlgorithmComponent,
) -> Result<LameEvaluation<T>, SciOxideError> {
    let derivative = result.as_ref().ok().and_then(|e| e.derivative);
    let value = kernel::report::<T>(result.map(|e| e.value), work, method)?;
    let derivative = derivative
        .map(|d| kernel::report::<T>(Ok(d), work, method))
        .transpose()?
        .and_then(|d| d.form().copied().map(|form| (d.ordinary(), form)));
    Ok(value.map_form(|value| Lame { value, derivative }))
}

/// Computes the Lamé function of the first kind `E_n^p(s)` for `h² = h2` and `k² = k2`.
///
/// The definition, the classes and the branch of the focal factors are those of
/// [`EllipsoidalHarmonicPlan`] and [`EllipsoidalHarmonicPlan::first_kind`]. Each call prepares
/// a plan with default [`EllipsoidalOptions`] and evaluates it once; preparation and evaluation
/// share one work budget. Prepare an [`EllipsoidalHarmonicPlan`] to evaluate many points.
///
/// # Accuracy
///
/// No fixed ulp bound is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// The errors of [`EllipsoidalHarmonicPlan::new`] and [`EllipsoidalHarmonicPlan::first_kind`],
/// and `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type.
///
/// # Examples
///
/// ```
/// use scioxide_special::{EllipsoidalSigns, ellipsoidal_harmonic};
///
/// // E₀¹(s) = 1 and E₁¹(s) = s
/// let signs = EllipsoidalSigns::default();
/// assert!((ellipsoidal_harmonic::<f64>(0.5, 1.0, 0, 1, 1.5, signs)? - 1.0).abs() < 1e-10);
/// assert!((ellipsoidal_harmonic::<f64>(0.5, 1.0, 1, 1, 1.5, signs)? - 1.5).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ellipsoidal_harmonic<T: NumericScalar>(
    h2: T::Real,
    k2: T::Real,
    n: usize,
    p: usize,
    s: T,
    signs: EllipsoidalSigns,
) -> Result<T, SciOxideError> {
    widen(s, "spectral parameter s must be finite")?;
    prepared_call(
        h2,
        k2,
        n,
        p,
        EllipsoidalOptions::default(),
        |plan, child| plan.first_run(s, signs, Some(child)),
    )?
    .value()
}

/// Computes the Lamé function of the second kind `F_n^p(s)` for `h² = h2` and `k² = k2`.
///
/// The definition and the domain `s > k` (real) or `|s| > k` (complex) are those of
/// [`EllipsoidalHarmonicPlan::second_kind`]. Each call prepares a plan with default
/// [`EllipsoidalOptions`] and evaluates it once; preparation and evaluation share one work
/// budget. Prepare an [`EllipsoidalHarmonicPlan`] to evaluate many points.
///
/// # Accuracy
///
/// No fixed ulp bound is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// The errors of [`EllipsoidalHarmonicPlan::new`] and
/// [`EllipsoidalHarmonicPlan::second_kind`], and `NumericalInstability` (`Overflow`) if the
/// value exceeds the range of the scalar type.
///
/// # Examples
///
/// ```
/// use scioxide_special::ellipsoidal_harmonic_second_kind;
///
/// // F₀¹(s) = ∫ₛ^∞ dt / √((t² − h²)(t² − k²)), an elliptic integral
/// let f = ellipsoidal_harmonic_second_kind::<f64>(0.5, 1.0, 0, 1, 1.5)?;
/// assert!((f - 0.762_320_146_639_776_9).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ellipsoidal_harmonic_second_kind<T: NumericScalar>(
    h2: T::Real,
    k2: T::Real,
    n: usize,
    p: usize,
    s: T,
) -> Result<T, SciOxideError> {
    widen(s, "spectral parameter s must be finite")?;
    prepared_call(
        h2,
        k2,
        n,
        p,
        EllipsoidalOptions::default(),
        |plan, child| plan.second_run(s, Some(child)),
    )?
    .value()
}

/// Computes the normalization constant `γ_n^p` of a surface ellipsoidal harmonic for `h² = h2`
/// and `k² = k2`.
///
/// The definition is that of [`EllipsoidalHarmonicPlan::normalization`]. Each call prepares a
/// plan with default [`EllipsoidalOptions`] and evaluates it once; preparation and evaluation
/// share one work budget.
///
/// # Accuracy
///
/// No fixed ulp bound is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// The errors of [`EllipsoidalHarmonicPlan::new`] and
/// [`EllipsoidalHarmonicPlan::normalization`], and `NumericalInstability` (`Overflow`) if the
/// value exceeds the range of the scalar type.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
///
/// // γ₀¹ = 4π
/// let gamma = scioxide_special::ellipsoidal_normalization(0.5_f64, 1.0, 0, 1)?;
/// assert!((gamma - 4.0 * PI).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ellipsoidal_normalization<R: RealScalar>(
    h2: R,
    k2: R,
    n: usize,
    p: usize,
) -> Result<R, SciOxideError> {
    prepared_call(
        h2,
        k2,
        n,
        p,
        EllipsoidalOptions::default(),
        |plan, child| plan.normalization_run(Some(child)),
    )?
    .value()
}

// One allowance covers preparation, its nested eigensolver, and evaluation.
fn prepared_call<T: NumericScalar>(
    h2: T::Real,
    k2: T::Real,
    n: usize,
    p: usize,
    options: EllipsoidalOptions<T::Real>,
    run: impl FnOnce(
        &EllipsoidalHarmonicPlan<T::Real>,
        &mut SolveContext<'_, '_>,
    ) -> Result<LameEvaluation<T>, SciOxideError>,
) -> Result<LameEvaluation<T>, SciOxideError> {
    scoped(
        ExecutionContext::sequential_native(),
        options.iteration_budget(),
        |child| {
            let plan = EllipsoidalHarmonicPlan::prepare(h2, k2, n, p, options, child)?;
            Ok(run(&plan, child)?.with_counts(child.counts().total()?))
        },
    )
}
