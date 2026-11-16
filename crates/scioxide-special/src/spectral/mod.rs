//! Converged real tridiagonal spectral preparation shared by special families.
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Argument, Budget, BudgetKind, BudgetLimit, Complex, NumericScalar, RealScalar, SciOxideError,
    SolveContext, Tolerance,
};
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_linalg::{
    EigenIndexRange, EigenvalueAccuracy, HermitianEigenDecomposition, HermitianEigenOptions,
    LinalgWorkspace,
};
/// Selects the kind of a radial Mathieu or spheroidal function.
///
/// The kinds follow DLMF §28.20 and §30.11: they are built from the cylindrical (Mathieu) or
/// spherical (spheroidal) Bessel functions of the matching kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialKind {
    /// The first kind, built from Bessel functions `J` or `j`.
    First,
    /// The second kind, built from Bessel functions `Y` or `y`.
    Second,
    /// The third kind, first kind + i · second kind (outgoing, Hankel `H⁽¹⁾`).
    Third,
    /// The fourth kind, first kind − i · second kind (incoming, Hankel `H⁽²⁾`).
    Fourth,
}
impl RadialKind {
    pub(crate) const fn bessel_kind(self) -> crate::bessel_airy::complex_bessel_common::Kind {
        use crate::bessel_airy::complex_bessel_common::Kind;
        match self {
            Self::First => Kind::J,
            Self::Second => Kind::Y,
            Self::Third => Kind::H1,
            Self::Fourth => Kind::H2,
        }
    }
}
/// Selects the normalization of a Mathieu or spheroidal angular function.
///
/// Radial functions have their own normalization, which this choice does not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralNormalization {
    /// `∫₀^2π ce_n(z)² dz = π` for Mathieu functions (DLMF §28.2), and the norm of the Ferrers
    /// function, `∫₋₁¹ Ps_n^m(x)² dx = 2(n+m)!/((2n+1)(n−m)!)`, for spheroidal functions
    /// (DLMF §30.4).
    Standard,
    /// A unit `L²` norm on the angular domain: `[0, 2π]` for Mathieu functions, `[−1, 1]`
    /// for spheroidal functions.
    UnitL2,
}
/// Sets the minimum and maximum number of expansion coefficients of a spectral plan.
///
/// A plan keeps the coefficients of one parity class, every other term of the Fourier
/// (Mathieu) or Ferrers (spheroidal) expansion. The defaults are 16 and 256; see
/// [`SpectralOptions`] for how preparation chooses the count between them.
///
/// # Examples
///
/// ```
/// let terms = scioxide_special::SpectralTerms::new(16, 256)?;
/// assert_eq!(terms.minimum(), 16);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectralTerms {
    minimum: usize,
    maximum: usize,
}
impl SpectralTerms {
    /// Creates truncation bounds.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) unless `2 ≤ minimum ≤ maximum`.
    pub fn new(minimum: usize, maximum: usize) -> Result<Self, SciOxideError> {
        if minimum < 2 || maximum < minimum {
            return Err(invalid(
                scioxide_core::argument!("terms"),
                "require 2 <= minimum <= maximum",
            ));
        }
        Ok(Self { minimum, maximum })
    }
    /// Returns the minimum number of coefficients.
    pub const fn minimum(self) -> usize {
        self.minimum
    }
    /// Returns the maximum number of coefficients.
    pub const fn maximum(self) -> usize {
        self.maximum
    }
}
impl Default for SpectralTerms {
    fn default() -> Self {
        Self {
            minimum: 16,
            maximum: 256,
        }
    }
}

/// Sets the tolerance, truncation bounds and limits of a [`crate::MathieuPlan`] or
/// [`crate::SpheroidalPlan`].
///
/// Preparation solves a truncated symmetric tridiagonal eigenproblem for the expansion
/// coefficients. It starts with `max(minimum, mode_index + 8)` coefficients, where
/// `mode_index` is the position of the requested mode within its parity class, and doubles
/// the count until two successive truncations agree. The tolerance, `1e-13` absolute and
/// relative by default for both `f64` and `f32`, bounds the change of the characteristic value
/// and of the coefficients and the estimated omitted tail; a looser tolerance does less work.
///
/// The maximum count does not adapt to the order or the parameter. If preparation returns
/// `ConvergenceFailure` (`TruncationLimit`), raise [`SpectralTerms::maximum`] with
/// [`Self::with_terms`]. For a nonzero parameter the maximum must leave room for two
/// truncations, `max(minimum, mode_index + 8) < maximum`.
///
/// Preparation and each evaluation have their own work budget, 2,097,152 units by default.
/// The memory limit, 64 MiB by default, caps the storage of preparation, about
/// `32·maximum²` bytes.
///
/// # Examples
///
/// ```
/// use scioxide_special::{SpectralOptions, SpectralTerms};
/// let options = SpectralOptions::<f64>::default()
///     .with_terms(SpectralTerms::new(32, 256)?);
/// assert_eq!(options.terms().minimum(), 32);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct SpectralOptions<S: RealScalar> {
    tolerance: Tolerance<S>,
    terms: SpectralTerms,
    iteration_budget: Budget,
    preparation_bytes: Budget,
}
impl<S: RealScalar> SpectralOptions<S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the tolerance that steers the truncation.
    pub const fn tolerance(self) -> Tolerance<S> {
        self.tolerance
    }
    /// Returns the truncation bounds.
    pub const fn terms(self) -> SpectralTerms {
        self.terms
    }
    /// Returns the work budget of preparation and of each evaluation, including nested
    /// Bessel-function and eigenvalue computations.
    pub const fn iteration_budget(self) -> Budget {
        self.iteration_budget
    }
    /// Returns the limit on the memory used by preparation, in bytes, excluding allocator
    /// overhead.
    pub const fn preparation_bytes(self) -> Budget {
        self.preparation_bytes
    }
    /// Returns a copy with a different tolerance.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns a copy with different truncation bounds.
    pub const fn with_terms(mut self, terms: SpectralTerms) -> Self {
        self.terms = terms;
        self
    }
    /// Returns a copy with a different work budget.
    pub const fn with_iteration_budget(mut self, iteration_budget: Budget) -> Self {
        self.iteration_budget = iteration_budget;
        self
    }
    /// Returns a copy with a different memory limit.
    pub const fn with_preparation_bytes(mut self, preparation_bytes: Budget) -> Self {
        self.preparation_bytes = preparation_bytes;
        self
    }
}
impl<S: RealScalar> Default for SpectralOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: S::try_from_f64_lossy(1e-13)
                .and_then(|value| Tolerance::new(value, value))
                .unwrap_or_else(|_| Tolerance::from_machine_epsilon()),
            terms: SpectralTerms::default(),
            iteration_budget: Budget::ONE.saturating_add(2_097_151),
            preparation_bytes: Budget::ONE.saturating_add(64 * 1024 * 1024 - 1),
        }
    }
}
/// Reports how the expansion of a Mathieu or spheroidal plan converged.
///
/// The residual, the coefficient change and the tail bound are absolute values computed in
/// `f64`, whatever the scalar type of the plan.
#[derive(Debug, Clone, Copy)]
pub struct SpectralDiagnostics {
    /// Number of coefficients kept.
    terms: usize,
    /// Number of truncation doublings.
    refinements: usize,
    /// Maximum eigen-equation residual, including the first omitted row.
    eigen_residual: f64,
    /// Maximum coefficient change between the last two truncations.
    coefficient_change: f64,
    /// Estimated real-axis sum of omitted coefficient magnitudes.
    truncation_bound: f64,
    /// Preparation work, including the eigensolver iterations.
    work: usize,
    /// Upper bound on preparation memory.
    preparation_bytes: usize,
}
impl SpectralDiagnostics {
    /// Returns the number of coefficients kept.
    pub const fn terms(self) -> usize {
        self.terms
    }
    /// Returns the number of times the truncation was doubled.
    pub const fn refinements(self) -> usize {
        self.refinements
    }
    /// Returns the largest residual of the truncated eigenvalue equation, including the first
    /// omitted row.
    pub const fn eigen_residual(self) -> f64 {
        self.eigen_residual
    }
    /// Returns the largest change of a coefficient between the last two truncations.
    pub const fn coefficient_change(self) -> f64 {
        self.coefficient_change
    }
    /// Returns the estimated sum of the magnitudes of the omitted coefficients, for a real
    /// coordinate.
    pub const fn truncation_bound(self) -> f64 {
        self.truncation_bound
    }
    /// Returns the work units used by preparation, including the eigenvalue solver.
    pub const fn work(self) -> usize {
        self.work
    }
    /// Returns an upper bound on the memory used by preparation, in bytes.
    pub const fn preparation_bytes(self) -> usize {
        self.preparation_bytes
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Spectrum {
    pub characteristic: f64,
    pub coefficients: Vec<Scaled<f64>>,
    pub errors: Vec<Scaled<f64>>,
    pub tail: Scaled<f64>,
    pub tail_ratio: f64,
    pub diagnostics: SpectralDiagnostics,
}
use crate::support::invalid;
/// Loss-of-significance error.
pub(crate) fn instability(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::LossOfSignificance, detail)
}
/// Iterations counted by this scope and its nested linear-algebra epochs.
pub(crate) fn used(context: &SolveContext<'_, '_>) -> Result<usize, SciOxideError> {
    Ok(context.counts().total()?.iterations())
}
/// Shared allowance left after own and nested work, or `None` when unbounded.
///
/// Linear-algebra stages run in child epochs that do not consume this scope's
/// own allowance, so the shared work limit is enforced on the combined count.
pub(crate) fn remaining(context: &SolveContext<'_, '_>) -> Result<Option<usize>, SciOxideError> {
    let used = used(context)?;
    Ok(match context.budget_report(BudgetKind::Iteration).limit() {
        BudgetLimit::Finite(limit) => Some(limit.saturating_sub(used)),
        BudgetLimit::Unbounded => None,
    })
}
/// Counts one unit of work.
#[inline]
pub(crate) fn step(context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    context.execution().check()?;
    // The local ledger already counts own iterations. Subtract child work
    // from its remaining allowance without merging every unrelated provider
    // counter on every coefficient. Admission checks ledger integrity/count
    // overflow; `used` and eigensolver boundaries still validate full totals.
    let exhausted = context
        .budget_report(BudgetKind::Iteration)
        .remaining()
        .is_some_and(|left| context.counts().nested().iterations() >= left);
    if exhausted || !context.record_iteration()?.is_granted() {
        return Err(crate::result::exhausted(
            scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
        ));
    }
    Ok(())
}
/// Checks the truncation bounds and the memory limit of a preparation.
///
/// `mode` names the argument reported when the mode lies outside the truncation bounds.
fn storage<S: RealScalar>(
    index: usize,
    diagonal_only: bool,
    options: SpectralOptions<S>,
    mode: Argument,
) -> Result<usize, SciOxideError> {
    if index >= options.terms.maximum {
        return Err(invalid(
            mode,
            "mode index must be below the maximum retained coefficient count",
        ));
    }
    if !diagonal_only
        && checked_add(index, 8, scioxide_core::quantity!("terms"))?.max(options.terms.minimum)
            >= options.terms.maximum
    {
        return Err(invalid(
            mode,
            "max(minimum, mode_index + 8) must be below the maximum retained coefficient count to compare two truncations",
        ));
    }
    let n = options.terms.maximum;
    // Four dense f64 matrices cover eigen output/scratch and their growth.
    // Linear storage includes matrix bands, current/prior scaled coefficients
    // and errors, recurrence scratch, and at most 2*n spheroidal step pairs
    // (64*n bytes). The bound is checked before any family allocates.
    let bytes = checked_add(
        checked_mul(
            checked_mul(n, n, scioxide_core::quantity!("preparation_bytes"))?,
            32,
            scioxide_core::quantity!("preparation_bytes"),
        )?,
        checked_add(
            checked_mul(n, 512, scioxide_core::quantity!("preparation_bytes"))?,
            4096,
            scioxide_core::quantity!("preparation_bytes"),
        )?,
        scioxide_core::quantity!("preparation_bytes"),
    )?;
    if bytes > options.preparation_bytes.get() {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("preparation_bytes"),
            bytes,
            options.preparation_bytes.get(),
        ));
    }
    Ok(bytes)
}

fn upward(x: Scaled<f64>) -> Result<f64, SciOxideError> {
    let value = x.abs().round()?;
    Ok(if !x.is_zero() && value == 0.0 {
        f64::from_bits(1)
    } else {
        value
    })
}
fn normalized_coefficients(
    diagonal: &[f64],
    off: &[f64],
    lambda: f64,
    pivot: usize,
    context: &mut SolveContext<'_, '_>,
) -> Result<Vec<Scaled<f64>>, SciOxideError> {
    let n = diagonal.len();
    let mut left = try_filled(
        n,
        Scaled::ZERO,
        scioxide_core::quantity!("left_coefficients"),
    )?;
    let mut right = try_filled(
        n,
        Scaled::ZERO,
        scioxide_core::quantity!("right_coefficients"),
    )?;
    left[0] = Scaled::from_value(1.0)?;
    right[n - 1] = Scaled::from_value(1.0)?;
    for k in 0..pivot {
        step(context)?;
        let mut value =
            left[k].mul(Scaled::from_value(lambda)?.sub(Scaled::from_value(diagonal[k])?)?)?;
        if k > 0 {
            value = value.sub(left[k - 1].mul(Scaled::from_value(off[k - 1])?)?)?;
        }
        left[k + 1] = value.div(Scaled::from_value(off[k])?)?;
    }
    for k in (pivot + 1..n).rev() {
        step(context)?;
        let mut value =
            right[k].mul(Scaled::from_value(lambda)?.sub(Scaled::from_value(diagonal[k])?)?)?;
        if k + 1 < n {
            value = value.sub(right[k + 1].mul(Scaled::from_value(off[k])?)?)?;
        }
        right[k - 1] = value.div(Scaled::from_value(off[k - 1])?)?;
    }
    let (a, b) = (left[pivot], right[pivot]);
    if a.is_zero() || b.is_zero() {
        return Err(instability("spectral recurrence normalization vanished"));
    }
    for k in 0..n {
        step(context)?;
        left[k] = if k <= pivot {
            left[k].div(a)?
        } else {
            right[k].div(b)?
        };
    }
    let norm = left
        .iter()
        .try_fold(Scaled::ZERO, |sum, &v| sum.add(v.mul(v)?))?
        .sqrt()?;
    let negative = left[0].is_sign_negative();
    for v in &mut left {
        step(context)?;
        *v = v.div(norm)?;
        if negative {
            *v = v.neg();
        }
    }
    Ok(left)
}
/// Each callback supplies a diagonal/off-diagonal pair or a rigorous future
/// diagonal lower bound and off-diagonal magnitude upper bound. `mode` names the
/// argument reported when the mode lies outside the truncation bounds.
pub(crate) fn prepare<S: RealScalar>(
    index: usize,
    diagonal_only: bool,
    options: SpectralOptions<S>,
    mode: Argument,
    context: &mut SolveContext<'_, '_>,
    matrix: impl Fn(usize) -> Result<(f64, f64), SciOxideError>,
    tail_bounds: impl Fn(usize) -> Result<(f64, f64), SciOxideError>,
) -> Result<Spectrum, SciOxideError> {
    let bytes = storage(index, diagonal_only, options, mode)?;
    if diagonal_only {
        step(context)?;
        let value = crate::support::finite(matrix(index)?.0, || {
            instability("spectral matrix or characteristic exceeds working range")
        })?;
        let mut coefficients = try_filled(
            index + 1,
            Scaled::ZERO,
            scioxide_core::quantity!("coefficients"),
        )?;
        coefficients[index] = Scaled::from_value(1.0)?;
        return Ok(Spectrum {
            characteristic: value,
            errors: try_filled(index + 1, Scaled::ZERO, scioxide_core::quantity!("errors"))?,
            coefficients,
            tail: Scaled::ZERO,
            tail_ratio: 0.0,
            diagnostics: SpectralDiagnostics {
                terms: index + 1,
                refinements: 0,
                eigen_residual: 0.0,
                coefficient_change: 0.0,
                truncation_bound: 0.0,
                work: used(context)?,
                preparation_bytes: bytes,
            },
        });
    }
    // The tolerance steers the binary64 truncation refinement for both scalar
    // widths: radial cross-products amplify coefficient errors even for f32.
    let (absolute, relative) = (
        options.tolerance.absolute().try_to_f64_exact()?,
        options.tolerance.relative().try_to_f64_exact()?,
    );
    // The eigen-residual includes the matrix rounding that truncation cannot
    // remove; beyond this documented floor the preparation has broken down.
    let residual_floor = S::SQRT_EPSILON.try_to_f64_exact()?;
    let mut n = options
        .terms
        .minimum
        .max(index + 8)
        .min(options.terms.maximum);
    let mut previous: Option<Spectrum> = None;
    let mut refinements = 0;
    let mut eigen = HermitianEigenDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    let eigen_options = HermitianEigenOptions::default()
        .with_index_range(EigenIndexRange::new(index, index + 1)?)
        .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative);
    loop {
        let mut diagonal = try_filled(n, 0.0, scioxide_core::quantity!("diagonal"))?;
        let mut off = try_filled(n, 0.0, scioxide_core::quantity!("off_diagonal"))?;
        for k in 0..n {
            step(context)?;
            let (d, e) = matrix(k)?;
            diagonal[k] = crate::support::finite(d, || {
                instability("spectral matrix or characteristic exceeds working range")
            })?;
            off[k] = crate::support::finite(e, || {
                instability("spectral matrix or characteristic exceeds working range")
            })?;
        }
        // The eigensolver's child epoch is bounded by the remaining shared work.
        let allowance = match remaining(context)? {
            Some(0) => {
                return Err(crate::result::exhausted(
                    scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                ));
            }
            Some(left) => Budget::try_from(left)?,
            None => options.iteration_budget,
        };
        HermitianEigenDecomposition::tridiagonal_into(
            &diagonal,
            &off[..n - 1],
            &eigen_options.with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(allowance),
            ),
            &mut eigen,
            &mut workspace,
            &mut *context,
        )?;
        let lambda = eigen.eigenvalues()[0];
        let mut pivot = 0;
        for k in 1..n {
            if eigen.eigenvectors().view().get(k, 0)?.abs()
                > eigen.eigenvectors().view().get(pivot, 0)?.abs()
            {
                pivot = k;
            }
        }
        let coefficients = normalized_coefficients(&diagonal, &off, lambda, pivot, context)?;
        let mut residual = 0.0_f64;
        let mut weighted = Scaled::ZERO;
        for k in 0..n {
            step(context)?;
            let mut row = ScaledAccumulator::ZERO;
            row.add_product(Scaled::from_value(diagonal[k])?, coefficients[k])?;
            row.add_product(Scaled::from_value(-lambda)?, coefficients[k])?;
            if k > 0 {
                row.add_product(Scaled::from_value(off[k - 1])?, coefficients[k - 1])?;
            }
            if k + 1 < n {
                row.add_product(Scaled::from_value(off[k])?, coefficients[k + 1])?;
            }
            residual = residual.max(upward(row.total()?)?);
            let weight = Scaled::from_value(diagonal[k].abs())?
                .add(Scaled::from_value(off[k].abs())?)?
                .add(Scaled::from_value(if k > 0 {
                    off[k - 1].abs()
                } else {
                    0.0
                })?)?;
            weighted = weighted.add(coefficients[k].abs().mul(weight)?)?;
        }
        residual = residual.max(upward(
            coefficients[n - 1].mul(Scaled::from_value(off[n - 1])?)?,
        )?);
        residual += upward(weighted.mul(Scaled::from_value(8.0 * f64::EPSILON)?)?)?;
        let (lower, upper) = tail_bounds(n)?;
        let gap = Scaled::from_value(lower)?
            .sub(Scaled::from_value(lambda)?)?
            .sub(Scaled::from_value(residual)?)?;
        let ratio = if !gap.is_sign_negative()
            && upper > 0.0
            && !gap.abs_le(Scaled::from_value(upper)?.mul(Scaled::from_value(2.0)?)?)
        {
            let t = Scaled::from_value(upper)?.div(gap)?.round()?;
            (2.0 * t / (1.0 + (1.0 - 4.0 * t * t).sqrt())).max(f64::from_bits(1))
        } else {
            1.0
        };
        let tail = if ratio < 1.0 {
            coefficients[n - 1]
                .abs()
                .mul(Scaled::from_value(ratio / (1.0 - ratio))?)?
        } else {
            Scaled::from_value(f64::MAX)?
        };
        let mut change = 0.0_f64;
        let mut errors = try_filled(n, Scaled::ZERO, scioxide_core::quantity!("errors"))?;
        for (k, error) in errors.iter_mut().enumerate() {
            step(context)?;
            let prior = previous
                .as_ref()
                .and_then(|p| p.coefficients.get(k))
                .copied()
                .unwrap_or(Scaled::ZERO);
            let delta = coefficients[k].sub(prior)?.abs();
            change = change.max(upward(delta)?);
            *error = delta.add(
                coefficients[k]
                    .abs()
                    .mul(Scaled::from_value(16.0 * f64::EPSILON * n as f64)?)?,
            )?;
        }
        let diagnostic = SpectralDiagnostics {
            terms: n,
            refinements,
            eigen_residual: residual,
            coefficient_change: change,
            truncation_bound: upward(tail)?,
            work: used(context)?,
            preparation_bytes: bytes,
        };
        let ready = previous.as_ref().is_some_and(|p| {
            (lambda - p.characteristic).abs() <= absolute.max(relative * lambda.abs())
        }) && residual <= residual_floor * lambda.abs().max(1.0)
            && change <= absolute.max(relative)
            && diagnostic.truncation_bound <= absolute.max(relative);
        let result = Spectrum {
            characteristic: lambda,
            coefficients,
            errors,
            tail,
            tail_ratio: ratio,
            diagnostics: diagnostic,
        };
        if ready {
            return Ok(result);
        }
        if n == options.terms.maximum {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::TruncationLimit,
                scioxide_core::ConvergenceDiagnostics::new()
                    .with_counts(context.counts())
                    .with_residual(scioxide_core::ResidualReport::unscaled(
                        diagnostic.eigen_residual,
                        scioxide_core::NormSelection::Infinity,
                    )?),
                alloc::format!(
                    "SpectralTerms::maximum={} reached; increase maximum with SpectralOptions::with_terms; terms={}, refinements={}, eigen_residual={}, coefficient_change={}, truncation_bound={}, work={}, preparation_bytes={}",
                    options.terms.maximum,
                    diagnostic.terms,
                    diagnostic.refinements,
                    diagnostic.eigen_residual,
                    diagnostic.coefficient_change,
                    diagnostic.truncation_bound,
                    diagnostic.work,
                    diagnostic.preparation_bytes,
                ),
            ));
        }
        previous = Some(result);
        n = n.saturating_mul(2).min(options.terms.maximum);
        refinements += 1;
    }
}
/// Widens a real or complex coordinate before an evaluation epoch opens; `detail` describes the
/// rejection of a NaN or infinite component.
pub(crate) fn coordinate<T: NumericScalar>(
    z: T,
    detail: &'static str,
) -> Result<Complex<f64>, SciOxideError> {
    if !z.has_finite_components() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            detail,
        ));
    }
    complex(Complex::new(z.real_component(), z.imaginary_component()))
}
/// Widen a finite coordinate.
pub(crate) fn complex<S: RealScalar>(z: Complex<S>) -> Result<Complex<f64>, SciOxideError> {
    if !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "spectral coordinate must be finite",
        ));
    }
    Ok(Complex::new(
        z.re.try_to_f64_exact()?,
        z.im.try_to_f64_exact()?,
    ))
}

pub(crate) mod ellipsoidal;
pub(crate) mod mathieu;
pub(crate) mod spheroidal;

/// Returns family estimates with a single work-exhaustion status.
pub(crate) fn report<T: scioxide_core::NumericScalar>(
    result: Result<crate::support::JetEstimate, SciOxideError>,
    derivative: bool,
    family: scioxide_core::AlgorithmComponent,
    context: &mut SolveContext<'_, '_>,
) -> Result<crate::SpecialFunctionResult<T>, SciOxideError> {
    context.execution().check()?;
    let exhausted = result.as_ref().err().is_some_and(|error| {
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        )
    });
    crate::complex_support::report_with_status(
        result,
        derivative,
        family,
        context.counts().own().iterations(),
        exhausted,
    )?
    .publish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use mathieu::{MathieuParity, MathieuPlan};
    use spheroidal::{SpheroidalKind, SpheroidalPlan};

    #[test]
    fn preparation_counts_include_selected_solver_and_cached_factors() {
        let options = SpectralOptions::<f64>::default();
        let mathieu = |options| {
            MathieuPlan::new(
                7,
                MathieuParity::Even,
                -3.0,
                SpectralNormalization::Standard,
                options,
            )
        };
        let spheroidal = |options| {
            SpheroidalPlan::new(
                SpheroidalKind::Oblate,
                2,
                7,
                3.0,
                SpectralNormalization::Standard,
                options,
            )
        };
        let a = mathieu(options).unwrap();
        let b = spheroidal(options).unwrap();
        fn check(
            work: usize,
            run: impl Fn(SpectralOptions<f64>) -> Result<SpectralDiagnostics, SciOxideError>,
        ) {
            let options = SpectralOptions::default();
            let exact = options.with_iteration_budget(Budget::try_from(work).unwrap());
            assert_eq!(run(exact).unwrap().work(), work);
            let short = options.with_iteration_budget(Budget::try_from(work - 1).unwrap());
            let error = run(short).unwrap_err();
            assert!(
                matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: scioxide_core::ConvergenceReason::IterationLimit,
                        ..
                    }
                ),
                "{error}"
            );
        }
        check(a.diagnostics().work(), |options| {
            mathieu(options).map(|p| p.diagnostics())
        });
        check(b.diagnostics().work(), |options| {
            spheroidal(options).map(|p| p.diagnostics())
        });
    }

    #[test]
    fn angular_allowances_stop_at_the_actual_recurrence_boundary() {
        let mut mathieu = MathieuPlan::new(
            3,
            MathieuParity::Odd,
            2.0_f64,
            SpectralNormalization::UnitL2,
            SpectralOptions::default(),
        )
        .unwrap();
        let mut spheroidal = SpheroidalPlan::new(
            SpheroidalKind::Prolate,
            1,
            4,
            2.0_f64,
            SpectralNormalization::UnitL2,
            SpectralOptions::default(),
        )
        .unwrap();
        macro_rules! check {
            ($plan:ident, $point:expr) => {{
                $plan.options = SpectralOptions::default();
                let reference = $plan.angular($point).unwrap();
                let used = reference.counts().iterations();
                $plan.options = $plan
                    .options
                    .with_iteration_budget(Budget::try_from(used).unwrap());
                assert_eq!($plan.angular($point).unwrap(), reference);
                $plan.options = $plan
                    .options
                    .with_iteration_budget(Budget::try_from(used - 1).unwrap());
                let short = $plan.angular($point).unwrap_err();
                let short_counts = crate::test_support::exhausted_counts(&short);
                assert_eq!(short_counts.iterations(), used - 1);
            }};
        }
        check!(mathieu, 0.3_f64);
        check!(mathieu, Complex::new(0.3_f64, 0.125));
        check!(spheroidal, 0.3_f64);
        check!(spheroidal, Complex::new(0.3_f64, 0.125));
    }
}
