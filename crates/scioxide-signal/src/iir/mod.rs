//! IIR filter representations, design and filtering.
use crate::util::{check, copy, inputs, zeros};
use crate::util::{empty, inconsistent, invalid, scalar_from_usize_rounded as scalar};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AlgorithmId, Argument, Complex, NumericScalar, ProblemToken, RealScalar, ScaledValue,
    SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState,
};
use scioxide_poly::{ApproxPolynomial, PolynomialBasis};

mod design;
mod execute;
mod sections;
mod transfer;
mod zpk;
pub use design::*;
pub use execute::*;
#[cfg(test)]
mod tests;

/// Frequency variable and stability region of a rational filter.
///
/// An analog filter is a function of `s` and is stable when every pole has
/// `Re s < 0`. A digital filter is a function of `z` and is stable when every
/// pole has `|z| < 1`. Response and group-delay frequencies follow the
/// [crate conventions](crate#conventions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirDomain {
    /// Continuous time: coefficients are descending powers of `s`.
    Analog,
    /// Discrete time: coefficients are ascending powers of `z⁻¹`.
    Digital,
}

fn add(a: usize, b: usize) -> Result<usize, SciOxideError> {
    a.checked_add(b)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("iir_storage")))
}
fn c<S: RealScalar>(x: S) -> Complex<S> {
    Complex::new(x, S::ZERO)
}
fn number<S: RealScalar>(x: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(x)
}
fn failure(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::LossOfSignificance, message)
}
/// A finite caller parameter; NaN and infinities are `NonFinite` input, never
/// an arithmetic overflow.
fn finite_parameter<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    crate::util::require_finite(x, || {
        SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "filter parameter must be finite",
        )
    })
}
/// A finite, strictly positive caller parameter `argument`.
fn positive_parameter<S: RealScalar>(x: S, argument: Argument) -> Result<S, SciOxideError> {
    crate::util::positive(finite_parameter(x)?, argument, crate::util::output_overflow)
}

/// Extended-range accumulator for every zero-pole-gain gain that a design or
/// conversion forms or rescales.
///
/// The same factors are multiplied in ordinary IEEE arithmetic and in
/// [`ScaledValue`](scioxide_core::ScaledValue) range. The ordinary product is
/// kept, bitwise, while every partial product is finite with a normal larger
/// component; otherwise [`ZpkGain::finish`] rounds the extended product once,
/// so intermediate overflow or underflow never loses a representable gain.
/// A nonzero exact gain must round to the normal range: below `S::MIN_POSITIVE`
/// the spacing of representable values exceeds `epsilon` relative to the
/// gain, so the zero-pole-gain form cannot hold the design to working
/// precision (a subnormal gain can lose every digit, and a gain that rounds to
/// zero gives an identically zero filter). `finish` reports such a gain as
/// `NumericalInstability(Underflow)`, and one above the finite range as
/// `NumericalInstability(Overflow)`. An exactly zero gain stays zero.
#[derive(Clone, Copy)]
struct ZpkGain<S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    ordinary: Option<Complex<S>>,
    extended: scioxide_core::ScaledValue<Complex<S>>,
}
impl<S: RealScalar> ZpkGain<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn new(gain: Complex<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            ordinary: Some(gain).filter(|g| Self::normal(*g)),
            extended: scioxide_core::ScaledValue::from_value(check(gain)?)?,
        })
    }
    /// Finite with a normal larger component, so no partial-product precision was lost.
    fn normal(gain: Complex<S>) -> bool {
        gain.has_finite_components() && gain.re.abs().max(gain.im.abs()) >= S::MIN_POSITIVE
    }
    fn with(
        self,
        ordinary: impl FnOnce(Complex<S>) -> Complex<S>,
        extended: scioxide_core::ScaledValue<Complex<S>>,
    ) -> Self {
        Self {
            ordinary: self.ordinary.map(ordinary).filter(|g| Self::normal(*g)),
            extended,
        }
    }
    /// Multiply by a finite complex factor.
    fn times(self, factor: Complex<S>) -> Result<Self, SciOxideError> {
        let extended = self
            .extended
            .mul(scioxide_core::ScaledValue::from_value(check(factor)?)?)?;
        Ok(self.with(|g| g * factor, extended))
    }
    /// Divide by a finite complex factor; a zero divisor has no finite gain.
    fn over(self, divisor: Complex<S>) -> Result<Self, SciOxideError> {
        if divisor.re == S::ZERO && divisor.im == S::ZERO {
            return Err(crate::util::output_overflow());
        }
        let extended = self
            .extended
            .div(scioxide_core::ScaledValue::from_value(check(divisor)?)?)?;
        Ok(self.with(|g| g / divisor, extended))
    }
    /// Multiply by a finite real factor.
    fn scale(self, factor: S) -> Result<Self, SciOxideError> {
        let extended = self
            .extended
            .mul_real(scioxide_core::Scaled::from_value(check(factor)?)?)?;
        Ok(self.with(|g| g.scale(factor), extended))
    }
    /// Divide by a finite nonzero real factor.
    fn unscale(self, divisor: S) -> Result<Self, SciOxideError> {
        if divisor == S::ZERO {
            return Err(crate::util::output_overflow());
        }
        let divisor_scaled = scioxide_core::Scaled::from_value(check(divisor)?)?;
        let extended = scioxide_core::ScaledValue::from_components(
            self.extended.real().div(divisor_scaled)?,
            self.extended.imaginary().div(divisor_scaled)?,
        );
        Ok(self.with(|g| g.unscale(divisor), extended))
    }
    /// The gain, rounded once; see the type documentation for its errors.
    fn finish(self) -> Result<Complex<S>, SciOxideError> {
        if let Some(gain) = self.ordinary {
            return Ok(gain);
        }
        let gain = check(self.extended.round()?)?;
        if !self.extended.is_zero() && gain.re.abs().max(gain.im.abs()) < S::MIN_POSITIVE {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "zero-pole-gain gain is below the normal range",
            ));
        }
        Ok(gain)
    }
}

/// Error for a nested root solve that stopped or missed its certificate.
///
/// Cancellation and deadlines stay execution stops; exhausted work maps to the
/// matching convergence reason with the solve's counts.
fn unsuccessful_root(
    termination: scioxide_rootfind::RootTermination,
    counts: scioxide_core::SolveCounts,
    detail: &'static str,
) -> SciOxideError {
    use scioxide_core::{BudgetKind, ConvergenceReason, ExecutionStopReason};
    use scioxide_rootfind::RootTermination;
    let reason = match termination {
        RootTermination::Cancelled => {
            return SciOxideError::execution_stopped(ExecutionStopReason::Cancelled);
        }
        RootTermination::TimeLimit => {
            return SciOxideError::execution_stopped(ExecutionStopReason::DeadlineReached);
        }
        RootTermination::IterationLimit | RootTermination::WorkLimit(BudgetKind::Iteration) => {
            ConvergenceReason::IterationLimit
        }
        RootTermination::WorkLimit(BudgetKind::Evaluation) => ConvergenceReason::EvaluationLimit,
        RootTermination::WorkLimit(kind) => ConvergenceReason::ResourceLimit(kind),
        RootTermination::Stagnation => ConvergenceReason::Stagnation,
        _ => ConvergenceReason::NestedSolver,
    };
    SciOxideError::convergence_failure(
        reason,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts),
        detail,
    )
}

/// Run one iterative design as a new solve under `scope`.
///
/// The local allowance is unbounded: iteration counts are scope-local and
/// reach a parent as nested counts, while shared resources, cancellation and
/// deadlines follow the parent.
pub(crate) fn run_design<'scope, 'capability, T>(
    scope: impl Into<SolveScope<'scope, 'capability>>,
    algorithm: AlgorithmId,
    run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scope.into().run(
        &mut SolveState::new(),
        SolveIdentity::new(ProblemToken::new(0), algorithm, 0, 0),
        SolveBudgets::unbounded(),
        run,
    )
}

/// Rational transfer function `H = B/A` of an analog or digital filter.
///
/// Digital coefficients are ascending powers of `z⁻¹` and analog coefficients
/// are descending powers of `s`:
///
/// ```text
/// H(z) = (b₀ + b₁ z⁻¹ + … + b_M z⁻ᴹ) / (a₀ + a₁ z⁻¹ + … + a_N z⁻ᴺ)
/// H(s) = (b₀ sᴹ + … + b_M) / (a₀ sᴺ + … + a_N)
/// ```
///
/// where `b` is the numerator and `a` the denominator. An analog function must
/// be proper (`M ≤ N`). [`TransferFunction::new`] divides every coefficient by
/// `a₀`, so the stored denominator starts with one. Leading zeros of a digital
/// numerator are kept as delays, and trailing zeros of digital coefficients
/// are dropped.
///
/// The filter need not be stable: check `to_zpk()?.is_stable()` when it
/// matters. The coefficients of a high-order filter are ill-conditioned, so
/// store such a filter as [`SecondOrderSections`].
///
/// # Examples
///
/// ```
/// use scioxide_signal::{IirDomain, TransferFunction};
/// // H(z) = 0.5 / (1 - 0.5 z^-1) after normalization by a[0] = 2.
/// let transfer = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[2.0, -1.0])?;
/// assert_eq!(transfer.denominator(), [1.0, -0.5]);
/// assert!((transfer.response(0.0)?.re - 1.0).abs() < 1e-15);
/// // The group delay at DC is p / (1 - p) = 1 sample for the pole p = 0.5.
/// assert!((transfer.group_delay(0.0)? - 1.0).abs() < 1e-14);
/// let zpk = transfer.to_zpk()?;
/// assert!(zpk.is_stable() && (zpk.poles()[0].re - 0.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct TransferFunction<T: NumericScalar> {
    domain: IirDomain,
    b: Vec<T>,
    a: Vec<T>,
}
type PolynomialEvaluation<S> = (ScaledValue<Complex<S>>, ScaledValue<Complex<S>>);

/// Group delay of one rational factor `b/a` at the domain variable `x`.
fn rational_group_delay<T: NumericScalar>(
    b: &[T],
    a: &[T],
    x: Complex<T::Real>,
    domain: IirDomain,
) -> Result<T::Real, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let digital = domain == IirDomain::Digital;
    let (b, db) = polynomial(b, x, digital)?;
    let (a, da) = polynomial(a, x, digital)?;
    if b.is_zero() || a.is_zero() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "group delay is undefined at a pole or zero",
        ));
    }
    let delay = match domain {
        IirDomain::Analog => da.div(a)?.sub(db.div(b)?)?,
        IirDomain::Digital => ScaledValue::from_value(x)?.mul(db.div(b)?.sub(da.div(a)?)?)?,
    };
    delay.real().round()
}

fn polynomial<T: NumericScalar>(
    a: &[T],
    x: Complex<T::Real>,
    reverse: bool,
) -> Result<PolynomialEvaluation<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    use scioxide_core::ScaledValue;
    use scioxide_poly::backend::polynomial_basis::newton_value_and_derivative_from_iter;
    let coefficient = |k: usize| {
        let v = a[if reverse { k } else { a.len() - 1 - k }];
        Complex::new(v.real_component(), v.imaginary_component())
    };
    let last = a.len() - 1;
    let terms = (0..last).map(|k| (coefficient(k), c(T::Real::ZERO)));
    let (value, derivative) = newton_value_and_derivative_from_iter(
        terms.map(|(v, node)| Ok((ScaledValue::from_value(v)?, ScaledValue::from_value(node)?))),
        ScaledValue::from_value(coefficient(last))?,
        ScaledValue::from_value(x)?,
        ScaledValue::ZERO,
        |v, x, node| v.mul(x.sub(node)?),
        |a, b| a.add(b),
    )?;
    Ok((value, derivative))
}
fn frequency_variable<S: RealScalar>(
    domain: IirDomain,
    frequency: S,
) -> Result<Complex<S>, SciOxideError> {
    finite_parameter(frequency)?;
    Ok(match domain {
        IirDomain::Analog => Complex::new(S::ZERO, frequency),
        IirDomain::Digital => Complex::from_polar(S::ONE, -S::PI * (frequency % S::TWO)),
    })
}
/// Roots of ascending coefficients `a`, certified as a factorization.
///
/// A candidate root set `r` is accepted when the expanded factorization
/// reproduces the coefficients:
/// `max_j |a_n e_j - a_j| <= 16 n epsilon max(max_j |a_j|, |a_n| max_j q_j)`,
/// where `e` and `q` are the coefficients of `prod(x - r_k)` and
/// `prod(x + |r_k|)`; the second term covers rounding of the expansion itself.
/// The roots are then exactly those of a polynomial within that normwise
/// backward error, so the zero-pole-gain form reproduces the coefficients.
/// Acceptance does not depend on step convergence. Per-root backward error
/// does not suffice: individually verified members of a cluster, such as the
/// `N`-fold Nyquist zero of a digital Butterworth numerator, can scatter
/// unevenly and reproduce the polynomial only to about `epsilon^(1/N)`.
///
/// The polynomial solver's roots are tried first. When they miss the
/// certificate, the eigenvalues of the balanced companion matrix, whose root
/// sets are normwise backward stable, are tried. A set that cannot be
/// certified, including one whose expansion leaves the finite range, is
/// numerical loss of significance; cancellation, deadlines and exhausted
/// shared work are stops. Forward accuracy follows root conditioning: an
/// `m`-fold root is accurate to about `epsilon^(1/m)`.
fn roots<T: NumericScalar>(
    a: &[T],
    mut context: Option<&mut SolveContext<'_, '_>>,
) -> Result<Vec<Complex<T::Real>>, SciOxideError>
where
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    use scioxide_rootfind::{
        PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace, RootTermination,
        polynomial_roots_into,
    };
    if a.len() <= 1 {
        return Ok(Vec::new());
    }
    let tolerance = T::Real::EPSILON * number::<T::Real>(16.)?;
    let options = PolynomialRootOptions::default()
        .with_method(PolynomialRootMethod::Auto)
        .with_backward_error_tolerance(scioxide_core::ResidualTolerance::new(tolerance, tolerance)?)
        .with_step_tolerance(scioxide_core::StepTolerance::new(tolerance, tolerance)?);
    let mut output = zeros(a.len() - 1)?;
    let mut workspace = PolynomialRootWorkspace::workspace(a.len() - 1, &options)?;
    let report = if let Some(context) = context.as_deref_mut() {
        polynomial_roots_into(a, &options, &mut output, &mut workspace, context)?
    } else {
        polynomial_roots_into(
            a,
            &options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?
    };
    if let termination @ (RootTermination::Cancelled
    | RootTermination::TimeLimit
    | RootTermination::WorkLimit(_)) = report.termination()
    {
        return Err(unsuccessful_root(
            termination,
            report.accounting(),
            "polynomial root solve exhausted shared work",
        ));
    }
    if reproduces(a, &output)? {
        return Ok(output);
    }
    let output = companion_roots(a, context)?;
    if reproduces(a, &output)? {
        return Ok(output);
    }
    Err(failure(
        "polynomial roots do not reproduce the coefficients",
    ))
}

/// Whether `a_n prod(x - r)` reproduces ascending coefficients `a`; see [`roots`].
fn reproduces<T: NumericScalar>(a: &[T], roots: &[Complex<T::Real>]) -> Result<bool, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let expand = |roots: &[Complex<T::Real>]| {
        ApproxPolynomial::from_roots(roots).ok().filter(|p| {
            p.basis()
                .coefficients()
                .iter()
                .all(|c| c.has_finite_components())
        })
    };
    let Some(product) = expand(roots) else {
        return Ok(false);
    };
    let mut magnitudes = zeros::<Complex<T::Real>>(roots.len())?;
    for (slot, root) in magnitudes.iter_mut().zip(roots) {
        *slot = c(-root.norm());
    }
    let Some(envelope) = expand(&magnitudes) else {
        return Ok(false);
    };
    let complex = |x: T| Complex::new(x.real_component(), x.imaginary_component());
    let lead = complex(a[a.len() - 1]);
    let mut scale = T::Real::ZERO;
    let mut error = T::Real::ZERO;
    for ((&coefficient, &expanded), &bound) in a
        .iter()
        .zip(product.basis().coefficients())
        .zip(envelope.basis().coefficients())
    {
        scale = scale
            .max(coefficient.abs())
            .max(lead.norm() * bound.re.abs());
        error = error.max((lead * expanded - complex(coefficient)).norm());
    }
    let tolerance = T::Real::EPSILON * number::<T::Real>(16.)? * scalar::<T::Real>(roots.len())?;
    Ok(error.is_finite() && error <= tolerance * scale)
}

/// Eigenvalues of the balanced companion matrix of ascending coefficients `a`.
fn companion_roots<T: NumericScalar>(
    a: &[T],
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<Vec<Complex<T::Real>>, SciOxideError>
where
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    use scioxide_linalg::{GeneralEigenOptions, GeneralEigenvaluesBuffer, LinalgWorkspace, Matrix};
    let n = a.len() - 1;
    let lead = a[n];
    let mut matrix = Matrix::<T>::zeros(n, n)?;
    let values = matrix.as_mut_slice();
    for (slot, &coefficient) in values[..n].iter_mut().zip(a[..n].iter().rev()) {
        *slot = check(-(coefficient / lead))?;
    }
    for row in 1..n {
        values[row * n + row - 1] = T::ONE;
    }
    let mut result = GeneralEigenvaluesBuffer::empty();
    let options = GeneralEigenOptions::new();
    let mut workspace = LinalgWorkspace::new();
    if let Some(context) = context {
        GeneralEigenvaluesBuffer::compute_into(
            &matrix,
            &options,
            &mut result,
            &mut workspace,
            context,
        )?;
    } else {
        GeneralEigenvaluesBuffer::compute_into(
            &matrix,
            &options,
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
    }
    copy(result.eigenvalues())
}

/// Zeros, poles and gain of a proper rational filter.
///
/// ```text
/// H(x) = k ∏ᵢ (x − zᵢ) / ∏ⱼ (x − pⱼ)
/// ```
///
/// where `x = s` for an analog filter and `x = z` for a digital one, `zᵢ` are
/// the zeros, `pⱼ` the poles and `k` the gain. All roots are finite, and there
/// are at most as many zeros as poles. Roots are stored as given: nothing is
/// cancelled and unstable poles are not reflected. A digital delay appears as
/// a pole at the origin.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_signal::{IirDomain, PoleZeroGain};
/// let c = |re: f64| Complex::new(re, 0.0);
/// // H(z) = 0.25 (z + 1) / (z - 0.5): unit DC gain and a Nyquist zero.
/// let zpk = PoleZeroGain::new(IirDomain::Digital, &[c(-1.0)], &[c(0.5)], c(0.25))?;
/// assert!((zpk.response(0.0)? - c(1.0)).norm() < 1e-15);
/// assert!(zpk.response(1.0)?.norm() < 1e-15);
/// assert!((zpk.stability_margin() - 0.5).abs() < 1e-15);
/// let transfer = zpk.to_transfer()?;
/// assert!((transfer.response(0.3)? - zpk.response(0.3)?).norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct PoleZeroGain<S: RealScalar> {
    domain: IirDomain,
    zeros: Vec<Complex<S>>,
    poles: Vec<Complex<S>>,
    gain: Complex<S>,
}
fn expand<S: RealScalar>(roots: &[Complex<S>]) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let polynomial = ApproxPolynomial::from_roots(roots)?;
    let mut out = copy(polynomial.basis().coefficients())?;
    out.reverse();
    Ok(out)
}

/// Cascade of strictly stable second-order sections.
///
/// Each row `[b₀, b₁, b₂, a₀, a₁, a₂]` is one factor of the transfer function:
///
/// ```text
/// H(z) = ∏ₖ (b₀ₖ + b₁ₖ z⁻¹ + b₂ₖ z⁻²) / (a₀ₖ + a₁ₖ z⁻¹ + a₂ₖ z⁻²)
/// H(s) = ∏ₖ (b₀ₖ s² + b₁ₖ s + b₂ₖ) / (a₀ₖ s² + a₁ₖ s + a₂ₖ)
/// ```
///
/// for digital and analog sections, where `k` runs over the rows. A
/// first-order analog section has `a₀ = b₀ = 0`. Construction divides each row
/// by its leading nonzero denominator coefficient and rejects a row whose
/// rounded poles are not strictly stable. Responses, group delays and
/// pole-zero conversion work section by section, so they avoid the
/// ill-conditioned expansion of a high-order polynomial.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_signal::{IirDomain, SecondOrderSections};
/// let sos = SecondOrderSections::new(
///     IirDomain::Digital,
///     &[
///         [0.5_f64, 0.0, 0.0, 1.0, -0.5, 0.0],
///         [1.0, 1.0, 0.0, 2.0, 0.0, 0.0],
///     ],
/// )?;
/// assert_eq!(sos.sections()[1], [0.5, 0.5, 0.0, 1.0, 0.0, 0.0]);
/// // Both sections have unit DC gain; the cascade expands to one transfer.
/// assert!((sos.response(0.0)?.re - 1.0).abs() < 1e-15);
/// assert!((sos.to_transfer()?.response(0.4)? - sos.response(0.4)?).norm() < 1e-14);
/// // A pole at z = 2 is outside the unit circle.
/// let unstable = SecondOrderSections::new(IirDomain::Digital, &[[1.0, 0.0, 0.0, 1.0, -2.0, 0.0]]);
/// assert_eq!(unstable.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SecondOrderSections<T: NumericScalar> {
    domain: IirDomain,
    sections: Vec<[T; 6]>,
}
/// Divides `signal` by `divisor` as polynomials, inverting a linear convolution.
///
/// Both slices are coefficients in descending powers, which is also the sample
/// order of a causal sequence. The result satisfies
///
/// ```text
/// signal = convolve(quotient, divisor) + remainder
/// ```
///
/// with [`convolve`](crate::convolve). The quotient has
/// `signal.len() − divisor.len() + 1` coefficients and the remainder has
/// `signal.len()`. A signal shorter than the divisor gives an empty quotient
/// and a remainder equal to the signal. The method is polynomial long
/// division, which amplifies rounding when `divisor[0]` is small compared with
/// the other divisor coefficients.
///
/// # Errors
///
/// - `NonFinite` if a coefficient of `signal` or `divisor` is NaN or infinite.
/// - `InvalidInput` if `divisor` is empty or `divisor[0]` is zero.
/// - `NumericalInstability` if a quotient or remainder coefficient overflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{convolve, deconvolve};
/// let signal = convolve(&[1.0_f64, 2.0], &[1.0, 3.0])?;
/// let (quotient, remainder) = deconvolve(&signal, &[1.0, 3.0])?;
/// assert!(
///     quotient
///         .iter()
///         .zip([1.0, 2.0])
///         .all(|(q, e)| (q - e).abs() < 1e-14)
/// );
/// assert!(remainder.len() == 3 && remainder.iter().all(|r| r.abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn deconvolve<T: NumericScalar>(
    signal: &[T],
    divisor: &[T],
) -> Result<(Vec<T>, Vec<T>), SciOxideError> {
    inputs(signal)?;
    inputs(divisor)?;
    if divisor.is_empty() {
        return Err(empty(
            scioxide_core::argument!("divisor"),
            "divisor must be nonempty",
        ));
    }
    if divisor[0] == T::ZERO {
        return Err(invalid(
            scioxide_core::argument!("divisor"),
            "divisor must have a nonzero leading coefficient",
        ));
    }
    let mut remainder = copy(signal)?;
    if signal.len() < divisor.len() {
        return Ok((Vec::new(), remainder));
    }
    let mut dividend = copy(signal)?;
    let mut denominator = copy(divisor)?;
    dividend.reverse();
    denominator.reverse();
    let dividend = ApproxPolynomial::monomial(dividend)?;
    let denominator = ApproxPolynomial::monomial(denominator)?;
    let division = dividend.div_rem(&denominator)?;
    let mut quotient = zeros(signal.len() - divisor.len() + 1)?;
    for (out, &value) in quotient
        .iter_mut()
        .rev()
        .zip(division.quotient().basis().coefficients())
    {
        *out = value;
    }
    remainder.fill(T::ZERO);
    for (out, &value) in remainder
        .iter_mut()
        .rev()
        .zip(division.remainder().basis().coefficients())
    {
        *out = value;
    }
    Ok((quotient, remainder))
}
