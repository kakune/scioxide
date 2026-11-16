//! Condition estimates for computed polynomial products and compositions.
use crate::PolynomialConditionReport;
use crate::util::{invalid, validate_slice};
use scioxide_core::AlgebraicScalar;
use scioxide_core::Scaled;
use scioxide_core::{NumericScalar, SciOxideError};

/// Returns the coefficient 1-norm in extended exponent range.
fn norm<C: NumericScalar>(a: &[C]) -> Result<Scaled<C::Real>, SciOxideError> {
    validate_slice(a)?;
    a.iter().try_fold(Scaled::ZERO, |s, &x| {
        s.add(crate::util::scaled_complex(x)?.magnitude()?)
    })
}
/// Estimates the normwise condition number of the coefficients of a computed
/// product `c = a · b`.
///
/// `a`, `b` and `product` are ascending monomial coefficient slices, and
/// `product` must have `len(a) + len(b) − 1` entries. The condition estimate is
///
/// ```text
/// κ = max(1, ‖a‖₁ ‖b‖₁ / ‖c‖₁),
/// ```
///
/// where `‖·‖₁` is the sum of the coefficient moduli. Since
/// `‖c‖₁ ≤ ‖a‖₁ ‖b‖₁`, a large `κ` means cancellation in the product. `κ` is
/// `+∞` when `c = 0` while `‖a‖₁ ‖b‖₁ > 0`, or when the ratio
/// overflows. The report also holds the relative forward-error estimate `γₙ κ`,
/// where `γₙ = nε/(1 − nε)`, `n = len(c)` and `ε` is the machine epsilon. The
/// cost is O(len(a) + len(b)) time with no allocation. The estimate is a
/// heuristic, not a bound.
///
/// # Errors
///
/// - `InvalidInput` if `a` or `b` is empty, or if `product` has the wrong
///   length.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `SizeOverflow` if the product length overflows `usize`.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
///
/// // (1 + x)(1 − x) = 1 − x²: ‖a‖₁‖b‖₁ = 4 and ‖c‖₁ = 2, so κ = 2.
/// let (a, b, c) = ([1.0_f64, 1.0], [1.0_f64, -1.0], [1.0_f64, 0.0, -1.0]);
/// let report = polynomial_product_condition(&a, &b, &c)?;
/// assert!(report.condition_estimate() >= 1.0);
/// assert!((report.condition_estimate() - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_product_condition<C: NumericScalar>(
    a: &[C],
    b: &[C],
    product: &[C],
) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
    if a.is_empty() || b.is_empty() {
        return Err(crate::util::empty_error(if a.is_empty() {
            scioxide_core::argument!("a")
        } else {
            scioxide_core::argument!("b")
        }));
    }
    if product.len()
        != scioxide_core::backend::checked_add(
            a.len(),
            b.len(),
            scioxide_core::quantity!("product_coefficients"),
        )? - 1
    {
        return Err(invalid(
            scioxide_core::argument!("product"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "product condition shape",
        ));
    }
    report(product, norm(a)?.mul(norm(b)?)?)
}
/// Estimates the normwise condition number of the coefficients of a computed
/// composition `c(x) = p(q(x))` in the monomial basis.
///
/// `composition` must have `(len(p) − 1)(len(q) − 1) + 1` entries. Since
/// `‖qⁱ‖₁ ≤ ‖q‖₁ⁱ`, the sum `Σᵢ |pᵢ| ‖q‖₁ⁱ` bounds the 1-norm of the
/// coefficients of `p(q)` computed without cancellation. The condition estimate
/// is
///
/// ```text
/// κ = max(1, Σᵢ |pᵢ| ‖q‖₁ⁱ / ‖c‖₁),
/// ```
///
/// and is `+∞` when `c = 0` but the bound is not, or when the ratio overflows.
/// The cost is O(len(p) + len(q)) time with no allocation. The estimate is a
/// heuristic, not a bound.
///
/// # Errors
///
/// - `InvalidInput` if `p` or `q` is empty, or if `composition` has the wrong
///   length.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `SizeOverflow` if the composition length overflows `usize`.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
///
/// // p(x) = 1 + x and q(x) = 1 + x give p(q(x)) = 2 + x, with no cancellation.
/// let (p, q, c) = ([1.0_f64, 1.0], [1.0_f64, 1.0], [2.0_f64, 1.0]);
/// let report = polynomial_composition_condition(&p, &q, &c)?;
/// assert!(report.condition_estimate() >= 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_composition_condition<C: NumericScalar>(
    p: &[C],
    q: &[C],
    composition: &[C],
) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
    let length = crate::composition::length(
        p.len(),
        q.len(),
        scioxide_core::argument!("p"),
        scioxide_core::argument!("q"),
    )?;
    if composition.len() != length {
        return Err(invalid(
            scioxide_core::argument!("composition"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "composition condition output shape",
        ));
    }
    validate_slice(p)?;
    let qnorm = norm(q)?;
    // `||q^i||_1 <= ||q||_1^i`, so Horner on `|p_i|` at `||q||_1` gives
    // `sum_i |p_i| ||q||_1^i`, a majorant of the coefficient one-norm of `p(q)`
    // computed without cancellation.
    let bound = p.iter().rev().try_fold(Scaled::ZERO, |s, &x| {
        s.mul(qnorm)?
            .add(crate::util::scaled_complex(x)?.magnitude()?)
    })?;
    report(composition, bound)
}
/// Builds the condition report of a computed `output` from the bound `bound`.
fn report<C: NumericScalar>(
    output: &[C],
    bound: Scaled<C::Real>,
) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
    let actual = norm(output)?;
    // A zero majorant means every input coefficient is zero, so the zero output
    // is exact and perfectly conditioned. A zero output under a nonzero majorant
    // means total cancellation, where relative sensitivity is unbounded.
    let condition = crate::basis::scaled_condition_ratio(bound, actual)?;
    crate::basis::condition_report(output, condition, C::Real::ONE)
}
