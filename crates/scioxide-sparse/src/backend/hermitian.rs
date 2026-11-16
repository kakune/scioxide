//! Arithmetic on 2×2 Hermitian pivot blocks `[a conj(b); b c]`.
//!
//! Each function takes a block with finite components and real diagonal
//! entries `a` and `c`. Intermediate results use [`Scaled`](scioxide_core::Scaled) arithmetic, which
//! keeps a separate exponent, so they neither overflow nor underflow, and
//! subnormal inputs keep their value. The functions return scaled values; you
//! decide when to round them to the scalar type, for example with `round`. They
//! use constant stack storage and do not allocate.
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledValue};

/// Computes the determinant `a c − |b|²` of the block `[a conj(b); b c]`.
///
/// The difference of products is formed in scaled arithmetic with one rounding
/// at a common exponent. A zero determinant, for a singular block, is a valid
/// result.
///
/// # Errors
///
/// - `NonFinite` if a component is NaN or infinite.
/// - `InvalidInput` if `a` or `c` has a nonzero imaginary part.
/// - `NumericalInstability` if the result exceeds the exponent range of
///   [`Scaled`].
///
/// ```
/// use scioxide_sparse::backend::hermitian::determinant;
///
/// // det [[2, 1], [1, 3]] = 2·3 − 1² = 5.
/// assert_eq!(determinant(2.0_f64, 1.0, 3.0)?.round()?, 5.0);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn determinant<S: NumericScalar>(a: S, b: S, c: S) -> Result<Scaled<S::Real>, SciOxideError> {
    validate_block(a, b, c)?;
    // For the block [a conj(b); b c] with real a and c,
    // det = a c - |b|^2 = a c - br^2 - bi^2. The difference of products is
    // formed with one rounding at a common exponent.
    let aa = Scaled::from_value(a.real_component())?;
    let cc = Scaled::from_value(c.real_component())?;
    let br = Scaled::from_value(b.real_component())?;
    let bi = Scaled::from_value(b.imaginary_component())?;
    aa.difference_of_products(cc, br, br)?.sub(bi.mul(bi)?)
}

/// Computes the two eigenvalues of the block `[a conj(b); b c]`, larger
/// magnitude first.
///
/// ```text
/// λ = (a + c)/2 ± √(((a − c)/2)² + |b|²)
/// ```
///
/// The eigenvalue of larger magnitude adds two terms of the same sign. The
/// other is computed as `det/λ₁`, which avoids cancellation. A zero block gives
/// two zeros.
///
/// # Errors
///
/// The same as [`determinant`].
///
/// ```
/// use scioxide_sparse::backend::hermitian::block_values;
///
/// let values = block_values(2.0_f64, 0.0, 1.0)?;
/// assert_eq!(values[0].round()?, 2.0);
/// assert_eq!(values[1].round()?, 1.0);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn block_values<S: NumericScalar>(
    a: S,
    b: S,
    c: S,
) -> Result<[Scaled<S::Real>; 2], SciOxideError> {
    validate_block(a, b, c)?;
    let half = Scaled::from_value(S::Real::ONE / S::Real::try_from_usize_exact(2)?)?;
    let a = Scaled::from_value(a.real_component())?;
    let c = Scaled::from_value(c.real_component())?;
    let middle = a.add(c)?.mul(half)?;
    let difference = a.sub(c)?.mul(half)?;
    let bm = crate::helpers::scaled(b)?.magnitude()?;
    let radius = difference.mul(difference)?.add(bm.mul(bm)?)?.sqrt()?;
    // The eigenvalues are middle +/- radius. The one of larger magnitude adds
    // terms of the same sign, and the other follows from the product of the
    // eigenvalues (det / large) instead of the cancelling difference.
    let large = if middle.is_sign_negative() {
        middle.sub(radius)?
    } else {
        middle.add(radius)?
    };
    if large.is_zero() {
        return Ok([large, large]);
    }
    let br = Scaled::from_value(b.real_component())?;
    let bi = Scaled::from_value(b.imaginary_component())?;
    let det = a.difference_of_products(c, br, br)?.sub(bi.mul(bi)?)?;
    Ok([large, det.div(large)?])
}

/// Solves `[a conj(b); b c] [u; v] = [x; y]` by Cramer's rule and returns
/// `[u, v]` as scaled complex values.
///
/// The determinant and the numerators `c x − conj(b) y` and `a y − b x` are
/// differences of products, each formed with one rounding. The function does
/// not test the conditioning of the block, so a nonzero determinant does not
/// guarantee a small residual.
///
/// # Errors
///
/// - `Singular` if the computed determinant is zero.
/// - `NonFinite` if a component of the block, `x` or `y` is NaN or infinite.
/// - `InvalidInput` if `a` or `c` has a nonzero imaginary part.
/// - `NumericalInstability` if a result exceeds the exponent range of
///   [`Scaled`].
///
/// ```
/// use scioxide_core::Complex64;
/// use scioxide_sparse::backend::hermitian::solve_block;
///
/// // diag(2, 4) [u; v] = [6; 8] gives u = 3, v = 2.
/// let solution = solve_block(2.0_f64, 0.0, 4.0, 6.0, 8.0)?;
/// assert_eq!(solution[0].round()?, Complex64::new(3.0, 0.0));
/// assert_eq!(solution[1].round()?, Complex64::new(2.0, 0.0));
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn solve_block<S: NumericScalar>(
    a: S,
    b: S,
    c: S,
    x: S,
    y: S,
) -> Result<[ScaledValue<scioxide_core::Complex<S::Real>>; 2], SciOxideError> {
    let det = ScaledValue::from_components(determinant(a, b, c)?, Scaled::ZERO);
    if det.is_zero() {
        return Err(scioxide_core::SciOxideError::singular(None));
    }
    // Cramer's rule for [a conj(b); b c] [u; v] = [x; y]: the numerators
    // c x - conj(b) y and a y - b x are single-rounding differences of products.
    Ok([
        crate::helpers::scaled(x)?
            .difference_of_products(
                crate::helpers::scaled(c)?,
                crate::helpers::scaled(b.conj())?,
                crate::helpers::scaled(y)?,
            )?
            .div(det)?,
        crate::helpers::scaled(y)?
            .difference_of_products(
                crate::helpers::scaled(a)?,
                crate::helpers::scaled(b)?,
                crate::helpers::scaled(x)?,
            )?
            .div(det)?,
    ])
}

fn validate_block<S: NumericScalar>(a: S, b: S, c: S) -> Result<(), SciOxideError> {
    if [a, b, c].iter().any(|value| {
        !value.real_component().is_finite() || !value.imaginary_component().is_finite()
    }) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "Hermitian block components must be finite",
        ));
    }
    if a.imaginary_component() != S::Real::ZERO || c.imaginary_component() != S::Real::ZERO {
        return Err(SciOxideError::invalid_input_static(
            scioxide_core::argument!("diagonal"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Hermitian diagonals must be real",
        ));
    }
    Ok(())
}
