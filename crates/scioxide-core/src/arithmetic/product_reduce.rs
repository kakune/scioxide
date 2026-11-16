//! Correctly rounded sums of products over iterators.
use super::{ProductSum, RealProductSum, certified_short_sum_products};
use crate::{NumericScalar, RealScalar, SciOxideError};

/// Computes `Σ aᵢ bᵢ` over an iterator of finite real or complex pairs, with
/// each component rounded once.
///
/// The result equals that of [`ProductSum`](crate::ProductSum): exact
/// accumulation followed by one rounding to nearest, ties to even. The iterator
/// is consumed once, up to its first `None`. Cost is `O(n)` time with a fixed
/// stack block of 64 pairs. An empty input or exact cancellation gives `+0`; a
/// nonzero sum that underflows keeps its sign.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
///   infinite, even when the other factor of its pair is zero.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the rounded sum is
///   outside the finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::checked_product_sum;
///
/// let pairs = [(1e100_f64, 1.0), (1.0, 1.0), (-1e100, 1.0)];
/// assert_eq!(checked_product_sum(pairs)?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_product_sum<T: NumericScalar>(
    products: impl IntoIterator<Item = (T, T)>,
) -> Result<T, SciOxideError> {
    if T::IS_REAL {
        return real_sum(
            products
                .into_iter()
                .map(|(a, b)| (a.real_component(), b.real_component())),
        )
        .map(T::from_real);
    }
    let mut sum = ProductSum::new();
    for (a, b) in products {
        sum.add_product(a, b)?;
    }
    sum.finish()
}

#[inline]
fn real_sum<S: RealScalar>(products: impl Iterator<Item = (S, S)>) -> Result<S, SciOxideError> {
    let mut products = products;
    let mut prefix = [(S::ZERO, S::ZERO); 4];
    let mut length = 0;
    for (slot, pair) in prefix.iter_mut().zip(products.by_ref()) {
        *slot = pair;
        length += 1;
    }
    let next = if length == prefix.len() {
        products.next()
    } else {
        None
    };
    if next.is_none()
        && let Some(value) = certified_short_sum_products(&prefix[..length])
    {
        return Ok(value);
    }
    let mut sum = RealProductSum::new();
    if let Some(next) = next {
        add_blocks(&mut sum, prefix, next, products)?;
    } else {
        for &(a, b) in &prefix[..length] {
            sum.add_product(a, b)?;
        }
    }
    sum.finish()
}

fn add_blocks<S: RealScalar>(
    sum: &mut RealProductSum<S>,
    prefix: [(S, S); 4],
    fifth: (S, S),
    mut products: impl Iterator<Item = (S, S)>,
) -> Result<(), SciOxideError> {
    let mut pairs = [(S::ZERO, S::ZERO); 64];
    pairs[..prefix.len()].copy_from_slice(&prefix);
    pairs[4] = fifth;
    let mut start = 5;
    loop {
        let mut length = start;
        for (slot, pair) in pairs[start..].iter_mut().zip(products.by_ref()) {
            *slot = pair;
            length += 1;
        }
        if length == 0 {
            return Ok(());
        }
        sum.add_block(&pairs[..length])?;
        if length < pairs.len() {
            return Ok(());
        }
        start = 0;
    }
}
