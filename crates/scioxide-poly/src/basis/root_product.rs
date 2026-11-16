//! Root product representation and operations.
use super::*;

pub(super) fn balanced_coefficients_from_roots<C: NumericScalar>(
    roots: &[C],
) -> Result<Vec<C>, SciOxideError> {
    roots
        .iter()
        .copied()
        .try_for_each(|root| finite_input(root, "root must have finite components").map(drop))?;
    // Leja ordering distributes roots over subproducts instead of separating
    // negative and positive roots into cancelling large coefficient vectors.
    let mut ordered = try_copy_slice(roots, scioxide_core::quantity!("ordered_roots"))?;
    leja_order(&mut ordered, |root| *root)?;
    let mut factors = try_with_capacity(
        ordered.len(),
        scioxide_core::quantity!("balanced_polynomial_root_factors"),
    )?;
    for root in ordered {
        factors.push(scaled_factor(&[-root, C::ONE])?);
    }
    round_monic_product(balanced_product(factors)?)
}

pub(super) fn scaled_factor<C: NumericScalar>(
    coefficients: &[C],
) -> Result<Vec<ScaledValue<C>>, SciOxideError> {
    let mut scaled = try_with_capacity(
        coefficients.len(),
        scioxide_core::quantity!("scaled_polynomial_factor"),
    )?;
    for &coefficient in coefficients {
        scaled.push(ScaledValue::from_value(coefficient)?);
    }
    Ok(scaled)
}

// Canonical ties make the greedy Leja sequence permutation invariant. Extended
// distance products avoid overflow and distribute roots over each subproduct.
pub(super) fn leja_order<C: NumericScalar, T>(
    factors: &mut [T],
    root: impl Fn(&T) -> C,
) -> Result<(), SciOxideError> {
    factors.sort_unstable_by(|left, right| crate::util::compare_complex(&root(left), &root(right)));
    if native_leja_order(factors, &root)? {
        return Ok(());
    }
    // Restart from the same canonical order after any lost range. Scratch and
    // partially selected factors must not influence the extended sequence.
    factors.sort_unstable_by(|left, right| crate::util::compare_complex(&root(left), &root(right)));
    let mut scores = crate::util::try_collect(
        factors
            .iter()
            .map(|factor| scaled_complex(root(factor))?.magnitude()),
        scioxide_core::quantity!("root_order_scores"),
    )?;
    for i in 0..factors.len() {
        let mut best = i;
        for j in i + 1..factors.len() {
            let comparison = scores[j].abs_cmp(scores[best]);
            if comparison.is_gt()
                || (comparison.is_eq()
                    && crate::util::compare_complex(&root(&factors[j]), &root(&factors[best]))
                        .is_lt())
            {
                best = j;
            }
        }
        factors.swap(i, best);
        scores.swap(i, best);
        let selected = scaled_complex(root(&factors[i]))?;
        for (factor, score) in factors[i + 1..].iter().zip(&mut scores[i + 1..]) {
            let distance = scaled_complex(root(factor))?.sub(selected)?.magnitude()?;
            *score = if i == 0 {
                distance
            } else {
                score.mul(distance)?
            };
        }
    }
    Ok(())
}

// Common binary rescaling preserves every comparison of the distance products.
// Native products are admitted only while normal (or an exact duplicate zero).
// Complex distances use the same sqrt-range admission as monomial evaluation.
pub(super) fn native_leja_order<C: NumericScalar, T>(
    factors: &mut [T],
    root: &impl Fn(&T) -> C,
) -> Result<bool, SciOxideError> {
    use scioxide_core::backend::standard;
    let safe = crate::admission::native_value;
    if !factors.iter().all(|factor| safe(root(factor))) {
        return Ok(false);
    }
    let mut scores = crate::util::try_collect(
        factors
            .iter()
            .map(|factor| Ok(crate::admission::native_magnitude(root(factor)))),
        scioxide_core::quantity!("native_root_order_scores"),
    )?;
    for i in 0..factors.len() {
        let mut best = i;
        for j in i + 1..factors.len() {
            if scores[j] > scores[best]
                || (scores[j] == scores[best]
                    && crate::util::compare_complex(&root(&factors[j]), &root(&factors[best]))
                        .is_lt())
            {
                best = j;
            }
        }
        factors.swap(i, best);
        scores.swap(i, best);
        let maximum = scores[i];
        if maximum != C::Real::ZERO && !crate::admission::native_range(maximum) {
            let exponent = standard::exponent(maximum)?;
            for score in &mut scores[i + 1..] {
                let scaled = score.scalbn(-exponent);
                if *score != C::Real::ZERO && scaled.abs() < C::Real::MIN_POSITIVE {
                    return Ok(false);
                }
                *score = scaled;
            }
        }
        let selected = root(&factors[i]);
        for (factor, score) in factors[i + 1..].iter().zip(&mut scores[i + 1..]) {
            let difference = root(factor) - selected;
            if !safe(difference) {
                return Ok(false);
            }
            let distance = crate::admission::native_magnitude(difference);
            let candidate = if i == 0 { distance } else { *score * distance };
            if i != 0 && standard::product_needs_retry(*score, distance, candidate) {
                return Ok(false);
            }
            *score = candidate;
        }
    }
    Ok(true)
}

pub(super) fn balanced_product<C: NumericScalar>(
    mut level: Vec<Vec<ScaledValue<C>>>,
) -> Result<Vec<ScaledValue<C>>, SciOxideError> {
    if level.is_empty() {
        return try_filled(
            1,
            ScaledValue::ONE,
            scioxide_core::quantity!("constant_root_product"),
        );
    }
    while level.len() > 1 {
        let mut next = try_with_capacity(
            level.len().div_ceil(2),
            scioxide_core::quantity!("balanced_polynomial_product"),
        )?;
        let mut factors = level.into_iter();
        while let Some(left) = factors.next() {
            next.push(match factors.next() {
                Some(right) => crate::fast::multiply_scaled(&left, &right)?,
                None => left,
            });
        }
        level = next;
    }
    level.pop().ok_or_else(|| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "balanced product tree lost its final factor",
        )
    })
}

pub(super) fn round_monic_product<C: NumericScalar>(
    coefficients: Vec<ScaledValue<C>>,
) -> Result<Vec<C>, SciOxideError> {
    let mut rounded = try_with_capacity(
        coefficients.len(),
        scioxide_core::quantity!("polynomial_from_roots"),
    )?;
    for coefficient in coefficients {
        // Every leaf is monic, so the product's leading coefficient is one.
        rounded.push(coefficient.round()?);
    }
    Ok(rounded)
}
