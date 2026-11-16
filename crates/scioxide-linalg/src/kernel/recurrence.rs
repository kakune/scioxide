//! Fused reductions and updates of finite private recurrence vectors.
//!
//! Bounds belong to a particular vector value, not its storage. Callers replace
//! them after every mutation and move them with swapped recurrence vectors.

use scioxide_core::backend::KernelDispatch;
use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

use super::standard;

/// Bound on the lane partial sums and tail terms a vector Dot2 leaf combines
/// after its lanes, added to the term count of [`Products::compensated_error`].
pub(crate) const DOT2_PARTIALS: usize = 64;

/// Unrounded range bounds and an ordinary candidate for the squared norm.
/// The square may overflow or underflow; it is used only after its range proof.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Moments<R: RealScalar> {
    pub(crate) maximum: R,
    pub(crate) square: R,
}

impl<R: RealScalar> Moments<R> {
    pub(crate) fn norm<S: NumericScalar<Real = R>>(self, values: &[S]) -> Result<R, SciOxideError> {
        if ordinary(self.maximum, self.maximum, values.len())? {
            crate::checked::finite_candidate(self.square.sqrt())
        } else {
            standard::norm_workspace(values.iter().copied())
        }
    }

    pub(crate) fn squared<S: NumericScalar<Real = R>>(
        self,
        values: &[S],
    ) -> Result<ScaledValue<S>, SciOxideError> {
        if ordinary(self.maximum, self.maximum, values.len())? {
            ScaledValue::from_value(S::from_real(self.square))
        } else {
            scaled_dot(values, values)
        }
    }
}

/// Candidates and bounds accumulated together, before any range decision.
pub(crate) struct Products<S: NumericScalar> {
    pub(crate) value: S,
    pub(crate) left: Moments<S::Real>,
    pub(crate) right: Moments<S::Real>,
}

impl<S: NumericScalar> Products<S> {
    pub(crate) fn scaled(&self, left: &[S], right: &[S]) -> Result<ScaledValue<S>, SciOxideError> {
        if ordinary(self.left.maximum, self.right.maximum, left.len())? {
            ScaledValue::from_value(self.value)
        } else {
            scaled_dot(left, right)
        }
    }

    /// A priori absolute error `E` of a [`compensated_products`] candidate `ŝ`,
    /// or `None` outside the ordinary frame, where only exact accumulation is
    /// valid.
    ///
    /// Let `k` count the real products per component (`n` real, `2n` complex)
    /// plus [`DOT2_PARTIALS`], `u = epsilon/2`, `gamma_k = k u/(1 - k u)` and
    /// `eta` the smallest subnormal. Ogita, Rump and Oishi bound each component
    /// by `u|s| + gamma_k² sum|terms|`, and each TwoProduct residual below the
    /// subnormal grid adds at most `eta/2`. Hence
    /// `|ŝ - s| <= E = 2u|ŝ| + 4 gamma_k² ||left||₂ ||right||₂ + k eta`, where the
    /// factor four covers both complex components and the rounding of the
    /// supplied norms. `k u > 1/4` also returns `None`.
    pub(crate) fn compensated_error(
        &self,
        left_norm: S::Real,
        right_norm: S::Real,
        len: usize,
    ) -> Result<Option<Scaled<S::Real>>, SciOxideError> {
        if !ordinary(self.left.maximum, self.right.maximum, len)? {
            return Ok(None);
        }
        let terms = len
            .checked_mul(if S::IS_REAL { 1 } else { 2 })
            .and_then(|terms| terms.checked_add(DOT2_PARTIALS))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("dot2_terms")))?;
        let count = crate::checked::rounded_count::<S::Real>(terms)?;
        let unit = S::Real::EPSILON / S::Real::TWO;
        let ku = count * unit;
        if ku > S::Real::ONE / (S::Real::TWO * S::Real::TWO) {
            return Ok(None);
        }
        let gamma = Scaled::from_value(ku / (S::Real::ONE - ku))?;
        let subnormal = S::Real::MIN_POSITIVE * S::Real::EPSILON;
        Scaled::from_value(S::Real::TWO * S::Real::TWO)?
            .mul(gamma.square()?)?
            .mul(Scaled::from_value(left_norm)?)?
            .mul(Scaled::from_value(right_norm)?)?
            .add(
                Scaled::from_value(S::Real::TWO * unit)?
                    .mul(ScaledValue::from_value(self.value)?.magnitude()?)?,
            )?
            .add(Scaled::from_value(count * subnormal)?)
            .map(Some)
    }
}

/// Same accumulation headroom and gradual-underflow guard as the extended
/// recurrence dot. Zero vectors also accept the ordinary (exact zero) result.
fn ordinary<R: RealScalar>(a: R, b: R, n: usize) -> Result<bool, SciOxideError> {
    if a == R::ZERO || b == R::ZERO {
        return Ok(true);
    }
    let product = a * b;
    let length = crate::checked::rounded_count::<R>(n.max(1))?;
    Ok(product.is_normal()
        && (product * length * R::try_from_usize_exact(8)?).is_finite()
        && (product * R::EPSILON).is_normal()
        && standard::product_scales(a, b, n)? == (0, 0))
}

fn scaled_dot<S: NumericScalar>(left: &[S], right: &[S]) -> Result<ScaledValue<S>, SciOxideError> {
    let mut real = scioxide_core::ScaledAccumulator::ZERO;
    let mut imaginary = scioxide_core::ScaledAccumulator::ZERO;
    for (&left, &right) in left.iter().zip(right) {
        let value = ScaledValue::from_value(left)?.mul(ScaledValue::from_value(right.conj())?)?;
        real.add_scaled(value.real())?;
        imaginary.add_scaled(value.imaginary())?;
    }
    Ok(ScaledValue::from_components(
        real.total()?,
        imaginary.total()?,
    ))
}

/// `left * conj(right)`, both squared norms and both component bounds in one
/// traversal. Inputs have already been validated by staging or their producer.
pub(crate) fn products<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<Products<S>, SciOxideError> {
    if left.len() != right.len() {
        return Err(SciOxideError::dimension_mismatch(
            [left.len()],
            [right.len()],
        ));
    }
    if let Some(result) = super::arch::recurrence::products(left, right, false, dispatch)? {
        return Ok(result);
    }
    let (sums, left, right) = traverse(left, right, [S::ZERO; 4], |mut sums, lane, a, b| {
        sums[lane] = a * b.conj() + sums[lane];
        sums
    });
    Ok(Products {
        value: (sums[0] + sums[1]) + (sums[2] + sums[3]),
        left,
        right,
    })
}

/// [`products`] with the Ogita–Rump–Oishi Dot2 candidate of `left * conj(right)`.
///
/// TwoProduct (one fused residual) and TwoSum are error-free in the ordinary
/// frame, and their residuals accumulate in working precision, so the candidate
/// is as accurate as twice-working-precision evaluation rounded once. Complex
/// components are real Dot2 sums of `re = ar br + ai bi` and
/// `im = ai br - ar bi`. [`Products::compensated_error`] bounds the candidate.
pub(crate) fn compensated_products<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<Products<S>, SciOxideError> {
    if left.len() != right.len() {
        return Err(SciOxideError::dimension_mismatch(
            [left.len()],
            [right.len()],
        ));
    }
    if let Some(result) = super::arch::recurrence::products(left, right, true, dispatch)? {
        return Ok(result);
    }
    let ((real, imaginary), left, right) = traverse(
        left,
        right,
        (Dot2::ZERO, Dot2::ZERO),
        |(real, imaginary), _, a, b| {
            let (ar, ai) = (a.real_component(), a.imaginary_component());
            let (br, bi) = (b.real_component(), b.imaginary_component());
            if S::IS_REAL {
                (real.add_product(ar, br), imaginary)
            } else {
                (
                    real.add_product(ar, br).add_product(ai, bi),
                    imaginary.add_product(ai, br).add_product(-ar, bi),
                )
            }
        },
    );
    Ok(Products {
        value: S::from_components(real.total(), imaginary.total()),
        left,
        right,
    })
}

/// One portable traversal folding `value` over four lanes with both moments.
fn traverse<S: NumericScalar, V>(
    left: &[S],
    right: &[S],
    mut value: V,
    accumulate: impl Fn(V, usize, S, S) -> V,
) -> (V, Moments<S::Real>, Moments<S::Real>) {
    let mut left_squares = [S::Real::ZERO; 4];
    let mut right_squares = [S::Real::ZERO; 4];
    let mut lm = S::Real::ZERO;
    let mut rm = S::Real::ZERO;
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        let lane = i % 4;
        value = accumulate(value, lane, a, b);
        left_squares[lane] = square_add(a, left_squares[lane]);
        right_squares[lane] = square_add(b, right_squares[lane]);
        lm = lm.max(standard::magnitude(a));
        rm = rm.max(standard::magnitude(b));
    }
    (
        value,
        Moments {
            maximum: lm,
            square: sum_four(left_squares),
        },
        Moments {
            maximum: rm,
            square: sum_four(right_squares),
        },
    )
}

/// Ogita–Rump–Oishi Dot2 state: an error-free running sum and the working
/// precision sum of every TwoSum and TwoProduct residual.
#[derive(Clone, Copy)]
pub(crate) struct Dot2<R: RealScalar> {
    sum: R,
    error: R,
}

impl<R: RealScalar> Dot2<R> {
    pub(crate) const ZERO: Self = Self {
        sum: R::ZERO,
        error: R::ZERO,
    };

    /// TwoSum `value` into the sum; `residual` joins the residual sum.
    #[inline]
    pub(crate) fn add_term(self, value: R, residual: R) -> Self {
        let sum = self.sum + value;
        let virtual_value = sum - self.sum;
        let rounding = (self.sum - (sum - virtual_value)) + (value - virtual_value);
        Self {
            sum,
            error: self.error + (rounding + residual),
        }
    }

    /// TwoProduct `a * b`, whose fused residual is exact above the subnormal grid.
    #[inline]
    pub(crate) fn add_product(self, a: R, b: R) -> Self {
        let product = a * b;
        self.add_term(product, a.mul_add(b, -product))
    }

    pub(crate) fn total(self) -> R {
        self.sum + self.error
    }
}

#[inline]
fn square_add<S: NumericScalar>(x: S, sum: S::Real) -> S::Real {
    let sum = x.real_component() * x.real_component() + sum;
    if S::IS_REAL {
        sum
    } else {
        x.imaginary_component() * x.imaginary_component() + sum
    }
}

fn sum_four<R: RealScalar>(s: [R; 4]) -> R {
    (s[0] + s[1]) + (s[2] + s[3])
}

/// `output = alpha * input + beta * output`, using producer-established bounds.
/// Output statistics and unchanged-update detection share the writing pass.
pub(crate) fn update<S: NumericScalar>(
    output: &mut [S],
    alpha: S,
    input: &[S],
    beta: S,
    bounds: (S::Real, S::Real),
    dispatch: KernelDispatch,
) -> Result<(bool, Moments<S::Real>), SciOxideError> {
    if output.len() != input.len() {
        return Err(SciOxideError::dimension_mismatch(
            [output.len()],
            [input.len()],
        ));
    }
    crate::checked::finite_candidate(alpha)?;
    crate::checked::finite_candidate(beta)?;
    let affine = standard::Affine::new(alpha, bounds.0, beta, bounds.1)?;
    if affine.is_unscaled()
        && let Some(result) =
            super::arch::recurrence::update(output, alpha, Some(input), beta, dispatch)?
    {
        return Ok(result);
    }
    let mut squares = [S::Real::ZERO; 4];
    let mut maximum = S::Real::ZERO;
    let mut changed = false;
    for (i, (out, &value)) in output.iter_mut().zip(input).enumerate() {
        let next = affine.apply(value, *out);
        let next = if affine.publication == standard::Publication::Checked {
            crate::checked::finite_candidate(next)?
        } else {
            next
        };
        changed |= next != *out;
        *out = next;
        maximum = maximum.max(standard::magnitude(next));
        squares[i % 4] = square_add(next, squares[i % 4]);
    }
    Ok((
        changed,
        Moments {
            maximum,
            square: sum_four(squares),
        },
    ))
}

/// Dispatched scaling with the same checked division fallback for tiny divisors.
pub(crate) fn normalize<S: NumericScalar>(
    values: &mut [S],
    norm: S::Real,
    maximum: S::Real,
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let reciprocal = S::Real::ONE / norm;
    if reciprocal.is_normal() {
        let factor = S::from_real(reciprocal);
        let affine = standard::Affine::new(factor, maximum, S::ZERO, S::Real::ZERO)?;
        if affine.is_unscaled()
            && super::arch::recurrence::update(values, S::ZERO, None, factor, dispatch)?.is_some()
        {
            return Ok(());
        }
        for value in values {
            *value = crate::checked::finite_candidate(*value * factor)?;
        }
    } else {
        if !norm.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Intermediate,
                "norm division inputs must be finite",
            ));
        }
        if norm == S::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "norm denominator is zero",
            ));
        }
        for value in values {
            let quotient = *value / S::from_real(norm);
            if !quotient.has_finite_components() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "division by norm did not produce a finite value",
                ));
            }
            *value = quotient;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::backend::KernelBackend;
    use scioxide_core::{Complex, ProductSum};

    fn backends() -> impl Iterator<Item = KernelDispatch> {
        let processor = crate::kernel::tests::test_processor();
        KernelBackend::available(processor)
            .map(move |backend| KernelDispatch::for_backend(backend, processor).unwrap())
    }

    #[test]
    fn native_boundaries_reject_mismatched_spans_without_writing() {
        for dispatch in backends() {
            for compensated in [false, true] {
                assert!(
                    super::super::arch::recurrence::products(
                        &[1_f64],
                        &[1., 2.],
                        compensated,
                        dispatch
                    )
                    .is_err()
                );
                assert!(
                    super::super::arch::recurrence::products(
                        &[1_f32, 2.],
                        &[1.],
                        compensated,
                        dispatch
                    )
                    .is_err()
                );
            }
            let mut out = [3_f64];
            assert!(
                super::super::arch::recurrence::update(&mut out, 1., Some(&[1., 2.]), 1., dispatch)
                    .is_err()
            );
            assert_eq!(out, [3.]);
            let mut out = [3_f32, 4.];
            assert!(
                super::super::arch::recurrence::update(&mut out, 1., Some(&[1.]), 1., dispatch)
                    .is_err()
            );
            assert_eq!(out, [3., 4.]);
        }
    }

    fn moderate<S: NumericScalar>() {
        let real = |x| S::Real::try_from_f64_lossy(x).unwrap();
        let scalar = |i: usize| {
            S::from_components(
                real((i % 17) as f64 / 8. - 1.),
                if S::IS_REAL {
                    S::Real::ZERO
                } else {
                    real((i % 13) as f64 / 16. - 0.25)
                },
            )
        };
        for dispatch in backends() {
            for n in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129] {
                let a: alloc::vec::Vec<_> = (0..n + 2).map(scalar).collect();
                let b: alloc::vec::Vec<_> = (7..n + 9).map(scalar).collect();
                let (a, b) = (&a[1..n + 1], &b[1..n + 1]);
                let products = products(a, b, dispatch).unwrap();
                let mut exact = ProductSum::new();
                for (&a, &b) in a.iter().zip(b) {
                    exact.add_product(a, b.conj()).unwrap();
                }
                let expected: S = exact.finish().unwrap();
                let tolerance = real(128. * (n + 1) as f64) * S::Real::EPSILON;
                assert!(
                    (products.scaled(a, b).unwrap().round().unwrap() - expected).abs() <= tolerance
                );
                assert_eq!(
                    products.left.maximum,
                    a.iter()
                        .fold(S::Real::ZERO, |m, &v| m.max(standard::magnitude(v)))
                );
                assert_eq!(
                    products.right.maximum,
                    b.iter()
                        .fold(S::Real::ZERO, |m, &v| m.max(standard::magnitude(v)))
                );
                assert!(
                    (products.left.norm(a).unwrap()
                        - standard::norm_workspace(a.iter().copied()).unwrap())
                    .abs()
                        <= tolerance
                );
                assert!(
                    (products.right.norm(b).unwrap()
                        - standard::norm_workspace(b.iter().copied()).unwrap())
                    .abs()
                        <= tolerance
                );

                let alpha = S::from_real(real(-0.75));
                let beta = S::from_real(real(0.5));
                let mut out = b.to_vec();
                let (changed, moments) = update(
                    &mut out,
                    alpha,
                    a,
                    beta,
                    (products.left.maximum, products.right.maximum),
                    dispatch,
                )
                .unwrap();
                let mut expected_changed = false;
                for ((&a, &b), &actual) in a.iter().zip(b).zip(&out) {
                    let mut sum = ProductSum::new();
                    sum.add_product(alpha, a).unwrap();
                    sum.add_product(beta, b).unwrap();
                    let expected: S = sum.finish().unwrap();
                    assert!((actual - expected).abs() <= tolerance);
                    expected_changed |= actual != b;
                }
                assert_eq!(changed, expected_changed);
                assert_eq!(
                    moments.maximum,
                    out.iter()
                        .fold(S::Real::ZERO, |m, &v| m.max(standard::magnitude(v)))
                );
                assert!(
                    (moments.norm(&out).unwrap()
                        - standard::norm_workspace(out.iter().copied()).unwrap())
                    .abs()
                        <= tolerance
                );
                if n > 0 {
                    let norm = moments.norm(&out).unwrap();
                    normalize(&mut out, norm, moments.maximum, dispatch).unwrap();
                    assert!(
                        (standard::norm_workspace(out.iter().copied()).unwrap() - S::Real::ONE)
                            .abs()
                            <= tolerance
                    );
                }
                let bound = out
                    .iter()
                    .fold(S::Real::ZERO, |m, &v| m.max(standard::magnitude(v)));
                let before = out.clone();
                assert!(
                    !update(
                        &mut out,
                        S::ZERO,
                        a,
                        S::ONE,
                        (products.left.maximum, bound),
                        dispatch
                    )
                    .unwrap()
                    .0
                );
                assert_eq!(out, before);
            }
        }
    }

    #[test]
    fn fused_real32() {
        moderate::<f32>();
    }
    #[test]
    fn fused_real64() {
        moderate::<f64>();
    }
    #[test]
    fn fused_complex32() {
        moderate::<Complex<f32>>();
    }
    #[test]
    fn fused_complex64() {
        moderate::<Complex<f64>>();
    }

    fn extreme<S: NumericScalar>(small: f64, large: f64) {
        let real = |x| S::Real::try_from_f64_lossy(x).unwrap();
        for dispatch in backends() {
            for scale in [small, large] {
                let a = [S::from_real(real(scale)), S::from_real(real(-scale))];
                let product = products(&a, &a, dispatch).unwrap();
                let scale = ScaledValue::from_value(a[0]).unwrap();
                let ratio = product
                    .scaled(&a, &a)
                    .unwrap()
                    .div(scale)
                    .unwrap()
                    .div(scale)
                    .unwrap()
                    .round()
                    .unwrap();
                assert!((ratio - S::from_real(real(2.))).abs() <= real(8.) * S::Real::EPSILON);
                let length = product.left.norm(&a).unwrap();
                assert!(
                    (length / real(2.).sqrt() / a[0].real_component() - S::Real::ONE).abs()
                        <= real(8.) * S::Real::EPSILON
                );
                let mut normalized = a;
                normalize(&mut normalized, length, product.left.maximum, dispatch).unwrap();
                assert!(
                    (standard::norm_workspace(normalized.into_iter()).unwrap() - S::Real::ONE)
                        .abs()
                        <= real(8.) * S::Real::EPSILON
                );
            }
            let maximum = S::from_real(S::Real::MAX);
            let a = [maximum, maximum];
            let b = [S::from_real(real(2.)), S::from_real(real(-2.))];
            assert_eq!(
                products(&a, &b, dispatch)
                    .unwrap()
                    .scaled(&a, &b)
                    .unwrap()
                    .round()
                    .unwrap(),
                S::ZERO
            );
            let mut out = [-maximum];
            update(
                &mut out,
                S::from_real(real(2.)),
                &[maximum],
                S::ONE,
                (S::Real::MAX, S::Real::MAX),
                dispatch,
            )
            .unwrap();
            assert_eq!(out, [maximum]);
            assert!(
                update(
                    &mut out,
                    S::from_real(real(2.)),
                    &[maximum],
                    S::ZERO,
                    (S::Real::MAX, S::Real::MAX),
                    dispatch
                )
                .is_err()
            );
            let tiny = S::from_real(S::Real::MIN_POSITIVE * S::Real::EPSILON);
            let mut out = [S::ZERO];
            let (_, moments) = update(
                &mut out,
                S::ONE,
                &[tiny],
                S::ZERO,
                (standard::magnitude(tiny), S::Real::ZERO),
                dispatch,
            )
            .unwrap();
            assert_eq!(out, [tiny]);
            assert_eq!(moments.norm(&out).unwrap(), tiny.real_component());
            assert!(
                update(
                    &mut out,
                    S::from_real(S::Real::NAN),
                    &[tiny],
                    S::ZERO,
                    (standard::magnitude(tiny), S::Real::ZERO),
                    dispatch
                )
                .is_err()
            );
            assert!(products(&out, &[], dispatch).is_err());
            assert!(
                update(
                    &mut out,
                    S::ONE,
                    &[],
                    S::ZERO,
                    (S::Real::ZERO, S::Real::ZERO),
                    dispatch
                )
                .is_err()
            );
        }
    }

    #[test]
    fn extended_real32() {
        extreme::<f32>(1e-30, 1e30);
    }
    #[test]
    fn extended_real64() {
        extreme::<f64>(1e-300, 1e300);
    }
    #[test]
    fn extended_complex32() {
        extreme::<Complex<f32>>(1e-30, 1e30);
    }
    #[test]
    fn extended_complex64() {
        extreme::<Complex<f64>>(1e-300, 1e300);
    }

    fn exact<S: NumericScalar>(left: &[S], right: &[S]) -> ScaledValue<S> {
        let mut sum = ProductSum::new();
        for (&a, &b) in left.iter().zip(right) {
            sum.add_product(a, b.conj()).unwrap();
        }
        let value = scioxide_core::backend::product_sum_scaled(&sum).unwrap();
        ScaledValue::from_components(value.real(), value.imaginary())
    }

    /// `huge + 1` rounds to `huge`, and `(1 + d)² = 1 + 2d + d²` rounds to
    /// `1 + 2d`, so ordinary dots lose both results on every lane layout.
    fn cancellation<R: RealScalar>(huge: R, d: R) {
        for dispatch in backends() {
            for n in [2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
                let mut left = alloc::vec![R::ONE; n];
                let right = alloc::vec![R::ONE; n];
                left[0] = huge;
                left[n - 1] = -huge;
                let products = compensated_products(&left, &right, dispatch).unwrap();
                assert_eq!(
                    products.value,
                    crate::checked::rounded_count::<R>(n - 2).unwrap()
                );
                let mut left = alloc::vec![R::ZERO; n];
                let mut right = alloc::vec![R::ZERO; n];
                (left[0], right[0]) = (R::ONE + d, R::ONE + d);
                (left[n - 1], right[n - 1]) = (-R::ONE, R::ONE + R::TWO * d);
                let products = compensated_products(&left, &right, dispatch).unwrap();
                assert_eq!(products.value, d * d);
            }
        }
    }

    #[test]
    fn compensated_products_recover_lost_sums_and_products() {
        cancellation::<f32>(2_f32.powi(30), 2_f32.powi(-12));
        cancellation::<f64>(2_f64.powi(60), 2_f64.powi(-30));
    }

    /// Ill-conditioned products stay within the documented Dot2 bound, and the
    /// moments match the ordinary traversal.
    fn compensated_bound<S: NumericScalar>() {
        let real = |x| S::Real::try_from_f64_lossy(x).unwrap();
        let scalar = |i: usize| {
            let scale = if i.is_multiple_of(5) { 1e6 } else { 1. };
            S::from_components(
                real(scale * ((i % 17) as f64 / 8. - 1.)),
                if S::IS_REAL {
                    S::Real::ZERO
                } else {
                    real(scale * ((i % 13) as f64 / 16. - 0.25))
                },
            )
        };
        for dispatch in backends() {
            for n in [
                0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 1000,
            ] {
                let a: alloc::vec::Vec<_> = (0..n).map(scalar).collect();
                // Cancel the large entries against their negated partners.
                let b: alloc::vec::Vec<_> = (0..n)
                    .map(|i| {
                        if i.is_multiple_of(10) {
                            -scalar(i + 5)
                        } else {
                            scalar(i + 5)
                        }
                    })
                    .collect();
                let compensated = compensated_products(&a, &b, dispatch).unwrap();
                let ordinary = products(&a, &b, dispatch).unwrap();
                assert_eq!(compensated.left.maximum, ordinary.left.maximum);
                assert_eq!(compensated.right.maximum, ordinary.right.maximum);
                let (left_norm, right_norm) = (
                    compensated.left.norm(&a).unwrap(),
                    compensated.right.norm(&b).unwrap(),
                );
                let tolerance = real(8. * (n + 1) as f64) * S::Real::EPSILON;
                assert!(
                    (left_norm - ordinary.left.norm(&a).unwrap()).abs() <= tolerance * left_norm
                );
                let error = compensated
                    .compensated_error(left_norm, right_norm, n)
                    .unwrap()
                    .unwrap();
                let difference = ScaledValue::from_value(compensated.value)
                    .unwrap()
                    .sub(exact(&a, &b))
                    .unwrap()
                    .magnitude()
                    .unwrap();
                assert!(difference.abs_le(error), "n={n}");
            }
        }
    }

    #[test]
    fn compensated_error_bounds_every_scalar_and_backend() {
        compensated_bound::<f32>();
        compensated_bound::<f64>();
        compensated_bound::<Complex<f32>>();
        compensated_bound::<Complex<f64>>();
    }

    #[test]
    fn compensated_error_requires_the_ordinary_frame() {
        for dispatch in backends() {
            let (left, right) = ([1e300_f64, 1e300], [1e10, -1e10]);
            let products = compensated_products(&left, &right, dispatch).unwrap();
            assert!(
                products
                    .compensated_error(1e300, 1e10, 2)
                    .unwrap()
                    .is_none()
            );
            let zero = compensated_products(&[0_f64; 3], &[1., 2., 3.], dispatch).unwrap();
            assert_eq!(zero.value, 0.);
            assert!(zero.compensated_error(0., 3., 3).unwrap().is_some());
        }
    }
}
