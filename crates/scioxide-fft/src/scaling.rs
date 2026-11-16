//! Binary range scaling, finite maxima and all-or-nothing restoration.
use crate::validation::arithmetic_overflow;
use scioxide_core::{Complex, RealScalar, SciOxideError};
/// Exponent making a nonzero finite maximum lie in `[0.5, 1)`.
/// Zero has exponent zero; no power of two is materialized, including at
/// the smallest subnormal and largest finite scalar boundaries.
pub(crate) fn range_exponent<S: RealScalar>(maximum: S) -> i32 {
    match maximum.decompose() {
        scioxide_core::FloatDecomposition::Finite { significand: 0, .. } => 0,
        scioxide_core::FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } => i32::from(exponent) + significand.ilog2() as i32 + 1,
        _ => unreachable!("range scaling requires a finite maximum"),
    }
}

/// One prepared binary scaling operation. A representable power of two gives
/// exactly the same IEEE rounding as `scalbn`, including subnormal results.
/// Exponents outside that range use core's single-rounding implementation.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BinaryScale<S> {
    Identity,
    Multiply(S),
    Extended(i32),
}
impl<S: RealScalar> BinaryScale<S> {
    pub(crate) fn new(exponent: i32) -> Self {
        if exponent == 0 {
            Self::Identity
        } else {
            let factor = S::ONE.scalbn(exponent);
            if factor.is_finite() && factor != S::ZERO {
                Self::Multiply(factor)
            } else {
                Self::Extended(exponent)
            }
        }
    }
    /// The ordinary power-of-two factor, or `None` for an extended exponent.
    /// Multiplying by one reproduces [`Self::Identity`] bit for bit.
    #[inline]
    pub(crate) fn factor(self) -> Option<S> {
        match self {
            Self::Identity => Some(S::ONE),
            Self::Multiply(factor) => Some(factor),
            Self::Extended(_) => None,
        }
    }
    #[inline]
    pub(crate) fn apply(self, value: S) -> S {
        match self {
            Self::Identity => value,
            Self::Multiply(factor) => value * factor,
            Self::Extended(exponent) => value.scalbn(exponent),
        }
    }
    /// Restore with one rounding, making exact zeros positive and keeping
    /// the sign of nonzero components that underflow.
    #[inline]
    pub(crate) fn restore(self, value: S) -> S {
        if value == S::ZERO {
            S::ZERO
        } else {
            self.apply(value)
        }
    }
    #[inline]
    pub(crate) fn restore_complex(self, value: Complex<S>) -> Complex<S> {
        Complex::new(self.restore(value.re), self.restore(value.im))
    }
    /// Range-scale each destination/value pair without canonicalizing zeros.
    /// Selecting the binary arm once keeps staging free of per-value dispatch.
    pub(crate) fn apply_all<'a>(self, pairs: impl Iterator<Item = (&'a mut Complex<S>, Complex<S>)>)
    where
        S: 'a,
    {
        fn each<'a, S: RealScalar + 'a>(
            pairs: impl Iterator<Item = (&'a mut Complex<S>, Complex<S>)>,
            scale: impl Fn(S) -> S,
        ) {
            for (destination, value) in pairs {
                *destination = Complex::new(scale(value.re), scale(value.im));
            }
        }
        match self {
            Self::Identity => each(pairs, |value| value),
            Self::Multiply(factor) => each(pairs, |value| value * factor),
            Self::Extended(exponent) => each(pairs, |value| value.scalbn(exponent)),
        }
    }

    /// Restore a candidate in place through the shared publication arithmetic.
    pub(crate) fn restore_in_place(self, values: &mut [Complex<S>]) {
        self.restore_all(in_place(values));
    }

    /// Apply [`Self::restore_complex`] to each `(destination, value)` pair,
    /// selecting the scaling arm once rather than per component.
    pub(crate) fn restore_all<'a>(
        self,
        pairs: impl Iterator<Item = (&'a mut Complex<S>, Complex<S>)>,
    ) where
        S: 'a,
    {
        fn each<'a, S: RealScalar + 'a>(
            pairs: impl Iterator<Item = (&'a mut Complex<S>, Complex<S>)>,
            scale: impl Fn(S) -> S,
        ) {
            let restore = |value: S| {
                if value == S::ZERO {
                    S::ZERO
                } else {
                    scale(value)
                }
            };
            for (destination, value) in pairs {
                *destination = Complex::new(restore(value.re), restore(value.im));
            }
        }
        match self {
            Self::Identity => each(pairs, |value| value),
            Self::Multiply(factor) => each(pairs, |value| value * factor),
            Self::Extended(exponent) => each(pairs, |value| value.scalbn(exponent)),
        }
    }
}

/// Pair each mutable destination with its current value for in-place mapping.
pub(crate) fn in_place<S: Copy>(
    values: &mut [Complex<S>],
) -> impl Iterator<Item = (&mut Complex<S>, Complex<S>)> {
    values.iter_mut().map(|value| {
        let current = *value;
        (value, current)
    })
}

/// Validate and measure complex components, reporting non-finite values as input errors.
pub(crate) fn complex_max<S: RealScalar>(
    values: impl IntoIterator<Item = Complex<S>>,
) -> Result<S, SciOxideError> {
    component_max(values.into_iter().flat_map(|value| [value.re, value.im]))
}

/// Validate and measure real samples, reporting non-finite values as input errors.
pub(crate) fn real_max<S: RealScalar>(
    values: impl IntoIterator<Item = S>,
) -> Result<S, SciOxideError> {
    component_max(values.into_iter())
}

/// Contiguous real samples use the shared vectorizable component scan.
pub(crate) fn real_max_slice<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    slice_max::<_, _, 8>(values, |chunk, maxima| {
        maxima.measure(chunk.iter().copied())
    })
}

/// Contiguous complex samples use the same scan as real samples, with two
/// components per value. No layout conversion or temporary allocation is needed.
pub(crate) fn complex_max_slice<S: RealScalar>(values: &[Complex<S>]) -> Result<S, SciOxideError> {
    slice_max::<_, _, 4>(values, |chunk, maxima| {
        let components = match chunk {
            [a, b, c, d] => [a.re, a.im, b.re, b.im, c.re, c.im, d.re, d.im],
            tail => {
                let mut components = [S::ZERO; 8];
                for (pair, value) in components.chunks_exact_mut(2).zip(tail) {
                    pair.copy_from_slice(&[value.re, value.im]);
                }
                components
            }
        };
        maxima.measure(components);
    })
}

/// Copy a dense packed candidate while measuring its original components.
/// The four-value reduction matches [`complex_max_slice`], including tails.
pub(crate) fn copy_complex_max<S: RealScalar>(
    source: &[Complex<S>],
    target: &mut [Complex<S>],
    conjugate: bool,
) -> Result<S, SciOxideError> {
    let mut maxima = Maxima::new();
    for (source, target) in source.chunks(4).zip(target.chunks_mut(4)) {
        let mut components = [S::ZERO; 8];
        for ((pair, &value), slot) in components.chunks_exact_mut(2).zip(source).zip(target) {
            pair.copy_from_slice(&[value.re, value.im]);
            *slot = if conjugate { value.conj() } else { value };
        }
        maxima.measure(components);
    }
    maxima.finish()
}

/// Both traversal forms feed one eight-component reduction. Independent
/// maxima and finiteness lanes keep the vectorizable contiguous arithmetic;
/// padding a final block with positive zeros cannot change its result.
struct Maxima<S> {
    values: [S; 8],
    finite: [bool; 8],
}

impl<S: RealScalar> Maxima<S> {
    fn new() -> Self {
        Self {
            values: [S::ZERO; 8],
            finite: [true; 8],
        }
    }

    #[inline(always)]
    fn measure(&mut self, values: impl IntoIterator<Item = S>) {
        for ((maximum, finite), value) in self.values.iter_mut().zip(&mut self.finite).zip(values) {
            let magnitude = value.abs();
            // Ordered comparison rejects NaN as well as either infinity.
            // Sharing the absolute value keeps this check vectorizable.
            *finite &= magnitude < S::INFINITY;
            *maximum = if magnitude > *maximum {
                magnitude
            } else {
                *maximum
            };
        }
    }

    fn finish(self) -> Result<S, SciOxideError> {
        if self.finite.into_iter().all(|finite| finite) {
            Ok(self.values.into_iter().fold(S::ZERO, |a, b| a.max(b)))
        } else {
            Err(crate::validation::nonfinite())
        }
    }
}

/// One slice traversal for both component shapes. Real chunks borrow their
/// components directly; complex chunks expose their components in registers.
/// Both adapters feed the same maximum and finiteness reduction.
fn slice_max<S: RealScalar, T, const N: usize>(
    values: &[T],
    measure: impl Fn(&[T], &mut Maxima<S>),
) -> Result<S, SciOxideError> {
    let mut maxima = Maxima::new();
    let chunks = values.chunks_exact(N);
    let tail = chunks.remainder();
    for chunk in chunks {
        measure(chunk, &mut maxima);
    }
    if !tail.is_empty() {
        measure(tail, &mut maxima);
    }
    maxima.finish()
}

/// Strided and conjugated views pack the same component blocks without
/// allocating. Stop at the first absent iterator item, including a short tail.
fn component_max<S: RealScalar>(values: impl Iterator<Item = S>) -> Result<S, SciOxideError> {
    let mut maxima = Maxima::new();
    let mut values = values;
    loop {
        let mut block = [S::ZERO; 8];
        let mut count = 0;
        for destination in &mut block {
            let Some(value) = values.next() else {
                break;
            };
            *destination = value;
            count += 1;
        }
        if count == 0 {
            break;
        }
        maxima.measure(block);
        if count < 8 {
            break;
        }
    }
    maxima.finish()
}

/// Restore a candidate after bounded arithmetic. The bound includes all
/// direction factors and rounding margin. External providers use an infinite
/// bound, ensuring their outputs are checked individually.
pub(crate) fn restore_complex<S: RealScalar>(
    values: &mut [Complex<S>],
    exponent: i32,
    bound: S,
) -> Result<(), SciOxideError> {
    let check = !bound.scalbn(exponent).is_finite();
    let restore = BinaryScale::new(exponent);
    for value in values {
        *value = restore.restore_complex(*value);
        if check && !value.is_finite() {
            return Err(arithmetic_overflow(
                "FFT output exceeds the finite scalar range",
            ));
        }
    }
    Ok(())
}

/// Conservative binary headroom for sums and complex products of `n` terms.
pub(crate) fn growth_exponent(n: usize) -> i32 {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as i32 + 2
    }
}

pub(crate) fn scale_complex<S: scioxide_core::RealScalar>(data: &mut [Complex<S>], scale: S) {
    if scale != S::ONE {
        data.iter_mut().for_each(|value| *value *= scale);
    }
}

/// A singleton has no butterfly: canonicalize exact zeros before its only
/// multiplication, keeping the sign when a nonzero product rounds to zero.
pub(crate) fn scale_singleton<S: RealScalar>(value: Complex<S>, scale: S) -> Complex<S> {
    let component = |x| {
        if x == S::ZERO || scale == S::ZERO {
            S::ZERO
        } else {
            x * scale
        }
    };
    Complex::new(component(value.re), component(value.im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use scioxide_core::RealScalar;

    fn same_maximum<S: RealScalar>(actual: Result<S, SciOxideError>, expected: S) {
        let actual = actual.unwrap();
        assert_eq!(actual.decompose(), expected.decompose());
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }

    #[test]
    fn real_and_complex_scans_agree_at_chunk_and_scalar_boundaries() {
        macro_rules! check {
            ($scalar:ty) => {{
                let cases: [$scalar; 10] = [
                    0.0,
                    -0.0,
                    <$scalar>::from_bits(1),
                    -<$scalar>::from_bits(3),
                    <$scalar>::MIN_POSITIVE,
                    -<$scalar>::MIN_POSITIVE,
                    0.5,
                    -0.75,
                    <$scalar>::MAX,
                    -<$scalar>::MAX,
                ];
                for len in 0..=19 {
                    for start in 0..cases.len() {
                        let real: Vec<_> =
                            (0..len).map(|i| cases[(start + i) % cases.len()]).collect();
                        let expected = real
                            .iter()
                            .fold(0.0 as $scalar, |maximum, x| maximum.max(x.abs()));
                        same_maximum(real_max_slice(&real), expected);
                        same_maximum(real_max(real.iter().copied()), expected);
                        for imaginary in [false, true] {
                            let complex: Vec<_> = real
                                .iter()
                                .map(|&x| {
                                    if imaginary {
                                        Complex::new(-0.0, x)
                                    } else {
                                        Complex::new(x, -0.0)
                                    }
                                })
                                .collect();
                            same_maximum(complex_max_slice(&complex), expected);
                            same_maximum(complex_max(complex.iter().copied()), expected);
                        }
                    }
                }
                for len in 1..=19 {
                    for index in 0..len {
                        for bad in [<$scalar>::NAN, <$scalar>::INFINITY, -<$scalar>::INFINITY] {
                            let mut real = vec![0.5 as $scalar; len];
                            real[index] = bad;
                            let mut results =
                                vec![real_max_slice(&real), real_max(real.iter().copied())];
                            for imaginary in [false, true] {
                                let complex: Vec<_> = real
                                    .iter()
                                    .map(|&x| {
                                        if imaginary {
                                            Complex::new(0.5, x)
                                        } else {
                                            Complex::new(x, 0.5)
                                        }
                                    })
                                    .collect();
                                results.extend([
                                    complex_max_slice(&complex),
                                    complex_max(complex.iter().copied()),
                                ]);
                            }
                            for result in results {
                                assert!(matches!(
                                    result.unwrap_err().view(),
                                    scioxide_core::SciOxideErrorView::NonFinite {
                                        origin: scioxide_core::NonFiniteOrigin::Input,
                                        ..
                                    }
                                ));
                            }
                        }
                    }
                }
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn prepared_binary_scaling_preserves_component_bits() {
        macro_rules! check {
            ($scalar:ty, $limit:expr) => {
                for exponent in -$limit..=$limit {
                    let scale = BinaryScale::<$scalar>::new(exponent);
                    for value in [
                        0.0,
                        -0.0,
                        0.5,
                        -0.75,
                        1.0,
                        <$scalar>::MIN_POSITIVE,
                        -<$scalar>::MIN_POSITIVE,
                        <$scalar>::from_bits(1),
                        -<$scalar>::from_bits(1),
                        <$scalar>::from_bits(3),
                        -<$scalar>::from_bits(3),
                        <$scalar>::MAX,
                        -<$scalar>::MAX,
                        (1.0 as $scalar).next_after(0.0),
                        (1.0 as $scalar).next_after(2.0),
                    ] {
                        assert_eq!(
                            scale.apply(value).to_bits(),
                            value.scalbn(exponent).to_bits(),
                            "{value:e} * 2^{exponent}"
                        );
                        let expected = if value == 0.0 {
                            0.0
                        } else {
                            value.scalbn(exponent)
                        };
                        assert_eq!(scale.restore(value).to_bits(), expected.to_bits());
                        let input = Complex::new(value, -value);
                        let mut scaled = [input];
                        scale.apply_all(in_place(&mut scaled));
                        assert_eq!(scaled[0].re.to_bits(), value.scalbn(exponent).to_bits());
                        assert_eq!(scaled[0].im.to_bits(), (-value).scalbn(exponent).to_bits());
                        let mut restored = [input];
                        scale.restore_in_place(&mut restored);
                        assert_eq!(restored[0].re.to_bits(), expected.to_bits());
                        let expected_im = if value == 0.0 {
                            0.0
                        } else {
                            (-value).scalbn(exponent)
                        };
                        assert_eq!(restored[0].im.to_bits(), expected_im.to_bits());
                    }
                }
            };
        }
        check!(f32, 320);
        check!(f64, 2200);
    }

    #[test]
    fn binary_restore_matches_checked_products() {
        for exponent in [-1074, -1022, -500, 0, 500, 1023, 1024] {
            for value in [
                0.0_f64,
                -0.0,
                0.5,
                -0.75,
                1.0,
                f64::MIN_POSITIVE,
                -f64::MIN_POSITIVE,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::MAX,
                1.0_f64.next_after(0.0),
            ] {
                let first = exponent.clamp(-1022, 1023);
                let factors = [
                    value,
                    1.0_f64.scalbn(first),
                    1.0_f64.scalbn(exponent - first),
                ];
                let restored = BinaryScale::new(exponent).restore(value);
                match scioxide_core::checked_float_product_factors(&factors) {
                    Ok(expected) => assert_eq!(restored.to_bits(), expected.to_bits()),
                    Err(_) => assert!(!restored.is_finite()),
                }
            }
        }
        for exponent in [-149_i32, -126, -70, 0, 70, 127, 128] {
            for value in [
                0.0_f32,
                -0.0,
                0.5,
                -0.75,
                1.0,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MAX,
            ] {
                let first = exponent.clamp(-126, 127);
                let factors = [
                    value,
                    1.0_f32.scalbn(first),
                    1.0_f32.scalbn(exponent - first),
                ];
                let restored = BinaryScale::new(exponent).restore(value);
                match scioxide_core::checked_float_product_factors(&factors) {
                    Ok(expected) => assert_eq!(restored.to_bits(), expected.to_bits()),
                    Err(_) => assert!(!restored.is_finite()),
                }
            }
        }
        for maximum in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1e-300,
            1.0,
            1e300,
            f64::MAX,
        ] {
            let bounded = maximum.scalbn(-range_exponent(maximum));
            assert!((0.5..1.0).contains(&bounded));
        }
    }
}
