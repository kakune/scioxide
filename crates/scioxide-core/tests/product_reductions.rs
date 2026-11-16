//! Iterator reductions agree with independent exact scalar accumulation.
#![allow(clippy::unwrap_used)]
use scioxide_core::backend::checked_product_sum;
use scioxide_core::{Complex, NumericScalar, ProductSum, RealScalar, SciOxideError};

fn verify<T: NumericScalar>(pairs: &[(T, T)]) {
    let expected = pairs
        .iter()
        .try_fold(ProductSum::new(), |mut sum, &(a, b)| {
            sum.add_product(a, b)?;
            Ok::<_, SciOxideError>(sum)
        })
        .and_then(|sum| sum.finish());
    match (checked_product_sum(pairs.iter().copied()), expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(
                actual.real_component().into_f64().to_bits(),
                expected.real_component().into_f64().to_bits()
            );
            assert_eq!(
                actual.imaginary_component().into_f64().to_bits(),
                expected.imaginary_component().into_f64().to_bits()
            );
        }
        (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
        (actual, expected) => assert_eq!(actual, expected),
    }
}

fn real_cases<S: RealScalar>() {
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    for length in [0, 1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129, 1024, 1025] {
        for exponent in [-200, -24, 0, 24, 200] {
            let pairs: Vec<_> = (0..length)
                .map(|i| {
                    let a = convert(((i * 7919 % 65537) as f64 - 32768.0) / 32749.0);
                    let b = convert(((i * 104729 % 65537) as f64 - 32768.0) / 32749.0);
                    (a.scalbn(exponent), b.scalbn(-exponent))
                })
                .collect();
            // Skip overflowing source scales in f32; overflow itself is tested below.
            if pairs.iter().all(|&(a, b)| a.is_finite() && b.is_finite()) {
                verify(&pairs);
            }
        }
        let tiny = S::ZERO.next_after(S::ONE);
        for pair in [
            (S::MAX, S::TWO),
            (tiny, S::HALF),
            (-tiny, S::HALF),
            (S::ZERO, -S::ZERO),
        ] {
            verify(&vec![pair; length]);
        }
        let mut cancellation = vec![(S::MAX, S::TWO); length];
        for (i, pair) in cancellation.iter_mut().enumerate() {
            if i % 2 == 1 {
                pair.0 = -pair.0;
            }
        }
        cancellation.push((S::ONE, S::ONE));
        verify(&cancellation);
        let mut ordinary = vec![(convert(1.25), convert(1.75)); length];
        for weight in [S::ZERO, S::ONE, -S::ONE, convert(1.5), S::MAX, tiny] {
            let weighted: Vec<_> = (0..length)
                .map(|i| (weight, convert((i as f64 - 17.0) / 31.0)))
                .collect();
            verify(&weighted);
        }
        for invalid in [S::NAN, S::INFINITY, -S::INFINITY] {
            ordinary.push((S::ZERO, invalid));
            verify(&ordinary);
            ordinary.pop();
        }
    }
    // Product rounding and final cancellation must not discard these low bits.
    let next = S::ONE.next_after(S::TWO);
    let mut pairs = vec![(next, next); 64];
    pairs.extend([(S::ONE, -S::ONE); 64]);
    verify(&pairs);
    let tiny = S::ZERO.next_after(S::ONE);
    for sign in [S::ONE, -S::ONE] {
        let mut pairs = vec![(sign * tiny, tiny); 128];
        pairs.push((sign * tiny, S::HALF));
        // Exact products below the destination grid break the final midpoint.
        assert_eq!(
            checked_product_sum(pairs.iter().copied()).unwrap(),
            sign * tiny
        );
        verify(&pairs);
    }
    // Exercise signed carries and shifts on both sides of the 128/192-bit
    // block boundaries, including exponent spans that require scalar fallback.
    for exponent in [
        0, 1, 63, 64, 125, 126, 127, 128, 129, 184, 185, 186, 190, 191, 192, 256, 500,
    ] {
        let large = S::ONE.scalbn(exponent);
        if !large.is_finite() {
            continue;
        }
        for sign in [S::ONE, -S::ONE] {
            let mut pairs = vec![(sign * large, S::ONE); 129];
            for (i, pair) in pairs.iter_mut().enumerate() {
                if i % 3 == 0 {
                    pair.0 = -pair.0;
                }
                if i % 5 == 0 {
                    pair.0 = sign;
                }
            }
            verify(&pairs);
        }
    }
}

#[test]
fn both_real_precisions_preserve_exact_rounding_and_errors() {
    real_cases::<f32>();
    real_cases::<f64>();
}

#[test]
fn complex_components_preserve_exact_products_and_cancellation() {
    for length in [0, 1, 4, 5, 64, 65, 129] {
        let pairs: Vec<_> = (0..length)
            .map(|i| {
                (
                    Complex::new(i as f64 / 17.0 - 1.0, i as f64 / 31.0),
                    Complex::new(-0.5, 1.5),
                )
            })
            .collect();
        verify(&pairs);
        verify(
            &pairs
                .iter()
                .map(|&(a, b)| {
                    (
                        Complex::new(a.re as f32, a.im as f32),
                        Complex::new(b.re as f32, b.im as f32),
                    )
                })
                .collect::<Vec<_>>(),
        );
    }
    verify(&[
        (Complex::new(f64::MAX, f64::MAX), Complex::new(1.0, 1.0)),
        (Complex::new(-f64::MAX, -f64::MAX), Complex::new(1.0, 1.0)),
    ]);
    verify(&[(Complex::new(0.0, f64::NAN), Complex::new(0.0, 0.0))]);
}

#[test]
fn iterator_stops_at_first_absence() {
    for length in [0, 1, 2, 3, 4, 5, 63, 64, 65, 128, 129] {
        let mut calls = 0;
        let values = core::iter::from_fn(|| {
            calls += 1;
            if calls <= length {
                Some((1.0_f64, 2.0))
            } else if calls == length + 1 {
                None
            } else {
                Some((f64::NAN, 0.0))
            }
        });
        assert_eq!(checked_product_sum(values).unwrap(), 2.0 * length as f64);
        assert_eq!(calls, length + 1);
    }
}
