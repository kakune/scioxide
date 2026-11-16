//! Allocation contract for scalar arithmetic kernels.

#![allow(clippy::unwrap_used)]

use core::ops::Neg;

use scioxide_core::{
    checked_affine, checked_difference_of_products, checked_float_product,
    checked_float_product_factors, checked_float_quotient, checked_midpoint, checked_mul_add,
    checked_next_after, checked_product_quotient, checked_quadratic_roots, checked_scalbn,
    checked_scaled_norm, checked_sum, checked_sum_of_products,
};

#[test]
fn certified_long_sums_and_their_exact_fallbacks_execute() {
    {
        let tail = f64::EPSILON * 0.375;
        assert_eq!(checked_sum(&[1.0, tail, 0.0, 0.0]).unwrap(), 1.0);
        assert_eq!(
            checked_sum(&[1.0, f64::EPSILON * 0.5, 0.0, 0.0]).unwrap(),
            1.0
        );
        assert_eq!(
            checked_sum(&[
                f64::MIN_POSITIVE,
                -f64::MIN_POSITIVE,
                f64::from_bits(1),
                0.0
            ])
            .unwrap(),
            f64::from_bits(1)
        );
        assert_eq!(checked_sum(&[1.0_f32; 1025]).unwrap(), 1025.0);
        assert!(checked_sum(&[f32::MAX; 5]).is_err());
        assert!(checked_sum(&[1.0_f32, 2.0, f32::NAN, 3.0, 4.0]).is_err());
    };
}

#[test]
fn exact_product_blocks_execute() {
    use scioxide_core::{Complex, backend::checked_product_sum};
    {
        assert_eq!(
            checked_product_sum([(1.25_f64, 1.75); 1025]).unwrap(),
            2242.1875
        );
        assert_eq!(
            checked_product_sum([(f64::MAX, 2.0), (-f64::MAX, 2.0)]).unwrap(),
            0.0
        );
        assert!(checked_product_sum([(0.0_f64, f64::NAN); 65]).is_err());
        assert!(checked_product_sum([(f32::MAX, f32::MAX); 65]).is_err());
        assert_eq!(
            checked_product_sum([(Complex::new(1.0_f32, 1.0), Complex::new(1.0, -1.0)); 65])
                .unwrap(),
            Complex::new(130.0, 0.0)
        );
    };
}

#[test]
fn compact_product_prefixes_and_promotion_execute() {
    use scioxide_core::backend::RealProductSum;
    {
        let mut sum = RealProductSum::<f64>::new();
        for _ in 0..65 {
            sum.add_weighted_pair(0.5, 1.0, 2.0).unwrap();
            assert!(sum.finish().is_ok());
        }
        sum.add_product(f64::from_bits(1), f64::from_bits(1))
            .unwrap();
        assert_eq!(sum.finish().unwrap(), 97.5);
        assert!(sum.add_weighted_pair(0.0, f64::NAN, 0.0).is_err());
        sum.clear();
        sum.add_product(-f64::from_bits(1), 0.5).unwrap();
        assert_eq!(sum.finish().unwrap().to_bits(), (-0.0_f64).to_bits());
    };
}

#[test]
fn compensated_blocks_execute_on_success_or_failure() {
    let data = [1.0_f64; 1025];
    {
        let [sum, squares] = scioxide_core::compensated_reduce(data.map(|x| [x, x * x])).unwrap();
        assert_eq!(sum.total().unwrap(), 1025.0);
        assert_eq!(squares.total().unwrap(), 1025.0);
        assert!(scioxide_core::compensated_reduce([[f64::NAN]; 259]).is_err());
        assert!(scioxide_core::compensated_reduce([[f64::MAX]; 259]).is_err());
    };
}

#[test]
fn scalar_kernel_family_performs_no_heap_allocation() {
    {
        let values = std::hint::black_box([f64::MAX, -f64::MAX, 1.0]);
        let left = std::hint::black_box([f64::MAX, f64::MIN_POSITIVE]);
        let right = std::hint::black_box([f64::MIN_POSITIVE, f64::MAX]);
        std::hint::black_box(checked_sum(&values)).unwrap();
        std::hint::black_box(checked_sum_of_products(&left, &right)).unwrap();
        std::hint::black_box(checked_difference_of_products(f64::MAX, 2.0, f64::MAX, 2.0)).unwrap();
        std::hint::black_box(checked_float_product(3.0_f64, 4.0)).unwrap();
        std::hint::black_box(checked_float_product_factors(&[
            f64::from_bits(1),
            f64::MAX,
        ]))
        .unwrap();
        std::hint::black_box(checked_float_quotient(-f64::from_bits(1), 2.0)).unwrap();
        std::hint::black_box(checked_mul_add(-f64::from_bits(1), 0.25, 0.0)).unwrap();
        std::hint::black_box(checked_scaled_norm(&[f64::MIN_POSITIVE, f64::from_bits(1)])).unwrap();
        assert!(checked_float_product_factors(&[0.0, f64::NAN]).is_err());
        assert!(checked_float_quotient(1.0_f64, 0.0).is_err());
        assert!(checked_product_quotient(1.0_f64, 2.0, 0.0).is_err());
        assert!(checked_affine(1.0_f64, 2.0, 1.1).is_err());
        assert!(checked_scaled_norm(&[f64::MAX, f64::MAX]).is_err());
        std::hint::black_box(checked_product_quotient(f64::MAX, 2.0, 2.0)).unwrap();
        std::hint::black_box(checked_midpoint(-f64::MAX, f64::MAX)).unwrap();
        std::hint::black_box(checked_next_after(0.0_f64, 1.0)).unwrap();
        std::hint::black_box(checked_scalbn(f64::MIN_POSITIVE, -52)).unwrap();
        std::hint::black_box(checked_affine(-f64::MAX, f64::MAX, 0.5)).unwrap();
        let _ = std::hint::black_box(checked_quadratic_roots(1.0_f64, -1.0e100, 1.0)).unwrap();
    };
}

#[test]
fn exact_fallback_and_complex_updates_execute() {
    use scioxide_core::{Complex, ProductSum, RangeAccumulator};
    {
        let tiny = std::hint::black_box(f64::from_bits(1));
        assert_eq!(
            checked_sum_of_products(&[0.75; 3], &[tiny; 3]).unwrap(),
            2.0 * tiny
        );
        let mut real = RangeAccumulator::new();
        real.add_triple_products(2.0, [(tiny, 0.5), (tiny, 0.5)])
            .unwrap();
        assert_eq!(real.finish().unwrap(), 2.0 * tiny);
        let mut complex = ProductSum::new();
        let left = Complex::new(1.5, -2.0);
        let right = Complex::new(-4.0, 0.5);
        complex.add_product(left, right).unwrap();
        let mut other = ProductSum::new();
        other
            .subtract_triple_product(Complex::new(1.0, -0.0), left, right)
            .unwrap();
        complex.merge(&other).unwrap();
        assert_eq!(complex.finish().unwrap(), Complex::new(0.0, 0.0));
        assert!(
            complex
                .add_product(Complex::new(0.0, f64::NAN), right)
                .is_err()
        );
        assert_eq!(complex.finish().unwrap(), Complex::new(0.0, 0.0));
    };
}

#[test]
fn elementary_families_execute() {
    use scioxide_core::{Complex, RealScalar};
    {
        let x = std::hint::black_box(1e20_f64);
        std::hint::black_box((
            RealScalar::sin_cos(x),
            RealScalar::tan(x),
            RealScalar::cbrt(x),
            RealScalar::exp10(3.0_f64),
        ));
        let z = std::hint::black_box(Complex::new(f64::MAX, f64::from_bits(1)));
        std::hint::black_box((z.sqrt(), z.ln(), z.asin(), z.acos(), z.atanh(), z.powi(3)));
    };
}

#[test]
fn scaled_values_execute_on_success_or_failure() {
    use scioxide_core::{Complex64, LogValue, ScaledValue};
    use scioxide_core::{Scaled, ScaledAccumulator};
    let phase = Complex64::new(
        core::f64::consts::FRAC_1_SQRT_2,
        core::f64::consts::FRAC_1_SQRT_2,
    );
    let logarithmic = LogValue::from_log(phase, f64::MAX.ln() + 0.2).unwrap();
    let large = Scaled::<f64>::ONE.shifted(i64::MAX).unwrap();
    let small = Scaled::<f64>::ONE.shifted(i64::MIN).unwrap();
    let mixed = Complex64::new(f64::MAX, f64::from_bits(1));
    (|| {
        assert_eq!(
            ScaledValue::from_value(mixed).unwrap().try_value().unwrap(),
            mixed
        );
        assert!(Scaled::from_value(f64::NAN).is_err());
        assert!(ScaledValue::from_log(1.0, 1e30).is_err());
        assert!(
            ScaledValue::from_value(mixed)
                .unwrap()
                .try_projective()
                .is_err()
        );
        assert!(logarithmic.try_value().unwrap().is_finite());
        assert_eq!(large.add(small).unwrap(), large);
        let mut sum = ScaledAccumulator::from_scaled(large);
        let before = sum.parts();
        assert!(sum.add_scaled(large).is_err());
        assert_eq!(sum.parts(), before);
        assert!(large.try_value().is_err());
        assert!(small.try_value().is_err());
        assert!(large.round_to::<f32>().is_err());
        assert!(small.try_value_to::<f32>().is_err());
        assert!(Scaled::<f32>::ZERO.ln_abs().is_err());
        let frame =
            ScaledValue::from_value(scioxide_core::Complex32::new(-0.0, f32::from_bits(1)))?
                .try_projective()?;
        assert_eq!(
            frame.into_scaled()?.try_value()?.re.to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(small.neg().round().unwrap().to_bits(), (-0.0_f64).to_bits());
        Ok::<(), scioxide_core::SciOxideError>(())
    })()
    .unwrap();
}
