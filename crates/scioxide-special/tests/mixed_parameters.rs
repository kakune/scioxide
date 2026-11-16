//! Real parameters beside complex arguments equal the calls with promoted complex parameters.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_special::*;

/// The complex number with imaginary part `+0` that a real parameter stands for.
fn promote<S: RealScalar>(value: S) -> Complex<S> {
    Complex::new(value, S::ZERO)
}

fn bits<V: NumericScalar>(value: V) -> (u64, u64) {
    (
        value.real_component().into_f64().to_bits(),
        value.imaginary_component().into_f64().to_bits(),
    )
}

/// Asserts that two plain outcomes have the same bits or the same error.
fn same_value<V: NumericScalar>(
    name: &str,
    mixed: Result<V, SciOxideError>,
    promoted: Result<V, SciOxideError>,
) {
    match (mixed, promoted) {
        (Ok(a), Ok(b)) => assert_eq!(bits(a), bits(b), "{name}: {a:?} vs {b:?}"),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{name}"),
        (a, b) => panic!("{name}: {a:?} vs {b:?}"),
    }
}

/// Asserts that two detailed outcomes are equal and their values have the same bits.
fn same_result<V: NumericScalar>(
    name: &str,
    mixed: Result<SpecialFunctionResult<V>, SciOxideError>,
    promoted: Result<SpecialFunctionResult<V>, SciOxideError>,
) {
    match (mixed, promoted) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "{name}");
            same_value(name, a.try_value(), b.try_value());
        }
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{name}"),
        (a, b) => panic!("{name}: {a:?} vs {b:?}"),
    }
}

/// Calls `$plain` and `$detailed` once with the real parameters and once with their promotions.
/// Each argument is `param <real value>` or `arg <complex value>`.
macro_rules! mixed {
    ($count:ident, $plain:ident, $detailed:ident, $($kind:ident $value:expr),+) => {{
        same_value(
            stringify!($plain),
            $plain($(mixed!(@real $kind $value)),+),
            $plain($(mixed!(@promoted $kind $value)),+),
        );
        same_result(
            stringify!($detailed),
            $detailed($(mixed!(@real $kind $value),)+ Default::default()),
            $detailed($(mixed!(@promoted $kind $value),)+ Default::default()),
        );
        $count += 1;
    }};
    (@real param $value:expr) => { $value };
    (@promoted param $value:expr) => { promote($value) };
    (@real arg $value:expr) => { $value };
    (@promoted arg $value:expr) => { $value };
    (@real degree $value:expr) => { $value };
    (@promoted degree $value:expr) => { $value };
}

#[allow(clippy::too_many_lines)]
fn every_function<S: RealScalar>() -> usize
where
    Complex<S>: NumericScalar<Real = S>,
{
    let r = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let c = |re: f64, im: f64| Complex::new(r(re), r(im));
    let z = c(2.0, 1.0);
    let mut n = 0;

    mixed!(n, bessel_j, bessel_j_with_options, param r(2.5), arg z);
    mixed!(n, bessel_y, bessel_y_with_options, param r(2.5), arg z);
    mixed!(n, bessel_i, bessel_i_with_options, param r(2.5), arg z);
    mixed!(n, bessel_k, bessel_k_with_options, param r(2.5), arg z);
    mixed!(n, bessel_i_scaled, bessel_i_scaled_with_options, param r(2.5), arg z);
    mixed!(n, bessel_k_scaled, bessel_k_scaled_with_options, param r(2.5), arg z);
    mixed!(n, hankel_1, hankel_1_with_options, param r(2.5), arg z);
    mixed!(n, hankel_2, hankel_2_with_options, param r(2.5), arg z);
    mixed!(n, bessel_j_scaled, bessel_j_scaled_with_options, param r(2.5), arg z);
    mixed!(n, bessel_y_scaled, bessel_y_scaled_with_options, param r(2.5), arg z);
    mixed!(n, hankel_1_scaled, hankel_1_scaled_with_options, param r(2.5), arg z);
    mixed!(n, hankel_2_scaled, hankel_2_scaled_with_options, param r(2.5), arg z);
    mixed!(n, bessel_j_derivative, bessel_j_derivative_with_options, param r(2.5), arg z, degree 2);
    mixed!(n, bessel_y_derivative, bessel_y_derivative_with_options, param r(2.5), arg z, degree 2);
    mixed!(n, bessel_i_derivative, bessel_i_derivative_with_options, param r(2.5), arg z, degree 2);
    mixed!(n, bessel_k_derivative, bessel_k_derivative_with_options, param r(2.5), arg z, degree 2);
    mixed!(
        n,
        bessel_i_scaled_derivative,
        bessel_i_scaled_derivative_with_options,
        param r(2.5),
        arg z,
        degree 1
    );
    mixed!(
        n,
        bessel_k_scaled_derivative,
        bessel_k_scaled_derivative_with_options,
        param r(2.5),
        arg z,
        degree 1
    );
    mixed!(n, hankel_1_derivative, hankel_1_derivative_with_options, param r(2.5), arg z, degree 1);
    mixed!(n, hankel_2_derivative, hankel_2_derivative_with_options, param r(2.5), arg z, degree 1);

    let x = c(1.5, 0.5);
    mixed!(n, gamma_p, gamma_p_with_options, param r(2.5), arg x);
    mixed!(n, gamma_q, gamma_q_with_options, param r(2.5), arg x);
    mixed!(n, gamma_lower, gamma_lower_with_options, param r(2.5), arg x);
    mixed!(n, gamma_upper, gamma_upper_with_options, param r(2.5), arg x);
    mixed!(n, beta_inc, beta_inc_with_options, param r(2.0), param r(3.0), arg c(0.3, 0.1));

    let w = c(0.5, 0.25);
    mixed!(n, hyp0f1, hyp0f1_with_options, param r(1.5), arg w);
    mixed!(n, hyp1f1, hyp1f1_with_options, param r(0.5), param r(1.25), arg w);
    mixed!(n, hyp2f0, hyp2f0_with_options, param r(0.5), param r(0.25), arg c(-0.5, 0.1));
    mixed!(n, hyperu, hyperu_with_options, param r(0.5), param r(1.25), arg c(-2.0, -0.0));
    mixed!(n, hyp2f1, hyp2f1_with_options, param r(1.0), param r(1.0), param r(2.0), arg w);
    mixed!(
        n,
        hyp2f1_regularized,
        hyp2f1_regularized_with_options,
        param r(1.0),
        param r(1.0),
        param r(0.0),
        arg w
    );

    mixed!(n, hurwitz_zeta, hurwitz_zeta_with_options, arg c(2.0, 1.0), param r(0.5));
    mixed!(n, polylog, polylog_with_options, param r(2.0), arg w);
    mixed!(n, lerch_phi, lerch_phi_with_options, arg w, param r(2.0), param r(0.5));
    mixed!(n, elliptic_f, elliptic_f_with_options, arg w, param r(0.5));
    mixed!(n, elliptic_e_inc, elliptic_e_inc_with_options, arg w, param r(0.5));

    for m in [0.0, 0.5, 1.0, 2.5, -1.5] {
        mixed!(n, jacobi_sn, jacobi_sn_with_options, arg w, param r(m));
        mixed!(n, jacobi_cn, jacobi_cn_with_options, arg w, param r(m));
        mixed!(n, jacobi_dn, jacobi_dn_with_options, arg w, param r(m));
        mixed!(n, jacobi_amplitude, jacobi_amplitude_with_options, arg w, param r(m));
    }

    // Errors pass through unchanged: a pole of `Γ(a)` and a non-finite parameter.
    mixed!(n, gamma_lower, gamma_lower_with_options, param r(-2.0), arg x);
    mixed!(n, bessel_j, bessel_j_with_options, param S::NAN, arg z);
    n
}

#[test]
fn real_parameters_equal_promoted_complex_parameters() {
    // 36 functions plus four Jacobi functions at five parameters, and two error cases.
    assert_eq!(every_function::<f64>(), 36 + 4 * 5 + 2);
    assert_eq!(every_function::<f32>(), 36 + 4 * 5 + 2);
}

#[test]
fn literals_infer_the_scalar_type_of_the_argument() -> Result<(), SciOxideError> {
    let j: f32 = bessel_j(0.0, 1.5_f32)?;
    assert!((f64::from(j) - bessel_j(0.0, 1.5_f64)?).abs() < 1e-6);
    let j: f32 = bessel_j(0.0_f32, 1.5)?;
    assert!((f64::from(j) - bessel_j(0.0, 1.5_f64)?).abs() < 1e-6);
    let p: f32 = gamma_p(2.0_f32, 3.0)?;
    assert!((f64::from(p) - gamma_p(2.0, 3.0_f64)?).abs() < 1e-6);
    let sn: f32 = jacobi_sn(0.5_f32, 0.5)?;
    assert!((f64::from(sn) - jacobi_sn(0.5_f64, 0.5)?).abs() < 1e-6);
    let f: Complex<f32> = hyp2f1(1.0, 1.0, 2.0, Complex::new(0.5_f32, 0.25))?;
    let wide = hyp2f1(1.0, 1.0, 2.0, Complex::new(0.5_f64, 0.25))?;
    assert!((f64::from(f.re) - wide.re).abs() < 1e-6 && (f64::from(f.im) - wide.im).abs() < 1e-6);
    let j: Complex<f64> = bessel_j(2.5, Complex::new(2.0, 1.0))?;
    assert_eq!(j, bessel_j(Complex::new(2.5, 0.0), Complex::new(2.0, 1.0))?);
    Ok(())
}

#[test]
fn functions_as_values_infer_both_types() {
    fn map2(f: impl Fn(f64, f64) -> Result<f64, SciOxideError>) -> f64 {
        f(0.0, 1.0).unwrap()
    }
    assert!((map2(bessel_j) - 0.7651976865579666).abs() < 1e-15);
    assert!((map2(gamma_p) - 1.0).abs() < 1e-15);
}
