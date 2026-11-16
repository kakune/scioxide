//! Ordinary kernels and their range-recovery paths obey the same certificates.

use scioxide_core::{Budget, Complex, Interval, ResidualTolerance, StepTolerance};
use scioxide_rootfind::{
    ComplexHalley, ComplexNewton, ComplexRootSolver, ComplexScalarFn, ComplexSecant, Halley,
    Newton, ScalarFn, ScalarRootInput, ScalarRootMethod, ScalarRootOptions, ScalarRootSolver,
    Secant,
};

macro_rules! range_tests {
    ($module:ident, $scalar:ty, $tiny:expr, $huge:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn secant_recovers_a_root_after_weight_underflow() {
                let small = $tiny;
                let large = $huge;
                let options = ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::new(0.0, 0.0).unwrap())
                    .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap());
                let real = ScalarFn::new(|x: $scalar| Ok(x + small));
                let result = Secant.solve(&real, (large, 0.0), &options).unwrap();
                assert!(result.is_success(), "{result:?}");
                assert!((result.point() / small + 1.0).abs() <= 4.0 * <$scalar>::EPSILON);
                let complex =
                    ComplexScalarFn::new(|z: Complex<$scalar>| Ok(z + Complex::new(small, small)));
                let result = ComplexSecant
                    .solve(
                        &complex,
                        (Complex::new(large, large), Complex::new(0.0, 0.0)),
                        &options,
                    )
                    .unwrap();
                assert!(result.is_success(), "{result:?}");
                assert!((result.point().re / small + 1.0).abs() <= 4.0 * <$scalar>::EPSILON);
                assert!((result.point().im / small + 1.0).abs() <= 4.0 * <$scalar>::EPSILON);
            }

            #[test]
            fn brent_contracts_extreme_brackets_when_the_value_ratio_underflows() {
                let function = ScalarFn::new(|x: $scalar| Ok((x - 3.0) - <$scalar>::MIN_POSITIVE));
                let options = ScalarRootOptions::default().with_max_iterations(Budget::ONE);
                for method in [ScalarRootMethod::Brent, ScalarRootMethod::Brenth] {
                    let result = method
                        .solve(
                            &function,
                            ScalarRootInput::Bracket(
                                Interval::new(3.0, <$scalar>::MAX / 4.0).unwrap(),
                            ),
                            &options,
                        )
                        .unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert_eq!(
                        result.termination(),
                        scioxide_rootfind::RootTermination::BracketTolerance
                    );
                    assert!(result.residual() < 0.0);
                    let bracket = result.bracket().unwrap();
                    assert!(bracket.upper() > 3.0);
                    assert!(bracket.upper() - bracket.lower() <= 32.0 * <$scalar>::EPSILON);
                    assert_eq!(
                        bracket.verify(&function).unwrap(),
                        scioxide_rootfind::BracketVerification::Verified
                    );
                }
            }

            #[test]
            fn brent_preserves_tiny_interpolants_with_a_loose_width_tolerance() {
                let target = <$scalar>::EPSILON;
                let function = ScalarFn::new(|x: $scalar| Ok(x - target));
                let options = ScalarRootOptions::default()
                    .with_step_tolerance(StepTolerance::new(1e-3, 0.0).unwrap());
                for method in [ScalarRootMethod::Brent, ScalarRootMethod::Brenth] {
                    let result = method
                        .solve(
                            &function,
                            ScalarRootInput::Bracket(Interval::new(0.0, 1.0).unwrap()),
                            &options,
                        )
                        .unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!(
                        (result.point() - target).abs() <= 8.0 * <$scalar>::EPSILON * target,
                        "requested width displaced an accurate interpolant: {result:?}"
                    );
                }
            }

            #[test]
            fn coordinate_and_function_scales_preserve_bracket_certificates() {
                let tolerance = 32.0 * <$scalar>::EPSILON;
                let options = ScalarRootOptions::default()
                    .with_step_tolerance(StepTolerance::new(0.0, tolerance).unwrap())
                    .with_max_iterations(Budget::ONE.saturating_add(511));
                for method in [
                    ScalarRootMethod::Bisection,
                    ScalarRootMethod::Brent,
                    ScalarRootMethod::Brenth,
                    ScalarRootMethod::Chandrupatla,
                    ScalarRootMethod::Toms748,
                    ScalarRootMethod::RegulaFalsi,
                    ScalarRootMethod::Illinois,
                    ScalarRootMethod::Ridder,
                ] {
                    for coordinate_scale in [$tiny, 1.0, $huge] {
                        for function_scale in [$tiny, 1.0, $huge] {
                            let function = ScalarFn::new(|x: $scalar| {
                                Ok((x / coordinate_scale - 0.25) * function_scale)
                            });
                            let result = method
                                .solve(
                                    &function,
                                    ScalarRootInput::Bracket(
                                        Interval::new(-coordinate_scale, coordinate_scale).unwrap(),
                                    ),
                                    &options,
                                )
                                .unwrap();
                            assert!(result.is_success(), "{method:?}: {result:?}");
                            assert!((result.point() / coordinate_scale - 0.25).abs() <= tolerance);
                            let bracket = result.bracket().unwrap();
                            assert_eq!(
                                bracket.verify(&function).unwrap(),
                                scioxide_rootfind::BracketVerification::Verified,
                            );
                        }
                    }
                }
            }

            #[test]
            fn unrepresentable_open_steps_recover_representable_roots() {
                let maximum = <$scalar>::MAX;
                let tolerance = 32.0 * <$scalar>::EPSILON;
                let options = ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::new(0.0, tolerance).unwrap());
                let real = ScalarFn::new(|x: $scalar| Ok((x / maximum + 1.0) * (maximum / 2.0)))
                    .with_derivative(|_| Ok(0.5))
                    .with_second_derivative(|_| Ok(0.0));
                for result in [
                    Newton::new(scioxide_differentiate::DerivativeStrategy::Analytic)
                        .solve(&real, maximum, &options),
                    Halley.solve(&real, maximum, &options),
                    Secant.solve(&real, (maximum / 2.0, maximum), &options),
                ] {
                    let result = result.unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!((result.point() / maximum + 1.0).abs() <= tolerance);
                }
                let complex = ComplexScalarFn::new(|z: Complex<$scalar>| {
                    Ok((z / maximum + Complex::new(1.0, 0.0)) * (maximum / 2.0))
                })
                .with_derivative(|_| Ok(Complex::new(0.5, 0.0)))
                .with_second_derivative(|_| Ok(Complex::new(0.0, 0.0)));
                let initial = Complex::new(maximum, 0.0);
                for result in [
                    ComplexNewton.solve(&complex, initial, &options),
                    ComplexHalley.solve(&complex, initial, &options),
                    ComplexSecant.solve(
                        &complex,
                        (Complex::new(maximum / 2.0, 0.0), initial),
                        &options,
                    ),
                ] {
                    let result = result.unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!((result.point().re / maximum + 1.0).abs() <= tolerance);
                    assert!(result.point().im.abs() <= tolerance);
                }
            }
        }
    };
}

range_tests!(single, f32, 1e-30_f32, 1e30_f32);
range_tests!(double, f64, 1e-300_f64, 1e300_f64);

#[test]
fn overflowing_step_thresholds_compare_in_extended_range() {
    use scioxide_rootfind::{Bisection, RootTermination};
    // `2·max(|a|, |b|)` overflows at these coordinates, but the steps and the
    // bracket width it bounds are representable comparisons.
    let options =
        ScalarRootOptions::default().with_step_tolerance(StepTolerance::new(0.0, 2.0).unwrap());
    let identity = ScalarFn::new(|x: f64| Ok(x));
    let result = Bisection
        .solve(&identity, Interval::new(-1e308, 1e308).unwrap(), &options)
        .unwrap();
    assert_eq!(result.termination(), RootTermination::BracketTolerance);
    let real = ScalarFn::new(|x: f64| Ok(x - 1e308));
    let complex = ComplexScalarFn::new(|z: Complex<f64>| Ok(z - Complex::new(1e308, 0.0)));
    let real = Secant.solve(&real, (0.0, 1e307), &options).unwrap();
    let complex = ComplexSecant
        .solve(
            &complex,
            (Complex::new(0.0, 0.0), Complex::new(1e307, 0.0)),
            &options,
        )
        .unwrap();
    assert!(real.is_success(), "{real:?}");
    assert!(complex.is_success(), "{complex:?}");
    assert_eq!(real.point(), complex.point().re);
}
