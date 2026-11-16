//! Real Airy values from anchored ODE/asymptotic kernels under a work budget.
//! The negative-axis phase reducer covers every finite IEEE exponent.
use crate::bessel_airy::airy_kernel::{self as kernel, Kind};
use crate::complex_support::{RoundedZero, ZeroOrScaled};
use crate::result::{Bound, Either, Evaluation, Form};
use crate::support::Work;
use crate::{AiryOptions, ScalarRepresentation, SpecialFunctionTermination};
use scioxide_core::{
    AlgorithmComponent, NumericScalar, RealScalar, SciOxideError, SolveContext, algorithm_component,
};

/// Evaluation of a real or complex Airy function.
pub(crate) type AiryEvaluation<T> = Evaluation<T, ZeroOrScaled>;
type RealAiryEvaluation<T> = Evaluation<T, Either<Form<T>, ZeroOrScaled>>;

/// Positive-axis Ai whose exponent is beyond the scalar range: its rounded zero.
pub(crate) fn underflow<T: NumericScalar>(used: usize) -> AiryEvaluation<T> {
    // The positive value is below half the smallest subnormal quantum.
    Evaluation::from_parts(
        Some(T::ZERO),
        ScalarRepresentation::Finite,
        SpecialFunctionTermination::Converged,
        scioxide_core::EvaluationCounts::from_iterations(used),
        scioxide_core::algorithm_id!("airy", "positive_underflow_bound", 1),
        Some(Either::Right(RoundedZero)),
    )
}

fn evaluate<S: RealScalar>(
    x: S,
    kind: Kind,
    scaled: bool,
    options: AiryOptions,
    family: AlgorithmComponent,
    parent: Option<&mut SolveContext<'_, '_>>,
) -> Result<RealAiryEvaluation<S>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "Airy input must be finite",
        ));
    }
    let wide = x.try_to_f64_exact()?;
    let mut work = Work::with_batched_context(options.iteration_budget().get(), parent)?;
    if wide > 0.0 && !scaled {
        let exponent = ((2.0 / 3.0) * wide) * wide.sqrt();
        if !exponent.is_finite() || exponent > S::MAX.try_to_f64_exact()? {
            if work.poll_step()? {
                work.check()?;
                return if matches!(kind, Kind::Ai) {
                    Ok(underflow(work.used).map_form(Either::Right))
                } else {
                    Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Overflow,
                        "Airy value and logarithm exceed the scalar range",
                    ))
                };
            }
            return Ok(crate::complex_support::report_with_status(
                Ok(crate::support::interrupted()),
                false,
                family,
                work.used,
                true,
            )?
            .map_form(|form| Either::Right(Either::Left(form))));
        }
    }
    let result = kernel::real_pair(wide, kind, scaled && wide > 0.0, &mut work);
    work.check()?;
    let ordinary_pair = result.as_ref().is_ok_and(|e| {
        e.scale.v.re == 0.0
            && e.scale.v.im == 0.0
            && e.y.v.re.abs() >= S::MIN_POSITIVE.into_f64()
            && e.y.v.re.abs() <= S::MAX.into_f64()
            && e.y.d.re.is_finite()
            && e.y.d.re.abs() <= S::MAX.into_f64()
    });
    if wide <= 0.0 || ordinary_pair {
        if !work.complete {
            return Err(crate::result::exhausted_work(work.used));
        }
        let e = result?;
        // Ordinary real Airy kernels have zero exponential scale. Publish their
        // real pair directly, retaining the same kernel uncertainty and final
        // rounding allowances without complex logarithmic/phase conversion.
        let value = e.y.v.re;
        let derivative = e.y.d.re;
        let error = e.error + 16.0 * f64::EPSILON * value.abs();
        let derivative_error =
            e.derivative_error.unwrap_or(e.error) + 24.0 * f64::EPSILON * derivative.abs();
        let projected = S::try_from_f64_lossy(value)?;
        let slope = S::try_from_f64_lossy(derivative)?;
        return Ok(Evaluation::finite(
            projected,
            Bound::derived(crate::result::outward::<S>(
                error + (projected.into_f64() - value).abs(),
            )?),
            Some((
                slope,
                Bound::derived(crate::result::outward::<S>(
                    derivative_error + (slope.into_f64() - derivative).abs(),
                )?),
            )),
            scioxide_core::AlgorithmId::from_components(family, e.method, 1),
            work.used,
            SpecialFunctionTermination::Converged,
        )
        .map_form(Either::Left));
    }
    let reported = crate::complex_support::report_with_status(
        result,
        true,
        family,
        work.used,
        !work.complete,
    )?;
    // The rounded logarithm cannot resolve the final binary64 overflow edge.
    // Its first overflowing IEEE input is independently generated at 160/240 digits.
    if !scaled
        && matches!(kind, Kind::Bi)
        && wide >= f64::from_bits(crate::bessel_airy::airy_anchors::BI_OVERFLOW_START)
        && reported.termination().is_success()
    {
        return Ok(reported
            .map_form(|form| Either::Right(Either::Left(form)))
            .with_projection(None, ScalarRepresentation::Overflow));
    }
    Ok(reported.map_form(|form| Either::Right(Either::Left(form))))
}
pub(crate) fn airy_ai<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    evaluate(
        x,
        Kind::Ai,
        false,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_ai)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_ai_evaluation<S: RealScalar>(
    x: S,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(
        x,
        Kind::Ai,
        false,
        options,
        algorithm_component!(stringify!(airy_ai)),
        context,
    )
}

pub(crate) fn airy_bi<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    evaluate(
        x,
        Kind::Bi,
        false,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_bi)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_bi_evaluation<S: RealScalar>(
    x: S,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(
        x,
        Kind::Bi,
        false,
        options,
        algorithm_component!(stringify!(airy_bi)),
        context,
    )
}

pub(crate) fn airy_ai_scaled<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    evaluate(
        x,
        Kind::Ai,
        true,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_ai_scaled)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_ai_scaled_evaluation<S: RealScalar>(
    x: S,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(
        x,
        Kind::Ai,
        true,
        options,
        algorithm_component!(stringify!(airy_ai_scaled)),
        context,
    )
}

pub(crate) fn airy_bi_scaled<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    evaluate(
        x,
        Kind::Bi,
        true,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_bi_scaled)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_bi_scaled_evaluation<S: RealScalar>(
    x: S,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(
        x,
        Kind::Bi,
        true,
        options,
        algorithm_component!(stringify!(airy_bi_scaled)),
        context,
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::excessive_precision
)]
mod tests {
    use super::*;

    #[test]
    fn test_airy_taylor_reports_iteration_exhaustion() {
        let options = AiryOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
        let result = crate::airy_ai_with_options(1.0_f64, options).unwrap_err();
        let result_counts = crate::test_support::exhausted_counts(&result);
        assert_eq!(result_counts.iterations(), 1);
    }

    // Reference values computed via high-precision series evaluation
    // (Python with extended precision, verified against DLMF tables).

    // ---- Ai(0) and Bi(0) ----

    #[test]
    fn test_airy_ai_zero_f64() {
        let val = airy_ai(0.0_f64).unwrap();
        let expected = 0.355_028_053_887_817_24_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Ai(0) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_zero_f64() {
        let val = airy_bi(0.0_f64).unwrap();
        let expected = 0.614_926_627_446_000_7_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Bi(0) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_zero_f32() {
        let val = airy_ai(0.0_f32).unwrap();
        let expected = 0.355_028_06_f32;
        assert!(
            (val - expected).abs() < 1e-5,
            "Ai(0) f32 = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_zero_f32() {
        let val = airy_bi(0.0_f32).unwrap();
        let expected = 0.614_926_6_f32;
        assert!(
            (val - expected).abs() < 1e-5,
            "Bi(0) f32 = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_zero_values_are_correctly_rounded() {
        assert_eq!(airy_ai(0.0_f64).unwrap().to_bits(), 0x3fd6_b8c7_9627_15b8);
        assert_eq!(airy_bi(0.0_f64).unwrap().to_bits(), 0x3fe3_ad7a_9b4a_3ea9);
        assert_eq!(airy_ai(0.0_f32).unwrap().to_bits(), 0x3eb5_c63d);
        assert_eq!(airy_bi(0.0_f32).unwrap().to_bits(), 0x3f1d_6bd5);
    }

    #[test]
    fn test_airy_zero_shortcuts_preserve_two_sided_neighborhoods() {
        let delta64 = 1e-8_f64;
        let ai_left = airy_ai(-delta64).unwrap();
        let ai_zero = airy_ai(0.0_f64).unwrap();
        let ai_right = airy_ai(delta64).unwrap();
        assert!(ai_left > ai_zero && ai_zero > ai_right);

        let bi_left = airy_bi(-delta64).unwrap();
        let bi_zero = airy_bi(0.0_f64).unwrap();
        let bi_right = airy_bi(delta64).unwrap();
        assert!(bi_left < bi_zero && bi_zero < bi_right);

        let delta32 = 1e-4_f32;
        assert!(airy_ai(-delta32).unwrap() > airy_ai(0.0_f32).unwrap());
        assert!(airy_ai(delta32).unwrap() < airy_ai(0.0_f32).unwrap());
        assert!(airy_bi(-delta32).unwrap() < airy_bi(0.0_f32).unwrap());
        assert!(airy_bi(delta32).unwrap() > airy_bi(0.0_f32).unwrap());
    }

    // ---- Ai for positive x (exponentially decaying) ----

    #[test]
    fn test_airy_ai_positive_1_f64() {
        let val = airy_ai(1.0_f64).unwrap();
        let expected = 0.135_292_416_312_881_4_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Ai(1) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_positive_2_f64() {
        let val = airy_ai(2.0_f64).unwrap();
        let expected = 0.034_924_130_423_274_4_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Ai(2) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_positive_5_f64() {
        let val = airy_ai(5.0_f64).unwrap();
        let expected = 1.083_444_281_360_744_2e-4_f64;
        assert!(
            (val - expected).abs() / expected < 8e-15,
            "Ai(5) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_positive_10_f64() {
        let val = airy_ai(10.0_f64).unwrap();
        let expected = 1.104_753_255_289_868_6e-10_f64;
        assert!(
            (val - expected).abs() / expected < 8e-15,
            "Ai(10) = {val}, expected {expected}"
        );
    }

    // ---- Bi for positive x (exponentially growing) ----

    #[test]
    fn test_airy_bi_positive_1_f64() {
        let val = airy_bi(1.0_f64).unwrap();
        let expected = 1.207_423_594_952_871_4_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Bi(1) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_positive_2_f64() {
        let val = airy_bi(2.0_f64).unwrap();
        let expected = 3.298_094_999_978_215_f64;
        assert!(
            (val - expected).abs() < 1e-9,
            "Bi(2) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_positive_5_f64() {
        let val = airy_bi(5.0_f64).unwrap();
        let expected = 657.792_044_171_171_1_f64;
        assert!(
            (val - expected).abs() / expected < 8e-15,
            "Bi(5) = {val}, expected {expected}"
        );
    }

    // ---- Ai for negative x (oscillatory) ----

    #[test]
    fn test_airy_ai_negative_1_f64() {
        let val = airy_ai(-1.0_f64).unwrap();
        let expected = 0.535_560_883_292_352_1_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Ai(-1) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_negative_3_f64() {
        // Ai(-3) is negative (oscillatory region)
        let val = airy_ai(-3.0_f64).unwrap();
        let expected = -0.378_814_293_677_658_f64;
        assert!(
            (val - expected).abs() < 1e-9,
            "Ai(-3) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_negative_5_f64() {
        let val = airy_ai(-5.0_f64).unwrap();
        let expected = 0.350_761_009_024_114_3_f64;
        assert!(
            (val - expected).abs() < 8e-15 * expected.abs(),
            "Ai(-5) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_negative_10_f64() {
        let val = airy_ai(-10.0_f64).unwrap();
        let expected = 0.040_241_238_486_443_19_f64;
        assert!(
            (val - expected).abs() < 8e-15 * expected.abs(),
            "Ai(-10) = {val}, expected {expected}"
        );
    }

    // ---- Bi for negative x (oscillatory) ----

    #[test]
    fn test_airy_bi_negative_1_f64() {
        let val = airy_bi(-1.0_f64).unwrap();
        let expected = 0.103_997_389_496_945_f64;
        assert!(
            (val - expected).abs() < 1e-12,
            "Bi(-1) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_negative_3_f64() {
        let val = airy_bi(-3.0_f64).unwrap();
        let expected = -0.198_289_626_374_927_f64;
        assert!(
            (val - expected).abs() < 1e-9,
            "Bi(-3) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_negative_5_f64() {
        let val = airy_bi(-5.0_f64).unwrap();
        let expected = -0.138_369_134_901_600_58_f64;
        assert!(
            (val - expected).abs() < 8e-15 * expected.abs(),
            "Bi(-5) = {val}, expected {expected}"
        );
    }

    // ---- f32 tests ----

    #[test]
    fn test_airy_ai_positive_1_f32() {
        let val = airy_ai(1.0_f32).unwrap();
        let expected = 0.135_292_42_f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "Ai(1) f32 = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_positive_1_f32() {
        let val = airy_bi(1.0_f32).unwrap();
        let expected = 1.207_423_6_f32;
        assert!(
            (val - expected).abs() < 1e-3,
            "Bi(1) f32 = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_ai_negative_1_f32() {
        let val = airy_ai(-1.0_f32).unwrap();
        let expected = 0.535_560_9_f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "Ai(-1) f32 = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_bi_negative_1_f32() {
        let val = airy_bi(-1.0_f32).unwrap();
        let expected = 0.103_997_4_f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "Bi(-1) f32 = {val}, expected {expected}"
        );
    }

    // ---- Non-finite input tests ----

    #[test]
    fn test_airy_ai_nan() {
        {
            let outcome = &(airy_ai(f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn test_airy_bi_nan() {
        {
            let outcome = &(airy_bi(f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn test_airy_ai_inf() {
        {
            let outcome = &(airy_ai(f64::INFINITY));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn test_airy_bi_inf() {
        {
            let outcome = &(airy_bi(f64::INFINITY));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn test_airy_ai_neg_inf() {
        {
            let outcome = &(airy_ai(f64::NEG_INFINITY));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    // ---- Near-zero region ----

    #[test]
    fn test_airy_ai_small_positive_f64() {
        let val = airy_ai(0.01_f64).unwrap();
        assert!(val > 0.35, "Ai(0.01) = {val}");
        assert!(val < 0.36, "Ai(0.01) = {val}");
    }

    #[test]
    fn test_airy_bi_small_positive_f64() {
        let val = airy_bi(0.01_f64).unwrap();
        assert!(val > 0.61, "Bi(0.01) = {val}");
        assert!(val < 0.63, "Bi(0.01) = {val}");
    }

    #[test]
    fn test_scaled_airy_large_positive_arguments() {
        let ai_scaled = airy_ai_scaled(1_000.0_f64).unwrap();
        let bi_scaled = airy_bi_scaled(1_000.0_f64).unwrap();
        assert!(ai_scaled.is_finite() && ai_scaled > 0.0);
        assert!(bi_scaled.is_finite() && bi_scaled > 0.0);
        assert_eq!(airy_ai(1_000.0_f64).unwrap().to_bits(), 0.0_f64.to_bits());
        {
            let outcome = &(airy_bi(1_000.0_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_large_negative_airy_phase_is_scaled_before_multiplication() {
        [airy_ai(-1.0e300_f64), airy_bi(-1.0e300_f64)]
            .into_iter()
            .for_each(|result| assert!(result.unwrap().is_finite()));
    }

    #[test]
    fn negative_exponent_bins_match_high_precision_oracles() {
        let f64_cases = [
            (
                f64::from_bits((1_123_u64) << 52),
                f64::from_bits(0xbe50_da3e_13e9_0f69),
                f64::from_bits(0xbe39_e7b6_af27_dbe7),
            ),
            (
                f64::from_bits((1_323_u64) << 52),
                f64::from_bits(0xbb31_eb29_9e40_aff1),
                f64::from_bits(0xbb01_a8ea_840a_dc35),
            ),
            (
                f64::MAX,
                f64::from_bits(0x2fd6_7e7a_030a_b76a),
                f64::from_bits(0xafdc_3ee0_f1a7_fff5),
            ),
        ];
        f64_cases
            .into_iter()
            .for_each(|(magnitude, expected_ai, expected_bi)| {
                let ai = airy_ai(-magnitude).unwrap();
                let bi = airy_bi(-magnitude).unwrap();
                assert!(((ai - expected_ai) / expected_ai).abs() < 2.0e-13);
                assert!(((bi - expected_bi) / expected_bi).abs() < 2.0e-13);
            });

        let magnitude = f32::MAX;
        let expected_ai = f32::from_bits(0xaec6_2e5d);
        let expected_bi = f32::from_bits(0xaed2_28fe);
        let ai = airy_ai(-magnitude).unwrap();
        let bi = airy_bi(-magnitude).unwrap();
        assert!(((ai - expected_ai) / expected_ai).abs() < 3.0e-6);
        assert!(((bi - expected_bi) / expected_bi).abs() < 3.0e-6);
    }

    #[test]
    fn test_airy_ai_small_negative_f64() {
        let val = airy_ai(-0.01_f64).unwrap();
        assert!(val > 0.35, "Ai(-0.01) = {val}");
        assert!(val < 0.36, "Ai(-0.01) = {val}");
    }

    // ---- Monotonicity checks ----

    #[test]
    fn test_airy_ai_positive_decays_f64() {
        let a1 = airy_ai(1.0_f64).unwrap();
        let a2 = airy_ai(2.0_f64).unwrap();
        let a3 = airy_ai(3.0_f64).unwrap();
        assert!(a1 > a2, "Ai should decay: Ai(1)={a1} > Ai(2)={a2}");
        assert!(a2 > a3, "Ai should decay: Ai(2)={a2} > Ai(3)={a3}");
        assert!(a3 > 0.0, "Ai should be positive for x > 0");
    }

    #[test]
    fn test_airy_bi_positive_grows_f64() {
        let b1 = airy_bi(1.0_f64).unwrap();
        let b2 = airy_bi(2.0_f64).unwrap();
        let b3 = airy_bi(3.0_f64).unwrap();
        assert!(b1 < b2, "Bi should grow: Bi(1)={b1} < Bi(2)={b2}");
        assert!(b2 < b3, "Bi should grow: Bi(2)={b2} < Bi(3)={b3}");
    }

    // ---- Wronskian identity ----
    //
    // W(Ai, Bi) = Ai(x) Bi'(x) - Ai'(x) Bi(x) = 1/pi
    // Verified numerically with central differences.

    #[test]
    fn test_airy_wronskian_at_zero_f64() {
        let h = 1e-6_f64;
        let ai0 = airy_ai(0.0_f64).unwrap();
        let bi0 = airy_bi(0.0_f64).unwrap();
        let ai_h = airy_ai(h).unwrap();
        let ai_mh = airy_ai(-h).unwrap();
        let bi_h = airy_bi(h).unwrap();
        let bi_mh = airy_bi(-h).unwrap();

        let ai_prime = (ai_h - ai_mh) / (2.0 * h);
        let bi_prime = (bi_h - bi_mh) / (2.0 * h);

        let wronskian = ai0.mul_add(bi_prime, -(ai_prime * bi0));
        let expected = 1.0 / core::f64::consts::PI;
        assert!(
            (wronskian - expected).abs() < 1e-5,
            "Wronskian at x=0: {wronskian}, expected {expected}"
        );
    }

    #[test]
    fn test_airy_wronskian_at_2_f64() {
        let h = 1e-6_f64;
        let x = 2.0_f64;
        let ai = airy_ai(x).unwrap();
        let bi = airy_bi(x).unwrap();
        let ai_h = airy_ai(x + h).unwrap();
        let ai_mh = airy_ai(x - h).unwrap();
        let bi_h = airy_bi(x + h).unwrap();
        let bi_mh = airy_bi(x - h).unwrap();

        let ai_prime = (ai_h - ai_mh) / (2.0 * h);
        let bi_prime = (bi_h - bi_mh) / (2.0 * h);

        let wronskian = ai.mul_add(bi_prime, -(ai_prime * bi));
        let expected = 1.0 / core::f64::consts::PI;
        assert!(
            (wronskian - expected).abs() < 1e-4,
            "Wronskian at x=2: {wronskian}, expected {expected}"
        );
    }
}
