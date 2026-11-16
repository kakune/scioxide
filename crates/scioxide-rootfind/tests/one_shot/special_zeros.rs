//! `special_zeros`.
use scioxide_core::{
    Argument, ExecutionContext, InvalidInputReason, RealScalar, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, argument,
};
use scioxide_rootfind::{
    KelvinZeroFunction, RootTermination, SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions,
    SpecialZeroWorkspace, special_zeros, special_zeros_into, special_zeros_with_options,
};

/// `j₀,₁, …, j₀,₅`, the first positive zeros of `J₀`.
const BESSEL_J0_ZEROS: [f64; 5] = [
    2.404_825_557_695_773,
    5.520_078_110_286_311,
    8.653_727_912_911_013,
    11.791_534_439_014_281,
    14.930_917_708_487_787,
];

fn invalid(error: &SciOxideError) -> Option<(Argument, InvalidInputReason)> {
    match error.view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}

fn configured_real_parts<S: RealScalar>(family: SpecialZeroFamily<S>, count: usize) -> Vec<S> {
    let zeros = special_zeros_with_options(
        family,
        SpecialZeroBranch::Real,
        1,
        count,
        &SpecialZeroOptions::default(),
    )
    .unwrap();
    assert!(zeros.report().is_success(), "{:?}", zeros.report());
    zeros.roots().iter().map(|zero| zero.point().re).collect()
}

fn matches_configured<S: RealScalar>(family: SpecialZeroFamily<S>) {
    let one_shot = special_zeros(family, 8).unwrap();
    let configured = configured_real_parts(family, 8);
    assert_eq!(one_shot.len(), 8);
    assert_eq!(one_shot, configured, "{family:?}");
    for (a, b) in one_shot.iter().zip(&configured) {
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}

#[test]
fn special_zeros_equals_the_configured_real_branch() {
    matches_configured(SpecialZeroFamily::BesselJ(0.0_f64));
    matches_configured(SpecialZeroFamily::BesselJ(0.0_f32));
    matches_configured(SpecialZeroFamily::<f64>::AiryAi);
    matches_configured(SpecialZeroFamily::<f32>::AiryAi);
    matches_configured(SpecialZeroFamily::<f64>::Kelvin(KelvinZeroFunction::Ber));
    matches_configured(SpecialZeroFamily::<f32>::Kelvin(KelvinZeroFunction::Ber));
}

#[test]
fn special_zeros_reach_reference_values() {
    let zeros = special_zeros(SpecialZeroFamily::BesselJ(0.0_f64), 5).unwrap();
    assert_eq!(zeros.len(), 5);
    for (zero, expected) in zeros.iter().zip(BESSEL_J0_ZEROS) {
        assert!((zero - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }
    let zeros = special_zeros(SpecialZeroFamily::BesselJ(0.0_f32), 5).unwrap();
    assert_eq!(zeros.len(), 5);
    for (zero, expected) in zeros.iter().zip(BESSEL_J0_ZEROS) {
        assert!((f64::from(*zero) - expected).abs() <= f64::from(f32::EPSILON) * expected);
    }
    // Airy zeros are negative and decrease: a₁ = −2.338107410459767….
    let zeros = special_zeros(SpecialZeroFamily::<f64>::AiryAi, 3).unwrap();
    assert!((zeros[0] + 2.338_107_410_459_767).abs() < 1e-14);
    assert!(zeros.windows(2).all(|pair| pair[1] < pair[0]));
}

#[test]
fn special_zeros_report_their_own_arguments() {
    let order = special_zeros(SpecialZeroFamily::BesselJ(-1.0_f64), 3).unwrap_err();
    assert_eq!(
        invalid(&order),
        Some((argument!("family"), InvalidInputReason::OutOfRange))
    );
    let order = special_zeros(SpecialZeroFamily::BesselY(-0.5_f32), 3).unwrap_err();
    assert_eq!(
        invalid(&order),
        Some((argument!("family"), InvalidInputReason::OutOfRange))
    );
    let order = special_zeros(SpecialZeroFamily::BesselY(f32::NAN), 3).unwrap_err();
    assert_eq!(order.kind(), SciOxideErrorKind::ScalarConversion);
    // 1 + 2⁵² exceeds the last exactly representable index 2⁵².
    let count = special_zeros(SpecialZeroFamily::<f64>::AiryAi, 1 << 52).unwrap_err();
    assert_eq!(
        invalid(&count),
        Some((argument!("count"), InvalidInputReason::OutOfRange))
    );
}

#[test]
fn special_zeros_without_real_zeros_are_empty() {
    assert!(
        special_zeros(SpecialZeroFamily::BesselJ(0.0_f64), 0)
            .unwrap()
            .is_empty()
    );
    for family in [
        SpecialZeroFamily::<f64>::Erf,
        SpecialZeroFamily::FresnelC,
        SpecialZeroFamily::FresnelS,
    ] {
        assert!(special_zeros(family, 3).unwrap().is_empty(), "{family:?}");
        let configured = special_zeros_with_options(
            family,
            SpecialZeroBranch::Real,
            1,
            3,
            &SpecialZeroOptions::default(),
        )
        .unwrap();
        assert_eq!(configured.report().termination(), RootTermination::NoZeros);
    }
}

#[test]
fn a_workspace_for_a_count_serves_special_zeros_into() {
    let family = SpecialZeroFamily::BesselJ(0.0_f64);
    let options = SpecialZeroOptions::default();
    let mut workspace = SpecialZeroWorkspace::workspace(5, &options).unwrap();
    assert_eq!(workspace.capacity(), 5);
    for count in [5, 2, 0] {
        let mut output = vec![None; count];
        let report = special_zeros_into(
            family,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(report.termination(), RootTermination::Complete);
        assert_eq!(report.completed(), count);
        let zeros: Vec<_> = output.iter().map(|zero| zero.unwrap().point().re).collect();
        assert_eq!(zeros, special_zeros(family, count).unwrap());
    }

    // An output longer than the workspace is rejected before any work.
    let mut output = [None; 6];
    let error = special_zeros_into(
        family,
        SpecialZeroBranch::Real,
        1,
        &options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap_err();
    assert_eq!(
        invalid(&error),
        Some((argument!("workspace"), InvalidInputReason::OutOfRange))
    );
    assert!(output.iter().all(Option::is_none));
}

#[test]
fn special_zeros_into_names_output_for_the_count() {
    let options = SpecialZeroOptions::<f64>::default();
    let mut workspace = SpecialZeroWorkspace::workspace(1, &options).unwrap();
    let mut solve = |start: usize, output: &mut [_]| {
        special_zeros_into(
            SpecialZeroFamily::AiryAi,
            SpecialZeroBranch::Real,
            start,
            &options,
            output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
    };
    // `start + output.len()` may equal 2⁵² but not exceed it.
    let report = solve(1 << 52, &mut []).unwrap();
    assert_eq!(report.completed(), 0);
    let error = solve(1 << 52, &mut [None]).unwrap_err();
    assert_eq!(
        invalid(&error),
        Some((argument!("output"), InvalidInputReason::OutOfRange))
    );
    let error = special_zeros_with_options(
        SpecialZeroFamily::<f64>::AiryAi,
        SpecialZeroBranch::Real,
        1 << 52,
        1,
        &options,
    )
    .unwrap_err();
    assert_eq!(
        invalid(&error),
        Some((argument!("count"), InvalidInputReason::OutOfRange))
    );
}
