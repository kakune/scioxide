//! Validated configuration and reusable prepared special-function contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, ExecutionContext, RelativeTolerance, SciOxideErrorKind, Tolerance};
use scioxide_special::*;

fn momentum(j: i32) -> AngularMomentum {
    AngularMomentum::new(j).unwrap()
}

fn state(twice_j: i32, twice_m: i32) -> AngularState {
    AngularState::new(twice_j, twice_m).unwrap()
}

#[test]
fn configuration_is_valid_before_preparation() {
    for (minimum, maximum) in [(0, 16), (1, 16), (3, 2)] {
        assert_eq!(
            SpectralTerms::new(minimum, maximum).unwrap_err().kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    let terms = SpectralTerms::new(2, 2).unwrap();
    let options = SpectralOptions::<f64>::new()
        .with_terms(terms)
        .with_tolerance(Tolerance::new(0.0, 1e-8).unwrap())
        .with_iteration_budget(Budget::ONE)
        .with_preparation_bytes(Budget::ONE);
    assert_eq!(options.terms().minimum(), 2);
    assert_eq!(options.terms().maximum(), 2);
    assert_eq!(options.tolerance().relative(), 1e-8);
    assert_eq!(options.iteration_budget(), Budget::ONE);
    assert_eq!(options.preparation_bytes(), Budget::ONE);
    assert!(Budget::try_from(0).is_err());
    let options = AngularOptions::new().with_preparation_bytes(Budget::ONE);
    assert_eq!(options.preparation_bytes(), Budget::ONE);
    let error = AngularPlan::new(momentum(20), options).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(
        AngularPlan::new(
            momentum(20),
            AngularOptions::new().with_workspace_bytes(Budget::ONE)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        MathieuPlan::new(
            0,
            MathieuParity::Even,
            1.0_f64,
            SpectralNormalization::Standard,
            SpectralOptions::new().with_preparation_bytes(Budget::ONE)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        MathieuPlan::new(
            4,
            MathieuParity::Even,
            0.0_f64,
            SpectralNormalization::Standard,
            SpectralOptions::new().with_terms(terms)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        MathieuPlan::new(
            0,
            MathieuParity::Even,
            1.0_f64,
            SpectralNormalization::Standard,
            SpectralOptions::new().with_terms(SpectralTerms::new(2, usize::MAX).unwrap())
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let options = EllipsoidalOptions::new(RelativeTolerance::new(0.0_f64).unwrap(), Budget::ONE)
        .with_relative_tolerance(RelativeTolerance::new(1e-8).unwrap());
    assert_eq!(options.relative_tolerance().relative(), 1e-8);
}

#[test]
fn angular_inputs_reject_invalid_states_at_construction() {
    assert!(AngularMomentum::new(-1).is_err());
    assert_eq!(momentum(i32::MAX).twice_j(), i32::MAX);
    for (j, m) in [(0, 1), (1, 0), (2, 3), (i32::MAX, i32::MIN)] {
        assert_eq!(
            AngularState::new(j, m).unwrap_err().kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    for (j, m) in [(0, 0), (1, -1), (1, 1), (i32::MAX, -i32::MAX)] {
        let value = state(j, m);
        assert_eq!(value.momentum(), momentum(j));
        assert_eq!(value.twice_m(), m);
    }
    let plan = AngularPlan::new(momentum(20), AngularOptions::default()).unwrap();
    assert_eq!(
        plan.gaunt::<f64>([state(1, 1); 3]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        plan.wigner_6j::<f64>([momentum(22); 6]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let mut incompatible = AngularPlan::new(momentum(2), AngularOptions::default())
        .unwrap()
        .workspace()
        .unwrap();
    assert_eq!(
        plan.wigner_3j_scoped::<f64>(
            [state(0, 0); 3],
            &mut incompatible,
            ExecutionContext::sequential_native()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn all_angular_operations_reuse_reserved_storage_and_clone() {
    fn traits<T: Clone + Send + Sync>() {}
    traits::<AngularPlan>();
    let plan = AngularPlan::new(momentum(20), AngularOptions::default()).unwrap();
    let cloned = plan.clone();
    let mut workspace = plan.workspace().unwrap();
    let q = [state(2, 2), state(2, -2), state(0, 0)];
    // The owned methods allocate a workspace; the scoped methods reuse the reserved one.
    macro_rules! check {
        ($plain:ident, $scoped:ident, ($($arg:expr),+), ($($integer:expr),+)) => {{
            let ordinary = plan.$plain::<f64>($($arg),+).unwrap();
            let scoped = cloned.$scoped::<f64>($($arg,)+ &mut workspace, ExecutionContext::sequential_native()).unwrap();
            assert_eq!(ordinary, scoped);
            let narrow = plan.$plain::<f32>($($arg),+).unwrap().try_value().unwrap();
            let value = ordinary.try_value().unwrap();
            assert!((f64::from(narrow)-value).abs() <= 2e-7 * value.abs().max(1.0));
            assert_eq!($plain::<f64>($($integer),+).unwrap(), value);
        }};
    }
    check!(wigner_3j, wigner_3j_scoped, (q), ([2, 2, 0], [2, -2, 0]));
    check!(
        clebsch_gordan,
        clebsch_gordan_scoped,
        (q[0], q[1], q[2]),
        ([2, 2, 0], [2, -2, 0])
    );
    check!(wigner_6j, wigner_6j_scoped, ([momentum(2); 6]), ([2; 6]));
    check!(
        wigner_9j,
        wigner_9j_scoped,
        ([2, 2, 2, 2, 2, 2, 2, 2, 0].map(momentum)),
        ([2, 2, 2, 2, 2, 2, 2, 2, 0])
    );
    check!(racah_w, racah_w_scoped, ([momentum(2); 6]), ([2; 6]));
    check!(gaunt, gaunt_scoped, (q), ([1, 1, 0], [1, -1, 0]));
    let zeros = plan.wigner_3j::<f64>([state(2, 0); 3]).unwrap();
    assert_eq!(zeros.termination(), SpecialFunctionTermination::Exact);
    assert_eq!(zeros.try_value().unwrap(), 0.0);
}

#[test]
fn spectral_clones_preserve_positional_semantics() {
    fn traits<T: Clone + Send + Sync>() {}
    traits::<MathieuPlan<f32>>();
    traits::<MathieuPlan<f64>>();
    traits::<SpheroidalPlan<f32>>();
    traits::<SpheroidalPlan<f64>>();
    traits::<EllipsoidalHarmonicPlan<f32>>();
    traits::<EllipsoidalHarmonicPlan<f64>>();
    traits::<HypergeometricFunction<f64>>();
    traits::<HypergeometricFunction<scioxide_core::Complex<f64>>>();
    for normalization in [
        SpectralNormalization::Standard,
        SpectralNormalization::UnitL2,
    ] {
        let mathieu = MathieuPlan::new(
            2,
            MathieuParity::Even,
            1.0_f64,
            normalization,
            SpectralOptions::default(),
        )
        .unwrap();
        assert_eq!(mathieu.normalization(), normalization);
        assert_eq!(
            mathieu.angular(0.3).unwrap(),
            mathieu
                .clone()
                .angular_scoped(0.3, ExecutionContext::sequential_native())
                .unwrap()
        );
        let spheroidal = SpheroidalPlan::new(
            SpheroidalKind::Prolate,
            1,
            2,
            1.0_f64,
            normalization,
            SpectralOptions::default(),
        )
        .unwrap();
        assert_eq!(spheroidal.normalization(), normalization);
        assert_eq!(
            spheroidal.angular(0.3).unwrap(),
            spheroidal
                .clone()
                .angular_scoped(0.3, ExecutionContext::sequential_native())
                .unwrap()
        );
    }
    let plan =
        EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 1, 2, EllipsoidalOptions::default()).unwrap();
    let positive = plan
        .first_kind(1.5_f64, EllipsoidalSigns::new(false, false))
        .unwrap();
    let cloned = plan.clone();
    assert_eq!(
        plan.first_kind(1.5_f64, EllipsoidalSigns::default())
            .unwrap(),
        positive
    );
    drop(plan);
    let negative = cloned
        .first_kind_scoped(
            1.5_f64,
            EllipsoidalSigns::new(true, false),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!((positive.try_value().unwrap() + negative.try_value().unwrap()).abs() < 1e-14);
    assert!(
        (*positive.derivative().unwrap().value() + *negative.derivative().unwrap().value()).abs()
            < 1e-14
    );
    assert_eq!(
        ellipsoidal_harmonic(0.5, 1.0, 1, 2, 1.5, EllipsoidalSigns::new(true, false)).unwrap(),
        negative.try_value().unwrap()
    );
}
