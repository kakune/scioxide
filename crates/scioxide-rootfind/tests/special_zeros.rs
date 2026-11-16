#![expect(
    clippy::panic,
    reason = "Tests report unexpected fixture cases or solver outcomes."
)]
use SpecialZeroBranch as B;
use SpecialZeroFamily as F;
use scioxide_rootfind::KelvinZeroFunction as K;
use scioxide_rootfind::*;
#[test]
fn named_sequences() {
    for f in [
        F::BesselJ(0.0),
        F::BesselJ(0.5),
        F::BesselJ(-0.49),
        F::BesselJ(32.0),
        F::BesselJ(443.0),
        F::BesselY(0.0),
        F::BesselY(32.0),
        F::BesselJPrime(0.0),
        F::BesselJPrime(1.0),
        F::BesselYPrime(1.0),
        F::AiryAi,
        F::AiryAiPrime,
        F::AiryBi,
        F::AiryBiPrime,
        F::Kelvin(K::Ber),
        F::Kelvin(K::Bei),
        F::Kelvin(K::Ker),
        F::Kelvin(K::Kei),
        F::Kelvin(K::BerPrime),
        F::Kelvin(K::BeiPrime),
        F::Kelvin(K::KerPrime),
        F::Kelvin(K::KeiPrime),
        F::FresnelCPrime,
        F::FresnelSPrime,
    ] {
        let result = special_zeros_with_options(f, B::Real, 1, 5, &SpecialZeroOptions::default())
            .unwrap_or_else(|e| panic!("{f:?}: {e}"));
        assert_eq!(
            result.report().termination(),
            RootTermination::Complete,
            "{f:?}: {:?}",
            result.report()
        );
        assert_eq!(result.roots().len(), 5);
        eprintln!(
            "{f:?}: {:?}",
            result
                .roots()
                .iter()
                .map(|z| z.point().re)
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn complex_sequences() {
    for f in [
        F::BesselY(0.0),
        F::BesselY(1.0),
        F::BesselYPrime(1.0),
        F::AiryBi,
        F::AiryBiPrime,
        F::Erf,
        F::FresnelC,
        F::FresnelS,
    ] {
        let result = special_zeros_with_options(f, B::UPPER, 1, 5, &SpecialZeroOptions::default())
            .unwrap_or_else(|e| panic!("{f:?}: {e}"));
        assert_eq!(
            result.report().termination(),
            RootTermination::Complete,
            "{f:?}: {:?}",
            result.report()
        );
        assert_eq!(result.roots().len(), 5);
        eprintln!(
            "{f:?}: {:?}",
            result.roots().iter().map(|z| z.point()).collect::<Vec<_>>()
        );
    }
}
#[test]
fn high_indices_and_turning_points() {
    for (f, branch, index) in [
        (F::BesselJ(1024.0), B::Real, 1),
        (F::BesselY(1024.0), B::Real, 1),
        (F::BesselJPrime(1024.0), B::Real, 1),
        (F::BesselYPrime(1024.0), B::Real, 1),
        (F::BesselJ(0.0), B::Real, 10000),
        (F::BesselY(0.0), B::Real, 10000),
        (F::AiryAi, B::Real, 10000),
        (F::AiryBiPrime, B::Real, 10000),
        (F::Kelvin(K::BerPrime), B::Real, 10000),
        (F::Kelvin(K::Kei), B::Real, 10000),
        (F::Erf, B::UPPER, 10000),
        (F::FresnelC, B::UPPER, 10000),
        (F::FresnelS, B::UPPER, 10000),
        (F::BesselY(0.0), B::UPPER, 10000),
        (F::AiryBi, B::UPPER, 10000),
        (F::AiryBiPrime, B::UPPER, 10000),
    ] {
        let r = special_zeros_with_options(f, branch, index, 1, &SpecialZeroOptions::default())
            .unwrap_or_else(|e| panic!("{f:?} {index}: {e}"));
        assert_eq!(
            r.report().termination(),
            RootTermination::Complete,
            "{f:?} {index}: {:?}",
            r.report()
        );
    }
}

#[test]
fn derivative_zero_stops_never_publish_a_base_zero_as_candidate() {
    use scioxide_core::Budget;
    // These derivative zeros are bracketed by zeros of J₁₀ or Y₂.₅. With one
    // refinement step the base zero stops first; it only bounds the derivative
    // zero, so no candidate is an estimate of it.
    for family in [F::BesselJPrime(10.0_f64), F::BesselYPrime(2.5)] {
        for start in 1..=3 {
            let options = SpecialZeroOptions::default().with_refinement_steps(Budget::ONE);
            let zeros = special_zeros_with_options(family, B::Real, start, 1, &options).unwrap();
            let report = zeros.report();
            assert_eq!(report.termination(), RootTermination::IterationLimit);
            assert_eq!(report.candidate(), None, "{family:?} {start}");
        }
    }
}
