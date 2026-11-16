//! Extended-precision forward references, independent of round trips.
#![expect(clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;

use scioxide_core::RealScalar;
use scioxide_fft::{DctType, DstType, Normalization, RealTransformPlan, RealTransformPlanSpec};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    mantissa_bits: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    length: usize,
    family: String,
    kind: usize,
    expected: Vec<f64>,
}

fn check<S: RealScalar>() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("fixtures/real_forward.json")).unwrap();
    assert!(fixtures.mantissa_bits > 53);
    for case in fixtures.cases {
        let n = case.length;
        let mut state = 0x1234_5678_9abc_def0_u64;
        let input: Vec<S> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                S::try_from_f64_lossy(((state >> 40) as f64 - 8_388_608.0) / 8_388_608.0).unwrap()
            })
            .collect();
        let mut output = vec![S::ZERO; n];
        let cx = scioxide_fft::FftPlanningContext::<S>::default().execution();
        if case.family == "dct" {
            let ty = [DctType::II, DctType::III, DctType::IV][case.kind - 2];
            let plan = RealTransformPlan::new(RealTransformPlanSpec::new(
                n,
                (ty).into(),
                scioxide_fft::FftDirection::Forward,
                Normalization::Orthonormal,
            ))
            .unwrap();
            plan.execute_into(&input, &mut output, &mut plan.workspace(()).unwrap(), cx)
                .unwrap();
        } else {
            let ty = [DstType::II, DstType::III, DstType::IV][case.kind - 2];
            let plan = RealTransformPlan::new(RealTransformPlanSpec::new(
                n,
                (ty).into(),
                scioxide_fft::FftDirection::Forward,
                Normalization::Orthonormal,
            ))
            .unwrap();
            plan.execute_into(&input, &mut output, &mut plan.workspace(()).unwrap(), cx)
                .unwrap();
        };
        assert_eq!(output.len(), case.expected.len());
        let (error, norm) = output.iter().zip(case.expected).fold(
            (0.0, 0.0),
            |(error, norm), (actual, expected)| {
                let difference = actual.try_to_f64_exact().unwrap() - expected;
                (error + difference * difference, norm + expected * expected)
            },
        );
        let relative = (error / norm).sqrt();
        let bound = 2.0 * S::EPSILON.try_to_f64_exact().unwrap() * (n as f64).log2();
        numerics::assert_close(
            relative,
            bound,
            format_args!("{}{} n={n}", case.family, case.kind),
        );
    }
}

#[test]
fn real_forward_f32() {
    check::<f32>();
}

#[test]
fn real_forward_f64() {
    check::<f64>();
}
