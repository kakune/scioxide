//! Fourier and trigonometric normalization, energy and symmetry identities.
#![expect(clippy::cast_precision_loss, clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;
use numerics::{epsilon, scalar, wide};
#[path = "support/real_definitions.rs"]
mod real_definitions;
use real_definitions::{dct_roundtrip_factor, direct_dct, direct_dst, dst_roundtrip_factor};

use scioxide_core::{Complex, ExecutionPolicy, NumericScalar, RealScalar};
use scioxide_fft::{
    DctType, DstType, FftBackendStrategy, FftDirection, FftInput, FftLayout, FftOutput, FftPlan,
    FftPlanSpec, HermitianSpectrum, Normalization, RealTransformPlan, dct, dst, fft, idct, idst,
    ifft, irfft, rfft,
};

#[cfg(feature = "std")]
const TEST_PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
#[cfg(not(feature = "std"))]
const TEST_PROCESSOR: scioxide_core::PortableProcessor = scioxide_core::PortableProcessor;

const LENGTHS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16];
const NORMALIZATIONS: &[Normalization] = &[
    Normalization::Forward,
    Normalization::Inverse,
    Normalization::Orthonormal,
    Normalization::Unscaled,
];

fn complex_signal<S: RealScalar>(n: usize) -> Vec<Complex<S>> {
    (0..n)
        .map(|index| {
            let sample = (index + 1) as f64;
            Complex::new(
                scalar::<S>((0.37 * sample).sin() - 0.2 * (0.11 * sample).cos()),
                scalar::<S>(0.5 * (0.23 * sample).cos() + 0.1 * (0.19 * sample).sin()),
            )
        })
        .collect()
}

fn real_signal<S: RealScalar>(n: usize) -> Vec<S> {
    (0..n)
        .map(|index| {
            let sample = (index + 1) as f64;
            scalar::<S>((0.41 * sample).sin() - 0.25 * (0.17 * sample).cos())
        })
        .collect()
}

fn fourier_scales(normalization: Normalization, n: usize) -> (f64, f64) {
    let n = n as f64;
    match normalization {
        Normalization::Forward => (n.recip(), 1.0),
        Normalization::Inverse => (1.0, n.recip()),
        Normalization::Orthonormal => {
            let scale = n.sqrt().recip();
            (scale, scale)
        }
        Normalization::Unscaled => (1.0, 1.0),
    }
}

fn complex_energy<S: RealScalar>(values: &[Complex<S>]) -> f64 {
    values.iter().map(|value| wide(value.norm_sqr())).sum()
}

fn real_energy<S: RealScalar>(values: &[S]) -> f64 {
    values
        .iter()
        .map(|value| {
            let value = wide(*value);
            value * value
        })
        .sum()
}

fn packed_real_energy<S: RealScalar>(values: &[Complex<S>], n: usize) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let unpaired = index == 0 || (n.is_multiple_of(2) && index == n / 2);
            let weight = if unpaired { 1.0 } else { 2.0 };
            weight * wide(value.norm_sqr())
        })
        .sum()
}

fn length_factor(n: usize) -> f64 {
    (n as f64).log2().max(1.0)
}

fn comparison_bound<S: RealScalar>(n: usize, expected_scale: f64) -> f64 {
    256.0 * epsilon::<S>() * length_factor(n) * expected_scale.max(1.0)
}

fn energy_bound<S: RealScalar>(n: usize, actual: f64, expected: f64) -> f64 {
    512.0 * epsilon::<S>() * length_factor(n) * actual.abs().max(expected.abs()).max(1.0)
}

fn assert_energy_close<S: RealScalar>(actual: f64, expected: f64, n: usize, context: &str) {
    numerics::assert_close(
        (actual - expected).abs(),
        energy_bound::<S>(n, actual, expected),
        context,
    );
}

#[track_caller]
fn assert_complex_close<S: RealScalar>(actual: &[Complex<S>], expected: &[Complex<S>], n: usize) {
    let scale = expected.iter().map(|z| wide(z.norm())).fold(0.0, f64::max);
    let bound = comparison_bound::<S>(n, scale);
    numerics::assert_complex_close(actual, expected, Complex::norm, |_| bound);
}

#[track_caller]
fn assert_real_close<S: RealScalar>(actual: &[S], expected: &[S], n: usize) {
    let scale = expected.iter().map(|z| wide(*z).abs()).fold(0.0, f64::max);
    let bound = comparison_bound::<S>(n, scale);
    numerics::assert_real_close(actual, expected, |_| bound);
}

fn scale_complex<S: RealScalar>(values: &[Complex<S>], scale: f64) -> Vec<Complex<S>> {
    let scale = scalar::<S>(scale);
    values.iter().map(|value| *value * scale).collect()
}

fn scale_real<S: RealScalar>(values: &[S], scale: f64) -> Vec<S> {
    let scale = scalar::<S>(scale);
    values.iter().map(|value| *value * scale).collect()
}

fn dct_real_fft_len(ty: DctType, n: usize) -> usize {
    match ty {
        DctType::I => 2 * (n - 1),
        DctType::II | DctType::III => 2 * n,
        DctType::IV => 4 * n,
    }
}

fn dst_real_fft_len(ty: DstType, n: usize) -> usize {
    match ty {
        DstType::I => 2 * (n + 1),
        DstType::II | DstType::III => 2 * n,
        DstType::IV => 4 * n,
    }
}

fn oracle_bound<S: RealScalar>(input: &[S], real_fft_len: usize) -> f64 {
    512.0
        * epsilon::<S>()
        * length_factor(real_fft_len)
        * input
            .iter()
            .map(|value| wide(*value).abs())
            .sum::<f64>()
            .max(1.0)
        * 2.0
}

fn assert_complex_fft_invariants<S: RealScalar>() {
    for &n in LENGTHS {
        let input = complex_signal::<S>(n);
        let input_energy = complex_energy(&input);
        let raw_forward = fft(&input, Normalization::Unscaled).unwrap();
        let raw_inverse = ifft(&input, Normalization::Unscaled).unwrap();

        for &normalization in NORMALIZATIONS {
            let (forward_scale, inverse_scale) = fourier_scales(normalization, n);
            let forward = fft(&input, normalization).unwrap();
            let inverse = ifft(&input, normalization).unwrap();

            assert_complex_close(&forward, &scale_complex(&raw_forward, forward_scale), n);
            assert_complex_close(&inverse, &scale_complex(&raw_inverse, inverse_scale), n);
            assert_energy_close::<S>(
                complex_energy(&forward),
                n as f64 * forward_scale * forward_scale * input_energy,
                n,
                "complex forward Parseval",
            );
            assert_energy_close::<S>(
                complex_energy(&inverse),
                n as f64 * inverse_scale * inverse_scale * input_energy,
                n,
                "complex inverse Parseval",
            );

            let recovered = ifft(&forward, normalization).unwrap();
            let roundtrip_factor = if normalization == Normalization::Unscaled {
                n as f64
            } else {
                1.0
            };
            let expected_roundtrip = scale_complex(&input, roundtrip_factor);
            assert_complex_close(&recovered, &expected_roundtrip, n);

            let spec = |direction, backend_strategy| {
                FftPlanSpec::new(n, direction, normalization, FftLayout::Complex)
                    .with_execution_policy(ExecutionPolicy::Sequential)
                    .with_backend_strategy(backend_strategy)
            };
            let allocating_plan = FftPlan::<S>::build(
                spec(FftDirection::Forward, FftBackendStrategy::Automatic),
                scioxide_fft::FftPlanningContext::default(),
            )
            .unwrap();
            let mut allocating_forward = input.clone();
            let mut allocating_scratch = allocating_plan.workspace(()).unwrap();
            allocating_plan
                .execute_in_place(
                    &mut allocating_forward,
                    &mut allocating_scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
            assert_complex_close(&allocating_forward, &forward, n);
            let inverse_plan = FftPlan::<S>::build(
                spec(FftDirection::Inverse, FftBackendStrategy::Automatic),
                scioxide_fft::FftPlanningContext::default(),
            )
            .unwrap();
            let mut allocating_inverse = input.clone();
            inverse_plan
                .execute_in_place(
                    &mut allocating_inverse,
                    &mut allocating_scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
            assert_complex_close(&allocating_inverse, &inverse, n);

            for backend in numerics::fft_backends(&TEST_PROCESSOR) {
                let plan = FftPlan::<S>::build(
                    spec(FftDirection::Forward, FftBackendStrategy::Required(backend)),
                    scioxide_fft::FftPlanningContext::default(),
                )
                .unwrap();
                let inverse_plan = FftPlan::<S>::build(
                    spec(FftDirection::Inverse, FftBackendStrategy::Required(backend)),
                    scioxide_fft::FftPlanningContext::default(),
                )
                .unwrap();
                let mut scratch = plan.workspace(()).unwrap();
                let mut planned_forward = input.clone();
                plan.execute_in_place(
                    &mut planned_forward,
                    &mut scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
                assert_complex_close(&planned_forward, &forward, n);

                let mut planned_inverse = input.clone();
                inverse_plan
                    .execute_in_place(
                        &mut planned_inverse,
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                assert_complex_close(&planned_inverse, &inverse, n);

                inverse_plan
                    .execute_in_place(
                        &mut planned_forward,
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                assert_complex_close(&planned_forward, &expected_roundtrip, n);
            }
        }
    }
}

fn assert_real_fft_invariants<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    for &n in LENGTHS {
        let input = real_signal::<S>(n);
        let input_energy = real_energy(&input);
        let raw_forward = rfft(&input, Normalization::Unscaled).unwrap();
        let raw_inverse = irfft(&raw_forward, Normalization::Unscaled).unwrap();
        let packed_energy = packed_real_energy(raw_forward.bins(), n);

        for &normalization in NORMALIZATIONS {
            let (forward_scale, inverse_scale) = fourier_scales(normalization, n);
            let forward = rfft(&input, normalization).unwrap();
            let inverse = irfft(&raw_forward, normalization).unwrap();

            assert_complex_close(
                forward.bins(),
                &scale_complex(raw_forward.bins(), forward_scale),
                n,
            );
            assert_real_close(&inverse, &scale_real(&raw_inverse, inverse_scale), n);
            assert_energy_close::<S>(
                packed_real_energy(forward.bins(), n),
                n as f64 * forward_scale * forward_scale * input_energy,
                n,
                "real FFT packed forward Parseval",
            );
            assert_energy_close::<S>(
                real_energy(&inverse),
                n as f64 * inverse_scale * inverse_scale * packed_energy,
                n,
                "real FFT inverse Parseval",
            );

            let recovered = irfft(&forward, normalization).unwrap();
            let roundtrip_factor = if normalization == Normalization::Unscaled {
                n as f64
            } else {
                1.0
            };
            let expected_roundtrip = scale_real(&input, roundtrip_factor);
            assert_real_close(&recovered, &expected_roundtrip, n);

            for backend in numerics::fft_backends(&TEST_PROCESSOR) {
                let spec = |direction| {
                    FftPlanSpec::new(n, direction, normalization, FftLayout::Real)
                        .with_execution_policy(ExecutionPolicy::Sequential)
                        .with_backend_strategy(FftBackendStrategy::Required(backend))
                };
                let plan = FftPlan::<S>::build(
                    spec(FftDirection::Forward),
                    scioxide_fft::FftPlanningContext::default(),
                )
                .unwrap();
                let inverse_plan = FftPlan::<S>::build(
                    spec(FftDirection::Inverse),
                    scioxide_fft::FftPlanningContext::default(),
                )
                .unwrap();
                let mut scratch = plan.workspace(()).unwrap();
                let mut planned_forward = HermitianSpectrum::zeros(n).unwrap();
                plan.execute_into(
                    FftInput::from(&input),
                    FftOutput::Hermitian {
                        logical_len: planned_forward.logical_len(),
                        bins: scioxide_linalg::VectorViewMut::from_slice(
                            planned_forward.bins_mut(),
                        ),
                    },
                    &mut scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
                assert_complex_close(planned_forward.bins(), forward.bins(), n);

                let mut planned_inverse = vec![<S as scioxide_core::AlgebraicScalar>::ZERO; n];
                inverse_plan
                    .execute_into(
                        FftInput::Hermitian {
                            logical_len: raw_forward.logical_len(),
                            bins: scioxide_linalg::VectorView::from_slice(raw_forward.bins()),
                        },
                        FftOutput::from(&mut planned_inverse),
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                assert_real_close(&planned_inverse, &inverse, n);

                inverse_plan
                    .execute_into(
                        FftInput::Hermitian {
                            logical_len: planned_forward.logical_len(),
                            bins: scioxide_linalg::VectorView::from_slice(planned_forward.bins()),
                        },
                        FftOutput::from(&mut planned_inverse),
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                assert_real_close(&planned_inverse, &expected_roundtrip, n);
            }
        }
    }
}

fn assert_dct_invariants<S: RealScalar>() {
    for &n in LENGTHS {
        let input = real_signal::<S>(n);
        let oracle_input: Vec<_> = input.iter().copied().map(wide).collect();
        let input_energy = real_energy(&input);

        for ty in [DctType::I, DctType::II, DctType::III, DctType::IV] {
            if ty == DctType::I && n == 1 {
                continue;
            }
            let roundtrip_factor = dct_roundtrip_factor(ty, n);
            let bound = oracle_bound::<S>(&input, dct_real_fft_len(ty, n));
            let raw_forward = dct(&input, ty, Normalization::Unscaled).unwrap();
            let raw_inverse = idct(&input, ty, Normalization::Unscaled).unwrap();

            for &normalization in NORMALIZATIONS {
                let forward = dct(&input, ty, normalization).unwrap();
                let inverse = idct(&input, ty, normalization).unwrap();
                let expected_forward = direct_dct(&oracle_input, ty, normalization, false);
                let expected_inverse = direct_dct(&oracle_input, ty, normalization, true);
                numerics::assert_real_close(&forward, &expected_forward, |_| bound);
                numerics::assert_real_close(&inverse, &expected_inverse, |_| bound);

                if normalization != Normalization::Orthonormal {
                    let forward_scale = if normalization == Normalization::Forward {
                        roundtrip_factor.recip()
                    } else {
                        1.0
                    };
                    let inverse_scale = if normalization == Normalization::Inverse {
                        roundtrip_factor.recip()
                    } else {
                        1.0
                    };
                    assert_real_close(&forward, &scale_real(&raw_forward, forward_scale), n);
                    assert_real_close(&inverse, &scale_real(&raw_inverse, inverse_scale), n);
                } else {
                    assert_energy_close::<S>(
                        real_energy(&forward),
                        input_energy,
                        n,
                        "orthonormal DCT forward energy",
                    );
                    assert_energy_close::<S>(
                        real_energy(&inverse),
                        input_energy,
                        n,
                        "orthonormal DCT inverse energy",
                    );
                }

                let expected_roundtrip = scale_real(
                    &input,
                    if normalization == Normalization::Unscaled {
                        roundtrip_factor
                    } else {
                        1.0
                    },
                );
                let recovered = idct(&forward, ty, normalization).unwrap();
                assert_real_close(&recovered, &expected_roundtrip, n);

                for backend in numerics::fft_backends(&TEST_PROCESSOR) {
                    let plan = RealTransformPlan::new(
                        scioxide_fft::RealTransformPlanSpec::new(
                            n,
                            (ty).into(),
                            scioxide_fft::FftDirection::Forward,
                            normalization,
                        )
                        .with_execution_policy(ExecutionPolicy::Sequential)
                        .with_backend_strategy(scioxide_fft::FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let plan_inverse = RealTransformPlan::new(
                        scioxide_fft::RealTransformPlanSpec::new(
                            n,
                            (ty).into(),
                            scioxide_fft::FftDirection::Inverse,
                            normalization,
                        )
                        .with_execution_policy(ExecutionPolicy::Sequential)
                        .with_backend_strategy(scioxide_fft::FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let mut scratch = plan.workspace(()).unwrap();
                    let mut planned_forward = vec![S::ZERO; n];
                    plan.execute_into(
                        &input,
                        &mut planned_forward,
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                    numerics::assert_real_close(&planned_forward, &expected_forward, |_| bound);

                    let mut planned_inverse = vec![S::ZERO; n];
                    plan_inverse
                        .execute_into(
                            &input,
                            &mut planned_inverse,
                            &mut scratch,
                            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                        )
                        .unwrap();
                    numerics::assert_real_close(&planned_inverse, &expected_inverse, |_| bound);

                    plan_inverse
                        .execute_into(
                            &planned_forward,
                            &mut planned_inverse,
                            &mut scratch,
                            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                        )
                        .unwrap();
                    assert_real_close(&planned_inverse, &expected_roundtrip, n);
                }
            }
        }
    }
}

fn assert_dst_invariants<S: RealScalar>() {
    for &n in LENGTHS {
        let input = real_signal::<S>(n);
        let oracle_input: Vec<_> = input.iter().copied().map(wide).collect();
        let input_energy = real_energy(&input);

        for ty in [DstType::I, DstType::II, DstType::III, DstType::IV] {
            let roundtrip_factor = dst_roundtrip_factor(ty, n);
            let bound = oracle_bound::<S>(&input, dst_real_fft_len(ty, n));
            let raw_forward = dst(&input, ty, Normalization::Unscaled).unwrap();
            let raw_inverse = idst(&input, ty, Normalization::Unscaled).unwrap();

            for &normalization in NORMALIZATIONS {
                let forward = dst(&input, ty, normalization).unwrap();
                let inverse = idst(&input, ty, normalization).unwrap();
                let expected_forward = direct_dst(&oracle_input, ty, normalization, false);
                let expected_inverse = direct_dst(&oracle_input, ty, normalization, true);
                numerics::assert_real_close(&forward, &expected_forward, |_| bound);
                numerics::assert_real_close(&inverse, &expected_inverse, |_| bound);

                if normalization != Normalization::Orthonormal {
                    let forward_scale = if normalization == Normalization::Forward {
                        roundtrip_factor.recip()
                    } else {
                        1.0
                    };
                    let inverse_scale = if normalization == Normalization::Inverse {
                        roundtrip_factor.recip()
                    } else {
                        1.0
                    };
                    assert_real_close(&forward, &scale_real(&raw_forward, forward_scale), n);
                    assert_real_close(&inverse, &scale_real(&raw_inverse, inverse_scale), n);
                } else {
                    assert_energy_close::<S>(
                        real_energy(&forward),
                        input_energy,
                        n,
                        "orthonormal DST forward energy",
                    );
                    assert_energy_close::<S>(
                        real_energy(&inverse),
                        input_energy,
                        n,
                        "orthonormal DST inverse energy",
                    );
                }

                let expected_roundtrip = scale_real(
                    &input,
                    if normalization == Normalization::Unscaled {
                        roundtrip_factor
                    } else {
                        1.0
                    },
                );
                let recovered = idst(&forward, ty, normalization).unwrap();
                assert_real_close(&recovered, &expected_roundtrip, n);

                for backend in numerics::fft_backends(&TEST_PROCESSOR) {
                    let plan = RealTransformPlan::new(
                        scioxide_fft::RealTransformPlanSpec::new(
                            n,
                            (ty).into(),
                            scioxide_fft::FftDirection::Forward,
                            normalization,
                        )
                        .with_execution_policy(ExecutionPolicy::Sequential)
                        .with_backend_strategy(scioxide_fft::FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let plan_inverse = RealTransformPlan::new(
                        scioxide_fft::RealTransformPlanSpec::new(
                            n,
                            (ty).into(),
                            scioxide_fft::FftDirection::Inverse,
                            normalization,
                        )
                        .with_execution_policy(ExecutionPolicy::Sequential)
                        .with_backend_strategy(scioxide_fft::FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let mut scratch = plan.workspace(()).unwrap();
                    let mut planned_forward = vec![S::ZERO; n];
                    plan.execute_into(
                        &input,
                        &mut planned_forward,
                        &mut scratch,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                    numerics::assert_real_close(&planned_forward, &expected_forward, |_| bound);

                    let mut planned_inverse = vec![S::ZERO; n];
                    plan_inverse
                        .execute_into(
                            &input,
                            &mut planned_inverse,
                            &mut scratch,
                            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                        )
                        .unwrap();
                    numerics::assert_real_close(&planned_inverse, &expected_inverse, |_| bound);

                    plan_inverse
                        .execute_into(
                            &planned_forward,
                            &mut planned_inverse,
                            &mut scratch,
                            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                        )
                        .unwrap();
                    assert_real_close(&planned_inverse, &expected_roundtrip, n);
                }
            }
        }
    }
}

fn assert_endpoint_weight_examples() {
    let constant = [1.0_f64; 4];
    let dct_two = dct(&constant, DctType::II, Normalization::Orthonormal).unwrap();
    assert_real_close(&dct_two, &[2.0, 0.0, 0.0, 0.0], constant.len());

    let dc = [1.0_f64, 0.0, 0.0, 0.0];
    assert_real_close(
        &dct(&dc, DctType::III, Normalization::Orthonormal).unwrap(),
        &[0.5; 4],
        dc.len(),
    );

    let input = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
    let raw_dst_two = dst(&input, DstType::II, Normalization::Unscaled).unwrap();
    let orthonormal_dst_two = dst(&input, DstType::II, Normalization::Orthonormal).unwrap();
    let scale = (2.0 / input.len() as f64).sqrt();
    let mut expected = scale_real(&raw_dst_two, scale);
    let last = expected.len() - 1;
    expected[last] /= std::f64::consts::SQRT_2;
    assert_real_close(&orthonormal_dst_two, &expected, input.len());
}

#[derive(serde::Deserialize)]
struct RealTransformFixtures {
    metadata: FixtureMetadata,
    cases: Vec<RealTransformFixture>,
}

#[derive(serde::Deserialize)]
struct FixtureMetadata {
    mpmath: String,
    numpy: String,
    scipy: String,
    precisions: [usize; 2],
}

#[derive(serde::Deserialize)]
struct RealTransformFixture {
    width: usize,
    family: String,
    kind: usize,
    normalization: String,
    inverse: bool,
    input: Vec<f64>,
    expected: Vec<f64>,
    scipy_norm: String,
    orthogonalize: bool,
    scale: f64,
    scipy_output: Vec<f64>,
}

fn assert_scipy_fixtures<S: RealScalar>(width: usize) {
    let fixtures: RealTransformFixtures =
        serde_json::from_str(include_str!("fixtures/real_transforms.json")).unwrap();
    assert_eq!(fixtures.metadata.mpmath, "1.3.0");
    assert_eq!(fixtures.metadata.numpy, "2.5.3");
    assert_eq!(fixtures.metadata.scipy, "1.18.1");
    assert_eq!(fixtures.metadata.precisions, [100, 160]);
    let cases: Vec<_> = fixtures
        .cases
        .iter()
        .filter(|case| case.width == width)
        .collect();
    // Lock the fixture matrix, including endpoint impulses and minimum lengths.
    for family in ["dct", "dst"] {
        for kind in 1..=4 {
            for n in [1, 2, 5, 6] {
                for normalization in ["Forward", "Inverse", "Orthonormal", "Unscaled"] {
                    for inverse in [false, true] {
                        let count = cases
                            .iter()
                            .filter(|case| {
                                case.family == family
                                    && case.kind == kind
                                    && case.input.len() == n
                                    && case.normalization == normalization
                                    && case.inverse == inverse
                            })
                            .count();
                        assert_eq!(
                            count,
                            if family == "dct" && kind == 1 && n == 1 {
                                0
                            } else if n == 1 {
                                1
                            } else {
                                3
                            }
                        );
                    }
                }
            }
        }
    }
    for case in cases {
        let normalization = match case.normalization.as_str() {
            "Forward" => Normalization::Forward,
            "Inverse" => Normalization::Inverse,
            "Orthonormal" => Normalization::Orthonormal,
            "Unscaled" => Normalization::Unscaled,
            _ => unreachable!("unknown fixture normalization"),
        };
        let cosine = case.family == "dct";
        let ct = [DctType::I, DctType::II, DctType::III, DctType::IV][case.kind - 1];
        let st = [DstType::I, DstType::II, DstType::III, DstType::IV][case.kind - 1];
        let raw_scales = [if cosine { 1.0 } else { 0.5 }, 0.5, 1.0, 0.5];
        let a = raw_scales[case.kind - 1];
        let b = raw_scales[[0, 2, 1, 3][case.kind - 1]];
        let (norm, orthogonalize, scale) = match normalization {
            Normalization::Unscaled if case.inverse => ("forward", false, b),
            Normalization::Unscaled => ("backward", false, a),
            Normalization::Forward => ("forward", false, if case.inverse { b } else { 1.0 / b }),
            Normalization::Inverse => ("backward", false, if case.inverse { 1.0 / a } else { a }),
            Normalization::Orthonormal => ("ortho", true, 1.0),
        };
        assert_eq!(case.scipy_norm, norm);
        assert_eq!(case.orthogonalize, orthogonalize);
        assert_eq!(case.scale, scale);
        let input: Vec<_> = case.input.iter().copied().map(scalar::<S>).collect();
        let n = input.len();
        let work = if cosine {
            dct_real_fft_len(ct, n)
        } else {
            dst_real_fft_len(st, n)
        };
        let bound = oracle_bound::<S>(&input, work);
        let direct = if cosine {
            direct_dct(&case.input, ct, normalization, case.inverse)
        } else {
            direct_dst(&case.input, st, normalization, case.inverse)
        };
        let direct: Vec<_> = direct.into_iter().map(scalar::<S>).collect();
        numerics::assert_real_close(&direct, &case.expected, |_| bound);
        let mapped: Vec<_> = case
            .scipy_output
            .iter()
            .map(|&v| scalar::<S>(v * scale))
            .collect();
        numerics::assert_real_close(&mapped, &case.expected, |_| bound);
        let owned = match (cosine, case.inverse) {
            (true, false) => dct(&input, ct, normalization),
            (true, true) => idct(&input, ct, normalization),
            (false, false) => dst(&input, st, normalization),
            (false, true) => idst(&input, st, normalization),
        }
        .unwrap();
        numerics::assert_real_close(&owned, &case.expected, |_| bound);
        let mut planned = vec![S::ZERO; n];
        let cx = scioxide_fft::FftPlanningContext::<S>::default().execution();
        if cosine {
            let plan = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
                n,
                (ct).into(),
                scioxide_fft::FftDirection::Forward,
                normalization,
            ))
            .unwrap();
            let plan_inverse = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
                n,
                (ct).into(),
                scioxide_fft::FftDirection::Inverse,
                normalization,
            ))
            .unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            if case.inverse {
                plan_inverse.execute_into(&input, &mut planned, &mut workspace, cx)
            } else {
                plan.execute_into(&input, &mut planned, &mut workspace, cx)
            }
            .unwrap();
        } else {
            let plan = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
                n,
                (st).into(),
                scioxide_fft::FftDirection::Forward,
                normalization,
            ))
            .unwrap();
            let plan_inverse = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
                n,
                (st).into(),
                scioxide_fft::FftDirection::Inverse,
                normalization,
            ))
            .unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            if case.inverse {
                plan_inverse.execute_into(&input, &mut planned, &mut workspace, cx)
            } else {
                plan.execute_into(&input, &mut planned, &mut workspace, cx)
            }
            .unwrap();
        }
        numerics::assert_real_close(&planned, &case.expected, |_| bound);
    }
}

#[test]
fn scipy_mapping_and_defining_sum_fixtures_f32() {
    assert_scipy_fixtures::<f32>(32);
}

#[test]
fn scipy_mapping_and_defining_sum_fixtures_f64() {
    assert_scipy_fixtures::<f64>(64);
}

#[test]
fn complex_fft_normalization_invariants_f32() {
    assert_complex_fft_invariants::<f32>();
}

#[test]
fn complex_fft_normalization_invariants_f64() {
    assert_complex_fft_invariants::<f64>();
}

#[test]
fn real_fft_normalization_invariants_f32() {
    assert_real_fft_invariants::<f32>();
}

#[test]
fn real_fft_normalization_invariants_f64() {
    assert_real_fft_invariants::<f64>();
}

#[test]
fn dct_normalization_invariants_f32() {
    assert_dct_invariants::<f32>();
}

#[test]
fn dct_normalization_invariants_f64() {
    assert_dct_invariants::<f64>();
}

#[test]
fn dst_normalization_invariants_f32() {
    assert_dst_invariants::<f32>();
}

#[test]
fn dst_normalization_invariants_f64() {
    assert_dst_invariants::<f64>();
}

#[test]
fn orthonormal_endpoint_weight_examples_are_retained() {
    assert_endpoint_weight_examples();
}

#[test]
fn shift_modulation_and_conjugate_symmetry_hold_for_awkward_lengths() {
    let length = 23;
    let shift = 5;
    let modulation = 4;
    let real = (0..length)
        .map(|index| (0.31 * index as f64).sin() - 0.2 * (0.17 * index as f64).cos())
        .collect::<Vec<_>>();
    let input = real
        .iter()
        .copied()
        .map(|value| Complex::new(value, 0.0))
        .collect::<Vec<_>>();
    let plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            length,
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        )
        .with_execution_policy(ExecutionPolicy::Sequential)
        .with_backend_strategy(FftBackendStrategy::Required(
            scioxide_core::backend::KernelBackend::Scalar,
        )),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let mut spectrum = input.clone();
    plan.execute_in_place(
        &mut spectrum,
        &mut plan.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();

    (1..length).for_each(|frequency| {
        let mirror = spectrum[length - frequency].conj();
        assert!((spectrum[frequency] - mirror).norm() <= 2e-12);
    });

    let shifted = (0..length)
        .map(|index| input[(index + length - shift) % length])
        .collect::<Vec<_>>();
    let mut shifted_spectrum = shifted;
    plan.execute_in_place(
        &mut shifted_spectrum,
        &mut plan.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    shifted_spectrum.iter().zip(&spectrum).enumerate().for_each(
        |(frequency, (actual, original))| {
            let angle = -core::f64::consts::TAU * frequency as f64 * shift as f64 / length as f64;
            let expected = *original * Complex::new(angle.cos(), angle.sin());
            assert!((*actual - expected).norm() <= 3e-12);
        },
    );

    let mut modulated = input
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            let angle = core::f64::consts::TAU * modulation as f64 * index as f64 / length as f64;
            value * Complex::new(angle.cos(), angle.sin())
        })
        .collect::<Vec<_>>();
    plan.execute_in_place(
        &mut modulated,
        &mut plan.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    modulated
        .iter()
        .zip((0..length).map(|frequency| spectrum[(frequency + length - modulation) % length]))
        .for_each(|(actual, expected)| assert!((*actual - expected).norm() <= 3e-12));
}
