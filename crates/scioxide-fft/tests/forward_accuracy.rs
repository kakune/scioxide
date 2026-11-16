//! Independent 100/160-digit DFT references, not round trips or shared twiddles.
#![expect(clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;

use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_fft::{
    FftAlgorithmKind, FftBackendStrategy, FftDirection, FftLayout, FftOutput, FftPlan, FftPlanSpec,
    FftPlanningContext, Normalization,
};
use scioxide_linalg::{VectorView, VectorViewMut};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    length: usize,
    real: bool,
    expected: Vec<[f64; 2]>,
}

fn signal<S: RealScalar>(n: usize, real: bool) -> Vec<Complex<S>> {
    // Matches the generator; each component is an exact 24-bit dyadic number.
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut component = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        S::try_from_f64_lossy(((state >> 40) as f64 - 8_388_608.0) / 8_388_608.0).unwrap()
    };
    (0..n)
        .map(|_| Complex::new(component(), if real { S::ZERO } else { component() }))
        .collect()
}

fn check_references<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/forward.json")).unwrap();
    for length in [2401, 3430, 2039, 4099, 10007, 65537] {
        for real in [false, true] {
            assert!(
                fixtures
                    .cases
                    .iter()
                    .any(|case| case.length == length && case.real == real),
                "missing forward fixture: n={length}, real={real}"
            );
        }
    }
    assert!(
        fixtures
            .cases
            .iter()
            .any(|case| case.length == 3375 && case.real),
        "missing odd-smooth packed-real fixture"
    );
    let context = FftPlanningContext::<S>::default();
    let epsilon = S::EPSILON.try_to_f64_exact().unwrap();
    // A native ARM run must execute NEON, not silently reduce to scalar coverage.
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    assert!(
        numerics::fft_backends(context.execution().processor())
            .any(|b| b == scioxide_core::backend::KernelBackend::Neon)
    );
    for case in fixtures.cases {
        let n = case.length;
        let input = signal::<S>(n, case.real);
        let real_input: Vec<_> = input.iter().map(|z| z.re).collect();
        let layout = if case.real {
            FftLayout::Real
        } else {
            FftLayout::Complex
        };
        for backend in numerics::fft_backends(context.execution().processor()) {
            let plan = FftPlan::<S>::new(
                FftPlanSpec::new(n, FftDirection::Forward, Normalization::Unscaled, layout)
                    .with_backend_strategy(FftBackendStrategy::Required(backend)),
            )
            .unwrap();
            let child_len = if case.real && n.is_multiple_of(2) {
                n / 2
            } else {
                n
            };
            assert_eq!(
                plan.diagnostics().algorithm(),
                match child_len {
                    1009 | 65537 => FftAlgorithmKind::Rader,
                    // Non-smooth n-1 deliberately uses a smooth Bluestein
                    // convolution, never a nested Rader/Bluestein plan.
                    1019 | 2039 | 4099 | 10007 => FftAlgorithmKind::Bluestein,
                    _ => FftAlgorithmKind::MixedRadix,
                }
            );
            assert_eq!(case.expected.len(), if case.real { n / 2 + 1 } else { n });
            assert_eq!(
                plan.resolved_backends().next(),
                Some(scioxide_fft::FftBackend::Cpu(backend))
            );
            let mut output = vec![Complex::new(S::ZERO, S::ZERO); case.expected.len()];
            let mut workspace = plan.workspace(()).unwrap();
            if case.real {
                plan.execute_into(
                    real_input.as_slice(),
                    FftOutput::Hermitian {
                        logical_len: n,
                        bins: VectorViewMut::from_slice(&mut output),
                    },
                    &mut workspace,
                    context.execution(),
                )
                .unwrap();
            } else {
                plan.execute_into(
                    VectorView::from_slice(&input),
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    context.execution(),
                )
                .unwrap();
            }
            let (error, norm) = output.iter().zip(&case.expected).enumerate().fold(
                (0.0, 0.0),
                |(error, norm), (k, (actual, expected))| {
                    // Count the negative-frequency partners of packed interior bins.
                    let weight = if case.real && k != 0 && !(n.is_multiple_of(2) && k == n / 2) {
                        2.0
                    } else {
                        1.0
                    };
                    let re = actual.re.try_to_f64_exact().unwrap() - expected[0];
                    let im = actual.im.try_to_f64_exact().unwrap() - expected[1];
                    (
                        error + weight * (re * re + im * im),
                        norm + weight * (expected[0] * expected[0] + expected[1] * expected[1]),
                    )
                },
            );
            let relative = (error / norm).sqrt();
            // A regression envelope, independent of implementation diagnostics.
            // Binary64 fixture rounding contributes at most epsilon/2 in L2;
            // f32 is compared with the same higher-precision rounded reference.
            let bound = 2.0 * epsilon * (n as f64).log2().max(1.0);
            numerics::assert_close(
                relative,
                bound,
                format_args!("{layout:?} {backend:?} n={n}"),
            );
        }
    }
}

#[test]
fn independent_forward_f32() {
    check_references::<f32>();
}

#[test]
fn independent_forward_f64() {
    check_references::<f64>();
}

fn check_tone<S: RealScalar>() {
    let context = FftPlanningContext::<S>::default();
    let epsilon = S::EPSILON.try_to_f64_exact().unwrap();
    for n in [1024, 4096, 4200, 6720, 16_384, 65_536] {
        for backend in numerics::fft_backends(context.execution().processor()) {
            let plan = FftPlan::<S>::new(
                FftPlanSpec::new(
                    n,
                    FftDirection::Forward,
                    Normalization::Unscaled,
                    FftLayout::Complex,
                )
                .with_backend_strategy(FftBackendStrategy::Required(backend)),
            )
            .unwrap();
            let mut output = vec![Complex::new(S::ZERO, S::ZERO); n];
            output[1].re = S::ONE;
            plan.execute_in_place(
                &mut output,
                &mut plan.workspace(()).unwrap(),
                context.execution(),
            )
            .unwrap();
            let worst = output
                .iter()
                .enumerate()
                .map(|(k, z)| {
                    let centered = if k > n / 2 {
                        k as i64 - n as i64
                    } else {
                        k as i64
                    };
                    let angle = -core::f64::consts::TAU * (centered as f64 / n as f64);
                    (z.re.try_to_f64_exact().unwrap() - angle.cos())
                        .abs()
                        .max((z.im.try_to_f64_exact().unwrap() - angle.sin()).abs())
                })
                .fold(0.0, f64::max);
            numerics::assert_close(worst, 6.0 * epsilon, format_args!("tone {backend:?} n={n}"));
        }
    }
}

#[test]
fn single_tone_has_length_independent_component_error() {
    check_tone::<f32>();
    check_tone::<f64>();
}
