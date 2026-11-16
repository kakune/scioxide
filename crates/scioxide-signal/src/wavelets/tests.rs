#![allow(clippy::unwrap_used)]
use super::*;
use crate::util::error_kind;
use scioxide_core::{Complex, SciOxideErrorKind};

fn families() -> Vec<WaveletFamily> {
    core::iter::once(WaveletFamily::Haar)
        .chain((4..=20).step_by(2).map(WaveletFamily::Daubechies))
        .chain(
            [103, 105, 202, 204, 206, 208, 301, 303, 305, 307, 309]
                .into_iter()
                .map(WaveletFamily::BiorthogonalSpline),
        )
        .collect()
}
/// Round-trip envelope for unit-scale data: every level's analysis and
/// synthesis each accumulate one dot product of at most `taps` terms, so the
/// error stays within `2 * levels * taps` unit roundoffs per axis pass
/// (observed at most a quarter of this over every family and boundary).
fn round_trip_bound<S: RealScalar>(family: WaveletFamily, levels: usize, axes: usize) -> f64 {
    let filters = WaveletFilters::<f64>::new(family).unwrap();
    let taps = [
        filters.analysis_low(),
        filters.analysis_high(),
        filters.synthesis_low(),
        filters.synthesis_high(),
    ]
    .iter()
    .map(|f| f.len())
    .max()
    .unwrap();
    (2 * axes * levels.max(1) * taps) as f64 * S::EPSILON.into_f64()
}
fn reconstruction<T: NumericScalar>() {
    for family in families() {
        for phase in [WaveletPhase::Centered, WaveletPhase::Uncentered] {
            for boundary in [
                SignalBoundary::Periodic,
                SignalBoundary::Zero,
                SignalBoundary::Reflect,
            ] {
                for n in [0, 1, 2, 3, 5, 8, 17, 32] {
                    let x: Vec<T> = (0..n)
                        .map(|i| {
                            T::from_components(
                                T::Real::try_from_f64_lossy(((i * 7 % 19) as f64 - 9.) / 9.)
                                    .unwrap(),
                                T::Real::try_from_f64_lossy(((i * 3 % 11) as f64 - 5.) / 5.)
                                    .unwrap(),
                            )
                        })
                        .collect();
                    let p = WaveletPlan::new(
                        family,
                        n,
                        maximum_depth(n, family, boundary).unwrap(),
                        WaveletOptions::default()
                            .with_phase(phase)
                            .with_boundary(boundary),
                    )
                    .unwrap();
                    let c = p.analyze(&x).unwrap();
                    let y = p.synthesize(&c).unwrap();
                    let tol = round_trip_bound::<T::Real>(family, p.levels().len(), 1);
                    for (&a, &b) in x.iter().zip(&y) {
                        assert!(
                            (a - b).real_component().abs().into_f64() <= tol
                                && (a - b).imaginary_component().abs().into_f64() <= tol,
                            "{family:?} {phase:?} {boundary:?} {n}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn all_family_reconstruction_f64() {
    reconstruction::<f64>();
}
#[test]
fn all_family_reconstruction_f32() {
    reconstruction::<f32>();
}
#[test]
fn all_family_reconstruction_complex64() {
    reconstruction::<Complex<f64>>();
}
#[test]
fn all_family_reconstruction_complex32() {
    reconstruction::<Complex<f32>>();
}
#[test]
fn matrix_reconstruction() {
    for family in families() {
        for boundary in [
            SignalBoundary::Periodic,
            SignalBoundary::Zero,
            SignalBoundary::Reflect,
        ] {
            for layout in [Wavelet2dLayout::Tensor, Wavelet2dLayout::Pyramid] {
                let shape = [5, 7];
                let x: Vec<_> = (0..35).map(|i| (i as f64 * 0.23).sin()).collect();
                let depth = maximum_depth(5, family, boundary)
                    .unwrap()
                    .min(maximum_depth(7, family, boundary).unwrap());
                let p = Wavelet2dPlan::new(
                    family,
                    shape,
                    [depth; 2],
                    layout,
                    WaveletOptions::default().with_boundary(boundary),
                )
                .unwrap();
                let c = p.analyze(&x).unwrap();
                let y = p.synthesize(&c).unwrap();
                let tol = round_trip_bound::<f64>(family, depth, 2);
                for (&a, b) in x.iter().zip(y) {
                    assert!(
                        (a - b).abs() <= tol,
                        "{family:?} {boundary:?} {layout:?}: {}",
                        (a - b).abs()
                    );
                }
            }
        }
    }
}

#[test]
fn haar_analytical_and_boundary_coefficients() {
    let p = WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 2, WaveletOptions::default()).unwrap();
    let c = p.analyze(&[1., 2., 3., 4.]).unwrap();
    let expected = [
        5.,
        -2.,
        -core::f64::consts::FRAC_1_SQRT_2,
        -core::f64::consts::FRAC_1_SQRT_2,
    ];
    for (a, b) in c.iter().zip(expected) {
        assert!((a - b).abs() < 2e-15);
    }
    // Centered Haar pairs [-1,0], [1,2], [3,4]: explicit endpoint conventions.
    for (boundary, x0, x4) in [
        (SignalBoundary::Zero, 0., 0.),
        (SignalBoundary::Reflect, 2., 3.),
    ] {
        let p = WaveletPlan::<f64>::new(
            WaveletFamily::Haar,
            4,
            1,
            WaveletOptions::default()
                .with_boundary(boundary)
                .with_phase(WaveletPhase::Centered),
        )
        .unwrap();
        let c = p.analyze(&[1., 2., 3., 4.]).unwrap();
        let expected = [x0 + 1., 5., 4. + x4, x0 - 1., -1., 4. - x4];
        for (a, b) in c.iter().zip(expected) {
            assert!((a - b * core::f64::consts::FRAC_1_SQRT_2).abs() < 2e-15);
        }
    }
}
#[test]
fn filter_identities_moments_and_energy() {
    for family in families() {
        let f = WaveletFilters::<f64>::new(family).unwrap();
        let (analysis_moments, synthesis_moments) = match family {
            WaveletFamily::Haar => (1, 1),
            WaveletFamily::Daubechies(n) => (n / 2, n / 2),
            WaveletFamily::BiorthogonalSpline(n) => (n / 100, n % 100),
        };
        for (taps, moments) in [
            (f.analysis_high(), analysis_moments),
            (f.synthesis_high(), synthesis_moments),
        ] {
            for power in 0..moments {
                // Scale the monomials to [-1,1] to avoid raw high-degree magnitude growth.
                let moment: f64 = taps
                    .iter()
                    .enumerate()
                    .map(|(i, &h)| {
                        h * (2. * i as f64 / (taps.len() - 1) as f64 - 1.).powi(power as i32)
                    })
                    .sum();
                assert!(moment.abs() < 3e-14, "{family:?} moment {power}: {moment}");
            }
        }
        for shift in -20_i32..=20 {
            for (a, b, identity) in [
                (f.analysis_low(), f.synthesis_low(), true),
                (f.analysis_high(), f.synthesis_high(), true),
                (f.analysis_low(), f.synthesis_high(), false),
                (f.analysis_high(), f.synthesis_low(), false),
            ] {
                let dot: f64 = a
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &a)| {
                        let j = i as i32 + 2 * shift;
                        (j >= 0)
                            .then(|| b.get(j as usize).map(|&b| a * b))
                            .flatten()
                    })
                    .sum();
                let target = if identity && shift == 0 { 1. } else { 0. };
                assert!(
                    (dot - target).abs() < 3e-14,
                    "{family:?} shift {shift}: {dot}"
                );
            }
        }
        if !matches!(family, WaveletFamily::BiorthogonalSpline(_)) {
            let p =
                WaveletPlan::<Complex<f64>>::new(family, 64, 6, WaveletOptions::default()).unwrap();
            let x: Vec<_> = (0..64)
                .map(|i| Complex::new((i as f64 * 0.2).sin(), (i as f64 * 0.7).cos()))
                .collect();
            let y = p.analyze(&x).unwrap();
            let a: f64 = x.iter().map(|x| x.re * x.re + x.im * x.im).sum();
            let b: f64 = y.iter().map(|x| x.re * x.re + x.im * x.im).sum();
            assert!((a - b).abs() < 3e-12, "{family:?}: {}", a - b);
        }
    }
}
fn gsl_pack(p: &Wavelet2dPlan<f64>, native: &[f64]) -> Vec<f64> {
    let mut packed = vec![0.; native.len()];
    for band in p.subbands() {
        let starts = band
            .detail_levels()
            .map(|l| if l == 0 { 0 } else { 4 >> l });
        let shape = band.shape();
        for r in 0..shape[0] {
            for c in 0..shape[1] {
                packed[band.offset() + r * shape[1] + c] =
                    native[(starts[0] + r) * 4 + starts[1] + c];
            }
        }
    }
    packed
}
#[test]
fn gsl_28_forward_and_independent_inverse_fixtures() {
    let mut cases = 0;
    for line in include_str!("../../tests/fixtures/wavelets.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let mut fields = line.split('\t');
        let order: usize = fields.next().unwrap().parse().unwrap();
        let phase = if fields.next().unwrap() == "0" {
            WaveletPhase::Uncentered
        } else {
            WaveletPhase::Centered
        };
        let geometry: usize = fields.next().unwrap().parse().unwrap();
        let inverse = fields.next().unwrap() == "1";
        let expected: Vec<f64> = fields.map(|x| x.parse().unwrap()).collect();
        let family = if order == 2 {
            WaveletFamily::Haar
        } else if order <= 20 {
            WaveletFamily::Daubechies(order)
        } else {
            WaveletFamily::BiorthogonalSpline(order)
        };
        let options = WaveletOptions::default().with_phase(phase);
        let x: Vec<_> = (0..expected.len())
            .map(|i| (((i * 7 + 3) % 17) as f64 - 8.) / 8.)
            .collect();
        let (actual, expected) = if geometry == 0 {
            let p = WaveletPlan::<f64>::new(family, 8, 3, options).unwrap();
            (
                if inverse {
                    p.synthesize(&x).unwrap()
                } else {
                    p.analyze(&x).unwrap()
                },
                expected,
            )
        } else {
            let p = Wavelet2dPlan::<f64>::new(
                family,
                [4, 4],
                [2, 2],
                if geometry == 1 {
                    Wavelet2dLayout::Tensor
                } else {
                    Wavelet2dLayout::Pyramid
                },
                options,
            )
            .unwrap();
            if inverse {
                (p.synthesize(&gsl_pack(&p, &x)).unwrap(), expected)
            } else {
                (p.analyze(&x).unwrap(), gsl_pack(&p, &expected))
            }
        };
        for (a, b) in actual.iter().zip(expected) {
            assert!(
                (a - b).abs() < 3e-13,
                "{family:?} {phase:?} geometry={geometry} inverse={inverse}: {a} {b}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 252);
}
#[test]
fn invalid_parameters_and_transactional_errors() {
    assert_eq!(
        error_kind(WaveletPlan::<f64>::new(
            WaveletFamily::Daubechies(3),
            8,
            1,
            WaveletOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(WaveletPlan::<f64>::new(
            WaveletFamily::BiorthogonalSpline(999),
            8,
            0,
            WaveletOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(WaveletPlan::<f64>::new(
            WaveletFamily::Haar,
            usize::MAX,
            0,
            WaveletOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(WaveletPlan::<f64>::new(
            WaveletFamily::Haar,
            8,
            4,
            WaveletOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    let p = WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 2, WaveletOptions::default()).unwrap();
    let cx = ExecutionContext::sequential_native();
    let requirements = p.workspace_requirements(1).unwrap();
    // A record needs `max(n, m) + 2·width = 12` values; each lane of an axis
    // form holds its samples, its coefficients and the level scratch, 16.
    let lanes = |k| p.workspace_requirements(k).unwrap().values_len();
    assert_eq!([lanes(0), lanes(1), lanes(3)], [12, 16, 48]);
    let mut ws = p.workspace(1).unwrap();
    assert_eq!(
        ws.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    let mut out = vec![7.; 4];
    assert_eq!(
        error_kind(p.analyze_into(&[0.; 3], &mut out, &mut ws, cx)),
        SciOxideErrorKind::DimensionMismatch
    );
    let mut small = SignalWorkspace::new();
    small
        .try_reserve(&SignalRequirements::values(p.input_len()))
        .unwrap();
    assert_eq!(
        small.check(&requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    assert_eq!(
        p.analyze_into(&[1.; 4], &mut out, &mut small, cx)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        ws.check(&requirements.with_max_bytes(8))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        error_kind(p.analyze_into(&[f64::NAN; 4], &mut out, &mut ws, cx)),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        error_kind(p.analyze_into(&[f64::MAX; 4], &mut out, &mut ws, cx)),
        SciOxideErrorKind::NumericalInstability
    );
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        error_kind(p.analyze_into(
            &[1.; 4],
            &mut out,
            &mut ws,
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(
        error_kind(p.synthesize_into(
            &[1.; 4],
            &mut out,
            &mut ws,
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, vec![7.; 4]);
}
#[test]
fn owned_forms_equal_native_borrowed_forms() {
    let cx = ExecutionContext::sequential_native();
    let p = WaveletPlan::<f64>::new(
        WaveletFamily::Daubechies(6),
        33,
        2,
        WaveletOptions::default().with_boundary(SignalBoundary::Reflect),
    )
    .unwrap();
    let x: Vec<_> = (0..33).map(|i| (i as f64 * 0.37).sin()).collect();
    let mut ws = p.workspace(1).unwrap();
    let mut coefficients = vec![0.; p.coefficient_len()];
    p.analyze_into(&x, &mut coefficients, &mut ws, cx).unwrap();
    assert_eq!(p.analyze(&x).unwrap(), coefficients);
    let mut restored = vec![0.; 33];
    p.synthesize_into(&coefficients, &mut restored, &mut ws, cx)
        .unwrap();
    assert_eq!(p.synthesize(&coefficients).unwrap(), restored);
    let matrix = Wavelet2dPlan::<f64>::new(
        WaveletFamily::Haar,
        [4, 6],
        [1, 2],
        Wavelet2dLayout::Tensor,
        WaveletOptions::default(),
    )
    .unwrap();
    let x: Vec<_> = (0..24).map(|i| (i as f64 * 0.11).cos()).collect();
    let mut ws = matrix.workspace(()).unwrap();
    let mut coefficients = vec![0.; matrix.coefficient_len()];
    matrix
        .analyze_into(&x, &mut coefficients, &mut ws, cx)
        .unwrap();
    assert_eq!(matrix.analyze(&x).unwrap(), coefficients);
    let mut restored = vec![0.; 24];
    matrix
        .synthesize_into(&coefficients, &mut restored, &mut ws, cx)
        .unwrap();
    assert_eq!(matrix.synthesize(&coefficients).unwrap(), restored);
}

use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{
    ExecutionPlan, Executor, NdView, NdViewMut, PartitionFn, SequentialExecutor, StridedLayout,
};
struct Counting {
    calls: AtomicUsize,
    fail: bool,
}
impl Executor for Counting {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        c: ExecutionContext<'_>,
        p: ExecutionPlan,
        o: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail {
            return Err(invalid(
                scioxide_core::argument!("context"),
                "injected wavelet executor failure",
            ));
        }
        SequentialExecutor.execute(c, p, o)
    }
}

fn matrix_scalar<T: NumericScalar>() {
    for family in [
        WaveletFamily::Haar,
        WaveletFamily::Daubechies(20),
        WaveletFamily::BiorthogonalSpline(309),
    ] {
        for boundary in [
            SignalBoundary::Periodic,
            SignalBoundary::Zero,
            SignalBoundary::Reflect,
        ] {
            for phase in [WaveletPhase::Centered, WaveletPhase::Uncentered] {
                for layout in [Wavelet2dLayout::Tensor, Wavelet2dLayout::Pyramid] {
                    let x: Vec<_> = (0..35)
                        .map(|i| {
                            T::from_components(
                                T::Real::try_from_f64_lossy(i as f64 / 35.).unwrap(),
                                T::Real::try_from_f64_lossy((i % 7) as f64 / 7.).unwrap(),
                            )
                        })
                        .collect();
                    let depth = maximum_depth(5, family, boundary)
                        .unwrap()
                        .min(maximum_depth(7, family, boundary).unwrap());
                    let p = Wavelet2dPlan::<T>::new(
                        family,
                        [5, 7],
                        [depth; 2],
                        layout,
                        WaveletOptions::default()
                            .with_boundary(boundary)
                            .with_phase(phase),
                    )
                    .unwrap();
                    let y = p.synthesize(&p.analyze(&x).unwrap()).unwrap();
                    let tol = round_trip_bound::<T::Real>(family, depth, 2);
                    for (a, b) in x.iter().zip(y) {
                        assert!((*a - b).real_component().abs().into_f64() <= tol);
                        assert!((*a - b).imaginary_component().abs().into_f64() <= tol);
                    }
                }
            }
        }
    }
}
#[test]
fn matrix_f32_and_complex_domains() {
    matrix_scalar::<f32>();
    matrix_scalar::<Complex<f32>>();
    matrix_scalar::<Complex<f64>>();
}
#[test]
fn empty_identity_and_matrix_length_errors() {
    let cx = ExecutionContext::sequential_native();
    let empty =
        WaveletPlan::<f64>::new(WaveletFamily::Haar, 0, 0, WaveletOptions::default()).unwrap();
    let mut ws = empty.workspace(1).unwrap();
    empty.analyze_into(&[], &mut [], &mut ws, cx).unwrap();
    empty.synthesize_into(&[], &mut [], &mut ws, cx).unwrap();
    let p = Wavelet2dPlan::<f64>::new(
        WaveletFamily::Haar,
        [2, 2],
        [1, 1],
        Wavelet2dLayout::Pyramid,
        WaveletOptions::default(),
    )
    .unwrap();
    let mut ws = p.workspace(()).unwrap();
    let mut out = [99.; 4];
    assert_eq!(
        error_kind(p.analyze_into(&[2.; 3], &mut out, &mut ws, cx)),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        error_kind(p.analyze_into(&[f64::NAN; 4], &mut out, &mut ws, cx)),
        SciOxideErrorKind::NonFinite
    );
    let mut small = SignalWorkspace::new();
    assert_eq!(
        p.synthesize_into(&[2.; 4], &mut out, &mut small, cx)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [99.; 4]);
}

#[test]
fn arbitrary_axis_broadcast_empty_batches_and_partial_depths() {
    for depth in 0..=3 {
        let p = WaveletPlan::<f64>::new(
            WaveletFamily::Daubechies(8),
            7,
            depth,
            WaveletOptions::default(),
        )
        .unwrap();
        let m = p.coefficient_len();
        // Broadcast one seven-sample signal over both batch axes.
        let input = [1., -2., 3., -4., 5., -6., 7.];
        let mut coeff = vec![0.; 2 * m * 3];
        let mut output = vec![0.; 42];
        let mut work = p.workspace(6).unwrap();
        p.analyze_axis_into(
            NdView::new(
                &input,
                StridedLayout::new_broadcast(&[2, 7, 3], &[0, 1, 0]).unwrap(),
            )
            .unwrap(),
            1,
            NdViewMut::new(
                &mut coeff,
                StridedLayout::new(&[2, m, 3], &[m * 3, 3, 1]).unwrap(),
            )
            .unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
        p.synthesize_axis_into(
            NdView::new(
                &coeff,
                StridedLayout::new(&[2, m, 3], &[m * 3, 3, 1]).unwrap(),
            )
            .unwrap(),
            1,
            NdViewMut::new(
                &mut output,
                StridedLayout::new(&[2, 7, 3], &[21, 3, 1]).unwrap(),
            )
            .unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (i, &x) in output.iter().enumerate() {
            assert!((x - input[(i / 3) % 7]).abs() < 1e-13);
        }
        // An empty batch has no lanes and needs no workspace.
        p.analyze_axis_into(
            NdView::new(&[], StridedLayout::new(&[0, 7, 3], &[21, 3, 1]).unwrap()).unwrap(),
            1,
            NdViewMut::new(
                &mut [],
                StridedLayout::new(&[0, m, 3], &[m * 3, 3, 1]).unwrap(),
            )
            .unwrap(),
            &mut SignalWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
}
#[test]
fn matrix_executor_failure_after_first_pass_is_atomic() {
    struct FailSecond(AtomicUsize);
    impl Executor for FailSecond {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            c: ExecutionContext<'_>,
            p: ExecutionPlan,
            o: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
                Err(invalid(
                    scioxide_core::argument!("context"),
                    "wavelet second pass failure",
                ))
            } else {
                SequentialExecutor.execute(c, p, o)
            }
        }
    }
    for inverse in [false, true] {
        let ex = FailSecond(AtomicUsize::new(0));
        let p = Wavelet2dPlan::<f64>::new(
            WaveletFamily::Haar,
            [4, 4],
            [2, 2],
            Wavelet2dLayout::Pyramid,
            WaveletOptions::default()
                .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())),
        )
        .unwrap();
        let mut work = p.workspace(()).unwrap();
        let mut out = [99.; 16];
        let cx = ExecutionContext::new(&ex);
        let result = if inverse {
            p.synthesize_into(&[1.; 16], &mut out, &mut work, cx)
        } else {
            p.analyze_into(&[1.; 16], &mut out, &mut work, cx)
        };
        assert_eq!(error_kind(result), SciOxideErrorKind::InvalidInput);
        assert_eq!(ex.0.load(Ordering::Relaxed), 2);
        assert_eq!(out, [99.; 16]);
    }
}

#[test]
fn dense_column_overflow_preserves_matrix_output() {
    let plan = Wavelet2dPlan::<f64>::new(
        WaveletFamily::Daubechies(4),
        [8, 8],
        [1, 1],
        Wavelet2dLayout::Pyramid,
        WaveletOptions::default(),
    )
    .unwrap();
    // Each horizontal lowpass is finite; vertical lowpasses exceed MAX.
    let input = vec![0.6 * f64::MAX; 64];
    let mut output = vec![17.; plan.coefficient_len()];
    let mut work = plan.workspace(()).unwrap();
    assert_eq!(
        error_kind(plan.analyze_into(
            &input,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert!(output.iter().all(|&value| value == 17.));
}

/// Separable single-stage 2-D analysis from independent 1-D row and column
/// plans: rows of `x` (`shape[0]` by `shape[1]`), then the resulting columns.
fn separable_stage(
    rows: &WaveletPlan<f64>,
    columns: &WaveletPlan<f64>,
    x: &[f64],
    shape: [usize; 2],
) -> Vec<f64> {
    let m = rows.coefficient_len();
    let mut horizontal = vec![0.; shape[0] * m];
    for (input, output) in x.chunks_exact(shape[1]).zip(horizontal.chunks_exact_mut(m)) {
        output.copy_from_slice(&rows.analyze(input).unwrap());
    }
    let mut result = vec![0.; columns.coefficient_len() * m];
    for column in 0..m {
        let lane: Vec<_> = (0..shape[0]).map(|r| horizontal[r * m + column]).collect();
        for (r, value) in columns.analyze(&lane).unwrap().into_iter().enumerate() {
            result[r * m + column] = value;
        }
    }
    result
}
/// Packed `(offset, len)` of the band with this detail level (0 is lowpass).
fn band(plan: &WaveletPlan<f64>, level: usize) -> (usize, usize) {
    if level == 0 {
        (0, plan.approximation_len())
    } else {
        let l = plan.levels()[level - 1];
        (l.detail_offset(), l.band_len())
    }
}
/// Pack the subbands of `plan` from separable 1-D stages.
fn separable_matrix(
    family: WaveletFamily,
    shape: [usize; 2],
    levels: [usize; 2],
    layout: Wavelet2dLayout,
    options: WaveletOptions,
    plan: &Wavelet2dPlan<f64>,
    x: &[f64],
) -> Vec<f64> {
    // Stage s holds (row plan, column plan, packed stage coefficients).
    let mut stages = Vec::new();
    if layout == Wavelet2dLayout::Tensor {
        let rows = WaveletPlan::<f64>::new(family, shape[1], levels[1], options).unwrap();
        let columns = WaveletPlan::<f64>::new(family, shape[0], levels[0], options).unwrap();
        let c = separable_stage(&rows, &columns, x, shape);
        stages.push((rows, columns, c));
    } else {
        let (mut current, mut block) = (shape, x.to_vec());
        for _ in 0..levels[0] {
            let rows = WaveletPlan::<f64>::new(family, current[1], 1, options).unwrap();
            let columns = WaveletPlan::<f64>::new(family, current[0], 1, options).unwrap();
            let c = separable_stage(&rows, &columns, &block, current);
            let next = [columns.approximation_len(), rows.approximation_len()];
            block = (0..next[0])
                .flat_map(|r| c[r * rows.coefficient_len()..][..next[1]].to_vec())
                .collect();
            stages.push((rows, columns, c));
            current = next;
        }
        if stages.is_empty() {
            return x.to_vec();
        }
    }
    let mut packed = vec![f64::NAN; plan.coefficient_len()];
    for subband in plan.subbands() {
        let [lr, lc] = subband.detail_levels();
        let (stage, (lr, lc)) = if layout == Wavelet2dLayout::Tensor {
            (0, (lr, lc))
        } else if lr == 0 && lc == 0 {
            (stages.len() - 1, (0, 0))
        } else {
            let level = lr.max(lc);
            (level - 1, (lr.min(1), lc.min(1)))
        };
        let (rows, columns, c) = &stages[stage];
        let (r0, nr) = band(columns, lr);
        let (c0, nc) = band(rows, lc);
        assert_eq!(subband.shape(), [nr, nc]);
        for r in 0..nr {
            for k in 0..nc {
                packed[subband.offset() + r * nc + k] =
                    c[(r0 + r) * rows.coefficient_len() + c0 + k];
            }
        }
    }
    packed
}
#[test]
fn matrix_analysis_is_bitwise_policy_independent_and_separable() {
    let executor = Counting {
        calls: AtomicUsize::new(0),
        fail: false,
    };
    let parallel = ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap());
    let mut checked = 0;
    // Short periodic extents wrap analysis windows around the record end.
    for family in [
        WaveletFamily::Haar,
        WaveletFamily::Daubechies(4),
        WaveletFamily::Daubechies(6),
        WaveletFamily::BiorthogonalSpline(103),
    ] {
        for boundary in [SignalBoundary::Periodic, SignalBoundary::Zero] {
            for phase in [WaveletPhase::Uncentered, WaveletPhase::Centered] {
                for shape in [[2, 2], [2, 6], [4, 4], [6, 4], [8, 10], [16, 6], [5, 6]] {
                    let options = WaveletOptions::default()
                        .with_boundary(boundary)
                        .with_phase(phase);
                    let x: Vec<_> = (0..shape[0] * shape[1])
                        .map(|i| (i as f64 * 0.37).sin() * 1.3 - (i as f64 * 0.11).cos() / 3.)
                        .collect();
                    for levels in (0..4).flat_map(|a| (0..4).map(move |b| [a, b])) {
                        for layout in [Wavelet2dLayout::Tensor, Wavelet2dLayout::Pyramid] {
                            let Ok(sequential) =
                                Wavelet2dPlan::<f64>::new(family, shape, levels, layout, options)
                            else {
                                continue;
                            };
                            let threaded = Wavelet2dPlan::<f64>::new(
                                family,
                                shape,
                                levels,
                                layout,
                                options.with_execution_policy(parallel),
                            )
                            .unwrap();
                            let expected = sequential.analyze(&x).unwrap();
                            let mut actual = vec![0.; expected.len()];
                            let mut ws = threaded.workspace(()).unwrap();
                            threaded
                                .analyze_into(
                                    &x,
                                    &mut actual,
                                    &mut ws,
                                    ExecutionContext::new(&executor),
                                )
                                .unwrap();
                            let bits =
                                |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                            let label = alloc::format!(
                                "{family:?} {boundary:?} {phase:?} {shape:?} {levels:?} {layout:?}"
                            );
                            assert_eq!(bits(&actual), bits(&expected), "{label}");
                            let separable = separable_matrix(
                                family,
                                shape,
                                levels,
                                layout,
                                options,
                                &sequential,
                                &x,
                            );
                            assert_eq!(bits(&separable), bits(&expected), "{label}");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 200 && executor.calls.load(Ordering::Relaxed) > 0);
}
