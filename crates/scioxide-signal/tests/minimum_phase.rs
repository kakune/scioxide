#![allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::{error, error_kind};

use scioxide_core::{
    CancellationToken, Capacity, Complex, ExecutionContext, ExecutionPolicy, NumericScalar,
    NumericalIssue, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_signal::*;

fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn options<S: RealScalar>(convention: MinimumPhaseConvention) -> MinimumPhaseOptions<S> {
    MinimumPhaseOptions::new()
        .with_convention(convention)
        .with_fft_length(Some(1024))
}
fn reconstruct<T: NumericScalar>(
    plan: &MinimumPhasePlan<T>,
    input: &[T],
    output: &mut [T],
    workspace: &mut SignalWorkspace<T>,
) -> Result<MinimumPhaseDiagnostics<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    plan.reconstruct_into(
        input,
        output,
        workspace,
        ExecutionContext::sequential_native(),
    )
}
// Independent trigonometric sums, including the full circle for complex FIRs.
fn response<T: NumericScalar>(h: &[T], w: f64) -> (f64, f64) {
    h.iter().enumerate().fold((0., 0.), |(re, im), (k, z)| {
        let (s, c) = (-w * k as f64).sin_cos();
        let a = z.real_component().into_f64();
        let b = z.imaginary_component().into_f64();
        (re + a * c - b * s, im + a * s + b * c)
    })
}
fn magnitude<T: NumericScalar>(h: &[T], w: f64) -> f64 {
    let (re, im) = response(h, w);
    re.hypot(im)
}
fn exact_real<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let tol = if core::mem::size_of::<S>() == 4 {
        8e-6
    } else {
        5e-12
    };
    // Root reflection: .5 - z^-1 has root 2, its minimum-phase partner has root .5.
    let (q, d) = minimum_phase_with_options(
        &[r::<S>(0.5), r(-1.)],
        options(MinimumPhaseConvention::SameLength),
    )
    .unwrap();
    assert!((q[0].into_f64() - 1.).abs() < tol);
    assert!((q[1].into_f64() + 0.5).abs() < tol);
    assert!((-q[1] / q[0]).abs() < S::ONE);
    assert!(d.magnitude_error().into_f64() < tol);
    assert_eq!(d.regularized_bins(), 0);
    // Exact autocorrelation of 1 - .5 z^-1, shifted into causal coefficient order.
    let h = [r::<S>(-0.5), r(1.25), r(-0.5)];
    for (mode, expected) in [
        (MinimumPhaseConvention::HalfLength, vec![1., -0.5]),
        (MinimumPhaseConvention::SameLength, vec![1., -1., 0.25]),
    ] {
        let (out, d) = minimum_phase_with_options(&h, options(mode)).unwrap();
        assert_eq!(out.len(), expected.len());
        for (&a, &b) in out.iter().zip(&expected) {
            assert!((a.into_f64() - b).abs() < tol);
        }
        if mode == MinimumPhaseConvention::SameLength {
            // Independent second-order Schur/Jury criterion for both zeros inside |z|<1.
            assert!(out[2].abs() < out[0]);
            assert!(out[0] + out[1] + out[2] > S::ZERO);
            assert!(out[0] - out[1] + out[2] > S::ZERO);
        }
        for k in 0..257 {
            let w = core::f64::consts::TAU * k as f64 / 257.;
            let target = magnitude(&h, w);
            let target = if mode == MinimumPhaseConvention::HalfLength {
                target.sqrt()
            } else {
                target
            };
            assert!((magnitude(&out, w) - target).abs() < tol * 2.);
        }
        assert!(d.discarded_energy_fraction().into_f64() < tol * tol);
    }
    // Hilbert's unity normalization is exact for this nonnegative unity-peak power spectrum.
    let hilbert =
        options(MinimumPhaseConvention::HalfLength).with_method(MinimumPhaseMethod::Hilbert);
    let (out, d) =
        minimum_phase_with_options(&[r::<S>(-0.1875), r(0.625), r(-0.1875)], hilbert).unwrap();
    assert!((out[0].into_f64() - 0.75).abs() < tol);
    assert!((out[1].into_f64() + 0.25).abs() < tol);
    assert!((-out[1] / out[0]).abs() < S::ONE);
    assert_eq!(d.hilbert_lift(), S::ZERO);
    assert!((d.hilbert_normalization().into_f64() - 1.).abs() < tol);
}
#[test]
fn exact_real_both_widths_and_methods() {
    exact_real::<f32>();
    exact_real::<f64>();
}
fn exact_complex<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let tol = if core::mem::size_of::<S>() == 4 {
        1e-5
    } else {
        5e-12
    };
    let h = [
        Complex::new(r::<S>(0.5), S::ZERO),
        Complex::new(S::ZERO, r(-1.)),
    ];
    let (out, d) =
        minimum_phase_with_options(&h, options(MinimumPhaseConvention::SameLength)).unwrap();
    assert!((out[0].re.into_f64() - 1.).abs() < tol);
    assert!(out[0].im.into_f64().abs() < tol);
    assert!(out[1].re.into_f64().abs() < tol);
    assert!((out[1].im.into_f64() + 0.5).abs() < tol);
    assert!(out[1].norm() / out[0].norm() < S::ONE);
    for k in 0..257 {
        let w = core::f64::consts::TAU * k as f64 / 257.;
        assert!((magnitude(&out, w) - magnitude(&h, w)).abs() < tol);
    }
    assert!(d.magnitude_error().into_f64() < tol);
    // Complex modulated autocorrelation has exact complex half-length factor.
    let h = [
        Complex::new(S::ZERO, r::<S>(0.5)),
        Complex::new(r(1.25), S::ZERO),
        Complex::new(S::ZERO, r(-0.5)),
    ];
    let (out, _) =
        minimum_phase_with_options(&h, options(MinimumPhaseConvention::HalfLength)).unwrap();
    assert!((out[0].re.into_f64() - 1.).abs() < tol);
    assert!((out[1].im.into_f64() + 0.5).abs() < tol);
    for k in 0..257 {
        let w = core::f64::consts::TAU * k as f64 / 257.;
        assert!((magnitude(&out, w).powi(2) - magnitude(&h, w)).abs() < tol * 3.);
    }
}
#[test]
fn exact_complex_both_widths_and_conventions() {
    exact_complex::<f32>();
    exact_complex::<f64>();
}
#[test]
fn gain_delay_short_inputs_and_odd_meshes() {
    for n in [1, 3, 17, 128] {
        let opt = MinimumPhaseOptions::new()
            .with_fft_length(Some(n))
            .with_convention(MinimumPhaseConvention::SameLength);
        let (out, _) = minimum_phase_with_options(&[-4.0_f64], opt).unwrap();
        assert!((out[0] - 4.).abs() < 1e-12);
    }
    let (out, _) = minimum_phase_with_options(
        &[0., 1., -0.5, 0.],
        options(MinimumPhaseConvention::SameLength),
    )
    .unwrap();
    out.iter()
        .zip([1., -0.5, 0., 0.])
        .for_each(|(a, b)| assert!((a - b).abs() < 1e-12));
    for gain in [1e-300_f64, 1e300] {
        for mode in [
            MinimumPhaseConvention::SameLength,
            MinimumPhaseConvention::HalfLength,
        ] {
            let h = [-0.5 * gain, 1.25 * gain, -0.5 * gain];
            let (out, _) = minimum_phase_with_options(&h, options(mode)).unwrap();
            let scale = if mode == MinimumPhaseConvention::SameLength {
                gain
            } else {
                gain.sqrt()
            };
            assert!((out[0] / scale - 1.).abs() < 2e-12);
        }
    }
    for mesh in [127, 128, 129] {
        let opt = options(MinimumPhaseConvention::SameLength).with_fft_length(Some(mesh));
        let (out, _) = minimum_phase_with_options(&[0.5_f64, -1.], opt).unwrap();
        assert!((out[0] - 1.).abs() < 1e-12);
        assert!((out[1] + 0.5).abs() < 1e-12);
    }
}
#[test]
fn floor_diagnostics_and_grid_refinement() {
    let opt = options(MinimumPhaseConvention::SameLength).with_spectral_zeros(
        SpectralZeroPolicy::Regularize {
            relative_floor: 1e-3,
        },
    );
    for h in [[1.0, -1.], [1., -0.999999]] {
        let (out, d) = minimum_phase_with_options(&h, opt).unwrap();
        assert!(d.regularized_bins() > 0);
        assert!(d.spectral_ratio() < 1e-3);
        assert_eq!(d.relative_floor(), 1e-3);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(d.magnitude_error() < 0.03);
        let strict = opt.with_spectral_zeros(SpectralZeroPolicy::Reject {
            relative_threshold: 1e-3,
        });
        assert_eq!(
            error_kind(minimum_phase_with_options(&h, strict)),
            SciOxideErrorKind::InvalidInput
        );
    }
    let error = |n| {
        let (out, d) = minimum_phase_with_options(
            &[0.95_f64, -1.],
            options(MinimumPhaseConvention::SameLength).with_fft_length(Some(n)),
        )
        .unwrap();
        (
            (out[0] - 1.).abs() + (out[1] + 0.95).abs(),
            d.magnitude_error(),
        )
    };
    let small = error(32);
    let large = error(512);
    assert!(large.0 < small.0 * 0.01);
    assert!(large.1 < small.1 * 0.01);
}
#[test]
fn invalid_inputs_are_transactional() {
    let p = MinimumPhasePlan::<f64>::new(3, options(MinimumPhaseConvention::SameLength)).unwrap();
    let mut workspace = p.workspace(()).unwrap();
    let mut out = [42.; 3];
    for (x, kind) in [
        (&[0., 0., 0.][..], SciOxideErrorKind::DomainError),
        (&[1., f64::NAN, 1.], SciOxideErrorKind::NonFinite),
        (&[1., f64::INFINITY, 1.], SciOxideErrorKind::NonFinite),
        (&[1., 2.], SciOxideErrorKind::DimensionMismatch),
    ] {
        assert_eq!(
            error_kind(reconstruct(&p, x, &mut out, &mut workspace)),
            kind
        );
        assert_eq!(out, [42.; 3]);
    }
    // Output must have exactly `output_len()` coefficients.
    for len in [2, 4] {
        let mut wrong = vec![42.; len];
        assert_eq!(
            reconstruct(&p, &[1., 2., 1.], &mut wrong, &mut workspace)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        assert!(wrong.iter().all(|&v| v == 42.));
    }
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::new(0, Default::default())),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::new(usize::MAX, Default::default())),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::new(
            3,
            MinimumPhaseOptions::new().with_fft_length(Some(2))
        )),
        SciOxideErrorKind::InvalidInput
    );
    for (floor, kind) in [
        (0., SciOxideErrorKind::InvalidInput),
        (-1., SciOxideErrorKind::InvalidInput),
        (1., SciOxideErrorKind::InvalidInput),
        (f64::NAN, SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(
            error_kind(MinimumPhasePlan::<f64>::new(
                3,
                MinimumPhaseOptions::new().with_spectral_zeros(SpectralZeroPolicy::Regularize {
                    relative_floor: floor
                })
            )),
            kind
        );
    }
    let hilbert = MinimumPhaseOptions::new().with_method(MinimumPhaseMethod::Hilbert);
    assert_eq!(
        error_kind(MinimumPhasePlan::<Complex<f64>>::new(3, hilbert)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::new(2, hilbert)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::new(
            3,
            hilbert.with_convention(MinimumPhaseConvention::SameLength)
        )),
        SciOxideErrorKind::InvalidInput
    );
    let p = MinimumPhasePlan::<f64>::new(3, hilbert).unwrap();
    let mut workspace = p.workspace(()).unwrap();
    let mut half = [42.; 2];
    assert_eq!(
        error_kind(reconstruct(&p, &[0.1, 0.8, 0.2], &mut half, &mut workspace)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(half, [42.; 2]);
    assert_eq!(
        error_kind(reconstruct(&p, &[0.1, 4., 0.1], &mut half, &mut workspace)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(half, [42.; 2]);
    // A valid finite input may require unrepresentable output gain.
    let p = MinimumPhasePlan::<f64>::new(3, options(MinimumPhaseConvention::SameLength)).unwrap();
    let mut workspace = p.workspace(()).unwrap();
    assert_eq!(
        error_kind(reconstruct(
            &p,
            &[0.5 * f64::MAX, f64::MAX, -f64::MAX],
            &mut out,
            &mut workspace
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(out, [42.; 3]);
}
#[test]
fn workspace_protocol_reserves_checks_and_rejects_undersized_storage() {
    let p = MinimumPhasePlan::<f64>::new(2, options(MinimumPhaseConvention::SameLength)).unwrap();
    let requirements = p.workspace_requirements(()).unwrap();
    // Mesh, target and verification spectra plus the FFT scratch.
    assert!(requirements.complex_len() >= 3 * p.fft_length());
    assert_eq!((requirements.values_len(), requirements.real_len()), (0, 0));
    let mut workspace = SignalWorkspace::<f64>::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let mut out = [42.; 2];
    assert_eq!(
        reconstruct(&p, &[0.5, -1.], &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [42.; 2]);
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let first = reconstruct(&p, &[0.5, -1.], &mut out, &mut workspace).unwrap();
    assert!((out[0] - 1.).abs() < 1e-12 && (out[1] + 0.5).abs() < 1e-12);
    assert_eq!(first.fft_length(), p.fft_length());
    // A workspace reserved for a larger plan serves a smaller one.
    let small = MinimumPhasePlan::<f64>::new(
        2,
        options(MinimumPhaseConvention::SameLength).with_fft_length(Some(64)),
    )
    .unwrap();
    assert_eq!(
        workspace
            .check(&small.workspace_requirements(()).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
    reconstruct(&small, &[0.5, -1.], &mut out, &mut workspace).unwrap();
    // Retained capacity beyond a configured byte limit is a resource limit.
    let limited = requirements.with_max_bytes(requirements.bytes::<f64>().unwrap() - 1);
    assert_eq!(
        workspace.check(&limited).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        SignalWorkspace::<f64>::new()
            .try_reserve(&limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}
#[test]
fn subnormal_breakdown_preserves_output() {
    let tiny = f64::from_bits(1);
    let p = MinimumPhasePlan::<f64>::new(
        2,
        options(MinimumPhaseConvention::SameLength).with_fft_length(Some(4)),
    )
    .unwrap();
    let mut workspace = p.workspace(()).unwrap();
    let mut output = [42.; 2];
    assert!(matches!(
        error(reconstruct(&p, &[tiny, -tiny], &mut output, &mut workspace)).view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
    assert_eq!(output, [42.; 2]);
    let p = MinimumPhasePlan::<f64>::new(
        3,
        options(MinimumPhaseConvention::HalfLength)
            .with_method(MinimumPhaseMethod::Hilbert)
            .with_spectral_zeros(SpectralZeroPolicy::Regularize {
                relative_floor: tiny,
            }),
    )
    .unwrap();
    let mut workspace = p.workspace(()).unwrap();
    assert!(matches!(
        error(reconstruct(
            &p,
            &[tiny, tiny * 2., tiny],
            &mut output,
            &mut workspace
        ))
        .view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
    assert_eq!(output, [42.; 2]);
}
#[test]
fn complex_input_and_first_call_allocations() {
    let p = MinimumPhasePlan::<Complex<f64>>::build(
        2,
        options(MinimumPhaseConvention::SameLength),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let input = [Complex::new(0.5, 0.), Complex::new(0., 1.)];
    let mut output = [Complex::new(42., 42.); 2];
    let mut workspace = p.workspace(()).unwrap();
    p.reconstruct_into(
        &input,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((output[0].re - 1.).abs() < 1e-12);
    assert!((output[1].im - 0.5).abs() < 1e-12);
    let p = MinimumPhasePlan::<f64>::new(
        3,
        options(MinimumPhaseConvention::HalfLength).with_method(MinimumPhaseMethod::Hilbert),
    )
    .unwrap();
    let mut workspace = p.workspace(()).unwrap();
    let mut out = [0.; 2];
    reconstruct(&p, &[-0.1875, 0.625, -0.1875], &mut out, &mut workspace).unwrap();
}
#[test]
fn cancellation_preserves_output() {
    let p = MinimumPhasePlan::<f64>::build(
        2,
        options(MinimumPhaseConvention::SameLength),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut workspace = p.workspace(()).unwrap();
    let mut out = [42.; 2];
    let token = CancellationToken::new();
    token.cancel();
    assert_eq!(
        p.reconstruct_into(
            &[0.5, -1.],
            &mut out,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, [42.; 2]);
    assert_eq!(
        error_kind(MinimumPhasePlan::<f64>::build(
            2,
            options(MinimumPhaseConvention::SameLength),
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    // A completed call is repeatable with the same workspace.
    reconstruct(&p, &[0.5, -1.], &mut out, &mut workspace).unwrap();
    let previous = out;
    reconstruct(&p, &[0.5, -1.], &mut out, &mut workspace).unwrap();
    assert_eq!(out, previous);
}
#[test]
fn options_round_trip_and_default_to_sequential() {
    let defaults = MinimumPhaseOptions::<f64>::default();
    assert_eq!(defaults.method(), MinimumPhaseMethod::Homomorphic);
    assert_eq!(defaults.convention(), MinimumPhaseConvention::HalfLength);
    assert_eq!(defaults.fft_length(), None);
    assert_eq!(defaults.execution_policy(), ExecutionPolicy::Sequential);
    assert!(matches!(
        defaults.spectral_zeros(),
        SpectralZeroPolicy::Regularize { relative_floor } if relative_floor == 32. * f64::EPSILON
    ));
    let parallel = ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap());
    let tolerance = scioxide_core::ResidualTolerance::new(1e-9, 1e-8).unwrap();
    let options = defaults
        .with_execution_policy(parallel)
        .with_symmetry_tolerance(tolerance);
    assert_eq!(options.execution_policy(), parallel);
    assert_eq!(options.symmetry_tolerance(), tolerance);
}
#[test]
fn hilbert_equiripple_lift_and_independent_response() {
    let h = design_fir_remez_with_options::<f64>(
        31,
        FirSymmetry::Even,
        &[FirBand::new(0., 0.2, 1., 1.), FirBand::new(0.5, 1., 0., 0.)],
        RemezOptions::default(),
    )
    .unwrap();
    let (out, d) = minimum_phase_with_options(
        h.coefficients(),
        MinimumPhaseOptions::new()
            .with_method(MinimumPhaseMethod::Hilbert)
            .with_fft_length(Some(8192)),
    )
    .unwrap();
    assert_eq!(out.len(), 16);
    assert!(d.hilbert_lift() > 0.);
    assert!(d.regularized_bins() > 0);
    assert!(d.magnitude_error() < 0.01);
    // Oracle applies the explicitly reported ripple lift to independent zero-phase sums.
    for k in 0..513 {
        let w = core::f64::consts::PI * k as f64 / 512.;
        let (re, im) = response(h.coefficients(), w);
        let (sin, cos) = (w * 15.).sin_cos();
        let a = re * cos - im * sin;
        let target = ((a + d.hilbert_lift()).max(0.) * d.hilbert_normalization()).sqrt();
        assert!((magnitude(&out, w) - target).abs() < 0.012);
    }
    // Compose reconstructed taps with the existing FIR executor.
    let filter = PolyphasePlan::fir(&out).unwrap();
    assert_eq!(filter.coefficients(), out.as_slice());
}
#[test]
fn executor_failure_is_transactional() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn, SequentialExecutor};
    struct FailAfter {
        calls: AtomicUsize,
        fail_at: usize,
    }
    impl Executor for FailAfter {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_at {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("injected_failure"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "executor fault",
                ));
            }
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    let exec = FailAfter {
        calls: AtomicUsize::new(0),
        fail_at: 1,
    };
    let context = ExecutionContext::new(&exec);
    let parallel = options(MinimumPhaseConvention::SameLength)
        .with_fft_length(Some(65536))
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let p = MinimumPhasePlan::<f64>::build(2, parallel, context)?;
    exec.calls.store(0, Ordering::Relaxed);
    let mut workspace = p.workspace(())?;
    let mut out = [42.; 2];
    assert_eq!(
        error_kind(p.reconstruct_into(&[0.5, -1.], &mut out, &mut workspace, context)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(out, [42.; 2]);
    assert!(exec.calls.load(Ordering::Relaxed) >= 2);
    let successful = FailAfter {
        calls: AtomicUsize::new(0),
        fail_at: usize::MAX,
    };
    let context = ExecutionContext::new(&successful);
    let p = MinimumPhasePlan::<f64>::build(2, parallel, context)?;
    let mut workspace = p.workspace(())?;
    p.reconstruct_into(&[0.5, -1.], &mut out, &mut workspace, context)?;
    assert!((out[0] - 1.).abs() < 1e-12 && (out[1] + 0.5).abs() < 1e-12);
    let previous = out;
    p.reconstruct_into(&[0.5, -1.], &mut out, &mut workspace, context)?;
    assert_eq!(out, previous);
    assert!(successful.calls.load(Ordering::Relaxed) > 0);
    Ok(())
}
