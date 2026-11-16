//! IIR gain representability, Bessel prototype accuracy and certified
//! pole-zero conversion of clustered roots.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, ExecutionContext, NumericalIssue, RealScalar, ResidualTolerance, SciOxideError,
    SciOxideErrorView,
};
use scioxide_linalg::Matrix;
use scioxide_signal::{
    BesselNormalization, IirBand, IirDomain, IirFamily, IirOrderFamily, IirOrderOptions, LtiTime,
    PoleZeroGain, StateSpace, design_iir, design_iir_scoped, design_real_gammatone_scoped,
    iir_prototype, select_iir_order_with_options,
};

fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

/// Bit patterns of the zeros, poles and gain, in order.
fn zpk_bits(zpk: &PoleZeroGain<f64>) -> Vec<u64> {
    let gain = [zpk.gain()];
    let values = zpk.zeros().iter().chain(zpk.poles()).chain(&gain);
    values
        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
        .collect()
}

fn underflow<T>(result: Result<T, SciOxideError>) -> bool {
    result.err().is_some_and(|error| {
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        )
    })
}

fn butterworth<S>(order: usize, band: IirBand<S>, domain: IirDomain) -> Result<(), SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    design_iir(order, IirFamily::Butterworth, band, domain).map(|_| ())
}

#[test]
fn unrepresentable_zero_pole_gain_is_an_underflow_error() {
    // The order `select_iir_order` recommends for pass 0.001, stop 0.0011,
    // 0.1/100 dB has an exact digital gain far below the binary64 range.
    let selected = select_iir_order_with_options(
        IirOrderFamily::Butterworth,
        IirBand::Lowpass(0.001_f64),
        IirBand::Lowpass(0.0011),
        0.1,
        100.,
        IirDomain::Digital,
        IirOrderOptions::new().with_maximum_order(1000),
    )
    .unwrap();
    assert!(selected.is_success() && selected.order() == 141);
    assert!(underflow(butterworth(
        selected.order(),
        selected.band(),
        IirDomain::Digital
    )));
    assert!(underflow(butterworth(
        60,
        IirBand::Lowpass(1e-6_f64),
        IirDomain::Digital
    )));
    assert!(underflow(butterworth(
        100,
        IirBand::Bandpass(0.001_f64, 0.0011),
        IirDomain::Digital
    )));
    assert!(underflow(butterworth(
        40,
        IirBand::Lowpass(1e-9_f64),
        IirDomain::Analog
    )));
    assert!(underflow(butterworth(
        16,
        IirBand::Lowpass(0.001_f32),
        IirDomain::Digital
    )));
    // Lowpass 0.01 has digital gain about tan(0.005 pi)^order: order 170 is
    // 4.06e-308 (normal), order 171 is 6.3e-310 (subnormal, below working
    // precision); in binary32 order 20 is 6.9e-37 and order 21 is 1.07e-38.
    for (order, representable) in [(170, true), (171, false)] {
        let design = design_iir(
            order,
            IirFamily::Butterworth,
            IirBand::Lowpass(0.01_f64),
            IirDomain::Digital,
        );
        if representable {
            let design = design.unwrap();
            assert!(design.zpk().gain().re >= f64::MIN_POSITIVE);
            assert!((design.zpk().response(0.0).unwrap().norm() - 1.0).abs() < 1e-12);
        } else {
            assert!(underflow(design));
        }
    }
    for (order, representable) in [(20, true), (21, false)] {
        let design = design_iir(
            order,
            IirFamily::Butterworth,
            IirBand::Lowpass(0.01_f32),
            IirDomain::Digital,
        );
        if representable {
            let design = design.unwrap();
            assert!(design.zpk().gain().re >= f32::MIN_POSITIVE);
            assert!((design.zpk().response(0.0).unwrap().norm() - 1.0).abs() < 1e-4);
        } else {
            assert!(underflow(design));
        }
    }
}

#[test]
fn section_gains_multiply_in_extended_range() {
    use scioxide_signal::SecondOrderSections;
    // Partial products leave the binary64 range although each final gain is
    // ordinary: sequential rounding gave a silent zero gain (or an overflow).
    for (gains, expected) in [
        ([1e-200, 1e-200, 1e300], 1e-100),
        ([1e200, 1e200, 1e-300], 1e100),
    ] {
        let rows: Vec<[f64; 6]> = gains
            .iter()
            .map(|&g| [g, 0.0, 0.0, 1.0, -0.5, 0.0])
            .collect();
        let sections = SecondOrderSections::new(IirDomain::Digital, &rows).unwrap();
        let zpk = sections.to_zpk().unwrap();
        assert!((zpk.gain().re - expected).abs() <= 4.0 * f64::EPSILON * expected);
        assert_eq!(zpk.gain().im, 0.0);
    }
    // A product below the normal range is reported, never published as zero.
    let rows = [
        [1e-200, 0.0, 0.0, 1.0, -0.5, 0.0],
        [1e-200, 0.0, 0.0, 1.0, 0.5, 0.0],
    ];
    let sections = SecondOrderSections::new(IirDomain::Digital, &rows).unwrap();
    assert!(underflow(sections.to_zpk()));
}

/// Fixture rows `(order, normalization, poles)` from mpmath roots.
fn bessel_reference() -> Vec<(usize, BesselNormalization, Vec<Complex<f64>>)> {
    include_str!("fixtures/bessel_poles.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let order: usize = fields[0].parse().unwrap();
            let normalization = match fields[1] {
                "delay" => BesselNormalization::Delay,
                "phase" => BesselNormalization::Phase,
                _ => BesselNormalization::Magnitude,
            };
            let values: Vec<f64> = fields[2..2 + 2 * order]
                .iter()
                .map(|x| x.parse().unwrap())
                .collect();
            let poles = values
                .chunks_exact(2)
                .map(|pair| Complex::new(pair[0], pair[1]))
                .collect();
            (order, normalization, poles)
        })
        .collect()
}

fn largest_relative_error(actual: &[Complex<f64>], reference: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    reference
        .iter()
        .map(|r| {
            actual
                .iter()
                .map(|p| (*p - *r).norm())
                .fold(f64::INFINITY, f64::min)
                / r.norm()
        })
        .fold(0.0, f64::max)
}

#[test]
fn bessel_prototypes_match_mpmath_through_order_25() {
    let reference = bessel_reference();
    assert_eq!(reference.len(), 75);
    for (order, normalization, poles) in &reference {
        let family = IirFamily::Bessel(*normalization);
        let wide = iir_prototype(*order, family).unwrap();
        let error = largest_relative_error(wide.poles(), poles);
        assert!(
            error <= 16.0 * f64::EPSILON,
            "f64 order {order} {normalization:?}: {error:e}"
        );
        // Conjugate pairs are exact and the gain is real.
        assert!(
            wide.poles()
                .iter()
                .all(|p| wide.poles().iter().any(|q| q.re == p.re && q.im == -p.im))
        );
        assert_eq!(wide.gain().im, 0.0);
        let narrow = iir_prototype(*order, IirFamily::Bessel(*normalization))
            .map(|z: scioxide_signal::PoleZeroGain<f32>| {
                z.poles()
                    .iter()
                    .map(|p| Complex::new(f64::from(p.re), f64::from(p.im)))
                    .collect::<Vec<_>>()
            })
            .unwrap();
        let error = largest_relative_error(&narrow, poles);
        assert!(
            error <= 8.0 * f64::from(f32::EPSILON),
            "f32 order {order} {normalization:?}: {error:e}"
        );
    }
}

#[test]
fn bessel_designs_count_their_iterations_in_the_callers_scope() {
    use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            cx(),
            SolveIdentity::new(
                ProblemToken::new(434),
                scioxide_core::algorithm_id!("signal", "bessel_test", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let design = design_iir_scoped(
        25,
        IirFamily::Bessel(BesselNormalization::Magnitude),
        IirBand::Lowpass(0.2_f64),
        IirDomain::Digital,
        &mut context,
    )
    .unwrap();
    assert!(design.stability_margin() > 0.0);
    assert!((design.zpk().response(0.2).unwrap().norm_sqr() - 0.5).abs() < 1e-12);
    assert_eq!(context.counts().own().iterations(), 0);
    assert!(context.counts().nested().iterations() > 0);
}

#[test]
fn clustered_numerator_roots_reproduce_the_transfer_function() {
    // A digital Butterworth numerator is (1 + z^-1)^N. Individually verified
    // members of its N-fold root can scatter unevenly; the certified set must
    // reproduce the coefficients, so the pole-zero response matches the
    // transfer response within the coefficient reconstruction tolerance.
    for order in 8..=16 {
        let design = design_iir(
            order,
            IirFamily::Butterworth,
            IirBand::Lowpass(0.3_f64),
            IirDomain::Digital,
        )
        .unwrap();
        let sections = design
            .zpk()
            .to_sos()
            .unwrap()
            .to_real(ResidualTolerance::default())
            .unwrap();
        let transfer = sections.to_transfer().unwrap();
        let zpk = transfer.to_zpk().unwrap();
        assert_eq!((zpk.zeros().len(), zpk.poles().len()), (order, order));
        assert_eq!(zpk.gain().re, transfer.numerator()[0]);
        for frequency in [0.0, 0.1, 0.25, 0.3, 0.5, 0.7] {
            let expected = transfer.response(frequency).unwrap();
            let actual = zpk.response(frequency).unwrap();
            assert!(
                (actual - expected).norm() <= 1e-9 * expected.norm(),
                "order {order} at {frequency}: {actual:?} vs {expected:?}"
            );
        }
        let pole_error = largest_relative_error(zpk.poles(), design.zpk().poles());
        assert!(pole_error <= 1e-9, "order {order}: poles {pole_error:e}");
        let section_zpk = sections.to_zpk().unwrap();
        for frequency in [0.0, 0.3, 0.7] {
            let expected = sections.response(frequency).unwrap();
            let actual = section_zpk.response(frequency).unwrap();
            assert!((actual - expected).norm() <= 1e-12 * expected.norm());
        }
    }
}

/// Deterministic uniform samples in `[-1, 1)`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }
}

#[test]
fn random_state_space_channels_convert_to_pole_zero_form() {
    let mut random = Lcg(12345);
    let n = 10;
    for _ in 0..20 {
        let a: Vec<f64> = (0..n * n)
            .map(|i| random.next() - if i % (n + 1) == 0 { 2.0 } else { 0.0 })
            .collect();
        let a = Matrix::from_vec(n, n, a).unwrap();
        let b = Matrix::from_vec(n, 1, vec![1.0; n]).unwrap();
        let c = Matrix::from_vec(1, n, (0..n).map(|_| random.next()).collect()).unwrap();
        let d = Matrix::zeros(1, 1).unwrap();
        let system =
            StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
        let transfer = system.to_transfer(0, 0).unwrap();
        let zpk = system.to_zpk(0, 0).unwrap();
        for frequency in [0.0, 0.5, 2.0] {
            let expected = transfer.response(frequency).unwrap();
            let actual = zpk.response(frequency).unwrap();
            assert!((actual - expected).norm() <= 1e-10 * expected.norm());
        }
    }
}

#[test]
fn root_finding_conversions_run_in_the_callers_scope() {
    use scioxide_core::{
        CancellationToken, ExecutionStopReason, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(434),
        scioxide_core::algorithm_id!("signal", "conversion_test", 1),
        0,
        0,
    );
    let design = design_iir(
        6,
        IirFamily::ChebyshevI { ripple: 1.0 },
        IirBand::Lowpass(0.3_f64),
        IirDomain::Digital,
    )
    .unwrap();
    let sections = design
        .zpk()
        .to_sos()
        .unwrap()
        .to_real(ResidualTolerance::default())
        .unwrap();
    let transfer = sections.to_transfer().unwrap();
    let system = transfer
        .to_state_space(LtiTime::Discrete { sample_period: 1.0 })
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state.begin_solve(cx(), identity, &mut budgets).unwrap();
    let zpk = transfer.to_zpk_scoped(&mut context).unwrap();
    let after_transfer = context.counts().nested().iterations();
    assert!(after_transfer > 0);
    system.to_zpk_scoped(0, 0, &mut context).unwrap();
    sections.to_zpk_scoped(&mut context).unwrap();
    design_real_gammatone_scoped(3, 0.25_f64, 0.05, &mut context).unwrap();
    assert!(context.counts().nested().iterations() > after_transfer);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(zpk.poles().len(), 6);
    // The owned conversions are the scoped ones in a fresh native root scope.
    let native = ExecutionContext::sequential_native;
    assert_eq!(
        [
            zpk_bits(&transfer.to_zpk().unwrap()),
            zpk_bits(&system.to_zpk(0, 0).unwrap()),
            zpk_bits(&sections.to_zpk().unwrap()),
        ],
        [
            zpk_bits(&transfer.to_zpk_scoped(native()).unwrap()),
            zpk_bits(&system.to_zpk_scoped(0, 0, native()).unwrap()),
            zpk_bits(&sections.to_zpk_scoped(native()).unwrap()),
        ]
    );

    let token = CancellationToken::new();
    token.cancel();
    let cancelled = || cx().with_cancellation(&token);
    let stopped = |error: SciOxideError| {
        matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::Cancelled
            }
        )
    };
    assert!(stopped(transfer.to_zpk_scoped(cancelled()).unwrap_err()));
    assert!(stopped(
        system.to_zpk_scoped(0, 0, cancelled()).unwrap_err()
    ));
    assert!(stopped(sections.to_zpk_scoped(cancelled()).unwrap_err()));
    assert!(stopped(
        design_real_gammatone_scoped(3, 0.25_f64, 0.05, cancelled()).unwrap_err()
    ));
}
#[test]
fn responses_keep_range_until_the_final_quotient() {
    use scioxide_core::Complex;
    use scioxide_signal::{IirDomain, PoleZeroGain, SecondOrderSections, TransferFunction};
    macro_rules! run {
        ($s:ty, $large:expr) => {{
            let large: $s = $large;
            let c = |x| Complex::new(x, 0.0 as $s);
            let tf =
                TransferFunction::new(IirDomain::Analog, &[1.0 as $s, 1.0, 1.0], &[1.0, 1.0, 1.0])
                    .unwrap();
            let response = tf.response(large).unwrap();
            assert!((response.re - 1.0).abs() < 32.0 * <$s>::EPSILON);
            assert!(response.im.abs() < 32.0 * <$s>::EPSILON);
            assert!(tf.group_delay(large).unwrap().abs() < 32.0 * <$s>::EPSILON);
            let roots = [c(-large), c(-large)];
            let zpk = PoleZeroGain::new(IirDomain::Analog, &roots, &roots, c(large)).unwrap();
            assert!((zpk.response(0.0).unwrap().re / large - 1.0).abs() < 32.0 * <$s>::EPSILON);
            // Intermediate cascade gains overflow or underflow in either ordering.
            for gains in [
                [large, large, 1.0 / large, 1.0 / large],
                [1.0 / large, 1.0 / large, large, large],
            ] {
                let rows = gains.map(|g| [g, 0.0, 0.0, 1.0, 0.0, 0.0]);
                let sos = SecondOrderSections::new(IirDomain::Digital, &rows).unwrap();
                assert!((sos.response(0.0).unwrap().re - 1.0).abs() < 64.0 * <$s>::EPSILON);
            }
            let digital =
                TransferFunction::new(IirDomain::Digital, &[1.0 as $s], &[1.0, -0.5]).unwrap();
            let expected = digital.response(0.0).unwrap();
            let actual = digital.response(<$s>::MAX).unwrap();
            assert!((actual - expected).norm() < 32.0 * <$s>::EPSILON);
        }};
    }
    run!(f32, 1e30);
    run!(f64, 1e200);
}

#[test]
fn complex_sos_gain_does_not_require_a_representable_magnitude() {
    for count in [1, 4] {
        let roots = vec![Complex::new(0.0_f64, 0.0); count];
        let gain = Complex::new(f64::MAX * 0.75, -f64::MAX * 0.75);
        let zpk = PoleZeroGain::new(IirDomain::Digital, &roots, &roots, gain).unwrap();
        let response = zpk.to_sos().unwrap().response(0.0).unwrap();
        assert!((response.re / gain.re - 1.0).abs() < 2e-13);
        assert!((response.im / gain.im - 1.0).abs() < 2e-13);
    }
    for gain in [
        Complex::new(f64::NAN, 0.0),
        Complex::new(0.0, f64::INFINITY),
    ] {
        let error = PoleZeroGain::new(IirDomain::Digital, &[], &[], gain).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

#[test]
fn discrete_state_space_frequency_reduces_before_forming_the_angle() {
    let a = Matrix::from_vec(1, 1, vec![0.5_f64]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let d = Matrix::zeros(1, 1).unwrap();
    let model = StateSpace::new(
        LtiTime::Discrete { sample_period: 1.0 },
        a.view(),
        b.view(),
        b.view(),
        d.view(),
    )
    .unwrap();
    let response = model
        .frequency_response(&[0.0, f64::MAX, -f64::MAX])
        .unwrap();
    for value in response.values() {
        assert!((value.re - 2.0).abs() < 1e-14);
        assert!(value.im.abs() < 1e-14);
    }
}
