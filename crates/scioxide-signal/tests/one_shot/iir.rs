//! One-shot forms: IIR sections, transfer functions and forward–backward filtering.
use super::{bits, error, error_kind, invalid};
use scioxide_core::{
    Complex, DomainReason, ExecutionContext, InvalidInputReason, NumericScalar, ResidualTolerance,
    SciOxideErrorKind, SciOxideErrorView, argument,
};
use scioxide_signal::{
    ForwardBackwardOptions, IirBand, IirDomain, IirFamily, IirFilter, IirInitialization,
    IirOrderFamily, IirPadding, IirRequest, SecondOrderSections, SignalExecutionOptions,
    SignalLayout, TransferFunction, design_iir, select_iir_order,
};

type C = Complex<f64>;

fn lowpass(order: usize) -> SecondOrderSections<f64> {
    let band = IirBand::Lowpass(0.2);
    design_iir(order, IirFamily::Butterworth, band, IirDomain::Digital)
        .unwrap()
        .to_sos()
        .unwrap()
}

fn record(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0.37 * i as f64).sin() + 0.25 * (1.9 * i as f64).cos() + 0.5)
        .collect()
}

fn complex_record(n: usize) -> Vec<C> {
    record(n)
        .iter()
        .zip(record(n + 3).iter().skip(3))
        .map(|(&re, &im)| C::new(re, -im))
        .collect()
}

/// Odd extension over `length` samples and steady-state delays in both passes.
fn conventional<'a, U: NumericScalar>(length: usize) -> ForwardBackwardOptions<'a, U> {
    ForwardBackwardOptions::new()
        .with_padding(IirPadding::Odd(length))
        .with_forward(IirInitialization::SteadyState)
        .with_backward(IirInitialization::SteadyState)
}

/// The prepared forward–backward form with a caller workspace.
fn prepared<T, U>(
    filter: &IirFilter<T>,
    input: &[U],
    options: ForwardBackwardOptions<'_, U>,
) -> Vec<U>
where
    T: NumericScalar,
    U: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let request =
        IirRequest::ForwardBackward(SignalLayout::Contiguous(input.len()), options.padding());
    let mut workspace = filter.workspace::<U>(request).unwrap();
    let mut output = vec![U::ZERO; input.len()];
    let cx = ExecutionContext::sequential_native();
    filter
        .forward_backward_into(input, options, &mut output, &mut workspace, cx)
        .unwrap();
    output
}

#[test]
fn to_sos_is_the_real_conversion_at_the_default_tolerance() {
    for (family, band) in [
        (IirFamily::Butterworth, IirBand::Lowpass(0.2)),
        (
            IirFamily::ChebyshevI { ripple: 1.0 },
            IirBand::Bandpass(0.2, 0.4),
        ),
        (
            IirFamily::Elliptic {
                ripple: 0.5,
                attenuation: 60.0,
            },
            IirBand::Highpass(0.3),
        ),
    ] {
        for domain in [IirDomain::Digital, IirDomain::Analog] {
            let design = design_iir(5, family, band, domain).unwrap();
            let configured = design
                .zpk()
                .to_sos()
                .unwrap()
                .to_real(ResidualTolerance::default())
                .unwrap();
            let sos = design.to_sos().unwrap();
            assert_eq!(sos.domain(), domain);
            assert_eq!(
                bits(sos.sections().as_flattened()),
                bits(configured.sections().as_flattened())
            );
        }
    }
}

#[test]
fn order_reports_carry_the_family_of_the_specification() {
    let (pass, stop) = (IirBand::Lowpass(0.2_f64), IirBand::Lowpass(0.3));
    let (ripple, attenuation) = (1.0, 60.0);
    for family in [
        IirOrderFamily::Butterworth,
        IirOrderFamily::ChebyshevI,
        IirOrderFamily::ChebyshevII,
        IirOrderFamily::Elliptic,
    ] {
        let report =
            select_iir_order(family, pass, stop, ripple, attenuation, IirDomain::Digital).unwrap();
        let expected = match family {
            IirOrderFamily::Butterworth => IirFamily::Butterworth,
            IirOrderFamily::ChebyshevI => IirFamily::ChebyshevI { ripple },
            IirOrderFamily::ChebyshevII => IirFamily::ChebyshevII { attenuation },
            IirOrderFamily::Elliptic => IirFamily::Elliptic {
                ripple,
                attenuation,
            },
        };
        assert_eq!(format!("{:?}", report.family()), format!("{expected:?}"));
        // Designing from the report equals designing from the restated family.
        let from_report = design_iir(
            report.order(),
            report.family(),
            report.band(),
            IirDomain::Digital,
        )
        .unwrap();
        let restated =
            design_iir(report.order(), expected, report.band(), IirDomain::Digital).unwrap();
        assert_eq!(
            bits(from_report.zpk().poles()),
            bits(restated.zpk().poles())
        );
        assert_eq!(
            bits(from_report.zpk().zeros()),
            bits(restated.zpk().zeros())
        );
    }
}

#[test]
fn forward_backward_with_options_equals_the_prepared_form() {
    let sos = lowpass(4);
    let filter = IirFilter::from_sos(&sos, SignalExecutionOptions::new()).unwrap();
    let x = record(40);
    let state = [0.1, -0.2, 0.05, 0.3];
    let options = [
        ForwardBackwardOptions::new(),
        conventional(15),
        ForwardBackwardOptions::new()
            .with_padding(IirPadding::Even(7))
            .with_forward(IirInitialization::State(&state))
            .with_backward(IirInitialization::SteadyState),
        ForwardBackwardOptions::new().with_padding(IirPadding::Constant(39)),
    ];
    for options in options {
        let owned = filter.forward_backward_with_options(&x, options).unwrap();
        assert_eq!(bits(&owned), bits(&prepared(&filter, &x, options)));
    }
    // Complex samples through real sections, and complex sections.
    let z = complex_record(40);
    let owned = filter
        .forward_backward_with_options(&z, conventional(15))
        .unwrap();
    assert_eq!(bits(&owned), bits(&prepared(&filter, &z, conventional(15))));
    let rotated: Vec<[C; 6]> = sos
        .sections()
        .iter()
        .map(|row| row.map(|b| C::new(b, 0.25 * b)))
        .collect();
    let complex = SecondOrderSections::new(IirDomain::Digital, &rotated).unwrap();
    let filter = IirFilter::from_sos(&complex, SignalExecutionOptions::new()).unwrap();
    let owned = filter
        .forward_backward_with_options(&z, conventional(15))
        .unwrap();
    assert_eq!(bits(&owned), bits(&prepared(&filter, &z, conventional(15))));
}

#[test]
fn forward_backward_with_options_reports_the_errors_of_the_prepared_form() {
    let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1.0, -0.5]).unwrap();
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new()).unwrap();
    let x = record(6);
    let padded = ForwardBackwardOptions::new().with_padding(IirPadding::Odd(6));
    let (argument, reason, _) = invalid(filter.forward_backward_with_options(&x, padded));
    assert_eq!(
        (argument, reason),
        (
            argument!("options.padding"),
            InvalidInputReason::Inconsistent
        )
    );
    let state = [0.0; 2];
    let explicit = ForwardBackwardOptions::new().with_forward(IirInitialization::State(&state));
    assert_eq!(
        error_kind(filter.forward_backward_with_options(&x, explicit)),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        error_kind(filter.forward_backward_with_options(&[1.0, f64::NAN], padded)),
        SciOxideErrorKind::NonFinite
    );
    // y[n] = x[n] + x[n − 1] overflows on two samples of f64::MAX.
    let sum = TransferFunction::new(IirDomain::Digital, &[1.0_f64, 1.0], &[1.0]).unwrap();
    let sum = IirFilter::from_transfer(&sum, SignalExecutionOptions::new()).unwrap();
    let none = ForwardBackwardOptions::new();
    assert_eq!(
        error_kind(sum.forward_backward_with_options(&[f64::MAX, f64::MAX], none)),
        SciOxideErrorKind::NumericalInstability
    );
    let integrator = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[1.0, -1.0]).unwrap();
    let filter = IirFilter::from_transfer(&integrator, SignalExecutionOptions::new()).unwrap();
    let steady = ForwardBackwardOptions::new().with_forward(IirInitialization::SteadyState);
    let complex = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[C::new(1.0, 1.0)], &[C::new(1.0, 0.0)])
            .unwrap(),
        SignalExecutionOptions::new(),
    )
    .unwrap();
    assert!(matches!(
        error(filter.forward_backward_with_options(&x, steady)).view(),
        SciOxideErrorView::DomainError { argument, reason: DomainReason::Undefined, .. }
            if argument == argument!("self")
    ));
    let (argument, reason, _) = invalid(complex.forward_backward_with_options(&x, none));
    assert_eq!(
        (argument, reason),
        (
            argument!("self"),
            InvalidInputReason::UnsupportedConfiguration
        )
    );
}

#[test]
fn sections_filter_and_forward_backward_equal_their_configured_forms() {
    // Order 8 has four sections and eight delays: 3·(8 + 1) = 27 samples.
    let sos = lowpass(8);
    let filter = IirFilter::from_sos(&sos, SignalExecutionOptions::new()).unwrap();
    assert_eq!(filter.delays(), 8);
    let x = record(100);
    assert_eq!(
        bits(&sos.filter(&x).unwrap()),
        bits(&filter.filter(&x).unwrap())
    );
    let z = sos.forward_backward(&x).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &x, conventional(27))));
    // Complex samples.
    let y = complex_record(100);
    assert_eq!(
        bits(&sos.filter(&y).unwrap()),
        bits(&filter.filter(&y).unwrap())
    );
    let z = sos.forward_backward(&y).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &y, conventional(27))));
    // A record shorter than 28 samples is extended by one sample less than its length.
    let short = record(10);
    let z = sos.forward_backward(&short).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &short, conventional(9))));
}

#[test]
fn transfer_filter_and_forward_backward_equal_their_configured_forms() {
    // max(len(a), len(b)) = 4, so the extension is 12 samples.
    let (b, a) = ([0.2_f64, 0.3, 0.1], [1.0, -0.6, 0.2, -0.05]);
    let transfer = TransferFunction::new(IirDomain::Digital, &b, &a).unwrap();
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new()).unwrap();
    let x = record(50);
    assert_eq!(
        bits(&transfer.filter(&x).unwrap()),
        bits(&filter.filter(&x).unwrap())
    );
    let z = transfer.forward_backward(&x).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &x, conventional(12))));
    let y = complex_record(50);
    assert_eq!(
        bits(&transfer.filter(&y).unwrap()),
        bits(&filter.filter(&y).unwrap())
    );
    let z = transfer.forward_backward(&y).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &y, conventional(12))));
    // Complex coefficients with complex samples.
    let complex: Vec<C> = b.iter().map(|&v| C::new(v, -0.5 * v)).collect();
    let transfer =
        TransferFunction::new(IirDomain::Digital, &complex, &[C::new(1.0, 0.0)]).unwrap();
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new()).unwrap();
    assert_eq!(
        bits(&transfer.filter(&y).unwrap()),
        bits(&filter.filter(&y).unwrap())
    );
    let z = transfer.forward_backward(&y).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &y, conventional(9))));
}

#[test]
fn empty_and_one_sample_records() {
    let sos = lowpass(4);
    let transfer = sos.to_transfer().unwrap();
    assert!(sos.filter::<f64>(&[]).unwrap().is_empty());
    assert!(sos.forward_backward::<f64>(&[]).unwrap().is_empty());
    assert!(transfer.filter::<C>(&[]).unwrap().is_empty());
    assert!(transfer.forward_backward::<C>(&[]).unwrap().is_empty());
    // One sample admits no extension; the passes still start in steady state.
    let filter = IirFilter::from_sos(&sos, SignalExecutionOptions::new()).unwrap();
    let one = [1.5];
    let z = sos.forward_backward(&one).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &one, conventional(0))));
    let unpadded = ForwardBackwardOptions::new()
        .with_forward(IirInitialization::SteadyState)
        .with_backward(IirInitialization::SteadyState);
    assert_eq!(bits(&z), bits(&prepared(&filter, &one, unpadded)));
    // The unit DC gain keeps a constant sample.
    assert!((z[0] - 1.5).abs() < 1e-12);
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new()).unwrap();
    let z = transfer.forward_backward(&one).unwrap();
    assert_eq!(bits(&z), bits(&prepared(&filter, &one, unpadded)));
}

#[test]
fn analog_representations_are_rejected_under_self() {
    let analog =
        SecondOrderSections::new(IirDomain::Analog, &[[0.0, 0.0, 1.0, 1.0, 1.4, 1.0]]).unwrap();
    let expected = invalid(IirFilter::from_sos(&analog, SignalExecutionOptions::new()));
    assert_eq!(expected.0, argument!("sos"));
    let wanted = (argument!("self"), expected.1, expected.2);
    assert_eq!(invalid(analog.filter(&[1.0])), wanted);
    assert_eq!(invalid(analog.forward_backward(&[1.0])), wanted);
    let transfer = TransferFunction::new(IirDomain::Analog, &[1.0_f64], &[1.0, 1.0]).unwrap();
    let expected = invalid(IirFilter::from_transfer(
        &transfer,
        SignalExecutionOptions::new(),
    ));
    assert_eq!(expected.0, argument!("transfer"));
    let wanted = (
        argument!("self"),
        InvalidInputReason::OutOfRange,
        expected.2,
    );
    assert_eq!(invalid(transfer.filter(&[C::new(1.0, 0.0)])), wanted);
    assert_eq!(invalid(transfer.forward_backward(&[1.0])), wanted);
    // The domain is checked before the samples.
    assert_eq!(invalid(transfer.filter(&[f64::NAN])), wanted);
}

#[test]
fn representation_errors_follow_the_filter() {
    let rows = [[
        C::new(1.0, 0.5),
        C::new(0.0, 0.0),
        C::new(0.0, 0.0),
        C::new(1.0, 0.0),
        C::new(-0.5, 0.0),
        C::new(0.0, 0.0),
    ]];
    let complex = SecondOrderSections::new(IirDomain::Digital, &rows).unwrap();
    for result in [
        complex.filter(&[1.0_f64]),
        complex.forward_backward(&[1.0_f64]),
    ] {
        let (argument, reason, _) = invalid(result);
        assert_eq!(
            (argument, reason),
            (
                argument!("self"),
                InvalidInputReason::UnsupportedConfiguration
            )
        );
    }
    let sos = lowpass(2);
    let transfer = TransferFunction::new(IirDomain::Digital, &[1.0_f64, 1.0], &[1.0]).unwrap();
    for result in [
        sos.filter(&[0.0, f64::NAN]),
        sos.forward_backward(&[0.0, f64::INFINITY]),
        transfer.filter(&[f64::NAN]),
        transfer.forward_backward(&[0.0, 1.0, f64::NAN]),
    ] {
        assert_eq!(error_kind(result), SciOxideErrorKind::NonFinite);
    }
    // y[n] = x[n] + x[n − 1] overflows on two samples of f64::MAX.
    let large = [f64::MAX, f64::MAX];
    for result in [transfer.filter(&large), transfer.forward_backward(&large)] {
        assert!(matches!(
            error(result).view(),
            SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
    // An integrator has no steady state: its denominator vanishes at z = 1.
    let integrator = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[1.0, -1.0]).unwrap();
    assert!(integrator.filter(&[1.0, 2.0]).is_ok());
    assert!(matches!(
        error(integrator.forward_backward(&[1.0, 2.0])).view(),
        SciOxideErrorView::DomainError { argument, reason: DomainReason::Undefined, .. }
            if argument == argument!("self")
    ));
}

#[test]
fn conventional_forward_backward_suppresses_end_transients() {
    // A zero-phase lowpass with unit DC gain passes a ramp; only the ends deviate.
    let sos = lowpass(6);
    let ramp: Vec<f64> = (0..200).map(|i| 1.0 + 0.001 * i as f64).collect();
    let deviation = |z: &[f64]| {
        z.iter()
            .zip(&ramp)
            .fold(0.0_f64, |e, (y, x)| e.max((y - x).abs()))
    };
    let conventional = deviation(&sos.forward_backward(&ramp).unwrap());
    let filter = IirFilter::from_sos(&sos, SignalExecutionOptions::new()).unwrap();
    let plain = filter.forward_backward_with_options(&ramp, ForwardBackwardOptions::new());
    let plain = deviation(&plain.unwrap());
    assert!(
        conventional < 1e-4 && plain > 0.1,
        "{conventional:e}, {plain:e}"
    );
}
