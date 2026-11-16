use super::*;

#[test]
fn iir_first_order_impulse() -> Result<(), SciOxideError> {
    let tf = TransferFunction::new(IirDomain::Digital, &[0.5], &[1.0, -0.5])?;
    let plan = IirFilter::from_transfer(&tf, SignalExecutionOptions::new())?;
    assert_eq!(
        plan.filter(&[1.0, 0.0, 0.0, 0.0])?,
        [0.5, 0.25, 0.125, 0.0625]
    );
    Ok(())
}
fn scalar_domains<S>() -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let s = |x| number::<S>(x);
    for family in [
        IirFamily::Butterworth,
        IirFamily::ChebyshevI { ripple: s(1.)? },
        IirFamily::ChebyshevII {
            attenuation: s(40.)?,
        },
        IirFamily::Elliptic {
            ripple: s(1.)?,
            attenuation: s(40.)?,
        },
        IirFamily::Bessel(BesselNormalization::Delay),
    ] {
        let design = design_iir(5, family, IirBand::Lowpass(s(0.3)?), IirDomain::Digital)?;
        let sos = design
            .zpk()
            .to_sos()?
            .to_real(ResidualTolerance::new(s(2e-5)?, s(2e-5)?)?)?;
        let plan = IirFilter::from_sos(&sos, SignalExecutionOptions::new())?;
        let input: Vec<S> = (0..100)
            .map(|i| s((i as f64 * 0.19).sin()))
            .collect::<Result<_, _>>()?;
        let out = plan.filter(&input)?;
        let complex: Vec<_> = input.iter().map(|&x| Complex::new(x, -x)).collect();
        let co = plan.filter(&complex)?;
        for (a, b) in out.iter().zip(co) {
            close(a.into_f64(), b.re.into_f64(), 1e-5);
            close(a.into_f64(), -b.im.into_f64(), 1e-5);
        }
        let mut state = IirState::new(&plan, 1)?;
        let mut work = plan.workspace(IirRequest::Filter(SignalLayout::Contiguous(100)))?;
        let mut chunked = zeros(100)?;
        for (x, y) in input.chunks(7).zip(chunked.chunks_mut(7)) {
            plan.filter_into(x, y, &mut state, &mut work, ExecutionContext::sequential())?;
        }
        assert_eq!(out, chunked);
    }
    Ok(())
}
#[test]
fn iir_both_precisions_real_complex_chunks() -> Result<(), SciOxideError> {
    scalar_domains::<f32>()?;
    scalar_domains::<f64>()
}
#[test]
fn iir_initialization_and_boundaries() -> Result<(), SciOxideError> {
    let tf = TransferFunction::new(IirDomain::Digital, &[0.5], &[1., -0.5])?;
    let plan = IirFilter::from_transfer(&tf, SignalExecutionOptions::new())?;
    // The unit-step steady state of 0.5 / (1 - 0.5 z^-1) holds y = 1 with delay 0.5,
    // so filtering a constant from it stays at the constant.
    let mut steady = [0.];
    plan.initialize(&mut steady, IirInitialization::SteadyState, 1.)?;
    assert_eq!(steady, [0.5]);
    let mut state = IirState::new(&plan, 1)?;
    state.values_mut().copy_from_slice(&steady);
    let mut w = plan.workspace(IirRequest::Filter(SignalLayout::Contiguous(4)))?;
    let mut out = [0.; 4];
    plan.filter_into(
        &[1.; 4],
        &mut out,
        &mut state,
        &mut w,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(out, [1.; 4]);
    assert_eq!(state.values(), steady);
    for padding in [
        IirPadding::None,
        IirPadding::Odd(3),
        IirPadding::Even(3),
        IirPadding::Constant(3),
    ] {
        let pad = padding.length();
        let mut w = plan.workspace(IirRequest::ForwardBackward(
            SignalLayout::Contiguous(8),
            padding,
        ))?;
        let mut out = [0.; 8];
        plan.forward_backward_into(
            &[2.; 8],
            ForwardBackwardOptions::new()
                .with_padding(padding)
                .with_forward(IirInitialization::SteadyState)
                .with_backward(IirInitialization::SteadyState),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )?;
        assert_eq!(out, [2.; 8]);
        // Independent direct convolution of the geometrical impulse response for both passes.
        let input = [1., 3., -2., 4., 1., 0., -1., 2.];
        let mut extended = Vec::new();
        for j in (1..=pad).rev() {
            extended.push(match padding {
                IirPadding::Odd(_) => 2. * input[0] - input[j],
                IirPadding::Even(_) => input[j],
                _ => input[0],
            });
        }
        extended.extend(input);
        for j in 1..=pad {
            extended.push(match padding {
                IirPadding::Odd(_) => 2. * input[7] - input[7 - j],
                IirPadding::Even(_) => input[7 - j],
                _ => input[7],
            });
        }
        let direct = |x: &[f64]| -> Vec<f64> {
            (0..x.len())
                .map(|i| {
                    (0..=i)
                        .map(|j| x[j] * 0.5_f64.powi((i - j + 1) as i32))
                        .sum()
                })
                .collect()
        };
        let mut first = direct(&extended);
        first.reverse();
        let mut expected = direct(&first);
        expected.reverse();
        plan.forward_backward_into(
            &input,
            ForwardBackwardOptions::new().with_padding(padding),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )?;
        for (i, v) in out.iter().enumerate() {
            close(*v, expected[pad + i], 1e-14);
        }
    }
    Ok(())
}

#[test]
fn iir_invalid_inputs_and_empty_records() -> Result<(), SciOxideError> {
    assert_eq!(
        error_kind(TransferFunction::new(IirDomain::Digital, &[1.], &[0.])),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SecondOrderSections::new(
            IirDomain::Digital,
            &[[1., 0., 0., 1., -1., 0.]]
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(iir_prototype(0, IirFamily::<f64>::Butterworth)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(iir_prototype(
            3,
            IirFamily::Elliptic {
                ripple: 2.,
                attenuation: 1.
            }
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(design_iir(
            2,
            IirFamily::Butterworth,
            IirBand::Lowpass(1.),
            IirDomain::Digital
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert!(
        select_iir_order_with_options(
            IirOrderFamily::Butterworth,
            IirBand::Lowpass(0.2),
            IirBand::Lowpass(0.201),
            1.,
            80.,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(2)
        )?
        .termination()
            == IirOrderTermination::OrderLimit
    );
    let plan = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[1.], &[1., -1.])?,
        SignalExecutionOptions::new(),
    )?;
    // An integrator has a zero DC denominator, so its steady state is undefined.
    let mut out = [99.; 2];
    let mut w = plan.workspace(IirRequest::ForwardBackward(
        SignalLayout::Contiguous(2),
        IirPadding::None,
    ))?;
    assert_eq!(
        plan.forward_backward_into(
            &[1.; 2],
            ForwardBackwardOptions::new().with_forward(IirInitialization::SteadyState),
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DomainError
    );
    assert_eq!(out, [99.; 2]);
    assert_eq!(plan.filter::<f64>(&[])?, Vec::<f64>::new());
    let mut out = [99.];
    let mut w = plan.workspace(IirRequest::ForwardBackward(
        SignalLayout::Contiguous(1),
        IirPadding::Odd(1),
    ))?;
    assert_eq!(
        error_kind(plan.forward_backward_into(
            &[1.],
            ForwardBackwardOptions::new().with_padding(IirPadding::Odd(1)),
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(out, [99.]);
    let cplan = IirFilter::from_transfer(
        &TransferFunction::new(
            IirDomain::Digital,
            &[Complex::new(1., 1.)],
            &[Complex::new(1., 0.)],
        )?,
        SignalExecutionOptions::new(),
    )?;
    assert_eq!(
        error_kind(cplan.filter(&[1.])),
        SciOxideErrorKind::InvalidInput
    );
    Ok(())
}

#[test]
fn iir_complex_coefficients_and_supplied_states() -> Result<(), SciOxideError> {
    let b = [Complex::new(0.2, 0.1), Complex::new(-0.1, 0.05)];
    let a = [Complex::new(1., 0.), Complex::new(-0.4, -0.3)];
    let tf = TransferFunction::new(IirDomain::Digital, &b, &a)?;
    let p = IirFilter::from_transfer(&tf, SignalExecutionOptions::new())?;
    let input = [
        Complex::new(1., 2.),
        Complex::new(-2., 0.5),
        Complex::new(0., 0.),
        Complex::new(1., 0.),
    ];
    let y = p.filter(&input)?;
    let mut last = Complex::new(0., 0.);
    for i in 0..input.len() {
        let expected = b[0] * input[i]
            + if i == 0 {
                Complex::new(0., 0.)
            } else {
                b[1] * input[i - 1]
            }
            - a[1] * last;
        close((y[i] - expected).norm(), 0., 1e-14);
        last = expected;
    }
    // The single transposed direct-form II delay after the record, derived from
    // the newest input/output pair: d = b1 x[3] - a1 y[3].
    let s = [b[1] * input[3] - a[1] * y[3]];
    let mut state = IirState::new(&p, 1)?;
    state.values_mut().copy_from_slice(&s);
    let mut w = p.workspace(IirRequest::Filter(SignalLayout::Contiguous(4)))?;
    let mut continuation = [Complex::new(0., 0.); 4];
    p.filter_into(
        &input,
        &mut continuation,
        &mut state,
        &mut w,
        ExecutionContext::sequential(),
    )?;
    let twice: Vec<_> = input.iter().chain(&input).copied().collect();
    assert_eq!(continuation, p.filter(&twice)?[4..]);
    let mut w = p.workspace(IirRequest::ForwardBackward(
        SignalLayout::Contiguous(4),
        IirPadding::None,
    ))?;
    let mut out = [Complex::new(0., 0.); 4];
    p.forward_backward_into(
        &input,
        ForwardBackwardOptions::new()
            .with_forward(IirInitialization::State(&s))
            .with_backward(IirInitialization::State(&s)),
        &mut out,
        &mut w,
        ExecutionContext::sequential(),
    )?;
    assert!(out.iter().all(|x| x.is_finite()));
    Ok(())
}

#[test]
fn iir_arithmetic_and_executor_failures_preserve_state() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};
    struct Failing;
    impl Executor for Failing {
        fn worker_budget(&self) -> NonZeroUsize {
            const { NonZeroUsize::new(2).expect("resource limit must be positive") }
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(invalid(
                scioxide_core::argument!("context"),
                "injected failure",
            ))
        }
    }
    let p = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[2.], &[1., -0.5])?,
        SignalExecutionOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
            const { NonZeroUsize::new(2).expect("resource limit must be positive") },
        )),
    )?;
    let mut out = [99.; 4];
    let mut state = IirState::new(&p, 1)?;
    let mut w = p.workspace(IirRequest::Filter(SignalLayout::Contiguous(4)))?;
    assert_eq!(
        error_kind(p.filter_into(
            &[1., 2., f64::MAX, 0.],
            &mut out,
            &mut state,
            &mut w,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(out, [99.; 4]);
    assert_eq!(state.values(), [0.]);
    let shape = [2, 2];
    let layout = StridedLayout::new(&shape, &[2, 1])?;
    let mut state = IirState::new(&p, 2)?;
    let mut w = p.workspace(IirRequest::Filter(SignalLayout::Axis {
        shape: &shape,
        axis: 0,
    }))?;
    assert_eq!(
        error_kind(p.filter_axis_into(
            NdView::new(&[1.; 4], layout)?,
            0,
            NdViewMut::new(&mut out, layout)?,
            &mut state,
            &mut w,
            ExecutionContext::new(&Failing)
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(out, [99.; 4]);
    assert_eq!(state.values(), [0., 0.]);
    Ok(())
}

#[test]
fn iir_forward_backward_preflight_and_workspace_errors_preserve_output() -> Result<(), SciOxideError>
{
    use scioxide_core::{Capacity, SciOxideErrorKind};
    let p = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[0.5], &[1., -0.5])?,
        SignalExecutionOptions::new(),
    )?;
    let mut out = [99.; 4];
    let request = IirRequest::ForwardBackward(SignalLayout::Contiguous(4), IirPadding::None);
    let mut w = p.workspace(request)?;
    assert_eq!(
        error_kind(p.forward_backward_into(
            &[f64::NAN; 4],
            ForwardBackwardOptions::default(),
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, [99.; 4]);
    let mut short = p.workspace(IirRequest::ForwardBackward(
        SignalLayout::Contiguous(3),
        IirPadding::None,
    ))?;
    assert_eq!(
        p.forward_backward_into(
            &[1.; 4],
            ForwardBackwardOptions::default(),
            &mut out,
            &mut short,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [99.; 4]);
    let requirements = p.workspace_requirements(request)?;
    assert_eq!(requirements.values_len(), 2 * 4 + p.delays());
    let mut empty = crate::SignalWorkspace::<f64>::new();
    assert_eq!(empty.check(&requirements)?, Capacity::NeedsGrowth);
    assert_eq!(
        empty
            .try_reserve(&requirements.with_max_bytes(8))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    empty.try_reserve(&requirements)?;
    assert_eq!(empty.check(&requirements)?, Capacity::Sufficient);
    assert_eq!(
        error_kind(
            p.workspace_requirements(IirRequest::Filter(SignalLayout::Axis {
                shape: &[2, 2],
                axis: 2
            }))
        ),
        SciOxideErrorKind::InvalidInput
    );
    let dc = design_real_gammatone(4, 0.0, 0.05)?;
    close(dc.response(0.)?.norm(), 1., 1e-12);
    Ok(())
}
