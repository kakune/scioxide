use super::*;

#[test]
fn iir_analytical_prototypes() -> Result<(), SciOxideError> {
    for n in [1, 2, 3, 6, 9] {
        for family in [
            IirFamily::Butterworth,
            IirFamily::ChebyshevI { ripple: 1. },
            IirFamily::ChebyshevII { attenuation: 40. },
        ] {
            let p = iir_prototype(n, family)?;
            assert!(p.is_stable());
            for w in [0.01, 0.25, 0.8, 1., 1.3, 2., 8.] {
                let expected = match family {
                    IirFamily::Butterworth => 1. / (1. + w.powi(2 * n as i32)),
                    IirFamily::ChebyshevI { .. } => {
                        1. / (1. + (10_f64.powf(0.1) - 1.) * cheb(n, w).powi(2))
                    }
                    _ => {
                        let t = cheb(n, 1. / w).powi(2);
                        t / (t + 9999.)
                    }
                };
                close(p.response(w)?.norm_sqr(), expected, 3e-12);
            }
        }
    }
    let b = iir_prototype(2, IirFamily::Bessel(BesselNormalization::Delay))?;
    for w in [0., 0.5, 1., 2.] {
        let h = Complex::new(3., 0.) / Complex::new(3. - w * w, 3. * w);
        close((b.response(w)? - h).norm(), 0., 2e-12);
    }
    for norm in [
        BesselNormalization::Delay,
        BesselNormalization::Phase,
        BesselNormalization::Magnitude,
    ] {
        let b = iir_prototype(5, IirFamily::Bessel(norm))?;
        match norm {
            BesselNormalization::Delay => close(b.to_transfer()?.group_delay(0.)?, 1., 2e-10),
            BesselNormalization::Phase => close(b.gain().re, 1., 2e-10),
            BesselNormalization::Magnitude => close(b.response(1.)?.norm_sqr(), 0.5, 2e-10),
        }
    }
    Ok(())
}
#[test]
fn iir_elliptic_ripple_zeros_and_stability() -> Result<(), SciOxideError> {
    for n in [1, 2, 3, 6, 9] {
        let p = iir_prototype(
            n,
            IirFamily::Elliptic {
                ripple: 1.0,
                attenuation: 60.0,
            },
        )?;
        assert!(p.is_stable());
        assert_eq!(p.poles().len(), n);
        assert_eq!(p.zeros().len(), 2 * (n / 2));
        for i in 0..=256 {
            let mag = p.response(i as f64 / 256.)?.norm();
            assert!(
                mag <= 1. + 2e-10 && mag >= 10_f64.powf(-0.05) - 2e-10,
                "order {n}, mag {mag}"
            );
        }
        for zero in p.zeros() {
            close(p.response(zero.im)?.norm(), 0., 1e-12);
        }
        let sos = p.to_sos()?.to_real(ResidualTolerance::new(1e-12, 1e-12)?)?;
        for w in [0., 0.3, 1., 2., 10.] {
            close((sos.response(w)? - p.response(w)?).norm(), 0., 2e-10);
        }
    }
    Ok(())
}
#[test]
fn iir_transform_substitution_and_sos() -> Result<(), SciOxideError> {
    for n in [1, 2, 5, 8] {
        let p = iir_prototype(n, IirFamily::Butterworth)?;
        for band in [
            IirBand::Lowpass(2.),
            IirBand::Highpass(2.),
            IirBand::Bandpass(1., 3.),
            IirBand::Bandstop(1., 3.),
        ] {
            let t = p.transform(band)?;
            let tf = t.to_transfer()?;
            let sos = t.to_sos()?.to_real(ResidualTolerance::new(1e-11, 1e-11)?)?;
            for w in [0.2, 0.8, 1.1, 2.1, 5.] {
                let mapped = match band {
                    IirBand::Lowpass(c) => w / c,
                    IirBand::Highpass(c) => -c / w,
                    IirBand::Bandpass(a, b) => (w * w - a * b) / (w * (b - a)),
                    IirBand::Bandstop(a, b) => w * (b - a) / (a * b - w * w),
                };
                close((t.response(w)? - p.response(mapped)?).norm(), 0., 1e-10);
                close((tf.response(w)? - t.response(w)?).norm(), 0., 1e-9);
                close((sos.response(w)? - t.response(w)?).norm(), 0., 1e-9);
            }
            let digital = t.bilinear(10.)?;
            for f in [0., 0.1, 0.3, 0.9] {
                close(
                    (digital.response(f)?
                        - t.response(20. * (core::f64::consts::PI * f / 2.).tan())?)
                    .norm(),
                    0.,
                    1e-9,
                );
            }
        }
    }
    Ok(())
}
#[test]
fn iir_transfer_delays_and_conversions() -> Result<(), SciOxideError> {
    for (b, a) in [
        (vec![0., 0., 0.3, 0.2], vec![1., -0.5]),
        (vec![1., -0.2], vec![1., -0.7, 0.1]),
        (vec![0.], vec![1., -0.5]),
        (vec![2.], vec![1.]),
    ] {
        let tf = TransferFunction::new(IirDomain::Digital, &b, &a)?;
        let p = tf.to_zpk()?;
        let rt = p.to_transfer()?;
        let sos = p.to_sos()?;
        for f in [0., 0.2, 0.5, 0.9] {
            let expected = tf.response(f)?;
            close((rt.response(f)? - expected).norm(), 0., 1e-10);
            close((sos.response(f)? - expected).norm(), 0., 1e-10);
        }
    }
    let delay = TransferFunction::new(IirDomain::Digital, &[0., 0., 1.], &[1.])?;
    close(delay.group_delay(0.3)?, 2., 1e-14);
    let (tf, r) = deconvolve(&[1., 3., 3., 1.], &[1., 1.])?;
    assert_eq!(tf, [1., 2., 1.]);
    assert_eq!(r, [0.; 4]);
    Ok(())
}
#[test]
fn iir_high_order_narrow_band() -> Result<(), SciOxideError> {
    for (n, band) in [
        (12, IirBand::Bandpass(0.2, 0.21)),
        (16, IirBand::Lowpass(0.05)),
    ] {
        let d = design_iir(n, IirFamily::Butterworth, band, IirDomain::Digital)?;
        let sos = d
            .zpk()
            .to_sos()?
            .to_real(ResidualTolerance::new(1e-12, 1e-12)?)?;
        let plan = IirFilter::from_sos(&sos, SignalExecutionOptions::new())?;
        let mut impulse = vec![0.; 8192];
        impulse[0] = 1.;
        let y = plan.filter(&impulse)?;
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().map(|v| v.abs()).fold(0., f64::max) < 1.);
        for f in [0.01, 0.05, 0.2, 0.205, 0.21, 0.3] {
            let h = sos.response(f)?;
            close((h - d.zpk().response(f)?).norm(), 0., 2e-10);
            let sum = y
                .iter()
                .enumerate()
                .fold(Complex::new(0., 0.), |v, (i, x)| {
                    v + Complex::from_polar(*x, -core::f64::consts::PI * f * i as f64)
                });
            close((sum - h).norm(), 0., 2e-5);
        }
    }
    Ok(())
}
#[test]
fn iir_order_and_special_filters() -> Result<(), SciOxideError> {
    for family in [
        IirOrderFamily::Butterworth,
        IirOrderFamily::ChebyshevI,
        IirOrderFamily::ChebyshevII,
        IirOrderFamily::Elliptic,
    ] {
        for (pass, stop, fp, fs) in [
            (
                IirBand::Lowpass(0.2),
                IirBand::Lowpass(0.35),
                vec![0.2],
                vec![0.35],
            ),
            (
                IirBand::Highpass(0.35),
                IirBand::Highpass(0.2),
                vec![0.35],
                vec![0.2],
            ),
            (
                IirBand::Bandpass(0.3, 0.5),
                IirBand::Bandpass(0.2, 0.6),
                vec![0.3, 0.5],
                vec![0.2, 0.6],
            ),
            (
                IirBand::Bandstop(0.2, 0.6),
                IirBand::Bandstop(0.3, 0.5),
                vec![0.2, 0.6],
                vec![0.3, 0.5],
            ),
        ] {
            let order = select_iir_order_with_options(
                family,
                pass,
                stop,
                1.,
                40.,
                IirDomain::Digital,
                IirOrderOptions::new().with_maximum_order(64),
            )?;
            let f = match family {
                IirOrderFamily::Butterworth => IirFamily::Butterworth,
                IirOrderFamily::ChebyshevI => IirFamily::ChebyshevI { ripple: 1. },
                IirOrderFamily::ChebyshevII => IirFamily::ChebyshevII { attenuation: 40. },
                IirOrderFamily::Elliptic => IirFamily::Elliptic {
                    ripple: 1.,
                    attenuation: 40.,
                },
            };
            assert!(order.is_success());
            let d = design_iir(order.order(), f, order.band(), IirDomain::Digital)?;
            for f in fp {
                assert!(d.zpk().response(f)?.norm() >= 10_f64.powf(-0.05) - 1e-9);
            }
            for f in fs {
                assert!(d.zpk().response(f)?.norm() <= 0.010000001);
            }
        }
    }
    let notch = design_notch_peak(0.3, 20., ResonanceKind::Notch)?;
    close(notch.response(0.3)?.norm(), 0., 1e-12);
    close(notch.response(0.)?.norm(), 1., 1e-12);
    let peak = design_notch_peak(0.3, 20., ResonanceKind::Peak)?;
    close(peak.response(0.3)?.norm(), 1., 1e-12);
    let comb = design_comb(5, 0.8, ResonanceKind::Notch)?;
    close(comb.response(0.4)?.norm(), 0., 1e-12);
    let g = design_gammatone(4, 0.3, 0.04)?;
    close(g.response(0.3)?.norm(), 1., 1e-12);
    Ok(())
}
#[test]
fn iir_high_precision_fixtures() -> Result<(), SciOxideError> {
    fn run<S>(width: &str) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        Complex<S>: NumericScalar<Real = S>,
    {
        for line in include_str!("../../../tests/fixtures/iir.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0] != width {
                continue;
            }
            let read = |i: usize| {
                fields[i]
                    .parse::<f64>()
                    .map_err(|_| invalid(scioxide_core::argument!("fields"), "fixture number"))
            };
            let order = fields[2]
                .parse::<usize>()
                .map_err(|_| invalid(scioxide_core::argument!("fields"), "fixture order"))?;
            let family = match fields[1] {
                "butter" => IirFamily::Butterworth,
                "cheby1" => IirFamily::ChebyshevI { ripple: S::ONE },
                "cheby2" => IirFamily::ChebyshevII {
                    attenuation: number(60.)?,
                },
                "bessel" => IirFamily::Bessel(BesselNormalization::Delay),
                _ => IirFamily::Elliptic {
                    ripple: S::ONE,
                    attenuation: number(60.)?,
                },
            };
            let response = iir_prototype(order, family)?.response(number(read(3)?)?)?;
            let error = (Complex::new(response.re.into_f64(), response.im.into_f64())
                - Complex::new(read(4)?, read(5)?))
            .norm();
            // The unit-bounded response is a product over `order` poles (and
            // zeros), each designed to O(order) unit roundoffs: a `4 order^2`
            // eps envelope. Bessel poles, as polynomial roots, are the most
            // sensitive: the worst point (order 8 at w = 3) is 61 eps in f32
            // and 78 eps in f64, under a third of the envelope.
            let tolerance = (4 * order * order) as f64 * S::EPSILON.into_f64();
            assert!(error <= tolerance, "fixture {line}: error {error}");
        }
        Ok(())
    }
    run::<f32>("32")?;
    run::<f64>("64")
}
#[test]
fn iir_real_gammatone_and_sos_utilities() -> Result<(), SciOxideError> {
    for n in [1, 2, 4] {
        let g = design_real_gammatone(n, 0.3, 0.05)?;
        close(g.response(0.3)?.norm(), 1., 1e-10);
        let zpk = g.to_zpk()?;
        let tf = g.to_transfer()?;
        for f in [0.1, 0.3, 0.5, 0.9] {
            close((zpk.response(f)? - g.response(f)?).norm(), 0., 1e-9);
            close(g.group_delay(f)?, tf.group_delay(f)?, 1e-8);
        }
    }
    Ok(())
}
#[test]
fn iir_budget_exhaustion_is_transactional() -> Result<(), SciOxideError> {
    use scioxide_core::{
        Budget, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(519),
        scioxide_core::algorithm_id!("signal", "iir", 1),
        1,
        0,
    );
    for family in [
        IirFamily::Butterworth,
        IirFamily::Bessel(BesselNormalization::Delay),
        IirFamily::Elliptic {
            ripple: 1.,
            attenuation: 60.,
        },
    ] {
        // Iteration allowances are scope-local: the design runs in a child
        // epoch, and the parent receives its iterations as nested counts.
        let mut state = SolveState::new();
        let mut budget = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context =
            state.begin_solve(ExecutionContext::sequential(), identity, &mut budget)?;
        let design = design_iir_scoped(
            6,
            family,
            IirBand::Lowpass(0.2),
            IirDomain::Digital,
            &mut context,
        )?;
        assert!(design.zpk().is_stable());
        assert_eq!(context.counts().own().iterations(), 0);
        assert!(context.counts().nested().iterations() > 0);
        let cancel = scioxide_core::CancellationToken::new();
        cancel.cancel();
        let stopped = design_iir_scoped(
            6,
            family,
            IirBand::Lowpass(0.2),
            IirDomain::Digital,
            ExecutionContext::sequential().with_cancellation(&cancel),
        );
        assert_eq!(error_kind(stopped), SciOxideErrorKind::ExecutionStopped);
    }
    Ok(())
}
/// Bit patterns of the zeros, poles and gain, in order.
fn zpk_bits(zpk: &PoleZeroGain<f64>) -> Vec<u64> {
    let gain = [zpk.gain()];
    let values = zpk.zeros().iter().chain(zpk.poles()).chain(&gain);
    values
        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
        .collect()
}
#[test]
fn iir_successful_shared_design_and_conversion() -> Result<(), SciOxideError> {
    use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let identity = SolveIdentity::new(
        ProblemToken::new(519),
        scioxide_core::algorithm_id!("signal", "iir", 1),
        1,
        0,
    );
    for family in [
        IirFamily::Bessel(BesselNormalization::Delay),
        IirFamily::Elliptic {
            ripple: 1.,
            attenuation: 40.,
        },
    ] {
        let mut state = SolveState::new();
        let mut budget = SolveBudgets::unbounded();
        let mut context =
            state.begin_solve(ExecutionContext::sequential(), identity, &mut budget)?;
        let d = design_iir_scoped(
            4,
            family,
            IirBand::Lowpass(0.25),
            IirDomain::Digital,
            &mut context,
        )?;
        assert_eq!(context.counts().own().iterations(), 0);
        assert!(context.counts().nested().iterations() > 0);
        let free = design_iir(4, family, IirBand::Lowpass(0.25), IirDomain::Digital)?;
        // The owned design is the scoped one in a fresh native root scope.
        let native = ExecutionContext::sequential_native();
        let scoped = design_iir_scoped(
            4,
            family,
            IirBand::Lowpass(0.25),
            IirDomain::Digital,
            native,
        )?;
        assert_eq!(zpk_bits(free.zpk()), zpk_bits(scoped.zpk()));
        for (a, b) in d.zpk().poles().iter().zip(free.zpk().poles()) {
            close((*a - *b).norm(), 0., 1e-15);
        }
        let tf = TransferFunction::new(IirDomain::Digital, &[0.2, 0.1], &[1., -0.5, 0.1])?;
        let p = tf.to_zpk()?;
        for f in [0., 0.2, 0.5] {
            close((p.response(f)? - tf.response(f)?).norm(), 0., 2e-9);
        }
        assert!(d.zpk().is_stable());
    }
    Ok(())
}

#[test]
fn iir_inversion_zero_roots_and_complex_pairing() -> Result<(), SciOxideError> {
    let p = PoleZeroGain::new(
        IirDomain::Analog,
        &[Complex::new(0., 0.)],
        &[Complex::new(-1., 0.), Complex::new(-2., 0.)],
        Complex::new(1., 0.),
    )?;
    for band in [IirBand::Highpass(2.), IirBand::Bandstop(1., 3.)] {
        let t = p.transform(band)?;
        for w in [0.1, 0.5, 2., 4.] {
            let mapped = match band {
                IirBand::Highpass(c) => -c / w,
                _ => 2. * w / (3. - w * w),
            };
            close((t.response(w)? - p.response(mapped)?).norm(), 0., 1e-12);
        }
    }
    let complex = PoleZeroGain::new(
        IirDomain::Digital,
        &[Complex::new(0.1, 0.2), Complex::new(-0.4, -0.1)],
        &[
            Complex::new(0.2, 0.3),
            Complex::new(0.4, 0.),
            Complex::new(-0.3, 0.1),
        ],
        Complex::new(0.3, 0.2),
    )?;
    let sos = complex.to_sos()?;
    for f in [0., 0.2, 0.8] {
        close((complex.response(f)? - sos.response(f)?).norm(), 0., 1e-12);
    }
    let constant = PoleZeroGain::new(IirDomain::Analog, &[], &[], Complex::new(2., 0.))?;
    assert_eq!(constant.to_sos()?.response(0.)?, Complex::new(2., 0.));
    Ok(())
}
#[test]
fn iir_f32_narrowband_and_real_gammatone() -> Result<(), SciOxideError> {
    let d = design_iir(
        8,
        IirFamily::Butterworth,
        IirBand::Bandpass(0.2_f32, 0.21),
        IirDomain::Digital,
    )?;
    let sos = d
        .zpk()
        .to_sos()?
        .to_real(ResidualTolerance::new(1e-5, 1e-5)?)?;
    for f in [0.2, 0.21] {
        close(sos.response(f)?.norm() as f64, 0.5_f64.sqrt(), 0.003);
    }
    let g = design_real_gammatone(4, 0.3_f32, 0.05)?;
    close(g.response(0.3)?.norm() as f64, 1., 0.003);
    Ok(())
}
