//! Feature extraction, waveform, detrending and local-filter tests against
//! independent definitions and exact-rational fixtures, one module per API.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
#[macro_use]
#[path = "support/views.rs"]
mod views;
use errors::{error, error_kind};
use scioxide_core::{
    Complex, ExecutionContext, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideErrorKind,
    SciOxideErrorView,
};
use scioxide_signal::*;
fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
/// `a` matches the exact value `b` within 200 unit roundoffs of `max(|b|, 1)`.
fn close<S: RealScalar>(a: S, b: f64) {
    assert!(
        (a.into_f64() - b).abs() <= 200. * S::EPSILON.into_f64() * b.abs().max(1.),
        "{} != {b}",
        a.into_f64()
    );
}

mod analytic {
    use super::*;

    fn cosine<S: RealScalar>() {
        for n in [7, 8, 23] {
            let x: Vec<S> = (0..n)
                .map(|i| s::<S>(0.3) + s::<S>(0.7) * (S::TAU * s::<S>(i as f64 / n as f64)).cos())
                .collect();
            let p = AnalyticSignalNdPlan::new(&[n], &[0], SignalExecutionOptions::new()).unwrap();
            let mut scratch = p.workspace(()).unwrap();
            let mut z = vec![Complex::new(S::ZERO, S::ZERO); n];
            p.analytic_into(
                view!(&x),
                view_mut!(&mut z),
                &mut scratch,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
            let tolerance = s::<S>(150.0) * S::EPSILON;
            for (i, value) in z.iter().enumerate() {
                assert!((value.re - x[i]).abs() < tolerance);
                assert!(
                    (value.im - s::<S>(0.7) * (S::TAU * s::<S>(i as f64 / n as f64)).sin()).abs()
                        < tolerance
                );
            }
            let before = z.clone();
            assert_eq!(
                error_kind(p.analytic_into(
                    view!(&x),
                    view_mut!(&mut z),
                    &mut SignalWorkspace::new(),
                    ExecutionContext::sequential_native()
                )),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(z, before);
        }
    }
    #[test]
    fn cosine_quadrature_both_widths() {
        cosine::<f32>();
        cosine::<f64>();
    }

    fn direct_dft<S: RealScalar>() {
        for n in [1, 2, 5, 6, 11] {
            let x: Vec<S> = (0..n).map(|i| s(((i * 7 + 3) % 11) as f64 / 11.)).collect();
            let bins: Vec<(f64, f64)> = (0..n)
                .map(|k| {
                    x.iter().enumerate().fold((0., 0.), |(re, im), (j, x)| {
                        let a = std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        (re + x.into_f64() * a.cos(), im - x.into_f64() * a.sin())
                    })
                })
                .collect();
            let analytic = hilbert(&x).unwrap();
            let env = envelope(&x).unwrap();
            for (j, z) in analytic.iter().enumerate() {
                let (re, im) = bins
                    .iter()
                    .enumerate()
                    .fold((0., 0.), |(re, im), (k, &(r, i))| {
                        let weight = if k == 0 || (n % 2 == 0 && k == n / 2) {
                            1.
                        } else if k <= n / 2 {
                            2.
                        } else {
                            0.
                        };
                        let a = std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        (
                            re + weight * (r * a.cos() - i * a.sin()) / n as f64,
                            im + weight * (r * a.sin() + i * a.cos()) / n as f64,
                        )
                    });
                close(z.re, re);
                close(z.im, im);
                close(env[j], re.hypot(im));
            }
        }
    }
    #[test]
    fn direct_dft_independent_of_fft_planner() {
        direct_dft::<f32>();
        direct_dft::<f64>();
    }

    #[test]
    fn envelope_failures_preserve_outputs() {
        let plan = AnalyticSignalNdPlan::new(&[3], &[0], SignalExecutionOptions::new()).unwrap();
        let mut scratch = plan.workspace(()).unwrap();
        let mut out = [42.; 3];
        assert_eq!(
            error_kind(plan.envelope_into(
                view!(&[0., f64::NAN, 1.]),
                view_mut!(&mut out),
                &mut scratch,
                ExecutionContext::sequential_native()
            )),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(out, [42.; 3]);
        assert_eq!(
            error_kind(complex_envelope_into(
                &[Complex::new(f64::MAX, f64::MAX)],
                &mut out,
                ExecutionContext::sequential_native()
            )),
            SciOxideErrorKind::NumericalInstability
        );
        assert_eq!(out, [42.; 3]);
    }
}

mod peaks {
    use super::*;

    fn plateaus_and_distance<S: RealScalar>() {
        let x: Vec<S> = [0., 2., 2., 0., 2., 0., 3., 0.]
            .into_iter()
            .map(s)
            .collect();
        let mut out = vec![Peak::default(); 4];
        let mut peak_work = PeakWorkspace::new();
        peak_work
            .try_reserve(&PeakWorkspace::<S>::workspace_requirements(x.len()))
            .unwrap();
        let count = find_peaks_into(
            &x,
            PeakOptions::default(),
            &mut out,
            &mut peak_work,
            native(),
        )
        .unwrap();
        assert_eq!(count, 3);
        assert_eq!(
            out[..count].iter().map(|p| p.index()).collect::<Vec<_>>(),
            [1, 4, 6]
        );
        assert_eq!((out[0].plateau_start(), out[0].plateau_end()), (1, 2));
        assert_eq!(out[0].prominence(), s(2.));
        assert_eq!(out[0].width(), s(2.));
        let count = find_peaks_into(
            &x,
            PeakOptions::default().with_distance(4),
            &mut out,
            &mut peak_work,
            native(),
        )
        .unwrap();
        assert_eq!(
            out[..count].iter().map(|p| p.index()).collect::<Vec<_>>(),
            [1, 6]
        );
        let before = out.clone();
        assert_eq!(
            error_kind(peak_properties_into(&x, &[0], S::HALF, &mut out, native())),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(before, out);
        assert_eq!(
            error_kind(peak_properties_into(
                &x,
                &[usize::MAX],
                S::HALF,
                &mut out,
                native()
            )),
            SciOxideErrorKind::InvalidInput
        );
    }
    #[test]
    fn plateaus_and_distance_both_widths() {
        plateaus_and_distance::<f32>();
        plateaus_and_distance::<f64>();
    }

    #[test]
    fn contours_range_and_ties() {
        let x = [-f64::MAX, f64::MAX, 0.];
        let mut out = [Peak::default()];
        peak_properties_into(&x, &[1], 0.5, &mut out, native()).unwrap();
        close(out[0].left_crossing(), 0.75);
        close(out[0].right_crossing(), 1.5);
        close(out[0].width(), 0.75);
        let x = [0., 0., 2., 0., 2., 0., 0.];
        let mut out = [Peak::default(); 3];
        let mut work = PeakWorkspace::new();
        work.try_reserve(&PeakWorkspace::<f64>::workspace_requirements(x.len()))
            .unwrap();
        let n = find_peaks_into(
            &x,
            PeakOptions::default().with_distance(3),
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(out[0].index(), 2);
        assert_eq!(out[0].left_base(), 1);
        assert_eq!(out[0].right_base(), 3);
        assert_eq!(
            find_peaks_into(
                &[1., 1., 0., 2., 2.],
                PeakOptions::default(),
                &mut out,
                &mut work,
                native()
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn nonfinite_selection_parameters_report_input_origin() {
        let mut out = [Peak::default()];
        let mut work = PeakWorkspace::new();
        assert!(matches!(
            error(find_peaks_into(
                &[0., 1., 0.],
                PeakOptions::default().with_relative_height(f64::NAN),
                &mut out,
                &mut work,
                native()
            ))
            .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

mod waveforms {
    use super::*;

    fn chunks_and_sweeps<S: RealScalar>() {
        let families = [
            Waveform::Sine {
                frequency: s(1.),
                phase: S::ZERO,
            },
            Waveform::Cosine {
                frequency: s(1.),
                phase: S::ZERO,
            },
            Waveform::Square {
                frequency: s(1.),
                phase: S::ZERO,
                duty: S::HALF,
            },
            Waveform::Sawtooth {
                frequency: s(1.),
                phase: S::ZERO,
                width: S::HALF,
            },
            Waveform::GaussianPulse {
                frequency: s(1.),
                phase: S::ZERO,
                sigma: s(0.2),
            },
            Waveform::Chirp {
                initial: s(1.),
                final_frequency: s(3.),
                duration: s(1.),
                phase: S::ZERO,
                law: ChirpLaw::Linear,
            },
            Waveform::Chirp {
                initial: s(1.),
                final_frequency: s(3.),
                duration: s(1.),
                phase: S::ZERO,
                law: ChirpLaw::Quadratic {
                    vertex_at_end: false,
                },
            },
            Waveform::Chirp {
                initial: s(1.),
                final_frequency: s(3.),
                duration: s(1.),
                phase: S::ZERO,
                law: ChirpLaw::Logarithmic,
            },
            Waveform::Chirp {
                initial: s(1.),
                final_frequency: s(3.),
                duration: s(1.),
                phase: S::ZERO,
                law: ChirpLaw::Hyperbolic,
            },
        ];
        for w in families {
            let mut batch = vec![S::ZERO; 31];
            let mut chunks = batch.clone();
            waveform_grid_into(w, S::ZERO, s(0.01), 0, &mut batch, native()).unwrap();
            for (i, chunk) in chunks.chunks_mut(4).enumerate() {
                waveform_grid_into(w, S::ZERO, s(0.01), i * 4, chunk, native()).unwrap();
            }
            assert_eq!(batch, chunks);
        }
        let w = Waveform::Chirp {
            initial: s::<S>(2.),
            final_frequency: s(4.),
            duration: s(1.),
            phase: S::ZERO,
            law: ChirpLaw::Linear,
        };
        let coeff = [s(2.), s(2.)];
        let poly = Waveform::PolynomialSweep {
            coefficients: &coeff,
            phase: S::ZERO,
        };
        for t in [0., 0.125, 0.25, 0.5, 1.] {
            assert!(
                (w.sample(s(t)).unwrap() - poly.sample(s(t)).unwrap()).abs()
                    < s::<S>(20.) * S::EPSILON
            );
        }
        let mut impulse = vec![S::ZERO; 5];
        unit_impulse_into(2, 0, &mut impulse, native()).unwrap();
        assert_eq!(impulse, [S::ZERO, S::ZERO, S::ONE, S::ZERO, S::ZERO]);
    }
    #[test]
    fn chunked_grids_and_polynomial_sweeps_both_widths() {
        chunks_and_sweeps::<f32>();
        chunks_and_sweeps::<f64>();
    }

    fn spectra<S: RealScalar>() {
        let n = 64;
        let mut out = vec![S::ZERO; n];
        waveform_grid_into(
            Waveform::Square {
                frequency: S::ONE,
                phase: S::ZERO,
                duty: S::HALF,
            },
            S::ZERO,
            s(1. / 64.),
            0,
            &mut out,
            native(),
        )
        .unwrap();
        // Half-wave antisymmetry makes every even DFT bin vanish, including DC.
        for k in (0..n).step_by(2) {
            let (re, im) = out.iter().enumerate().fold((0., 0.), |(re, im), (j, x)| {
                let a = std::f64::consts::TAU * (k * j) as f64 / n as f64;
                (re + x.into_f64() * a.cos(), im + x.into_f64() * a.sin())
            });
            assert!(re.hypot(im) < 1e-10);
        }
        for (law, cycles) in [
            (ChirpLaw::Linear, 1.25),
            (
                ChirpLaw::Quadratic {
                    vertex_at_end: false,
                },
                1. + 1. / 12.,
            ),
            (
                ChirpLaw::Quadratic {
                    vertex_at_end: true,
                },
                1. + 5. / 12.,
            ),
            (
                ChirpLaw::Logarithmic,
                2. * ((2_f64).sqrt() - 1.) / 2_f64.ln(),
            ),
            (ChirpLaw::Hyperbolic, -4. * 0.75_f64.ln()),
        ] {
            let w = Waveform::Chirp {
                initial: s::<S>(2.),
                final_frequency: s(4.),
                duration: S::ONE,
                phase: S::ZERO,
                law,
            };
            close(
                w.sample(S::HALF).unwrap(),
                (std::f64::consts::TAU * cycles).cos(),
            );
        }
    }
    #[test]
    fn spectra_and_integrated_frequency_both_widths() {
        spectra::<f32>();
        spectra::<f64>();
    }

    #[test]
    fn tiny_negative_phase_respects_degenerate_fractions() {
        let t = -f64::from_bits(1);
        assert_eq!(
            Waveform::Square {
                frequency: 1.,
                phase: 0.,
                duty: 1.
            }
            .sample(t)
            .unwrap(),
            1.
        );
        assert_eq!(
            Waveform::Sawtooth {
                frequency: 1.,
                phase: 0.,
                width: 1.
            }
            .sample(t)
            .unwrap(),
            1.
        );
    }

    #[test]
    fn invalid_sweeps_and_grids_preserve_outputs() {
        let wave = Waveform::Chirp {
            initial: 1.,
            final_frequency: -1.,
            duration: 1.,
            phase: 0.,
            law: ChirpLaw::Logarithmic,
        };
        assert_eq!(error_kind(wave.sample(0.)), SciOxideErrorKind::DomainError);
        let mut out = [42.; 3];
        assert_eq!(
            error_kind(waveform_grid_into(
                Waveform::Sine {
                    frequency: 1.,
                    phase: 0.
                },
                0.,
                1.,
                usize::MAX,
                &mut out,
                native()
            )),
            SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(out, [42.; 3]);
    }
}

mod detrend {
    use super::*;

    fn rational_least_squares<S: RealScalar + NumericScalar<Real = S>>() {
        let x: Vec<S> = [1., 4., 2., 8.].into_iter().map(s).collect();
        let mut out = vec![s::<S>(0.); 4];
        detrend_into(&x, DetrendKind::Linear, &[], &mut out, native()).unwrap();
        for (&a, b) in out.iter().zip([0.1, 1.2, -2.7, 1.4]) {
            close(a, b);
        }
        detrend_into(&x, DetrendKind::Linear, &[1, 3], &mut out, native()).unwrap();
        for a in out {
            close(a, 0.);
        }
    }
    #[test]
    fn exact_rational_least_squares_both_widths() {
        rational_least_squares::<f32>();
        rational_least_squares::<f64>();
    }

    #[test]
    fn complex_lines_and_invalid_breakpoints() {
        let input: Vec<_> = (0..9)
            .map(|i| Complex::new(2. + 3. * i as f64, 7. - 2. * i as f64))
            .collect();
        let mut out = vec![Complex::new(0., 0.); 9];
        detrend_into(&input, DetrendKind::Linear, &[], &mut out, native()).unwrap();
        assert!(out.iter().all(|z| z.norm() < 1e-13));
        let before = out.clone();
        assert_eq!(
            error_kind(detrend_into(
                &input,
                DetrendKind::Linear,
                &[4, 4],
                &mut out,
                native()
            )),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(before, out);
    }

    #[test]
    fn extreme_mean_keeps_tiny_residual() {
        let t = 1e-300;
        let input = [f64::MAX, 3. * t, -f64::MAX];
        let mut out = [0.; 3];
        detrend_into(&input, DetrendKind::Constant, &[], &mut out, native()).unwrap();
        assert!((out[1] / t - 2.).abs() < 1e-14);
    }
}

mod vector_strength {
    use super::*;

    #[test]
    fn locked_and_uniform_phases() {
        let mut v = [VectorStrength::default(); 2];
        vector_strength_into(&[0., 1., 2., 3.], &[1., 4.], &mut v, native()).unwrap();
        assert!((v[0].strength() - 1_f64).abs() < 1e-15);
        assert!(v[1].strength() < 1e-15);
    }

    #[test]
    fn f32_and_extreme_remainders() {
        let mut out = [VectorStrength::default(); 2];
        vector_strength_into(&[0_f32, 0.25, 0.5, 0.75], &[1., 0.25], &mut out, native()).unwrap();
        close(out[0].strength(), 0.);
        close(out[1].strength(), 1.);
        let mut extreme = [VectorStrength::default()];
        vector_strength_into(&[f64::MAX], &[f64::MIN_POSITIVE], &mut extreme, native()).unwrap();
        close(extreme[0].strength(), 1.);
    }
}

mod local_filters {
    use super::*;

    fn chunked_streams<S: RealScalar>() {
        let x: Vec<S> = [3., 3., 3., 30., 3., 3., 2., 1., 0., -1., 0.]
            .into_iter()
            .map(s)
            .collect();
        let plans = [
            LocalFilterPlan::new(2, LocalFilter::Median).unwrap(),
            LocalFilterPlan::new(2, LocalFilter::Rank { rank: 0 }).unwrap(),
            LocalFilterPlan::new(2, LocalFilter::Hampel { threshold: s(3.) }).unwrap(),
            LocalFilterPlan::new(
                2,
                LocalFilter::Wiener {
                    noise_variance: s(1.),
                },
            )
            .unwrap(),
            LocalFilterPlan::order(&[-3, 0, 1], 1).unwrap(),
            LocalFilterPlan::order(&[1, 2, 3], 1).unwrap(),
            LocalFilterPlan::order(&[-3, -2, -1], 1).unwrap(),
        ];
        for plan in plans {
            let mut batch = vec![S::ZERO; x.len()];
            let mut scratch = plan.workspace(x.len()).unwrap();
            let mut indices = vec![0; x.len()];
            let changed = plan
                .filter_into(&x, &mut batch, &mut indices, &mut scratch, native())
                .unwrap();
            for chunk_len in 1..=x.len() + 1 {
                let mut stream = plan.stream().unwrap();
                let mut work = stream.workspace(chunk_len).unwrap();
                let mut result = Vec::new();
                let mut changes = Vec::new();
                for chunk in x.chunks(chunk_len).chain(core::iter::once(&[][..])) {
                    let finish = if chunk.is_empty() {
                        StreamChunk::Final
                    } else {
                        StreamChunk::Partial
                    };
                    let count = stream.output_len(chunk.len(), finish).unwrap();
                    let mut out = vec![S::ZERO; count];
                    let mut repl = vec![0; count];
                    let p = stream
                        .process_into(chunk, finish, &mut out, &mut repl, &mut work, native())
                        .unwrap();
                    assert_eq!(p.first_output_index(), result.len());
                    result.extend_from_slice(&out);
                    changes.extend_from_slice(&repl[..p.replacements()]);
                }
                assert_eq!(result, batch);
                assert_eq!(changes, &indices[..changed]);
            }
        }
    }
    #[test]
    fn chunked_streams_match_batch_both_widths() {
        chunked_streams::<f32>();
        chunked_streams::<f64>();
    }

    fn rational_filters<S: RealScalar>() {
        let x: Vec<S> = [1., 1., 9., 2., 2.].into_iter().map(s).collect();
        for (kind, expected) in [
            (LocalFilter::Median, [1., 1., 2., 2., 2.]),
            (
                LocalFilter::Hampel { threshold: s(3.) },
                [1., 1., 2., 2., 2.],
            ),
            (
                LocalFilter::Wiener {
                    noise_variance: s(1.),
                },
                [1., 19. / 16., 327. / 38., 31. / 14., 2.],
            ),
        ] {
            let plan = LocalFilterPlan::new(1, kind).unwrap();
            let mut out = [S::ZERO; 5];
            let mut indices = [0; 5];
            let mut work = plan.workspace(5).unwrap();
            plan.filter_into(&x, &mut out, &mut indices, &mut work, native())
                .unwrap();
            for (&a, b) in out.iter().zip(expected) {
                close(a, b);
            }
            let constant = [s::<S>(7.); 5];
            plan.filter_into(&constant, &mut out, &mut indices, &mut work, native())
                .unwrap();
            assert_eq!(out, constant);
        }
        close(estimate_wiener_noise(&x, 1).unwrap(), 68. / 9.);
    }
    #[test]
    fn rational_moments_and_order_statistics_both_widths() {
        rational_filters::<f32>();
        rational_filters::<f64>();
    }

    #[test]
    fn stream_failures_preserve_state_and_outputs() {
        let plan = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 3. }).unwrap();
        let mut stream = plan.stream().unwrap();
        let x = [1., 10., 1.];
        let mut work = stream.workspace(3).unwrap();
        let mut output = [42.; 3];
        assert_eq!(
            error_kind(stream.process_into(
                &x,
                StreamChunk::Final,
                &mut output,
                &mut [],
                &mut work,
                native()
            )),
            SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, [42.; 3]);
        assert_eq!(stream.output_len(3, StreamChunk::Final).unwrap(), 3);
        let mut indices = [99; 3];
        let p = stream
            .process_into(
                &x,
                StreamChunk::Final,
                &mut output,
                &mut indices,
                &mut work,
                native(),
            )
            .unwrap();
        assert_eq!(output, [1.; 3]);
        assert_eq!(p.replacements(), 1);
        assert_eq!(indices[0], 1);
        assert_eq!(
            error_kind(stream.output_len(0, StreamChunk::Final)),
            SciOxideErrorKind::InvalidInput
        );
        stream.reset();
        assert_eq!(stream.output_len(3, StreamChunk::Final).unwrap(), 3);
    }

    #[test]
    fn mixed_magnitude_outliers_do_not_erase_small_deviations() {
        let t = 1e-300;
        let input = [0., 2. * t, t, 3. * t, f64::MAX];
        let plan = LocalFilterPlan::new(2, LocalFilter::Hampel { threshold: 3. }).unwrap();
        let mut out = [0.; 5];
        let mut indices = [0; 5];
        let mut work = plan.workspace(5).unwrap();
        plan.filter_into(&input, &mut out, &mut indices, &mut work, native())
            .unwrap();
        assert_eq!(out[2], t);
        let tiny = 1e-30_f32;
        let input = [0., 2. * tiny, tiny, 3. * tiny, f32::MAX];
        let plan = LocalFilterPlan::new(2, LocalFilter::Hampel { threshold: 3. }).unwrap();
        let mut out = [0.; 5];
        let mut work = plan.workspace(5).unwrap();
        plan.filter_into(&input, &mut out, &mut indices, &mut work, native())
            .unwrap();
        assert_eq!(out[2], tiny);
    }

    #[test]
    fn invalid_parameters() {
        assert_eq!(
            error_kind(LocalFilterPlan::new(usize::MAX, LocalFilter::<f64>::Median)),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            error_kind(LocalFilterPlan::<f64>::order(&[], 0)),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            error_kind(LocalFilterPlan::<f64>::order(&[0, 0], 0)),
            SciOxideErrorKind::InvalidInput
        );
        assert!(matches!(
            error(LocalFilterPlan::new(
                1,
                LocalFilter::Wiener {
                    noise_variance: f64::INFINITY
                }
            ))
            .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert!(matches!(
            error(LocalFilterPlan::new(
                1,
                LocalFilter::Hampel {
                    threshold: f64::NAN
                }
            ))
            .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

mod ridges {
    use super::*;

    fn ricker<S: RealScalar>() {
        let x: Vec<S> = (0..65)
            .map(|i| {
                let t = s::<S>(i as f64 - 32.);
                (-t * t / s::<S>(18.)).exp()
            })
            .collect();
        let scales = [s(1.), s(2.), s(3.), s(4.)];
        let options = RidgeOptions::default()
            .with_support_radius(20)
            .with_max_distance(2)
            .with_max_gap(0)
            .with_min_length(4)
            .with_min_strength(S::ZERO);
        let mut out = vec![RidgePeak::default(); 32];
        let mut workspace = RidgePlan::new(&scales, options)
            .unwrap()
            .workspace(65)
            .unwrap();
        let count = RidgePlan::new(&scales, options)
            .and_then(|plan| plan.find_peaks_into(&x, &mut out, &mut workspace, native()))
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(out[0].index(), 32);
        assert_eq!(out[0].length(), 4);
        assert_eq!(out[0].scale_index(), 0);
        let before = out.clone();
        assert_eq!(
            error_kind(
                RidgePlan::new(&[S::ZERO], options).and_then(|plan| plan.find_peaks_into(
                    &x,
                    &mut out,
                    &mut workspace,
                    native()
                ))
            ),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(before, out);
    }
    #[test]
    fn ricker_ridges_both_widths() {
        ricker::<f32>();
        ricker::<f64>();
    }

    #[test]
    fn separated_peaks_and_deterministic_matching() {
        let x: Vec<f64> = (0..65)
            .map(|i| {
                let x = i as f64;
                (-((x - 20.) / 2.).powi(2)).exp() + (-((x - 44.) / 2.).powi(2)).exp()
            })
            .collect();
        let scales = [1., 2., 3.];
        let options = RidgeOptions::default()
            .with_support_radius(12)
            .with_max_distance(2)
            .with_max_gap(1)
            .with_min_length(3)
            .with_min_strength(0.05);
        let mut out = [RidgePeak::default(); 32];
        let mut workspace = RidgeWorkspace::new();
        let count = RidgePlan::new(&scales, options)
            .and_then(|plan| {
                workspace.try_reserve(&plan.workspace_requirements(x.len())?)?;
                plan.find_peaks_into(&x, &mut out, &mut workspace, native())
            })
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!([out[0].index(), out[1].index()], [20, 44]);
        let before = out;
        let again = RidgePlan::new(&scales, options)
            .and_then(|plan| plan.find_peaks_into(&x, &mut out, &mut workspace, native()))
            .unwrap();
        assert_eq!(again, count);
        assert_eq!(before, out);
        assert_eq!(
            error_kind(
                RidgePlan::new(&scales, options).and_then(|plan| plan.find_peaks_into(
                    &x,
                    &mut [],
                    &mut workspace,
                    native()
                ))
            ),
            SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn remote_outlier_does_not_remove_a_tiny_ridge() {
        let mut x = [0_f64; 41];
        x[2] = 1e300;
        x[30] = 1e-300;
        let mut out = [RidgePeak::default(); 20];
        let count = RidgePlan::new(
            &[1.],
            RidgeOptions::default()
                .with_support_radius(3)
                .with_max_distance(1)
                .with_max_gap(0)
                .with_min_length(1)
                .with_min_strength(0.),
        )
        .and_then(|plan| {
            let mut workspace = plan.workspace(x.len())?;
            plan.find_peaks_into(&x, &mut out, &mut workspace, native())
        })
        .unwrap();
        assert!(
            out[..count]
                .iter()
                .any(|p| p.index() == 30 && p.strength() > 0.)
        );
    }

    #[test]
    fn workspace_size_overflow() {
        assert_eq!(
            error_kind(
                RidgePlan::new(
                    &[1.0, 2.0, 3.0, 4.0],
                    RidgeOptions::default()
                        .with_support_radius(1)
                        .with_max_distance(1)
                        .with_max_gap(0)
                        .with_min_length(1)
                        .with_min_strength(0.0)
                )
                .unwrap()
                .workspace_requirements(usize::MAX)
            ),
            SciOxideErrorKind::SizeOverflow
        );
    }
}
