//! One-shot forms: peaks, detrending, smoothing, local filters and test signals.
use super::{argument_and_reason, error_kind, identical};
use scioxide_core::{
    Complex, DomainReason, ExecutionContext, Interval, InvalidInputReason, NumericScalar,
    SciOxideErrorKind, SciOxideErrorView, argument,
};
use scioxide_signal::{
    ChirpLaw, DetrendKind, LocalFilter, LocalFilterPlan, Peak, PeakOptions, PeakWorkspace,
    RidgeOptions, RidgePeak, RidgePlan, SavitzkyGolayBoundary, SavitzkyGolayOptions,
    SavitzkyGolayPlan, VectorStrength, Waveform, detrend, detrend_into, find_peaks,
    find_peaks_into, find_peaks_with_options, median_filter, peak_properties, peak_properties_into,
    savitzky_golay, vector_strength, vector_strength_into, waveform, waveform_into,
};

fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

fn real(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (0.61 * k as f64).sin() + 0.4 * (2.3 * k as f64).cos() + 0.01 * k as f64)
        .collect()
}

fn complex(n: usize) -> Vec<Complex<f64>> {
    real(n + 2)
        .windows(3)
        .map(|w| Complex::new(w[0], w[2] - w[1]))
        .collect()
}

mod peaks {
    use super::*;

    fn into(input: &[f64], options: PeakOptions<f64>) -> Vec<Peak<f64>> {
        let mut workspace = PeakWorkspace::new();
        workspace
            .try_reserve(&PeakWorkspace::<f64>::workspace_requirements(input.len()))
            .unwrap();
        let mut output = vec![Peak::default(); input.len()];
        let count = find_peaks_into(input, options, &mut output, &mut workspace, native()).unwrap();
        output.truncate(count);
        output
    }

    #[test]
    fn find_peaks_equal_the_workspace_form() {
        let filtered = PeakOptions::new()
            .with_prominence(Interval::new(0.3, 10.0).unwrap())
            .with_relative_height(0.25)
            .with_distance(4);
        for n in [0, 1, 2, 3, 64, 257] {
            let x = real(n);
            let all = find_peaks(&x).unwrap();
            identical(&all, &into(&x, PeakOptions::default()));
            assert!(n >= 3 || all.is_empty());
            assert!(n < 64 || all.len() > 5);
            identical(
                &find_peaks_with_options(&x, filtered).unwrap(),
                &into(&x, filtered),
            );
            let indices: Vec<_> = all.iter().map(|peak| peak.index()).collect();
            let mut expected = vec![Peak::default(); indices.len()];
            peak_properties_into(&x, &indices, 0.75, &mut expected, native()).unwrap();
            identical(&peak_properties(&x, &indices, 0.75).unwrap(), &expected);
        }
    }

    #[test]
    fn find_peaks_errors() {
        let x = [0.0, 1.0, 0.0];
        let options = PeakOptions::new().with_relative_height(1.5);
        assert_eq!(
            argument_and_reason(find_peaks_with_options(&x, options)),
            (
                argument!("options.relative_height"),
                InvalidInputReason::OutOfRange
            )
        );
        let options = PeakOptions::new().with_plateau_size((0, 2));
        assert_eq!(
            argument_and_reason(find_peaks_with_options(&x, options)),
            (
                argument!("options.plateau_size"),
                InvalidInputReason::OutOfRange
            )
        );
        assert_eq!(
            error_kind(find_peaks(&[0.0, f64::NAN, 0.0])),
            SciOxideErrorKind::NonFinite
        );
        // The prominence MAX − (−MAX) overflows.
        let tall = [-f64::MAX, f64::MAX, -f64::MAX];
        assert_eq!(
            error_kind(find_peaks(&tall)),
            SciOxideErrorKind::NumericalInstability
        );
        assert_eq!(
            error_kind(peak_properties(&tall, &[1], 0.5)),
            SciOxideErrorKind::NumericalInstability
        );
    }

    #[test]
    fn peak_properties_errors() {
        let x = [0.0, 2.0, 4.0, 2.0, 0.0];
        assert_eq!(
            argument_and_reason(peak_properties(&x, &[1], 0.5)),
            (argument!("indices"), InvalidInputReason::Inconsistent)
        );
        assert_eq!(
            argument_and_reason(peak_properties(&x, &[0], 0.5)),
            (argument!("indices"), InvalidInputReason::OutOfRange)
        );
        assert_eq!(
            argument_and_reason(peak_properties(&x, &[2], -0.5)),
            (argument!("relative_height"), InvalidInputReason::OutOfRange)
        );
        assert_eq!(
            error_kind(peak_properties(&x, &[2], f64::NAN)),
            SciOxideErrorKind::NonFinite
        );
        assert!(peak_properties(&x, &[], 0.5).unwrap().is_empty());
    }

    #[test]
    fn ridge_find_peaks_equals_the_workspace_form() {
        let options = RidgeOptions::default().with_support_radius(6);
        let plan = RidgePlan::new(&[1.0, 2.0, 4.0], options).unwrap();
        for n in [0, 1, 5, 80] {
            let x = real(n);
            let mut workspace = plan.workspace(n).unwrap();
            let mut output = vec![RidgePeak::default(); n];
            let count = plan
                .find_peaks_into(&x, &mut output, &mut workspace, native())
                .unwrap();
            output.truncate(count);
            identical(&plan.find_peaks(&x).unwrap(), &output);
        }
        assert_eq!(
            error_kind(plan.find_peaks(&[0.0, f64::INFINITY, 0.0])),
            SciOxideErrorKind::NonFinite
        );
        // A narrow kernel has a peak value above one, so the coefficient of a
        // spike of the largest finite height overflows.
        let narrow = RidgePlan::new(&[0.1], options).unwrap();
        assert_eq!(
            error_kind(narrow.find_peaks(&[0.0, f64::MAX, 0.0])),
            SciOxideErrorKind::NumericalInstability
        );
    }
}

mod detrending {
    use super::*;

    fn into<T: NumericScalar>(input: &[T], kind: DetrendKind) -> Vec<T> {
        let mut output = vec![T::ZERO; input.len()];
        detrend_into(input, kind, &[], &mut output, native()).unwrap();
        output
    }

    #[test]
    fn detrend_equals_the_unsegmented_form() {
        for kind in [DetrendKind::Constant, DetrendKind::Linear] {
            for n in [0, 1, 2, 31] {
                let x = real(n);
                identical(&detrend(&x, kind).unwrap(), &into(&x, kind));
                let z = complex(n);
                identical(&detrend(&z, kind).unwrap(), &into(&z, kind));
            }
        }
        assert!(detrend::<f64>(&[], DetrendKind::Linear).unwrap().is_empty());
    }

    #[test]
    fn detrend_errors() {
        assert_eq!(
            error_kind(detrend(&[1.0, f64::NAN], DetrendKind::Constant)),
            SciOxideErrorKind::NonFinite
        );
        // The line through ±MAX has residuals of 4·MAX/3.
        assert_eq!(
            error_kind(detrend(
                &[f64::MAX, -f64::MAX, f64::MAX],
                DetrendKind::Linear
            )),
            SciOxideErrorKind::NumericalInstability
        );
    }
}

mod smoothing {
    use super::*;

    fn plan<T: NumericScalar>(input: &[T], window: usize, degree: usize) -> Vec<T> {
        SavitzkyGolayPlan::<T::Real>::new(window, degree, SavitzkyGolayOptions::default())
            .unwrap()
            .filter(input, SavitzkyGolayBoundary::Polynomial)
            .unwrap()
    }

    #[test]
    fn savitzky_golay_equals_the_polynomial_boundary_plan() {
        for (window, degree) in [(1, 0), (5, 2), (7, 3), (11, 4)] {
            for n in [0, window, 40] {
                let x = real(n);
                identical(
                    &savitzky_golay(&x, window, degree).unwrap(),
                    &plan(&x, window, degree),
                );
                let z = complex(n);
                identical(
                    &savitzky_golay(&z, window, degree).unwrap(),
                    &plan(&z, window, degree),
                );
            }
        }
        assert!(savitzky_golay::<f64>(&[], 7, 2).unwrap().is_empty());
    }

    #[test]
    fn savitzky_golay_errors() {
        let x = real(9);
        assert_eq!(
            argument_and_reason(savitzky_golay(&x, 0, 0)),
            (argument!("window_length"), InvalidInputReason::Empty)
        );
        assert_eq!(
            argument_and_reason(savitzky_golay(&x, 5, 5)),
            (
                argument!("polynomial_degree"),
                InvalidInputReason::OutOfRange
            )
        );
        assert_eq!(
            argument_and_reason(savitzky_golay(&x, 4, 2)),
            (
                argument!("window_length"),
                InvalidInputReason::UnsupportedConfiguration
            )
        );
        // A nonempty record shorter than the window has no complete window to fit.
        assert_eq!(
            argument_and_reason(savitzky_golay(&x[..4], 5, 2)),
            (argument!("input"), InvalidInputReason::Inconsistent)
        );
        assert_eq!(
            error_kind(savitzky_golay(&[0.0, 1.0, f64::NAN], 3, 1)),
            SciOxideErrorKind::NonFinite
        );
        // The weights (−3, 12, 17, 12, −3)/35 have absolute sum 47/35.
        let m = f64::MAX;
        assert_eq!(
            error_kind(savitzky_golay(&[-m, m, m, m, -m], 5, 2)),
            SciOxideErrorKind::NumericalInstability
        );
    }
}

mod local_filters {
    use super::*;

    fn into(plan: &LocalFilterPlan<f64>, input: &[f64]) -> (Vec<f64>, usize) {
        let mut output = vec![0.0; input.len()];
        let mut replacements = vec![0; input.len()];
        let mut workspace = plan.workspace(input.len()).unwrap();
        let replaced = plan
            .filter_into(
                input,
                &mut output,
                &mut replacements,
                &mut workspace,
                native(),
            )
            .unwrap();
        (output, replaced)
    }

    #[test]
    fn median_filter_equals_the_median_plan() {
        for radius in [0, 1, 3, 40] {
            for n in [0, 1, 9, 50] {
                let x = real(n);
                let plan = LocalFilterPlan::new(radius, LocalFilter::Median).unwrap();
                identical(&median_filter(&x, radius).unwrap(), &into(&plan, &x).0);
            }
        }
        // Radius zero is the identity.
        let x = real(12);
        identical(&median_filter(&x, 0).unwrap(), &x);
        assert!(median_filter::<f64>(&[], 2).unwrap().is_empty());
    }

    #[test]
    fn local_filter_plan_filter_equals_filter_into() {
        let mut x = real(60);
        x[17] = 40.0;
        x[41] = -35.0;
        for kind in [
            LocalFilter::Median,
            LocalFilter::Rank { rank: 1 },
            LocalFilter::Wiener {
                noise_variance: 0.05,
            },
            LocalFilter::Hampel { threshold: 2.0 },
        ] {
            let plan = LocalFilterPlan::new(2, kind).unwrap();
            let (expected, replaced) = into(&plan, &x);
            identical(&plan.filter(&x).unwrap(), &expected);
            if matches!(kind, LocalFilter::Hampel { .. }) {
                assert!(replaced >= 2);
            }
        }
        let rank = LocalFilterPlan::order(&[-3, -1, 2], 2).unwrap();
        identical(&rank.filter(&x).unwrap(), &into(&rank, &x).0);
    }

    #[test]
    fn local_filter_errors() {
        assert_eq!(
            argument_and_reason(median_filter(&[1.0, 2.0], usize::MAX)),
            (argument!("radius"), InvalidInputReason::OutOfRange)
        );
        assert_eq!(
            error_kind(median_filter(&[1.0, f64::NAN, 2.0], 1)),
            SciOxideErrorKind::NonFinite
        );
        let plan = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 1.0 }).unwrap();
        assert_eq!(
            error_kind(plan.filter(&[1.0, f64::NEG_INFINITY])),
            SciOxideErrorKind::NonFinite
        );
    }
}

mod phase {
    use super::*;

    #[test]
    fn vector_strength_equals_the_single_period_form() {
        let events: Vec<f64> = real(40).iter().map(|x| 3.0 * x.abs()).collect();
        for period in [0.1, 0.7, 1.0, 13.0] {
            let mut expected = [VectorStrength::default()];
            vector_strength_into(&events, &[period], &mut expected, native()).unwrap();
            identical(&vector_strength(&events, period).unwrap(), &expected[0]);
        }
    }

    #[test]
    fn vector_strength_errors_name_its_own_arguments() {
        assert_eq!(
            argument_and_reason(vector_strength(&[], 1.0)),
            (argument!("events"), InvalidInputReason::Empty)
        );
        for period in [0.0, -0.0, -2.0, -f64::MAX] {
            assert_eq!(
                argument_and_reason(vector_strength(&[0.5], period)),
                (argument!("period"), InvalidInputReason::OutOfRange)
            );
        }
        for period in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert_eq!(
                error_kind(vector_strength(&[0.5], period)),
                SciOxideErrorKind::NonFinite
            );
        }
        assert_eq!(
            error_kind(vector_strength(&[0.5, f64::NAN], 1.0)),
            SciOxideErrorKind::NonFinite
        );
    }
}

mod waveforms {
    use super::*;

    #[test]
    fn waveform_equals_waveform_into() {
        let times: Vec<f64> = (0..33).map(|k| 0.03 * k as f64 - 0.2).collect();
        let coefficients = [0.5, -1.0, 2.0];
        for shape in [
            Waveform::Sine {
                frequency: 1.5,
                phase: 0.3,
            },
            Waveform::Square {
                frequency: 2.0,
                phase: 0.0,
                duty: 0.3,
            },
            Waveform::Chirp {
                initial: 1.0,
                final_frequency: 4.0,
                duration: 1.0,
                law: ChirpLaw::Logarithmic,
                phase: 0.0,
            },
            Waveform::PolynomialSweep {
                coefficients: &coefficients,
                phase: 0.1,
            },
        ] {
            let mut expected = vec![0.0; times.len()];
            waveform_into(shape, &times, &mut expected, native()).unwrap();
            identical(&waveform(shape, &times).unwrap(), &expected);
        }
        let sine = Waveform::Sine {
            frequency: 1.0,
            phase: 0.0,
        };
        assert!(waveform(sine, &[]).unwrap().is_empty());
    }

    #[test]
    fn waveform_errors() {
        let sine = Waveform::Sine {
            frequency: 1.0,
            phase: 0.0,
        };
        assert_eq!(
            error_kind(waveform(sine, &[0.0, f64::NAN])),
            SciOxideErrorKind::NonFinite
        );
        let square = Waveform::Square {
            frequency: 1.0,
            phase: 0.0,
            duty: 1.5,
        };
        assert_eq!(
            argument_and_reason(waveform(square, &[0.0])),
            (argument!("waveform"), InvalidInputReason::OutOfRange)
        );
        // The hyperbolic law 1/(1 − t/2) has its pole at t = 2.
        let hyperbolic = Waveform::Chirp {
            initial: 1.0,
            final_frequency: 2.0,
            duration: 1.0,
            law: ChirpLaw::Hyperbolic,
            phase: 0.0,
        };
        match waveform(hyperbolic, &[0.5, 2.5]).unwrap_err().view() {
            SciOxideErrorView::DomainError {
                argument, reason, ..
            } => assert_eq!(
                (argument, reason),
                (argument!("waveform"), DomainReason::Pole)
            ),
            other => unreachable!("{other:?}"),
        }
        let fast = Waveform::Sine {
            frequency: f64::MAX,
            phase: 0.0,
        };
        assert_eq!(
            error_kind(waveform(fast, &[4.0])),
            SciOxideErrorKind::NumericalInstability
        );
    }
}
