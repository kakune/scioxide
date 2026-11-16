use super::*;
use crate::test_support::{assert_complex_close, assert_real_close};
use crate::{HermitianSpectrum, Normalization, irfft, rfft};

/// Prepare a sequential plan, the public default policy, on the
/// automatically selected backend.
fn prepared<S: scioxide_core::RealScalar>(n: usize) -> Result<RealFftPlan<S>, SciOxideError> {
    RealFftPlan::with_context(
        n,
        ExecutionPolicy::Sequential,
        ExecutionContext::sequential_native(),
        scioxide_core::argument!("n"),
        scioxide_core::argument!("context"),
    )
}

/// Zeroed scratch of the length a plan requires.
fn workspace<S: scioxide_core::RealScalar>(plan: &RealFftPlan<S>) -> Vec<Complex<S>> {
    vec![Complex::new(S::ZERO, S::ZERO); plan.lane_scratch_len()]
}

use scioxide_core::RealScalar;

fn axis_zero<'a>(shape: &'a [usize], strides: &'a [usize]) -> crate::lanes::Placement<'a> {
    crate::lanes::Placement {
        base: 0,
        geometry: crate::lanes::LaneGeometry::new::<Complex<f64>>(shape, strides, 0),
        shape,
        strides,
    }
}

#[test]
fn bounded_unscaled_children_match_checked_transforms() {
    let cx = ExecutionContext::sequential_native();
    for n in [1, 5, 8, 15, 45] {
        let plan = prepared::<f64>(n).unwrap();
        let input: Vec<_> = (0..n)
            .map(|i| (i as f64 + 1.0) / (8 * n * n) as f64)
            .collect();
        let m = plan.spectrum_len();
        let mut expected = vec![Complex::new(0.0, 0.0); m];
        let mut scratch = workspace(&plan);
        plan.forward_into(
            1.0,
            VectorView::from_slice(&input),
            VectorViewMut::from_slice(&mut expected),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        let mut expected_real = vec![0.0; n];
        plan.inverse_into(
            1.0,
            VectorView::from_slice(&expected),
            VectorViewMut::from_slice(&mut expected_real),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        // Three interleaved lanes on axis zero exercise strided sources
        // and staged targets.
        let lanes = 3;
        let samples: Vec<_> = input.iter().flat_map(|&v| [v; 3]).collect();
        let mut bins = vec![Complex::new(f64::NAN, 0.0); m * lanes];
        let mut recovered = vec![Complex::new(f64::NAN, 0.0); n * lanes];
        let (full, packed, strides) = ([n, lanes], [m, lanes], [lanes, 1]);
        let place = |shape| axis_zero(shape, &strides);
        let mut staging = vec![
            Complex::new(0.0, 0.0);
            place(&full)
                .geometry
                .staging_len()
                .max(place(&packed).geometry.staging_len())
        ];
        plan.forward_lanes(
            LanePairs::new(
                (&samples, place(&full)),
                (&mut bins, place(&packed)),
                &mut staging,
            ),
            &mut scratch,
            LaneMode::Bounded,
            (BinaryScale::Identity, 1.0),
            cx,
        )
        .unwrap();
        for lane in 0..lanes {
            let actual: Vec<_> = bins.iter().skip(lane).step_by(lanes).copied().collect();
            assert_complex_close(&actual, &expected, Complex::norm, |_| 1e-12);
        }
        plan.inverse_lanes(
            LanePairs::new(
                (&bins, place(&packed)),
                (&mut recovered, place(&full)),
                &mut staging,
            ),
            &mut scratch,
            LaneMode::Bounded,
            (1.0, 1.0),
            cx,
        )
        .unwrap();
        for lane in 0..lanes {
            let actual: Vec<_> = recovered
                .iter()
                .skip(lane)
                .step_by(lanes)
                .map(|z| z.re)
                .collect();
            assert_real_close(&actual, &expected_real, |_| 1e-12);
        }
    }
}

fn bounded_lane_contract<S: RealScalar>() {
    let cx = ExecutionContext::sequential_native();
    for n in [1, 5, 8, 15, 35, 45] {
        for blocks in [false, true] {
            let mut plan = prepared::<S>(n).unwrap();
            if !blocks {
                plan.blocks = None;
            }
            let m = plan.spectrum_len();
            let lanes = 3;
            let (full, packed, strides) = ([n, lanes], [m, lanes], [lanes, 1]);
            let place = |shape| crate::lanes::Placement {
                base: 0,
                geometry: crate::lanes::LaneGeometry::new::<Complex<S>>(shape, &strides, 0),
                shape,
                strides: &strides,
            };
            let mut staging = vec![
                Complex::new(S::ZERO, S::ZERO);
                place(&full)
                    .geometry
                    .staging_len()
                    .max(place(&packed).geometry.staging_len())
            ];
            let mut scratch = workspace(&plan);
            let mut output = vec![Complex::new(S::ZERO, S::ZERO); n * lanes];
            for k in [0, m / 2, m - 1] {
                for bad in [
                    S::NAN,
                    S::INFINITY,
                    -S::INFINITY,
                    S::ONE + S::EPSILON,
                    -S::ONE - S::EPSILON,
                ] {
                    let mut bins = vec![Complex::new(S::HALF, S::ZERO); m * lanes];
                    bins[k * lanes + lanes - 1].re = bad;
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let _ = plan.inverse_lanes(
                                LanePairs::new(
                                    (&bins, place(&packed)),
                                    (&mut output, place(&full)),
                                    &mut staging,
                                ),
                                &mut scratch,
                                LaneMode::Bounded,
                                (S::ONE, S::ONE),
                                cx,
                            );
                        }))
                        .is_err(),
                        "inverse n={n}, k={k}, blocks={blocks}"
                    );
                }
            }
            for k in [0, m - 1]
                .into_iter()
                .filter(|&k| k == 0 || n.is_multiple_of(2))
            {
                let mut bins = vec![Complex::new(S::HALF, S::ZERO); m * lanes];
                bins[k * lanes + lanes - 1].im = S::HALF;
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = plan.inverse_lanes(
                            LanePairs::new(
                                (&bins, place(&packed)),
                                (&mut output, place(&full)),
                                &mut staging,
                            ),
                            &mut scratch,
                            LaneMode::Bounded,
                            (S::ONE, S::ONE),
                            cx,
                        );
                    }))
                    .is_err(),
                    "endpoint n={n}, k={k}, blocks={blocks}"
                );
            }
            for bad in [S::NAN, S::INFINITY, S::ONE + S::EPSILON] {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = plan.forward_bounded(
                            &mut scratch,
                            |values| {
                                values.fill(Complex::new(S::HALF, S::ZERO));
                                values[0].re = bad;
                            },
                            cx,
                        );
                    }))
                    .is_err(),
                    "forward n={n}"
                );
            }
        }
    }
}

#[test]
fn bounded_real_entries_reject_nonfinite_excess_and_unprojected_endpoints() {
    bounded_lane_contract::<f32>();
    bounded_lane_contract::<f64>();
}

#[test]
fn zero_is_rejected_and_one_is_exact() {
    let cx = ExecutionContext::sequential_native();
    let error = prepared::<f64>(0).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    let plan = prepared::<f64>(1).unwrap();
    assert_eq!(plan.spectrum_len(), 1);
    assert_eq!(plan.scratch_len(), 1);
    let forward = |spectrum: &mut [Complex<f64>], scratch: &mut [Complex<f64>]| {
        let input = VectorView::from_slice(&[5.0]);
        let output = VectorViewMut::from_slice(spectrum);
        plan.forward_into(1.0, input, output, false, scratch, cx)
    };
    let mut spectrum = [Complex::new(0.0, 0.0)];
    assert!(forward(&mut spectrum, &mut []).is_err());
    assert_eq!(spectrum, [Complex::new(0.0, 0.0)]);
    forward(&mut spectrum, &mut workspace(&plan)).unwrap();
    assert_eq!(spectrum, [Complex::new(5.0, 0.0)]);

    let inverse = |output: &mut [f64], scratch: &mut [Complex<f64>]| {
        let input = VectorView::from_slice(&spectrum);
        let output = VectorViewMut::from_slice(output);
        plan.inverse_into(1.0, input, output, false, scratch, cx)
    };
    let mut output = [0.0];
    assert!(inverse(&mut output, &mut []).is_err());
    assert_eq!(output, [0.0]);
    inverse(&mut output, &mut workspace(&plan)).unwrap();
    assert_eq!(output, [5.0]);
}

fn decimated_cases<S>(tolerance: f64)
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let cx = ExecutionContext::sequential_native();
    // Radices 3, 5 and 7, mixed chains, prime leaves 11 and 1009, and
    // the promoted primes 3, 5 and 7 themselves.
    for n in [
        3, 5, 7, 9, 15, 21, 25, 33, 35, 45, 49, 63, 77, 105, 125, 243, 343, 3027, 3375,
    ] {
        let input: Vec<S> = (0..n)
            .map(|i| S::try_from_f64_lossy(((i * 37 % 101) as f64 - 50.0) / 64.0).unwrap())
            .collect();
        let complex_input: Vec<_> = input.iter().map(|&x| Complex::new(x, S::ZERO)).collect();
        let expected = crate::fft(&complex_input, Normalization::Inverse).unwrap();
        let plan = prepared::<S>(n).unwrap();
        let (_, inverse) = Normalization::Inverse.fourier_scales::<S>(n).unwrap();
        assert_eq!(plan.is_packed(), decimation::radix(n).is_some(), "n={n}");
        let mut scratch = workspace(&plan);
        let mut spectrum = vec![Complex::new(S::TEN, S::TEN); plan.spectrum_len()];
        plan.forward_into(
            S::ONE,
            VectorView::from_slice(&input),
            VectorViewMut::from_slice(&mut spectrum),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        let bound = tolerance * (n as f64).sqrt();
        assert_complex_close(
            &spectrum,
            &expected[..plan.spectrum_len()],
            Complex::norm,
            |_| bound,
        );
        assert_eq!(spectrum[0].im, S::ZERO);
        // Strided views take the iterator paths.
        let strided_input: Vec<S> = input.iter().flat_map(|&x| [x, S::TEN]).collect();
        let mut strided = vec![Complex::new(S::TEN, S::TEN); 2 * plan.spectrum_len()];
        plan.forward_into(
            S::ONE,
            VectorView::from_slice_with_stride(&strided_input, n, 2).unwrap(),
            VectorViewMut::from_slice_with_stride(&mut strided, plan.spectrum_len(), 2).unwrap(),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        let strided: Vec<_> = strided.iter().step_by(2).copied().collect();
        assert_eq!(strided, spectrum);
        let mut output = vec![S::TEN; n];
        plan.inverse_into(
            inverse,
            VectorView::from_slice(&spectrum),
            VectorViewMut::from_slice(&mut output),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        assert_real_close(&output, &input, |_| tolerance);
        let mut strided = vec![S::TEN; 2 * n];
        plan.inverse_into(
            inverse,
            VectorView::from_slice(&spectrum),
            VectorViewMut::from_slice_with_stride(&mut strided, n, 2).unwrap(),
            false,
            &mut scratch,
            cx,
        )
        .unwrap();
        let strided: Vec<_> = strided.iter().step_by(2).copied().collect();
        assert_eq!(strided, output);
    }
}

#[test]
fn odd_decimation_matches_complex_fft_f32() {
    decimated_cases::<f32>(2e-5);
}

#[test]
fn decimation_rejects_nonfinite_pair_blocks_before_running_children() {
    let plan = prepared::<f64>(15).unwrap();
    let RealAlgorithm::Decimated(decimation) = &plan.algorithm else {
        unreachable!()
    };
    for bad in [f64::NAN, f64::INFINITY] {
        for inverse in [false, true] {
            let mut data = vec![Complex::new(0.0, 0.0); plan.spectrum_len()];
            if inverse {
                data[1].re = bad;
            } else {
                data[0].im = bad;
            }
            let mut work = vec![Complex::new(0.0, 0.0); decimation.work_len().unwrap()];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if inverse {
                    decimation.inverse(
                        &mut data,
                        &mut work,
                        BinaryScale::Identity,
                        ExecutionContext::sequential(),
                    )
                } else {
                    decimation.forward(&mut data, &mut work, ExecutionContext::sequential())
                }
            }));
            assert!(result.is_err());
        }
    }
}

#[test]
fn odd_decimation_matches_complex_fft_f64() {
    decimated_cases::<f64>(2e-13);
}

#[test]
fn decimated_overflow_is_checked_and_transactional() {
    let n = 45;
    let cx = ExecutionContext::sequential_native();
    let plan = prepared::<f64>(n).unwrap();
    let (forward, _) = Normalization::Forward.fourier_scales::<f64>(n).unwrap();
    let mut spectrum = vec![Complex::new(7.0, 7.0); plan.spectrum_len()];
    let mut scratch = workspace(&plan);
    plan.forward_into(
        forward,
        VectorView::from_slice(&[f64::MAX; 45]),
        VectorViewMut::from_slice(&mut spectrum),
        false,
        &mut scratch,
        cx,
    )
    .unwrap();
    assert!(spectrum[0].re > f64::MAX * 0.99 && spectrum[0].re.is_finite());
    assert!(spectrum[1..].iter().all(|z| z.norm() < f64::MAX * 1e-12));

    // Unscaled: a lone DC bin is representable, a second full bin is not.
    let mut bins = vec![Complex::new(0.0, 0.0); plan.spectrum_len()];
    bins[0].re = f64::MAX;
    let mut output = vec![0.0; n];
    plan.inverse_into(
        1.0,
        VectorView::from_slice(&bins),
        VectorViewMut::from_slice(&mut output),
        false,
        &mut scratch,
        cx,
    )
    .unwrap();
    assert_real_close(&output, &[f64::MAX; 45], |_| f64::MAX * 1e-14);
    bins[1].re = f64::MAX;
    let mut sentinel = vec![9.0; n];
    let error = plan
        .inverse_into(
            1.0,
            VectorView::from_slice(&bins),
            VectorViewMut::from_slice(&mut sentinel),
            false,
            &mut scratch,
            cx,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(sentinel, vec![9.0; n]);
}

#[test]
fn even_reduction_and_odd_fallback_match_complex_fft() {
    for input in [
        vec![1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0],
        vec![1.0_f64, -2.0, 3.0, -4.0, 5.0],
    ] {
        let complex_input: Vec<_> = input
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        let expected = crate::fft(&complex_input, Normalization::Inverse).unwrap();
        let actual = rfft(&input, Normalization::Inverse).unwrap();
        assert_complex_close(
            actual.bins(),
            &expected[..actual.len()],
            Complex::norm,
            |_| 2e-11,
        );
        assert_real_close(
            &irfft(&actual, Normalization::Inverse).unwrap(),
            &input,
            |_| 2e-11,
        );
    }
}

#[test]
fn inverse_endpoints_and_all_structural_errors_are_transactional() {
    let cx = ExecutionContext::sequential_native();
    let plan = prepared::<f64>(8).unwrap();
    let (_, inverse) = Normalization::Inverse.fourier_scales::<f64>(8).unwrap();
    let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut spectrum = rfft(&input, Normalization::Inverse).unwrap().into_vec();
    let mut output = [99.0; 8];
    let mut scratch = workspace(&plan);
    let mut execute = |spectrum: &[Complex<f64>], output: &mut [f64]| {
        plan.inverse_into(
            inverse,
            VectorView::from_slice(spectrum),
            VectorViewMut::from_slice(output),
            false,
            &mut scratch,
            cx,
        )
    };

    spectrum[0].im = 1.0;
    let error = execute(&spectrum, &mut output).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    assert_eq!(output, [99.0; 8]);

    spectrum = rfft(&input, Normalization::Inverse).unwrap().into_vec();
    let allowance = 8.0 * f64::EPSILON * spectrum[0].re.abs();
    spectrum[0].im = allowance;
    spectrum[4].im = -allowance;
    execute(&spectrum, &mut output).unwrap();
    assert_real_close(&output, &input, |_| 1e-12);
    output.fill(99.0);

    spectrum = rfft(&input, Normalization::Inverse).unwrap().into_vec();
    spectrum[4].im = 1.0;
    let error = execute(&spectrum, &mut output).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    assert_eq!(output, [99.0; 8]);

    let error = plan
        .forward_into(
            1.0,
            VectorView::from_slice(&input),
            VectorViewMut::from_slice(&mut vec![Complex::new(7.0, 7.0); plan.spectrum_len()]),
            false,
            &mut vec![Complex::new(0.0, 0.0); plan.scratch_len() - 1],
            cx,
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn allocating_delegates_validate_empty_non_finite_and_shape() {
    let error = rfft::<f64>(&[], Normalization::Inverse).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    let error = rfft(&[1.0_f64, f64::NAN], Normalization::Inverse).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    let error = HermitianSpectrum::<f64>::from_vec(0, vec![]).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    let error = HermitianSpectrum::from_vec(4, vec![Complex::new(1.0_f64, 0.0)]).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn normalized_real_transforms_avoid_unscaled_intermediate_overflow() {
    let cx = ExecutionContext::sequential_native();
    let plan = prepared::<f64>(8).unwrap();
    let (forward, _) = Normalization::Forward.fourier_scales::<f64>(8).unwrap();
    let mut scratch = workspace(&plan);
    let mut spectrum = vec![Complex::new(7.0, 7.0); plan.spectrum_len()];
    plan.forward_into(
        forward,
        VectorView::from_slice(&[f64::MAX; 8]),
        VectorViewMut::from_slice(&mut spectrum),
        false,
        &mut scratch,
        cx,
    )
    .unwrap();
    assert!(
        spectrum
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(spectrum[0].re > f64::MAX * 0.99);

    let mut sentinel = vec![Complex::new(7.0, 7.0); plan.spectrum_len()];
    let before = sentinel.clone();
    let error = plan
        .forward_into(
            1.0,
            VectorView::from_slice(&[f64::MAX; 8]),
            VectorViewMut::from_slice(&mut sentinel),
            false,
            &mut scratch,
            cx,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(sentinel, before);
}

fn hermitian_input_cases<S>()
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let cx = ExecutionContext::sequential_native();
    for n in [1, 2, 7, 8] {
        let plan = prepared::<S>(n).unwrap();
        let (_, inverse) = Normalization::Inverse.fourier_scales::<S>(n).unwrap();
        let mut scratch = workspace(&plan);
        let mut execute = |spectrum: &[Complex<S>], output: &mut [S]| {
            plan.inverse_into(
                inverse,
                VectorView::from_slice(spectrum),
                VectorViewMut::from_slice(output),
                false,
                &mut scratch,
                cx,
            )
        };
        let exponents: &[i32] = if S::MANTISSA_DIGITS == 53 {
            &[-900, -100, 0, 100, 900]
        } else {
            &[-100, 0, 100]
        };
        for &exponent in exponents {
            let scale = S::ONE.scalbn(exponent);
            let tolerance = S::EPSILON.scalbn(3) * scale;
            for imaginary in [false, true] {
                for accepted in [true, false] {
                    let noise = (tolerance * S::TWO).next_after(if accepted {
                        S::ZERO
                    } else {
                        S::INFINITY
                    });
                    let value = if imaginary {
                        Complex::new(S::ZERO, noise)
                    } else {
                        Complex::new(noise, S::ZERO)
                    };
                    let projected = crate::hermitian::project_pair(
                        value,
                        Some(Complex::new(S::ZERO, S::ZERO)),
                        scale,
                    );
                    if accepted {
                        let expected = if imaginary {
                            Complex::new(S::ZERO, noise * S::HALF)
                        } else {
                            Complex::new(noise * S::HALF, S::ZERO)
                        };
                        assert_eq!(projected.unwrap(), expected);
                    } else {
                        assert_eq!(
                            projected.unwrap_err().kind(),
                            scioxide_core::SciOxideErrorKind::InvalidInput
                        );
                    }
                }
            }
            let mut exact = vec![Complex::new(S::ZERO, S::ZERO); plan.spectrum_len()];
            // A tiny DC real part must use the rest of the spectrum's scale.
            exact[if n == 1 { 0 } else { 1 }].re = scale;
            let mut expected = vec![S::ZERO; n];
            execute(&exact, &mut expected).unwrap();
            let endpoints: &[usize] = if n.is_multiple_of(2) {
                &[0, n / 2]
            } else {
                &[0]
            };
            for &endpoint in endpoints {
                let mut spectrum = exact.clone();
                spectrum[endpoint].im = tolerance.next_after(S::ZERO);
                let mut output = vec![S::ZERO; n];
                execute(&spectrum, &mut output).unwrap();
                assert_eq!(output, expected);
                spectrum[endpoint].im = tolerance.next_after(S::INFINITY);
                let mut sentinel = vec![S::TEN; n];
                let error = execute(&spectrum, &mut sentinel).unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
                assert_eq!(sentinel, vec![S::TEN; n]);
            }
        }
        let mut spectrum = vec![Complex::new(S::ZERO, S::ZERO); plan.spectrum_len()];
        spectrum[if n == 1 { 0 } else { 1 }].re = S::MIN_SUBNORMAL.scalbn(4);
        spectrum[0].im = S::MIN_SUBNORMAL;
        let mut output = vec![S::ZERO; n];
        execute(&spectrum, &mut output).unwrap();
        spectrum[0].im = S::MIN_SUBNORMAL * S::TWO;
        let mut sentinel = vec![S::TEN; n];
        assert_eq!(
            execute(&spectrum, &mut sentinel).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
        assert_eq!(sentinel, vec![S::TEN; n]);
        for index in 0..spectrum.len() {
            for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
                let mut invalid = spectrum.clone();
                invalid[index] = Complex::new(bad, bad);
                assert_eq!(
                    execute(&invalid, &mut sentinel).unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::NonFinite
                );
                assert_eq!(sentinel, vec![S::TEN; n]);
            }
        }
    }
    // The final stored odd-length bin is not a Nyquist endpoint.
    let plan = prepared::<S>(7).unwrap();
    let mut spectrum = vec![Complex::new(S::ZERO, S::ZERO); plan.spectrum_len()];
    spectrum[3].im = S::ONE;
    plan.inverse_into(
        S::ONE,
        VectorView::from_slice(&spectrum),
        VectorViewMut::from_slice(&mut [S::ZERO; 7]),
        false,
        &mut workspace(&plan),
        cx,
    )
    .unwrap();
}

#[test]
fn hermitian_input_policy_f32() {
    hermitian_input_cases::<f32>();
}

#[test]
fn hermitian_input_policy_f64() {
    hermitian_input_cases::<f64>();
}

#[test]
fn odd_residual_breakdown_is_separate_from_input_rejection() {
    let mass = 2.0_f64;
    let bound = crate::hermitian::residual_bound(7, mass).unwrap();
    let mut values = [Complex::new(0.0, 0.0); 7];
    values[3].im = bound.next_after(0.0);
    check_odd_inverse_residual(&values, 7, mass).unwrap();
    values[3].im = bound.next_after(f64::INFINITY);
    assert!(matches!(
        check_odd_inverse_residual(&values, 7, mass)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    ));
}

#[test]
fn odd_roundtrips_use_one_global_scale_aware_residual() {
    for amplitude in [1e-100_f64, 1.0, 1e8, 1e300] {
        let mut input = [0.0; 7];
        input[3] = amplitude;
        let spectrum = rfft(&input, Normalization::Inverse).unwrap();
        let output = irfft(&spectrum, Normalization::Inverse).unwrap();
        let tolerance = amplitude.abs().max(1.0) * 2e-13;
        assert_real_close(&output, &input, |_| tolerance);
    }
}
