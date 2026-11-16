//! Allocating convenience transforms match prepared plans and independent fixtures.
#[path = "support/hankel_zeros.rs"]
mod hankel_zeros;
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/owned.rs"]
mod owned;
use owned::ExecuteOwned;

use numerics::{assert_close, assert_complex_close, assert_real_close};
use scioxide_core::{Complex, ExecutionPolicy, NdView, RealScalar, StridedLayout};
use scioxide_fft::{
    CztContour, CztPlan, CztPlanSpec, DiscreteHankelPlan, DiscreteHankelPlanSpec, FftDirection,
    FftLogLimits, FftLogPlan, FftLogPlanSpec, HermitianSpectrum, Normalization, NufftPlan,
    NufftPlanSpec, NufftTransform, czt, dht, fftlog_low_ringing_offset, fht, hfft, idht, ifht,
    ihfft, nufft_type1, nufft_type2, nufft_type3, zoom_fft,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    czt: CztFixture,
    zoom: ZoomFixture,
    fftlog: FftLogFixture,
    fhtoffset: FhtOffsetFixture,
    dht: DhtFixture,
    nufft: NufftFixture,
    hfft: Vec<HfftFixture>,
    ihfft: Vec<IhfftFixture>,
}
#[derive(Deserialize)]
struct CztFixture {
    input: Vec<[f64; 2]>,
    a: [f64; 2],
    w: [f64; 2],
    m: usize,
    expected: Vec<[f64; 2]>,
}
#[derive(Deserialize)]
struct ZoomFixture {
    input: Vec<[f64; 2]>,
    start: f64,
    end: f64,
    m: usize,
    sample_rate: f64,
    endpoint: bool,
    expected: Vec<[f64; 2]>,
}
#[derive(Deserialize)]
struct FftLogFixture {
    input: Vec<f64>,
    spacing: f64,
    order: f64,
    offset: f64,
    bias: f64,
    forward: Vec<f64>,
    inverse: Vec<f64>,
}
#[derive(Deserialize)]
struct FhtOffsetFixture {
    spacing: f64,
    order: f64,
    initial: f64,
    bias: f64,
    expected: f64,
}
#[derive(Deserialize)]
struct DhtFixture {
    input: Vec<f64>,
    order: f64,
    extent: f64,
    expected: Vec<f64>,
}
#[derive(Deserialize)]
struct NufftFixture {
    points: Vec<f64>,
    frequencies: Vec<f64>,
    input: Vec<[f64; 2]>,
    type2_input: Vec<[f64; 2]>,
    modes: usize,
    type1: Vec<[f64; 2]>,
    type2: Vec<[f64; 2]>,
    type3: Vec<[f64; 2]>,
}
#[derive(Deserialize)]
struct HfftFixture {
    input: Vec<[f64; 2]>,
    length: usize,
    expected: Vec<f64>,
}
#[derive(Deserialize)]
struct IhfftFixture {
    input: Vec<f64>,
    expected: Vec<[f64; 2]>,
}
#[expect(clippy::unwrap_used, reason = "decode the checked-in fixture")]
fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!("fixtures/simple.json")).unwrap()
}
fn complex(values: &[[f64; 2]]) -> Vec<Complex<f64>> {
    values
        .iter()
        .map(|&[re, im]| Complex::new(re, im))
        .collect()
}
#[expect(clippy::unwrap_used, reason = "f32 and f64 widen exactly")]
fn bits<S: RealScalar>(values: &[Complex<S>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|z| {
            let wide = |x: S| x.try_to_f64_exact().unwrap().to_bits();
            (wide(z.re), wide(z.im))
        })
        .collect()
}
fn real_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn czt_and_zoom_match_plans_and_scipy_1_18_1() {
    let case = fixtures().czt;
    let input = complex(&case.input);
    let a = Complex::new(case.a[0], case.a[1]);
    let w = Complex::new(case.w[0], case.w[1]);
    let actual = czt(&input, case.m, w, a, Normalization::Unscaled).unwrap();
    let plan = CztPlan::new(CztPlanSpec::new(
        input.len(),
        case.m,
        CztContour::from_wa(w, a).unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    assert_eq!(bits(&actual), bits(&plan.execute_owned(&input).unwrap()));
    assert_complex_close(&actual, &complex(&case.expected), Complex::norm, |z| {
        2e-12 * (1.0 + z.norm())
    });
    assert!(czt(&input, 0, w, a, Normalization::Unscaled).is_err());

    let case = fixtures().zoom;
    let input = complex(&case.input);
    let actual = zoom_fft(
        &input,
        case.start,
        case.end,
        case.m,
        case.sample_rate,
        case.endpoint,
        Normalization::Unscaled,
    )
    .unwrap();
    let plan = CztPlan::new(CztPlanSpec::new(
        input.len(),
        case.m,
        CztContour::zoom(
            case.start,
            case.end,
            case.m,
            case.sample_rate,
            case.endpoint,
        )
        .unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    assert_eq!(bits(&actual), bits(&plan.execute_owned(&input).unwrap()));
    assert_complex_close(&actual, &complex(&case.expected), Complex::norm, |z| {
        2e-12 * (1.0 + z.norm())
    });
    assert!(zoom_fft(&input, 0.0, 1.0, 5, 0.0, false, Normalization::Unscaled).is_err());
    real_contour_input::<f32>();
    real_contour_input::<f64>();
}

/// Real samples give the bits of the same samples with zero imaginary parts.
#[expect(clippy::unwrap_used, reason = "fixed test samples")]
fn real_contour_input<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let real: Vec<S> = [1.0, -0.5, 2.0, 0.25, 0.75].map(scalar).to_vec();
    let complexified: Vec<_> = real.iter().map(|&x| Complex::new(x, S::ZERO)).collect();
    let (w, a) = (
        Complex::new(scalar(0.6), scalar(-0.8)),
        Complex::new(scalar(1.1), S::ZERO),
    );
    let norm = Normalization::Orthonormal;
    assert_eq!(
        bits(&czt(&real, 4, w, a, norm).unwrap()),
        bits(&czt(&complexified, 4, w, a, norm).unwrap())
    );
    let (start, end, rate) = (scalar(-0.1), scalar(0.3), S::ONE);
    assert_eq!(
        bits(&zoom_fft(&real, start, end, 6, rate, true, norm).unwrap()),
        bits(&zoom_fft(&complexified, start, end, 6, rate, true, norm).unwrap())
    );
}

#[test]
fn low_ringing_offset_matches_scipy_1_18_1_fhtoffset() {
    let case = fixtures().fhtoffset;
    let offset =
        fftlog_low_ringing_offset(case.spacing, case.order, case.initial, case.bias).unwrap();
    // Both libraries round the Hamilton log-gamma phase independently; dln / pi
    // scales that rounding into the offset.
    assert_close(
        (offset - case.expected).abs(),
        64.0 * f64::EPSILON,
        "fhtoffset",
    );
}

#[test]
fn real_fftlog_matches_plan_and_scipy_1_18_1() {
    let case = fixtures().fftlog;
    let input: Vec<_> = case.input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    for (direction, expected) in [
        (scioxide_fft::FftDirection::Forward, &case.forward),
        (scioxide_fft::FftDirection::Inverse, &case.inverse),
    ] {
        let plan = FftLogPlan::new(
            FftLogPlanSpec::new(
                input.len(),
                case.spacing,
                case.order,
                case.offset,
                Complex::new(case.bias, 0.0),
                direction,
            )
            .with_execution_policy(ExecutionPolicy::Sequential)
            .with_limits(FftLogLimits::default()),
        )
        .unwrap();
        let mut output = vec![Complex::new(0.0, 0.0); input.len()];
        plan.execute_into(
            &input,
            &mut output,
            &mut plan.workspace(()).unwrap(),
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
        let actual = if direction == scioxide_fft::FftDirection::Forward {
            fht(
                &case.input,
                case.spacing,
                case.order,
                case.offset,
                case.bias,
            )
        } else {
            ifht(
                &case.input,
                case.spacing,
                case.order,
                case.offset,
                case.bias,
            )
        }
        .unwrap();
        assert_real_close(
            &actual,
            &output.iter().map(|z| z.re).collect::<Vec<_>>(),
            |_| 1e-13,
        );
        for z in output {
            numerics::assert_close(z.im.abs(), 1e-12, "FFTLog imaginary residual");
        }
        assert_real_close(&actual, expected, |_| 2e-12);
    }
    let forward = fht(
        &case.input,
        case.spacing,
        case.order,
        case.offset,
        case.bias,
    )
    .unwrap();
    let back = ifht(&forward, case.spacing, case.order, case.offset, case.bias).unwrap();
    assert_real_close(&back, &case.input, |_| 2e-12);
    assert!(fht(&case.input, 0.0, case.order, case.offset, case.bias).is_err());
}

/// Runs a discrete Hankel plan with the default spec on real samples.
#[expect(clippy::unwrap_used, reason = "fixed test zeros")]
fn hankel_plan(
    input: &[f64],
    zeros: &[f64],
    order: f64,
    direction: FftDirection,
) -> Vec<Complex<f64>> {
    let spec = DiscreteHankelPlanSpec::new(order, 4.0, direction);
    let plan = DiscreteHankelPlan::new(zeros, spec).unwrap();
    let complex: Vec<_> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut output = vec![Complex::new(0.0, 0.0); input.len()];
    let cx = scioxide_core::ExecutionContext::sequential_native();
    plan.execute_into(&complex, &mut output, &mut plan.workspace(()).unwrap(), cx)
        .unwrap();
    output
}

#[test]
fn dht_matches_plan_and_gsl_2_8() {
    let case = fixtures().dht;
    assert_eq!(case.extent, 4.0);
    let zeros = hankel_zeros::zeros(case.order);
    let zeros = &zeros[..case.input.len() + 1];
    let expected = hankel_plan(&case.input, zeros, case.order, FftDirection::Forward);
    assert!(expected.iter().all(|z| z.im == 0.0));
    let actual = dht(&case.input, zeros, case.order, case.extent).unwrap();
    let real: Vec<_> = expected.iter().map(|z| z.re).collect();
    assert_eq!(real_bits(&actual), real_bits(&real));
    assert_real_close(&actual, &case.expected, |_| 2e-11);
    assert!(dht(&[], &zeros[..1], case.order, case.extent).is_err());
}

#[test]
fn dht_and_idht_match_forward_and_inverse_plans() {
    let zeros = hankel_zeros::zeros(0.5_f64);
    let zeros = &zeros[..17];
    let x: Vec<_> = (0..16).map(|j| j as f64).collect();
    for (direction, transform) in [
        (
            FftDirection::Forward,
            dht::<f64> as fn(&[f64], &[f64], f64, f64) -> _,
        ),
        (FftDirection::Inverse, idht::<f64>),
    ] {
        let expected = hankel_plan(&x, zeros, 0.5, direction);
        assert!(expected.iter().all(|z| z.im == 0.0));
        let real: Vec<_> = expected.iter().map(|z| z.re).collect();
        assert_eq!(
            real_bits(&transform(&x, zeros, 0.5, 4.0).unwrap()),
            real_bits(&real)
        );
    }
    let transformed = dht(&x, zeros, 0.5, 4.0).unwrap();
    let back = idht(&transformed, zeros, 0.5, 4.0).unwrap();
    assert!(x.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-9));
}

#[test]
fn nufft_functions_match_plans_and_finufft_2_5_1() {
    let case = fixtures().nufft;
    let shape = [case.points.len(), 1];
    let freq_shape = [case.frequencies.len(), 1];
    let strides = [1, 1];
    let view = || NdView::new(&case.points, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
    let freq = || {
        NdView::new(
            &case.frequencies,
            StridedLayout::new(&freq_shape, &strides).unwrap(),
        )
        .unwrap()
    };
    let input = complex(&case.input);
    let type2_input = complex(&case.type2_input);
    let modes = [case.modes];
    let accuracy = 1e-8;
    let plan = |transform, direction, norm| {
        NufftPlan::new(transform, NufftPlanSpec::new(direction, norm, accuracy)).unwrap()
    };
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        for norm in [Normalization::Unscaled, Normalization::Forward] {
            let type1 = NufftTransform::Type1 {
                points: view(),
                modes: &modes,
            };
            let type2 = NufftTransform::Type2 {
                points: view(),
                modes: &modes,
            };
            let type3 = NufftTransform::Type3 {
                points: view(),
                frequencies: freq(),
            };
            let cases = [
                (
                    nufft_type1(view(), &input, &modes, direction, norm, accuracy).unwrap(),
                    plan(type1, direction, norm).execute_owned(&input).unwrap(),
                    &case.type1,
                ),
                (
                    nufft_type2(view(), &type2_input, &modes, direction, norm, accuracy).unwrap(),
                    plan(type2, direction, norm)
                        .execute_owned(&type2_input)
                        .unwrap(),
                    &case.type2,
                ),
                (
                    nufft_type3(view(), &input, freq(), direction, norm, accuracy).unwrap(),
                    plan(type3, direction, norm).execute_owned(&input).unwrap(),
                    &case.type3,
                ),
            ];
            for (actual, plan, finufft) in cases {
                assert_eq!(bits(&actual), bits(&plan));
                if (direction, norm) == (FftDirection::Forward, Normalization::Unscaled) {
                    assert_complex_close(&actual, &complex(finufft), Complex::norm, |z| {
                        2e-7 * (1.0 + z.norm())
                    });
                }
            }
        }
    }
    let real: Vec<_> = input.iter().map(|z| z.re).collect();
    let complexified: Vec<_> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let real_modes: Vec<_> = type2_input.iter().map(|z| z.re).collect();
    let complex_modes: Vec<_> = real_modes.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let (direction, norm) = (FftDirection::Forward, Normalization::Unscaled);
    assert_eq!(
        bits(&nufft_type1(view(), &real, &modes, direction, norm, accuracy).unwrap()),
        bits(&nufft_type1(view(), &complexified, &modes, direction, norm, accuracy).unwrap())
    );
    assert_eq!(
        bits(&nufft_type2(view(), &real_modes, &modes, direction, norm, accuracy).unwrap()),
        bits(&nufft_type2(view(), &complex_modes, &modes, direction, norm, accuracy).unwrap())
    );
    assert_eq!(
        bits(&nufft_type3(view(), &real, freq(), direction, norm, accuracy).unwrap()),
        bits(&nufft_type3(view(), &complexified, freq(), direction, norm, accuracy).unwrap())
    );
    let error = nufft_type1(view(), &input[..3], &modes, direction, norm, accuracy).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn one_dimensional_hermitian_functions_match_nd_and_scipy_1_18_1() {
    for case in fixtures().hfft {
        let bins = complex(&case.input);
        let packed = HermitianSpectrum::from_vec(case.length, bins.clone()).unwrap();
        let actual = hfft(&packed, Normalization::Inverse).unwrap();
        let shape = [case.length];
        let nd = scioxide_fft::HermitianSpectrumNd::from_vec(&shape, &[0], bins).unwrap();
        let plan = scioxide_fft::hfftn(&nd, Normalization::Inverse).unwrap();
        assert_real_close(&actual, &plan, |_| 1e-13);
        assert_real_close(&actual, &case.expected, |_| 2e-12);
    }
    for case in fixtures().ihfft {
        let actual = ihfft(&case.input, Normalization::Inverse).unwrap();
        assert_eq!(actual.logical_len(), case.input.len());
        let shape = [case.input.len()];
        let plan = scioxide_fft::ihfftn(&case.input, &shape, &[0], Normalization::Inverse).unwrap();
        assert_complex_close(actual.bins(), &plan.into_vec(), Complex::norm, |z| {
            1e-13 * (1.0 + z.norm())
        });
        assert_complex_close(
            actual.bins(),
            &complex(&case.expected),
            Complex::norm,
            |z| 2e-12 * (1.0 + z.norm()),
        );
    }
    assert!(ihfft::<f64>(&[], Normalization::Inverse).is_err());
}

#[test]
fn single_precision_one_call_paths() {
    let x = [
        Complex::new(1.0_f32, 0.0),
        Complex::new(0.5, -0.25),
        Complex::new(-0.75, 0.5),
        Complex::new(0.25, 0.125),
        Complex::new(1.25, -0.5),
    ];
    let a = Complex::new(1.0, 0.0);
    let w = Complex::new(0.0, -1.0);
    assert_eq!(czt(&x, 4, w, a, Normalization::Unscaled).unwrap().len(), 4);
    assert_eq!(
        zoom_fft(&x, 0.0, 0.25, 4, 1.0, false, Normalization::Unscaled)
            .unwrap()
            .len(),
        4
    );
    let real = [1.0_f32, -0.5, 2.0, 0.25, 0.75];
    let transformed = fht(&real, 0.3, 0.5, 0.1, 0.2).unwrap();
    let restored = ifht(&transformed, 0.3, 0.5, 0.1, 0.2).unwrap();
    assert!(
        restored
            .iter()
            .zip(real)
            .all(|(&a, b)| (a - b).abs() < 2e-5)
    );
    let zeros = &hankel_zeros::zeros(0.0_f32)[..real.len() + 1];
    let transformed = dht(&real, zeros, 0.0, 4.0).unwrap();
    assert_eq!(transformed.len(), 5);
    assert_eq!(idht(&transformed, zeros, 0.0, 4.0).unwrap().len(), 5);
    let points = [-1.5_f32, -0.25, 0.7, 1.3, 0.2];
    let frequencies = [0.2_f32, -0.8, 1.1];
    let shape = [5, 1];
    let freq_shape = [3, 1];
    let strides = [1, 1];
    let view = || NdView::new(&points, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
    let freq = || {
        NdView::new(
            &frequencies,
            StridedLayout::new(&freq_shape, &strides).unwrap(),
        )
        .unwrap()
    };
    let (direction, norm) = (FftDirection::Forward, Normalization::Unscaled);
    let type1 = nufft_type1(view(), &x, &[5], direction, norm, 1e-4_f32).unwrap();
    assert_eq!(type1.len(), 5);
    assert_eq!(
        nufft_type2(view(), &x, &[5], direction, norm, 1e-4)
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        nufft_type3(view(), &real, freq(), direction, norm, 1e-4)
            .unwrap()
            .len(),
        3
    );
    let packed = ihfft(&real, Normalization::Inverse).unwrap();
    let restored = hfft(&packed, Normalization::Inverse).unwrap();
    assert!(
        restored
            .iter()
            .zip(real)
            .all(|(&a, b)| (a - b).abs() < 2e-5)
    );
}
