//! Chirp-z and zoom transforms, independent fixtures and transactional execution.
#![expect(clippy::unwrap_used)]
#[path = "support/execution.rs"]
mod execution;
use execution::CancelAfter;
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/owned.rs"]
mod owned;
#[path = "support/provider.rs"]
mod provider;
use owned::ExecuteOwned;
use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, RealScalar, StridedLayout};
use scioxide_fft::{CztAlgorithm, CztContour, CztPlan, CztPlanSpec, Normalization};

fn plan<S: RealScalar>(n: usize, m: usize, ar: f64, dr: f64, ap: f64, dp: f64) -> CztPlan<S>
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    CztPlan::build(
        CztPlanSpec::new(
            n,
            m,
            CztContour::new(s(ar), s(dr), s(ap), s(dp)).unwrap(),
            Normalization::Unscaled,
        ),
        ExecutionContext::sequential(),
    )
    .unwrap()
}
fn reference(
    input: &[Complex<f64>],
    m: usize,
    ar: f64,
    dr: f64,
    ap: f64,
    dp: f64,
) -> Vec<Complex<f64>> {
    (0..m)
        .map(|k| {
            numerics::direct_sum(input, |j| {
                (
                    -std::f64::consts::TAU * (ap + dp * k as f64) * j as f64,
                    (-(j as f64) * (ar + dr * k as f64)).exp(),
                )
            })
        })
        .collect()
}
fn accuracy<S: RealScalar>(tol: f64)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    for (n, m) in [(1, 1), (1, 7), (7, 1), (17, 23), (23, 17), (60, 61)] {
        for (ar, dr) in [(0.0, 0.0), (0.001, 0.00001), (0.2, -0.005), (-0.1, 0.004)] {
            let ap = 0.125;
            let dp = 0.03125;
            let p = plan::<S>(n, m, ar, dr, ap, dp);
            let input: Vec<_> = (0..n)
                .map(|j| {
                    Complex::new(
                        S::try_from_f64_lossy((j as f64 * 0.7).cos()).unwrap(),
                        S::try_from_f64_lossy((j as f64 * 0.3).sin()).unwrap(),
                    )
                })
                .collect();
            let x: Vec<_> = input
                .iter()
                .map(|v| {
                    Complex::new(
                        v.re.try_to_f64_exact().unwrap(),
                        v.im.try_to_f64_exact().unwrap(),
                    )
                })
                .collect();
            let c = p.spec().contour();
            let expected = reference(
                &x,
                m,
                c.log_radius().try_to_f64_exact().unwrap(),
                c.log_radius_step().try_to_f64_exact().unwrap(),
                ap,
                dp,
            );
            let actual = p.execute_owned(&input).unwrap();
            for (a, e) in actual.iter().zip(expected) {
                let err = (a.re.try_to_f64_exact().unwrap() - e.re)
                    .abs()
                    .max((a.im.try_to_f64_exact().unwrap() - e.im).abs());
                assert!(
                    err < tol * (1.0 + e.re.abs() + e.im.abs()),
                    "n={n} m={m} contour={ar},{dr} err={err}"
                );
            }
        }
    }
}
#[test]
fn f64_direct_references() {
    accuracy::<f64>(2e-11);
}
#[test]
fn f32_direct_references() {
    accuracy::<f32>(3e-5);
}

fn rounded_unit_circle<S: RealScalar>(tolerance: f64)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    for n in [2, 3, 8, 32, 63, 64] {
        let w = Complex::from_polar(S::ONE, -S::TAU / s(n as f64));
        for turns in [0.0, 0.05] {
            let a = Complex::from_polar(S::ONE, S::TAU * s(turns));
            let contour = CztContour::from_wa(w, a).unwrap();
            assert_eq!(contour.log_radius().decompose(), S::ZERO.decompose());
            assert_eq!(contour.log_radius_step().decompose(), S::ZERO.decompose());
            let plan =
                CztPlan::new(CztPlanSpec::new(n, n, contour, Normalization::Unscaled)).unwrap();
            assert_eq!(plan.algorithm(), CztAlgorithm::Bluestein);
            let input: Vec<_> = (0..n)
                .map(|j| Complex::new(s((j % 7) as f64 / 8.0), s((j % 3) as f64 / 4.0)))
                .collect();
            // A nontrivial starting phase is modulation before the DFT.
            let modulated: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    x * Complex::from_polar(
                        S::ONE,
                        s(-core::f64::consts::TAU * contour.start_turns() * j as f64),
                    )
                })
                .collect();
            let dft = CztPlan::new(CztPlanSpec::new(
                n,
                n,
                CztContour::dft(n).unwrap(),
                Normalization::Unscaled,
            ))
            .unwrap();
            let expected = dft.execute_owned(&modulated).unwrap();
            let actual = plan.execute_owned(&input).unwrap();
            let convenience = scioxide_fft::czt(&input, n, w, a, Normalization::Unscaled).unwrap();
            assert_eq!(actual, convenience);
            for (a, e) in actual.iter().zip(expected) {
                assert!(
                    (a.re - e.re)
                        .abs()
                        .into_f64()
                        .max((a.im - e.im).abs().into_f64())
                        <= tolerance * n as f64
                );
            }
            let radius = S::ONE + s(16.0) * S::EPSILON;
            let off = CztContour::from_wa(Complex::new(w.re * radius, w.im * radius), a).unwrap();
            assert!(!off.log_radius_step().is_zero());
            assert_eq!(
                CztPlan::new(CztPlanSpec::new(n, n, off, Normalization::Unscaled))
                    .unwrap()
                    .algorithm(),
                CztAlgorithm::ScaledDirect
            );
        }
    }
}

#[test]
fn rounded_unit_circle_parameters_keep_bluestein() {
    rounded_unit_circle::<f32>(3e-5);
    rounded_unit_circle::<f64>(2e-11);
}
#[test]
fn growth_decay_and_transaction() {
    for ar in [-800.0, 800.0] {
        let p = plan::<f64>(2, 3, ar, 0.0, 0.0, 0.0);
        assert_eq!(p.algorithm(), CztAlgorithm::ScaledDirect);
        let x = if ar < 0.0 {
            vec![Complex::new(1.0, 0.0), Complex::new(1e-300, 0.0)]
        } else {
            vec![Complex::new(1.0, 0.0), Complex::new(1e300, 0.0)]
        };
        let y = p.execute_owned(&x).unwrap();
        let expected = if ar < 0.0 {
            (800.0_f64 + 1e-300_f64.ln()).exp()
        } else {
            1.0
        };
        assert!((y[0].re / expected - 1.0).abs() < 2e-13);
    }
    let p = plan::<f64>(2, 3, -800.0, 0.0, 0.0, 0.0);
    let mut out = vec![Complex::new(17.0, 19.0); 3];
    let before = out.clone();
    let mut scratch = p.workspace(()).unwrap();
    assert!(
        p.execute_into(
            &[1.0, 1.0],
            &mut out,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(out, before);
    assert!(
        p.execute_into(
            &[1.0, f64::NAN],
            &mut out,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(out, before);
    assert!(
        p.execute_into(
            &[1.0, 0.0],
            &mut out,
            &mut [],
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(out, before);
}
#[test]
fn zoom_normalization_endpoints() {
    for normalization in [
        Normalization::Unscaled,
        Normalization::Inverse,
        Normalization::Forward,
        Normalization::Orthonormal,
    ] {
        let zoom = CztPlan::new(CztPlanSpec::new(
            8,
            8,
            CztContour::zoom(0.0, 8.0, 8, 8.0, false).unwrap(),
            normalization,
        ))
        .unwrap();
        let input: Vec<_> = (0..8).map(|j| Complex::new(j as f64, 0.0)).collect();
        let expected = scioxide_fft::fft(&input, normalization).unwrap();
        let actual = zoom.execute_owned(&input).unwrap();
        for (a, e) in actual.iter().zip(expected) {
            assert!((a.re - e.re).abs() + (a.im - e.im).abs() < 1e-12);
        }
    }
    for (start, end, endpoint) in [(1.0, 3.0, true), (3.0, -1.0, false), (2.0, 2.0, true)] {
        let zoom = CztPlan::new(CztPlanSpec::new(
            2,
            5,
            CztContour::zoom(start, end, 5, 8.0, endpoint).unwrap(),
            Normalization::Unscaled,
        ))
        .unwrap();
        let actual = zoom
            .execute_owned(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)])
            .unwrap();
        let denominator = if endpoint { 4.0 } else { 5.0 };
        for (k, a) in actual.iter().enumerate() {
            let phase =
                -std::f64::consts::TAU * (start + (end - start) * k as f64 / denominator) / 8.0;
            assert!((a.re - phase.cos()).abs() + (a.im - phase.sin()).abs() < 1e-13);
        }
    }
}
#[test]
fn strided_batches_and_real_views() {
    let p = plan::<f64>(3, 5, 0.0, 0.0, 0.125, 0.0625);
    let input: Vec<_> = (0..24).map(|i| i as f64).collect();
    let shape = [2, 3, 2];
    let strides = [12, 4, 2];
    let output_shape = [2, 5, 2];
    let output_strides = [20, 4, 2];
    let view = NdView::new(&input, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
    let mut output = vec![Complex::new(-1.0, -1.0); 40];
    let mut scratch = scioxide_fft::FftWorkspace::from_vec(vec![
        Complex::new(0.0, 0.0);
        p.workspace_requirements(
            scioxide_fft::AxisWorkspaceRequest::Axis {
                shape: &shape,
                axis: 1
            }
        )
        .unwrap()
        .complex_len()
    ]);
    p.execute_axis_into(
        view,
        1,
        NdViewMut::new(
            &mut output,
            StridedLayout::new(&output_shape, &output_strides).unwrap(),
        )
        .unwrap(),
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for b in 0..2 {
        for i in 0..2 {
            let x: Vec<_> = (0..3)
                .map(|j| Complex::new(input[b * 12 + j * 4 + i * 2], 0.0))
                .collect();
            let expected = p.execute_owned(&x).unwrap();
            for (k, e) in expected.iter().enumerate() {
                assert_eq!(output[b * 20 + k * 4 + i * 2], *e);
            }
        }
    }
    assert_eq!(output[1], Complex::new(-1.0, -1.0));
    assert_eq!(
        p.workspace_requirements(scioxide_fft::AxisWorkspaceRequest::Axis {
            shape: &[0, 3, usize::MAX],
            axis: 1
        })
        .unwrap()
        .complex_len(),
        0
    );
}
#[test]
fn contour_points_are_queried_through_the_spec_without_allocating() {
    fn check<S: RealScalar>()
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        let plan = plan::<S>(2, 4, 2.0_f64.ln(), 0.0, 0.0, 0.25);
        let expected = [(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)];
        for (k, (re, im)) in expected.into_iter().enumerate() {
            let point = plan.spec().contour().point(k).unwrap();
            let expected = Complex::new(
                S::try_from_f64_lossy(re).unwrap(),
                S::try_from_f64_lossy(im).unwrap(),
            );
            assert!((point - expected).norm() <= S::EPSILON * S::TEN);
        }
    }
    check::<f32>();
    check::<f64>();
    let contour = CztContour::new(1000.0_f64, 0.0, 0.0, 0.0).unwrap();
    assert_eq!(
        contour.point(0).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn contour_validation_and_wa_convention() {
    assert!(CztContour::new(f64::NAN, 0.0, 0.0, 0.0).is_err());
    assert!(CztContour::from_wa(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)).is_err());
    let c = CztContour::from_wa(Complex::new(0.0_f64, -1.0), Complex::new(2.0, 0.0)).unwrap();
    let p = CztPlan::new(CztPlanSpec::new(2, 4, c, Normalization::Unscaled)).unwrap();
    let out = p
        .execute_owned(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)])
        .unwrap();
    let expected = [
        Complex::new(0.5, 0.0),
        Complex::new(0.0, -0.5),
        Complex::new(-0.5, 0.0),
        Complex::new(0.0, 0.5),
    ];
    for (a, e) in out.iter().zip(expected) {
        assert!((a.re - e.re).abs() + (a.im - e.im).abs() < 1e-14);
    }
    assert!(CztPlan::new(CztPlanSpec::new(0, 1, c, Normalization::Unscaled)).is_err());
    assert!(CztPlan::new(CztPlanSpec::new(usize::MAX, 2, c, Normalization::Unscaled)).is_err());
}

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    width: u32,
    parameters: [f64; 4],
    input: Vec<[f64; 2]>,
    expected: Vec<[f64; 2]>,
}
#[derive(serde::Deserialize)]
struct Fixtures {
    cases: Vec<Fixture>,
}
fn fixture<S: RealScalar>(f: &Fixture, tolerance: f64)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let [ar, dr, ap, dp] = f.parameters;
    let p = plan::<S>(f.input.len(), f.expected.len(), ar, dr, ap, dp);
    let input: Vec<_> = f
        .input
        .iter()
        .map(|v| {
            Complex::new(
                S::try_from_f64_lossy(v[0]).unwrap(),
                S::try_from_f64_lossy(v[1]).unwrap(),
            )
        })
        .collect();
    let actual = p.execute_owned(&input).unwrap();
    for (a, e) in actual.iter().zip(&f.expected) {
        for (v, r) in [(a.re, e[0]), (a.im, e[1])] {
            let error = (v.try_to_f64_exact().unwrap() - r).abs();
            let scale = e[0].abs().max(e[1].abs()).max(1e-100);
            assert!(
                error <= tolerance * scale,
                "{} f{}: error={error:e} scale={scale:e}",
                f.name,
                f.width
            );
        }
    }
}
#[test]
fn independent_100_digit_fixtures() {
    let registry: Fixtures = serde_json::from_str(include_str!("fixtures/czt.json")).unwrap();
    for case in registry.cases {
        if case.width == 32 {
            fixture::<f32>(&case, 3e-5);
        } else {
            fixture::<f64>(&case, 3e-11);
        }
    }
}

#[derive(serde::Deserialize)]
enum FixtureNormalization {
    Unscaled,
    Inverse,
    Forward,
    Orthonormal,
}
impl FixtureNormalization {
    fn value(&self) -> Normalization {
        match self {
            Self::Unscaled => Normalization::Unscaled,
            Self::Inverse => Normalization::Inverse,
            Self::Forward => Normalization::Forward,
            Self::Orthonormal => Normalization::Orthonormal,
        }
    }
}
#[derive(serde::Deserialize)]
struct BoundedFixture {
    #[serde(flatten)]
    fixture: Fixture,
    normalization: FixtureNormalization,
    mass: Vec<f64>,
    coefficient_error: Vec<f64>,
}
#[derive(serde::Deserialize)]
struct BoundedFixtures {
    cases: Vec<BoundedFixture>,
}

fn bounded_fixture<S: RealScalar>(case: &BoundedFixture) {
    let f = &case.fixture;
    let [ar, dr, ap, dp] = f.parameters.map(|x| S::try_from_f64_lossy(x).unwrap());
    let plan = CztPlan::new(CztPlanSpec::new(
        f.input.len(),
        f.expected.len(),
        CztContour::new(ar, dr, ap, dp).unwrap(),
        case.normalization.value(),
    ))
    .unwrap();
    assert_eq!(plan.algorithm(), CztAlgorithm::ScaledDirect);
    let input: Vec<_> = f
        .input
        .iter()
        .map(|v| {
            Complex::new(
                S::try_from_f64_lossy(v[0]).unwrap(),
                S::try_from_f64_lossy(v[1]).unwrap(),
            )
        })
        .collect();
    let actual = plan.execute_owned(&input).unwrap();
    for (k, (a, e)) in actual.iter().zip(&f.expected).enumerate() {
        // Independent direct sums and L1 mass supply a conditioning-aware
        // allowance. The radial/phase contribution is separate from the
        // scalar-dependent accumulation envelope, even near cancellation.
        let bound = 64.0 * S::EPSILON.into_f64() * (input.len() as f64 + 2.0) * case.mass[k]
            + case.coefficient_error[k]
            + 2.0 * S::MIN_SUBNORMAL.into_f64();
        for (value, expected) in [(a.re, e[0]), (a.im, e[1])] {
            let error = (value.into_f64() - expected).abs();
            assert!(
                error <= bound,
                "{} f{} k={k}: error={error:e}, bound={bound:e}",
                f.name,
                f.width
            );
        }
    }
}

#[test]
fn independent_bounded_and_fallback_high_precision_sums() {
    let registry: BoundedFixtures =
        serde_json::from_str(include_str!("fixtures/czt_bounded.json")).unwrap();
    for case in registry.cases {
        if case.fixture.width == 32 {
            bounded_fixture::<f32>(&case);
        } else {
            bounded_fixture::<f64>(&case);
        }
    }
}

#[derive(serde::Deserialize)]
struct FastFixture {
    #[serde(flatten)]
    fixture: Fixture,
    weighted_mass: f64,
    log_mass: f64,
}

fn fast_fixture<S: RealScalar>(case: &FastFixture) {
    let f = &case.fixture;
    let [ar, dr, ap, dp] = f.parameters.map(|v| S::try_from_f64_lossy(v).unwrap());
    let n = f.input.len();
    let m = f.expected.len();
    let chi = dr.into_f64().abs() * n.max(m).pow(2) as f64;
    let l = scioxide_fft::next_fast_len(n + m - 1).unwrap();
    let input: Vec<_> = f
        .input
        .iter()
        .map(|&[re, im]| {
            Complex::new(
                S::try_from_f64_lossy(re).unwrap(),
                S::try_from_f64_lossy(im).unwrap(),
            )
        })
        .collect();
    for normalization in [
        Normalization::Unscaled,
        Normalization::Forward,
        Normalization::Inverse,
        Normalization::Orthonormal,
    ] {
        let p = CztPlan::new(CztPlanSpec::new(
            n,
            m,
            CztContour::new(ar, dr, ap, dp).unwrap(),
            normalization,
        ))
        .unwrap();
        assert_eq!(p.algorithm(), CztAlgorithm::Bluestein);
        let scale = match normalization {
            Normalization::Forward => 1.0 / n as f64,
            Normalization::Orthonormal => 1.0 / (n as f64).sqrt(),
            _ => 1.0,
        };
        let bound = scale
            * (2.0 * chi).exp()
            * (128.0 * S::EPSILON.into_f64() * (l as f64).log2().max(1.0) * case.weighted_mass
                + 32.0 * f64::EPSILON * (case.log_mass + chi * case.weighted_mass));
        let mut workspace = p.workspace(()).unwrap();
        let mut output = vec![Complex::new(S::ZERO, S::ZERO); m];
        {
            p.execute_into(
                &input,
                &mut output,
                &mut workspace,
                scioxide_fft::FftPlanningContext::<S>::default().execution(),
            )
            .unwrap();
        };
        for (k, (value, expected)) in output.iter().zip(&f.expected).enumerate() {
            for (actual, reference) in [
                (value.re.into_f64(), expected[0]),
                (value.im.into_f64(), expected[1]),
            ] {
                assert!(
                    (actual - scale * reference).abs() <= bound,
                    "{} f{} k={k}: error={} bound={bound}",
                    f.name,
                    f.width,
                    (actual - scale * reference).abs()
                );
            }
        }
    }
}

#[test]
fn large_radius_and_mild_spiral_high_precision_fixtures() {
    #[derive(serde::Deserialize)]
    struct Registry {
        cases: Vec<FastFixture>,
    }
    let registry: Registry = serde_json::from_str(include_str!("fixtures/czt_fast.json")).unwrap();
    for case in registry.cases {
        if case.fixture.width == 32 {
            fast_fixture::<f32>(&case);
        } else {
            fast_fixture::<f64>(&case);
        }
    }
}

#[test]
fn exact_selection_boundary_high_precision_fixtures() {
    #[derive(serde::Deserialize)]
    struct Boundary {
        direct: Vec<BoundedFixture>,
        fast: Vec<FastFixture>,
    }
    let cases: Boundary = serde_json::from_str(include_str!("fixtures/czt_boundary.json")).unwrap();
    for case in cases.direct {
        assert_eq!(case.fixture.input.len() * case.fixture.expected.len(), 4096);
        if case.fixture.width == 32 {
            bounded_fixture::<f32>(&case);
        } else {
            bounded_fixture::<f64>(&case);
        }
    }
    for case in cases.fast {
        assert_eq!(case.fixture.input.len() * case.fixture.expected.len(), 4097);
        if case.fixture.width == 32 {
            fast_fixture::<f32>(&case);
        } else {
            fast_fixture::<f64>(&case);
        }
    }
}

#[test]
fn fast_radius_strided_axes_real_input_and_late_failure() {
    let (n, m) = (65, 67);
    let p = plan::<f64>(n, m, 0.01, 0.0, 0.125, 1.0 / 128.0);
    let shape = [2, n];
    let target_shape = [2, m];
    let input_strides = [2 * n + 3, 2];
    let output_strides = [2 * m + 3, 2];
    let source_layout = StridedLayout::new(&shape, &input_strides).unwrap();
    let target_layout = StridedLayout::new(&target_shape, &output_strides).unwrap();
    let mut input: Vec<_> = (0..4 * n + 3).map(|j| (j % 17) as f64 / 16.0).collect();
    let complex: Vec<_> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let sentinel = Complex::new(17.0, 19.0);
    let mut output = vec![sentinel; 4 * m + 3];
    let mut workspace = p
        .workspace(scioxide_fft::AxisWorkspaceRequest::Axis {
            shape: &shape,
            axis: 1,
        })
        .unwrap();
    {
        p.execute_axis_into(
            NdView::new(&input, source_layout).unwrap(),
            1,
            NdViewMut::new(&mut output, target_layout).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    };
    let real_output = output.clone();
    p.execute_axis_into(
        NdView::new(&complex, source_layout).unwrap(),
        1,
        NdViewMut::new(&mut output, target_layout).unwrap(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, real_output);
    let mut addressed = vec![false; output.len()];
    for lane in 0..2 {
        let values: Vec<_> = (0..n)
            .map(|j| complex[lane * input_strides[0] + 2 * j])
            .collect();
        for (k, expected) in p.execute_owned(&values).unwrap().iter().enumerate() {
            let index = lane * output_strides[0] + 2 * k;
            assert_eq!(output[index], *expected);
            addressed[index] = true;
        }
    }
    assert!(
        output
            .iter()
            .zip(addressed)
            .all(|(&v, used)| used || v == sentinel)
    );
    input[input_strides[0] + 4] = f64::NAN;
    output.fill(sentinel);
    assert!(
        p.execute_axis_into(
            NdView::new(&input, source_layout).unwrap(),
            1,
            NdViewMut::new(&mut output, target_layout).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert!(output.iter().all(|&v| v == sentinel));

    let cancellation = CancelAfter::new(10);
    assert!(
        p.execute_into(
            &complex[..n],
            &mut output[..m],
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert!(output.iter().all(|&v| v == sentinel));
}

fn weighted_ranges<S: RealScalar>(ar: f64, power: i32) {
    let n = 65;
    let scalar = |v| S::try_from_f64_lossy(v).unwrap();
    for sign in [-1.0, 1.0] {
        let p = plan::<S>(n, n, ar * sign, 0.0, 0.0, 0.0);
        assert_eq!(p.algorithm(), CztAlgorithm::Bluestein);
        let x = S::ONE.scalbn(if sign < 0.0 { -power } else { power });
        let mut input = vec![Complex::new(S::ZERO, S::ZERO); n];
        input[1] = Complex::new(x, -x);
        let expected = (x.into_f64().ln() - sign * ar).exp();
        for value in p.execute_owned(&input).unwrap() {
            assert!((value.re.into_f64() / expected - 1.0).abs() < 512.0 * S::EPSILON.into_f64());
            assert!((value.im.into_f64() / expected + 1.0).abs() < 512.0 * S::EPSILON.into_f64());
        }
        input.fill(Complex::new(S::ZERO, S::ZERO));
        input[0] = Complex::new(S::MIN_SUBNORMAL, S::ZERO);
        for value in p.execute_owned(&input).unwrap() {
            assert!((value.re - S::MIN_SUBNORMAL).abs() <= S::MIN_SUBNORMAL);
        }
        input.fill(Complex::new(S::ZERO, S::ZERO));
        assert!(
            p.execute_owned(&input)
                .unwrap()
                .iter()
                .all(|z| z.re.is_zero() && z.im.is_zero())
        );
    }
    let p = plan::<S>(n, n, -ar, 0.0, 0.0, 0.0);
    let mut input = vec![Complex::new(S::ZERO, S::ZERO); n];
    input[1] = Complex::new(S::ONE, S::ZERO);
    let sentinel = Complex::new(scalar(17.0), scalar(19.0));
    let mut output = vec![sentinel; n];
    let mut workspace = p.workspace(()).unwrap();
    assert!(
        p.execute_into(
            &input,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(output.iter().all(|&z| z == sentinel));
    input[1] = Complex::new(S::NAN, S::ZERO);
    assert!(
        p.execute_into(
            &input,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(output.iter().all(|&z| z == sentinel));
}

#[test]
fn fast_radius_range_scaling_and_transaction() {
    weighted_ranges::<f32>(100.0, 120);
    weighted_ranges::<f64>(800.0, 1000);
    let mut input = vec![Complex::new(0.0, 0.0); 65];
    input[1] = Complex::new(1.0, -1.0);
    let decay = plan::<f64>(65, 65, 1e10, 0.0, 0.0, 0.0);
    assert!(
        decay
            .execute_owned(&input)
            .unwrap()
            .iter()
            .all(|z| z.re == 0.0 && z.im == 0.0)
    );
    let growth = plan::<f64>(65, 65, -1e10, 0.0, 0.0, 0.0);
    assert!(growth.execute_owned(&input).is_err());
    let spec = CztPlanSpec::new(
        65,
        65,
        CztContour::new(f64::MAX, 0.0, 0.0, 0.0).unwrap(),
        Normalization::Unscaled,
    );
    assert!(matches!(
        CztPlan::new(spec).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
}
#[test]
fn exact_grid_long_phases_and_reuse() {
    let n = 4093;
    // Existing FFT forward/round-trip rounding allowance at this length.
    let bound = 128.0 * f32::EPSILON as f64 * (n as f64).log2();
    let p = CztPlan::new(CztPlanSpec::new(
        n,
        n,
        CztContour::<f32>::dft(n).unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    let mut input = vec![Complex::new(0.0_f32, 0.0); n];
    input[n - 1] = Complex::new(1.0, 0.0);
    let output = p.execute_owned(&input).unwrap();
    for (k, a) in output.iter().enumerate() {
        let angle = std::f64::consts::TAU * k as f64 / n as f64;
        assert!(
            (a.re as f64 - angle.cos()).abs() + (a.im as f64 - angle.sin()).abs() < bound,
            "k={k} actual={a:?} angle={angle}"
        );
    }
    assert_eq!(output, p.execute_owned(&input).unwrap());
    let zoom = CztPlan::new(CztPlanSpec::new(
        n,
        257,
        CztContour::zoom(0.123_f32, 0.128, 257, 1.0, false).unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    let out = zoom.execute_owned(&input).unwrap();
    for (k, a) in out.iter().enumerate() {
        let frequency = 0.123_f32 as f64 + (0.128_f32 as f64 - 0.123_f32 as f64) * k as f64 / 257.0;
        let angle = -std::f64::consts::TAU * frequency * (n - 1) as f64;
        assert!(
            (a.re as f64 - angle.cos()).abs() + (a.im as f64 - angle.sin()).abs() < bound,
            "k={k} actual={a:?} angle={angle}"
        );
    }
}
#[test]
fn conjugated_views_and_cancellation_are_transactional() {
    use scioxide_linalg::{VectorView, VectorViewMut};
    let p = plan::<f64>(3, 5, 0.0, 0.0, 0.125, 0.0625);
    let input = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -4.0),
        Complex::new(-1.0, 0.5),
    ];
    let mut output = vec![Complex::new(0.0, 0.0); 5];
    let mut scratch = p.workspace(()).unwrap();
    p.execute_into(
        VectorView::from_slice(&input).conjugate(),
        VectorViewMut::from_slice(&mut output),
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let expected = p.execute_owned(&input.map(|v| v.conj())).unwrap();
    assert_eq!(output, expected);
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let before = output.clone();
    assert!(
        p.execute_into(
            &input,
            &mut output,
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(before, output);
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential() {
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    use std::num::NonZeroUsize;
    let executor = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let context = ExecutionContext::new(&executor);
    let spec = CztPlanSpec::new(
        32769,
        32769,
        CztContour::<f64>::dft(32769).unwrap(),
        Normalization::Forward,
    )
    .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let p = CztPlan::build(spec, context).unwrap();
    let x = vec![Complex::new(1.0, 0.0); 32769];
    let mut out = vec![Complex::new(0.0, 0.0); 32769];
    let mut scratch = p.workspace(()).unwrap();
    p.execute_into(&x, &mut out, &mut scratch, context).unwrap();
    let sequential = CztPlan::build(
        spec.with_execution_policy(ExecutionPolicy::Sequential),
        ExecutionContext::sequential(),
    )
    .unwrap()
    .execute_owned(&x)
    .unwrap();
    for (a, e) in out.iter().zip(sequential) {
        assert!((a.re - e.re).abs() + (a.im - e.im).abs() < 1e-12);
    }
}

#[test]
fn cancellation_after_validation_preserves_zero_input_output() {
    let cancellation = CancelAfter::new(1);
    let p = plan::<f64>(3, 5, 0.0, 0.0, 0.125, 0.0625);
    let mut output = vec![Complex::new(17.0, 19.0); 5];
    let original = output.clone();
    let mut scratch = p.workspace(()).unwrap();
    assert!(
        p.execute_into(
            &[0.0; 3],
            &mut output,
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert_eq!(output, original);
}

#[test]
fn native_limits_include_cached_storage_and_retained_axis_capacity() {
    use scioxide_core::{Capacity, SciOxideErrorView};
    use scioxide_fft::{AxisWorkspaceRequest, CztLimits, FftWorkspace};
    for ar in [0.0, 0.25, 800.0] {
        let spec = CztPlanSpec::new(
            17,
            23,
            CztContour::new(ar, 0.0, 0.125, 0.03125).unwrap(),
            Normalization::Unscaled,
        );
        let baseline = CztPlan::new(spec).unwrap();
        let plain = baseline.workspace_requirements(()).unwrap();
        let request = AxisWorkspaceRequest::Axis {
            shape: &[2, 17],
            axis: 1,
        };
        let axis = baseline.workspace_requirements(request).unwrap();
        let plan = CztPlan::new(spec.with_limits(CztLimits {
            preparation_bytes: usize::MAX,
            workspace_bytes: axis.bytes(),
        }))
        .unwrap();
        let requirements = plan.workspace_requirements(request).unwrap();
        let work = plan.workspace(request).unwrap();
        assert_eq!(work.check(&requirements).unwrap(), Capacity::Sufficient);
        let oversized =
            FftWorkspace::from_vec(vec![Complex::new(0.0, 0.0); requirements.complex_len() + 1]);
        assert!(matches!(
            oversized.check(&requirements).unwrap_err().view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        let limited = CztPlan::new(spec.with_limits(CztLimits {
            preparation_bytes: usize::MAX,
            workspace_bytes: plain.bytes(),
        }))
        .unwrap();
        assert!(matches!(
            limited.workspace_requirements(request).unwrap_err().view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        assert!(matches!(
            CztPlan::new(spec.with_limits(CztLimits {
                preparation_bytes: usize::MAX,
                workspace_bytes: plain.bytes() - 1
            }))
            .unwrap_err()
            .view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        assert!(matches!(
            CztPlan::new(spec.with_limits(CztLimits {
                preparation_bytes: 1,
                workspace_bytes: usize::MAX
            }))
            .unwrap_err()
            .view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        if ar != 0.0 {
            let bytes = baseline.plan_bytes();
            assert!(
                CztPlan::new(spec.with_limits(CztLimits {
                    preparation_bytes: bytes,
                    workspace_bytes: usize::MAX
                }))
                .is_ok()
            );
            assert!(matches!(
                CztPlan::new(spec.with_limits(CztLimits {
                    preparation_bytes: bytes - 1,
                    workspace_bytes: usize::MAX
                }))
                .unwrap_err()
                .view(),
                SciOxideErrorView::ResourceLimit { .. }
            ));
        }
    }
}

#[test]
fn direct_workspace_overflow_short_storage_and_late_range_errors_are_structured() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    let p = plan::<f64>(2, 3, 1.0, -400.0, 0.0, 0.0);
    let mut out = vec![Complex::new(17.0, 19.0); 3];
    let original = out.clone();
    let required = p.workspace_requirements(()).unwrap().complex_len();
    assert_eq!(required, 3 + 2);
    let mut scratch = vec![Complex::new(0.0, 0.0); required - 1];
    let error = p
        .execute_into(
            &[1.0, 1.0],
            &mut out,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(out, original);
    scratch.resize(required, Complex::new(0.0, 0.0));
    let error = p
        .execute_into(
            &[1.0, 1.0],
            &mut out,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(out, original);
    let spec = CztPlanSpec::new(
        usize::MAX,
        1,
        CztContour::new(1.0_f64, 0.0, 0.0, 0.0).unwrap(),
        Normalization::Unscaled,
    );
    assert!(matches!(
        CztPlan::new(spec).unwrap_err().view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
    let p = plan::<f64>(17, 23, 0.25, 0.0, 0.0, 0.0);
    assert!(matches!(
        p.workspace_requirements(scioxide_fft::AxisWorkspaceRequest::Axis {
            shape: &[usize::MAX / 17, 17],
            axis: 1
        })
        .unwrap_err()
        .view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn direct_real_and_complex_strided_axes_preserve_every_gap_and_failed_lane() {
    let shape = [2, 3];
    let out_shape = [2, 5];
    let layout = StridedLayout::new(&shape, &[8, 2]).unwrap();
    let out_layout = StridedLayout::new(&out_shape, &[12, 2]).unwrap();
    let input: Vec<_> = (0..16)
        .map(|j| Complex::new(j as f64 + 1.0, j as f64 / 4.0))
        .collect();
    let real: Vec<_> = input.iter().map(|x| x.re).collect();
    let sentinel = Complex::new(17.0, 19.0);
    for ar in [0.25, -0.25, 800.0] {
        let p = plan::<f64>(3, 5, ar, 0.0, 0.0, 0.03125);
        let mut scratch = p
            .workspace(scioxide_fft::AxisWorkspaceRequest::Axis {
                shape: &shape,
                axis: 1,
            })
            .unwrap();
        for real_input in [false, true] {
            let mut output = vec![sentinel; 24];
            if real_input {
                p.execute_axis_into(
                    NdView::new(&real, layout).unwrap(),
                    1,
                    NdViewMut::new(&mut output, out_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            } else {
                p.execute_axis_into(
                    NdView::new(&input, layout).unwrap(),
                    1,
                    NdViewMut::new(&mut output, out_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            }
            let mut addressed = [false; 24];
            for batch in 0..2 {
                let lane: Vec<_> = (0..3)
                    .map(|j| {
                        let x = input[batch * 8 + j * 2];
                        if real_input {
                            Complex::new(x.re, 0.0)
                        } else {
                            x
                        }
                    })
                    .collect();
                for (k, expected) in p.execute_owned(&lane).unwrap().iter().enumerate() {
                    let index = batch * 12 + k * 2;
                    assert_eq!(output[index], *expected);
                    addressed[index] = true;
                }
            }
            for (value, written) in output.iter().zip(addressed) {
                if !written {
                    assert_eq!(*value, sentinel);
                }
            }
        }
    }
    let p = plan::<f64>(3, 5, -800.0, 0.0, 0.0, 0.0);
    let mut scratch = p
        .workspace(scioxide_fft::AxisWorkspaceRequest::Axis {
            shape: &shape,
            axis: 1,
        })
        .unwrap();
    let mut late_failure = vec![Complex::new(0.0, 0.0); 16];
    late_failure[10] = Complex::new(1.0, 0.0);
    let mut output = vec![sentinel; 24];
    assert!(
        p.execute_axis_into(
            NdView::new(&late_failure, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, out_layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(output, vec![sentinel; 24]);
}

#[test]
fn cancellation_after_an_admitted_row_preserves_output() {
    let p = plan::<f64>(3, 5, 0.25, 0.0, 0.0, 0.0);
    let cancellation = CancelAfter::new(5);
    let mut output = vec![Complex::new(17.0, 19.0); 5];
    let original = output.clone();
    assert!(
        p.execute_into(
            &[1.0; 3],
            &mut output,
            &mut p.workspace(()).unwrap(),
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert_eq!(output, original);
}

#[test]
fn zero_input_preserves_late_exponent_error_and_signed_gradual_underflow() {
    let p = plan::<f64>(2, 3, f64::MAX, 0.0, 0.0, 0.0);
    assert_eq!(
        p.execute_owned(&[Complex::new(0.0, 0.0); 2]).unwrap(),
        vec![Complex::new(0.0, 0.0); 3]
    );
    assert!(
        p.execute_owned(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)])
            .is_err()
    );
    let p = plan::<f64>(2, 3, 800.0, 0.0, 0.0, 0.0);
    let output = p
        .execute_owned(&[Complex::new(0.0, 0.0), Complex::new(-1.0, 1.0)])
        .unwrap();
    assert!(output.iter().all(|x| x.re == 0.0
        && x.re.is_sign_negative()
        && x.im == 0.0
        && !x.im.is_sign_negative()));
}

#[cfg(feature = "std")]
#[test]
fn direct_component_bits_are_identical_across_worker_counts() {
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    use std::num::NonZeroUsize;
    for ar in [0.25, 800.0] {
        let spec = CztPlanSpec::new(
            17,
            23,
            CztContour::new(ar, 0.0, 0.125, 0.03125).unwrap(),
            Normalization::Orthonormal,
        );
        let input: Vec<_> = (0..17)
            .map(|j| Complex::new(1.0 + j as f64 / 32.0, j as f64 / 64.0))
            .collect();
        let expected = CztPlan::new(spec).unwrap().execute_owned(&input).unwrap();
        for workers in [1, 2, 4] {
            let count = NonZeroUsize::new(workers).unwrap();
            let executor = ThreadExecutor::new(count);
            let cx = ExecutionContext::new(&executor);
            let p = CztPlan::build(
                spec.with_execution_policy(ExecutionPolicy::Parallel(count)),
                cx,
            )
            .unwrap();
            let mut output = vec![Complex::new(0.0, 0.0); 23];
            p.execute_into(&input, &mut output, &mut p.workspace(()).unwrap(), cx)
                .unwrap();
            for (a, e) in output.iter().zip(&expected) {
                assert_eq!(
                    (a.re.to_bits(), a.im.to_bits()),
                    (e.re.to_bits(), e.im.to_bits())
                );
            }
        }
    }
}

#[test]
fn zoom_matches_defining_sums_in_both_directions() {
    fn check<S: RealScalar>()
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        let scalar = |value| S::try_from_f64_lossy(value).unwrap();
        let source: Vec<_> = (0..8)
            .map(|j| Complex::new(j as f64 / 8.0, (j % 3) as f64 / 4.0))
            .collect();
        let input: Vec<_> = source
            .iter()
            .map(|z| Complex::new(scalar(z.re), scalar(z.im)))
            .collect();
        for (m, start, end, endpoint) in [
            (8, 0.0, 1.0, false),
            (5, 0.125, 0.375, true),
            (5, 0.375, -0.125, false),
            (1, 0.2, 0.4, true),
        ] {
            let contour =
                CztContour::zoom(scalar(start), scalar(end), m, S::ONE, endpoint).unwrap();
            let plan =
                CztPlan::new(CztPlanSpec::new(8, m, contour, Normalization::Unscaled)).unwrap();
            let step = if m == 1 {
                0.0
            } else {
                (end - start) / if endpoint { (m - 1) as f64 } else { m as f64 }
            };
            let expected = reference(&source, m, 0.0, 0.0, start, step);
            // A short defining sum plus Bluestein convolution: absolute
            // component error scales with input L1 mass and scalar precision.
            let mass: f64 = source.iter().map(|z| z.re.abs() + z.im.abs()).sum();
            let bound = 64.0 * S::EPSILON.try_to_f64_exact().unwrap() * mass;
            numerics::assert_complex_close(
                &plan.execute_owned(&input).unwrap(),
                &expected,
                numerics::component_max,
                |_| bound,
            );
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn zoom_contour_validation_and_uniform_plan_controls() {
    use scioxide_fft::{CztLimits, FftBackend, FftBackendStrategy};
    for (m, start, end, sample_rate) in [
        (0, 0.0, 1.0, 1.0),
        (8, f64::NAN, 1.0, 1.0),
        (8, 0.0, f64::INFINITY, 1.0),
        (8, 0.0, 1.0, 0.0),
        (8, 0.0, f64::MAX, f64::MIN_POSITIVE),
    ] {
        assert!(CztContour::zoom(start, end, m, sample_rate, false).is_err());
    }
    let contour = CztContour::zoom(0.0, 1.0, 8, 1.0, false).unwrap();
    let spec = CztPlanSpec::new(8, 8, contour, Normalization::Unscaled)
        .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential)
        .with_backend_strategy(FftBackendStrategy::Required(
            scioxide_core::backend::KernelBackend::Scalar,
        ));
    let plan = CztPlan::new(spec).unwrap();
    assert_eq!(
        plan.resolved_backends().next(),
        Some(FftBackend::Cpu(
            scioxide_core::backend::KernelBackend::Scalar
        ))
    );
    for limits in [
        CztLimits {
            preparation_bytes: 1,
            ..CztLimits::default()
        },
        CztLimits {
            workspace_bytes: 1,
            ..CztLimits::default()
        },
    ] {
        assert!(CztPlan::new(spec.with_limits(limits)).is_err());
    }
}

#[test]
fn zoom_divides_extreme_spans_before_projecting() {
    for (m, endpoint) in [(4, false), (5, true)] {
        for sign in [-1.0, 1.0] {
            let contour =
                CztContour::zoom(-sign * f64::MAX, sign * f64::MAX, m, 1.0, endpoint).unwrap();
            let expected = sign * (f64::MAX / 2.0);
            assert!((contour.step_turns() / expected - 1.0).abs() <= f64::EPSILON);
            assert!((contour.point(1).unwrap() - Complex::new(1.0, 0.0)).norm() < 1e-14);
        }
    }
    let error = CztContour::zoom(-f64::MAX, f64::MAX, 2, 1.0, true).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
}

#[test]
fn empty_czt_axes_keep_the_prepared_capacity_limit() {
    use scioxide_fft::{
        AxisWorkspaceRequest, CztContour, CztPlan, CztPlanSpec, FftPlanningContext,
        FftProviderLimits, FftProviderPolicy,
    };
    let provider = provider::DirectProvider::new(provider::Mode::Direct);
    let plan = CztPlan::build(
        CztPlanSpec::new(
            3,
            3,
            CztContour::<f64>::dft(3).unwrap(),
            Normalization::Unscaled,
        ),
        FftPlanningContext::with_provider(
            ExecutionContext::sequential(),
            &provider,
            FftProviderPolicy::Required,
        )
        .with_provider_limits(FftProviderLimits {
            workspace_bytes: 4096,
            ..FftProviderLimits::default()
        }),
    )
    .unwrap();
    let plain = plan.workspace_requirements(()).unwrap();
    let empty = plan
        .workspace_requirements(AxisWorkspaceRequest::Axis {
            shape: &[0, 3],
            axis: 1,
        })
        .unwrap();
    assert_eq!(empty.complex_len(), 0);
    assert_eq!(empty.byte_limit(), plain.byte_limit());
    let retained = scioxide_fft::FftWorkspace::from_vec(vec![
        Complex::new(0.0, 0.0);
        empty.byte_limit()
            / size_of::<Complex<f64>>()
            + 1
    ]);
    assert_eq!(
        retained.check(&empty).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn axis_requirements_distinguish_overflow_from_zero_extent() {
    use scioxide_fft::{AxisWorkspaceRequest, CztContour, CztPlan, CztPlanSpec};
    let plan = CztPlan::<f64>::new(CztPlanSpec::new(
        2,
        2,
        CztContour::dft(2).unwrap(),
        Normalization::Inverse,
    ))
    .unwrap();
    assert_eq!(
        plan.workspace_requirements(AxisWorkspaceRequest::Axis {
            shape: &[2, usize::MAX],
            axis: 0
        })
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        plan.workspace_requirements(AxisWorkspaceRequest::Axis {
            shape: &[2, usize::MAX, 0],
            axis: 0
        })
        .unwrap()
        .complex_len(),
        0
    );
}

#[test]
fn default_limits_are_unbounded() {
    assert_eq!(
        scioxide_fft::CztLimits::default().preparation_bytes,
        usize::MAX
    );
}
