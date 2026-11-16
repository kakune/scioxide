//! NUFFT types 1–3: fixtures, accuracy, resources, operators and axis execution.
#![expect(clippy::unwrap_used, clippy::panic)]
#[path = "support/views.rs"]
mod views;
use views::view;
#[path = "support/execution.rs"]
mod execution;
use execution::CancelAfter;
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/owned.rs"]
mod owned;

use owned::ExecuteOwned;
use scioxide_core::{AdjointOperator, Complex, NdView, RealScalar, StridedLayout};
use scioxide_fft::{FftDirection, Normalization, NufftLimits, NufftPlan, NufftPlanSpec};

fn limits() -> NufftLimits {
    NufftLimits {
        preparation_bytes: 256 << 20,
        workspace_bytes: 256 << 20,
        grid_points: 1 << 22,
        stencil_visits: 20_000_000,
        kernel_support: 128,
    }
}
fn options<S: RealScalar>(eps: f64) -> NufftPlanSpec<S>
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    NufftPlanSpec::new(
        scioxide_fft::FftDirection::Forward,
        scioxide_fft::Normalization::Unscaled,
        S::try_from_f64_lossy(eps).unwrap(),
    )
    .with_limits(limits())
}
fn direct(
    points: &[f64],
    frequencies: &[f64],
    d: usize,
    input: &[Complex<f64>],
    outputs: usize,
    sign: f64,
) -> Vec<Complex<f64>> {
    (0..outputs)
        .map(|k| {
            numerics::direct_sum(input, |j| {
                (
                    (0..d)
                        .map(|a| points[j * d + a] * frequencies[k * d + a])
                        .sum::<f64>()
                        * sign,
                    1.0,
                )
            })
        })
        .collect()
}
fn error(actual: &[Complex<f64>], reference: &[Complex<f64>], input: &[Complex<f64>]) -> f64 {
    let scale: f64 = input.iter().map(|z| z.norm()).sum();
    actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a - *b).norm() / scale)
        .fold(0.0, f64::max)
}
fn modes(shape: &[usize]) -> Vec<f64> {
    let n: usize = shape.iter().product();
    (0..n)
        .flat_map(|i| {
            shape.iter().enumerate().map(move |(a, &n)| {
                let stride: usize = shape[a + 1..].iter().product();
                (i / stride % n) as f64 - (n / 2) as f64
            })
        })
        .collect()
}
#[test]
fn direct_accuracy_and_adjoint_all_types() {
    for (shape, count) in [(vec![19], 23), (vec![5, 4], 11), (vec![3, 2, 2], 5)] {
        let d = shape.len();
        let points: Vec<_> = (0..count * d)
            .map(|j| (j as f64 * 1.31).sin() * 2.9)
            .collect();
        let coords = [count, d];
        let strides = [d, 1];
        let frequencies = modes(&shape);
        let bins: usize = shape.iter().product();
        let fc = [bins, d];
        for kind in 1..=3 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let mut opt = options(1e-7);
                opt = NufftPlanSpec::new(direction, opt.normalization(), opt.accuracy())
                    .with_limits(opt.limits());
                opt = opt.with_oversampling(3.0);
                let plan = match kind {
                    1 => NufftPlan::build(
                        scioxide_fft::NufftTransform::Type1 {
                            points: view(&points, &coords, &strides),
                            modes: &shape,
                        },
                        opt,
                        scioxide_fft::FftPlanningContext::default(),
                    ),
                    2 => NufftPlan::build(
                        scioxide_fft::NufftTransform::Type2 {
                            points: view(&points, &coords, &strides),
                            modes: &shape,
                        },
                        opt,
                        scioxide_fft::FftPlanningContext::default(),
                    ),
                    _ => NufftPlan::build(
                        scioxide_fft::NufftTransform::Type3 {
                            points: view(&points, &coords, &strides),
                            frequencies: view(&frequencies, &fc, &strides),
                        },
                        opt,
                        scioxide_fft::FftPlanningContext::default(),
                    ),
                }
                .unwrap();
                let input: Vec<_> = (0..plan.input_len())
                    .map(|j| Complex::new((j as f64 * 0.7).cos(), (j as f64 * 1.3).sin()))
                    .collect();
                let sign = if direction == FftDirection::Forward {
                    -1.0
                } else {
                    1.0
                };
                let reference = if kind == 2 {
                    direct(&frequencies, &points, d, &input, count, sign)
                } else {
                    direct(&points, &frequencies, d, &input, bins, sign)
                };
                let actual = plan.execute_owned(&input).unwrap();
                let e = error(&actual, &reference, &input);
                assert!(e <= 1e-7, "type {kind} d={d} error {e:e}");
                let y: Vec<_> = (0..plan.output_len())
                    .map(|j| Complex::new((j as f64 * 0.2).sin(), 0.4))
                    .collect();
                let mut adj = vec![Complex::new(0.0, 0.0); plan.input_len()];
                plan.apply_adjoint_into(&y, &mut adj, &mut plan.workspace(()).unwrap())
                    .unwrap();
                let lhs: Complex<f64> = actual.iter().zip(&y).map(|(x, y)| x.conj() * *y).sum();
                let rhs: Complex<f64> = input.iter().zip(&adj).map(|(x, y)| x.conj() * *y).sum();
                assert!(
                    (lhs - rhs).norm() < 1e-10 * (1.0 + lhs.norm()),
                    "adjoint type {kind}: {lhs:?} vs {rhs:?}"
                );
            }
        }
    }
}

#[test]
fn single_precision_accuracy() {
    let points: Vec<_> = (0..31).map(|j| (j as f32 * 0.7).sin() * 3.0).collect();
    let freq: Vec<_> = (0..19).map(|j| j as f32 * 0.61 - 5.0).collect();
    let pshape = [points.len(), 1];
    let fshape = [freq.len(), 1];
    let strides = [1, 1];
    let pview = || NdView::new(&points, StridedLayout::new(&pshape, &strides).unwrap()).unwrap();
    for kind in 1..=3 {
        let mut opt = options::<f32>(1e-4);
        opt = opt.with_oversampling(4.0);
        let plan = match kind {
            1 => NufftPlan::build(
                scioxide_fft::NufftTransform::Type1 {
                    points: pview(),
                    modes: &[19],
                },
                opt,
                scioxide_fft::FftPlanningContext::default(),
            ),
            2 => NufftPlan::build(
                scioxide_fft::NufftTransform::Type2 {
                    points: pview(),
                    modes: &[19],
                },
                opt,
                scioxide_fft::FftPlanningContext::default(),
            ),
            _ => NufftPlan::build(
                scioxide_fft::NufftTransform::Type3 {
                    points: pview(),
                    frequencies: NdView::new(&freq, StridedLayout::new(&fshape, &strides).unwrap())
                        .unwrap(),
                },
                opt,
                scioxide_fft::FftPlanningContext::default(),
            ),
        }
        .unwrap();
        let input: Vec<_> = (0..plan.input_len())
            .map(|j| Complex::new((j as f32).cos(), 0.25))
            .collect();
        let dp: Vec<_> = points.iter().map(|&x| x as f64).collect();
        let df = if kind == 3 {
            freq.iter().map(|&x| x as f64).collect()
        } else {
            modes(&[19])
        };
        let di: Vec<_> = input
            .iter()
            .map(|z| Complex::new(z.re as f64, z.im as f64))
            .collect();
        let reference = if kind == 2 {
            direct(&df, &dp, 1, &di, points.len(), -1.0)
        } else {
            direct(&dp, &df, 1, &di, 19, -1.0)
        };
        let actual: Vec<_> = plan
            .execute_owned(&input)
            .unwrap()
            .iter()
            .map(|z| Complex::new(z.re as f64, z.im as f64))
            .collect();
        assert!(error(&actual, &reference, &di) <= 1e-4);
    }
}

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    width: u32,
    kind: usize,
    direction: String,
    normalization: String,
    dimension: usize,
    shape: Vec<usize>,
    points: Vec<Vec<f64>>,
    frequencies: Vec<Vec<f64>>,
    input: Vec<[f64; 2]>,
    expected: Vec<[f64; 2]>,
}
#[derive(serde::Deserialize)]
struct Fixtures {
    cases: Vec<Fixture>,
}
fn fixture<S: RealScalar>(f: &Fixture, eps: f64)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    let points: Vec<_> = f.points.iter().flatten().copied().map(convert).collect();
    let frequencies: Vec<_> = f
        .frequencies
        .iter()
        .flatten()
        .copied()
        .map(convert)
        .collect();
    let ps = [f.points.len(), f.dimension];
    let fs = [f.frequencies.len(), f.dimension];
    let strides = [f.dimension, 1];
    let pv = NdView::new(&points, StridedLayout::new(&ps, &strides).unwrap()).unwrap();
    let fv = NdView::new(&frequencies, StridedLayout::new(&fs, &strides).unwrap()).unwrap();
    let direction = match f.direction.as_str() {
        "forward" => FftDirection::Forward,
        "inverse" => FftDirection::Inverse,
        _ => unreachable!("unknown fixture direction"),
    };
    let normalization = match f.normalization.as_str() {
        "unscaled" => Normalization::Unscaled,
        "forward" => Normalization::Forward,
        "inverse" => Normalization::Inverse,
        "orthonormal" => Normalization::Orthonormal,
        _ => unreachable!("unknown fixture normalization"),
    };
    let opt = NufftPlanSpec::new(direction, normalization, convert(eps))
        .with_limits(limits())
        .with_oversampling(4.0);
    let plan = match f.kind {
        1 => NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: pv,
                modes: &f.shape,
            },
            opt,
            scioxide_fft::FftPlanningContext::default(),
        ),
        2 => NufftPlan::build(
            scioxide_fft::NufftTransform::Type2 {
                points: pv,
                modes: &f.shape,
            },
            opt,
            scioxide_fft::FftPlanningContext::default(),
        ),
        _ => NufftPlan::build(
            scioxide_fft::NufftTransform::Type3 {
                points: pv,
                frequencies: fv,
            },
            opt,
            scioxide_fft::FftPlanningContext::default(),
        ),
    }
    .unwrap();
    let input: Vec<_> = f
        .input
        .iter()
        .map(|x| Complex::new(convert(x[0]), convert(x[1])))
        .collect();
    let actual: Vec<_> = plan
        .execute_owned(&input)
        .unwrap()
        .iter()
        .map(|z| {
            Complex::new(
                z.re.try_to_f64_exact().unwrap(),
                z.im.try_to_f64_exact().unwrap(),
            )
        })
        .collect();
    let reference: Vec<_> = f
        .expected
        .iter()
        .map(|x| Complex::new(x[0], x[1]))
        .collect();
    let di: Vec<_> = f.input.iter().map(|x| Complex::new(x[0], x[1])).collect();
    let count = f.input.len() as f64;
    let scale = match normalization {
        Normalization::Forward if direction == FftDirection::Forward => 1.0 / count,
        Normalization::Inverse if direction == FftDirection::Inverse => 1.0 / count,
        Normalization::Orthonormal => 1.0 / count.sqrt(),
        _ => 1.0,
    };
    let e = error(&actual, &reference, &di) / scale;
    assert!(
        e <= eps,
        "{} type {} f{} {:?} {:?}: {e:e} > {eps:e}",
        f.name,
        f.kind,
        f.width,
        direction,
        normalization
    );
}
#[test]
fn independent_120_digit_fixtures_and_requested_accuracy_sweeps() {
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/nufft.json")).unwrap();
    for f in fixtures.cases {
        if f.width == 32 {
            for eps in [1e-2, 1e-3, 1e-4] {
                fixture::<f32>(&f, eps);
            }
        } else {
            for eps in [1e-3, 1e-6, 1e-9, 1e-11] {
                fixture::<f64>(&f, eps);
            }
        }
    }
}

#[test]
fn dimensions_empty_sums_and_degenerate_type3_boxes() {
    for d in [0, 1, 4, 8] {
        let points = vec![1.25; 3 * d];
        let freq = vec![0.75; 2 * d];
        let ps = [3, d];
        let fs = [2, d];
        let strides = [d, 1];
        let plan = NufftPlan::build(
            scioxide_fft::NufftTransform::Type3 {
                points: view(&points, &ps, &strides),
                frequencies: view(&freq, &fs, &strides),
            },
            options(1e-6),
            scioxide_fft::FftPlanningContext::default(),
        )
        .unwrap();
        assert_eq!(plan.dimension(), d);
        assert_eq!(plan.diagnostics().grid_points, 1);
        let input = vec![Complex::new(1.0, 0.25); 3];
        let expected = direct(&points, &freq, d, &input, 2, -1.0);
        assert!(error(&plan.execute_owned(&input).unwrap(), &expected, &input) < 1e-12);
    }
    let ps = [0, 2];
    let strides = [2, 1];
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&[], &ps, &strides),
            modes: &[2, 3],
        },
        options(1e-6),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    assert_eq!(
        plan.execute_owned(&[]).unwrap(),
        vec![Complex::new(0.0, 0.0); 6]
    );
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type2 {
            points: view(&[], &ps, &strides),
            modes: &[2, 3],
        },
        options(1e-6),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    assert!(
        plan.execute_owned(&[Complex::new(1.0, 0.0); 6])
            .unwrap()
            .is_empty()
    );
    let points = [0.25, 0.5, -0.1, 0.7, 0.2, 0.3, -0.3, 0.1];
    let ps = [2, 4];
    let strides = [4, 1];
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &ps, &strides),
            modes: &[1, 3, 1, 1],
        },
        options(1e-8),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let input = [Complex::new(1.0, 0.0); 2];
    let freq = modes(&[1, 3, 1, 1]);
    assert!(
        error(
            &plan.execute_owned(&input).unwrap(),
            &direct(&points, &freq, 4, &input, 3, -1.0),
            &input
        ) < 1e-8
    );
}

#[test]
fn strided_real_complex_batches_and_block_operators() {
    use scioxide_core::{
        BlockOperator, BlockVectorView, BlockVectorViewMut, ExecutionContext, NdViewMut,
    };
    use scioxide_linalg::{VectorView, VectorViewMut};
    let points = [0.25, -1.5, 0.25];
    let shape = [3, 1];
    let strides = [1, 1];
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[5],
        },
        options(1e-8),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let input = [
        Complex::new(1.0, 0.25),
        Complex::new(-0.5, 0.75),
        Complex::new(2.0, -0.25),
    ];
    let reference = plan.execute_owned(&input).unwrap();
    let mut out = vec![Complex::new(99.0, 0.0); 10];
    let mut work = scioxide_fft::FftWorkspace::from_vec(vec![
        Complex::new(0.0, 0.0);
        plan.workspace_requirements(
            scioxide_fft::AxisWorkspaceRequest::Axis {
                shape: &[2, plan.input_len()],
                axis: 1
            }
        )
        .map(|r| r.complex_len())
        .unwrap()
    ]);
    let ins = [2, 3];
    let inst = [0, 1];
    let outs = [2, 5];
    let outst = [1, 2];
    plan.execute_axis_into(
        NdView::new(&input, StridedLayout::new_broadcast(&ins, &inst).unwrap()).unwrap(),
        1,
        NdViewMut::new(&mut out, StridedLayout::new(&outs, &outst).unwrap()).unwrap(),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (j, &v) in reference.iter().enumerate() {
        assert_eq!(out[2 * j], v);
        assert_eq!(out[2 * j + 1], v);
    }
    let mut scalar_out = vec![Complex::new(0.0, 0.0); 5];
    plan.execute_into(
        VectorView::from_slice(&input).conjugate(),
        VectorViewMut::from_slice(&mut scalar_out),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(
        scalar_out,
        plan.execute_owned(&input.map(|z| z.conj())).unwrap()
    );
    let real = [1.0, -0.5, 2.0];
    let re_expected = plan
        .execute_owned(&real.map(|v| Complex::new(v, 0.0)))
        .unwrap();
    plan.execute_into(
        VectorView::from_slice(&real),
        VectorViewMut::from_slice(&mut scalar_out),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(scalar_out, re_expected);
    plan.execute_axis_into(
        NdView::new(&real, StridedLayout::new_broadcast(&ins, &inst).unwrap()).unwrap(),
        1,
        NdViewMut::new(&mut out, StridedLayout::new(&outs, &outst).unwrap()).unwrap(),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (j, &v) in re_expected.iter().enumerate() {
        assert_eq!(out[2 * j], v);
        assert_eq!(out[2 * j + 1], v);
    }
    let block_input: Vec<_> = input.iter().chain(&input).copied().collect();
    let mut block_work = vec![Complex::new(0.0, 0.0); plan.apply_block_workspace_len(2).unwrap()];
    plan.apply_block_into(
        BlockVectorView::new(&block_input, 3, 2).unwrap(),
        &mut BlockVectorViewMut::new(&mut out, 5, 2).unwrap(),
        &mut block_work,
    )
    .unwrap();
    assert_eq!(&out[..5], reference);
    assert_eq!(&out[5..], reference);
    let before = out.clone();
    let mut bad = block_input;
    bad[5].re = f64::NAN;
    assert!(
        plan.apply_block_into(
            BlockVectorView::new(&bad, 3, 2).unwrap(),
            &mut BlockVectorViewMut::new(&mut out, 5, 2).unwrap(),
            &mut block_work
        )
        .is_err()
    );
    assert_eq!(out, before);
    let bad_view = NdView::new(&bad, StridedLayout::new(&ins, &[3, 1]).unwrap()).unwrap();
    assert!(
        plan.execute_axis_into(
            bad_view,
            1,
            NdViewMut::new(&mut out, StridedLayout::new(&outs, &outst).unwrap()).unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(out, before);
    assert_eq!(
        plan.workspace_requirements(scioxide_fft::AxisWorkspaceRequest::Axis {
            shape: &[0, plan.input_len()],
            axis: 1
        })
        .map(|r| r.complex_len())
        .unwrap(),
        0
    );
    plan.execute_axis_into(
        NdView::<f64>::new(&[], StridedLayout::new(&[0, 3], &[3, 1]).unwrap()).unwrap(),
        1,
        NdViewMut::new(&mut [], StridedLayout::new(&[0, 5], &[5, 1]).unwrap()).unwrap(),
        &mut scioxide_fft::FftWorkspace::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn normalization_permutation_and_periodic_shift_properties() {
    use scioxide_fft::Normalization;
    let points = [-1.0, 0.0, 0.5, 2.0];
    let shape = [4, 1];
    let strides = [1, 1];
    let input = [Complex::new(1.0, 0.25); 4];
    let base = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[7],
        },
        options(1e-8),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap()
    .execute_owned(&input)
    .unwrap();
    for normalization in [
        Normalization::Forward,
        Normalization::Inverse,
        Normalization::Orthonormal,
        Normalization::Unscaled,
    ] {
        let mut opt = options(1e-8);
        opt = NufftPlanSpec::new(opt.direction(), normalization, opt.accuracy())
            .with_limits(opt.limits());
        let actual = NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&points, &shape, &strides),
                modes: &[7],
            },
            opt,
            scioxide_fft::FftPlanningContext::default(),
        )
        .unwrap()
        .execute_owned(&input)
        .unwrap();
        let scale = match normalization {
            Normalization::Forward => 1.0 / 4.0,
            Normalization::Orthonormal => 0.5,
            _ => 1.0,
        };
        for (a, b) in actual.iter().zip(&base) {
            assert!((*a - *b * scale).norm() < 1e-14);
        }
    }
    let shifted = points.map(|x| x + 8.0 * std::f64::consts::TAU);
    let actual = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&shifted, &shape, &strides),
            modes: &[7],
        },
        options(1e-8),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap()
    .execute_owned(&input)
    .unwrap();
    assert!(error(&actual, &base, &input) < 2e-8);
    let reversed: Vec<_> = points.iter().rev().copied().collect();
    let actual = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&reversed, &shape, &strides),
            modes: &[7],
        },
        options(1e-8),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap()
    .execute_owned(&input)
    .unwrap();
    assert!(error(&actual, &base, &input) < 1e-13);
}

#[test]
fn invalid_inputs_cancellation_and_overflow_are_transactional() {
    use scioxide_core::ExecutionContext;
    let points = [0.0, 0.25, -1.0];
    let shape = [3, 1];
    let strides = [1, 1];
    assert!(
        NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&points, &shape, &strides),
                modes: &[0]
            },
            options(1e-7),
            scioxide_fft::FftPlanningContext::default()
        )
        .is_err()
    );
    assert!(
        NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&points, &shape, &strides),
                modes: &[usize::MAX]
            },
            options(1e-7),
            scioxide_fft::FftPlanningContext::default()
        )
        .is_err()
    );
    let exact_index_limit = 1_usize << (usize::BITS.min(52) - 2);
    let error = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[exact_index_limit],
        },
        options(1e-7),
        scioxide_fft::FftPlanningContext::default(),
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::SizeOverflow);
    assert_eq!(
        match error.view() {
            scioxide_core::SciOxideErrorView::SizeOverflow { context, .. } => Some(context),
            _ => None,
        }
        .unwrap(),
        scioxide_core::quantity!("nufft_grid_extent")
    );
    assert!(
        NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&points, &shape, &strides),
                modes: &[4]
            },
            options(1e-17),
            scioxide_fft::FftPlanningContext::default()
        )
        .is_err()
    );
    let bad = [f64::NAN, 0.0, 1.0];
    assert!(
        NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&bad, &shape, &strides),
                modes: &[4]
            },
            options(1e-7),
            scioxide_fft::FftPlanningContext::default()
        )
        .is_err()
    );
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[4],
        },
        options(1e-7),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let input = [Complex::new(1.0, 0.5); 3];
    let mut output = vec![Complex::new(17.0, 19.0); 4];
    let before = output.clone();
    let mut work = plan.workspace(()).unwrap();
    for after in [0, 1, 3] {
        let stop = CancelAfter::new(after);
        assert!(
            plan.execute_into(
                &input,
                &mut output,
                &mut work,
                ExecutionContext::sequential().with_cancellation(&stop)
            )
            .is_err()
        );
        assert_eq!(output, before);
    }
    assert!(
        plan.execute_into(
            &[Complex::new(f64::MAX, 0.0); 3],
            &mut output,
            &mut work,
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(output, before);
    let mut opt = options(1e-7);
    opt = NufftPlanSpec::new(
        opt.direction(),
        scioxide_fft::Normalization::Forward,
        opt.accuracy(),
    )
    .with_limits(opt.limits());
    let normalized = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[4],
        },
        opt,
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    normalized
        .execute_into(
            &[Complex::new(f64::MAX / 2.0, 0.0); 3],
            &mut output,
            &mut work,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    assert!(output.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential_nufft() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, ThreadExecutor};
    use std::num::NonZeroUsize;
    let points: Vec<_> = (0..64).map(|j| (j as f64 * 0.17).sin()).collect();
    let shape = [64, 1];
    let strides = [1, 1];
    let input = vec![Complex::new(1.0, 0.5); 64];
    let executor = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let context = ExecutionContext::new(&executor);
    let mut opt = options(1e-7);
    opt = opt.with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let parallel = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[32769],
        },
        opt,
        context,
    )
    .unwrap();
    let mut output = vec![Complex::new(0.0, 0.0); 32769];
    let mut work = parallel.workspace(()).unwrap();
    parallel
        .execute_into(&input, &mut output, &mut work, context)
        .unwrap();
    opt = opt.with_execution_policy(ExecutionPolicy::Sequential);
    let sequential = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[32769],
        },
        opt,
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap()
    .execute_owned(&input)
    .unwrap();
    assert!(error(&output, &sequential, &input) < 1e-12);
}

#[test]
fn solve_context_counts_admission_and_stops_at_budget() {
    use scioxide_core::{
        Budget, BudgetKind, EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let points = [0.0, 0.5];
    let shape = [2, 1];
    let strides = [1, 1];
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &strides),
            modes: &[4],
        },
        options(1e-7),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(511),
        scioxide_core::algorithm_id!("fft", "nufft", 1),
        2,
        4,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let input = [Complex::new(1.0, 0.5); 2];
    let mut output = vec![Complex::new(0.0, 0.0); 4];
    let mut work = plan.workspace(()).unwrap();
    plan.execute_into(&input, &mut output, &mut work, &mut context)
        .unwrap();
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    let before = output.clone();
    let error = plan
        .execute_into(&input, &mut output, &mut work, &mut context)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let scioxide_core::SciOxideErrorView::ConvergenceFailure {
        reason: _failure_reason,
        diagnostics: _failure_diagnostics,
        detail: _failure_detail,
    } = error.view()
    else {
        unreachable!("unexpected error kind")
    };
    assert_eq!(
        _failure_reason,
        scioxide_core::ConvergenceReason::EvaluationLimit
    );
    let diagnostics = _failure_diagnostics;
    assert_eq!(diagnostics.budget().unwrap().consumed(), 1);
    assert_eq!(
        diagnostics
            .counts()
            .own()
            .provider(EvaluationRole::LinearOperator),
        0
    );
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    assert_eq!(output, before);
}

#[test]
fn scoped_nufft_adjoint_inherits_cancellation() {
    use scioxide_core::{
        CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let points = [0.0_f64, 0.5, 1.0];
    let shape = [3, 1];
    let p = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: view(&points, &shape, &[1, 1]),
            modes: &[4],
        },
        options(1e-6),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let input = vec![Complex::new(1.0, 0.25); p.output_len()];
    let mut output = vec![Complex::new(99.0, 0.0); p.input_len()];
    let initial = output.clone();
    let mut scratch = p.workspace(()).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "nufft-adjoint", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            id,
            &mut budgets,
        )
        .unwrap();
    let result = p.apply_adjoint_into_scoped(
        &input,
        &mut output,
        &mut scratch,
        &mut scioxide_core::OperatorContext::new(&mut context),
    );
    assert!(matches!(
        result.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: scioxide_core::ExecutionStopReason::Cancelled,
            ..
        }
    ));
    assert_eq!(output, initial);
}

#[test]
fn every_normalization_sign_and_type_uses_input_count() {
    use scioxide_fft::{Normalization, NufftTransform};
    let points = [-0.75, 0.125, 0.625];
    let frequencies = [-1.25, 0.5];
    let shape = [3, 1];
    let fs = [2, 1];
    let strides = [1, 1];
    let uniform = modes(&[5]);
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let sign = if direction == FftDirection::Forward {
            -1.0
        } else {
            1.0
        };
        for normalization in [
            Normalization::Unscaled,
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Orthonormal,
        ] {
            for kind in 1..=3 {
                let transform = match kind {
                    1 => NufftTransform::Type1 {
                        points: view(&points, &shape, &strides),
                        modes: &[5],
                    },
                    2 => NufftTransform::Type2 {
                        points: view(&points, &shape, &strides),
                        modes: &[5],
                    },
                    _ => NufftTransform::Type3 {
                        points: view(&points, &shape, &strides),
                        frequencies: view(&frequencies, &fs, &strides),
                    },
                };
                let plan = NufftPlan::new(
                    transform,
                    NufftPlanSpec::new(direction, normalization, 1e-10),
                )
                .unwrap();
                let input: Vec<_> = (0..plan.input_len())
                    .map(|j| Complex::new(0.5 + j as f64 / 8.0, -0.25))
                    .collect();
                let reference = match kind {
                    1 => direct(&points, &uniform, 1, &input, 5, sign),
                    2 => direct(&uniform, &points, 1, &input, 3, sign),
                    _ => direct(&points, &frequencies, 1, &input, 2, sign),
                };
                let n = input.len() as f64;
                let scale = match normalization {
                    Normalization::Forward if direction == FftDirection::Forward => 1.0 / n,
                    Normalization::Inverse if direction == FftDirection::Inverse => 1.0 / n,
                    Normalization::Orthonormal => 1.0 / n.sqrt(),
                    _ => 1.0,
                };
                let reference: Vec<_> = reference.into_iter().map(|z| z * scale).collect();
                assert!(error(&plan.execute_owned(&input).unwrap(), &reference, &input) <= 1e-10);
                let v = vec![Complex::new(0.25, -0.5); plan.output_len()];
                let mut adjoint = vec![Complex::new(0.0, 0.0); plan.input_len()];
                plan.apply_adjoint_into(&v, &mut adjoint, &mut plan.workspace(()).unwrap())
                    .unwrap();
                let forward = plan.execute_owned(&input).unwrap();
                let lhs: Complex<f64> = forward.iter().zip(&v).map(|(a, b)| a.conj() * *b).sum();
                let rhs: Complex<f64> =
                    input.iter().zip(&adjoint).map(|(a, b)| a.conj() * *b).sum();
                assert!((lhs - rhs).norm() <= 1e-10);
            }
        }
    }
}

#[test]
fn accuracy_errors_and_shared_unbounded_defaults() {
    use scioxide_core::{SciOxideErrorKind, SciOxideErrorView};
    use scioxide_fft::{Normalization, NufftTransform};
    for accuracy in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.1,
        0.0,
        1.0,
        2.0,
    ] {
        let error = NufftPlan::new(
            NufftTransform::Type1 {
                points: view(&[0.0], &[1, 1], &[1, 1]),
                modes: &[3],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy),
        )
        .unwrap_err();
        if accuracy.is_finite() {
            assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
            assert!(
                matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("spec.accuracy"))
            );
        } else {
            assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        }
    }
    for accuracy in [1e-18, 1e-30] {
        let error = NufftPlan::new(
            NufftTransform::Type1 {
                points: view(&[0.0], &[1, 1], &[1, 1]),
                modes: &[3],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    }
    assert_eq!(NufftLimits::default().preparation_bytes, usize::MAX);
    assert_eq!(NufftLimits::default().workspace_bytes, usize::MAX);
    let output = scioxide_fft::nufft_type1(
        view(&[0.0], &[1, 1], &[1, 1]),
        &[Complex::new(1.0, 0.0)],
        &[3],
        FftDirection::Forward,
        Normalization::Unscaled,
        1e-8,
    )
    .unwrap();
    assert!(
        output
            .iter()
            .all(|z| (*z - Complex::new(1.0, 0.0)).norm() < 1e-8)
    );
}

fn large_accuracy_sweep<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_fft::{Normalization, NufftTransform};
    let single = S::EPSILON.try_to_f64_exact().unwrap() > 1e-10;
    let requests: &[f64] = if single {
        &[1e-3, 1e-4, 1e-5]
    } else {
        &[1e-6, 1e-9, 1e-10, 1e-12]
    };
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    for shape in [
        vec![10_000],
        vec![100_000],
        vec![400, 250],
        vec![50, 50, 40],
    ] {
        let d = shape.len();
        let n: usize = shape.iter().product();
        // Small opposite dimension keeps independent full direct sums affordable.
        let points: Vec<S> = (0..7 * d)
            .map(|j| convert((((j * 17) % 31) as f64 - 15.0) / 256.0))
            .collect();
        let exact_points: Vec<_> = points
            .iter()
            .map(|v| v.try_to_f64_exact().unwrap())
            .collect();
        let uniform = modes(&shape);
        let frequencies: Vec<S> = (0..5 * d)
            .map(|j| convert(((j * 7 % 13) as f64 / 12.0 - 0.5) * shape[j % d] as f64))
            .collect();
        let exact_frequencies: Vec<_> = frequencies
            .iter()
            .map(|v| v.try_to_f64_exact().unwrap())
            .collect();
        for &accuracy in requests {
            for kind in 1..=3 {
                let ps = [7, d];
                let fs = [5, d];
                let strides = [d, 1];
                let pv = NdView::new(&points, StridedLayout::new(&ps, &strides).unwrap()).unwrap();
                let transform = match kind {
                    1 => NufftTransform::Type1 {
                        points: pv,
                        modes: &shape,
                    },
                    2 => NufftTransform::Type2 {
                        points: pv,
                        modes: &shape,
                    },
                    _ => NufftTransform::Type3 {
                        points: pv,
                        frequencies: NdView::new(
                            &frequencies,
                            StridedLayout::new(&fs, &strides).unwrap(),
                        )
                        .unwrap(),
                    },
                };
                let result = NufftPlan::new(
                    transform,
                    NufftPlanSpec::new(
                        FftDirection::Forward,
                        Normalization::Unscaled,
                        convert(accuracy),
                    ),
                );
                let plan =
                    result.unwrap_or_else(|e| panic!("{shape:?} type {kind}, {accuracy}: {e}"));
                let input: Vec<_> = (0..plan.input_len())
                    .map(|j| {
                        Complex::new(
                            convert(((j * 13 % 29) as f64 - 14.0) / 32.0),
                            convert(((j * 7 % 19) as f64 - 9.0) / 32.0),
                        )
                    })
                    .collect();
                let exact_input: Vec<_> = input
                    .iter()
                    .map(|z| {
                        Complex::new(
                            z.re.try_to_f64_exact().unwrap(),
                            z.im.try_to_f64_exact().unwrap(),
                        )
                    })
                    .collect();
                let reference = match kind {
                    1 => direct(&exact_points, &uniform, d, &exact_input, n, -1.0),
                    2 => direct(&uniform, &exact_points, d, &exact_input, 7, -1.0),
                    _ => direct(&exact_points, &exact_frequencies, d, &exact_input, 5, -1.0),
                };
                let actual: Vec<_> = plan
                    .execute_owned(&input)
                    .unwrap()
                    .iter()
                    .map(|z| {
                        Complex::new(
                            z.re.try_to_f64_exact().unwrap(),
                            z.im.try_to_f64_exact().unwrap(),
                        )
                    })
                    .collect();
                let observed = error(&actual, &reference, &exact_input);
                assert!(
                    observed <= accuracy,
                    "{shape:?} type {kind}, f{}: {observed:e}>{accuracy:e}",
                    if single { 32 } else { 64 }
                );
                let estimate =
                    plan.diagnostics().approximation_error + plan.diagnostics().roundoff_error;
                assert!(
                    observed <= estimate,
                    "{shape:?} type {kind}: {observed:e}>{estimate:e}"
                );
            }
        }
    }
}

#[test]
fn independent_large_accuracy_sweep_f64() {
    large_accuracy_sweep::<f64>();
}
#[test]
fn independent_large_accuracy_sweep_f32() {
    large_accuracy_sweep::<f32>();
}

#[test]
fn alias_estimates_track_worst_offset_errors() {
    use scioxide_fft::{Normalization, NufftTransform};
    for shape in [vec![32], vec![16, 16], vec![16, 16, 16]] {
        let d = shape.len();
        for accuracy in [1e-6, 1e-9, 1e-12] {
            let mut observed = 0.0_f64;
            let mut estimate = 0.0_f64;
            for offset in 0..17 {
                // A single unit source removes averaging/cancellation from the
                // input-L1 metric. Highest centered modes exercise the band edge.
                let coordinates: Vec<_> = shape
                    .iter()
                    .map(|&n| std::f64::consts::TAU * (offset as f64 / 17.0) / (2 * n) as f64)
                    .collect();
                let plan = NufftPlan::new(
                    NufftTransform::Type1 {
                        points: view(&coordinates, &[1, d], &[d, 1]),
                        modes: &shape,
                    },
                    NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy),
                )
                .unwrap();
                let actual = plan.execute_owned(&[Complex::new(1.0, 0.0)]).unwrap();
                let reference = direct(
                    &coordinates,
                    &modes(&shape),
                    d,
                    &[Complex::new(1.0, 0.0)],
                    actual.len(),
                    -1.0,
                );
                observed = observed.max(error(&actual, &reference, &[Complex::new(1.0, 0.0)]));
                estimate =
                    plan.diagnostics().approximation_error + plan.diagnostics().roundoff_error;
            }
            assert!(
                observed <= estimate,
                "{shape:?} {accuracy}: {observed:e} > {estimate:e}"
            );
            assert!(
                estimate <= 10.0 * observed,
                "{shape:?} {accuracy}: estimate {estimate:e} exceeds ten times {observed:e}"
            );
        }
    }
}

#[test]
fn kernel_support_uses_the_accuracy_remaining_above_roundoff() {
    use scioxide_fft::{Normalization, NufftTransform};
    let points = [-0.75, 0.125, 0.5, 1.25];
    let prepare = |accuracy| {
        NufftPlan::new(
            NufftTransform::Type1 {
                points: view(&points, &[4, 1], &[1, 1]),
                modes: &[64],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy),
        )
    };
    let floor = prepare(1e-9).unwrap().diagnostics().roundoff_error;
    let accuracy = 1.01 * floor;
    let plan = prepare(accuracy).unwrap();
    assert!(plan.diagnostics().approximation_error + plan.diagnostics().roundoff_error <= accuracy);
    let input = [Complex::new(1.0, 0.25); 4];
    let reference = direct(&points, &modes(&[64]), 1, &input, 64, -1.0);
    assert!(error(&plan.execute_owned(&input).unwrap(), &reference, &input) <= accuracy);
    assert_eq!(
        prepare(0.99 * floor).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}

fn large_type3_targets<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_fft::{Normalization, NufftTransform};
    let accuracy = if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
        1e-4
    } else {
        1e-12
    };
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    let input: Vec<_> = (0..4)
        .map(|j| {
            Complex::new(
                convert(1.0 / (j + 1) as f64),
                convert(if j % 2 == 0 { 0.125 } else { -0.125 }),
            )
        })
        .collect();
    let exact_input: Vec<_> = input
        .iter()
        .map(|z| {
            Complex::new(
                z.re.try_to_f64_exact().unwrap(),
                z.im.try_to_f64_exact().unwrap(),
            )
        })
        .collect();
    for d in 1..=3 {
        let points: Vec<_> = (0..4 * d)
            .map(|j| convert(((j / d * 11 + j % d * 7) % 17) as f64 / 16.0 - 0.5))
            .collect();
        let frequencies: Vec<_> = (0..100_000 * d)
            .map(|j| {
                convert(16.0 * ((j / d * [7, 13, 17][j % d]) % 100_003) as f64 / 100_003.0 - 8.0)
            })
            .collect();
        let plan = NufftPlan::new(
            NufftTransform::Type3 {
                points: NdView::new(&points, StridedLayout::new(&[4, d], &[d, 1]).unwrap())
                    .unwrap(),
                frequencies: NdView::new(
                    &frequencies,
                    StridedLayout::new(&[100_000, d], &[d, 1]).unwrap(),
                )
                .unwrap(),
            },
            NufftPlanSpec::new(
                FftDirection::Forward,
                Normalization::Unscaled,
                convert(accuracy),
            ),
        )
        .unwrap();
        let exact_points: Vec<_> = points
            .iter()
            .map(|x| x.try_to_f64_exact().unwrap())
            .collect();
        let exact_frequencies: Vec<_> = frequencies
            .iter()
            .map(|x| x.try_to_f64_exact().unwrap())
            .collect();
        let reference = direct(
            &exact_points,
            &exact_frequencies,
            d,
            &exact_input,
            100_000,
            -1.0,
        );
        let actual: Vec<_> = plan
            .execute_owned(&input)
            .unwrap()
            .iter()
            .map(|z| {
                Complex::new(
                    z.re.try_to_f64_exact().unwrap(),
                    z.im.try_to_f64_exact().unwrap(),
                )
            })
            .collect();
        let observed = error(&actual, &reference, &exact_input);
        assert!(
            observed <= accuracy,
            "type-3 {d}D, 100000 targets: {observed:e} > {accuracy:e}"
        );
        assert!(
            observed <= plan.diagnostics().approximation_error + plan.diagnostics().roundoff_error
        );
    }
}
#[test]
fn independent_100000_type3_targets_f64() {
    large_type3_targets::<f64>();
}
#[test]
fn independent_100000_type3_targets_f32() {
    large_type3_targets::<f32>();
}

#[test]
fn rounding_floor_selects_kernel_width() {
    use scioxide_fft::{Normalization, NufftTransform};
    fn column<S: RealScalar, const N: usize>(x: &[S; N]) -> NdView<'_, '_, S> {
        NdView::new(x, StridedLayout::new(&[N, 1], &[1, 1]).unwrap()).unwrap()
    }
    fn one<S: RealScalar, const N: usize>(points: &[S; N]) -> NufftTransform<'_, '_, S> {
        NufftTransform::Type1 {
            points: column(points),
            modes: &[64],
        }
    }
    fn three<'a, S: RealScalar>(
        points: &'a [S; 3],
        frequencies: &'a [S; 3],
    ) -> NufftTransform<'a, 'a, S> {
        NufftTransform::Type3 {
            points: column(points),
            frequencies: column(frequencies),
        }
    }
    fn prepared<S: RealScalar>(
        transform: NufftTransform<'_, '_, S>,
        accuracy: f64,
    ) -> Result<NufftPlan<S>, scioxide_core::SciOxideError> {
        let accuracy = S::try_from_f64_lossy(accuracy).unwrap();
        let spec = NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy);
        NufftPlan::new(transform, spec)
    }
    fn check<S: RealScalar>(plan: &NufftPlan<S>, width: usize, grid: usize) {
        let diagnostics = plan.diagnostics();
        assert_eq!(diagnostics.kernel_support, width);
        assert_eq!(diagnostics.grid_points, grid);
        assert_eq!(plan.grid_shape(), &[grid]);
        assert!(diagnostics.approximation_error < diagnostics.roundoff_error);
        assert!(
            diagnostics.approximation_error + diagnostics.roundoff_error
                <= plan.spec().accuracy().try_to_f64_exact().unwrap()
        );
    }
    // In every case aliasing alone admits one width less; the rounding floor of
    // the prepared grid then requires the next width.
    let points = [-0.75, 0.125, 0.5, 1.25];
    let single = points.map(|x| x as f32);
    check(&prepared(one(&points), 5e-15).unwrap(), 18, 128);
    check(&prepared(one(&single), 1.5e-6).unwrap(), 9, 128);
    check(&prepared(one(&single), 5e-6).unwrap(), 8, 128);
    // Periodic argument reduction raises the type-2 floor.
    let wrapped = [4.0, -5.0, 0.25];
    let two = NufftTransform::Type2 {
        points: column(&wrapped),
        modes: &[1000],
    };
    check(&prepared(two, 5e-13).unwrap(), 16, 2048);
    // The widened type-3 base grid crosses into the next power of two.
    let (sources, frequencies) = ([-1.0, 0.25, 2.0], [-30.0, 4.0, 17.5]);
    check(
        &prepared(three(&sources, &frequencies), 5e-15).unwrap(),
        18,
        256,
    );
    let (sources, frequencies) = (sources.map(|x| x as f32), frequencies.map(|x| x as f32));
    check(
        &prepared(three(&sources, &frequencies), 2e-6).unwrap(),
        9,
        128,
    );
    assert_eq!(
        prepared(one(&points), 4e-15).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn type2_interpolation_meets_the_f32_rounding_floor() {
    use scioxide_fft::{Normalization, NufftTransform};
    // Single band-edge or zero modes remove averaging from the input-L1 metric;
    // requests just above the rounding floor select the widest kernels and thus
    // the most interpolation products per target. A zero mode makes every
    // product positive, where plain summation rounding would grow with w^d.
    for shape in [vec![32], vec![16, 16], vec![8, 8, 8]] {
        let d = shape.len();
        let n: usize = shape.iter().product();
        let count = 400;
        let points: Vec<f32> = (0..count * d)
            .map(|j| ((j * 37 % 101) as f32 / 101.0 - 0.5) * std::f32::consts::TAU)
            .collect();
        let exact_points: Vec<_> = points.iter().map(|&x| f64::from(x)).collect();
        let prepare = |accuracy: f64| {
            NufftPlan::new(
                NufftTransform::Type2 {
                    points: NdView::new(&points, StridedLayout::new(&[count, d], &[d, 1]).unwrap())
                        .unwrap(),
                    modes: &shape,
                },
                NufftPlanSpec::new(
                    FftDirection::Forward,
                    Normalization::Unscaled,
                    accuracy as f32,
                ),
            )
        };
        let floor = prepare(1e-3).unwrap().diagnostics().roundoff_error;
        let (accuracy, plan) = (0..64)
            .map(|k| 1.02 * floor * 1.03_f64.powi(k))
            .find_map(|accuracy| prepare(accuracy).ok().map(|plan| (accuracy, plan)))
            .unwrap();
        let estimate = plan.diagnostics().approximation_error + plan.diagnostics().roundoff_error;
        let center: usize = (0..d)
            .map(|a| (shape[a] / 2) * shape[a + 1..].iter().product::<usize>())
            .sum();
        for mode in [0, center] {
            let mut input = vec![Complex::new(0.0_f32, 0.0); n];
            input[mode] = Complex::new(1.0, 0.0);
            let exact_input: Vec<_> = input
                .iter()
                .map(|z| Complex::new(f64::from(z.re), f64::from(z.im)))
                .collect();
            let reference = direct(&modes(&shape), &exact_points, d, &exact_input, count, -1.0);
            let actual: Vec<_> = plan
                .execute_owned(&input)
                .unwrap()
                .iter()
                .map(|z| Complex::new(f64::from(z.re), f64::from(z.im)))
                .collect();
            let observed = error(&actual, &reference, &exact_input);
            assert!(
                observed <= estimate && observed <= accuracy,
                "{shape:?} mode {mode}: {observed:e} > {estimate:e} or {accuracy:e}"
            );
        }
    }
}

#[test]
fn coordinate_preparation_respects_padding_and_broadcasts() {
    use scioxide_fft::NufftTransform;
    let packed = [-0.5, 0.25, 0.5, -0.25, 0.0, 0.75];
    let mut padded = [f64::NAN; 20];
    for j in 0..3 {
        for d in 0..2 {
            padded[j * 7 + d * 2] = packed[j * 2 + d];
        }
    }
    let prepare = |points| {
        NufftPlan::new(
            NufftTransform::Type1 {
                points,
                modes: &[4, 5],
            },
            options(1e-10),
        )
        .unwrap()
    };
    let input = [Complex::new(0.5, -0.25); 3];
    let expected = prepare(view(&packed, &[3, 2], &[2, 1]))
        .execute_owned(&input)
        .unwrap();
    let actual = prepare(view(&padded, &[3, 2], &[7, 2]))
        .execute_owned(&input)
        .unwrap();
    assert!(error(&actual, &expected, &input) < 1e-12);
    let broadcast = prepare(
        NdView::new(
            &packed[..2],
            StridedLayout::new_broadcast(&[3, 2], &[0, 1]).unwrap(),
        )
        .unwrap(),
    )
    .execute_owned(&input)
    .unwrap();
    let repeated = [
        packed[0], packed[1], packed[0], packed[1], packed[0], packed[1],
    ];
    let expected = prepare(view(&repeated, &[3, 2], &[2, 1]))
        .execute_owned(&input)
        .unwrap();
    assert!(error(&broadcast, &expected, &input) < 1e-12);
}

#[path = "support/operator_accounting.rs"]
mod operator_accounting;

#[test]
fn operator_evaluations_are_one_per_application_or_nonempty_axis() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::AxisWorkspaceRequest;
    use scioxide_fft::{FftPlanningContext, FftProviderPolicy};
    let provider = provider::DirectProvider::new(provider::Mode::Direct);
    for supplied in [false, true] {
        let planning = if supplied {
            FftPlanningContext::with_provider(
                scioxide_core::ExecutionContext::sequential_native(),
                &provider,
                FftProviderPolicy::Required,
            )
        } else {
            FftPlanningContext::default()
        };
        let points = [0.0, 0.5, -0.25, 0.75, 0.1, -0.1, 0.2, -0.2];
        let plan = NufftPlan::build(
            scioxide_fft::NufftTransform::Type1 {
                points: view(&points, &[8, 1], &[1, 1]),
                modes: &[8],
            },
            options(1e-7),
            planning,
        )
        .unwrap();
        for value in [0.0, 1.0] {
            let input = [Complex::new(value, 0.0); 8];
            let mut work = plan.workspace(()).unwrap();
            operator_accounting::admission(8, |output, context| {
                plan.execute_into(&input, output, &mut work, context)
            });
            operator_accounting::operators(&plan, &input);
            let shape = [2, 8];
            let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
            let mut work = plan
                .workspace(AxisWorkspaceRequest::Axis {
                    shape: &shape,
                    axis: 1,
                })
                .unwrap();
            operator_accounting::admission(16, |output, context| {
                plan.execute_axis_into(
                    NdView::new(&[value; 16], layout).unwrap(),
                    1,
                    NdViewMut::new(output, layout).unwrap(),
                    &mut work,
                    context,
                )
            });
            let shape = [0, 8];
            let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
            let mut work = plan
                .workspace(AxisWorkspaceRequest::Axis {
                    shape: &shape,
                    axis: 1,
                })
                .unwrap();
            operator_accounting::empty_axis(|context| {
                plan.execute_axis_into(
                    NdView::new(&[] as &[f64], layout).unwrap(),
                    1,
                    NdViewMut::new(&mut [], layout).unwrap(),
                    &mut work,
                    context,
                )
            });
        }
    }
}

#[path = "support/provider.rs"]
mod provider;

#[test]
fn nufft_operator_adapters_validate_shapes_before_publication() {
    use scioxide_core::{AdjointOperator, LinearOperator, NdView, StridedLayout};
    use scioxide_fft::{NufftLimits, NufftPlan, NufftPlanSpec};
    let points = [0.0_f64, 0.3];
    let shape = [2, 1];
    let strides = [1, 1];
    let options = NufftPlanSpec::new(
        scioxide_fft::FftDirection::Forward,
        scioxide_fft::Normalization::Unscaled,
        1e-6,
    )
    .with_limits(NufftLimits {
        preparation_bytes: 1 << 24,
        workspace_bytes: 1 << 24,
        grid_points: 1 << 16,
        stencil_visits: 1 << 20,
        kernel_support: 128,
    });
    let plan = NufftPlan::build(
        scioxide_fft::NufftTransform::Type1 {
            points: NdView::new(&points, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
            modes: &[4],
        },
        options,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = [Complex::new(17.0, 3.0); 4];
    let before = output;
    assert_eq!(
        plan.apply_into(&[Complex::new(1.0, 0.0)], &mut output, &mut workspace)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, before);
    assert_eq!(
        plan.apply_adjoint_into(&[Complex::new(1.0, 0.0); 4], &mut output, &mut workspace)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, before);
    let input = [Complex::new(1.0, 0.25); 2];
    let expected = plan.execute_owned(&input).unwrap();
    plan.apply_into(&input, &mut output, &mut workspace)
        .unwrap();
    assert!(
        output
            .iter()
            .zip(expected)
            .all(|(a, b)| (*a - b).norm() < 1e-12)
    );
}
