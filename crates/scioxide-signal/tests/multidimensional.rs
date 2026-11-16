#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    CancellationToken, Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar,
    SciOxideErrorKind, StridedLayout,
};
use scioxide_signal::{
    AnalyticSignalNdPlan, RieszPlan, SignalExecutionOptions, SignalWorkspace, monogenic_nd,
    riesz_nd,
};
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn dense_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    let mut out = vec![0; shape.len()];
    for (slot, n) in out.iter_mut().zip(shape).rev() {
        *slot = stride;
        stride *= n;
    }
    out
}
fn view<'a, T>(x: &'a [T], shape: &'a [usize], strides: &'a [usize]) -> NdView<'a, 'a, T> {
    NdView::new(x, StridedLayout::new(shape, strides).unwrap()).unwrap()
}
fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for (c, n) in out.iter_mut().zip(shape).rev() {
        *c = index % n;
        index /= n;
    }
    out
}
// Independent full finite sums: complementary coordinates select independent batches.
fn direct(
    x: &[Complex<f64>],
    shape: &[usize],
    axes: &[usize],
    spacing: &[f64],
    component: Option<usize>,
) -> Vec<Complex<f64>> {
    let n = x.len();
    let coords: Vec<_> = (0..n).map(|i| coordinates(i, shape)).collect();
    let transformed: usize = axes.iter().map(|&a| shape[a]).product();
    let mut spectrum = vec![Complex::new(0., 0.); n];
    for (k, out) in spectrum.iter_mut().enumerate() {
        for (j, &value) in x.iter().enumerate() {
            if (0..shape.len()).any(|a| !axes.contains(&a) && coords[k][a] != coords[j][a]) {
                continue;
            }
            let phase = -core::f64::consts::TAU
                * axes
                    .iter()
                    .map(|&a| (coords[k][a] * coords[j][a]) as f64 / shape[a] as f64)
                    .sum::<f64>();
            *out += value * Complex::new(phase.cos(), phase.sin());
        }
        let symbol = if let Some(component) = component {
            let frequencies: Vec<_> = axes
                .iter()
                .zip(spacing)
                .map(|(&a, &h)| {
                    let k = coords[k][a];
                    let signed = if k > shape[a] / 2 {
                        k as f64 - shape[a] as f64
                    } else {
                        k as f64
                    };
                    signed / (shape[a] as f64 * h)
                })
                .collect();
            let norm = frequencies.iter().map(|w| w * w).sum::<f64>().sqrt();
            let a = axes[component];
            let nyquist = shape[a].is_multiple_of(2) && coords[k][a] == shape[a] / 2;
            Complex::new(
                0.,
                if norm == 0. || nyquist {
                    0.
                } else {
                    -frequencies[component] / norm
                },
            )
        } else {
            let gain = axes
                .iter()
                .map(|&a| {
                    let k = coords[k][a];
                    let n = shape[a];
                    if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                        1.
                    } else if k <= n / 2 {
                        2.
                    } else {
                        0.
                    }
                })
                .product();
            Complex::new(gain, 0.)
        };
        *out *= symbol;
    }
    let mut result = vec![Complex::new(0., 0.); n];
    for (j, out) in result.iter_mut().enumerate() {
        for (k, &value) in spectrum.iter().enumerate() {
            if (0..shape.len()).any(|a| !axes.contains(&a) && coords[k][a] != coords[j][a]) {
                continue;
            }
            let phase = core::f64::consts::TAU
                * axes
                    .iter()
                    .map(|&a| (coords[k][a] * coords[j][a]) as f64 / shape[a] as f64)
                    .sum::<f64>();
            *out += value * Complex::new(phase.cos(), phase.sin()) / transformed as f64;
        }
    }
    result
}
fn references<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let tol = if S::EPSILON.into_f64() < 1e-10 {
        3e-12
    } else {
        3e-5
    };
    for shape in [&[3, 4][..], &[4, 5], &[2, 3, 4], &[1, 5]] {
        let strides = dense_strides(shape);
        let n: usize = shape.iter().product();
        let mut selections = vec![&[0][..], &[1], &[1, 0]];
        if shape.len() == 3 {
            selections.push(&[2, 0, 1]);
        }
        for axes in selections {
            let real: Vec<_> = (0..n)
                .map(|i| s::<S>((i as f64 * 0.47).sin() + 0.2))
                .collect();
            let complex: Vec<_> = real
                .iter()
                .enumerate()
                .map(|(i, &x)| Complex::new(x, s::<S>((i as f64 * 0.29).cos())))
                .collect();
            let spacing: Vec<_> = axes.iter().map(|&a| s::<S>(1. + a as f64 * 0.5)).collect();
            let spacing64: Vec<_> = axes.iter().map(|&a| 1. + a as f64 * 0.5).collect();
            let r64: Vec<_> = real
                .iter()
                .map(|&x| Complex::new(x.into_f64(), 0.))
                .collect();
            let z64: Vec<_> = complex
                .iter()
                .map(|z| Complex::new(z.re.into_f64(), z.im.into_f64()))
                .collect();
            let analytic = AnalyticSignalNdPlan::new(shape, axes, SignalExecutionOptions::new())
                .unwrap()
                .analytic(view(&real, shape, &strides))
                .unwrap();
            let expected = direct(&r64, shape, axes, &spacing64, None);
            for (actual, expected) in analytic.iter().zip(expected) {
                assert!((actual.re.into_f64() - expected.re).abs() < tol);
                assert!((actual.im.into_f64() - expected.im).abs() < tol);
            }
            let real_result = riesz_nd(view(&real, shape, &strides), axes, &spacing).unwrap();
            let complex_result = riesz_nd(view(&complex, shape, &strides), axes, &spacing).unwrap();
            let tuple = monogenic_nd(view(&real, shape, &strides), axes, &spacing).unwrap();
            assert_eq!(&tuple[..n], &real);
            assert_eq!(&tuple[n..], &real_result);
            for c in 0..axes.len() {
                let expected = direct(&r64, shape, axes, &spacing64, Some(c));
                let complex_expected = direct(&z64, shape, axes, &spacing64, Some(c));
                for j in 0..n {
                    assert!(expected[j].im.abs() < 1e-12);
                    assert!((real_result[c * n + j].into_f64() - expected[j].re).abs() < tol);
                    let actual = complex_result[c * n + j];
                    assert!((actual.re.into_f64() - complex_expected[j].re).abs() < tol);
                    assert!((actual.im.into_f64() - complex_expected[j].im).abs() < tol);
                }
            }
        }
    }
}
#[test]
fn direct_reference_both_precisions() {
    references::<f32>();
    references::<f64>();
}

fn plane_waves<S: RealScalar>() {
    let tol = if S::EPSILON.into_f64() < 1e-10 {
        3e-12
    } else {
        3e-5
    };
    let shape = [16, 16];
    let strides = [16, 1];
    for (wave, spacing) in [
        ([1., 2.], [1., 1.]),
        ([2., -1.], [1., 1.]),
        ([-1., -2.], [1., 1.]),
        ([1., 2.], [0.5, 2.]),
    ] {
        let x: Vec<_> = (0..256)
            .map(|j| {
                s::<S>(
                    (core::f64::consts::TAU
                        * (wave[0] * (j / 16) as f64 + wave[1] * (j % 16) as f64)
                        / 16.)
                        .cos(),
                )
            })
            .collect();
        let h = spacing.map(s::<S>);
        let p = RieszPlan::new(&shape, &[0, 1], &h, SignalExecutionOptions::new()).unwrap();
        let z = p.monogenic(view(&x, &shape, &strides)).unwrap();
        let norm = (wave[0] / spacing[0]).hypot(wave[1] / spacing[1]);
        for j in 0..256 {
            let phase = core::f64::consts::TAU
                * (wave[0] * (j / 16) as f64 + wave[1] * (j % 16) as f64)
                / 16.;
            for c in 0..2 {
                assert!(
                    (z[(c + 1) * 256 + j].into_f64() - wave[c] / spacing[c] / norm * phase.sin())
                        .abs()
                        < tol
                );
            }
            let amplitude = z[j].hypot(z[256 + j].hypot(z[512 + j]));
            assert!((amplitude - S::ONE).abs() < s::<S>(tol));
        }
    }
}
#[test]
fn physical_orientation_and_rotation_covariance() {
    plane_waves::<f32>();
    plane_waves::<f64>();
}

#[test]
fn strided_views_and_axis_order() {
    let x: Vec<_> = (0..12).map(|i| (i as f64).sin()).collect();
    let original = RieszPlan::new(&[3, 4], &[0, 1], &[1., 2.], SignalExecutionOptions::new())
        .unwrap()
        .riesz(view(&x, &[3, 4], &[4, 1]))
        .unwrap();
    // Transpose both sample axes and permute component order.
    let p = RieszPlan::new(&[4, 3], &[0, 1], &[2., 1.], SignalExecutionOptions::new()).unwrap();
    let mut output = vec![77.; 40];
    let mut scratch = p.workspace(()).unwrap();
    p.riesz_into(
        view(&x, &[4, 3], &[1, 4]),
        NdViewMut::new(
            &mut output,
            StridedLayout::new(&[2, 4, 3], &[20, 1, 5]).unwrap(),
        )
        .unwrap(),
        &mut scratch,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for c in 0..2 {
        for a in 0..4 {
            for b in 0..3 {
                assert!(
                    (output[c * 20 + a + b * 5] - original[(1 - c) * 12 + b * 4 + a]).abs() < 1e-12
                );
            }
        }
    }
    assert_eq!(output[4], 77.);
    let broadcast = NdView::new(
        &[2.],
        StridedLayout::new_broadcast(&[3, 4], &[0, 0]).unwrap(),
    )
    .unwrap();
    let z = p.monogenic(broadcast); // Wrong shape still fails before publication.
    assert_eq!(error_kind(z), SciOxideErrorKind::DimensionMismatch);
    let p = RieszPlan::new(&[3, 4], &[0, 1], &[1., 1.], SignalExecutionOptions::new()).unwrap();
    let broadcast = NdView::new(
        &[2.],
        StridedLayout::new_broadcast(&[3, 4], &[0, 0]).unwrap(),
    )
    .unwrap();
    let z = p.monogenic(broadcast).unwrap();
    assert_eq!(&z[..12], &[2.; 12]);
    assert!(z[12..].iter().all(|x| x.abs() < 1e-12));
}

#[test]
fn invalid_inputs_and_transactional_failure() {
    for (axes, kind) in [
        (&[][..], SciOxideErrorKind::InvalidInput),
        (&[2], SciOxideErrorKind::OutOfBounds),
        (&[0, 0], SciOxideErrorKind::InvalidInput),
    ] {
        assert_eq!(
            error_kind(AnalyticSignalNdPlan::<f64>::new(
                &[3, 4],
                axes,
                SignalExecutionOptions::new()
            )),
            kind
        );
    }
    assert_eq!(
        error_kind(AnalyticSignalNdPlan::<f64>::new(
            &[0, 4],
            &[0],
            SignalExecutionOptions::new()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(AnalyticSignalNdPlan::<f64>::new(
            &[usize::MAX, 2],
            &[0],
            SignalExecutionOptions::new()
        )),
        SciOxideErrorKind::SizeOverflow
    );
    for (h, kind) in [
        (&[][..], SciOxideErrorKind::DimensionMismatch),
        (&[0.], SciOxideErrorKind::InvalidInput),
        (&[-1.], SciOxideErrorKind::InvalidInput),
        (&[f64::NAN], SciOxideErrorKind::NonFinite),
        (&[f64::INFINITY], SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(
            error_kind(RieszPlan::<f64>::new(
                &[4],
                &[0],
                h,
                SignalExecutionOptions::new()
            )),
            kind
        );
    }
    let p = RieszPlan::new(&[4], &[0], &[1.], SignalExecutionOptions::new()).unwrap();
    let mut work = p.workspace(()).unwrap();
    let native = ExecutionContext::sequential_native();
    let mut out = [42.; 8];
    for x in [[1., 2., f64::NAN, 4.], [1., 2., f64::INFINITY, 4.]] {
        assert_eq!(
            error_kind(p.monogenic_into(
                view(&x, &[4], &[1]),
                NdViewMut::new(&mut out, StridedLayout::new(&[2, 4], &[4, 1]).unwrap()).unwrap(),
                &mut work,
                native
            )),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(out, [42.; 8]);
    }
    assert_eq!(
        error_kind(p.monogenic_into(
            view(&[1.; 4], &[4], &[1]),
            NdViewMut::new(&mut out, StridedLayout::new(&[2, 4], &[4, 1]).unwrap()).unwrap(),
            &mut SignalWorkspace::new(),
            native
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [42.; 8]);
    let token = CancellationToken::new();
    token.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&token);
    assert_eq!(
        error_kind(p.monogenic_into(
            view(&[1.; 4], &[4], &[1]),
            NdViewMut::new(&mut out, StridedLayout::new(&[2, 4], &[4, 1]).unwrap()).unwrap(),
            &mut work,
            context
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, [42.; 8]);
    let a = AnalyticSignalNdPlan::new(&[4], &[0], SignalExecutionOptions::new()).unwrap();
    let mut output = [Complex::new(42., 0.); 4];
    let mut scratch = a.workspace(()).unwrap();
    assert_eq!(
        error_kind(a.analytic_into(
            view(&[1.; 4], &[4], &[1]),
            NdViewMut::new(&mut output, StridedLayout::new(&[4], &[1]).unwrap()).unwrap(),
            &mut scratch,
            context
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(output, [Complex::new(42., 0.); 4]);
    // Multi-axis doubling makes this finite real input's analytic result unrepresentable.
    let huge: Vec<_> = (0..16)
        .map(|i| f64::MAX * 0.75 * (core::f64::consts::TAU * ((i / 4 + i % 4) as f64) / 4.).cos())
        .collect();
    let a = AnalyticSignalNdPlan::new(&[4, 4], &[0, 1], SignalExecutionOptions::new()).unwrap();
    let mut output = [Complex::new(42., 0.); 16];
    let mut scratch = a.workspace(()).unwrap();
    assert_eq!(
        error_kind(a.analytic_into(
            view(&huge, &[4, 4], &[4, 1]),
            NdViewMut::new(&mut output, StridedLayout::new(&[4, 4], &[4, 1]).unwrap()).unwrap(),
            &mut scratch,
            native
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(output, [Complex::new(42., 0.); 16]);
}
#[test]
fn empty_batches_and_extreme_spacing() {
    let a = AnalyticSignalNdPlan::<f64>::new(&[0, 4], &[1], SignalExecutionOptions::new()).unwrap();
    assert!(a.analytic(view(&[], &[0, 4], &[0, 0])).unwrap().is_empty());
    let p = RieszPlan::<f64>::new(&[0, 4], &[1], &[1.], SignalExecutionOptions::new()).unwrap();
    assert!(p.monogenic(view(&[], &[0, 4], &[0, 0])).unwrap().is_empty());
    let p = RieszPlan::new(
        &[4, 4],
        &[0, 1],
        &[f64::MIN_POSITIVE, f64::MAX],
        SignalExecutionOptions::new(),
    )
    .unwrap();
    let x: Vec<_> = (0..16)
        .map(|i| (core::f64::consts::TAU * (i % 4) as f64 / 4.).cos())
        .collect();
    let y = p.riesz(view(&x, &[4, 4], &[4, 1])).unwrap();
    for (j, &value) in y[16..].iter().enumerate() {
        assert!((value - (core::f64::consts::TAU * (j % 4) as f64 / 4.).sin()).abs() < 1e-12);
    }
}
