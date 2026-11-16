//! Native output adaptation and layout-independent packing contracts.
use super::*;
use alloc::vec;
use scioxide_core::{AlgebraicScalar, Complex, RealScalar};

fn native_tiles<R: RealScalar + NumericScalar<Real = R>>() {
    let value = |x| R::try_from_f64_lossy(x).unwrap();
    let epsilon = R::ONE.scalbn(-((R::MANTISSA_DIGITS / 2 + 1) as i32));
    let fused = (R::ONE + epsilon).mul_add(R::ONE - epsilon, -R::ONE);
    assert_ne!(fused, R::ZERO);
    for backend in KernelSpec::BACKENDS {
        let spec = KernelSpec::for_scalar::<R>(backend);
        assert!(spec.mr * spec.nr <= 128);
        let Ok(dispatch) = KernelDispatch::for_backend(backend, tests::test_processor()) else {
            continue;
        };
        let Some(kernel) = arch::select_microkernel::<R>(dispatch) else {
            continue;
        };
        for rows in 1..=spec.mr {
            for cols in 1..=spec.nr {
                for depth in [2, 3, spec.kc + 1] {
                    let mut a = vec![R::ZERO; depth * spec.mr];
                    let mut b = vec![R::ZERO; depth * spec.nr];
                    a[..rows].fill(-R::ONE);
                    a[spec.mr..spec.mr + rows].fill(R::ONE + epsilon);
                    b[..cols].fill(R::ONE);
                    b[spec.nr..spec.nr + cols].fill(R::ONE - epsilon);
                    for triangle in [None, Some(Triangle::Lower), Some(Triangle::Upper)] {
                        for beta in [R::ZERO, R::ONE, value(-0.5)] {
                            let height = rows + 2;
                            let width = cols + 6;
                            for (rs, cs) in [(width + 3, 1), (2 * width + 3, 2), (1, height + 3)] {
                                let span = (height - 1) * rs + (width - 1) * cs + 1;
                                let mut data = vec![value(73.0); span + 2];
                                let tile = OutputTile {
                                    row_start: 1,
                                    col_start: 5,
                                    rows,
                                    cols,
                                    shared: depth,
                                    mr: spec.mr,
                                    nr: spec.nr,
                                    global_row_start: 7,
                                    triangle,
                                };
                                let mut expected = data.clone();
                                for row in 0..rows {
                                    for col in 0..cols {
                                        let i = 1 + (row + 1) * rs + (col + 5) * cs;
                                        // Unselected entries must never be read or changed.
                                        data[i] = if beta == R::ZERO || !tile.contains(row, col) {
                                            R::NAN
                                        } else {
                                            value(0.25)
                                        };
                                        expected[i] = if tile.contains(row, col) {
                                            if beta == R::ZERO {
                                                fused
                                            } else {
                                                beta.mul_add(value(0.25), fused)
                                            }
                                        } else {
                                            R::NAN
                                        };
                                    }
                                }
                                let mut output = MatrixViewMut::from_slice_with_strides(
                                    height,
                                    width,
                                    &mut data[1..span + 1],
                                    rs,
                                    cs,
                                )
                                .unwrap();
                                assert!(kernel.apply_tile(&a, &b, &mut output, &tile, beta));
                                for (&actual, &expected) in data.iter().zip(&expected) {
                                    assert!(
                                        actual == expected
                                            || (actual.is_nan() && expected.is_nan()),
                                        "{backend:?}: {rows}x{cols}, depth {depth}, strides {rs}/{cs}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_native_leaf_handles_all_partial_tiles_and_masks() {
    native_tiles::<f32>();
    native_tiles::<f64>();
}

/// Complex products on the real leaves: every partial tile, mask, stride and
/// beta class, natively and through the portable fallback. Small integer data
/// make every sum exact, so both evaluations must match exactly.
fn complex_tiles<R: RealScalar + NumericScalar<Real = R>>() {
    type C<R> = Complex<R>;
    let c = |re: f64, im: f64| {
        C::new(
            R::try_from_f64_lossy(re).unwrap(),
            R::try_from_f64_lossy(im).unwrap(),
        )
    };
    let zero = C::<R>::ZERO;
    for backend in KernelSpec::BACKENDS {
        let spec = KernelSpec::for_scalar::<C<R>>(backend);
        let real = KernelSpec::for_scalar::<R>(backend);
        // Complex tiles keep the real register tile, depth and cache footprint.
        assert_eq!(
            (spec.mr, 2 * spec.nr, 2 * spec.kc, 2 * spec.nc, spec.mc),
            (real.mr, real.nr, real.kc, real.nc, real.mc)
        );
        let Ok(dispatch) = KernelDispatch::for_backend(backend, tests::test_processor()) else {
            continue;
        };
        let kernel = arch::select_microkernel::<C<R>>(dispatch);
        assert_eq!(
            kernel.map(Microkernel::shape),
            arch::select_microkernel::<R>(dispatch).map(|_| (real.mr, real.nr))
        );
        for rows in 1..=spec.mr {
            for cols in 1..=spec.nr {
                for depth in [1, 3, spec.kc + 1] {
                    let left = |inner: usize, lane: usize| {
                        c(
                            ((inner + 2 * lane) % 5) as f64 - 2.0,
                            ((3 * inner + lane) % 7) as f64 - 3.0,
                        )
                    };
                    let right = |inner: usize, lane: usize| {
                        c(
                            ((2 * inner + lane) % 7) as f64 - 3.0,
                            ((inner + 3 * lane) % 5) as f64 - 2.0,
                        )
                    };
                    let mut a = vec![zero; depth * spec.mr];
                    let mut b = vec![zero; 2 * depth * spec.nr];
                    pack_entries(&mut a, PanelSide::Left, spec.mr, depth, |inner, lane| {
                        if lane < rows { left(inner, lane) } else { zero }
                    });
                    pack_entries(&mut b, PanelSide::Right, spec.nr, depth, |inner, lane| {
                        if lane < cols {
                            right(inner, lane)
                        } else {
                            zero
                        }
                    });
                    for triangle in [None, Some(Triangle::Lower), Some(Triangle::Upper)] {
                        for beta in [zero, C::ONE, c(-0.5, 0.0), c(0.5, -0.25)] {
                            let height = rows + 2;
                            let width = cols + 6;
                            for (rs, cs) in [(width + 3, 1), (2 * width + 3, 2), (1, height + 3)] {
                                let span = (height - 1) * rs + (width - 1) * cs + 1;
                                let mut data = vec![c(73.0, -73.0); span + 2];
                                let tile = OutputTile {
                                    row_start: 1,
                                    col_start: 5,
                                    rows,
                                    cols,
                                    shared: depth,
                                    mr: spec.mr,
                                    nr: spec.nr,
                                    global_row_start: 7,
                                    triangle,
                                };
                                let mut expected = data.clone();
                                for row in 0..rows {
                                    for col in 0..cols {
                                        let i = 1 + (row + 1) * rs + (col + 5) * cs;
                                        let prior = c(0.25, 0.75);
                                        // Unselected entries must never be read or changed.
                                        data[i] = if beta == zero || !tile.contains(row, col) {
                                            C::new(R::NAN, R::NAN)
                                        } else {
                                            prior
                                        };
                                        expected[i] = if tile.contains(row, col) {
                                            let sum = (0..depth).fold(zero, |sum, inner| {
                                                sum + left(inner, row) * right(inner, col)
                                            });
                                            if beta == zero {
                                                sum
                                            } else {
                                                sum + beta * prior
                                            }
                                        } else {
                                            data[i]
                                        };
                                    }
                                }
                                for native in [kernel, None] {
                                    let mut actual = data.clone();
                                    let mut output = MatrixViewMut::from_slice_with_strides(
                                        height,
                                        width,
                                        &mut actual[1..span + 1],
                                        rs,
                                        cs,
                                    )
                                    .unwrap();
                                    let (a, b) = (components(&a), components(&b));
                                    if let Some(kernel) = native {
                                        assert!(kernel.apply_tile(a, b, &mut output, &tile, beta));
                                    } else {
                                        scalar_microkernel(a, b, &mut output, &tile, beta);
                                    }
                                    for (&actual, &expected) in actual.iter().zip(&expected) {
                                        assert!(
                                            actual == expected
                                                || (actual.is_nan() && expected.is_nan()),
                                            "{backend:?}: {rows}x{cols}, depth {depth}, strides {rs}/{cs}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn complex_products_run_the_real_leaves_on_component_panels() {
    complex_tiles::<f32>();
    complex_tiles::<f64>();
}

fn packing<S: NumericScalar>() {
    let scalar = |r, i| {
        S::from_components(
            S::Real::try_from_f64_lossy(r).unwrap(),
            S::Real::try_from_f64_lossy(i).unwrap(),
        )
    };
    let count = component_count::<S>();
    let data: alloc::vec::Vec<_> = (0..256)
        .map(|i| scalar(i as f64 / 16.0, -0.25 - i as f64 / 64.0))
        .collect();
    // Seven outer lanes in two 4-lane panels of six shared indices.
    let (lanes, depth) = (4, 6);
    let parts = |value: S| [value.real_component(), value.imaginary_component()];
    for (rs, cs) in [(13, 1), (1, 13), (25, 2)] {
        let matrix = MatrixView::from_slice_with_strides(9, 9, &data[1..], rs, cs).unwrap();
        for op in [GemmOp::None, GemmOp::Transpose, GemmOp::Adjoint] {
            for scale in [-2, 0, 2] {
                for alpha in [S::ONE, scalar(-0.75, 0.25)] {
                    let panel = PackingPanel::new(1, 2, 7, depth, lanes).scaled(scale);
                    let mut a = vec![scalar(73.0, 0.0); 2 * depth * lanes + 1];
                    let mut b = vec![scalar(73.0, 0.0); 2 * depth * lanes * count + 1];
                    pack_a(matrix, op, panel, alpha, &mut a);
                    pack_b(matrix, op, panel, &mut b);
                    let mut expected_a = vec![S::Real::ZERO; 2 * depth * lanes * count];
                    let mut expected_b = vec![S::Real::ZERO; 2 * depth * lanes * count * count];
                    for outer in 0..7 {
                        let (block, lane) = (outer / lanes, outer % lanes);
                        for inner in 0..depth {
                            let left = parts(
                                alpha
                                    * standard::scale(
                                        operated_get(matrix, op, 1 + outer, 2 + inner),
                                        scale,
                                    ),
                            );
                            let right = parts(standard::scale(
                                operated_get(matrix, op, 2 + inner, 1 + outer),
                                scale,
                            ));
                            let slot = block * depth + inner;
                            // A: one real row per component.
                            for component in 0..count {
                                expected_a[(slot * count + component) * lanes + lane] =
                                    left[component];
                            }
                            // B: `re, im` per lane, then `-im, re`.
                            let rows = [[right[0], right[1]], [-right[1], right[0]]];
                            for (row, values) in rows.iter().enumerate().take(count) {
                                for component in 0..count {
                                    expected_b[((slot * count + row) * lanes + lane) * count
                                        + component] = values[component];
                                }
                            }
                        }
                    }
                    let len = 2 * depth * lanes;
                    assert_eq!(components(&a[..len]), &expected_a[..]);
                    assert_eq!(components(&b[..len * count]), &expected_b[..]);
                    assert_eq!(a[len], scalar(73.0, 0.0));
                    assert_eq!(b[len * count], scalar(73.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn packers_preserve_transforms_scales_padding_and_strides() {
    packing::<f32>();
    packing::<f64>();
    packing::<Complex<f32>>();
    packing::<Complex<f64>>();
}

#[test]
fn finite_scan_covers_dense_orders_without_reading_padding() {
    for backend in KernelBackend::available(tests::test_processor()) {
        let dispatch = KernelDispatch::for_backend(backend, tests::test_processor()).unwrap();
        for (rs, cs) in [(5, 1), (1, 3), (7, 1), (1, 5), (13, 2)] {
            let mut data = vec![f64::NAN; 50];
            for row in 0..3 {
                for col in 0..5 {
                    data[1 + row * rs + col * cs] = (row * 5 + col) as f64 - 7.0;
                }
            }
            for op in [GemmOp::None, GemmOp::Transpose, GemmOp::Adjoint] {
                let view = MatrixView::from_slice_with_strides(3, 5, &data[1..], rs, cs).unwrap();
                let view = operated_view(view, op);
                let finite = view.try_finite_dispatched(dispatch).unwrap();
                assert_eq!(finite.maximum, view.try_finite().unwrap().maximum);
            }
            data[1 + 2 * rs + 4 * cs] = f64::INFINITY;
            let view = MatrixView::from_slice_with_strides(3, 5, &data[1..], rs, cs).unwrap();
            assert!(view.adjoint().try_finite_dispatched(dispatch).is_err());
        }
        let empty =
            MatrixView::<f64>::from_slice_with_strides(0, usize::MAX, &[], usize::MAX, usize::MAX)
                .unwrap();
        assert_eq!(empty.try_finite_dispatched(dispatch).unwrap().maximum, 0.0);
    }
}

/// Componentwise `|Re| + |Im|`, which bounds every component product sum.
fn magnitude<S: NumericScalar>(value: S) -> S::Real {
    value.real_component().abs() + value.imaginary_component().abs()
}

fn product_bounds<S: NumericScalar>() {
    let real =
        |i: usize| S::Real::try_from_f64_lossy(((i * 17 + 3) % 31) as f64 / 31.0 - 0.5).unwrap();
    let scalar = |i: usize| {
        if S::IS_REAL {
            S::from_real(real(i))
        } else {
            S::from_components(real(i), real(i + 5))
        }
    };
    let count = component_count::<S>();
    for (m, k, n) in [(33, 35, 37), (7, 513, 11), (129, 257, 9)] {
        let a = crate::Matrix::from_vec(m, k, (0..m * k).map(scalar).collect()).unwrap();
        let b =
            crate::Matrix::from_vec(k, n, (0..k * n).map(|i| scalar(i + 13)).collect()).unwrap();
        let at = a.view().transpose().try_to_matrix().unwrap();
        let bt = b.view().transpose().try_to_matrix().unwrap();
        let ah = a.view().adjoint().try_to_matrix().unwrap();
        let bh = b.view().adjoint().try_to_matrix().unwrap();
        let initial = crate::Matrix::from_vec(m, n, vec![scalar(1); m * n]).unwrap();
        let config = GemmConfig {
            alpha: scalar(2),
            beta: scalar(3),
            ..GemmConfig::ordinary()
        };
        let mut exact = initial.clone();
        tests::reference_gemm(a.view(), b.view(), exact.view_mut(), config).unwrap();
        // Each component is a sum of `count * k` real products plus prior terms.
        let eps = S::Real::EPSILON;
        let terms = S::Real::try_from_f64_lossy((count * (k + 3)) as f64).unwrap();
        let gamma = terms * eps / (S::Real::ONE - terms * eps);
        for backend in KernelBackend::available(tests::test_processor()) {
            let mut layout_reference = None;
            for left in [a.view(), at.view().transpose(), ah.view().adjoint()] {
                for right in [b.view(), bt.view().transpose(), bh.view().adjoint()] {
                    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
                        let mut actual = initial.clone();
                        tests::prepared_gemm(
                            left.try_finite().unwrap(),
                            right.try_finite().unwrap(),
                            actual.view_mut(),
                            GemmConfig {
                                backend: Some(backend),
                                ..config
                            },
                            ExecutionPolicy::Sequential,
                            &mut LinalgWorkspace::new(),
                            KernelContext::new(tests::test_context()).with_accuracy(accuracy),
                        )
                        .unwrap();
                        if accuracy == Accuracy::Exact {
                            assert_eq!(actual.as_slice(), exact.as_slice());
                        } else {
                            // Storage order does not change the packed reduction order.
                            if let Some(previous) = &layout_reference {
                                assert_eq!(&actual, previous);
                            }
                            for row in 0..m {
                                for col in 0..n {
                                    let bound = (0..k).fold(S::Real::ZERO, |sum, inner| {
                                        sum + magnitude(config.alpha)
                                            * magnitude(a.get_element(row, inner))
                                            * magnitude(b.get_element(inner, col))
                                    }) + magnitude(config.beta)
                                        * magnitude(initial.get_element(row, col));
                                    let error =
                                        actual.get_element(row, col) - exact.get_element(row, col);
                                    assert!(
                                        error.real_component().abs() <= gamma * bound
                                            && error.imaginary_component().abs() <= gamma * bound,
                                        "{backend:?}, {m}x{k}x{n}"
                                    );
                                }
                            }
                            layout_reference = Some(actual);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn standard_bounds_and_exact_results_cover_skinny_transformed_products() {
    product_bounds::<f32>();
    product_bounds::<f64>();
    product_bounds::<Complex<f32>>();
    product_bounds::<Complex<f64>>();
}

#[cfg(feature = "std")]
fn worker_independence<S: NumericScalar>() {
    let (m, k, n) = (961, 257, 9);
    let value = |i: usize, p: usize, q: usize| {
        let real =
            |i: usize| S::Real::try_from_f64_lossy(((i * p) % q) as f64 / q as f64 - 0.5).unwrap();
        if S::IS_REAL {
            S::from_real(real(i))
        } else {
            S::from_components(real(i), real(i + 3))
        }
    };
    let a = crate::Matrix::from_vec(k, m, (0..k * m).map(|i| value(i, 7, 31)).collect()).unwrap();
    let b = crate::Matrix::from_vec(k, n, (0..k * n).map(|i| value(i, 13, 29)).collect()).unwrap();
    for backend in KernelBackend::available(tests::test_processor()) {
        let mut reference = None;
        for workers in [1, 2, 8] {
            let workers = NonZeroUsize::new(workers).unwrap();
            let executor = scioxide_core::ThreadExecutor::new(workers);
            let context = ExecutionContext::new(&executor).with_processor(tests::test_processor());
            let mut output = crate::Matrix::zeros(m, n).unwrap();
            tests::prepared_gemm(
                a.view().adjoint().try_finite().unwrap(),
                b.view().try_finite().unwrap(),
                output.view_mut(),
                GemmConfig {
                    backend: Some(backend),
                    ..GemmConfig::ordinary()
                },
                ExecutionPolicy::Parallel(workers),
                &mut LinalgWorkspace::new(),
                KernelContext::new(context),
            )
            .unwrap();
            let bits: alloc::vec::Vec<_> = components(output.as_slice())
                .iter()
                .map(|&x| (x, x.is_sign_negative()))
                .collect();
            if let Some(expected) = &reference {
                assert_eq!(&bits, expected, "{backend:?}");
            } else {
                reference = Some(bits);
            }
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn partial_tiles_crossing_depth_blocks_are_bitwise_independent_of_workers() {
    worker_independence::<f64>();
    worker_independence::<Complex<f64>>();
}
