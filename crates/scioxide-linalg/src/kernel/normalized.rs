//! Standard packed products over validated, binary-scaled operands.
struct WorkspaceProduct<'a, 'b, S: NumericScalar> {
    left: MatrixView<'a, S>,
    right: MatrixView<'a, S>,
    output: MatrixViewMut<'b, S>,
    config: GemmConfig<S>,
}
struct PackedPanels<'a, S> {
    a: &'a mut [S],
    b: &'a mut [S],
}
use super::*;
use crate::FiniteMatrixView;
use scioxide_core::backend::PartitionSlot;
use scioxide_core::{AlgebraicScalar, RealScalar};

pub(crate) fn bounded<S: NumericScalar>(
    left: FiniteMatrixView<'_, S>,
    right: FiniteMatrixView<'_, S>,
    mut output: MatrixViewMut<'_, S>,
    config: GemmConfig<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
    plan: GemmPlan,
) -> Result<(), SciOxideError> {
    let ae = standard::exponent(left.maximum)?;
    let be = standard::exponent(right.maximum)?;
    let alpha_e = standard::exponent(standard::magnitude(config.alpha))?;
    let beta_e = standard::exponent(standard::magnitude(config.beta))?;
    let right_maximum = right.maximum;
    let left = left.as_view();
    let right = right.as_view();
    let context = context.execution();
    let mut prior_max = S::Real::ZERO;
    if config.beta != S::ZERO {
        for row in 0..left.rows {
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_none_or(|triangle| triangle_contains(triangle, row, col))
                {
                    let value = output.get_element(row, col);
                    crate::checked::validate_input(value, "prior output")?;
                    prior_max = prior_max.max(standard::magnitude(value));
                }
            }
        }
    }
    let ce = standard::exponent(prior_max)?;
    let growth = usize::BITS as i32 - left.cols.max(1).leading_zeros() as i32 + 4;
    let product_bound = if config.alpha == S::ZERO {
        0
    } else {
        ae + be + alpha_e + growth
    };
    let prior_bound = if config.beta == S::ZERO {
        0
    } else {
        ce + beta_e + 3
    };
    let bound = product_bound.max(prior_bound).max(0);
    // An overflow-prone range is first computed in the unit frame. Normalizing
    // each operand by its own maximum can flush a small factor whose product
    // with a large one is ordinary, so only entries that overflowed there are
    // recomputed with normalized operands below.
    let unit_first = bound >= standard::exponent(S::Real::MAX)?
        || (config.alpha != S::ZERO && ae + alpha_e + 2 >= standard::exponent(S::Real::MAX)?);
    // Folding alpha into A can round an entry below the normal range. The
    // normalized frame avoids that; an overflow-prone unit frame instead
    // recomputes the affected rows exactly below.
    let folding = standard::folding_underflows(
        config.alpha,
        || standard::nonzero_minimum(left),
        right_maximum,
    )?;
    let scaled = unit_first
        || folding
        || (ae + be + alpha_e
            < standard::exponent(S::Real::MIN_POSITIVE)? + S::Real::MANTISSA_DIGITS as i32);
    if !scaled && direct_product(left.rows, left.cols, right.cols) {
        // Finite bounds prove representability. All fallible work is complete.
        for row in 0..left.rows {
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_some_and(|t| !triangle_contains(t, row, col))
                {
                    continue;
                }
                let mut sum = if config.beta == S::ZERO {
                    S::ZERO
                } else {
                    config.beta * output.get_element(row, col)
                };
                for inner in 0..left.cols {
                    sum =
                        config.alpha * left.get_element(row, inner) * right.get_element(inner, col)
                            + sum;
                }
                output.set_element(row, col, sum);
            }
        }
        return Ok(());
    }
    if direct_product(left.rows, left.cols, right.cols) {
        workspace.require_candidate(left.rows * right.cols)?;
    } else {
        workspace.require_gemm(&plan)?;
    }
    let crate::workspace::ProductParts {
        packed_a,
        packed_b,
        candidate,
        ..
    } = workspace.product_parts()?;
    let candidate = &mut candidate[..left.rows * right.cols];
    let normalized_alpha = standard::scale(
        standard::scale(config.alpha, -alpha_e),
        ae + be + alpha_e - bound,
    );
    let normalized_prior = |row: usize, col: usize| {
        standard::scale(
            standard::scale(config.beta, -beta_e)
                * standard::scale(output.get_element(row, col), -ce),
            ce + beta_e - bound,
        )
    };
    let frame = scaled && !unit_first;
    let (a_scale, b_scale, restore, alpha) = if frame {
        (ae, be, bound, normalized_alpha)
    } else {
        (0, 0, 0, config.alpha)
    };
    let initialize = config.beta != S::ZERO
        || left.cols == 0
        || direct_product(left.rows, left.cols, right.cols);
    if initialize {
        for row in 0..left.rows {
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_none_or(|triangle| triangle_contains(triangle, row, col))
                {
                    candidate[row * right.cols + col] = if config.beta == S::ZERO {
                        S::ZERO
                    } else if frame {
                        normalized_prior(row, col)
                    } else {
                        config.beta * output.get_element(row, col)
                    };
                }
            }
        }
    }
    if direct_product(left.rows, left.cols, right.cols) {
        for row in 0..left.rows {
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_none_or(|triangle| triangle_contains(triangle, row, col))
                {
                    let mut sum = candidate[row * right.cols + col];
                    for inner in 0..left.cols {
                        sum = alpha
                            * standard::scale(left.get_element(row, inner), -a_scale)
                            * standard::scale(right.get_element(inner, col), -b_scale)
                            + sum;
                    }
                    candidate[row * right.cols + col] = sum;
                }
            }
        }
    } else {
        type Slot<'a, S> = PartitionSlot<(&'a mut [S], &'a mut [S], &'a mut [S])>;
        let mut slots: [Slot<'_, S>; 8] = core::array::from_fn(|_| PartitionSlot::empty());
        let mut remaining = &mut *candidate;
        let mut a = &mut packed_a[..];
        let mut b = &mut packed_b[..];
        for part in plan.execution.partitions() {
            let first = part.range().start * plan.spec.mc;
            let last = (part.range().end * plan.spec.mc).min(left.rows);
            let (chunk, rest) =
                crate::workspace::split_mut(remaining, (last - first) * right.cols)?;
            remaining = rest;
            let (packed_a, rest) = crate::workspace::split_mut(a, plan.packed_a_per_worker)?;
            a = rest;
            let (packed_b, rest) = crate::workspace::split_mut(b, plan.packed_b_per_worker)?;
            b = rest;
            slots[part.worker_index()] = PartitionSlot::new((chunk, packed_a, packed_b));
        }
        context.execute_partitioned(
            plan.execution,
            &slots[..plan.execution.worker_count()],
            &|part, (candidate, a, b), _| {
                let rows = candidate.len() / right.cols;
                workspace_tiles(
                    WorkspaceProduct {
                        left,
                        right,
                        output: MatrixViewMut::from_slice(rows, right.cols, candidate)?,
                        config: GemmConfig {
                            alpha,
                            beta: if initialize { S::ONE } else { S::ZERO },
                            ..config
                        },
                    },
                    part.range().start * plan.spec.mc,
                    plan,
                    PackedPanels { a, b },
                    context,
                    (-a_scale, -b_scale),
                )
            },
        )?;
    }

    if scaled {
        let normal = standard::exponent(S::Real::MIN_POSITIVE)?;
        for row in 0..left.rows {
            let folds_below = unit_first
                && folding
                && (0..left.cols).try_fold(false, |below, inner| {
                    let value = standard::magnitude(left.get_element(row, inner));
                    Ok::<_, SciOxideError>(
                        below
                            || (value != S::Real::ZERO
                                && alpha_e + standard::exponent(value)? - 1 < normal),
                    )
                })?;
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_none_or(|triangle| triangle_contains(triangle, row, col))
                {
                    let value = candidate[row * right.cols + col];
                    candidate[row * right.cols + col] = if folds_below {
                        let mut sum = scioxide_core::ProductSum::new();
                        for inner in 0..left.cols {
                            sum.add_triple_product(
                                config.alpha,
                                left.get_element(row, inner),
                                right.get_element(inner, col),
                            )?;
                        }
                        if config.beta != S::ZERO {
                            sum.add_product(config.beta, output.get_element(row, col))?;
                        }
                        sum.finish()?
                    } else if unit_first && !value.has_finite_components() {
                        let prior = if config.beta == S::ZERO {
                            S::ZERO
                        } else {
                            normalized_prior(row, col)
                        };
                        let sum = (0..left.cols).fold(prior, |sum, inner| {
                            normalized_alpha
                                * standard::scale(left.get_element(row, inner), -ae)
                                * standard::scale(right.get_element(inner, col), -be)
                                + sum
                        });
                        standard::restore(sum, bound)?
                    } else {
                        standard::restore(value, restore)?
                    };
                }
            }
        }
    }
    for row in 0..left.rows {
        let columns = match config.triangle {
            None => 0..right.cols,
            Some(Triangle::Lower) => 0..right.cols.min(row + 1),
            Some(Triangle::Upper) => row.min(right.cols)..right.cols,
        };
        if output.col_stride == 1 {
            let source = &candidate[row * right.cols..][columns.clone()];
            output.data[row * output.row_stride..][columns].copy_from_slice(source);
        } else {
            for col in columns {
                output.set_element(row, col, candidate[row * right.cols + col]);
            }
        }
    }
    Ok(())
}

/// Write bounded updates directly into the caller's staging workspace.
pub(crate) fn workspace_bounded<S: NumericScalar>(
    left: FiniteMatrixView<'_, S>,
    right: FiniteMatrixView<'_, S>,
    mut output: MatrixViewMut<'_, S>,
    config: GemmConfig<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    context.execution().check()?;
    if left.rows() == 0 || right.cols() == 0 {
        return Ok(());
    }
    let prior = if config.beta == S::ZERO {
        S::Real::ZERO
    } else {
        config
            .prior
            .unwrap_or_else(|| standard::component_maximum(output.as_view(), config.triangle))
    };
    // Workspace arithmetic can overflow before final factor publication.
    // Reject a non-finite bound before interpreting its binary exponent.
    crate::checked::finite_candidate(S::from_real(left.maximum))?;
    crate::checked::finite_candidate(S::from_real(right.maximum))?;
    crate::checked::finite_candidate(S::from_real(prior))?;
    let growth = usize::BITS as i32 - left.cols().max(1).leading_zeros() as i32 + 4;
    let ae = standard::exponent(left.maximum)?;
    let alpha_e = standard::exponent(standard::magnitude(config.alpha))?;
    let product = ae + standard::exponent(right.maximum)? + alpha_e;
    let bound = (product + growth).max(
        standard::exponent(prior)? + standard::exponent(standard::magnitude(config.beta))? + 3,
    );
    let dispatch = config.backend.map_or_else(
        || Ok(context.dispatch()),
        |backend| KernelDispatch::for_backend(backend, context.execution().processor()),
    )?;
    // The unit frame folds alpha into A, so `alpha * A` must itself stay in
    // range, not only the whole product.
    if bound >= standard::exponent(S::Real::MAX)?
        || (config.alpha != S::ZERO && ae + alpha_e + 2 >= standard::exponent(S::Real::MAX)?)
        || product < standard::exponent(S::Real::MIN_POSITIVE)? + S::Real::MANTISSA_DIGITS as i32
        || standard::folding_underflows(
            config.alpha,
            || standard::nonzero_minimum(left.as_view()),
            right.maximum,
        )?
    {
        let plan = GemmPlan::new::<S>(
            left.rows(),
            left.cols(),
            right.cols(),
            policy,
            context.execution(),
            dispatch,
        )?;
        return bounded(left, right, output, config, workspace, context, plan);
    }
    let left = left.as_view();
    let right = right.as_view();
    if left.cols == 0 || (!config.packed && direct_product(left.rows, left.cols, right.cols)) {
        for row in 0..left.rows {
            for col in 0..right.cols {
                if config
                    .triangle
                    .is_some_and(|triangle| !triangle_contains(triangle, row, col))
                {
                    continue;
                }
                let mut sum = if config.beta == S::ZERO {
                    S::ZERO
                } else {
                    config.beta * output.get_element(row, col)
                };
                for inner in 0..left.cols {
                    sum =
                        config.alpha * left.get_element(row, inner) * right.get_element(inner, col)
                            + sum;
                }
                output.set_element(row, col, sum);
            }
        }
        return Ok(());
    }
    let plan = GemmPlan::new::<S>(
        left.rows,
        left.cols,
        right.cols,
        policy,
        context.execution(),
        dispatch,
    )?;
    workspace.require_gemm(&plan)?;
    let crate::workspace::ProductParts {
        packed_a, packed_b, ..
    } = workspace.product_parts()?;
    // Factor workspaces own the surrounding transaction. Partition disjoint
    // row spans directly, preserving the same tile/reduction order per entry.
    // Interleaved layouts remain on the same serial tile kernel.
    let span = (output.cols - 1) * output.col_stride + 1;
    if output.row_stride < span {
        return workspace_tiles(
            WorkspaceProduct {
                left,
                right,
                output,
                config,
            },
            0,
            plan,
            PackedPanels {
                a: &mut packed_a[..plan.packed_a_per_worker],
                b: &mut packed_b[..plan.packed_b_per_worker],
            },
            context.execution(),
            (0, 0),
        );
    }
    type Slot<'a, S> = PartitionSlot<(MatrixViewMut<'a, S>, &'a mut [S], &'a mut [S])>;
    let mut slots: [Slot<'_, S>; 8] = core::array::from_fn(|_| PartitionSlot::empty());
    let row_stride = output.row_stride;
    let col_stride = output.col_stride;
    let cols = output.cols;
    let mut remaining = output.data;
    let mut a = &mut packed_a[..];
    let mut b = &mut packed_b[..];
    for part in plan.execution.partitions() {
        let first = part.range().start * plan.spec.mc;
        let last = (part.range().end * plan.spec.mc).min(left.rows);
        let length = (last - first)
            .saturating_mul(row_stride)
            .min(remaining.len());
        let (chunk, rest) = crate::workspace::split_mut(remaining, length)?;
        remaining = rest;
        let (pa, rest) = crate::workspace::split_mut(a, plan.packed_a_per_worker)?;
        a = rest;
        let (pb, rest) = crate::workspace::split_mut(b, plan.packed_b_per_worker)?;
        b = rest;
        let target = MatrixViewMut::from_slice_with_strides(
            last - first,
            cols,
            chunk,
            row_stride,
            col_stride,
        )?;
        slots[part.worker_index()] = PartitionSlot::new((target, pa, pb));
    }
    context.execution().execute_partitioned(
        plan.execution,
        &slots[..plan.execution.worker_count()],
        &|part, (target, a, b), _| {
            workspace_tiles(
                WorkspaceProduct {
                    left,
                    right,
                    output: target,
                    config,
                },
                part.range().start * plan.spec.mc,
                plan,
                PackedPanels { a, b },
                context.execution(),
                (0, 0),
            )
        },
    )
}

fn workspace_tiles<S: NumericScalar>(
    problem: WorkspaceProduct<'_, '_, S>,
    first: usize,
    plan: GemmPlan,
    panels: PackedPanels<'_, S>,
    context: ExecutionContext<'_>,
    scales: (i32, i32),
) -> Result<(), SciOxideError> {
    let PackedPanels { a, b } = panels;
    let WorkspaceProduct {
        left,
        right,
        mut output,
        config,
    } = problem;

    let spec = plan.spec;
    let kernel = arch::select_microkernel::<S>(plan.dispatch);
    for col in (0..right.cols).step_by(spec.nc) {
        let cols = spec.nc.min(right.cols - col);
        for shared in (0..left.cols).step_by(spec.kc) {
            context.check()?;
            let depth = spec.kc.min(left.cols - shared);
            pack_b(
                right,
                GemmOp::None,
                PackingPanel::new(col, shared, cols, depth, spec.nr).scaled(scales.1),
                b,
            );
            for row in (0..output.rows).step_by(spec.mc) {
                let rows = spec.mc.min(output.rows - row);
                pack_a(
                    left,
                    GemmOp::None,
                    PackingPanel::new(first + row, shared, rows, depth, spec.mr).scaled(scales.0),
                    config.alpha,
                    a,
                );
                apply_packed_block(
                    a,
                    b,
                    &mut output,
                    crate::kernel::PackedBlock {
                        row_start: row,
                        col_start: col,
                        rows,
                        cols,
                        shared: depth,
                        global_row_start: first + row,
                    },
                    crate::kernel::PackedKernel {
                        beta: if shared == 0 { config.beta } else { S::ONE },
                        spec,
                        microkernel: kernel,
                        triangle: config.triangle,
                    },
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod workspace_tests {
    use super::*;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{Executor, PartitionFn, SequentialExecutor};

    struct Recorder {
        workers: NonZeroUsize,
        calls: AtomicUsize,
    }
    impl Executor for Recorder {
        fn worker_budget(&self) -> NonZeroUsize {
            self.workers
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            task: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, task)
        }
    }

    #[test]
    fn bounded_products_dispatch_and_preserve_bits_across_strided_partitions() {
        let (m, k, n) = (147, 41, 37);
        let a = crate::Matrix::from_vec(
            m,
            k,
            (0..m * k)
                .map(|i| ((i * 7) % 31) as f64 / 31.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let b = crate::Matrix::from_vec(
            k,
            n,
            (0..k * n)
                .map(|i| ((i * 11) % 29) as f64 / 29.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let left = a.view().try_finite().unwrap();
        let right = b.view().try_finite().unwrap();

        for triangle in [
            None,
            Some(scioxide_core::Triangle::Lower),
            Some(scioxide_core::Triangle::Upper),
        ] {
            let mut reference = None;
            for workers in [1, 2, 8] {
                let workers = NonZeroUsize::new(workers).unwrap();
                let recorder = Recorder {
                    workers,
                    calls: AtomicUsize::new(0),
                };
                let context = KernelContext::new(ExecutionContext::new(&recorder));
                let policy = ExecutionPolicy::Parallel(workers);
                let mut work = LinalgWorkspace::new();
                crate::kernel::tests::prepare_product(&mut work, m, k, n, policy, context).unwrap();
                let mut output =
                    crate::Matrix::from_vec(m, n + 3, alloc::vec![0.25;m*(n+3)]).unwrap();
                workspace_bounded(
                    left,
                    right,
                    output.view_mut().block_mut(0, 0, m, n).unwrap(),
                    GemmConfig {
                        alpha: 0.7,
                        beta: -0.2,
                        triangle,
                        ..GemmConfig::ordinary()
                    },
                    policy,
                    &mut work,
                    context,
                )
                .unwrap();
                if workers.get() > 1 {
                    assert!(recorder.calls.load(Ordering::Relaxed) > 0);
                }
                for row in 0..m {
                    for col in 0..n + 3 {
                        if col >= n || triangle.is_some_and(|t| !triangle_contains(t, row, col)) {
                            assert_eq!(output.get_element(row, col), 0.25);
                        }
                    }
                }
                let bits: Vec<_> = output.as_slice().iter().map(|x| x.to_bits()).collect();
                if let Some(expected) = &reference {
                    assert_eq!(&bits, expected);
                } else {
                    reference = Some(bits);
                }
            }
        }
    }

    #[test]
    fn out_of_range_alpha_a_takes_the_binary_frame() {
        // Folding alpha into A overflows (2^1200) or underflows (2^-1200)
        // although every product alpha*a*b = 2^(+-600) is ordinary.
        for exponent in [600, -600] {
            let (k, n) = (3, 2);
            let a = crate::Matrix::from_vec(1, k, alloc::vec![2f64.powi(exponent); k]).unwrap();
            let b =
                crate::Matrix::from_vec(k, n, alloc::vec![2f64.powi(-exponent); k * n]).unwrap();
            let context = KernelContext::new(ExecutionContext::sequential());
            let policy = ExecutionPolicy::Sequential;
            let mut work = LinalgWorkspace::new();
            crate::kernel::tests::prepare_product(&mut work, 1, k, n, policy, context).unwrap();
            let mut output = crate::Matrix::from_vec(1, n, alloc::vec![0.0; n]).unwrap();
            workspace_bounded(
                a.view().try_finite().unwrap(),
                b.view().try_finite().unwrap(),
                output.view_mut(),
                GemmConfig {
                    alpha: 2f64.powi(exponent),
                    ..GemmConfig::ordinary()
                },
                policy,
                &mut work,
                context,
            )
            .unwrap();
            let expected = 3.0 * 2f64.powi(exponent);
            for &value in output.as_slice() {
                assert!((value - expected).abs() <= 4.0 * f64::EPSILON * expected);
            }
        }
    }
}
