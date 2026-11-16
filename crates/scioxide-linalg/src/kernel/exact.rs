//! Exact exponent-split products evaluated by the packed real microkernels.
//!
//! Each signed slice digit has q bits. With depth d, 2q + ceil(log2 d)
//! is at most the working precision: all products and all partial sums are
//! exact integers. Scale exponents are applied only inside the accumulator.

pub(super) struct ExactProduct<'a, S: NumericScalar> {
    pub(super) left: MatrixView<'a, S>,
    pub(super) right: MatrixView<'a, S>,
    pub(super) config: GemmConfig<S>,
    pub(super) addend: Option<(MatrixView<'a, S>, MatrixView<'a, S>, S)>,
}
struct ExactTile {
    row_start: usize,
    col_start: usize,
    spec: KernelSpec,
    kernel: Option<Microkernel>,
}

use super::*;
use scioxide_core::ExecutionPlan;
use scioxide_core::backend::PartitionSlot;
use scioxide_core::{RangeAccumulator, RealScalar};

const TILE: usize = 32;
const DEPTH: usize = 128;
const WORKERS: usize = 8;

/// Exact scratch geometry; all data live in the shared workspace arenas.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Layout {
    pub(super) tiles: usize,
    pub(super) workers: usize,
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) reals: usize,
    pub(super) accumulators: usize,
}

pub(super) fn tile_count(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
    checked_product(
        &[rows.div_ceil(TILE), cols.div_ceil(TILE)],
        scioxide_core::quantity!("exact_tile_count"),
    )
}

impl Layout {
    pub(super) fn new<S: NumericScalar>(
        rows: usize,
        cols: usize,
        workers: usize,
    ) -> Result<Self, SciOxideError> {
        if rows == 0 || cols == 0 || workers == 0 {
            return Ok(Self::default());
        }
        let q = (S::Real::MANTISSA_DIGITS as usize - 7) / 2;
        let (maximum, exponent, _) = crate::checked::finite_parts(S::Real::MAX)?;
        let (_, minimum, _) = crate::checked::finite_parts(S::Real::MIN_SUBNORMAL)?;
        let span = (i32::from(exponent) + (64 - maximum.leading_zeros()) as i32
            - i32::from(minimum)) as usize;
        let slices = span.div_ceil(q);
        let padded_rows = KernelSpec::BACKENDS
            .into_iter()
            .map(|backend| {
                let spec = KernelSpec::for_scalar::<S::Real>(backend);
                TILE.div_ceil(spec.mr) * spec.mr
            })
            .max()
            .unwrap_or(TILE);
        let a = checked_product(
            &[padded_rows, DEPTH, slices],
            scioxide_core::quantity!("exact_left_slices"),
        )?;
        let b = checked_product(
            &[TILE, DEPTH, slices],
            scioxide_core::quantity!("exact_right_slices"),
        )?;
        let real_stride = checked_add(
            checked_add(a, b, scioxide_core::quantity!("exact_real_stride"))?,
            TILE * TILE,
            scioxide_core::quantity!("exact_real_stride"),
        )?;
        let components = if S::IS_REAL { 1 } else { 2 };
        Ok(Self {
            tiles: checked_product(
                &[rows.div_ceil(TILE), cols.div_ceil(TILE), TILE, TILE],
                scioxide_core::quantity!("exact_tiles"),
            )?,
            workers,
            a,
            b,
            reals: checked_mul(
                real_stride,
                workers,
                scioxide_core::quantity!("exact_reals"),
            )?,
            accumulators: checked_product(
                &[components, TILE, TILE, workers],
                scioxide_core::quantity!("exact_accumulators"),
            )?,
        })
    }
}

pub(super) struct Workspace<'a, S: NumericScalar> {
    pub(super) tiles: &'a mut [S],
    pub(super) reals: &'a mut [S::Real],
    pub(super) accumulators: &'a mut [RangeAccumulator<S::Real>],
    pub(super) layout: Layout,
}

struct Worker<'a, R: RealScalar> {
    a: &'a mut [R],
    b: &'a mut [R],
    partial: &'a mut [R],
    real: &'a mut [RangeAccumulator<R>],
    imaginary: &'a mut [RangeAccumulator<R>],
}

pub(super) fn plan(
    rows: usize,
    shared: usize,
    cols: usize,
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
) -> Result<ExecutionPlan, SciOxideError> {
    let count = tile_count(rows, cols)?;
    let flops = checked_product(
        &[rows, shared, cols, 2],
        scioxide_core::quantity!("exact_gemm_work"),
    )?;
    let policy = if policy == ExecutionPolicy::Auto && flops < GEMM_AUTO_FLOPS {
        ExecutionPolicy::Sequential
    } else {
        policy
    };
    Ok(context.plan(
        policy,
        count,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: GEMM_AUTO_WORKERS,
        },
    ))
}

impl<S: NumericScalar> Workspace<'_, S> {
    pub(super) fn multiply(
        &mut self,
        product: crate::kernel::exact::ExactProduct<'_, S>,
        candidate: &mut [S],
        policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
        dispatch: KernelDispatch,
    ) -> Result<(), SciOxideError> {
        let crate::kernel::exact::ExactProduct {
            left,
            right,
            config,
            addend,
        } = product;

        let plan = plan(left.rows, left.cols, right.cols, policy, context)?;
        let columns = right.cols.div_ceil(TILE);
        let spec = KernelSpec::for_scalar::<S::Real>(dispatch.backend());
        let kernel = arch::select_microkernel::<S::Real>(dispatch);
        type Slot<'a, S> = PartitionSlot<(&'a mut [S], Worker<'a, <S as NumericScalar>::Real>)>;
        let mut slots: [Slot<'_, S>; WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
        let needed = checked_product(
            &[plan.work_items(), TILE, TILE],
            scioxide_core::quantity!("exact_tiles"),
        )?;
        if self.tiles.len() < needed
            || self.layout.workers < plan.worker_count()
            || self.reals.len() < self.layout.reals
            || self.accumulators.len() < self.layout.accumulators
        {
            return Err(SciOxideError::dimension_mismatch(
                [needed, plan.worker_count()],
                [self.tiles.len(), self.layout.workers],
            ));
        }
        let mut remaining = &mut self.tiles[..needed];
        let mut reals = &mut *self.reals;
        let mut accumulators = &mut *self.accumulators;
        let real_stride = self.layout.a + self.layout.b + TILE * TILE;
        let accumulator_stride = TILE * TILE * if S::IS_REAL { 1 } else { 2 };
        for part in plan.partitions() {
            let (chunk, rest) =
                crate::workspace::split_mut(remaining, part.range().len() * TILE * TILE)?;
            remaining = rest;
            let (worker_reals, rest) = crate::workspace::split_mut(reals, real_stride)?;
            reals = rest;
            let (a, rest) = crate::workspace::split_mut(worker_reals, self.layout.a)?;
            let (b, partial) = crate::workspace::split_mut(rest, self.layout.b)?;
            let (worker_accumulators, rest) =
                crate::workspace::split_mut(accumulators, accumulator_stride)?;
            accumulators = rest;
            let (real, imaginary) = crate::workspace::split_mut(worker_accumulators, TILE * TILE)?;
            slots[part.worker_index()] = PartitionSlot::new((
                chunk,
                Worker {
                    a,
                    b,
                    partial,
                    real,
                    imaginary,
                },
            ));
        }
        context.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|part, (output, mut worker), _| {
                for (index, tile) in part.range().zip(output.chunks_mut(TILE * TILE)) {
                    context.check()?;
                    let row = (index / columns) * TILE;
                    let col = (index % columns) * TILE;
                    if config.triangle.is_some_and(|triangle| match triangle {
                        Triangle::Lower => row + TILE.min(left.rows - row) <= col,
                        Triangle::Upper => col + TILE.min(right.cols - col) <= row,
                    }) {
                        continue;
                    }
                    worker.tile(
                        ExactProduct {
                            left,
                            right,
                            config,
                            addend,
                        },
                        candidate,
                        tile,
                        ExactTile {
                            row_start: row,
                            col_start: col,
                            spec,
                            kernel,
                        },
                        context,
                    )?;
                }
                Ok(())
            },
        )?;

        for (index, tile) in self.tiles[..plan.work_items() * TILE * TILE]
            .chunks(TILE * TILE)
            .enumerate()
        {
            let row_start = index / columns * TILE;
            let col_start = index % columns * TILE;
            for row in 0..TILE.min(left.rows - row_start) {
                let len = TILE.min(right.cols - col_start);
                let offset = (row_start + row) * right.cols + col_start;
                for col in 0..len {
                    if config.triangle.is_none_or(|triangle| {
                        triangle_contains(triangle, row_start + row, col_start + col)
                    }) {
                        candidate[offset + col] = tile[row * TILE + col];
                    }
                }
            }
        }
        Ok(())
    }
}

impl<R: RealScalar> Worker<'_, R> {
    fn tile<S: NumericScalar<Real = R>>(
        &mut self,
        product: ExactProduct<'_, S>,
        prior: &[S],
        output: &mut [S],
        tile: ExactTile,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let ExactProduct {
            left,
            right,
            config,
            addend,
        } = product;
        let ExactTile {
            row_start,
            col_start,
            spec,
            kernel,
        } = tile;

        let rows = TILE.min(left.rows - row_start);
        let cols = TILE.min(right.cols - col_start);
        let complex = !matches!(S::kind(), ScalarKind::F32 | ScalarKind::F64);
        self.real.iter_mut().for_each(RangeAccumulator::clear);
        self.imaginary.iter_mut().for_each(RangeAccumulator::clear);
        let selected = |row, col| {
            config.triangle.is_none_or(|triangle| {
                triangle_contains(triangle, row_start + row, col_start + col)
            })
        };
        for (left, right, alpha) in core::iter::once((left, right, config.alpha)).chain(addend) {
            if alpha != S::ZERO {
                for shared in (0..left.cols).step_by(DEPTH) {
                    context.check()?;
                    let depth = DEPTH.min(left.cols - shared);
                    // The fixed DEPTH bound makes slice width backend-independent.
                    let q = (R::MANTISSA_DIGITS as usize - 7) / 2;
                    for (ai, bi, imaginary, negative) in [
                        (false, false, false, false),
                        (true, true, false, true),
                        (false, true, true, false),
                        (true, false, true, false),
                    ]
                    .into_iter()
                    .take(if complex { 4 } else { 1 })
                    {
                        let mut ae = [0; TILE];
                        let mut be = [0; TILE];
                        let a_size = rows.div_ceil(spec.mr) * spec.mr * depth;
                        let b_size = cols.div_ceil(spec.nr) * spec.nr * depth;
                        let a_count = pack_slices(
                            SliceLayout {
                                outer: rows,
                                depth,
                                lanes: spec.mr,
                                q,
                            },
                            &mut ae,
                            self.a,
                            |row, inner| {
                                component(left.get_element(row_start + row, shared + inner), ai)
                            },
                        )?;
                        let b_count = pack_slices(
                            SliceLayout {
                                outer: cols,
                                depth,
                                lanes: spec.nr,
                                q,
                            },
                            &mut be,
                            self.b,
                            |col, inner| {
                                component(right.get_element(shared + inner, col_start + col), bi)
                            },
                        )?;
                        for a in 0..a_count {
                            if self.a[a * a_size..(a + 1) * a_size]
                                .iter()
                                .all(|value| *value == R::ZERO)
                            {
                                continue;
                            }
                            for b in 0..b_count {
                                if self.b[b * b_size..(b + 1) * b_size]
                                    .iter()
                                    .all(|value| *value == R::ZERO)
                                {
                                    continue;
                                }
                                let mut partial =
                                    MatrixViewMut::from_slice(TILE, TILE, self.partial)?;
                                apply_packed_block(
                                    &self.a[a * a_size..(a + 1) * a_size],
                                    &self.b[b * b_size..(b + 1) * b_size],
                                    &mut partial,
                                    crate::kernel::PackedBlock {
                                        row_start: 0,
                                        col_start: 0,
                                        rows,
                                        cols,
                                        shared: depth,
                                        global_row_start: 0,
                                    },
                                    crate::kernel::PackedKernel {
                                        beta: R::ZERO,
                                        spec,
                                        microkernel: kernel,
                                        triangle: None,
                                    },
                                );
                                for (row, &a_exponent) in ae[..rows].iter().enumerate() {
                                    for (col, &b_exponent) in be[..cols].iter().enumerate() {
                                        if !selected(row, col) {
                                            continue;
                                        }
                                        let index = row * TILE + col;
                                        let value = self.partial[index];
                                        if value == R::ZERO {
                                            continue;
                                        }
                                        let exponent =
                                            a_exponent + b_exponent + ((a + b) * q) as i32;
                                        let value = if negative { -value } else { value };
                                        let (re, im) = if imaginary {
                                            (-alpha.imaginary_component(), alpha.real_component())
                                        } else {
                                            (alpha.real_component(), alpha.imaginary_component())
                                        };
                                        self.real[index].add_scaled_product(value, re, exponent)?;
                                        if complex {
                                            self.imaginary[index]
                                                .add_scaled_product(value, im, exponent)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for row in 0..rows {
            for col in 0..cols {
                let index = row * TILE + col;
                let previous = prior[(row_start + row) * right.cols + col_start + col];
                if !selected(row, col) {
                    output[index] = previous;
                    continue;
                }
                if config.beta != S::ZERO {
                    self.real[index]
                        .add_product(config.beta.real_component(), previous.real_component())?;
                    if complex {
                        self.real[index].subtract_product(
                            config.beta.imaginary_component(),
                            previous.imaginary_component(),
                        )?;
                        self.imaginary[index].add_product(
                            config.beta.real_component(),
                            previous.imaginary_component(),
                        )?;
                        self.imaginary[index].add_product(
                            config.beta.imaginary_component(),
                            previous.real_component(),
                        )?;
                    }
                }
                output[index] = S::from_components(
                    self.real[index].finish()?,
                    if complex {
                        self.imaginary[index].finish()?
                    } else {
                        R::ZERO
                    },
                );
            }
        }
        Ok(())
    }
}

fn component<S: NumericScalar>(value: S, imaginary: bool) -> S::Real {
    if imaginary {
        value.imaginary_component()
    } else {
        value.real_component()
    }
}

// An exact radix-2^q decomposition. Each row/column chooses its own least
// occupied exponent; gaps extend the slice count but never lose low bits.
#[derive(Clone, Copy)]
struct SliceLayout {
    outer: usize,
    depth: usize,
    lanes: usize,
    q: usize,
}

fn pack_slices<R: RealScalar>(
    layout: SliceLayout,
    exponents: &mut [i32; TILE],
    packed: &mut [R],
    value: impl Fn(usize, usize) -> R,
) -> Result<usize, SciOxideError> {
    let SliceLayout {
        outer,
        depth,
        lanes,
        q,
    } = layout;
    let mut count = 0;
    for (row, exponent) in exponents[..outer].iter_mut().enumerate() {
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for inner in 0..depth {
            let (bits, e, _) = crate::checked::finite_parts(value(row, inner))?;
            if bits != 0 {
                lo = lo.min(i32::from(e) + bits.trailing_zeros() as i32);
                hi = hi.max(i32::from(e) + (64 - bits.leading_zeros()) as i32);
            }
        }
        *exponent = if lo == i32::MAX { 0 } else { lo };
        if hi != i32::MIN {
            count = count.max(((hi - lo) as usize).div_ceil(q));
        }
    }
    let size = outer.div_ceil(lanes) * lanes * depth;
    packed[..size * count].fill(R::ZERO);
    let mask = (1_u64 << q) - 1;
    for (row, &base) in exponents[..outer].iter().enumerate() {
        for inner in 0..depth {
            let (bits, e, negative) = crate::checked::finite_parts(value(row, inner))?;
            let sign = if negative { -1_i8 } else { 1_i8 };
            if bits == 0 {
                continue;
            }
            for slice in 0..count {
                let shift = base + (slice * q) as i32 - i32::from(e);
                let digit = if shift >= 64 || shift <= -64 {
                    0
                } else if shift >= 0 {
                    (bits >> shift) & mask
                } else {
                    (bits << -shift) & mask
                };
                let digit = R::from_u64_half_open01(digit << (64 - q)).scalbn(q as i32);
                packed[slice * size + (row / lanes * depth + inner) * lanes + row % lanes] =
                    if sign < 0 { -digit } else { digit };
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;
    use crate::kernel::tests::{test_context, test_processor};

    #[test]
    fn work_overflow_precedes_broadcast_input_traversal() {
        let input = [1.0_f64];
        let left = MatrixView::from_slice_with_strides(1, usize::MAX, &input, 0, 0).unwrap();
        let right = MatrixView::from_slice_with_strides(usize::MAX, 1, &input, 0, 0).unwrap();
        let mut output = [7.0];
        let result = left.mul_mat_into(
            right,
            &crate::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
            &mut LinalgWorkspace::new(),
            crate::LinalgExecutionContext::unscoped(
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            ),
        );
        assert!(matches!(
            (result).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert_eq!(output, [7.0]);
    }

    #[test]
    fn long_scalar_products_reserve_only_the_output_candidate() {
        let mut workspace = LinalgWorkspace::<f64>::new();
        crate::kernel::tests::prepare_product(
            &mut workspace,
            1,
            1024,
            1,
            ExecutionPolicy::Sequential,
            test_context(),
        )
        .unwrap();
        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_A).len(), 0);
        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_B).len(), 0);
        assert!(!workspace.scalar_mut(crate::kernel::CANDIDATE).is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn extreme_split_cancellation_matches_every_worker_and_backend() {
        let left = Matrix::from_vec(
            65,
            3,
            (0..195)
                .map(|index| match index % 3 {
                    0 => f64::MAX,
                    1 => -f64::MAX,
                    _ => f64::from_bits(1),
                })
                .collect(),
        )
        .unwrap();
        let right = Matrix::from_vec(3, 65, alloc::vec![1.0; 195]).unwrap();
        let pool = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(8).unwrap());
        let context =
            KernelContext::new(ExecutionContext::new(&pool).with_processor(test_processor()))
                .with_accuracy(Accuracy::Exact);
        let mut workspace = LinalgWorkspace::new();
        let mut output = Matrix::zeros(65, 65).unwrap();
        for backend in KernelBackend::available(test_processor()) {
            for workers in 1..=8 {
                (|| -> Result<(), SciOxideError> {
                    let config = GemmConfig {
                        backend: Some(backend),
                        ..GemmConfig::ordinary()
                    };
                    crate::kernel::tests::prepared_gemm(
                        operated_view(left.view(), config.left_op).try_finite()?,
                        operated_view(right.view(), config.right_op).try_finite()?,
                        output.view_mut(),
                        config,
                        ExecutionPolicy::Parallel(NonZeroUsize::new(workers).unwrap()),
                        &mut workspace,
                        context,
                    )
                })()
                .unwrap();
                assert!(output.as_slice().iter().all(|value| value.to_bits() == 1));
            }
        }
    }

    #[test]
    fn split_gemm_recovers_overflowing_partials_and_tiny_residuals() {
        for magnitude in [f64::MAX, 1.0, f64::from_bits(1)] {
            let left = Matrix::from_vec(
                33,
                3,
                (0..99)
                    .map(|index| match index % 3 {
                        0 => magnitude,
                        1 => -magnitude,
                        _ => f64::from_bits(1),
                    })
                    .collect(),
            )
            .unwrap();
            let right = Matrix::from_vec(3, 35, alloc::vec![1.0; 105]).unwrap();
            for backend in KernelBackend::available(test_processor()) {
                let mut result = Matrix::zeros(33, 35).unwrap();
                (|| -> Result<(), SciOxideError> {
                    let config = GemmConfig {
                        backend: Some(backend),
                        ..GemmConfig::ordinary()
                    };
                    crate::kernel::tests::prepared_gemm(
                        operated_view(left.view(), config.left_op).try_finite()?,
                        operated_view(right.view(), config.right_op).try_finite()?,
                        result.view_mut(),
                        config,
                        ExecutionPolicy::Sequential,
                        &mut LinalgWorkspace::new(),
                        KernelContext::new(crate::kernel::tests::test_context())
                            .with_accuracy(Accuracy::Exact),
                    )
                })()
                .unwrap();
                assert!(result.as_slice().iter().all(|value| value.to_bits() == 1));
            }
        }
    }

    #[test]
    fn split_scaled_expression_rounds_only_after_alpha_and_beta() {
        let left = Matrix::from_vec(33, 2, alloc::vec![f64::MAX; 66]).unwrap();
        let right = Matrix::from_vec(2, 35, alloc::vec![1.0; 70]).unwrap();
        for backend in KernelBackend::available(test_processor()) {
            let mut output = Matrix::from_vec(33, 35, alloc::vec![f64::MAX; 33 * 35]).unwrap();
            (|| -> Result<(), SciOxideError> {
                let config = GemmConfig {
                    alpha: 0.25,
                    beta: -0.25,
                    backend: Some(backend),
                    ..GemmConfig::ordinary()
                };
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    output.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut LinalgWorkspace::new(),
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert!(
                output
                    .as_slice()
                    .iter()
                    .all(|value| value.to_bits() == (f64::MAX / 4.0).to_bits())
            );
        }
    }
}
