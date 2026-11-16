//! Smooth transforms executed across lanes.
//!
//! Lane passes hold [`WIDTH`] lanes as structure-of-arrays [`Block`] rows, so
//! every codelet operation becomes element-wise operations across the lanes
//! and twiddles stay shared scalars. Rows are gathered in the plan's
//! digit-reversed order, every stage runs with the shared generic codelets,
//! and the result is in natural order. Portable rows use scalar arithmetic.
//! Native rows mirror the selected one-lane leaves: AVX2 products fuse only
//! the indices covered by a native register; scalar tails, the leading fused
//! radix-16 gather and final unit-twiddle products keep their own arithmetic.
//! This includes signed zeros, so regrouping lanes cannot change their bits.
//! Partial blocks repeat their last lane.
use scioxide_core::{Complex, RealScalar};

use super::block::{Block, BlockLanes, ROWS, Run, WIDTH};
use super::scalar::{Butterfly, codelet};
use super::{Stage, StagePlan};

/// Indexed rows of codelet values: a slice, or blocks kept in scratch.
pub(crate) trait Rows<V> {
    fn get(&self, i: usize) -> V;
    fn set(&mut self, i: usize, value: V);
}

impl<V: Copy> Rows<V> for [V] {
    #[inline(always)]
    fn get(&self, i: usize) -> V {
        self[i]
    }
    #[inline(always)]
    fn set(&mut self, i: usize, value: V) {
        self[i] = value;
    }
}

/// One decimation-in-time stage over the first `len` rows, as
/// `scalar::butterfly` applies it to single values.
#[inline(always)]
fn apply<S, V, R, const RADIX: usize, const INVERSE: bool>(
    rows: &mut R,
    len: usize,
    stage: &Stage<S>,
    (fused_width, rotate_zero): (usize, bool),
) where
    S: RealScalar,
    V: Butterfly<S>,
    R: Rows<V> + ?Sized,
{
    let width = stage.width;
    for base in (0..len).step_by(RADIX * width) {
        for index in base..base + width {
            let mut values = [V::zero(); 7];
            let first = rows.get(index);
            values[0] = if width == 1 { first.bounded() } else { first };
            for (branch, slot) in values.iter_mut().enumerate().take(RADIX).skip(1) {
                let value = rows.get(index + branch * width);
                let value = if width == 1 { value.bounded() } else { value };
                *slot = if index == base && !rotate_zero {
                    value
                } else if index - base < fused_width {
                    value.rotate_fused::<INVERSE>(
                        stage.twiddles[(branch - 1) * width + index - base],
                    )
                } else {
                    value.rotate::<INVERSE>(stage.twiddles[(branch - 1) * width + index - base])
                };
            }
            codelet::<S, _, RADIX, INVERSE>(&mut values, &stage.roots);
            for (branch, &value) in values.iter().enumerate().take(RADIX) {
                rows.set(index + branch * width, value);
            }
        }
    }
}

/// Every stage of `plan` over the first `plan.n` rows.
#[inline(always)]
fn stages<S, V, R, const INVERSE: bool>(plan: &StagePlan<S>, rows: &mut R, native: bool)
where
    S: RealScalar,
    V: Butterfly<S>,
    R: Rows<V> + ?Sized,
{
    for (i, stage) in plan.stages.iter().enumerate() {
        let last = i + 1 == plan.stages.len();
        let lanes = 32 / size_of::<Complex<S>>();
        let fused = native
            && plan.dispatch.backend() == scioxide_core::backend::KernelBackend::Avx2Fma
            && stage.width >= lanes
            && (!last || stage.radix == 4 && stage.width.is_multiple_of(lanes));
        // The fused leading radix-16 gather leaves the k=0 column unrotated.
        // Other vector stages multiply their unit twiddles too; final scalar
        // leaves also retain those products, including their signed zeros.
        let leading_pair = i == 1 && !last && stage.radix == 4 && plan.stages[0].radix == 4;
        let arithmetic = (
            if fused {
                stage.width / lanes * lanes
            } else {
                0
            },
            native && last || fused && !leading_pair,
        );
        match stage.radix {
            2 => apply::<S, V, R, 2, INVERSE>(rows, plan.n, stage, arithmetic),
            3 => apply::<S, V, R, 3, INVERSE>(rows, plan.n, stage, arithmetic),
            4 => apply::<S, V, R, 4, INVERSE>(rows, plan.n, stage, arithmetic),
            5 => apply::<S, V, R, 5, INVERSE>(rows, plan.n, stage, arithmetic),
            7 => apply::<S, V, R, 7, INVERSE>(rows, plan.n, stage, arithmetic),
            _ => unreachable!("stage plans prove supported radices"),
        }
    }
}

/// Transform the [`super::block::WIDTH`] lanes of `rows`, already in the
/// plan's digit order, in place and unscaled: every stage runs on whole
/// structure-of-arrays blocks, and each lane follows the selected arithmetic
/// exactly, whatever its position in the block.
#[inline(always)]
pub(crate) fn transform_rows<S: RealScalar>(
    plan: &StagePlan<S>,
    rows: &mut [Block<S, { super::block::WIDTH }>],
    inverse: bool,
    native: bool,
) {
    let rows = &mut rows[..plan.n];
    if plan.n == 1 {
        rows[0] = rows[0].bounded();
    }
    if inverse {
        stages::<S, _, _, true>(plan, rows, native);
    } else {
        stages::<S, _, _, false>(plan, rows, native);
    }
}

/// Transform every lane of `run` in place, a block of [`WIDTH`] lanes at a
/// time through `rows`, and multiply it by `scale`.
///
/// The caller proves `plan.n == run.len <= ROWS`; indexing is checked.
#[inline(always)]
pub(crate) fn transform_run<S: RealScalar>(
    plan: &StagePlan<S>,
    values: &mut [Complex<S>],
    run: Run,
    (inverse, scale): (bool, S),
    rows: &mut [Block<S, WIDTH>; ROWS],
) {
    for block in (0..run.count).step_by(WIDTH) {
        let lanes = BlockLanes::new(run, block);
        for (row, &source) in rows.iter_mut().zip(plan.permutation.iter()) {
            let values = lanes.read(values, source);
            *row = Block::from_fn(|lane| values[lane]);
        }
        transform_rows(plan, rows, inverse, false);
        let rows = &mut rows[..plan.n];
        if plan.n == 1 {
            let value = rows[0];
            rows[0] =
                Block::from_fn(|lane| crate::scaling::scale_singleton(value.lane(lane), scale));
        } else if scale != S::ONE {
            rows.iter_mut().for_each(|row| *row = *row * scale);
        }
        for (i, row) in rows.iter().enumerate() {
            lanes.write(values, i, core::array::from_fn(|lane| row.lane(lane)));
        }
    }
}
