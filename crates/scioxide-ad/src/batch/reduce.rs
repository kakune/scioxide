//! Pairwise reduction of block results over the block index.
//!
//! The reduction tree is fixed by the number of blocks (stride doubling), not by
//! the partition of blocks over workers, so results are identical for every
//! execution policy.

use scioxide_core::RealScalar;

use super::workspace::BlockStats;

fn count<S: RealScalar>(n: usize) -> S {
    S::from_f64_ieee(n as f64)
}

/// Combines two blocks: sums, and means with second central moments (Chan, Golub
/// and LeVeque 1983).
fn combine<S: RealScalar>(a: BlockStats<S>, b: BlockStats<S>) -> BlockStats<S> {
    if a.count == 0 {
        return BlockStats {
            sum: a.sum + b.sum,
            derivative: a.derivative + b.derivative,
            kinks: a.kinks + b.kinks,
            ..b
        };
    }
    if b.count == 0 {
        return BlockStats {
            sum: a.sum + b.sum,
            derivative: a.derivative + b.derivative,
            kinks: a.kinks + b.kinks,
            ..a
        };
    }
    let n = a.count + b.count;
    let (na, nb, total) = (count::<S>(a.count), count::<S>(b.count), count::<S>(n));
    let delta = b.mean - a.mean;
    BlockStats {
        count: n,
        sum: a.sum + b.sum,
        mean: a.mean + delta * (nb / total),
        m2: a.m2 + b.m2 + delta * delta * (na * nb / total),
        derivative: a.derivative + b.derivative,
        kinks: a.kinks + b.kinks,
    }
}

/// Reduces `blocks` in place and returns the total.
pub(crate) fn stats<S: RealScalar>(blocks: &mut [BlockStats<S>]) -> BlockStats<S> {
    let n = blocks.len();
    let mut stride = 1;
    while stride < n {
        let mut first = 0;
        while first + stride < n {
            blocks[first] = combine(blocks[first], blocks[first + stride]);
            first += 2 * stride;
        }
        stride *= 2;
    }
    blocks.first().copied().unwrap_or_else(BlockStats::empty)
}

/// Reduces `blocks` vectors of `width` values (block-major) in place; the total is
/// left in the first `width` values.
pub(crate) fn vectors<S: RealScalar>(values: &mut [S], blocks: usize, width: usize) {
    let mut stride = 1;
    while stride < blocks {
        let mut first = 0;
        while first + stride < blocks {
            let (head, tail) = values.split_at_mut((first + stride) * width);
            let target = &mut head[first * width..(first + 1) * width];
            for (t, &v) in target.iter_mut().zip(&tail[..width]) {
                *t += v;
            }
            first += 2 * stride;
        }
        stride *= 2;
    }
}
