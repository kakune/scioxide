//! Infallible arithmetic over finite contiguous rows, checked at block boundaries.

use super::{DISTANCE_BLOCK, DistanceMetric, HausdorffResult, PointSet, Row, finish, named, norm2};
use scioxide_core::backend::ScalarSliceMut;
use scioxide_core::{RealScalar, SciOxideError};

// The metric is a literal selected outside the batch. Preserve left-to-right
// binary64 accumulation, including sqrt-then-square for squared Euclidean.
#[inline(always)]
pub(super) fn raw<S: RealScalar>(metric: DistanceMetric, a: &[S], b: &[S]) -> f64 {
    let accumulate = |sum: f64, d: f64| match metric {
        DistanceMetric::CityBlock => sum + d.abs(),
        DistanceMetric::Chebyshev => sum.max(d.abs()),
        _ => sum + d * d,
    };
    if let ([a, b, c], [x, y, z]) = (a, b) {
        [
            a.into_f64() - x.into_f64(),
            b.into_f64() - y.into_f64(),
            c.into_f64() - z.into_f64(),
        ]
        .into_iter()
        .fold(0.0, accumulate)
    } else {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| a.into_f64() - b.into_f64())
            .fold(0.0, accumulate)
    }
}

#[inline(always)]
fn value(metric: DistanceMetric, raw: f64) -> f64 {
    match metric {
        DistanceMetric::Euclidean => RealScalar::sqrt(raw),
        DistanceMetric::SquaredEuclidean => {
            let norm = RealScalar::sqrt(raw);
            norm * norm
        }
        _ => raw,
    }
}

// Primitive projections accept every IEEE value without fallible per-pair
// conversions. Only private staging is written before the block check. Check
// the actual rounded f32 value: a value just above f32::MAX can round to MAX.
// Callers certify that zero square sums come from equal coordinates.
pub(super) struct BlockStatus {
    pub(super) ordinary: bool,
    pub(super) zero: bool,
}

#[inline]
pub(super) fn store<S: RealScalar>(
    metric: DistanceMetric,
    raw: &[f64],
    output: &mut [S],
) -> BlockStatus {
    let norm = matches!(
        metric,
        DistanceMetric::Euclidean | DistanceMetric::SquaredEuclidean
    );
    macro_rules! convert {
        ($output:expr, $cast:expr) => {
            $output.iter_mut().zip(raw).fold(
                BlockStatus {
                    ordinary: true,
                    zero: false,
                },
                |status, (out, &x)| {
                    *out = $cast(value(metric, x));
                    BlockStatus {
                        ordinary: status.ordinary
                            & x.is_finite()
                            & (!norm | (x == 0.0) | (x >= f64::MIN_POSITIVE))
                            & out.is_finite(),
                        zero: status.zero | (norm & (x == 0.0)),
                    }
                },
            )
        };
    }
    match S::slice_mut(output) {
        ScalarSliceMut::F32(output) => convert!(output, |x| x as f32),
        ScalarSliceMut::F64(output) => convert!(output, |x| x),
        // RealScalar is sealed to the two primitive real representations.
        _ => BlockStatus {
            ordinary: false,
            zero: false,
        },
    }
}

#[inline]
pub(super) fn fill<S: RealScalar>(
    metric: DistanceMetric,
    left: Row<'_, S>,
    right: &PointSet<'_, S>,
    first: usize,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let mut sums = [0.0; DISTANCE_BLOCK];
    let sums = &mut sums[..output.len()];
    if left.dimension == 3 && right.strides[0] == 3 {
        // Fixed-size row chunks expose the common 3D loop to the vectorizer;
        // both coordinate and row bounds are proved by these slice lengths.
        let data = &right.data[3 * first..3 * (first + output.len())];
        let a = &left.data[..3];
        for (sum, b) in sums.iter_mut().zip(data.chunks_exact(3)) {
            *sum = raw(metric, a, b);
        }
    } else {
        for (sum, column) in sums.iter_mut().zip(first..) {
            *sum = raw(metric, left.data, right.row::<true>(column).data);
        }
    }
    let status = store(metric, sums, output);
    if !status.ordinary || (status.zero && !exact_zeros(left, right, first, sums)) {
        replay(metric, left, right, first, output)?;
    }
    Ok(())
}

#[inline]
pub(super) fn exact_zeros<S: RealScalar>(
    left: Row<'_, S>,
    right: &PointSet<'_, S>,
    first: usize,
    sums: &[f64],
) -> bool {
    if left.dimension == 3 && right.strides[0] == 3 {
        let data = &right.data[3 * first..3 * (first + sums.len())];
        let a = &left.data[..3];
        // Non-short-circuit comparisons keep this certificate vectorizable on
        // duplicate-heavy grids. Zero sums alone cannot certify equal points.
        sums.iter()
            .zip(data.chunks_exact(3))
            .fold(true, |exact, (&sum, b)| {
                exact & ((sum != 0.0) | ((a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2])))
            })
    } else {
        sums.iter()
            .zip(first..)
            .all(|(&sum, column)| sum != 0.0 || left.data == right.row::<true>(column).data)
    }
}

#[cold]
#[inline(never)]
fn replay<S: RealScalar>(
    metric: DistanceMetric,
    left: Row<'_, S>,
    right: &PointSet<'_, S>,
    first: usize,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    for (out, column) in output.iter_mut().zip(first..) {
        let b = right.row::<true>(column);
        *out = finish(named(metric, left.dimension, |axis| left.pair(b, axis))?)?;
    }
    Ok(())
}

pub(super) fn hausdorff<S: RealScalar, const CONTIGUOUS: bool>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
    sources: &[usize],
    targets: &[usize],
) -> Result<HausdorffResult<S>, SciOxideError> {
    if CONTIGUOUS && source.dimension() == 3 {
        hausdorff_rows::<S, CONTIGUOUS, true>(source, target, sources, targets)
    } else {
        hausdorff_rows::<S, CONTIGUOUS, false>(source, target, sources, targets)
    }
}

fn hausdorff_rows<S: RealScalar, const CONTIGUOUS: bool, const THREE: bool>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
    sources: &[usize],
    targets: &[usize],
) -> Result<HausdorffResult<S>, SciOxideError> {
    // Short speculative blocks retain early-break efficiency; output kernels
    // amortize staging and conversion over longer contiguous segments.
    const HAUSDORFF_BLOCK: usize = 32;
    let mut result = HausdorffResult {
        distance: S::ZERO,
        source: usize::MAX,
        target: 0,
    };
    let mut sums = [0.0; HAUSDORFF_BLOCK];
    let mut distances = [S::ZERO; HAUSDORFF_BLOCK];
    for &i in sources {
        let a = source.row::<CONTIGUOUS>(i);
        let mut nearest = S::MAX;
        let mut witness = usize::MAX;
        let mut pruned = false;
        for indices in targets.chunks(HAUSDORFF_BLOCK) {
            let sums = &mut sums[..indices.len()];
            let distances = &mut distances[..indices.len()];
            for (sum, &j) in sums.iter_mut().zip(indices) {
                let b = target.row::<CONTIGUOUS>(j);
                *sum = if THREE {
                    raw(DistanceMetric::Euclidean, &a.data[..3], &b.data[..3])
                } else if CONTIGUOUS {
                    raw(DistanceMetric::Euclidean, a.data, b.data)
                } else {
                    (0..a.dimension).fold(0.0, |sum, axis| {
                        let (a, b) = a.pair(b, axis);
                        let d = a - b;
                        sum + d * d
                    })
                };
            }
            // Square root and destination rounding are monotone. Checking the
            // maximum certifies destination range, and the minimum gives its
            // rounded minimum without a square root for every pair. Each tiny
            // sum needs its own certificate: a zero minimum can hide a positive
            // subnormal sum elsewhere in the block.
            // Finite input differences and sums of squares cannot produce NaN.
            let (low, high) = sums
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(low, high), &x| {
                    (low.min(x), high.max(x))
                });
            let mut bounds = [S::ZERO; 2];
            let ordinary = store(DistanceMetric::Euclidean, &[low, high], &mut bounds).ordinary
                && (low != 0.0
                    || sums.iter().zip(indices).all(|(&sum, &j)| {
                        sum >= f64::MIN_POSITIVE
                            || (sum == 0.0 && {
                                let b = target.row::<CONTIGUOUS>(j);
                                (0..a.dimension).all(|axis| a.value(axis) == b.value(axis))
                            })
                    }));
            if ordinary {
                let minimum = bounds[0];
                if minimum < result.distance {
                    pruned = true;
                    break;
                }
                if minimum <= nearest {
                    if minimum < nearest {
                        nearest = minimum;
                        witness = usize::MAX;
                    }
                    // Resolve ties in destination precision only when this
                    // block can improve the witness. Bounds already certify
                    // every conversion; the repeated validity reduction is dead.
                    let _ = store(DistanceMetric::Euclidean, sums, distances);
                    for (&j, &d) in indices.iter().zip(distances.iter()) {
                        if d == nearest {
                            witness = witness.min(j);
                        }
                    }
                }
                continue;
            }
            for &j in indices {
                // Replay only the visited prefix. A speculative overflow after
                // an early break must not become an observable error.
                let b = target.row::<CONTIGUOUS>(j);
                let d = finish(norm2(a.dimension, |axis| {
                    let (a, b) = a.pair(b, axis);
                    a - b
                })?)?;
                if d < result.distance {
                    pruned = true;
                    break;
                }
                if d < nearest || (d == nearest && j < witness) {
                    nearest = d;
                    witness = j;
                }
            }
            if pruned {
                break;
            }
        }
        if !pruned
            && (nearest > result.distance || (nearest == result.distance && i < result.source))
        {
            result = HausdorffResult {
                distance: nearest,
                source: i,
                target: witness,
            };
        }
    }
    Ok(result)
}
