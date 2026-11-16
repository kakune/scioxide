//! Two-pass moments of complete slices in native `f64` arithmetic.
//!
//! The first pass sums differences from a shifted origin, so large raw power
//! sums are never subtracted. Weighted differences keep both TwoSum parts and
//! the exact rounding error of the leading weight product, so an origin far
//! from the center loses no accuracy. The second pass sums centered powers
//! with compensated summation. Values outside the native range fall back to
//! the extended-range online update.
use crate::arithmetic::{
    PRODUCT_CANCELLATION, exact_centered_products, native, normalized_deviation, promote, shift,
    shifted_center, weighted_shifted_center,
};
use crate::central::Central;
use crate::error_helpers::invalid;
use crate::{MomentOrder, WeightSemantics};
use scioxide_core::backend::two_product;
use scioxide_core::{
    CompensatedSum, NonFiniteOrigin, RealScalar, SciOxideError, compensated_reduce,
};
use scioxide_core::{Scaled, ScaledAccumulator};

/// Sums `count` terms with compensated summation, without buffering short slices.
fn reduce<const N: usize>(
    count: usize,
    term: impl FnMut(usize) -> [f64; N],
) -> Option<[CompensatedSum<f64>; N]> {
    if count <= 64 {
        scioxide_core::backend::compensated_indexed(count, term)
    } else {
        compensated_reduce((0..count).map(term)).ok()
    }
}

pub(crate) struct Batch<S: RealScalar> {
    pub count: usize,
    pub frequency_count: usize,
    pub central: Central<S>,
    pub scale: S,
    pub sum: Scaled<f64>,
    pub degrees_fraction: Scaled<f64>,
    pub concentration: Scaled<f64>,
}

/// Result of the native passes, before conversion of the weight state.
struct NativeBatch<S: RealScalar> {
    count: usize,
    frequency_count: usize,
    central: Central<S>,
    scale: S,
    total: f64,
    degrees_fraction: f64,
    concentration: f64,
}
impl<S: RealScalar> NativeBatch<S> {
    fn project(self) -> Result<Batch<S>, SciOxideError> {
        Ok(Batch {
            count: self.count,
            frequency_count: self.frequency_count,
            central: self.central,
            scale: self.scale,
            sum: if self.count == 0 {
                Scaled::ZERO
            } else {
                Scaled::from_value(self.total / self.scale.try_to_f64_exact()?)?
            },
            degrees_fraction: Scaled::from_value(self.degrees_fraction)?,
            concentration: Scaled::from_value(self.concentration)?,
        })
    }
}

#[inline]
pub(crate) fn moments<S: RealScalar>(
    values: &[S],
    order: MomentOrder,
) -> Result<Option<(usize, Central<S>)>, SciOxideError> {
    Ok(
        native_batch(values, None, WeightSemantics::Frequency, order)?
            .map(|batch| (batch.count, batch.central)),
    )
}

/// Marginal states and co-moment of two paired slices.
pub(crate) struct PairBatch<S: RealScalar> {
    pub x: Central<S>,
    pub y: Central<S>,
    pub covariance: Scaled<f64>,
}

pub(crate) fn paired<S: RealScalar>(
    x: &[S],
    y: &[S],
    order: MomentOrder,
) -> Result<Option<PairBatch<S>>, SciOxideError> {
    match order {
        MomentOrder::Second => paired_order::<S, 2, 3>(x, y),
        MomentOrder::Third => paired_order::<S, 3, 5>(x, y),
        MomentOrder::Fourth => paired_order::<S, 4, 7>(x, y),
    }
}

fn paired_order<S: RealScalar, const ORDER: usize, const N: usize>(
    x: &[S],
    y: &[S],
) -> Result<Option<PairBatch<S>>, SciOxideError> {
    let Some((&first_x, &first_y)) = x.first().zip(y.first()) else {
        return Ok(Some(PairBatch {
            x: Central::empty(),
            y: Central::empty(),
            covariance: Scaled::ZERO,
        }));
    };
    let origins = [first_x.into_f64(), first_y.into_f64()];
    let mut minimum = [first_x, first_y];
    let mut maximum = minimum;
    let mut admitted = true;
    let mut residuals = false;
    let shifted = compensated_reduce(x.iter().zip(y).map(|(&x, &y)| {
        for ((min, max), value) in minimum.iter_mut().zip(&mut maximum).zip([x, y]) {
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
        }
        let (x, y) = (x.into_f64(), y.into_f64());
        admitted &= native(x, 64) & native(y, 64);
        let [xh, xl] = shift(x, origins[0]);
        let [yh, yl] = shift(y, origins[1]);
        residuals |= (xl != 0.0) | (yl != 0.0);
        [xh, yh]
    }));
    let Ok([xh, yh]) = shifted else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    // Most bounded slices have exactly representable origin differences. Keep
    // their low streams implicit; any nonzero residual is reduced in full.
    let [xl, yl] = if residuals {
        compensated_reduce(x.iter().zip(y).map(|(&x, &y)| {
            [
                shift(x.into_f64(), origins[0])[1],
                shift(y.into_f64(), origins[1])[1],
            ]
        }))?
    } else {
        [CompensatedSum::ZERO; 2]
    };
    let count = x.len() as f64;
    let centers = [
        shifted_center(origins[0], [xh, xl], count)?,
        shifted_center(origins[1], [yh, yl], count)?,
    ];
    let moments = compensated_reduce(x.iter().zip(y).map(|(&x, &y)| {
        let dx = normalized_deviation(x.into_f64(), centers[0]);
        let dy = normalized_deviation(y.into_f64(), centers[1]);
        admitted &= native(dx, 128) & native(dy, 128);
        let (xx, yy) = (dx * dx, dy * dy);
        let mut terms = [0.0; N];
        terms[0] = xx;
        terms[1] = yy;
        terms[2] = dx * dy;
        if ORDER >= 3 {
            terms[3] = xx * dx;
            terms[4] = yy * dy;
        }
        if ORDER >= 4 {
            terms[5] = xx * xx;
            terms[6] = yy * yy;
        }
        terms
    }));
    let Ok(moments) = moments else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    let (xx, yy) = (moments[0].total()?, moments[1].total()?);
    let mut cross = moments[2].total()?;
    // Marginal energies bound the absolute product mass (Cauchy–Schwarz).
    // Cancellation replays exact products of both retained center parts.
    if xx > 0.0 && yy > 0.0 && cross.abs() <= PRODUCT_CANCELLATION * xx.sqrt() * yy.sqrt() {
        cross = exact_centered_products(
            x.iter().zip(y).map(|(&x, &y)| (x.into_f64(), y.into_f64())),
            [centers[0], centers[1]],
        )?;
    }
    let central = |axis: usize| -> Result<Central<S>, SciOxideError> {
        Central::from_extended(crate::central::ExtendedCentral {
            mean: if x.len() == 1 {
                ScaledAccumulator::from_value(origins[axis])?
            } else {
                promote(centers[axis])?
            },
            m2: Scaled::from_value(if axis == 0 { xx / count } else { yy / count })?,
            m3: Scaled::from_value(if ORDER >= 3 {
                moments[3 + axis].total()? / count
            } else {
                0.0
            })?,
            m4: Scaled::from_value(if ORDER >= 4 {
                moments[5 + axis].total()? / count
            } else {
                0.0
            })?,
            minimum: minimum[axis],
            maximum: maximum[axis],
        })
    };
    Ok(Some(PairBatch {
        x: central(0)?,
        y: central(1)?,
        covariance: Scaled::from_value(cross / count)?,
    }))
}

#[inline]
pub(crate) fn from_slices<S: RealScalar>(
    values: &[S],
    weights: Option<&[S]>,
    semantics: WeightSemantics,
    order: MomentOrder,
) -> Result<Option<Batch<S>>, SciOxideError> {
    native_batch(values, weights, semantics, order)?
        .map(NativeBatch::project)
        .transpose()
}

#[inline]
fn native_batch<S: RealScalar>(
    values: &[S],
    weights: Option<&[S]>,
    semantics: WeightSemantics,
    order: MomentOrder,
) -> Result<Option<NativeBatch<S>>, SciOxideError> {
    let weighted = weights.is_some();
    let weights = weights.unwrap_or(&[]);
    match (order, weighted) {
        (MomentOrder::Second, false) => summarize::<S, 2, false>(values, weights, semantics),
        (MomentOrder::Third, false) => summarize::<S, 3, false>(values, weights, semantics),
        (MomentOrder::Fourth, false) => summarize::<S, 4, false>(values, weights, semantics),
        (MomentOrder::Second, true) => summarize::<S, 2, true>(values, weights, semantics),
        (MomentOrder::Third, true) => summarize::<S, 3, true>(values, weights, semantics),
        (MomentOrder::Fourth, true) => summarize::<S, 4, true>(values, weights, semantics),
    }
}

fn summarize<S: RealScalar, const ORDER: usize, const WEIGHTED: bool>(
    values: &[S],
    weights: &[S],
    semantics: WeightSemantics,
) -> Result<Option<NativeBatch<S>>, SciOxideError> {
    // With source exponents bounded by 64 and slice length bounded by usize,
    // weighted sums through order four fit normal f64, including differences
    // between adjacent source values. Larger dynamic ranges use Scaled kernels.
    let first = if WEIGHTED {
        None
    } else {
        values.first().copied()
    };
    let mut origin = first.map(|x| x.into_f64());
    let (mut count, mut frequency_count) = (if WEIGHTED { 0 } else { values.len() }, 0_usize);
    let mut minimum = first.unwrap_or(S::ZERO);
    let mut maximum = minimum;
    let mut scale = if first.is_some() { S::ONE } else { S::ZERO };
    let mut unweighted_shifted = [CompensatedSum::ZERO; 2];
    if !WEIGHTED {
        let origin = origin.unwrap_or(0.0);
        let mut finite = true;
        let mut bounded = true;
        let shifted = reduce(values.len(), |i| {
            let value = values[i];
            let x = value.into_f64();
            finite &= x.is_finite();
            bounded &= native(x, 64);
            if value < minimum {
                minimum = value;
            }
            if value > maximum {
                maximum = value;
            }
            shift(x, origin)
        });
        if !finite {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "observations and weights must be finite",
            ));
        }
        if !bounded {
            return Ok(None);
        }
        let Some(shifted) = shifted else {
            return Ok(None);
        };
        // Admission and extrema share the unchanged shifted compensated pass.
        // Wider sources still select the extended streaming fallback.
        unweighted_shifted = shifted;
    } else {
        for (block_index, block) in values.chunks(256).enumerate() {
            let start = block_index * 256;
            let [finite, nonnegative, bounded] = block.iter().enumerate().fold(
                [true; 3],
                |[finite, nonnegative, bounded], (i, &value)| {
                    let w = weights[start + i].into_f64();
                    let x = value.into_f64();
                    [
                        finite & x.is_finite() & w.is_finite(),
                        nonnegative & (w >= 0.0),
                        bounded & ((w == 0.0) | (native(x, 64) & native(w, 64))),
                    ]
                },
            );
            if !finite {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "observations and weights must be finite",
                ));
            }
            if !nonnegative {
                return Err(invalid(
                    scioxide_core::argument!("weights"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "weight must be nonnegative",
                ));
            }
            if !bounded {
                return Ok(None);
            }
            for (offset, &value) in block.iter().enumerate() {
                let i = start + offset;
                let weight = weights[i];
                if weight == S::ZERO {
                    continue;
                }
                if semantics == WeightSemantics::Frequency {
                    frequency_count = frequency_count
                        .checked_add(weight.try_to_usize_exact()?)
                        .ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "replicated_observation_count"
                            ))
                        })?;
                }
                origin.get_or_insert(value.into_f64());
                if count == 0 {
                    minimum = value;
                    maximum = value;
                }
                count += 1;
                if weight > scale {
                    scale = weight;
                }
                // Inputs are already finite; ordered comparisons also retain the first
                // observation's sign for tied zero extrema.
                if value < minimum {
                    minimum = value;
                }
                if value > maximum {
                    maximum = value;
                }
            }
        }
    }
    let Some(origin) = origin else {
        return Ok(Some(NativeBatch {
            count,
            frequency_count,
            central: Central::empty(),
            scale: S::ZERO,
            total: 0.0,
            degrees_fraction: 0.0,
            concentration: 1.0,
        }));
    };
    let (center, total, square_mass) = if WEIGHTED {
        let Ok([high, low, mass, squares]) =
            compensated_reduce(values.iter().zip(weights).map(|(&x, &w)| {
                let w = w.into_f64();
                // Zero masses may carry unadmitted sources; they contribute
                // the origin, without a branch. Admitted sources and weights
                // keep the product and its error-free residual normal; the
                // low TwoSum part is already below the leading rounding.
                let x = if w == 0.0 { origin } else { x.into_f64() };
                let [high, low] = shift(x, origin);
                let (product, residual) = two_product(w, high);
                [product, w * low + residual, w, w * w]
            }))
        else {
            return Ok(None);
        };
        (
            weighted_shifted_center(origin, [high, low], mass)?,
            mass.total()?,
            squares.total()?,
        )
    } else {
        let count = count as f64;
        (
            shifted_center(origin, unweighted_shifted, count)?,
            count,
            count,
        )
    };
    let total_squared = total * total;
    let pair_mass = total_squared - square_mass;
    // Subtract only when the difference has condition number at most seven.
    // Dominant weights select the shared pair-product merger, which preserves
    // reliability degrees even when W² and Q round to the same f64 value.
    if WEIGHTED && count > 1 && pair_mass < total_squared * 0.25 {
        return Ok(None);
    }
    if !WEIGHTED {
        frequency_count = count;
    }
    let mut admitted = true;
    let mut terms = |i: usize| {
        let value: S = values[i];
        let w = if WEIGHTED { weights[i].into_f64() } else { 1.0 };
        if w == 0.0 {
            return [0.0; 3];
        }
        let delta = normalized_deviation(value.into_f64(), center);
        admitted &= native(delta, 128);
        let square = delta * delta;
        [
            w * square,
            if ORDER >= 3 {
                w * (square * delta)
            } else {
                0.0
            },
            if ORDER >= 4 {
                w * (square * square)
            } else {
                0.0
            },
        ]
    };
    let moments = if ORDER == 2 {
        reduce(values.len(), |i| {
            let [second, ..] = terms(i);
            [second]
        })
        .map(|[second]| [second, CompensatedSum::ZERO, CompensatedSum::ZERO])
    } else if ORDER == 3 {
        reduce(values.len(), |i| {
            let [second, third, _] = terms(i);
            [second, third]
        })
        .map(|[second, third]| [second, third, CompensatedSum::ZERO])
    } else {
        reduce(values.len(), terms)
    };
    let Some(moments) = moments else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    let mean = if count == 1 {
        [minimum.into_f64(), 0.0]
    } else {
        center
    };
    Ok(Some(NativeBatch {
        count,
        frequency_count,
        central: Central::from_native_parts(
            mean,
            [
                moments[0].total()? / total,
                moments[1].total()? / total,
                moments[2].total()? / total,
            ],
            minimum,
            maximum,
        )?,
        scale,
        total,
        degrees_fraction: if WEIGHTED {
            pair_mass / total_squared
        } else {
            (count - 1) as f64 / total
        },
        concentration: if WEIGHTED {
            square_mass / total_squared
        } else {
            1.0 / total
        },
    }))
}
