//! Richardson extrapolation, per component, on one nested trapezoidal grid.

use crate::LevelOptions;

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValue, AffineValueMut, ErrorTolerance, Interval, PartialResultTrust, RealScalar, Scaled,
    SciOxideError, SolveContext, write_value,
};

use crate::evaluation::sample_1d;
use crate::geometry::IntervalGeometry;
use crate::kernel::{
    WITNESS_FRACTION, WITNESS_WINDOW, WitnessInterpolant, scaled_roundoff_floor, witness_error,
};
use crate::report::{IntegrationErrorComponents, IntegrationTermination};
use crate::workspace::Outcome;
use scioxide_core::PromotingCompensatedSum;

/// Largest supported number of trapezoidal levels; level `j` has `2^j` panels.
pub(crate) const MAX_LEVELS: usize = 30;
pub(super) const MIN_CONVERGENCE_LEVEL: usize = 5;

const NATIVE_LEVELS: usize = 6;
const NATIVE_MIDPOINTS: usize = (1 << NATIVE_LEVELS) - 1;
/// New samples entering the witness window per level: its odd fine indices.
const FRESH: usize = WITNESS_WINDOW / 2;
/// The check point is sampled after this level's grid, the first whose errors count.
const WITNESS_LEVEL: usize = MIN_CONVERGENCE_LEVEL - 1;
/// Counted levels, from `WITNESS_LEVEL`.
const COUNTED: usize = MAX_LEVELS - WITNESS_LEVEL;
/// Stored window samples: at most `FRESH` per level.
const RETAINED: usize = FRESH * MAX_LEVELS;

/// Samples `lower..upper` of one level that enter its check-point window, stored
/// from slot `slot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Capture {
    lower: usize,
    upper: usize,
    slot: usize,
}

impl Capture {
    const NONE: Self = Self {
        lower: 0,
        upper: 0,
        slot: 0,
    };
}

/// The off-grid check point ("witness") and, for counted levels, its window
/// and interpolants, prepared once per plan.
///
/// The window of level `L` holds the grid samples at the eight fine indices
/// from `floor(2^L w) - 3` around the witness fraction `w`. Its odd indices are
/// midpoints of level `L` and its even indices lie in the window of level
/// `L - 1`, so every window sample is one of the at most four midpoints that
/// its own level captures. Refinement stores those captures and gathers a
/// window only where a witness error is needed.
#[derive(Debug, Clone, PartialEq)]
struct Witness<S: RealScalar> {
    fraction: S,
    /// Storage slot of every window sample.
    windows: [[u8; WITNESS_WINDOW]; COUNTED],
    interpolants: [WitnessInterpolant<S>; COUNTED],
}

impl<S: RealScalar> Witness<S> {
    fn new() -> Result<Self, SciOxideError> {
        let mut witness = Self {
            fraction: S::try_from_f64_lossy(WITNESS_FRACTION)?,
            windows: [[0; WITNESS_WINDOW]; COUNTED],
            interpolants: [WitnessInterpolant::at(S::ZERO); COUNTED],
        };
        for level in WITNESS_LEVEL..MAX_LEVELS {
            let (base, _) = witness.placement(level);
            for offset in 0..WITNESS_WINDOW {
                // An interior fine index first appears as a midpoint of the level
                // its trailing zeros reach back to.
                let index = base + offset as isize;
                let back = index.trailing_zeros() as usize;
                let position = if 0 < index && index < 1 << level {
                    ((index >> back) - 1) / 2 - witness.placement(level - back).1
                } else {
                    -1
                };
                if !(0..FRESH as isize).contains(&position) {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "witness window leaves the retained grid samples",
                    ));
                }
                witness.windows[level - WITNESS_LEVEL][offset] =
                    ((level - back) * FRESH + position as usize) as u8;
            }
            let offset = witness.fraction.into_f64() * f64::from(1_u32 << level) - base as f64;
            witness.interpolants[level - WITNESS_LEVEL] =
                WitnessInterpolant::at(S::try_from_f64_lossy(offset)?);
        }
        Ok(witness)
    }

    /// The first window fine index of `level` and the midpoint index of the
    /// window's first odd fine index.
    #[inline(always)]
    fn placement(&self, level: usize) -> (isize, isize) {
        // The fraction is a scalar value in (0, 1) and `level < MAX_LEVELS`, so
        // scaling is exact in binary64 and truncation is its floor.
        let scaled = self.fraction.into_f64() * f64::from(1_u32 << level);
        let base = scaled as isize - (WITNESS_WINDOW as isize / 2 - 1);
        (base, (base + isize::from(base % 2 == 0) - 1) / 2)
    }

    /// Midpoints of `level >= 1` that enter its window, stored at
    /// `FRESH * level` plus their position among the window's odd indices.
    #[inline(always)]
    fn capture(&self, level: usize) -> Capture {
        let (_, origin) = self.placement(level);
        let count = 1_isize << (level - 1);
        let lower = origin.clamp(0, count);
        Capture {
            lower: lower as usize,
            upper: (origin + FRESH as isize).clamp(0, count) as usize,
            slot: level * FRESH + (lower - origin) as usize,
        }
    }

    /// Integral-level witness error of a counted level from one component's
    /// stored samples and witness sample.
    #[inline(always)]
    fn error(
        &self,
        level: usize,
        retained: &[S; RETAINED],
        witness: S,
        geometry: IntervalGeometry<S>,
    ) -> Result<S, SciOxideError> {
        let window = self.windows[level - WITNESS_LEVEL].map(|slot| retained[usize::from(slot)]);
        witness_error(
            self.interpolants[level - WITNESS_LEVEL].excess(&window, witness)?,
            geometry,
        )
    }
}

/// Extends a Richardson functional, ordered from finest to coarsest trapezoid.
/// The newly appended coefficient must be zero. Extended arithmetic avoids
/// rounding intermediate coefficients before the caller prepares scalar weights.
pub(crate) fn richardson_step<S: RealScalar>(
    coefficients: &mut [Scaled<S>],
    divisor: S,
) -> Result<(), SciOxideError> {
    let inverse = Scaled::from_value(S::ONE)?.div(Scaled::from_value(divisor)?)?;
    let leading = Scaled::from_value(S::ONE)?.add(inverse)?;
    for j in (1..coefficients.len()).rev() {
        coefficients[j] = coefficients[j]
            .mul(leading)?
            .sub(coefficients[j - 1].mul(inverse)?)?;
    }
    coefficients[0] = coefficients[0].mul(leading)?;
    Ok(())
}

/// Richardson divisors and bounded paired functionals, rounded once into S,
/// and the prepared witness windows.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Factors<S: RealScalar> {
    ratios: [(S, S); MAX_LEVELS],
    steps: [S; MAX_LEVELS],
    paired: [[[S; 2]; NATIVE_LEVELS + 1]; NATIVE_LEVELS + 1],
    canonical_midpoints: [S; NATIVE_MIDPOINTS],
    witness: Witness<S>,
}
impl<S: RealScalar> Factors<S> {
    pub(super) fn new() -> Result<Self, SciOxideError> {
        let mut power = S::ONE;
        let ratios = core::array::from_fn(|_| {
            power *= S::TWO * S::TWO;
            let divisor = power - S::ONE;
            (divisor, S::ONE / divisor)
        });
        let mut step = S::ONE;
        let steps = core::array::from_fn(|_| {
            let current = step;
            step *= S::HALF;
            current
        });
        let mut paired = [[[S::ZERO; 2]; NATIVE_LEVELS + 1]; NATIVE_LEVELS + 1];
        let mut coefficients = [Scaled::ZERO; NATIVE_LEVELS + 1];
        coefficients[0] = Scaled::from_value(S::ONE)?;
        for level in 1..=NATIVE_LEVELS {
            for (j, value) in coefficients[..level].iter().enumerate() {
                paired[level][j][1] = value.round()?;
            }
            richardson_step(&mut coefficients[..=level], ratios[level - 1].0)?;
            for (j, value) in coefficients[..=level].iter().enumerate() {
                paired[level][j][0] = value.round()?;
            }
        }
        let mut canonical_midpoints = [S::ZERO; NATIVE_MIDPOINTS];
        for level in 1..=NATIVE_LEVELS {
            let count = 1 << (level - 1);
            let denominator = S::from_f64_ieee((2 * count) as f64);
            for (i, point) in canonical_midpoints[count - 1..2 * count - 1]
                .iter_mut()
                .enumerate()
            {
                *point = S::from_f64_ieee((2 * i + 1) as f64) / denominator;
            }
        }
        Ok(Self {
            ratios,
            steps,
            paired,
            canonical_midpoints,
            witness: Witness::new()?,
        })
    }
    #[inline(always)]
    fn native_diagonals(&self, level: usize, history: &[S; NATIVE_LEVELS + 1]) -> Option<[S; 2]> {
        match level {
            1 => self.native_diagonals_at::<1>(history),
            2 => self.native_diagonals_at::<2>(history),
            3 => self.native_diagonals_at::<3>(history),
            4 => self.native_diagonals_at::<4>(history),
            5 => self.native_diagonals_at::<5>(history),
            NATIVE_LEVELS => self.native_diagonals_at::<NATIVE_LEVELS>(history),
            _ => None,
        }
    }

    #[inline(always)]
    fn native_diagonals_at<const LEVEL: usize>(
        &self,
        history: &[S; NATIVE_LEVELS + 1],
    ) -> Option<[S; 2]> {
        // Earlier entries were admitted on append. The owner permanently
        // switches to the recurrence after the first failed bound.
        if !scioxide_core::backend::ShortReductionRange::new(1 << NATIVE_LEVELS)
            .admits(history[LEVEL])
        {
            return None;
        }
        // At most 32 new samples have addition depth ten. The paired
        // Richardson functionals satisfy sum |c_j|*2^j < 3; including their
        // short lane reductions stays within the existing 50-epsilon floor.
        // Anchoring enforces their unit coefficient sum exactly for constants.
        let anchor = history[LEVEL];
        let values = scioxide_core::backend::four_lane_sums(LEVEL, |i| {
            let j = i + 1;
            self.paired[LEVEL][j].map(|weight| weight * (history[LEVEL - j] - anchor))
        });
        Some(values.map(|value| anchor + value))
    }
}

/// Advances one tableau row in place and returns its diagonal extrapolant.
#[inline(always)]
fn update_tableau<S: RealScalar>(
    tableau: &mut [S; MAX_LEVELS],
    level: usize,
    trapezoidal: S,
    factors: &Factors<S>,
) -> Result<S, SciOxideError> {
    // The tableau is stored as a single anti-diagonal row: before the call `tableau[i]` holds
    // `R(level - 1, level - 1 - i)`, so `tableau[level - 1]` is the previous plain trapezoid
    // and `tableau[0]` the previous best extrapolant. Sweeping `i` downwards replaces each
    // entry with `R(level, level - i)` using
    //     R(n, k) = R(n, k-1) + (R(n, k-1) - R(n-1, k-1)) / (4^k - 1),
    // reading `tableau[i + 1]` (already the new row) and `tableau[i]` (still the old row)
    // before overwriting it. This needs O(levels) storage instead of the full triangle, and
    // `tableau[level]` keeps the raw trapezoid that the next level halves and extends.
    tableau[level] = trapezoidal;
    for (index, &(divisor, inverse)) in (0..level).rev().zip(&factors.ratios) {
        let lower_order = tableau[index + 1];
        let previous_row = tableau[index];
        let difference = lower_order - previous_row;
        let correction = difference * inverse;
        let native = lower_order + correction;
        tableau[index] = if native.is_finite()
            && (difference == S::ZERO || correction.abs() >= S::MIN_POSITIVE)
        {
            native
        } else {
            extended_correction(lower_order, previous_row, divisor)?
        };
    }
    Ok(tableau[0])
}

#[cold]
#[inline(never)]
fn extended_correction<S: RealScalar>(
    lower: S,
    previous: S,
    divisor: S,
) -> Result<S, SciOxideError> {
    let difference = Scaled::from_value(lower)?.sub(Scaled::from_value(previous)?)?;
    let correction = difference.div(Scaled::from_value(divisor)?)?;
    Scaled::from_value(lower)?.add(correction)?.round()
}

#[cold]
#[inline(never)]
fn extended_difference<S: RealScalar>(value: S, previous: S, guard: S) -> Result<S, SciOxideError> {
    Scaled::from_value(value)?
        .sub(Scaled::from_value(previous)?)?
        .abs()
        .mul(Scaled::from_value(guard)?)?
        .round()
}

/// Per-component numerical state is independent of sampling scratch.
#[derive(Debug, Clone, Copy)]
struct Component<S: RealScalar> {
    tableau: [S; MAX_LEVELS],
    history: [S; NATIVE_LEVELS + 1],
    exploratory_floors: [S; MIN_CONVERGENCE_LEVEL - 2],
    deferred: usize,
    last_trapezoid: S,
    previous_lower: S,
    recurrence: bool,
    previous: S,
    absolute: Scaled<S>,
    native_absolute: Option<S>,
    best_value: S,
    best_error: S,
    best_floor: S,
    /// Level with the smallest error, returned after an unsuccessful stop; ties
    /// keep the coarser counted level.
    best_level: usize,
    /// Value, error without the witness error, and floor of every counted level.
    counted: [[S; 3]; COUNTED],
    /// The off-grid witness sample.
    witness: S,
}
impl<S: RealScalar> Component<S> {
    fn new() -> Self {
        Self {
            best_level: 0,
            counted: [[S::ZERO; 3]; COUNTED],
            witness: S::ZERO,
            tableau: [S::ZERO; MAX_LEVELS],
            history: [S::ZERO; NATIVE_LEVELS + 1],
            exploratory_floors: [S::ZERO; MIN_CONVERGENCE_LEVEL - 2],
            deferred: 0,
            last_trapezoid: S::ZERO,
            previous_lower: S::ZERO,
            recurrence: false,
            previous: S::ZERO,
            absolute: Scaled::ZERO,
            native_absolute: None,
            best_value: S::ZERO,
            best_error: S::INFINITY,
            best_floor: S::ZERO,
        }
    }

    #[inline(always)]
    fn initialize(
        &mut self,
        geometry: IntervalGeometry<S>,
        sum: &PromotingCompensatedSum<S>,
        absolute_sum: &PromotingCompensatedSum<S>,
    ) -> Result<[S; 2], SciOxideError> {
        self.best_error = S::INFINITY;
        self.deferred = 0;
        let native = geometry.native_span().and_then(|span| {
            let half = span * S::HALF;
            Some((
                native_product(half, native_total(sum)?)?,
                native_product(half, native_total(absolute_sum)?)?,
            ))
        });
        let floor;
        if let Some((value, absolute)) = native {
            self.tableau[0] = value;
            self.native_absolute = Some(absolute);
            floor = crate::kernel::roundoff_floor(absolute)?;
        } else {
            self.tableau[0] = geometry.half_span()?.mul(sum.total()?)?.round()?;
            self.absolute = geometry.half_span()?.mul(absolute_sum.total()?)?;
            self.native_absolute = None;
            floor = scaled_roundoff_floor(self.absolute)?;
        }
        // Every later row overwrites its new column before reading it; old
        // columns beyond the active diagonal never need to be cleared.
        self.previous = self.tableau[0];
        self.last_trapezoid = self.previous;
        self.previous_lower = self.previous;
        self.history[0] = self.previous;
        self.recurrence = !scioxide_core::backend::ShortReductionRange::new(1 << NATIVE_LEVELS)
            .admits(self.previous);
        Ok([self.previous, floor])
    }

    /// Adds one level's trapezoid and diagonals. Counted levels return and
    /// record their error without the witness error, which `settle` adds.
    #[inline(always)]
    fn advance(
        &mut self,
        level: usize,
        factors: &Factors<S>,
        step: Option<S>,
        geometry: IntervalGeometry<S>,
        sum: &PromotingCompensatedSum<S>,
        absolute_sum: &PromotingCompensatedSum<S>,
    ) -> Result<[S; 4], SciOxideError> {
        let floor;

        let native = step.and_then(|h| {
            let old_abs = self.native_absolute?;
            let sum = native_total(sum)?;
            let absolute_sum = native_total(absolute_sum)?;
            // Before recurrence, the retained trapezoid passed the common
            // reduction-range bound; its half is normal or exactly zero.
            let previous_part = if self.recurrence {
                native_product(self.last_trapezoid, S::HALF)?
            } else {
                self.last_trapezoid * S::HALF
            };
            let increment = native_product(h, sum)?;
            let trap = previous_part + increment;
            let absolute = if old_abs == self.last_trapezoid.abs() && absolute_sum == sum.abs() {
                previous_part.abs() + increment.abs()
            } else {
                native_product(old_abs, S::HALF)? + native_product(h, absolute_sum)?
            };
            (trap.is_finite() && absolute.is_finite()).then_some((trap, absolute))
        });
        let trap = if let Some((trap, absolute)) = native {
            self.native_absolute = Some(absolute);
            floor = crate::kernel::roundoff_floor(absolute)?;
            trap
        } else {
            let (trap, extended_floor) =
                self.extended_trapezoid(level, geometry, sum, absolute_sum)?;
            floor = extended_floor;
            trap
        };
        if level < MIN_CONVERGENCE_LEVEL - 1
            && !self.recurrence
            && scioxide_core::backend::ShortReductionRange::new(1 << NATIVE_LEVELS).admits(trap)
        {
            self.history[level] = trap;
            self.exploratory_floors[level - 1] = floor;
            self.deferred = level;
            self.last_trapezoid = trap;
            // Coarse extrapolants cannot terminate the solve, and level four
            // discards their minima. Only the last pair enters that level's
            // change and consistency estimates. Short-range admission proves
            // these native functionals and differences cannot fail.
            if level == MIN_CONVERGENCE_LEVEL - 2
                && let Some([value, lower]) =
                    factors.native_diagonals_at::<{ MIN_CONVERGENCE_LEVEL - 2 }>(&self.history)
            {
                self.previous = value;
                self.previous_lower = lower;
            }
            return Ok([trap, S::INFINITY, floor, trap.abs()]);
        }
        if self.deferred > 0 {
            if level == MIN_CONVERGENCE_LEVEL - 1 {
                self.deferred = 0;
            } else {
                self.replay_exploration(factors)?;
            }
        }
        let native = if !self.recurrence && level <= NATIVE_LEVELS {
            self.history[level] = trap;
            factors.native_diagonals(level, &self.history)
        } else {
            None
        };
        self.last_trapezoid = trap;
        if let Some([value, lower]) = native {
            self.record_diagonals::<true>(level, value, lower, floor)
        } else {
            let [value, lower] = self.recurrence_diagonals(level, trap, factors)?;
            self.record_diagonals::<false>(level, value, lower, floor)
        }
    }

    #[inline(always)]
    fn record_diagonals<const ADMITTED: bool>(
        &mut self,
        level: usize,
        value: S,
        lower: S,
        floor: S,
    ) -> Result<[S; 4], SciOxideError> {
        let previous_lower = self.previous_lower;
        self.previous_lower = lower;
        let difference = value - self.previous;
        let guard = S::try_from_usize_exact(crate::kernel::EXTRAPOLATION_ERROR_GUARD)?;
        // With B = sqrt(MAX) / 512, each admitted functional is bounded by
        // B + 2B * sum|c_j| < 9B. Current, previous and lower diagonals therefore
        // have guarded differences below 72B, well below overflow. Scaling by
        // the exact power-of-two guard agrees with scaled arithmetic through
        // subnormal differences and signed zeros as well.
        let mut error = if ADMITTED {
            difference.abs() * guard
        } else if let Some(error) = native_product(difference.abs(), guard) {
            error
        } else {
            extended_difference(value, self.previous, guard)?
        }
        .max(floor);
        if level >= WITNESS_LEVEL {
            let difference = (lower - previous_lower).abs();
            let consistency = if ADMITTED {
                difference * guard
            } else {
                native_product(difference, guard)
                    .map(Ok)
                    .unwrap_or_else(|| extended_difference(lower, previous_lower, guard))?
            };
            error = error.max(consistency);
            // Forget potentially aliased coarse-grid minima after exploration;
            // counted levels compete once `settle` adds their witness errors.
            if level == WITNESS_LEVEL {
                self.best_error = S::INFINITY;
                self.best_level = usize::MAX;
            }
            self.counted[level - WITNESS_LEVEL] = [value, error, floor];
        } else if error < self.best_error {
            self.best_error = error;
            self.best_value = value;
            self.best_floor = floor;
        }
        let scale = value.abs().max(self.previous.abs());
        self.previous = value;

        Ok([value, error, floor, scale])
    }

    /// Lets a counted level compete for the smallest error with its final
    /// error, including the witness error; ties keep the coarser level, as a
    /// level-ordered strict minimum does.
    #[inline(always)]
    fn settle(&mut self, level: usize, error: S) {
        if error < self.best_error || (error == self.best_error && level < self.best_level) {
            let [value, _, floor] = self.counted[level - WITNESS_LEVEL];
            self.best_value = value;
            self.best_error = error;
            self.best_floor = floor;
            self.best_level = level;
        }
    }

    /// Settles the deferred counted levels in `unsettled` before an
    /// unsuccessful stop returns the smallest error.
    #[cold]
    #[inline(never)]
    fn settle_deferred(
        &mut self,
        unsettled: u32,
        witness: &Witness<S>,
        retained: &[S; RETAINED],
        geometry: IntervalGeometry<S>,
    ) -> Result<(), SciOxideError> {
        for level in (WITNESS_LEVEL..MAX_LEVELS).filter(|level| unsettled >> level & 1 == 1) {
            let raised = witness.error(level, retained, self.witness, geometry)?;
            self.settle(level, self.counted[level - WITNESS_LEVEL][1].max(raised));
        }
        Ok(())
    }
    /// Recomputes the exact early-stop diagnostics from the completed levels.
    #[cold]
    #[inline(never)]
    fn replay_exploration(&mut self, factors: &Factors<S>) -> Result<(), SciOxideError> {
        let levels = core::mem::take(&mut self.deferred);
        self.previous = self.history[0];
        self.previous_lower = self.previous;
        self.best_error = S::INFINITY;
        for level in 1..=levels {
            let floor = self.exploratory_floors[level - 1];
            if let Some([value, lower]) = factors.native_diagonals(level, &self.history) {
                self.record_diagonals::<true>(level, value, lower, floor)?
            } else {
                let [value, lower] =
                    self.recurrence_diagonals(level, self.history[level], factors)?;
                self.record_diagonals::<false>(level, value, lower, floor)?
            };
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn recurrence_diagonals(
        &mut self,
        level: usize,
        trap: S,
        factors: &Factors<S>,
    ) -> Result<[S; 2], SciOxideError> {
        if !self.recurrence {
            for row in 0..level {
                update_tableau(&mut self.tableau, row, self.history[row], factors)?;
            }
            self.recurrence = true;
        }
        let value = update_tableau(&mut self.tableau, level, trap, factors)?;
        Ok([value, self.tableau[1]])
    }

    #[cold]
    #[inline(never)]
    fn extended_trapezoid(
        &mut self,
        level: usize,
        geometry: IntervalGeometry<S>,
        sum: &PromotingCompensatedSum<S>,
        absolute_sum: &PromotingCompensatedSum<S>,
    ) -> Result<(S, S), SciOxideError> {
        let h = geometry
            .span()?
            .div(Scaled::from_value(S::try_from_usize_exact(
                1usize << level,
            )?)?)?;
        let old_abs = match self.native_absolute.take() {
            Some(v) => Scaled::from_value(v)?,
            None => self.absolute,
        };
        self.absolute = old_abs.shifted(-1)?.add(h.mul(absolute_sum.total()?)?)?;
        let floor = scaled_roundoff_floor(self.absolute)?;
        let trap = Scaled::from_value(self.last_trapezoid)?
            .shifted(-1)?
            .add(h.mul(sum.total()?)?)?
            .round()?;
        Ok((trap, floor))
    }
}

#[derive(Debug)]
struct Sampling<S: RealScalar> {
    points: Vec<S>,
    rows: Vec<S>,
    sums: Vec<[PromotingCompensatedSum<S>; 2]>,
    /// Per-component witness window samples that every level captures.
    retained: Vec<[S; RETAINED]>,
    /// The current level's captures.
    capture: Capture,
}
impl<S: RealScalar> Sampling<S> {
    fn allocate(components: usize, block: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            retained: try_filled(
                components,
                [S::ZERO; RETAINED],
                scioxide_core::quantity!("witness_samples"),
            )?,
            capture: Capture::NONE,
            points: try_filled(block, S::ZERO, scioxide_core::quantity!("romberg_points"))?,
            rows: try_filled(
                scioxide_core::backend::checked_mul(
                    block,
                    components,
                    scioxide_core::quantity!("romberg_rows"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("romberg_rows"),
            )?,
            sums: try_filled(
                components,
                [PromotingCompensatedSum::ZERO; 2],
                scioxide_core::quantity!("component_sums"),
            )?,
        })
    }
    /// Stores the captured samples among the `len` rows of `d` components
    /// that the current block holds, starting at level sample `start`.
    #[inline(always)]
    fn capture(&mut self, start: usize, len: usize, d: usize) {
        let Capture { lower, upper, slot } = self.capture;
        let (first, last) = (lower.max(start), upper.min(start + len));
        if first >= last {
            return;
        }
        let rows = &self.rows[(first - start) * d..(last - start) * d];
        let slots = slot + first - lower..slot + last - lower;
        for (component, retained) in self.retained[..d].iter_mut().enumerate() {
            for (target, row) in retained[slots.clone()].iter_mut().zip(rows.chunks_exact(d)) {
                *target = row[component];
            }
        }
    }

    #[inline(always)]
    fn append(
        totals: &mut [PromotingCompensatedSum<S>; 2],
        first: bool,
        sums: [PromotingCompensatedSum<S>; 2],
    ) -> Result<(), SciOxideError> {
        if first {
            *totals = sums;
        } else {
            for (total, sum) in totals.iter_mut().zip(sums) {
                total.merge(sum)?;
            }
        }
        Ok(())
    }
    /// Evaluates `count` grid points in blocks and add every row, in point order,
    /// to the component and absolute sums.
    fn sample<const WIDTH: usize, F, V>(
        &mut self,
        function: &F,
        count: usize,
        point: impl Fn(usize) -> Result<S, SciOxideError>,
        options: &LevelOptions<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let d = if WIDTH == 0 { self.sums.len() } else { WIDTH };
        self.sums.fill([PromotingCompensatedSum::ZERO; 2]);
        let mut start = 0;
        while start < count {
            let len = (count - start).min(self.points.len());
            for (offset, slot) in self.points[..len].iter_mut().enumerate() {
                *slot = point(start + offset)?;
            }
            let rows = &mut self.rows[..len * d];
            if d == 1
                && len <= 64
                && options.execution_policy() == scioxide_core::ExecutionPolicy::Sequential
            {
                let [sum, absolute] = crate::evaluation::sample_scalar_1d(
                    function,
                    &self.points[..len],
                    rows,
                    context,
                )?;
                Self::append(&mut self.sums[0], start == 0, [sum, absolute])?;
                self.capture(start, len, 1);
                start += len;
                continue;
            }
            sample_1d(
                function,
                &self.points[..len],
                d,
                rows,
                options.execution_policy(),
                context,
            )?;
            for component in 0..d {
                let values = rows.chunks_exact(d).map(|row| row[component]);
                let sum = PromotingCompensatedSum::from_values(values.clone())?;
                let absolute = PromotingCompensatedSum::from_values(values.map(|v| v.abs()))?;
                Self::append(&mut self.sums[component], start == 0, [sum, absolute])?;
            }
            self.capture(start, len, d);
            start += len;
        }
        Ok(())
    }

    /// Samples a level prepared on `[0, 1]`. A unit span needs only a
    /// translation; other spans round as a power-of-two multiplication of the
    /// dyadic grid does.
    #[inline(always)]
    fn sample_canonical_grid<F, V>(
        &mut self,
        function: &F,
        canonical: &[S],
        span: S,
        lower: S,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        if span == S::ONE && lower == S::ZERO {
            return self.sample_scalar_points(function, canonical, context);
        }
        let sums = if span == S::ONE {
            crate::evaluation::sample_scalar::<S, S, 2, false>(
                canonical,
                &mut self.rows[..canonical.len()],
                context,
                {
                    #[inline(always)]
                    |point: &S| crate::evaluation::scalar_value(&function(lower + *point)?)
                },
                |_| S::ONE,
            )?
        } else {
            crate::evaluation::sample_scalar::<S, S, 2, false>(
                canonical,
                &mut self.rows[..canonical.len()],
                context,
                {
                    #[inline(always)]
                    |point: &S| crate::evaluation::scalar_value(&function(lower + span * *point)?)
                },
                |_| S::ONE,
            )?
        };
        self.sums[0] = sums;
        self.capture(0, canonical.len(), 1);
        Ok(())
    }

    /// Maps one scalar level into scratch, with the same 64-row reduction
    /// boundaries as the general sampler.
    #[inline(always)]
    fn sample_scalar_grid<F, V>(
        &mut self,
        function: &F,
        count: usize,
        point: impl Fn(usize) -> Result<S, SciOxideError>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        for start in (0..count).step_by(64) {
            let len = (count - start).min(64);
            for (offset, slot) in self.points[..len].iter_mut().enumerate() {
                *slot = point(start + offset)?;
            }
            let sums = crate::evaluation::sample_scalar_1d(
                function,
                &self.points[..len],
                &mut self.rows[..len],
                context,
            )?;
            Self::append(&mut self.sums[0], start == 0, sums)?;
            self.capture(start, len, 1);
        }
        Ok(())
    }

    /// Evaluates scalar points in one four-row pass over their value rows;
    /// validation and evaluation counting stay in point order.
    #[inline(always)]
    fn sample_scalar_points<F, V>(
        &mut self,
        function: &F,
        points: &[S],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        if points.len() <= 64 {
            self.sums[0] = crate::evaluation::sample_scalar_1d(
                function,
                points,
                &mut self.rows[..points.len()],
                context,
            )?;
            self.capture(0, points.len(), 1);
            return Ok(());
        }
        for (block, points) in points.chunks(64).enumerate() {
            let sums = crate::evaluation::sample_scalar_1d(
                function,
                points,
                &mut self.rows[..points.len()],
                context,
            )?;
            Self::append(&mut self.sums[0], block == 0, sums)?;
            self.capture(block * 64, points.len(), 1);
        }
        Ok(())
    }
}

/// Numerical state, sampling scratch and output buffers.
#[derive(Debug)]
pub(super) struct State<S: RealScalar> {
    components: usize,
    data: Vec<Component<S>>,
    sampling: Sampling<S>,
    values: Vec<S>,
    errors: Vec<S>,
    scales: Vec<S>,
    floors: Vec<S>,
    pub(super) diagnostics: Vec<IntegrationErrorComponents<S>>,
}
impl<S: RealScalar> State<S> {
    pub(super) fn allocate(components: usize, block: usize) -> Result<Self, SciOxideError> {
        let filled = |quantity| try_filled(components, S::ZERO, quantity);
        Ok(Self {
            components,
            data: try_filled(
                components,
                Component::new(),
                scioxide_core::quantity!("romberg_components"),
            )?,
            sampling: Sampling::allocate(components, block)?,
            values: filled(scioxide_core::quantity!("component_values"))?,
            errors: filled(scioxide_core::quantity!("component_errors"))?,
            scales: filled(scioxide_core::quantity!("component_scales"))?,
            floors: filled(scioxide_core::quantity!("roundoff_floors"))?,
            diagnostics: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("component_diagnostics"),
            )?,
        })
    }
    /// Refines inside the solve context and writes the estimate to `out`.
    #[inline(always)]
    pub(super) fn run<F, V, O, const WIDTH: usize>(
        &mut self,
        options: &LevelOptions<S>,
        factors: &Factors<S>,
        function: &F,
        interval: Interval<S>,
        out: &mut O,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Outcome<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let components = if WIDTH == 0 { self.components } else { WIDTH };
        let Some(geometry) = IntervalGeometry::prepare(interval)?.filter(|_| components > 0) else {
            return crate::execution::zero(
                &mut self.values,
                out,
                if components == 0 {
                    IntegrationTermination::FixedRuleCompleted
                } else {
                    IntegrationTermination::DegenerateDomain
                },
                context,
            );
        };
        let (termination, has_error) = match self
            .refine::<F, V, WIDTH>(options, factors, function, interval, geometry, context)?
        {
            Ok(result) => result,
            Err(stop) => return Ok(Outcome::Unpublished(stop)),
        };
        for ((diagnostic, &error), &floor) in self.diagnostics[..components]
            .iter_mut()
            .zip(&self.errors[..components])
            .zip(&self.floors[..components])
        {
            let mut parts = IntegrationErrorComponents::discretization_only(error);
            if !has_error {
                parts.discretization = None;
            }
            *diagnostic = parts.with_roundoff(floor);
        }
        context.commit_accepted(PartialResultTrust::Admissible)?;
        write_value(&self.values[..components], out)?;
        Ok(Outcome::Published {
            termination,
            subdivisions: 0,
            errors: IntegrationErrorComponents::envelope(&self.diagnostics[..components]),
        })
    }

    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn refine<F, V, const WIDTH: usize>(
        &mut self,
        options: &LevelOptions<S>,
        factors: &Factors<S>,
        function: &F,
        interval: Interval<S>,
        geometry: IntervalGeometry<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Result<(IntegrationTermination, bool), IntegrationTermination>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let width = if WIDTH == 0 { self.components } else { WIDTH };
        let outputs = Outputs {
            values: &mut self.values[..width],
            errors: &mut self.errors[..width],
            scales: &mut self.scales[..width],
            floors: &mut self.floors[..width],
        };
        refine_components::<S, F, V, WIDTH>(
            options,
            factors,
            function,
            interval,
            geometry,
            &mut self.sampling,
            &mut self.data[..width],
            outputs,
            context,
        )
    }
}

struct Outputs<'a, S: RealScalar> {
    values: &'a mut [S],
    errors: &'a mut [S],
    scales: &'a mut [S],
    floors: &'a mut [S],
}

#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn refine_components<S: RealScalar, F, V, const WIDTH: usize>(
    options: &LevelOptions<S>,
    factors: &Factors<S>,
    function: &F,
    interval: Interval<S>,
    geometry: IntervalGeometry<S>,
    sampling: &mut Sampling<S>,
    data: &mut [Component<S>],
    output: Outputs<'_, S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Result<(IntegrationTermination, bool), IntegrationTermination>, SciOxideError>
where
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
{
    let width = if WIDTH == 0 { data.len() } else { WIDTH };
    let Outputs {
        values,
        errors,
        scales,
        floors,
    } = output;
    let tolerance = options.tolerance();
    values.fill(S::ZERO);
    errors.fill(S::ZERO);
    floors.fill(S::ZERO);
    if let Some(stop) = crate::workspace::admit(context, 2, false)? {
        return Ok(Err(stop));
    }
    let ends = [interval.lower(), interval.upper()];
    let witness = &factors.witness;
    sampling.capture = Capture::NONE;
    if width == 1 && options.execution_policy() == scioxide_core::ExecutionPolicy::Sequential {
        sampling.sample_scalar_points(function, &ends, context)?;
    } else {
        sampling.sample::<WIDTH, F, V>(function, 2, |index| Ok(ends[index]), options, context)?;
    }
    for (index, component) in data[..width].iter_mut().enumerate() {
        let [value, floor] =
            component.initialize(geometry, &sampling.sums[index][0], &sampling.sums[index][1])?;
        values[index] = value;
        floors[index] = floor;
    }
    let mut termination = IntegrationTermination::RefinementLimit;
    let mut has_error = false;
    let mut witnessed = false;
    // Counted levels whose witness errors are deferred.
    let mut unsettled = 0_u32;
    for level in 1..options.max_levels().get().min(MAX_LEVELS) {
        sampling.capture = witness.capture(level);
        // Level `level` adds the `n = 2^(level-1)` midpoints of the previous grid, so all
        // earlier callback values are reused through `T(level) = T(level-1)/2 + h * sum`.
        let n = 1usize << (level - 1);
        // The bounded level count makes this power of two exact in both formats.
        let denominator = S::from_f64_ieee(n as f64);
        let step = geometry
            .native_span()
            .map(|span| span * factors.steps[level]);
        // Eight ULPs of separation cover affine-map and grid rounding at both
        // endpoints, proving every new midpoint lies strictly inside its cell.
        let spacing_guard = S::try_from_usize_exact(8)?
            * S::EPSILON
            * interval.lower().abs().max(interval.upper().abs());
        let direct = step.is_some_and(|h| h >= S::MIN_POSITIVE && h > spacing_guard);
        let point = |index: usize| -> Result<Option<S>, SciOxideError> {
            let a = geometry.point_at_fraction(S::try_from_usize_exact(index)? / denominator)?;
            let b =
                geometry.point_at_fraction(S::try_from_usize_exact(index + 1)? / denominator)?;
            let m = Interval::new(a, b)?.midpoint()?;
            Ok((a < m && m < b).then_some(m))
        };
        // All new midpoints are validated before any callback is admitted: once the grid
        // is finer than the floating-point spacing, a midpoint coincides with a cell edge
        // and further halving only repeats samples, so the level is reported as
        // stagnation without spending evaluations on it.
        let mut valid = true;
        if !direct {
            for index in 0..n {
                if point(index)?.is_none() {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            termination = IntegrationTermination::Stagnation;
            break;
        }
        if let Some(stop) = crate::workspace::admit(context, n, false)? {
            termination = stop;
            break;
        }
        if direct
            && width == 1
            && options.execution_policy() == scioxide_core::ExecutionPolicy::Sequential
        {
            let h = step.ok_or_else(|| {
                crate::kernel::invalid(
                    scioxide_core::argument!("interval"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "native grid has no step",
                )
            })?;
            let point = {
                #[inline(always)]
                |i| {
                    if n == 1 {
                        return geometry.point_at_fraction(S::HALF);
                    }
                    // The bounded level count and spacing proof make this
                    // conversion exact; all coordinates use this same map.
                    let index = S::from_f64_ieee(f64::from(i as u32));
                    Ok(interval.lower() + h * (S::TWO * index + S::ONE))
                }
            };
            if n > 1 && n <= NATIVE_MIDPOINTS.div_ceil(2) {
                let span = geometry.native_span().ok_or_else(|| {
                    crate::kernel::invalid(
                        scioxide_core::argument!("interval"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "admitted dyadic grid has no native span",
                    )
                })?;
                sampling.sample_canonical_grid(
                    function,
                    &factors.canonical_midpoints[n - 1..2 * n - 1],
                    span,
                    interval.lower(),
                    context,
                )?;
            } else {
                sampling.sample_scalar_grid(function, n, point, context)?;
            }
        } else {
            sampling.sample::<WIDTH, F, V>(
                function,
                n,
                |index| {
                    if direct {
                        return geometry.point_at_fraction(
                            (S::try_from_usize_exact(index)? + S::HALF) / denominator,
                        );
                    }
                    point(index)?.ok_or_else(|| {
                        crate::kernel::invalid(
                            scioxide_core::argument!("interval"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "validated midpoint changed",
                        )
                    })
                },
                options,
                context,
            )?;
        }
        // From the first level whose errors count, one off-grid witness sample
        // checks the grid near it: every grid sample is dyadic, so integrands in
        // step with the grid look smooth and self-consistent on it. It is one
        // admitted unit after this level's grid; a stop there still records the level.
        let mut stopped = None;
        if level >= WITNESS_LEVEL && !witnessed {
            stopped = crate::workspace::admit(context, 1, false)?;
            if stopped.is_none() {
                let point = geometry.point_at_fraction(witness.fraction)?;
                sample_1d(
                    function,
                    core::slice::from_ref(&point),
                    width,
                    &mut sampling.rows[..width],
                    options.execution_policy(),
                    context,
                )?;
                for (component, &value) in data[..width].iter_mut().zip(&sampling.rows[..width]) {
                    component.witness = value;
                }
                witnessed = true;
            }
        }
        // Every output component shares the callback grid but keeps its own tableau,
        // error and tolerance scale, so components converge (and are judged) jointly
        // without one component's extrapolation affecting another's.
        for (index, component) in data[..width].iter_mut().enumerate() {
            let [value, error, floor, scale] = component.advance(
                level,
                factors,
                step,
                geometry,
                &sampling.sums[index][0],
                &sampling.sums[index][1],
            )?;
            values[index] = value;
            errors[index] = error;
            floors[index] = floor;
            scales[index] = scale;
        }
        has_error = true;
        if level < WITNESS_LEVEL {
            continue;
        }
        // Witness errors only raise level errors, so a level that cannot stop
        // defers its own: success never publishes it, and an unsuccessful stop
        // settles every deferred level before publishing the smallest error.
        let mut verdict = match stopped {
            Some(stop) => Some(stop),
            None => counted_verdict(level, &data[..width], errors, floors, scales, tolerance)?,
        };
        if verdict.is_none() {
            unsettled |= 1 << level;
            continue;
        }
        let mut raised = false;
        for (index, component) in data[..width].iter_mut().enumerate() {
            if witnessed {
                let error = witness.error(
                    level,
                    &sampling.retained[index],
                    component.witness,
                    geometry,
                )?;
                raised |= error > errors[index];
                errors[index] = errors[index].max(error);
            }
            component.settle(level, errors[index]);
        }
        if raised {
            verdict = counted_verdict(level, &data[..width], errors, floors, scales, tolerance)?;
        }
        if let Some(stop) = verdict {
            termination = stop;
            break;
        }
    }
    if has_error && termination != IntegrationTermination::ToleranceReached {
        for (index, component) in data[..width].iter_mut().enumerate() {
            if component.deferred > 0 {
                component.replay_exploration(factors)?;
            }
            if unsettled != 0 {
                component.settle_deferred(
                    unsettled,
                    witness,
                    &sampling.retained[index],
                    geometry,
                )?;
            }
            values[index] = component.best_value;
            errors[index] = component.best_error;
            floors[index] = component.best_floor;
        }
    }
    Ok(Ok((termination, has_error)))
}

/// Termination that a counted level's errors imply, if any.
#[inline(always)]
fn counted_verdict<S: RealScalar>(
    level: usize,
    data: &[Component<S>],
    errors: &[S],
    floors: &[S],
    scales: &[S],
    tolerance: &ErrorTolerance<S>,
) -> Result<Option<IntegrationTermination>, SciOxideError> {
    // Absolute tolerance alone can accept tiny, unresolved aliasing
    // residues. Require changes to resolve the sampled mass as well.
    let mut observed = false;
    let mut stable = true;
    for (component, &error) in data.iter().zip(errors) {
        if let Some(mass) = component.native_absolute
            && let Some(guarded) = native_product(
                error,
                S::from_f64_ieee(crate::kernel::EXTRAPOLATION_ERROR_GUARD as f64),
            )
        {
            observed |= mass != S::ZERO;
            stable &= guarded <= mass;
            continue;
        }
        let mass = match component.native_absolute {
            Some(value) => Scaled::from_value(value)?,
            None => component.absolute,
        };
        observed |= !mass.is_zero();
        stable &= Scaled::from_value(error)?
            .mul(Scaled::from_value(S::try_from_usize_exact(
                crate::kernel::EXTRAPOLATION_ERROR_GUARD,
            )?)?)?
            .abs_cmp(mass)
            .is_le();
    }
    if level >= MIN_CONVERGENCE_LEVEL
        && observed
        && stable
        && tolerance.is_satisfied(errors, scales)?
    {
        return Ok(Some(IntegrationTermination::ToleranceReached));
    }
    if errors.iter().zip(floors).all(|(e, f)| e <= f) && !tolerance.is_satisfied(floors, scales)? {
        return Ok(Some(IntegrationTermination::Stagnation));
    }
    Ok(None)
}

/// The rounded native total; promoted or overflowing sums take the extended path.
#[inline(always)]
fn native_total<S: RealScalar>(sum: &PromotingCompensatedSum<S>) -> Option<S> {
    sum.native()?.total().ok()
}

/// Only zero or normal products can enter an ordinary recurrence.
#[inline(always)]
fn native_product<S: RealScalar>(a: S, b: S) -> Option<S> {
    let value = a * b;
    (value.is_finite() && (a == S::ZERO || b == S::ZERO || value.abs() >= S::MIN_POSITIVE))
        .then_some(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "range_tests.rs"]
mod range_tests;
