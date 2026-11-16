//! Latin hypercube and Poisson disk designs with explicit work limits.
use super::DiscrepancyMethod;
use super::discrepancy::KernelSums;
use super::plan::invalid;
use crate::{AdvanceOnError, BitGenerator, SamplingRequirements, SamplingWorkspace, StatePolicy};
use alloc::vec::Vec;
use scioxide_core::backend::RealBackend;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

/// Why [`latin_hypercube_into`] stopped.
///
/// Every status comes with a complete design. See [`latin_hypercube_with_options`] for an
/// example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatinHypercubeTermination {
    /// No optimization was requested (`max_proposals` is 0).
    Complete,
    /// `stagnation_limit` consecutive swap proposals did not lower the discrepancy.
    Stagnation,
    /// All `max_proposals` swap proposals were made.
    BudgetExhausted,
    /// The execution context was cancelled or reached its deadline before a proposal.
    Stopped,
}
/// Termination status and diagnostics of [`latin_hypercube_with_options`] and
/// [`latin_hypercube_into`].
///
/// The design always has all `n` points. The discrepancies measure the improvement from
/// optimization. See [`latin_hypercube_with_options`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct LatinHypercubeReport {
    termination: LatinHypercubeTermination,
    proposals: usize,
    initial_discrepancy: Option<f64>,
    final_discrepancy: Option<f64>,
}
impl LatinHypercubeReport {
    /// Returns why generation stopped.
    pub const fn termination(&self) -> LatinHypercubeTermination {
        self.termination
    }
    /// Returns the number of swap proposals made, including rejected ones.
    pub const fn proposals(&self) -> usize {
        self.proposals
    }
    /// Returns the squared centered L² discrepancy before optimization, or `None` if no
    /// optimization was requested.
    pub const fn initial_discrepancy(&self) -> Option<f64> {
        self.initial_discrepancy
    }
    /// Returns the squared centered L² discrepancy of the returned design, or `None` if no
    /// optimization was requested.
    pub const fn final_discrepancy(&self) -> Option<f64> {
        self.final_discrepancy
    }
}
/// Why [`PoissonDisk::generate_into`] stopped.
///
/// Every status comes with a valid design: the accepted points are pairwise at least the radius
/// apart. See [`PoissonDisk`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonDiskTermination {
    /// The design holds `capacity` points.
    Complete,
    /// No active point is left: every accepted point failed `candidates_per_point` candidates in
    /// a row. The design is not maximal in general: with the default 30 candidates, points of
    /// the box up to about `1.7 r` from every accepted point remain near the boundary and up to
    /// about `1.2 r` in the interior, so more points would fit.
    ActiveListExhausted,
    /// The candidate limit or the neighbor-work limit of [`PoissonDiskOptions`] ran out.
    BudgetExhausted,
    /// The execution context was cancelled or reached its deadline before a candidate.
    Stopped,
}
/// Termination status and work counts of [`PoissonDisk::generate`] and
/// [`PoissonDisk::generate_into`].
///
/// The accepted points are the first [`points`](Self::points) rows of the output. See
/// [`PoissonDisk`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct PoissonDiskReport {
    termination: PoissonDiskTermination,
    points: usize,
    proposals: usize,
    distance_checks: usize,
}
impl PoissonDiskReport {
    /// Returns why generation stopped.
    pub const fn termination(&self) -> PoissonDiskTermination {
        self.termination
    }
    /// Returns the number of accepted points.
    pub const fn points(&self) -> usize {
        self.points
    }
    /// Returns the number of candidates drawn, including the initial point and candidates that
    /// fell outside the box.
    pub const fn proposals(&self) -> usize {
        self.proposals
    }
    /// Returns the neighbor work done: one unit per visited grid cell, empty or not, and one per
    /// distance comparison.
    pub const fn distance_checks(&self) -> usize {
        self.distance_checks
    }
}
/// Point placement within strata and optimization limits for [`latin_hypercube_with_options`]
/// and [`latin_hypercube_into`].
///
/// See [`latin_hypercube_with_options`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct LatinHypercubeOptions {
    centered: bool,
    max_proposals: usize,
    stagnation_limit: usize,
}
impl Default for LatinHypercubeOptions {
    fn default() -> Self {
        Self {
            centered: false,
            max_proposals: 0,
            stagnation_limit: 100,
        }
    }
}
impl LatinHypercubeOptions {
    /// Returns whether points sit at stratum midpoints instead of uniformly within their stratum
    /// (default `false`).
    pub const fn centered(self) -> bool {
        self.centered
    }
    /// Returns the options with midpoint placement set to `value`.
    pub const fn with_centered(mut self, value: bool) -> Self {
        self.centered = value;
        self
    }
    /// Returns the maximum number of swap proposals. Zero, the default, turns optimization off.
    pub const fn max_proposals(self) -> usize {
        self.max_proposals
    }
    /// Returns the options with at most `value` swap proposals.
    pub const fn with_max_proposals(mut self, value: usize) -> Self {
        self.max_proposals = value;
        self
    }
    /// Returns the number of consecutive proposals without improvement after which optimization
    /// stops (default 100, must be positive).
    pub const fn stagnation_limit(self) -> usize {
        self.stagnation_limit
    }
    /// Returns the options with stagnation limit `value`. [`latin_hypercube_into`] checks it.
    pub const fn with_stagnation_limit(mut self, value: usize) -> Self {
        self.stagnation_limit = value;
        self
    }
}
/// Returns a Latin hypercube design of `n` points in `[0, 1)ᵈ`, row by row.
///
/// Each coordinate axis is divided into `n` equal strata `[i/n, (i+1)/n)`, and each stratum of
/// each axis holds exactly one point (McKay, Beckman and Conover 1979). Column `k` places point
/// `i` at `(i + u)/n` with `u` uniform on `[0, 1)`, and then shuffles the column with an
/// independent random permutation. Generation costs O(n d) time and `n d` uniform draws, plus
/// about `n d` draws for the shuffles. On error the generator stays advanced by the words it
/// consumed.
///
/// It equals [`latin_hypercube_with_options`] with the default options without the report.
/// That form can also center points in their strata and optimize the design, and
/// [`latin_hypercube_into`] writes into a caller's buffer.
///
/// # Errors
///
/// The errors of [`latin_hypercube_with_options`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{SeedableRng, Xoshiro256StarStar, latin_hypercube};
/// let design: Vec<f64> = latin_hypercube(8, 3, &mut Xoshiro256StarStar::from_seed(42))?;
/// assert_eq!(design.len(), 24);
/// for column in 0..3 {
///     let mut strata: Vec<usize> =
///         design.iter().skip(column).step_by(3).map(|&x| (x * 8.0) as usize).collect();
///     strata.sort_unstable();
///     assert_eq!(strata, (0..8).collect::<Vec<_>>()); // one point per stratum
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting
/// values of input variables in the analysis of output from a computer code", *Technometrics*
/// 21(2), 1979, 239–245.
pub fn latin_hypercube<S: RealScalar, R: BitGenerator>(
    n: usize,
    dim: usize,
    rng: &mut R,
) -> Result<Vec<S>, SciOxideError> {
    latin_hypercube_with_options(n, dim, rng, LatinHypercubeOptions::default())
        .map(|(design, _)| design)
}
/// Returns a Latin hypercube design of `n` points in `[0, 1)ᵈ`, row by row, with its report.
///
/// The design is that of [`latin_hypercube`], with points at stratum midpoints `(i + 1/2)/n`
/// if [`LatinHypercubeOptions::centered`] is set. With `max_proposals > 0` the design is
/// improved by random swaps: each proposal picks a column and two rows, swaps their entries
/// (which keeps the Latin property), and keeps the swap only if the squared centered L²
/// discrepancy (Hickernell 1998) decreases. A proposal costs O(n d). Optimization stops after
/// `max_proposals` proposals or `stagnation_limit` proposals in a row without improvement, as
/// reported by [`LatinHypercubeTermination`].
///
/// The function runs on the calling thread and allocates the design and one staging copy of
/// `n d` elements. On error the generator stays advanced by the words it consumed. Use
/// [`latin_hypercube_into`] to reuse buffers, to restore the generator on error or to observe
/// a cancellation or deadline.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero, or if `S` is `f32` and `n > 2²⁴`. Nothing is allocated.
/// - `SizeOverflow` if `n · dim` overflows `usize`.
/// - `AllocationFailure` if the design cannot be allocated.
/// - The errors of [`latin_hypercube_into`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     LatinHypercubeOptions, LatinHypercubeTermination, SeedableRng, SplitMix64,
///     latin_hypercube_with_options,
/// };
/// let options = LatinHypercubeOptions::default().with_max_proposals(50);
/// let (design, report) =
///     latin_hypercube_with_options::<f64, _>(8, 2, &mut SplitMix64::from_seed(1), options)?;
/// assert_eq!(design.len(), 16);
/// assert_eq!(report.termination(), LatinHypercubeTermination::BudgetExhausted);
/// assert!(report.final_discrepancy() <= report.initial_discrepancy());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. J. Hickernell, "A generalized discrepancy and quadrature error bound", *Math. Comp.*
/// 67(221), 1998, 299–322.
pub fn latin_hypercube_with_options<S: RealScalar, R: BitGenerator>(
    n: usize,
    dim: usize,
    rng: &mut R,
    options: LatinHypercubeOptions,
) -> Result<(Vec<S>, LatinHypercubeReport), SciOxideError> {
    if n == 0 {
        return Err(invalid(
            scioxide_core::argument!("n"),
            "positive point count required",
        ));
    }
    validate_strata_count::<S>(n, scioxide_core::argument!("n"))?;
    let mut design = scioxide_core::backend::try_filled(
        scioxide_core::backend::checked_mul(n, dim, scioxide_core::quantity!("lhs_shape"))?,
        S::ZERO,
        scioxide_core::quantity!("lhs_output"),
    )?;
    let report = latin_hypercube_into(
        dim,
        rng,
        AdvanceOnError,
        options,
        &mut design,
        &mut SamplingWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    Ok((design, report))
}
/// Writes a Latin hypercube design of `out.len() / dim` points into `out`, row by row, and
/// returns its report.
///
/// The design and its optimization are those of [`latin_hypercube_with_options`]. It is built
/// in `workspace` and copied to `out` only on success, because its rows depend on each other.
/// `state` is [`AdvanceOnError`](crate::AdvanceOnError) or [`Rollback`](crate::Rollback) and
/// sets the generator state after an error. `workspace` grows to `out.len()` elements if it is
/// smaller, so a reused workspace makes later calls allocation-free. `cx` is checked once
/// before the first draw and again before each swap proposal: a cancellation or deadline seen
/// there ends optimization with a complete design and [`LatinHypercubeTermination::Stopped`].
///
/// # Errors
///
/// - `InvalidInput` if `dim` is zero, `out` is empty or `options.stagnation_limit` is zero.
///   Nothing is drawn.
/// - `DimensionMismatch` if `out.len()` is not a multiple of `dim`. Nothing is drawn.
/// - `InvalidInput` if `S` is `f32` and there are more than 2²⁴ points, so strata are not
///   representable.
/// - `NumericalInstability` if a discrepancy sum overflows or a stratum point cannot be
///   represented.
/// - `AllocationFailure` if the workspace must grow and cannot.
/// - `ExecutionStopped` if `cx` is cancelled or past its deadline before the first draw.
/// - Errors of `rng` and of the clock of `cx`.
pub fn latin_hypercube_into<S: RealScalar, R: BitGenerator, P: StatePolicy<R>>(
    dim: usize,
    rng: &mut R,
    state: P,
    options: LatinHypercubeOptions,
    out: &mut [S],
    workspace: &mut SamplingWorkspace<S>,
    cx: ExecutionContext<'_>,
) -> Result<LatinHypercubeReport, SciOxideError> {
    // The policy is a type-level choice; its value carries no data.
    let _ = state;
    if dim == 0 || out.is_empty() || options.stagnation_limit == 0 {
        return Err(invalid(
            if dim == 0 {
                scioxide_core::argument!("dim")
            } else if out.is_empty() {
                scioxide_core::argument!("out")
            } else {
                scioxide_core::argument!("options.stagnation_limit")
            },
            "positive dimension, output and stagnation limit required",
        ));
    }
    if !out.len().is_multiple_of(dim) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(dim),
            scioxide_core::Dimensions::one(out.len() % dim),
        ));
    }
    let n = out.len() / dim;
    validate_strata_count::<S>(n, scioxide_core::argument!("out"))?;
    workspace.stage(&SamplingRequirements::new(out.len()), S::ZERO)?;
    cx.check()?;
    crate::transaction::run::<_, _, P>(rng, |rng| {
        // Standard LHS: column by column, place one point in each of the n
        // strata (row i gets stratum i, jittered or centered), then shuffle
        // the column independently with Fisher-Yates so the strata pairing
        // across dimensions is a uniform random permutation.
        let points = workspace.as_mut_slice();
        for col in 0..dim {
            for row in 0..n {
                let u = if options.centered {
                    S::HALF
                } else {
                    S::from_u64_half_open01(rng.next_uniform_u64()?)
                };
                points[row * dim + col] = represented_stratum_sample(row, n, u)?;
            }
            for row in (1..n).rev() {
                let other = crate::sampling::uniform_index(row + 1, rng)?;
                points.swap(row * dim + col, other * dim + col);
            }
        }
        // Exact centered sums: each proposal removes and re-adds the terms
        // of its two rows, which equals full evaluation bit for bit.
        let mut sums = if options.max_proposals > 0 {
            Some(KernelSums::of(points, dim, DiscrepancyMethod::Centered)?)
        } else {
            None
        };
        let argument = scioxide_core::argument!("out");
        let initial = sums
            .as_ref()
            .map(|sums| sums.value(dim, n, argument))
            .transpose()?;
        let mut current = initial;
        let mut stale = 0;
        let mut report = LatinHypercubeReport {
            termination: LatinHypercubeTermination::Complete,
            proposals: 0,
            initial_discrepancy: initial,
            final_discrepancy: initial,
        };
        // Optional greedy optimization: propose swapping two entries within
        // one column (which keeps the Latin property) and keep the swap only
        // if the centered L2 discrepancy decreases.
        for _ in 0..options.max_proposals {
            if cx.stop_reason()?.is_some() {
                report.termination = LatinHypercubeTermination::Stopped;
                break;
            }
            report.proposals += 1;
            let col = crate::sampling::uniform_index(dim, rng)?;
            let (i, j) = (
                crate::sampling::uniform_index(n, rng)?,
                crate::sampling::uniform_index(n, rng)?,
            );
            let (a, b) = (i * dim + col, j * dim + col);
            let Some(committed) = sums.as_ref() else {
                break;
            };
            // A swap within one row changes nothing.
            let candidate = if i == j {
                None
            } else {
                let mut candidate = committed.clone();
                candidate.rows(points, dim, (i, j), -1)?;
                points.swap(a, b);
                candidate.rows(points, dim, (i, j), 1)?;
                let value = candidate.value(dim, n, argument)?;
                Some((candidate, value))
            };
            if let Some((candidate, value)) =
                candidate.filter(|&(_, new)| current.is_some_and(|old| new < old))
            {
                sums = Some(candidate);
                current = Some(value);
                stale = 0;
            } else {
                if i != j {
                    points.swap(a, b);
                }
                stale += 1;
            }
            report.termination = LatinHypercubeTermination::BudgetExhausted;
            if stale >= options.stagnation_limit {
                report.termination = LatinHypercubeTermination::Stagnation;
                break;
            }
        }
        report.final_discrepancy = current;
        out.copy_from_slice(points);
        Ok(report)
    })
}

/// Work limits for a [`PoissonDisk`] design.
///
/// See [`PoissonDisk`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct PoissonDiskOptions {
    candidates_per_point: usize,
    max_proposals: usize,
    max_grid_cells: usize,
    max_distance_checks: usize,
}
impl Default for PoissonDiskOptions {
    fn default() -> Self {
        Self {
            candidates_per_point: 30,
            max_proposals: 1 << 20,
            max_grid_cells: 1 << 20,
            max_distance_checks: 1 << 24,
        }
    }
}
impl PoissonDiskOptions {
    /// Returns the number `k` of candidates tried around an active point before it is retired
    /// (default 30, Bridson's choice; must be positive).
    pub const fn candidates_per_point(self) -> usize {
        self.candidates_per_point
    }
    /// Returns the options with `value` candidates per active point.
    /// [`PoissonDisk::with_options`] checks it.
    pub const fn with_candidates_per_point(mut self, value: usize) -> Self {
        self.candidates_per_point = value;
        self
    }
    /// Returns the maximum number of candidates, including the initial point and candidates
    /// outside the box (default 2²⁰).
    pub const fn max_proposals(self) -> usize {
        self.max_proposals
    }
    /// Returns the options with at most `value` candidates.
    pub const fn with_max_proposals(mut self, value: usize) -> Self {
        self.max_proposals = value;
        self
    }
    /// Returns the maximum number of background grid cells (default 2²⁰, must be positive).
    pub const fn max_grid_cells(self) -> usize {
        self.max_grid_cells
    }
    /// Returns the options with at most `value` grid cells. [`PoissonDisk::with_options`]
    /// checks it.
    pub const fn with_max_grid_cells(mut self, value: usize) -> Self {
        self.max_grid_cells = value;
        self
    }
    /// Returns the maximum neighbor work per design (default 2²⁴). Each visited grid cell and
    /// each distance comparison counts one unit.
    pub const fn max_distance_checks(self) -> usize {
        self.max_distance_checks
    }
    /// Returns the options with at most `value` units of neighbor work.
    pub const fn with_max_distance_checks(mut self, value: usize) -> Self {
        self.max_distance_checks = value;
        self
    }
}
/// Returns a Poisson disk design of at most `capacity` points in the box `[lower, upper]`,
/// pairwise at least `radius` apart, row by row.
///
/// It runs [`PoissonDisk::generate`] with the default [`PoissonDiskOptions`] and returns the
/// accepted points. Running out of active points before `capacity` is reached is a success
/// ([`PoissonDiskTermination::ActiveListExhausted`]), although more points may still fit. On
/// error the generator stays advanced by the words it consumed.
///
/// # Errors
///
/// - The errors of [`PoissonDisk::new`] and [`PoissonDisk::generate`].
/// - `ConvergenceFailure` if the candidate or neighbor-work limit of the default options runs
///   out ([`PoissonDiskTermination::BudgetExhausted`]).
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     GeometricDiscrepancy, SeedableRng, SplitMix64, geometric_discrepancy, poisson_disk,
/// };
/// let points = poisson_disk(&[0.0; 2], &[1.0; 2], 0.2, 64, &mut SplitMix64::from_seed(3))?;
/// assert!(points.len() % 2 == 0 && points.len() <= 2 * 64);
/// let separation = geometric_discrepancy(&points, 2, GeometricDiscrepancy::MinimumDistance)?;
/// assert!(separation >= 0.2); // the defining property of the design
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn poisson_disk<S: RealScalar, R: BitGenerator>(
    lower: &[S],
    upper: &[S],
    radius: S,
    capacity: usize,
    rng: &mut R,
) -> Result<Vec<S>, SciOxideError> {
    let (points, report) = PoissonDisk::new(lower, upper, radius, capacity)?.generate(rng)?;
    if report.termination == PoissonDiskTermination::BudgetExhausted {
        return Err(crate::law::iteration_limit(report.proposals));
    }
    Ok(points)
}
/// Poisson disk design in a box of 1 to 6 dimensions: random points pairwise at least `r` apart.
///
/// The method is Bridson's algorithm (2007). The first point is uniform in the box
/// `[lower, upper]` and becomes active. While active points remain, one is chosen at random
/// and up to `k` candidates are drawn uniformly by volume from the spherical shell of radii
/// `[r, 2r)` around it. A candidate inside the box and at least `r` from every accepted point
/// is accepted and becomes active. If all `k` candidates fail, the point is retired. A
/// background grid with cells of side `r/√d` limits each distance check to nearby cells, so
/// the expected cost is O(k N) for `N` points.
///
/// Generation stops when the design holds `capacity` points, when no active point is left, or
/// when a limit in [`PoissonDiskOptions`] runs out. The accepted points are always valid. The
/// type stores the validated box, the radius and the grid shape; reserve a
/// [`PoissonDiskWorkspace`] once with [`Self::workspace`] and reuse it with
/// [`Self::generate_into`]. [`poisson_disk`] is the one-shot form.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     GeometricDiscrepancy, PoissonDisk, PoissonDiskTermination, SeedableRng, SplitMix64,
///     geometric_discrepancy,
/// };
/// let design = PoissonDisk::new(&[0.0; 2], &[1.0; 2], 0.2, 64)?;
/// let (points, report) = design.generate(&mut SplitMix64::from_seed(3))?;
/// assert_eq!(report.termination(), PoissonDiskTermination::ActiveListExhausted);
/// assert_eq!(points.len(), 2 * report.points());
/// let separation = geometric_discrepancy(&points, 2, GeometricDiscrepancy::MinimumDistance)?;
/// assert!(separation >= 0.2); // the defining property of the design
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. Bridson, "Fast Poisson disk sampling in arbitrary dimensions", *ACM SIGGRAPH 2007
/// Sketches*, 2007, article 22.
#[derive(Debug, Clone)]
pub struct PoissonDisk<S> {
    lower: Vec<S>,
    upper: Vec<S>,
    radius: f64,
    cell: f64,
    shape: Vec<usize>,
    cells: usize,
    capacity: usize,
    options: PoissonDiskOptions,
}
/// Working storage for [`PoissonDisk::generate_into`]: the points, the active list and the
/// background grid.
///
/// Create it with [`PoissonDisk::workspace`]. See [`PoissonDisk`] for an example.
#[derive(Debug, Clone)]
pub struct PoissonDiskWorkspace<S> {
    points: Vec<S>,
    active: Vec<usize>,
    grid: Vec<usize>,
    candidate: Vec<S>,
    low: Vec<usize>,
    high: Vec<usize>,
    index: Vec<usize>,
}
impl<S: RealScalar> PoissonDisk<S> {
    /// Creates a design of at most `capacity` points in the box `[lower, upper]` with minimum
    /// distance `radius`.
    ///
    /// It uses the default [`PoissonDiskOptions`]: 30 candidates per point, at most 2²⁰
    /// candidates and 2²⁰ grid cells, and 2²⁴ units of neighbor work. [`Self::with_options`]
    /// sets the limits.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::with_options`].
    pub fn new(
        lower: &[S],
        upper: &[S],
        radius: S,
        capacity: usize,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            lower,
            upper,
            radius,
            capacity,
            PoissonDiskOptions::default(),
        )
    }
    /// Creates the design of [`Self::new`] with the limits of `options`.
    ///
    /// The background grid has `Πⱼ ⌈(upperⱼ − lowerⱼ) √d / radius⌉` cells.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `lower` does not have 1 to 6 coordinates, `upper` has a different
    ///   length, `radius` is not positive and finite, a bound is not finite or
    ///   `lowerⱼ ≥ upperⱼ`, `capacity` or the candidates per point is zero, or the grid exceeds
    ///   `max_grid_cells`.
    /// - `SizeOverflow` if the output or grid size overflows `usize`.
    /// - `AllocationFailure` if the bounds cannot be copied.
    pub fn with_options(
        lower: &[S],
        upper: &[S],
        radius: S,
        capacity: usize,
        options: PoissonDiskOptions,
    ) -> Result<Self, SciOxideError> {
        let dim = lower.len();
        if !(1..=6).contains(&dim)
            || upper.len() != dim
            || !radius.is_finite()
            || radius <= S::ZERO
            || capacity == 0
            || options.candidates_per_point == 0
            || options.max_grid_cells == 0
        {
            return Err(invalid(
                if !(1..=6).contains(&dim) {
                    scioxide_core::argument!("lower")
                } else if upper.len() != dim {
                    scioxide_core::argument!("upper")
                } else if !radius.is_finite() || radius <= S::ZERO {
                    scioxide_core::argument!("radius")
                } else if capacity == 0 {
                    scioxide_core::argument!("capacity")
                } else {
                    scioxide_core::argument!("options")
                },
                "invalid shape, radius, capacity or limits",
            ));
        }
        let _ = scioxide_core::backend::checked_mul(
            capacity,
            dim,
            scioxide_core::quantity!("poisson_capacity"),
        )?;
        // Bridson's background grid: with cell side r / sqrt(d), a cell's
        // diagonal equals r, so each cell holds at most one accepted point and
        // a single `usize` per cell suffices (usize::MAX marks empty).
        let radius = radius.into_f64();
        let cell = radius / RealScalar::sqrt(dim as f64);
        if cell == 0.0 {
            return Err(invalid(
                scioxide_core::argument!("radius"),
                "cell width underflow",
            ));
        }
        let mut shape = scioxide_core::backend::try_with_capacity(
            dim,
            scioxide_core::quantity!("poisson_grid_shape"),
        )?;
        let mut cells = 1;
        for (&lo, &hi) in lower.iter().zip(upper) {
            let width = hi.into_f64() - lo.into_f64();
            if !lo.is_finite() || !hi.is_finite() || width <= 0.0 || !width.is_finite() {
                return Err(invalid(
                    if lo.is_finite() {
                        scioxide_core::argument!("upper")
                    } else {
                        scioxide_core::argument!("lower")
                    },
                    "finite ordered representable box required",
                ));
            }
            let count = RealScalar::ceil(width / cell).try_to_usize_exact()?;
            if count == 0 {
                return Err(invalid(
                    scioxide_core::argument!("radius"),
                    "grid cell ratio underflow",
                ));
            }
            cells = scioxide_core::backend::checked_mul(
                cells,
                count,
                scioxide_core::quantity!("poisson_grid"),
            )?;
            if cells > options.max_grid_cells {
                return Err(invalid(
                    scioxide_core::argument!("options.max_grid_cells"),
                    "grid exceeds caller cell budget",
                ));
            }
            shape.push(count);
        }
        let copy = |x: &[S]| {
            let mut v = scioxide_core::backend::try_with_capacity(
                x.len(),
                scioxide_core::quantity!("poisson_bounds"),
            )?;
            v.extend_from_slice(x);
            Ok::<_, SciOxideError>(v)
        };
        Ok(Self {
            lower: copy(lower)?,
            upper: copy(upper)?,
            radius,
            cell,
            shape,
            cells,
            capacity,
            options,
        })
    }
    /// Returns a workspace for [`Self::generate_into`], with storage for `capacity` points and
    /// the background grid.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self) -> Result<PoissonDiskWorkspace<S>, SciOxideError> {
        let d = self.lower.len();
        Ok(PoissonDiskWorkspace {
            points: scioxide_core::backend::try_filled(
                scioxide_core::backend::checked_mul(
                    self.capacity,
                    d,
                    scioxide_core::quantity!("poisson_output"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("poisson_points"),
            )?,
            active: scioxide_core::backend::try_with_capacity(
                self.capacity,
                scioxide_core::quantity!("poisson_active_list"),
            )?,
            grid: scioxide_core::backend::try_filled(
                self.cells,
                usize::MAX,
                scioxide_core::quantity!("poisson_grid"),
            )?,
            candidate: scioxide_core::backend::try_filled(
                d,
                S::ZERO,
                scioxide_core::quantity!("poisson_candidate"),
            )?,
            low: scioxide_core::backend::try_filled(
                d,
                0,
                scioxide_core::quantity!("poisson_neighbor_box"),
            )?,
            high: scioxide_core::backend::try_filled(
                d,
                0,
                scioxide_core::quantity!("poisson_neighbor_box"),
            )?,
            index: scioxide_core::backend::try_filled(
                d,
                0,
                scioxide_core::quantity!("poisson_neighbor_iterator"),
            )?,
        })
    }
    /// Returns the length the output slice of [`Self::generate_into`] must have:
    /// `capacity · d`.
    pub fn output_elements(&self) -> usize {
        self.capacity * self.lower.len()
    }
    /// Returns a new design: the accepted points, row by row, and the report.
    ///
    /// The vector holds [`PoissonDiskReport::points`] rows. A stop on a work limit is not an
    /// error: it returns the valid design built so far. The function runs on the calling
    /// thread and allocates `capacity · d` elements and a workspace. On error the generator
    /// stays advanced by the words it consumed.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::generate_into`].
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn generate<R: BitGenerator>(
        &self,
        rng: &mut R,
    ) -> Result<(Vec<S>, PoissonDiskReport), SciOxideError> {
        let mut out = scioxide_core::backend::try_filled(
            self.output_elements(),
            S::ZERO,
            scioxide_core::quantity!("poisson_output"),
        )?;
        let report = self.generate_into(
            rng,
            AdvanceOnError,
            &mut out,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )?;
        out.truncate(report.points * self.lower.len());
        Ok((out, report))
    }
    /// Generates a new design and writes its accepted points to the first rows of `out`.
    ///
    /// The number of accepted points is [`PoissonDiskReport::points`]. The rows after them are
    /// not written. The design is built in `work` and copied to `out` only on success, because
    /// its points depend on each other. `state` is [`AdvanceOnError`](crate::AdvanceOnError)
    /// or [`Rollback`](crate::Rollback) and sets the generator state after an error. A stop on
    /// a work limit is not an error: it returns the valid design built so far. `cx` is checked
    /// once before the first draw and again before each candidate: a cancellation or deadline
    /// seen there ends generation with [`PoissonDiskTermination::Stopped`]. Nothing is
    /// allocated.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have [`Self::output_elements`] elements or the
    ///   sizes of `work` differ from those of [`Self::workspace`]. Nothing is drawn and `out` is
    ///   unchanged.
    /// - `ExecutionStopped` if `cx` is cancelled or past its deadline before the first draw.
    /// - `ConvergenceFailure` if drawing a direction exceeds the rejection limit.
    /// - `NumericalInstability` if a candidate cannot be represented in `S`.
    /// - Errors of `rng` and of the clock of `cx`.
    pub fn generate_into<R: BitGenerator, P: StatePolicy<R>>(
        &self,
        rng: &mut R,
        state: P,
        out: &mut [S],
        work: &mut PoissonDiskWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<PoissonDiskReport, SciOxideError> {
        // The policy is a type-level choice; its value carries no data.
        let _ = state;
        let dim = self.lower.len();
        for (expected, actual) in [
            (self.output_elements(), out.len()),
            (self.output_elements(), work.points.len()),
            (self.cells, work.grid.len()),
            (dim, work.candidate.len()),
        ] {
            if actual != expected {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(expected),
                    scioxide_core::Dimensions::one(actual),
                ));
            }
        }
        if work.active.capacity() < self.capacity {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.capacity),
                scioxide_core::Dimensions::one(work.active.capacity()),
            ));
        }
        cx.check()?;
        work.active.clear();
        work.grid.fill(usize::MAX);
        crate::transaction::run::<_, _, P>(rng, |rng| {
            let normal = crate::Normal::<f64>::standard();
            let mut report = PoissonDiskReport {
                termination: PoissonDiskTermination::BudgetExhausted,
                points: 0,
                proposals: 0,
                distance_checks: 0,
            };
            if self.options.max_proposals == 0 {
                return Ok(report);
            }
            report.proposals = 1;
            for (j, x) in work.candidate.iter_mut().enumerate() {
                let u = f64::from_u64_half_open01(rng.next_uniform_u64()?);
                *x = S::try_from_f64_lossy(
                    self.lower[j].into_f64()
                        + (self.upper[j].into_f64() - self.lower[j].into_f64()) * u,
                )?;
                // A rounded upper endpoint is valid for the closed box.
                *x = x.min(self.upper[j]);
            }
            self.accept(work, &mut report);
            'generation: while report.points < self.capacity && !work.active.is_empty() {
                if report.proposals >= self.options.max_proposals {
                    break;
                }
                // Bridson (2007): pick a random active point, try up to k
                // candidates in the annulus [r, 2r) around it, and retire the
                // point from the active list if all candidates fail.
                let slot = crate::sampling::uniform_index(work.active.len(), rng)?;
                let parent = work.active[slot];
                let mut accepted = false;
                for _ in 0..self.options.candidates_per_point {
                    if report.proposals >= self.options.max_proposals {
                        break 'generation;
                    }
                    if cx.stop_reason()?.is_some() {
                        report.termination = PoissonDiskTermination::Stopped;
                        break 'generation;
                    }
                    report.proposals += 1;
                    // Isotropic Gaussian direction and radius distributed as
                    // volume in [r,2r). Every configured attempt is an annulus
                    // candidate, including in the higher supported dimensions.
                    let mut words = crate::distribution::words::Words::new(rng, usize::MAX);
                    let norm = crate::util::rejection(|| {
                        let mut norm = 0.0_f64;
                        for x in &mut work.candidate {
                            let z = crate::Distribution::sample(&normal, &mut words)?;
                            *x = S::try_from_f64_lossy(z)?;
                            norm = RealScalar::hypot(norm, x.into_f64());
                        }
                        Ok((norm > 0.0).then_some(norm))
                    })?;
                    let u = crate::Rng::<f64>::next_float(&mut words)?;
                    let radius =
                        (1.0 + u * (2.0_f64.powi(dim as i32) - 1.0)).powf(1.0 / dim as f64);
                    for x in &mut work.candidate {
                        *x = S::try_from_f64_lossy(x.into_f64() * (radius / norm))?;
                    }
                    let mut inside = true;
                    for (j, x) in work.candidate.iter_mut().enumerate() {
                        let value =
                            work.points[parent * dim + j].into_f64() + self.radius * x.into_f64();
                        if !value.is_finite()
                            || value < self.lower[j].into_f64()
                            || value > self.upper[j].into_f64()
                        {
                            inside = false;
                            break;
                        }
                        *x = S::try_from_f64_lossy(value)?;
                    }
                    if !inside {
                        continue;
                    }
                    for j in 0..dim {
                        let x = work.candidate[j].into_f64() - self.lower[j].into_f64();
                        work.low[j] = RealScalar::floor(((x - self.radius) / self.cell).max(0.0))
                            .try_to_usize_exact()?
                            .min(self.shape[j] - 1);
                        work.high[j] = RealScalar::floor((x + self.radius) / self.cell)
                            .try_to_usize_exact()?
                            .min(self.shape[j] - 1);
                        // One-cell expansion covers boundary rounding in the grid mapping.
                        work.low[j] = work.low[j].saturating_sub(1);
                        work.high[j] = (work.high[j] + 1).min(self.shape[j] - 1);
                        work.index[j] = work.low[j];
                    }
                    // Compare either every accepted point or the grid box,
                    // whichever has fewer entries. Even empty grid visits count.
                    let cells = work
                        .low
                        .iter()
                        .zip(&work.high)
                        .fold(1usize, |n, (&lo, &hi)| n.saturating_mul(hi - lo + 1));
                    let mut separated = true;
                    if report.points <= cells {
                        for point in work.points[..report.points * dim].chunks_exact(dim) {
                            if !neighbor_work(&mut report, self.options.max_distance_checks) {
                                break 'generation;
                            }
                            if within_radius(&work.candidate, point, self.radius) {
                                separated = false;
                                break;
                            }
                        }
                    } else {
                        loop {
                            if !neighbor_work(&mut report, self.options.max_distance_checks) {
                                break 'generation;
                            }
                            let neighbor = work.grid[self.grid_index(&work.index)];
                            if neighbor != usize::MAX {
                                if !neighbor_work(&mut report, self.options.max_distance_checks) {
                                    break 'generation;
                                }
                                if within_radius(
                                    &work.candidate,
                                    &work.points[neighbor * dim..(neighbor + 1) * dim],
                                    self.radius,
                                ) {
                                    separated = false;
                                    break;
                                }
                            }
                            let mut axis = 0;
                            while axis < dim && work.index[axis] == work.high[axis] {
                                work.index[axis] = work.low[axis];
                                axis += 1;
                            }
                            if axis == dim {
                                break;
                            }
                            work.index[axis] += 1;
                        }
                    }
                    if separated {
                        // Grid occupancy is also checked: never overwrite a boundary-coincident cell.
                        self.coordinates(&work.candidate, &mut work.index);
                        if !neighbor_work(&mut report, self.options.max_distance_checks) {
                            break 'generation;
                        }
                        if work.grid[self.grid_index(&work.index)] != usize::MAX {
                            continue;
                        }
                        self.accept(work, &mut report);
                        accepted = true;
                        break;
                    }
                }
                if !accepted {
                    work.active.swap_remove(slot);
                }
            }
            if report.points == self.capacity {
                report.termination = PoissonDiskTermination::Complete;
            } else if work.active.is_empty() {
                report.termination = PoissonDiskTermination::ActiveListExhausted;
            }
            out[..report.points * dim].copy_from_slice(&work.points[..report.points * dim]);
            Ok(report)
        })
    }
    fn grid_index(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(&self.shape)
            .fold(0, |value, (&i, &shape)| value * shape + i)
    }
    fn coordinates(&self, point: &[S], index: &mut [usize]) {
        for (j, x) in index.iter_mut().enumerate() {
            *x = (((point[j].into_f64() - self.lower[j].into_f64()) / self.cell) as usize)
                .min(self.shape[j] - 1);
        }
    }
    fn accept(&self, work: &mut PoissonDiskWorkspace<S>, report: &mut PoissonDiskReport) {
        let dim = self.lower.len();
        self.coordinates(&work.candidate, &mut work.index);
        work.grid[self.grid_index(&work.index)] = report.points;
        work.points[report.points * dim..(report.points + 1) * dim]
            .copy_from_slice(&work.candidate);
        work.active.push(report.points);
        report.points += 1;
    }
}

/// Reject strata that the target scalar cannot separate; `argument` names the stratum-count
/// parameter.
fn validate_strata_count<S: RealScalar>(
    count: usize,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    // With more than 2^24 strata, adjacent f32 stratum boundaries collapse and
    // some strata would have no representable interior point.
    if S::MANTISSA_DIGITS == 24 && count > (1_usize << 24) {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "at most 2^24 strata are supported for f32",
        ));
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn represented_stratum_sample<S: RealScalar>(
    index: usize,
    count: usize,
    uniform: S,
) -> Result<S, SciOxideError> {
    // (index + u) / count is formed in f64 and then rounded to S. Rounding may
    // land on the next stratum boundary, so the value is pulled back to the
    // largest representable point below it, preserving the half-open stratum.
    let denominator = count as f64;
    let lower = S::try_from_f64_lossy(index as f64 / denominator)?;
    let upper = S::try_from_f64_lossy((index + 1) as f64 / denominator)?;
    let exact = (index as f64 + uniform.into_f64()) / denominator;
    let rounded = S::try_from_f64_lossy(exact)?;
    let value = if rounded >= upper {
        upper.next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)
    } else {
        rounded
    };
    if value.is_finite() && value >= lower && value < upper && value >= S::ZERO && value < S::ONE {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "target scalar cannot represent a distinct half-open stratum sample",
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::{SeedableRng, SplitMix64};

    #[test]
    fn test_latin_hypercube_shape() {
        let mut rng = SplitMix64::from_seed(42);
        let lhs = super::latin_hypercube::<f64, _>(10, 3, &mut rng).unwrap();
        assert_eq!(lhs.len(), 30);
        for p in lhs.chunks_exact(3) {
            assert_eq!(p.len(), 3);
        }
    }

    #[test]
    fn test_latin_hypercube_range() {
        let mut rng = SplitMix64::from_seed(42);
        let lhs = super::latin_hypercube::<f64, _>(20, 4, &mut rng).unwrap();
        for p in lhs.chunks_exact(4) {
            for &v in p {
                assert!((0.0..1.0).contains(&v), "out of range: {v}");
            }
        }
    }

    #[test]
    fn test_latin_hypercube_zero_n() {
        let mut rng = SplitMix64::from_seed(42);
        let error = super::latin_hypercube::<f64, _>(0, 3, &mut rng).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_latin_hypercube_zero_dim() {
        let mut rng = SplitMix64::from_seed(42);
        let error = super::latin_hypercube::<f64, _>(10, 0, &mut rng).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use crate::{SeedableRng, SerializableState, SplitMix64};

    #[test]
    fn lhs_output_shape_is_checked_before_consumption() {
        let mut rng = SplitMix64::from_seed(1);
        let saved = rng.export_state().unwrap();
        let mut out = [123.0_f64; 3];
        crate::error_assertions::dimensions(
            latin_hypercube_into(
                2,
                &mut rng,
                AdvanceOnError,
                LatinHypercubeOptions::default(),
                &mut out,
                &mut SamplingWorkspace::new(),
                ExecutionContext::sequential_native(),
            )
            .unwrap_err(),
            2,
            1,
        );
        assert_eq!(out, [123.0; 3]);
        assert_eq!(rng.export_state().unwrap(), saved);
    }

    #[test]
    fn poisson_output_and_each_workspace_shape_are_checked() {
        let plan = PoissonDisk::with_options(
            &[0.0_f64; 2],
            &[1.0; 2],
            0.2,
            4,
            PoissonDiskOptions::default()
                .with_candidates_per_point(30)
                .with_max_proposals(1000)
                .with_max_grid_cells(1000)
                .with_max_distance_checks(1000),
        )
        .unwrap();
        for defect in 0..6 {
            let mut work = plan.workspace().unwrap();
            let mut out = vec![123.0; plan.output_elements()];
            let (expected, actual) = match defect {
                0 => {
                    out.pop();
                    (plan.output_elements(), out.len())
                }
                1 => {
                    work.points.pop();
                    (plan.output_elements(), work.points.len())
                }
                2 => {
                    work.grid.pop();
                    (plan.cells, work.grid.len())
                }
                3 => {
                    work.candidate.pop();
                    (2, work.candidate.len())
                }
                4 => {
                    work.active = Vec::new();
                    (4, 0)
                }
                _ => {
                    out.push(123.0);
                    (plan.output_elements(), out.len())
                }
            };
            let mut rng = SplitMix64::from_seed(1);
            let saved = rng.export_state().unwrap();
            crate::error_assertions::dimensions(
                plan.generate_into(
                    &mut rng,
                    AdvanceOnError,
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err(),
                expected,
                actual,
            );
            assert!(out.iter().all(|&x| x == 123.0));
            assert_eq!(rng.export_state().unwrap(), saved);
        }
    }
}

fn neighbor_work(report: &mut PoissonDiskReport, limit: usize) -> bool {
    if report.distance_checks == limit {
        false
    } else {
        report.distance_checks += 1;
        true
    }
}
/// A scaled squared-distance filter avoids square roots and cannot overflow:
/// a single coordinate at least r already separates the points. Only the
/// rounding band around r needs the original stable hypot comparison.
fn within_radius<S: RealScalar>(a: &[S], b: &[S], radius: f64) -> bool {
    let mut square = 0.0;
    for (a, b) in a.iter().zip(b) {
        let delta = (a.into_f64() - b.into_f64()).abs();
        if delta >= radius {
            return false;
        }
        let ratio = delta / radius;
        square += ratio * ratio;
    }
    let slack = 16.0 * a.len() as f64 * f64::EPSILON;
    if square < 1.0 - slack {
        return true;
    }
    if square > 1.0 + slack {
        return false;
    }
    a.iter().zip(b).fold(0.0_f64, |norm, (a, b)| {
        RealScalar::hypot(norm, a.into_f64() - b.into_f64())
    }) < radius
}

#[cfg(test)]
mod neighbor_tests {
    use super::*;
    #[test]
    fn scaled_filter_matches_hypot_at_range_and_distance_boundaries() {
        for radius in [1e-300_f64, 1.0, 1e300] {
            for d in [1, 2, 6] {
                let zero = alloc::vec![0.0;d];
                for factor in [0.0, 0.5, 1.0 - f64::EPSILON, 1.0, 1.0 + f64::EPSILON, 2.0] {
                    let a = alloc::vec![radius*factor/(d as f64).sqrt();d];
                    let norm = a.iter().fold(0.0_f64, |n, &x| RealScalar::hypot(n, x));
                    assert_eq!(
                        within_radius(&a, &zero, radius),
                        norm < radius,
                        "radius={radius}, d={d}, factor={factor}"
                    );
                }
            }
        }
    }
}
