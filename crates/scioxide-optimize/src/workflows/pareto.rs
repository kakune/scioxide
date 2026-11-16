//! Pareto dominance, crowding distance and bounded archives of nondominated points.
use super::*;
use alloc::vec::Vec;

use scioxide_core::{
    FeasibilityTolerance,
    backend::{checked_mul, try_filled},
};

/// States the Pareto dominance relation between two objective vectors, for
/// minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParetoDominance {
    /// The first vector is no worse in every component and better in at least
    /// one.
    Dominates,
    /// The second vector dominates the first.
    Dominated,
    /// All components are equal; `−0` and `+0` compare equal.
    Equal,
    /// Each vector is better than the other in at least one component.
    Incomparable,
}
/// Returns the Pareto dominance relation of `a` to `b` for minimization.
///
/// `a` dominates `b` if `aᵢ ≤ bᵢ` for every `i` and `aᵢ < bᵢ` for some `i`.
/// Components are compared exactly, without a tolerance.
///
/// # Errors
///
/// Returns `InvalidInput` if the vectors are empty, have different lengths,
/// or contain NaN or infinity.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{ParetoDominance, pareto_dominance};
///
/// assert_eq!(pareto_dominance(&[1.0_f64, 2.0], &[1.0, 3.0])?, ParetoDominance::Dominates);
/// assert_eq!(pareto_dominance(&[1.0_f64, 3.0], &[2.0, 1.0])?, ParetoDominance::Incomparable);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn pareto_dominance<R: RealScalar>(a: &[R], b: &[R]) -> Result<ParetoDominance, SciOxideError> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err(invalid(
            scioxide_core::argument!("b"),
            "dominance requires equally sized finite nonempty vectors",
        ));
    }
    Ok(relation(a, b))
}
fn relation<R: RealScalar>(a: &[R], b: &[R]) -> ParetoDominance {
    let better = a.iter().zip(b).any(|(x, y)| x < y);
    let worse = a.iter().zip(b).any(|(x, y)| x > y);
    match (better, worse) {
        (true, false) => ParetoDominance::Dominates,
        (false, true) => ParetoDominance::Dominated,
        (false, false) => ParetoDominance::Equal,
        _ => ParetoDominance::Incomparable,
    }
}
/// Borrows one point of a population or archive with its objectives and
/// constraint values.
#[derive(Debug, Clone, Copy)]
pub struct ParetoEntry<'a, R: RealScalar> {
    pub(super) point: &'a [R],
    pub(super) values: &'a [R],
    pub(super) errors: &'a [R],
    pub(super) variances: &'a [R],
    pub(super) constraints: &'a [R],
    pub(super) violation: R,
    pub(super) id: u64,
    pub(super) samples: usize,
    pub(super) uncertainty: bool,
}
impl<'a, R: RealScalar> ParetoEntry<'a, R> {
    /// Returns the decision variables `x`.
    pub fn point(self) -> &'a [R] {
        self.point
    }
    /// Returns the objective values, or their sample means for noisy
    /// objectives.
    pub fn objectives(self) -> &'a [R] {
        self.values
    }
    /// Returns the standard errors of the objective means, or `None` for a
    /// single observation.
    pub fn standard_errors(self) -> Option<&'a [R]> {
        self.uncertainty.then_some(self.errors)
    }
    /// Returns the sample variances of the objectives, or `None` for a single
    /// observation. Deterministic objectives have zero variance.
    pub fn sample_variances(self) -> Option<&'a [R]> {
        self.uncertainty.then_some(self.variances)
    }
    /// Returns the constraint values: equalities `h(x)` first, then
    /// inequalities `g(x)`, in the order they were defined.
    pub fn constraints(self) -> &'a [R] {
        self.constraints
    }
    /// Returns the total constraint violation
    /// `Σ max(|hᵢ| − τ, 0) + Σ max(gⱼ − τ, 0)`, where `τ` is the feasibility
    /// threshold.
    pub fn violation(self) -> R {
        self.violation
    }
    /// Returns whether the total constraint violation is zero.
    pub fn is_feasible(self) -> bool {
        self.violation == R::ZERO
    }
    /// Returns the insertion number of the point, used to break ties.
    pub fn id(self) -> u64 {
        self.id
    }
    /// Returns the number of objective observations averaged in the values.
    pub fn replicates(self) -> usize {
        self.samples
    }
}

#[derive(Debug, Clone)]
pub(super) struct Population<R: RealScalar> {
    pub n: usize,
    pub m: usize,
    pub c: usize,
    pub len: usize,
    pub points: Vec<R>,
    pub values: Vec<R>,
    pub errors: Vec<R>,
    pub variances: Vec<R>,
    pub constraints: Vec<R>,
    pub violations: Vec<R>,
    pub ids: Vec<u64>,
    pub samples: Vec<usize>,
    pub uncertainty: Vec<bool>,
}
impl<R: RealScalar> Population<R> {
    pub(super) fn new(
        n: usize,
        m: usize,
        c: usize,
        capacity: usize,
    ) -> Result<Self, SciOxideError> {
        if n == 0 || m == 0 || capacity == 0 {
            return Err(invalid(
                scioxide_core::argument!("objectives"),
                "population dimensions and capacity must be positive",
            ));
        }
        let field = |w| {
            try_filled(
                checked_mul(w, capacity, scioxide_core::quantity!("population_field"))?,
                R::ZERO,
                scioxide_core::quantity!("population_field"),
            )
        };
        Ok(Self {
            n,
            m,
            c,
            len: 0,
            points: field(n)?,
            values: field(m)?,
            errors: field(m)?,
            variances: field(m)?,
            constraints: field(c)?,
            violations: field(1)?,
            ids: try_filled(capacity, 0, scioxide_core::quantity!("population.ids"))?,
            samples: try_filled(capacity, 0, scioxide_core::quantity!("population.samples"))?,
            uncertainty: try_filled(
                capacity,
                false,
                scioxide_core::quantity!("population.uncertainty"),
            )?,
        })
    }
    pub(super) fn capacity(&self) -> usize {
        self.ids.len()
    }
    pub(super) fn entry(&self, i: usize) -> ParetoEntry<'_, R> {
        ParetoEntry {
            point: &self.points[i * self.n..(i + 1) * self.n],
            values: &self.values[i * self.m..(i + 1) * self.m],
            errors: &self.errors[i * self.m..(i + 1) * self.m],
            variances: &self.variances[i * self.m..(i + 1) * self.m],
            constraints: &self.constraints[i * self.c..(i + 1) * self.c],
            violation: self.violations[i],
            id: self.ids[i],
            samples: self.samples[i],
            uncertainty: self.uncertainty[i],
        }
    }
    pub(super) fn put(&mut self, i: usize, e: ParetoEntry<'_, R>) {
        self.points[i * self.n..(i + 1) * self.n].copy_from_slice(e.point);
        self.values[i * self.m..(i + 1) * self.m].copy_from_slice(e.values);
        self.errors[i * self.m..(i + 1) * self.m].copy_from_slice(e.errors);
        self.variances[i * self.m..(i + 1) * self.m].copy_from_slice(e.variances);
        self.constraints[i * self.c..(i + 1) * self.c].copy_from_slice(e.constraints);
        self.violations[i] = e.violation;
        self.ids[i] = e.id;
        self.samples[i] = e.samples;
        self.uncertainty[i] = e.uncertainty;
    }
    pub(super) fn copy_from(&mut self, other: &Self) {
        for i in 0..other.len {
            self.put(i, other.entry(i));
        }
        self.len = other.len;
    }
    /// Fallible copy of the shape and live entries.
    pub(super) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let mut copy = Self::new(self.n, self.m, self.c, self.capacity())?;
        copy.copy_from(self);
        Ok(copy)
    }
    /// Integrity token over the shape and live entries; `Debug` prints each
    /// float in its shortest exactly round-tripping form.
    pub(super) fn state_token(&self) -> u128 {
        let live = |width: usize| self.len * width;
        crate::context::options_token(&(
            (self.n, self.m, self.c, self.len, self.capacity()),
            &self.points[..live(self.n)],
            &self.values[..live(self.m)],
            &self.errors[..live(self.m)],
            &self.variances[..live(self.m)],
            &self.constraints[..live(self.c)],
            &self.violations[..self.len],
            &self.ids[..self.len],
            &self.samples[..self.len],
            &self.uncertainty[..self.len],
        ))
    }
}
pub(super) fn violation<R: RealScalar>(
    values: &[R],
    equalities: usize,
    tolerance: FeasibilityTolerance<R>,
) -> Result<R, SciOxideError> {
    if equalities > values.len() {
        return Err(invalid(
            scioxide_core::argument!("constraints"),
            "equality count exceeds constraint values",
        ));
    }
    let threshold = tolerance.threshold(R::ONE)?;
    values.iter().enumerate().try_fold(R::ZERO, |sum, (i, &v)| {
        if !v.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("constraints"),
                "non-finite constraint value",
            ));
        }
        finite(sum + ((if i < equalities { v.abs() } else { v }) - threshold).max(R::ZERO))
    })
}
pub(super) fn constrained_dominates<R: RealScalar>(
    a: ParetoEntry<'_, R>,
    b: ParetoEntry<'_, R>,
) -> bool {
    if a.violation != b.violation {
        a.violation < b.violation
    } else if a.violation > R::ZERO {
        false
    } else {
        relation(a.values, b.values) == ParetoDominance::Dominates
    }
}
// Indices are scratch; distances are addressed by population index.
pub(super) fn crowding<R: RealScalar>(
    pop: &Population<R>,
    indices: &mut [usize],
    distance: &mut [R],
) -> Result<(), SciOxideError> {
    for &i in indices.iter() {
        distance[i] = R::ZERO;
    }
    for j in 0..pop.m {
        indices.sort_unstable_by(|&a, &b| {
            pop.values[a * pop.m + j]
                .partial_cmp(&pop.values[b * pop.m + j])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(pop.ids[a].cmp(&pop.ids[b]))
        });
        let (Some(&first), Some(&last)) = (indices.first(), indices.last()) else {
            continue;
        };
        let lo = pop.values[first * pop.m + j];
        let hi = pop.values[last * pop.m + j];
        if lo == hi {
            continue;
        }
        distance[first] = R::INFINITY;
        distance[last] = R::INFINITY;
        let scale = lo.abs().max(hi.abs());
        let span = hi / scale - lo / scale;
        if span <= R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "unrepresentable crowding span",
            ));
        }
        for group in indices.windows(3) {
            let [a, b, c] = [group[0], group[1], group[2]];
            if distance[b].is_finite() {
                distance[b] = finite(
                    distance[b]
                        + (pop.values[c * pop.m + j] / scale - pop.values[a * pop.m + j] / scale)
                            / span,
                )?;
            }
        }
    }
    Ok(())
}
/// Keeps at most `capacity` feasible, mutually nondominated points.
///
/// An inserted point is kept if it is feasible and no stored point dominates
/// or equals it; stored points that it dominates are removed. When the archive
/// would exceed its capacity, the point with the smallest crowding distance is
/// removed, so the archive keeps a spread-out approximation of the Pareto
/// front. Constraints are equalities `h(x) = 0` and inequalities `g(x) ≤ 0`,
/// each satisfied within the feasibility tolerance. Construction reserves
/// `O(capacity × (dimension + objectives + constraints))` storage; insertion
/// does not allocate.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::ParetoArchive;
///
/// // Three mutually nondominated points; (3, 3) is dominated by (2, 2).
/// let mut archive = ParetoArchive::<f64>::new(1, 2, 8)?;
/// for (x, f) in [
///     (0.0, [1.0, 3.0]),
///     (1.0, [2.0, 2.0]),
///     (2.0, [3.0, 1.0]),
///     (3.0, [3.0, 3.0]),
/// ] {
///     archive.insert(&[x], &f, &[])?;
/// }
/// assert_eq!(archive.len(), 3);
/// // Area dominated up to the reference point (4, 4): 1 + 2 + 3 = 6.
/// assert!((archive.hypervolume_2d([4.0, 4.0])? - 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
/// multiobjective genetic algorithm: NSGA-II", *IEEE Trans. Evol. Comput.*
/// 6(2), 182–197, 2002. <https://doi.org/10.1109/4235.996017>
#[derive(Debug, Clone)]
pub struct ParetoArchive<R: RealScalar> {
    pub(super) data: Population<R>,
    scratch: Population<R>,
    indices: Vec<usize>,
    distance: Vec<R>,
    published_distance: Vec<R>,
    pub(super) equalities: usize,
    pub(super) tolerance: FeasibilityTolerance<R>,
    pruned: usize,
    next_id: u64,
}
impl<R: RealScalar> ParetoArchive<R> {
    /// Creates an empty archive for points with `dimension` variables and
    /// `objectives` objectives, without constraints.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension`, `objectives` or `capacity` is zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved.
    pub fn new(
        dimension: usize,
        objectives: usize,
        capacity: usize,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            dimension,
            objectives,
            0,
            0,
            capacity,
            FeasibilityTolerance::default(),
        )
    }
    /// Returns an empty archive that stores `equalities` equality and
    /// `inequalities` inequality constraint values per point, judged feasible
    /// with `tolerance`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the archive is not empty.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved.
    pub fn with_constraints(
        self,
        equalities: usize,
        inequalities: usize,
        tolerance: FeasibilityTolerance<R>,
    ) -> Result<Self, SciOxideError> {
        if !self.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "cannot change constraints of a populated archive",
            ));
        }
        Self::prepare(
            self.data.n,
            self.data.m,
            equalities.checked_add(inequalities).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("constraint_rows"))
            })?,
            equalities,
            self.capacity(),
            tolerance,
        )
    }
    pub(super) fn prepare(
        n: usize,
        m: usize,
        c: usize,
        equalities: usize,
        capacity: usize,
        tolerance: FeasibilityTolerance<R>,
    ) -> Result<Self, SciOxideError> {
        if capacity == 0 || equalities > c {
            return Err(invalid(
                scioxide_core::argument!("capacity"),
                "archive capacity or constraints",
            ));
        }
        let size = capacity.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("archive_slots"))
        })?;
        Ok(Self {
            data: Population::new(n, m, c, size)?,
            scratch: Population::new(n, m, c, size)?,
            indices: try_filled(size, 0, scioxide_core::quantity!("archive.indices"))?,
            distance: try_filled(size, R::ZERO, scioxide_core::quantity!("archive.distance"))?,
            published_distance: try_filled(
                size,
                R::ZERO,
                scioxide_core::quantity!("archive.published_distance"),
            )?,
            equalities,
            tolerance,
            pruned: 0,
            next_id: 0,
        })
    }
    /// Returns the maximum number of stored points.
    pub fn capacity(&self) -> usize {
        self.data.capacity() - 1
    }
    /// Returns the number of stored points.
    pub fn len(&self) -> usize {
        self.data.len
    }
    /// Returns whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the number of nondominated points removed because the archive
    /// was full.
    pub const fn pruned(&self) -> usize {
        self.pruned
    }
    /// Returns the crowding distance of each stored point, in the order of
    /// [`Self::entries`].
    ///
    /// Each objective contributes the gap between the two neighboring values,
    /// divided by the range of that objective. Points at either end of an
    /// objective's range have infinite distance; an objective with zero range
    /// contributes zero.
    pub fn crowding_distances(&self) -> &[R] {
        &self.published_distance[..self.len()]
    }
    /// Returns the stored points in insertion order.
    pub fn entries(&self) -> impl Iterator<Item = ParetoEntry<'_, R>> {
        (0..self.len()).map(|i| self.data.entry(i))
    }
    /// Inserts a point with its objective and constraint values, and returns
    /// whether it is kept.
    ///
    /// `constraints` holds the equality values first, then the inequality
    /// values. The point is not kept if it is infeasible, if a stored point
    /// dominates or equals it, or if it is the one removed for capacity.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a length does not match the archive, or a value is
    ///   NaN or infinite.
    /// - `SizeOverflow` if the insertion counter overflows.
    ///
    /// On error, the archive is unchanged.
    pub fn insert(
        &mut self,
        point: &[R],
        objectives: &[R],
        constraints: &[R],
    ) -> Result<bool, SciOxideError> {
        if point.len() != self.data.n
            || objectives.len() != self.data.m
            || constraints.len() != self.data.c
            || point.iter().chain(objectives).any(|x| !x.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("point"),
                "archive observation shape or non-finite value",
            ));
        }
        let v = violation(constraints, self.equalities, self.tolerance)?;
        let id = self.next_id;
        let next = id.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("archive.next_id"))
        })?;
        // Stage directly in the spare slot, with no temporary per-insertion allocation.
        let i = self.data.len;
        self.data.points[i * self.data.n..(i + 1) * self.data.n].copy_from_slice(point);
        self.data.values[i * self.data.m..(i + 1) * self.data.m].copy_from_slice(objectives);
        self.data.constraints[i * self.data.c..(i + 1) * self.data.c].copy_from_slice(constraints);
        self.data.errors[i * self.data.m..(i + 1) * self.data.m].fill(R::ZERO);
        self.data.variances[i * self.data.m..(i + 1) * self.data.m].fill(R::ZERO);
        self.data.violations[i] = v;
        self.data.ids[i] = id;
        self.data.samples[i] = 1;
        self.data.uncertainty[i] = true;
        let retained = self.commit_spare()?;
        self.next_id = next;
        Ok(retained)
    }
    pub(super) fn insert_estimate(
        &mut self,
        entry: ParetoEntry<'_, R>,
    ) -> Result<bool, SciOxideError> {
        self.data.put(self.data.len, entry);
        self.commit_spare()
    }
    fn commit_spare(&mut self) -> Result<bool, SciOxideError> {
        let candidate = self.data.len;
        let e = self.data.entry(candidate);
        if e.violation > R::ZERO {
            return Ok(false);
        }
        let mut count = 0;
        for i in 0..candidate {
            match relation(self.data.entry(i).values, e.values) {
                ParetoDominance::Dominates | ParetoDominance::Equal => return Ok(false),
                ParetoDominance::Dominated => {}
                ParetoDominance::Incomparable => {
                    self.indices[count] = i;
                    count += 1;
                }
            }
        }
        self.indices[count] = candidate;
        count += 1;
        let mut dropped = None;
        if count > self.capacity() {
            crowding(&self.data, &mut self.indices[..count], &mut self.distance)?;
            dropped = self.indices[..count].iter().copied().min_by(|&a, &b| {
                self.distance[a]
                    .partial_cmp(&self.distance[b])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(self.data.ids[b].cmp(&self.data.ids[a]))
            });
        }
        let pruned = self
            .pruned
            .checked_add(usize::from(dropped.is_some()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("archive.pruned"))
            })?;
        self.indices[..count].sort_unstable_by_key(|&i| self.data.ids[i]);
        self.scratch.len = 0;
        for &i in &self.indices[..count] {
            if Some(i) != dropped {
                self.scratch.put(self.scratch.len, self.data.entry(i));
                self.scratch.len += 1;
            }
        }
        for (i, index) in self.indices[..self.scratch.len].iter_mut().enumerate() {
            *index = i;
        }
        crowding(
            &self.scratch,
            &mut self.indices[..self.scratch.len],
            &mut self.distance,
        )?;
        self.published_distance[..self.scratch.len]
            .copy_from_slice(&self.distance[..self.scratch.len]);
        core::mem::swap(&mut self.data, &mut self.scratch);
        self.pruned = pruned;
        Ok(dropped != Some(candidate))
    }
    /// Checks that every stored point is feasible and that no two stored points
    /// dominate or equal each other.
    ///
    /// Costs `O(len² × objectives + len × constraints)`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if a stored constraint value is not finite.
    pub fn verify(&self) -> Result<bool, SciOxideError> {
        for (i, a) in self.entries().enumerate() {
            if violation(a.constraints, self.equalities, self.tolerance)? != R::ZERO {
                return Ok(false);
            }
            for b in self.entries().skip(i + 1) {
                if relation(a.values, b.values) != ParetoDominance::Incomparable {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    /// Returns the hypervolume of a two-objective archive: the area dominated
    /// by the stored points and bounded by `reference`.
    ///
    /// `reference` must be no better than every stored point in both
    /// objectives. Costs `O(len²)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the archive does not have exactly two objectives, if
    ///   `reference` is not finite, or if a stored point is worse than
    ///   `reference` in some objective.
    /// - `NumericalInstability` if the area overflows.
    pub fn hypervolume_2d(&self, reference: [R; 2]) -> Result<R, SciOxideError> {
        if self.data.m != 2 || reference.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("reference"),
                "hypervolume requires two objectives and a finite reference",
            ));
        }
        if self
            .entries()
            .any(|e| e.values[0] > reference[0] || e.values[1] > reference[1])
        {
            return Err(invalid(
                scioxide_core::argument!("reference"),
                "reference must be no better than every archive objective",
            ));
        }
        let mut total = scioxide_core::RangeAccumulator::<R>::new();
        for e in self.entries() {
            let right = self
                .entries()
                .filter(|other| other.values[0] > e.values[0])
                .map(|other| other.values[0])
                .fold(reference[0], R::min);
            // Expand each rectangle into paired products so an overflowing width
            // times a tiny (or zero) height still rounds the finite complete area.
            total.add_product(right, reference[1])?;
            total.subtract_product(e.values[0], reference[1])?;
            total.subtract_product(right, e.values[1])?;
            total.add_product(e.values[0], e.values[1])?;
        }
        total.finish()
    }
    pub(super) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let mut copy = Self::prepare(
            self.data.n,
            self.data.m,
            self.data.c,
            self.equalities,
            self.capacity(),
            self.tolerance,
        )?;
        copy.copy_from(self);
        Ok(copy)
    }
    pub(super) fn state_token(&self) -> u128 {
        crate::context::options_token(&(
            self.data.state_token(),
            &self.published_distance[..self.len()],
            (self.equalities, self.tolerance, self.pruned, self.next_id),
        ))
    }
    pub(super) fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
        self.published_distance[..other.len()].copy_from_slice(other.crowding_distances());
        self.pruned = other.pruned;
        self.next_id = other.next_id;
    }
    pub(super) fn reset(&mut self) {
        self.data.len = 0;
        self.pruned = 0;
        self.next_id = 0;
    }
}
