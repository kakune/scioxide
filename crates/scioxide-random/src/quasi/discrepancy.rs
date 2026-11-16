//! L² discrepancies with product kernels and Euclidean separation measures.
use super::plan::invalid;
use alloc::vec::Vec;
use scioxide_core::RangeAccumulator;
use scioxide_core::{RealScalar, SciOxideError};

/// L² discrepancy measured by [`discrepancy`] on points in `[0, 1]ᵈ`; smaller is more uniform.
///
/// For `n` points `xᵢ = (xᵢ₁, …, xᵢd)`, each squared discrepancy has the form
///
/// ```text
/// D² = Cᵈ − (2/n) Σᵢ Πₖ g(xᵢₖ) + (1/n²) Σᵢ Σⱼ Πₖ h(xᵢₖ, xⱼₖ),
/// ```
///
/// with the constant `C` and kernels `g` and `h` given for each variant, `a = |x − 1/2|`,
/// `b = |y − 1/2|` and `δ = |x − y|`. See [`discrepancy`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscrepancyMethod {
    /// Squared centered L² discrepancy (Hickernell 1998): `C = 13/12`,
    /// `g(x) = 1 + a/2 − a²/2`, `h(x, y) = 1 + a/2 + b/2 − δ/2`.
    ///
    /// It is invariant under reflection of any coordinate about 1/2.
    Centered,
    /// Squared wrap-around L² discrepancy (Hickernell 1998): `C = 4/3` with the sign of the
    /// first term reversed, `g = 0`, `h(x, y) = 3/2 − δ + δ²`.
    ///
    /// It is invariant under shifts of the points modulo 1.
    WrapAround,
    /// Squared mixture discrepancy (Zhou, Fang and Ning 2013): `C = 19/12`,
    /// `g(x) = 5/3 − a/4 − a²/4`, `h(x, y) = 15/8 − a/4 − b/4 − 3δ/4 + δ²/2`.
    Mixture,
    /// L² star discrepancy, not squared (Warnock 1972): the square root of the formula with
    /// `C = 1/3`, `g(x) = (1 − x²)/2`, `h(x, y) = 1 − max(x, y)`.
    ///
    /// It is the root mean square, over anchored boxes `[0, t)`, of the difference between the
    /// fraction of points in the box and its volume. It is not the maximum (star) discrepancy.
    L2Star,
}

fn validate<S: RealScalar>(points: &[S], dim: usize) -> Result<usize, SciOxideError> {
    if dim == 0 || points.is_empty() || !points.len().is_multiple_of(dim) {
        return Err(invalid(
            scioxide_core::argument!("points"),
            "nonempty complete rows with positive dimension required",
        ));
    }
    for &x in points {
        if !x.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "finite coordinates required",
            ));
        }
        if x < S::ZERO || x > S::ONE {
            return Err(invalid(
                scioxide_core::argument!("points"),
                "coordinates must lie in [0,1]",
            ));
        }
    }
    Ok(points.len() / dim)
}

// Hickernell's closed forms share the shape
//   D^2 = C^d - (2/n) sum_i prod_k g(x_ik) + (1/n^2) sum_{i,j} prod_k h(x_ik, x_jk),
// with per-coordinate kernels g (`single`) and h (`pair`) and constant C
// (`finish`). The wrap-around discrepancy has no g term (its integral against
// the uniform law is the constant), which is encoded as g = 0 and a negated
// constant.
fn single<S: RealScalar>(row: &[S], method: DiscrepancyMethod) -> Result<f64, SciOxideError> {
    row.iter()
        .map(|x| {
            let x = x.into_f64();
            let a = (x - 0.5).abs();
            match method {
                DiscrepancyMethod::Centered => 1.0 + 0.5 * a - 0.5 * a * a,
                DiscrepancyMethod::WrapAround => 0.0,
                DiscrepancyMethod::Mixture => 5.0 / 3.0 - 0.25 * a - 0.25 * a * a,
                DiscrepancyMethod::L2Star => (1.0 - x * x) * 0.5,
            }
        })
        .try_fold(1.0, scioxide_core::checked_float_product)
}
fn pair<S: RealScalar>(a: &[S], b: &[S], method: DiscrepancyMethod) -> Result<f64, SciOxideError> {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let (a, b) = (a.into_f64(), b.into_f64());
            let delta = (a - b).abs();
            match method {
                DiscrepancyMethod::Centered => {
                    1.0 + 0.5 * (a - 0.5).abs() + 0.5 * (b - 0.5).abs() - 0.5 * delta
                }
                DiscrepancyMethod::WrapAround => 1.5 - delta + delta * delta,
                DiscrepancyMethod::Mixture => {
                    1.875 - 0.25 * (a - 0.5).abs() - 0.25 * (b - 0.5).abs() - 0.75 * delta
                        + 0.5 * delta * delta
                }
                DiscrepancyMethod::L2Star => 1.0 - a.max(b),
            }
        })
        .try_fold(1.0, scioxide_core::checked_float_product)
}
fn finish(
    dim: usize,
    n: usize,
    a: f64,
    b: f64,
    method: DiscrepancyMethod,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    let base: f64 = match method {
        DiscrepancyMethod::Centered => 13.0 / 12.0,
        DiscrepancyMethod::WrapAround => 4.0 / 3.0,
        DiscrepancyMethod::Mixture => 19.0 / 12.0,
        DiscrepancyMethod::L2Star => 1.0 / 3.0,
    };
    let constant = (0..dim).try_fold(1.0, |p, _| scioxide_core::checked_float_product(p, base))?;
    let n = n as f64;
    let terms = [
        if method == DiscrepancyMethod::WrapAround {
            -constant
        } else {
            constant
        },
        -2.0 * (a / n),
        (b / n) / n,
    ];
    let mut sum = RangeAccumulator::new();
    for term in terms {
        sum.add(term)?;
    }
    // D^2 is a difference of O(C^d) terms and can come out slightly negative
    // for near-optimal designs. Negative values within a rounding envelope
    // scaled by d and the term magnitudes are clamped to zero; anything larger
    // indicates a real inconsistency.
    let value = sum.finish()?;
    let floor =
        64.0 * f64::EPSILON * (dim as f64 + 1.0) * terms.iter().map(|x| x.abs()).sum::<f64>();
    if value < -floor {
        return Err(invalid(
            argument,
            "negative discrepancy exceeds roundoff envelope",
        ));
    }
    Ok(if method == DiscrepancyMethod::L2Star {
        RealScalar::sqrt(value.max(0.0))
    } else {
        value.max(0.0)
    })
}
// Mathematical centered factors lie in [1, 1.5], so through dimension 16 a
// rounded pair product lies in [1/2, 2^10) and is an exact integer multiple of
// 2^-53 below 2^63 units. Such pair sums accumulate exactly in i128 units and
// publish as four exact 32-bit pieces, without per-pair accumulator work.
const CENTERED_UNIT_DIMENSIONS: usize = 16;

/// Product-kernel sums: exact `Σ_i g(x_i)` and ordered `Σ_{i,j} h(x_i, x_j)`.
/// Full evaluation, incremental appends and Latin-hypercube swaps add or
/// remove exact terms, and each sum rounds once in [`KernelSums::value`], so
/// every evaluator agrees bit for bit.
#[derive(Debug, Clone)]
pub(super) struct KernelSums {
    method: DiscrepancyMethod,
    singles: RangeAccumulator<f64>,
    pairs: RangeAccumulator<f64>,
}
impl KernelSums {
    fn new(method: DiscrepancyMethod) -> Self {
        Self {
            method,
            singles: RangeAccumulator::new(),
            pairs: RangeAccumulator::new(),
        }
    }
    /// Sums over every row and ordered pair of `points`.
    pub(super) fn of<S: RealScalar>(
        points: &[S],
        dim: usize,
        method: DiscrepancyMethod,
    ) -> Result<Self, SciOxideError> {
        let mut sums = Self::new(method);
        for row in points.chunks_exact(dim) {
            sums.add_single(row, 1)?;
            sums.add_pairs(points.chunks_exact(dim).map(|other| (row, other)), 1)?;
        }
        Ok(sums)
    }
    fn add_single<S: RealScalar>(&mut self, row: &[S], weight: i32) -> Result<(), SciOxideError> {
        self.singles
            .add_product(f64::from(weight), single(row, self.method)?)
    }
    // `weight` times the sum of h over ordered pairs; centered sums through
    // dimension 16 accumulate exact units.
    fn add_pairs<'a, S: RealScalar + 'a>(
        &mut self,
        pairs: impl Iterator<Item = (&'a [S], &'a [S])>,
        weight: i32,
    ) -> Result<(), SciOxideError> {
        let mut pairs = pairs.peekable();
        let small = pairs.peek().is_some_and(|(row, _)| {
            self.method == DiscrepancyMethod::Centered && row.len() <= CENTERED_UNIT_DIMENSIONS
        });
        if !small {
            for (a, b) in pairs {
                self.pairs
                    .add_product(f64::from(weight), pair(a, b, self.method)?)?;
            }
            return Ok(());
        }
        let mut units = 0_i128;
        for (a, b) in pairs {
            let bits = pair(a, b, self.method)?.to_bits();
            let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
            units += i128::from(significand << ((bits >> 52) - 1022));
        }
        let units = units * i128::from(weight);
        let sign = if units < 0 { -1.0 } else { 1.0 };
        let magnitude = units.unsigned_abs();
        for shift in [0, 32, 64, 96] {
            let piece = ((magnitude >> shift) & u128::from(u32::MAX)) as u32;
            self.pairs
                .add_scaled_product(sign * f64::from(piece), 1.0, shift - 53)?;
        }
        Ok(())
    }
    /// Add (`weight = 1`) or remove (`weight = -1`) every term of the distinct
    /// rows `i` and `j`: their singles, and each ordered pair with either row.
    pub(super) fn rows<S: RealScalar>(
        &mut self,
        points: &[S],
        dim: usize,
        (i, j): (usize, usize),
        weight: i32,
    ) -> Result<(), SciOxideError> {
        let (left, right) = (
            &points[i * dim..(i + 1) * dim],
            &points[j * dim..(j + 1) * dim],
        );
        self.add_single(left, weight)?;
        self.add_single(right, weight)?;
        self.add_pairs(
            points
                .chunks_exact(dim)
                .flat_map(|other| [(left, other), (other, left), (right, other), (other, right)]),
            weight,
        )?;
        // Each pair within {i, j} was counted twice above.
        self.add_pairs(
            [(left, left), (right, left), (left, right), (right, right)].into_iter(),
            -weight,
        )
    }
    /// Append `row` to `points`: its single, its self pair and both orders of
    /// its pairs with every existing row.
    fn append<S: RealScalar>(&mut self, points: &[S], row: &[S]) -> Result<(), SciOxideError> {
        self.add_single(row, 1)?;
        self.add_pairs(
            core::iter::once((row, row)).chain(
                points
                    .chunks_exact(row.len())
                    .flat_map(|other| [(row, other), (other, row)]),
            ),
            1,
        )
    }
    /// The rounded discrepancy of `n` rows in `dim` coordinates.
    pub(super) fn value(
        &self,
        dim: usize,
        n: usize,
        argument: scioxide_core::Argument,
    ) -> Result<f64, SciOxideError> {
        finish(
            dim,
            n,
            self.singles.finish()?,
            self.pairs.finish()?,
            self.method,
            argument,
        )
    }
}

/// Computes an L² discrepancy of `n` points in `[0, 1]ᵈ`, stored row by row in `points`.
///
/// The closed forms of [`DiscrepancyMethod`] are evaluated in `f64` in O(n²d) time and O(1)
/// extra memory. The variants other than [`DiscrepancyMethod::L2Star`] return the squared
/// discrepancy. Use the value to compare point sets of the same size and dimension.
///
/// # Accuracy
///
/// The kernel sums are accumulated exactly and rounded once, so the result has an absolute error
/// of a few ulps of `Cᵈ` and the terms. The discrepancy is a difference of terms of size `Cᵈ`,
/// so its relative error grows when it is much smaller than `Cᵈ`. A negative result within
/// `64 ε (d + 1)` times the sum of the term magnitudes is returned as 0. For the centered
/// variant, the value matches [`CenteredDiscrepancy`] and the Latin hypercube optimizer bit for
/// bit.
///
/// # Errors
///
/// - `InvalidInput` if `dim` is zero, `points` is empty or not a whole number of rows, a
///   coordinate lies outside `[0, 1]`, or the result is negative beyond rounding error.
/// - `NonFinite` if a coordinate is NaN or infinite.
/// - `NumericalInstability` if a kernel product or sum overflows.
///
/// # Examples
///
/// ```
/// use scioxide_random::{DiscrepancyMethod, discrepancy};
/// // One point at the interval center: squared centered and wrap-around values.
/// let centered = discrepancy(&[0.5_f64], 1, DiscrepancyMethod::Centered)?;
/// let wrap = discrepancy(&[0.5_f64], 1, DiscrepancyMethod::WrapAround)?;
/// assert!((centered - 1.0 / 12.0).abs() < 1e-14 && (wrap - 1.0 / 6.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - F. J. Hickernell, "A generalized discrepancy and quadrature error bound", *Math. Comp.*
///   67(221), 1998, 299–322.
/// - Y.-D. Zhou, K.-T. Fang and J.-H. Ning, "Mixture discrepancy for quasi-random point sets",
///   *J. Complexity* 29(3–4), 2013, 283–301.
/// - T. T. Warnock, "Computational investigations of low-discrepancy point sets", in
///   *Applications of Number Theory to Numerical Analysis*, Academic Press, 1972, 319–343.
pub fn discrepancy<S: RealScalar>(
    points: &[S],
    dim: usize,
    method: DiscrepancyMethod,
) -> Result<f64, SciOxideError> {
    let n = validate(points, dim)?;
    KernelSums::of(points, dim, method)?.value(dim, n, scioxide_core::argument!("points"))
}

/// Squared centered L² discrepancy of a growing point set, updated in O(nd) per added point.
///
/// The type stores the points and the kernel sums of [`DiscrepancyMethod::Centered`]. It
/// evaluates the discrepancy after a candidate point is appended in O(nd) instead of O(n²d), for
/// greedy construction of designs. Results equal [`discrepancy`] on the same points bit for bit.
/// Storage for `capacity` points is allocated at construction.
///
/// # Examples
///
/// ```
/// use scioxide_random::{CenteredDiscrepancy, DiscrepancyMethod, discrepancy};
/// let points = [0.25_f64, 0.25, 0.75, 0.75];
/// let mut state = CenteredDiscrepancy::new(&points[..2], 2, 2)?;
/// let candidate = state.evaluate_append(&points[2..])?; // no mutation
/// let committed = state.append(&points[2..])?;
/// let direct = discrepancy(&points, 2, DiscrepancyMethod::Centered)?;
/// assert!((candidate - direct).abs() < 1e-14 && (committed - direct).abs() < 1e-14);
/// assert!((state.value()? - direct).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CenteredDiscrepancy<S> {
    points: Vec<S>,
    dim: usize,
    sums: KernelSums,
    capacity: usize,
}
impl<S: RealScalar> CenteredDiscrepancy<S> {
    /// Creates the state for the initial points `points` (row by row, `dim` coordinates) with
    /// room for `capacity` points in total.
    ///
    /// The cost is O(n²d) for `n` initial points.
    ///
    /// # Errors
    ///
    /// - The errors of [`discrepancy`] for invalid points.
    /// - `InvalidInput` if `capacity` is less than the number of initial points.
    /// - `SizeOverflow` or `AllocationFailure` if storage for `capacity` points cannot be
    ///   allocated.
    pub fn new(points: &[S], dim: usize, capacity: usize) -> Result<Self, SciOxideError> {
        let n = validate(points, dim)?;
        if capacity < n {
            return Err(invalid(
                scioxide_core::argument!("capacity"),
                "capacity below sample count",
            ));
        }
        let sums = KernelSums::of(points, dim, DiscrepancyMethod::Centered)?;
        let mut stored = scioxide_core::backend::try_with_capacity(
            scioxide_core::backend::checked_mul(
                capacity,
                dim,
                scioxide_core::quantity!("discrepancy_capacity"),
            )?,
            scioxide_core::quantity!("discrepancy_points"),
        )?;
        stored.extend_from_slice(points);
        Ok(Self {
            points: stored,
            dim,
            sums,
            capacity,
        })
    }
    /// Returns the points, row by row.
    pub fn points(&self) -> &[S] {
        &self.points
    }
    /// Returns the squared centered discrepancy of the current points.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a sum overflows.
    /// - `InvalidInput` if the result is negative beyond rounding error.
    pub fn value(&self) -> Result<f64, SciOxideError> {
        self.sums.value(
            self.dim,
            self.points.len() / self.dim,
            scioxide_core::argument!("self"),
        )
    }
    // O(nd) exact update instead of O(n^2 d) recomputation; the value equals
    // the full evaluation of the appended points bit for bit.
    fn candidate(&self, row: &[S]) -> Result<(KernelSums, f64), SciOxideError> {
        if row.len() != self.dim {
            return Err(invalid(
                scioxide_core::argument!("row"),
                "wrong append dimension",
            ));
        }
        validate(row, self.dim)?;
        let mut sums = self.sums.clone();
        sums.append(&self.points, row)?;
        let value = sums.value(
            self.dim,
            self.points.len() / self.dim + 1,
            scioxide_core::argument!("row"),
        )?;
        Ok((sums, value))
    }
    /// Returns the squared centered discrepancy after appending `row`, without changing the
    /// state, in O(nd).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `row` does not have `dim` coordinates or a coordinate lies outside
    ///   `[0, 1]`.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `NumericalInstability` if a sum overflows.
    pub fn evaluate_append(&self, row: &[S]) -> Result<f64, SciOxideError> {
        Ok(self.candidate(row)?.1)
    }
    /// Appends `row` and returns the new squared centered discrepancy, in O(nd).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the state already holds `capacity` points.
    /// - The errors of [`Self::evaluate_append`].
    ///
    /// On error, the state is unchanged.
    pub fn append(&mut self, row: &[S]) -> Result<f64, SciOxideError> {
        if self.points.len() / self.dim == self.capacity {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "discrepancy append capacity exhausted",
            ));
        }
        let (sums, value) = self.candidate(row)?;
        self.points.extend_from_slice(row);
        self.sums = sums;
        Ok(value)
    }
}
/// Euclidean separation measure computed by [`geometric_discrepancy`]; larger means points are
/// more evenly spread.
///
/// See [`geometric_discrepancy`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricDiscrepancy {
    /// Smallest Euclidean distance between two distinct points, `minᵢ≠ⱼ ‖xᵢ − xⱼ‖₂` (the maximin
    /// criterion).
    MinimumDistance,
    /// Mean edge length of the Euclidean minimum spanning tree of the points.
    MeanSpanningTreeEdge,
}
/// Computes a Euclidean separation measure of `n` points in `[0, 1]ᵈ`, stored row by row in
/// `points`.
///
/// Distances are computed in `f64` with `hypot`, so they do not overflow or underflow. Duplicate
/// points count as distance 0. [`GeometricDiscrepancy::MinimumDistance`] costs O(n²d) and
/// allocates nothing. [`GeometricDiscrepancy::MeanSpanningTreeEdge`] runs Prim's algorithm in
/// O(n²d) time and allocates `n` distances.
///
/// # Errors
///
/// - `InvalidInput` if `dim` is zero, there are fewer than two points, `points` is not a whole
///   number of rows, or a coordinate lies outside `[0, 1]`.
/// - `NonFinite` if a coordinate is NaN or infinite.
/// - `AllocationFailure` if the distances of the spanning tree cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_random::{GeometricDiscrepancy, geometric_discrepancy};
/// let square = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
/// let minimum = geometric_discrepancy(&square, 2, GeometricDiscrepancy::MinimumDistance)?;
/// let mean_edge = geometric_discrepancy(&square, 2, GeometricDiscrepancy::MeanSpanningTreeEdge)?;
/// assert!((minimum - 1.0).abs() < 1e-15 && (mean_edge - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. C. Prim, "Shortest connection networks and some generalizations", *Bell Syst. Tech. J.*
/// 36(6), 1957, 1389–1401.
pub fn geometric_discrepancy<S: RealScalar>(
    points: &[S],
    dim: usize,
    method: GeometricDiscrepancy,
) -> Result<f64, SciOxideError> {
    let n = validate(points, dim)?;
    if n < 2 {
        return Err(invalid(
            scioxide_core::argument!("points"),
            "geometric discrepancy requires two points",
        ));
    }
    let distance = |i: usize, j: usize| {
        points[i * dim..(i + 1) * dim]
            .iter()
            .zip(&points[j * dim..(j + 1) * dim])
            .fold(0.0_f64, |norm, (a, b)| {
                RealScalar::hypot(norm, a.into_f64() - b.into_f64())
            })
    };
    if method == GeometricDiscrepancy::MinimumDistance {
        let mut best = f64::INFINITY;
        for (i, row) in points.chunks_exact(dim).enumerate() {
            for other in points[..i * dim].chunks_exact(dim) {
                // The norm cannot improve once any coordinate separation reaches
                // best. Check this cheap bound first; surviving pairs retain the
                // original chained hypot, including its subnormal behavior.
                if row
                    .iter()
                    .zip(other)
                    .any(|(a, b)| (a.into_f64() - b.into_f64()).abs() >= best)
                {
                    continue;
                }
                let norm = row.iter().zip(other).fold(0.0_f64, |norm, (a, b)| {
                    RealScalar::hypot(norm, a.into_f64() - b.into_f64())
                });
                best = best.min(norm);
                if best == 0.0 {
                    return Ok(best);
                }
            }
        }
        return Ok(best);
    }
    // Dense Prim's algorithm, O(n^2) time and O(n) storage: `nearest` holds
    // each vertex's distance to the growing tree, and -1 marks vertices
    // already in the tree. Vertex 0 seeds the tree with a zero-length edge,
    // so the n accepted edges sum to the n - 1 real tree edges.
    let mut nearest = scioxide_core::backend::try_filled(
        n,
        f64::INFINITY,
        scioxide_core::quantity!("spanning_tree_distances"),
    )?;
    nearest[0] = 0.0;
    let mut sum = RangeAccumulator::new();
    for _ in 0..n {
        let next = nearest
            .iter()
            .enumerate()
            .filter(|(_, x)| **x >= 0.0)
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .ok_or_else(|| invalid(scioxide_core::argument!("points"), "no remaining vertex"))?;
        sum.add(nearest[next])?;
        nearest[next] = -1.0;
        for (j, best) in nearest.iter_mut().enumerate() {
            if *best >= 0.0 {
                *best = best.min(distance(next, j));
            }
        }
    }
    Ok(sum.finish()? / (n - 1) as f64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    #[test]
    fn incremental_and_swapped_sums_equal_full_evaluation_bit_for_bit() {
        let centered = DiscrepancyMethod::Centered;
        // Dimension 2 uses exact units; dimension 17 the exact accumulator.
        for dim in [2, 17] {
            let points: Vec<f64> = (0..8 * dim)
                .map(|i| ((i * 37 + 11) % 97) as f64 / 97.0)
                .collect();
            let full = discrepancy(&points, dim, centered).unwrap();
            let mut state = CenteredDiscrepancy::new(&points[..dim], dim, 8).unwrap();
            let mut appended = 0.0;
            for row in points[dim..].chunks_exact(dim) {
                appended = state.append(row).unwrap();
            }
            assert_eq!(appended.to_bits(), full.to_bits());
            let mut sums = KernelSums::of(&points, dim, centered).unwrap();
            let mut swapped = points.clone();
            sums.rows(&swapped, dim, (1, 5), -1).unwrap();
            swapped.swap(dim + 1, 5 * dim + 1);
            sums.rows(&swapped, dim, (1, 5), 1).unwrap();
            let value = sums
                .value(dim, 8, scioxide_core::argument!("points"))
                .unwrap();
            let direct = discrepancy(&swapped, dim, centered).unwrap();
            assert_eq!(value.to_bits(), direct.to_bits());
        }
    }
    #[test]
    fn spanning_tree_of_two_points_is_their_distance() {
        assert_eq!(
            geometric_discrepancy(
                &[0.0_f64, 1.0],
                1,
                GeometricDiscrepancy::MeanSpanningTreeEdge
            )
            .unwrap(),
            1.0
        );
    }
}
