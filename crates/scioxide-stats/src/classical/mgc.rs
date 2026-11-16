//! Multiscale graph correlation (MGC) test of independence.
use super::*;
use crate::{
    PermutationKind, PermutationMode, ResamplingInference, ResamplingMethod, ResamplingOptions,
    ResamplingPlan, ResamplingProblem, ResamplingResult, ResamplingWorkspace,
};
use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_random::{Beta, Quantile};

#[derive(Clone, Debug)]
struct Geometry {
    centered: Vec<f64>,
    ranks: Vec<usize>,
    variance: Vec<f64>,
    scales: usize,
}

/// Prepared multiscale graph correlation (MGC) test of independence between
/// two paired samples.
///
/// MGC computes local distance correlations restricted to the `k` nearest
/// neighbors in `x` and the `l` nearest in `y`, for all neighborhood sizes
/// `(k, l)`. It reports a smoothed maximum of this local correlation map.
/// The p-value comes from permutations of the `y` rows (upper tail), through
/// [`MgcPlan::permutation_plan`] and [`MgcPlan::test`].
///
/// Preparation computes and ranks the distance matrices: O(n² d + n² log n)
/// time and O(n²) memory for `n` observations in `d` dimensions. Each
/// evaluation of the statistic costs O(n²) time and uses the scratch from
/// [`MgcPlan::workspace`]. Computation is in `f64`. Of the
/// [`ClassicalOptions`], only [`ClassicalOptions::max_workspace_bytes`] is
/// used, as the byte limit of [`MgcPlan::workspace`].
///
/// # References
///
/// C. Shen, C. E. Priebe and J. T. Vogelstein, "From distance correlation to
/// multiscale graph correlation", *J. Amer. Statist. Assoc.* 115 (2020),
/// 280–291.
#[derive(Clone, Debug)]
pub struct MgcPlan {
    n: usize,
    x: Geometry,
    y: Geometry,
    threshold: f64,
    groups: Option<[usize; 2]>,
    max_workspace_bytes: usize,
}

/// MGC statistic and the neighborhood sizes at which it was attained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MgcStatistic {
    /// Smoothed local distance correlation, in `[−1, 1]`.
    pub correlation: f64,
    /// Selected neighborhood sizes `(k, l)` in `x` and `y`, counted from one.
    pub scale: [usize; 2],
}

fn cells(a: usize, b: usize) -> Result<usize, SciOxideError> {
    a.checked_mul(b)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))
}

fn geometry<S: RealScalar>(distance: MatrixView<'_, S>) -> Result<Geometry, SciOxideError> {
    let n = distance.rows();
    let size = cells(n, n)?;
    let mut values = try_filled(
        size,
        0.0,
        scioxide_core::quantity!("mgc_centered_distances"),
    )?;
    let mut max = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = finite(distance.get(i, j)?)?.into_f64();
            if v < 0.0 || (i == j && v != 0.0) {
                return Err(invalid(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "distance matrix needs nonnegative entries and a zero diagonal",
                ));
            }
            let other = finite(distance.get(j, i)?)?.into_f64();
            if (v - other).abs() > 32.0 * S::EPSILON.into_f64() * v.max(other) {
                return Err(invalid(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "distance matrix is not symmetric",
                ));
            }
            let value = v + (other - v) / 2.0;
            max = max.max(value);
            values[i * n + j] = value;
        }
    }
    if max > 0.0 {
        for value in &mut values {
            let normalized = *value / max;
            if *value != 0.0 && normalized == 0.0 {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "distance scale separation exceeds f64 normalization range",
                ));
            }
            *value = normalized;
        }
    }
    let mut rank = try_filled(size, 0, scioxide_core::quantity!("mgc_distance_ranks"))?;
    let mut column = try_filled(n, 0.0, scioxide_core::quantity!("mgc_column"))?;
    let mut scales = 1;
    for j in 0..n {
        for (i, value) in column.iter_mut().enumerate() {
            *value = values[i * n + j];
        }
        let ranks = ranks::ranks(&column, crate::RankMethod::Dense)?;
        let mean = column.iter().sum::<f64>() / (n - 1) as f64;
        for (i, &r) in ranks.iter().enumerate() {
            let r = r as usize;
            rank[i * n + j] = r - 1;
            scales = scales.max(r);
            values[i * n + j] = if i == j { 0.0 } else { column[i] - mean };
        }
    }
    let mut geometry = Geometry {
        centered: values,
        ranks: rank,
        variance: try_filled(scales, 0.0, scioxide_core::quantity!("mgc_local_variance"))?,
        scales,
    };
    let len = cells(scales, scales)?
        .checked_add(2 * scales)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let mut scratch = try_filled(
        len,
        0.0,
        scioxide_core::quantity!("mgc_variance_preparation"),
    )?;
    covariance(n, &geometry, &geometry, None, &mut scratch)?;
    for (i, v) in geometry.variance.iter_mut().enumerate() {
        *v = scratch[i * scales + i];
    }
    Ok(geometry)
}

fn covariance(
    n: usize,
    x: &Geometry,
    y: &Geometry,
    permutation: Option<&[usize]>,
    scratch: &mut [f64],
) -> Result<(), SciOxideError> {
    let size = cells(x.scales, y.scales)?;
    if scratch.len() < size + x.scales + y.scales {
        return Err(invalid(
            scioxide_core::argument!("scratch"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "short MGC covariance scratch",
        ));
    }
    let (map, rest) = scratch.split_at_mut(size);
    let (ex, rest) = rest.split_at_mut(x.scales);
    let ey = &mut rest[..y.scales];
    map.fill(0.0);
    ex.fill(0.0);
    ey.fill(0.0);
    for i in 0..n {
        for j in 0..n {
            let yi = permutation.map_or(i, |p| p[i]);
            let yj = permutation.map_or(j, |p| p[j]);
            let a = x.centered[i * n + j];
            let b = y.centered[yj * n + yi];
            let k = x.ranks[i * n + j];
            let l = y.ranks[yj * n + yi];
            map[k * y.scales + l] += a * b;
            ex[k] += a;
            ey[l] += b;
        }
    }
    for i in 1..ex.len() {
        ex[i] += ex[i - 1];
    }
    for j in 1..ey.len() {
        ey[j] += ey[j - 1];
    }
    for i in 0..x.scales {
        for j in 0..y.scales {
            let index = i * y.scales + j;
            if i > 0 {
                map[index] += map[index - y.scales];
            }
            if j > 0 {
                map[index] += map[index - 1];
            }
            if i > 0 && j > 0 {
                map[index] -= map[index - y.scales - 1];
            }
        }
    }
    for (i, row) in map.chunks_exact_mut(y.scales).enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v -= ex[i] * ey[j] / (n as f64 * n as f64);
        }
    }
    Ok(())
}

fn distances<S: RealScalar>(x: MatrixView<'_, S>) -> Result<Matrix<f64>, SciOxideError> {
    let n = x.rows();
    let d = x.cols();
    if d == 0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "MGC sample needs at least one coordinate",
        ));
    }
    let mut scale = 0.0_f64;
    for i in 0..n {
        for j in 0..d {
            scale = scale.max(finite(x.get(i, j)?)?.into_f64().abs());
        }
    }
    let mut matrix = Matrix::zeros(n, n)?;
    if scale == 0.0 {
        return Ok(matrix);
    }
    let denominator = scioxide_core::Scaled::<f64>::from_value(scale)?;
    for i in 0..n {
        for j in 0..i {
            let norm = (0..d).try_fold(0.0, |sum, k| -> Result<f64, SciOxideError> {
                let delta = scioxide_core::Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(x.get(i, k)?),
                )?
                .sub(scioxide_core::Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(x.get(j, k)?),
                )?)?
                .div(denominator)?
                .round_to::<f64>()?;
                Ok(sum + delta * delta)
            })?;
            let value = finite(norm.sqrt())?;
            matrix.view_mut().set(i, j, value)?;
            matrix.view_mut().set(j, i, value)?;
        }
    }
    Ok(matrix)
}

impl MgcPlan {
    /// Prepares the test from paired samples using Euclidean distances.
    ///
    /// Row `i` of `x` and row `i` of `y` form one observation. The distances are
    /// rescaled by their maximum, which leaves MGC unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` and `y` have different numbers of rows, fewer
    ///   than five rows, or no columns.
    /// - An error if `x` or `y` contains NaN or infinity.
    /// - The errors of [`Self::from_distances`].
    pub fn from_samples<S: RealScalar>(
        x: MatrixView<'_, S>,
        y: MatrixView<'_, S>,
        options: ClassicalOptions,
    ) -> Result<Self, SciOxideError> {
        if x.rows() < 5 || x.rows() != y.rows() {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "MGC requires at least five paired rows",
            ));
        }
        let dx = distances(x)?;
        let dy = distances(y)?;
        Self::from_distances(dx.view(), dy.view(), options)
    }
    /// Prepares the test from precomputed distance matrices.
    ///
    /// `x` and `y` are `n × n` distance matrices of the same observations. They
    /// must be symmetric to within `32 ε` relative, non-negative, and zero on
    /// the diagonal.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the matrices are not square of the same order, or
    ///   `n < 5`.
    /// - `InvalidInput` if a matrix has a negative entry, a nonzero diagonal
    ///   entry or is not symmetric.
    /// - An error if a matrix contains NaN or infinity.
    /// - `NumericalInstability` (`Underflow`) if a nonzero distance underflows
    ///   to zero after division by the largest distance.
    pub fn from_distances<S: RealScalar>(
        x: MatrixView<'_, S>,
        y: MatrixView<'_, S>,
        options: ClassicalOptions,
    ) -> Result<Self, SciOxideError> {
        let n = x.rows();
        if n < 5 || x.cols() != n || y.rows() != n || y.cols() != n {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "MGC needs equal square distance matrices of order at least five",
            ));
        }
        let a = (n - 1) as f64;
        let beta_shape = a * (a - 3.0) / 4.0 - 0.5;
        let threshold = 1.0 - 2.0 * Beta::new(beta_shape, beta_shape)?.quantile(0.02 / a)?;
        Ok(Self {
            n,
            x: geometry(x)?,
            y: geometry(y)?,
            threshold,
            groups: None,
            max_workspace_bytes: options.max_workspace_bytes(),
        })
    }
    /// Prepares a two-sample MGC test of equal distributions.
    ///
    /// The rows of `x` and `y` are pooled and tested for independence from the
    /// 0/1 group label. The matching permutation plan reassigns labels between
    /// the pooled observations with the group sizes fixed.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` and `y` have different numbers of columns, either
    ///   is empty, or they have fewer than five rows together.
    /// - The errors of [`Self::from_samples`].
    pub fn two_sample<S: RealScalar>(
        x: MatrixView<'_, S>,
        y: MatrixView<'_, S>,
        options: ClassicalOptions,
    ) -> Result<Self, SciOxideError> {
        if x.cols() != y.cols() || x.rows() == 0 || y.rows() == 0 {
            return Err(invalid(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "MGC two-sample shape mismatch",
            ));
        }
        let n = x.rows().checked_add(y.rows()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        let mut joined = Matrix::zeros(n, x.cols())?;
        let mut labels = Matrix::zeros(n, 1)?;
        for i in 0..n {
            let (source, row, label) = if i < x.rows() {
                (x, i, S::ZERO)
            } else {
                (y, i - x.rows(), S::ONE)
            };
            for j in 0..x.cols() {
                joined.view_mut().set(i, j, source.get(row, j)?)?;
            }
            labels.view_mut().set(i, 0, label)?;
        }
        let mut plan = Self::from_samples(labels.view(), joined.view(), options)?;
        plan.groups = Some([x.rows(), y.rows()]);
        Ok(plan)
    }
    /// Returns the number of observations `n`.
    pub const fn observations(&self) -> usize {
        self.n
    }
    /// Returns the dimensions of the local correlation map: the numbers of
    /// distinct neighborhood sizes in `x` and `y`. Tied distances share a size.
    pub fn map_shape(&self) -> [usize; 2] {
        [self.x.scales, self.y.scales]
    }
    /// Returns the number of `f64` scratch entries [`Self::statistic`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the count overflows `usize`.
    pub fn scratch_len(&self) -> Result<usize, SciOxideError> {
        cells(self.x.scales, self.y.scales)?
            .checked_mul(3)
            .and_then(|v| v.checked_add(self.x.scales + self.y.scales))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
    }
    /// Allocates scratch for [`Self::statistic`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the scratch exceeds the
    ///   [`ClassicalOptions::max_workspace_bytes`] given at preparation.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace(&self) -> Result<Vec<f64>, SciOxideError> {
        let len = self.scratch_len()?;
        crate::error_helpers::limit(
            cells(len, core::mem::size_of::<f64>())?,
            self.max_workspace_bytes,
            scioxide_core::quantity!("mgc_bytes"),
        )?;
        try_filled(
            len,
            0.0,
            scioxide_core::quantity!("mgc_statistic_workspace"),
        )
    }
    /// Computes the MGC statistic, optionally with the rows of `y` permuted.
    ///
    /// `permutation[i]` is the `y` row paired with `x` row `i`. On success the
    /// first `k × l` entries of `scratch` hold the local correlation map, where
    /// `[k, l]` is [`Self::map_shape`]. The cost is O(n²) time.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `scratch` is shorter than [`Self::scratch_len`].
    /// - `InvalidInput` if `permutation` is not a permutation of `0..n`.
    pub fn statistic(
        &self,
        permutation: Option<&[usize]>,
        scratch: &mut [f64],
    ) -> Result<MgcStatistic, SciOxideError> {
        if scratch.len() < self.scratch_len()? {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "short MGC scratch",
            ));
        }
        if let Some(p) = permutation
            && (p.len() != self.n
                || p.iter()
                    .enumerate()
                    .any(|(i, &v)| v >= self.n || p[..i].contains(&v)))
        {
            return Err(invalid(
                scioxide_core::argument!("permutation"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "MGC row permutation must be a bijection",
            ));
        }
        covariance(self.n, &self.x, &self.y, permutation, scratch)?;
        let [nx, ny] = self.map_shape();
        let size = nx * ny;
        let (map, rest) = scratch.split_at_mut(size);
        for (i, row) in map.chunks_exact_mut(ny).enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let a = self.x.variance[i];
                let b = self.y.variance[j];
                *v = if a <= 0.0 || b <= 0.0 {
                    0.0
                } else {
                    finite(*v / a.sqrt() / b.sqrt())?.clamp(-1.0, 1.0)
                };
            }
        }
        let global = map[size - 1];
        let mut best = MgcStatistic {
            correlation: global,
            scale: [nx, ny],
        };
        if nx == 1 || ny == 1 {
            return Ok(best);
        }
        let threshold = self.threshold.max(global);
        let (labels, rest) = rest.split_at_mut(size);
        let queue = &mut rest[..size];
        labels.fill(0.0);
        let (mut largest, mut best_component) = (0usize, 0usize);
        let mut component = 0usize;
        for start in 0..size {
            if labels[start] != 0.0 || map[start] <= threshold {
                continue;
            }
            component += 1;
            let id = component as f64;
            let (mut head, mut tail) = (0, 1);
            queue[0] = start as f64;
            labels[start] = id;
            while head < tail {
                let index = queue[head] as usize;
                head += 1;
                let row = index / ny;
                let col = index % ny;
                for next in [
                    if row > 0 { Some(index - ny) } else { None },
                    if row + 1 < nx { Some(index + ny) } else { None },
                    if col > 0 { Some(index - 1) } else { None },
                    if col + 1 < ny { Some(index + 1) } else { None },
                ]
                .into_iter()
                .flatten()
                {
                    if labels[next] == 0.0 && map[next] > threshold {
                        labels[next] = id;
                        queue[tail] = next as f64;
                        tail += 1;
                    }
                }
            }
            if tail > largest {
                largest = tail;
                best_component = component;
            }
        }
        let minimum = (0.02 * nx.max(ny) as f64).ceil() as usize * nx.min(ny);
        if largest >= minimum && largest > 0 {
            for (i, &v) in map.iter().enumerate() {
                if labels[i] == best_component as f64 && v >= best.correlation {
                    best = MgcStatistic {
                        correlation: v,
                        scale: [i / ny + 1, i % ny + 1],
                    };
                }
            }
        }
        Ok(best)
    }
    /// Creates the matching upper-tail permutation plan for [`Self::test`].
    ///
    /// The plan permutes the pairing of `x` and `y` rows, or the group labels for
    /// a [`Self::two_sample`] test. Exact mode enumerates all `n!` pairings, or
    /// all label assignments.
    ///
    /// # Errors
    ///
    /// - Any error of [`ResamplingPlan::new`].
    /// - `ResourceLimit` if the per-worker scratch exceeds the plan's workspace
    ///   byte limit.
    pub fn permutation_plan(
        &self,
        mode: PermutationMode,
        options: ResamplingOptions<f64>,
    ) -> Result<ResamplingPlan<f64>, SciOxideError> {
        let singleton = [self.n];
        let (lengths, kind) = match &self.groups {
            Some(groups) => (&groups[..], PermutationKind::Independent),
            None => (&singleton[..], PermutationKind::Pairings),
        };
        ResamplingPlan::new(
            lengths,
            ResamplingMethod::Permutation {
                kind,
                mode,
                alternative: TestAlternative::Greater,
            },
            options,
        )?
        .with_statistic_scratch(self.scratch_len()?)
    }
    fn report(
        &self,
        report: ResamplingResult<f64>,
    ) -> Result<ClassicalPermutationResult<f64>, SciOxideError> {
        let test = match (report.observed, report.inference) {
            (Some(statistic), Some(ResamplingInference::Test(p))) => {
                let mut result = result(
                    statistic,
                    p.p_value,
                    ClassicalMethod::Permutation {
                        exact: p.exact,
                        replicates: report.completed,
                    },
                    if self.groups.is_some() {
                        TestSamplePolicy::IndependentGroups
                    } else {
                        TestSamplePolicy::Paired
                    },
                    self.groups.unwrap_or([self.n, self.n]),
                )?;
                result.effect = Some((ClassicalEffect::Association, statistic));
                Some(result)
            }
            _ => None,
        };
        Ok(ClassicalPermutationResult {
            test,
            resampling: report,
        })
    }
    fn validate_plan(&self, plan: &ResamplingPlan<f64>) -> Result<(), SciOxideError> {
        let (shape, kind) = match self.groups {
            Some([a, b]) => (
                plan.offsets.as_slice() == [0, a, a + b],
                PermutationKind::Independent,
            ),
            None => (
                plan.offsets.as_slice() == [0, self.n],
                PermutationKind::Pairings,
            ),
        };
        if !shape
            || plan.statistic_scratch != self.scratch_len()?
            || !matches!(plan.method(), ResamplingMethod::Permutation {kind:k,alternative:TestAlternative::Greater,..} if k==kind)
        {
            return Err(invalid(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "MGC requires its matching upper-tail permutation plan",
            ));
        }
        Ok(())
    }
    /// Computes the permutation p-value of the MGC test.
    ///
    /// `plan` must come from [`Self::permutation_plan`]. The remaining arguments
    /// have the same meaning as in [`ResamplingPlan::evaluate_into`]. The effect of
    /// the returned test is the MGC statistic.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `plan` does not match this test.
    /// - Any error of [`ResamplingPlan::evaluate_into`].
    #[allow(clippy::too_many_arguments)]
    pub fn test_into<'scope, 'capability>(
        &self,
        plan: &ResamplingPlan<f64>,
        replicates: impl crate::ReplicateExecution<f64>,
        output: &mut [f64],
        workspace: &mut ResamplingWorkspace<f64>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ClassicalPermutationResult<f64>, SciOxideError> {
        workspace.reject_attempt();
        self.validate_plan(plan)?;
        let callback = |s: crate::ResampledIndices<'_>, scratch: &mut [f64]| {
            Ok(self.statistic(Some(s.indices()), scratch)?.correlation)
        };
        self.report(plan.evaluate_into(
            ResamplingProblem::IndexedWorkspace(&callback),
            replicates,
            output,
            workspace,
            scope,
        )?)
    }
    /// Evaluates the permutation test with owned replicate storage.
    ///
    /// # Errors
    ///
    /// Returns allocation errors and the errors of the caller-output form.
    pub fn test(
        &self,
        plan: &ResamplingPlan<f64>,
        replicates: impl crate::ReplicateExecution<f64>,
    ) -> Result<ClassicalPermutationResult<f64>, SciOxideError> {
        let mut output = scioxide_core::backend::try_filled(
            plan.replicates(),
            0.0,
            scioxide_core::quantity!("permutation_output"),
        )?;
        let mut workspace = plan.workspace()?;
        self.test_into(
            plan,
            replicates,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
