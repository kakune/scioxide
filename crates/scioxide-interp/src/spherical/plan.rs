//! Least-squares spline fits on the sphere with fixed breakpoints: sample
//! layouts, the factored design matrix, fit storage and fit results.

use crate::NodeSeparation;
use crate::difference_coefficients;
use crate::validate::validate_nodes;
use alloc::borrow::Cow;
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, ExecutionContext, ExecutionPolicy, ProblemToken,
    RealScalar, SciOxideError, SolveBudgets, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::{
    DirectSolveOptions, HouseholderDecomposition, HouseholderMinimumNormPlan,
    HouseholderOrientation, HouseholderPivoting, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace, Matrix, RankPolicy, VectorViewMut,
};

use super::basis::{SphereBasis, resolve};
use super::model::SphericalSpline;
use crate::SplineFitLimits;
use crate::adaptive::{AdaptiveSplineReport, sample_dimension};
use crate::checked::intermediate;
use crate::error::invalid;
use crate::protocol::ExtrapolationPolicy;
use crate::values::{scatter, zeroed};

/// Lists the sample sites of a spherical fit as `[θ, φ]` in radians.
///
/// Colatitudes `θ` lie in `[0, π]`. Longitudes `φ` are any finite angle and
/// wrap modulo `2π`. Samples at a pole with different longitudes, and samples
/// whose longitudes differ by a multiple of `2π`, are the same point on the
/// sphere.
///
/// # Examples
///
/// ```
/// use scioxide_interp::SphericalSamples;
/// let theta = [0.5_f64, 1.0];
/// let longitude = [0.0, 2.0, 4.0];
/// let grid = SphericalSamples::Rectangular {
///     theta: &theta,
///     longitude: &longitude,
/// };
/// let scattered = SphericalSamples::Scattered(&[[0.5_f64, 0.0], [1.0, 2.0]]);
/// assert!(matches!(grid, SphericalSamples::Rectangular { .. }));
/// assert!(matches!(scattered, SphericalSamples::Scattered(points) if points.len() == 2));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum SphericalSamples<'a, S: RealScalar> {
    /// Scattered points in any order, with one value per point.
    Scattered(&'a [[S; 2]]),
    /// The grid `theta × longitude`, with values ordered longitude fastest.
    ///
    /// Both axes are finite and strictly increasing. Rows at a pole and a
    /// longitude column that repeats the first one modulo `2π` are allowed.
    Rectangular {
        /// Colatitudes of the grid rows.
        theta: &'a [S],
        /// Longitudes of the grid columns.
        longitude: &'a [S],
    },
}

impl<'a, S: RealScalar> SphericalSamples<'a, S> {
    /// Number of sample sites; validates rectangular axes.
    pub(super) fn count(&self) -> Result<usize, SciOxideError> {
        match *self {
            Self::Scattered(points) => Ok(points.len()),
            Self::Rectangular { theta, longitude } => {
                for axis in [theta, longitude] {
                    validate_nodes(
                        axis,
                        1,
                        NodeSeparation::Exact,
                        scioxide_core::argument!("samples"),
                    )?;
                }
                checked_mul(
                    theta.len(),
                    longitude.len(),
                    scioxide_core::quantity!("grid_points"),
                )
            }
        }
    }

    /// Sample points; a rectangular grid is expanded with longitude fastest.
    pub(super) fn points(&self) -> Result<Cow<'a, [[S; 2]]>, SciOxideError> {
        match *self {
            Self::Scattered(points) => Ok(Cow::Borrowed(points)),
            Self::Rectangular { theta, longitude } => {
                let mut points =
                    try_with_capacity(self.count()?, scioxide_core::quantity!("grid_points"))?;
                for &t in theta {
                    for &p in longitude {
                        points.push([t, p]);
                    }
                }
                Ok(Cow::Owned(points))
            }
        }
    }
}

/// Deterministic groups of already normalized, finite sample coordinates.
struct GroupedSites<S> {
    sites: Vec<[S; 2]>,
    groups: Vec<usize>,
    weights: Vec<S>,
}

fn group_sites<S: RealScalar>(
    points: &[[S; 2]],
    weights: &[S],
) -> Result<GroupedSites<S>, SciOxideError> {
    let quantity = scioxide_core::quantity!("site_groups");
    let mut indices = try_with_capacity(points.len(), quantity)?;
    indices.extend(
        weights
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (w != S::ZERO).then_some(i)),
    );
    indices.sort_unstable_by(|&a, &b| {
        // Validation excludes NaNs. Numeric comparison treats signed zeros as
        // equal, unlike bitwise ordering. Original indices break exact ties.
        points[a][0]
            .partial_cmp(&points[b][0])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| {
                points[a][1]
                    .partial_cmp(&points[b][1])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .then_with(|| a.cmp(&b))
    });
    let mut groups = try_filled(points.len(), usize::MAX, quantity)?;
    let mut representative = None;
    for index in indices {
        let first = match representative {
            Some(first) if points[first] == points[index] => first,
            _ => {
                representative = Some(index);
                index
            }
        };
        groups[index] = first;
    }
    let mut sites = try_with_capacity(points.len(), quantity)?;
    let mut sums = try_with_capacity(points.len(), quantity)?;
    for (index, &weight) in weights.iter().enumerate() {
        let first = groups[index];
        if first == usize::MAX {
            continue;
        }
        let group = if first == index {
            let group = sites.len();
            sites.push(points[index]);
            sums.push(S::ZERO);
            group
        } else {
            // The smallest original index is visited before every duplicate.
            groups[first]
        };
        groups[index] = group;
        sums[group] = intermediate(sums[group] + weight)?;
    }
    Ok(GroupedSites {
        sites,
        groups,
        weights: sums,
    })
}

/// Fits a [`SphericalSpline`](crate::SphericalSpline) with fixed breakpoints
/// to samples on the sphere by penalized least squares.
///
/// The colatitude breakpoints span `[0, π]`, and the longitude breakpoints
/// span `[0, 2π]` with at least four intervals. For samples `(θ_k, φ_k, y_k)`
/// with weights `w_k ≥ 0` and a penalty `λ ≥ 0`, the fit minimizes
///
/// ```text
/// Σ_k w_k |y_k − f(θ_k, φ_k)|² + λ (‖Δ²_φ c‖² + ‖Δ²_θ c‖²)
/// ```
///
/// over the coefficients, where `Δ²_φ` and `Δ²_θ` are second differences of
/// the tensor coefficients `c_ij` along longitude (periodically) and colatitude.
/// The six polar modes are not penalized. Samples at the same point on the
/// sphere, such as pole samples or longitudes that differ by `2π`, are merged
/// into one site whose weight is the sum of theirs. Weights must be finite and
/// nonnegative with at least one positive; `None` gives unit weights.
///
/// The plan factors the dense weighted design matrix once with Householder
/// transformations. With at least as many rows as modes it uses QR with column
/// pivoting and solves the least-squares problem. With fewer rows it uses LQ
/// and returns the minimum-norm solution, which interpolates the merged
/// sites. A rank-deficient design is an error; no regularization is added.
/// Each fit solves one right-hand side per real value component, so every
/// component equals the scalar fit of that component.
///
/// For `r` rows (merged sites plus penalty rows) and `c` modes, the design and
/// its factor take `O(r c)` storage and the factorization costs
/// `O(r c min(r, c))`. Merging duplicate sites costs `O(m log m)` for `m`
/// samples. The memory limit is checked before the design is allocated. The
/// design is dense because scattered sites and the polar modes, which are
/// nonzero over the whole sphere, give no band structure.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, SphericalSamples, SphericalSplinePlan, SplineFitLimits,
/// };
/// use std::f64::consts::PI;
/// let theta: Vec<f64> = (0..=6).map(|i| PI * i as f64 / 6.0).collect();
/// let longitude: Vec<f64> = (0..12).map(|j| PI * j as f64 / 6.0).collect();
/// let samples = SphericalSamples::Rectangular {
///     theta: &theta,
///     longitude: &longitude,
/// };
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI / 2.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     samples,
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// // 1 and h01(θ/π) lie in the space of the polar modes, so the fit is exact.
/// let values: Vec<[f64; 2]> = theta
///     .iter()
///     .flat_map(|t| {
///         longitude.iter().map(move |_| {
///             let u = t / PI;
///             [1.0, u * u * (3.0 - 2.0 * u)]
///         })
///     })
///     .collect();
/// let fit = plan.fit(&values)?;
/// assert!(fit.report().residual() < 1e-24);
/// let mut workspace = plan.workspace(2)?;
/// let mut coefficients = vec![0.0; 2 * plan.mode_count()];
/// let report = plan.fit_into(
///     &values,
///     &mut coefficients,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(coefficients, fit.spline().coefficients());
/// assert_eq!(report.residual(), fit.report().residual());
/// let north = fit
///     .spline()
///     .evaluate([0.0, 2.0], ExtrapolationPolicy::Reject)?;
/// assert!((north[0] - 1.0).abs() < 1e-12 && north[1].abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SphericalSplinePlan<S: RealScalar> {
    basis: SphereBasis<S>,
    samples: Vec<[S; 2]>,
    weights: Vec<S>,
    groups: Vec<usize>,
    group_weights: Vec<S>,
    factor: HouseholderMinimumNormPlan<S>,
    penalty: S,
    rows: usize,
    limits: SplineFitLimits,
    reciprocal_condition: S,
}

/// Describes the scratch that [`SphericalSplinePlan::fit_into`] needs for values
/// of a given number of real components.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_interp::{SphericalSamples, SphericalSplinePlan, SplineFitLimits};
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (1..8).map(|i| [0.4 * i as f64, 0.9 * i as f64]).collect();
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     SphericalSamples::Scattered(&samples),
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// let requirements = plan.workspace_requirements(3)?;
/// assert_eq!(requirements.dimension(), 3);
/// let mut workspace = plan.workspace(1)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SphericalSplineFitRequirements {
    dimension: usize,
    rows: usize,
    columns: usize,
    samples: usize,
    linalg: LinalgRequirements,
}

impl SphericalSplineFitRequirements {
    /// Returns the number of real components per value.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of real coefficient components a fit writes,
    /// `modes · dimension`.
    pub const fn coefficients(&self) -> usize {
        self.columns * self.dimension
    }
}

/// Holds reusable scratch for [`SphericalSplinePlan::fit_into`].
///
/// Create it with [`SphericalSplinePlan::workspace`] and grow it with
/// [`Self::try_reserve`]. After a successful fit it holds the weighted squared
/// residual of each sample.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_interp::{SphericalSamples, SphericalSplinePlan, SplineFitLimits};
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (1..8).map(|i| [0.4 * i as f64, 0.9 * i as f64]).collect();
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     SphericalSamples::Scattered(&samples),
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// let mut workspace = plan.workspace(1)?;
/// let mut coefficients = vec![0.0; plan.mode_count()];
/// plan.fit_into(
///     &[1.0; 7],
///     &mut coefficients,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(workspace.residuals().len(), 7);
/// assert!(workspace.residuals().iter().all(|r| *r < 1e-20));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct SphericalSplineFitWorkspace<S: RealScalar> {
    rhs: Vec<S>,
    solution: Vec<S>,
    coefficients: Vec<S>,
    query: Vec<S>,
    residuals: Vec<S>,
    valid: usize,
    linalg: LinalgWorkspace<'static, S>,
}

impl<S: RealScalar> SphericalSplineFitWorkspace<S> {
    /// Returns whether this workspace covers `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit` if the solve scratch size cannot be
    ///   computed.
    pub fn check(
        &self,
        requirements: &SphericalSplineFitRequirements,
    ) -> Result<Capacity, SciOxideError> {
        let linalg = self.linalg.check(&requirements.linalg)?;
        Ok(
            if linalg == Capacity::Sufficient
                && self.rhs.len() >= requirements.rows
                && self.solution.len() >= requirements.columns
                && self.coefficients.len() >= requirements.coefficients()
                && self.query.len() >= query_len(requirements.dimension)
                && self.residuals.len() >= requirements.samples
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the workspace in place to cover `requirements`.
    ///
    /// On failure the existing capacity is kept.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &SphericalSplineFitRequirements,
    ) -> Result<(), SciOxideError> {
        grow(&mut self.rhs, requirements.rows)?;
        grow(&mut self.solution, requirements.columns)?;
        grow(&mut self.coefficients, requirements.coefficients())?;
        grow(&mut self.query, query_len(requirements.dimension))?;
        grow(&mut self.residuals, requirements.samples)?;
        self.linalg.try_reserve(&requirements.linalg)
    }

    /// Returns the weighted squared residual `w_k |y_k − f(θ_k, φ_k)|²` of each
    /// sample after the last successful fit, summed over value components, or
    /// an empty slice.
    pub fn residuals(&self) -> &[S] {
        &self.residuals[..self.valid]
    }
}

/// Query scratch of the residual evaluation: one staged value and two local rows.
const fn query_len(dimension: usize) -> usize {
    dimension.saturating_add(8)
}

/// Grow a zero-filled scratch vector to at least `length` entries.
fn grow<S: RealScalar>(values: &mut Vec<S>, length: usize) -> Result<(), SciOxideError> {
    if values.len() < length {
        values.try_reserve(length - values.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sphere_fit_scratch"),
                length,
            )
        })?;
        values.resize(length, S::ZERO);
    }
    Ok(())
}

/// Holds a fitted [`SphericalSpline`](crate::SphericalSpline) and its report.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     AdaptiveSplineTermination, SphericalSamples, SphericalSplinePlan, SplineFitLimits,
/// };
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (1..8).map(|i| [0.4 * i as f64, 0.9 * i as f64]).collect();
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     SphericalSamples::Scattered(&samples),
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// let fit = plan.fit(&[3.0; 7])?;
/// assert_eq!(
///     fit.report().termination(),
///     AdaptiveSplineTermination::FixedPenalty
/// );
/// assert_eq!(fit.spline().coefficients().len(), plan.mode_count());
/// let spline = fit.into_spline();
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SphericalSplineFit<S: RealScalar, V: AffineValue<Scale = S> = S> {
    pub(super) spline: SphericalSpline<S, V>,
    pub(super) report: AdaptiveSplineReport<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> SphericalSplineFit<S, V> {
    /// Returns the fitted spline.
    pub const fn spline(&self) -> &SphericalSpline<S, V> {
        &self.spline
    }

    /// Returns the fit report.
    pub const fn report(&self) -> &AdaptiveSplineReport<S> {
        &self.report
    }

    /// Returns the fitted spline, consuming the result.
    pub fn into_spline(self) -> SphericalSpline<S, V> {
        self.spline
    }
}

impl<S: RealScalar> SphericalSplinePlan<S> {
    /// Builds and factors the design on the calling thread.
    ///
    /// Equivalent to [`Self::build`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::build`].
    pub fn new(
        theta_breaks: &[S],
        longitude_breaks: &[S],
        samples: SphericalSamples<'_, S>,
        weights: Option<&[S]>,
        penalty: S,
        limits: SplineFitLimits,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            theta_breaks,
            longitude_breaks,
            samples,
            weights,
            penalty,
            limits,
            ExecutionContext::sequential_native(),
        )
    }

    /// Builds and factors the design within `scope`.
    ///
    /// `penalty` is the smoothing parameter `λ`. The design size is checked
    /// against `limits.max_workspace_bytes()` before any storage is reserved.
    /// The factorization and a condition estimate run in a child scope of
    /// `scope`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `penalty` is negative or not finite, there are fewer
    ///   than two colatitude or five longitude breakpoints, the breakpoints do
    ///   not run from `0` to `π` and `2π`, breakpoints or grid axes are not
    ///   strictly increasing, a weight is negative, or no weight is positive.
    /// - `DomainError` if a sample colatitude is outside `[0, π]`.
    /// - `DimensionMismatch` if the number of weights differs from the number of
    ///   samples.
    /// - `NonFinite` if a sample, breakpoint or weight is NaN or infinite.
    /// - `ResourceLimit` if the knot or memory limit in `limits` is exceeded.
    /// - `NumericalInstability` (`IllConditioned`) if the design is numerically
    ///   rank deficient.
    /// - `ExecutionStopped` on cancellation or a deadline, and
    ///   `AllocationFailure` if storage cannot be reserved.
    pub fn build<'scope, 'capability>(
        theta_breaks: &[S],
        longitude_breaks: &[S],
        samples: SphericalSamples<'_, S>,
        weights: Option<&[S]>,
        penalty: S,
        limits: SplineFitLimits,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        if !penalty.is_finite() || penalty < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("penalty"),
                "sphere penalty must be finite and nonnegative",
            ));
        }
        let count = samples.count()?;
        // Admit the full caller-selected axis shape before copying any knots.
        // A polar-only model still owns its longitude basis even though it has
        // no interior coefficient lanes, so charge that storage separately.
        // Too-short axes are rejected by `SphereBasis::new`.
        let quantity = scioxide_core::quantity!("sphere_axes");
        let predicted_columns = checked_add(
            6,
            checked_mul(
                theta_breaks.len().saturating_sub(2),
                longitude_breaks.len().saturating_sub(1),
                quantity,
            )?,
            quantity,
        )?;
        let axis_bytes = checked_mul(
            checked_add(theta_breaks.len(), longitude_breaks.len(), quantity)?,
            core::mem::size_of::<S>().max(core::mem::size_of::<usize>()) * 32,
            quantity,
        )?;
        let grouping_bytes = checked_mul(count, core::mem::size_of::<usize>(), quantity)?;
        limits.check_spherical::<S>(
            count,
            predicted_columns,
            1,
            checked_add(axis_bytes, grouping_bytes, quantity)?,
        )?;
        let points = samples.points()?;
        let basis = SphereBasis::new(theta_breaks, longitude_breaks, &limits)?;
        let columns = basis.columns()?;
        let interior = basis.interior();
        let q = basis.periodic();
        let penalty_rows = if penalty > S::ZERO && interior > 0 {
            let quantity = scioxide_core::quantity!("penalty_rows");
            checked_add(
                checked_mul(interior, q, quantity)?,
                checked_mul(interior.saturating_sub(2), q, quantity)?,
                quantity,
            )?
        } else {
            0
        };
        let weights = crate::adaptive::weights(weights, points.len())?;
        let mut normalized = try_with_capacity(points.len(), scioxide_core::quantity!("samples"))?;
        for &point in points.iter() {
            let mut point = resolve(
                point,
                ExtrapolationPolicy::Reject,
                scioxide_core::argument!("samples"),
            )?
            .effective();
            if point[0] == S::ZERO || point[0] == S::PI {
                point[1] = S::ZERO;
            }
            normalized.push(point);
        }
        let GroupedSites {
            sites,
            groups,
            weights: group_weights,
        } = group_sites(&normalized, &weights)?;
        let rows = checked_add(
            sites.len(),
            penalty_rows,
            scioxide_core::quantity!("design_rows"),
        )?;
        limits.check_spherical::<S>(rows, columns, 1, 0)?;
        let matrix = design(&basis, &sites, &group_weights, penalty, rows, columns)?;
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "spherical-spline", 1),
            ((rows as u128) << 64) | columns as u128,
            0,
        );
        let (factor, reciprocal_condition) = scope.into().run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |context| {
                context.execution().check()?;
                let mut factor = HouseholderDecomposition::empty();
                let mut workspace = LinalgWorkspace::new();
                HouseholderDecomposition::factor_into(
                    &matrix,
                    if rows >= columns {
                        HouseholderOrientation::Qr
                    } else {
                        HouseholderOrientation::Lq
                    },
                    HouseholderPivoting::Column,
                    &LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut factor,
                    &mut workspace,
                    &mut *context,
                )?;
                if !factor
                    .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
                    .is_full_rank()
                {
                    return Err(rank_deficient());
                }
                let factor = factor.into_minimum_norm(
                    &DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon)
                        .with_execution_options(LinalgExecutionOptions::new()),
                )?;
                let mut workspace = factor.workspace(1)?;
                let condition = factor.condition_estimate_scoped(&mut workspace, &mut *context)?;
                Ok((factor, condition.reciprocal().value()))
            },
        )?;
        Ok(Self {
            basis,
            samples: normalized,
            weights,
            groups,
            group_weights,
            factor,
            penalty,
            rows,
            limits,
            reciprocal_condition,
        })
    }

    /// Returns the colatitude knot vector and the longitude knot vector, which
    /// extends the periodic breakpoints by three knots at each end.
    pub fn knots(&self) -> [&[S]; 2] {
        self.basis.knots()
    }

    /// Returns the smoothing parameter `λ`.
    pub const fn penalty(&self) -> S {
        self.penalty
    }

    /// Returns the limits that fits check.
    pub const fn limits(&self) -> SplineFitLimits {
        self.limits
    }

    /// Returns the number of modes, which is the number of coefficients per real
    /// value component.
    pub fn mode_count(&self) -> usize {
        // Validated when the plan was built.
        self.factor.factor().cols()
    }

    /// Returns the workspace requirements for values of `dimension` real components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `ResourceLimit` if the fit exceeds the memory limit of the plan.
    /// - `SizeOverflow` if the size cannot be computed.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<SphericalSplineFitRequirements, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "sphere fit values need at least one real component",
            ));
        }
        let columns = self.mode_count();
        self.limits
            .check_spherical::<S>(self.rows, columns, dimension, 0)?;
        checked_mul(columns, dimension, scioxide_core::quantity!("coefficients"))?;
        Ok(SphericalSplineFitRequirements {
            dimension,
            rows: self.rows,
            columns,
            samples: self.samples.len(),
            linalg: self.factor.workspace_requirements(1)?,
        })
    }

    /// Allocates a workspace for values of `dimension` real components.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<SphericalSplineFitWorkspace<S>, SciOxideError> {
        let mut workspace = SphericalSplineFitWorkspace {
            rhs: Vec::new(),
            solution: Vec::new(),
            coefficients: Vec::new(),
            query: Vec::new(),
            residuals: Vec::new(),
            valid: 0,
            linalg: LinalgWorkspace::new(),
        };
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Fits one value per sample and returns the spline with its report.
    ///
    /// This runs [`Self::fit_into`] on the calling thread.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::fit_into`]; an empty `values` is a
    ///   `DimensionMismatch`.
    /// - `AllocationFailure` if the workspace or spline cannot be allocated.
    pub fn fit<V: AffineValue<Scale = S>>(
        &self,
        values: &[V],
    ) -> Result<SphericalSplineFit<S, V::Owned>, SciOxideError> {
        self.fit_scoped(values, ExecutionContext::sequential_native())
            .map(|(fit, _)| fit)
    }

    /// Owned fit within `scope`, returning the workspace holding its residuals.
    pub(super) fn fit_scoped<'scope, 'capability, V: AffineValue<Scale = S>>(
        &self,
        values: &[V],
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ScopedFit<S, V::Owned>, SciOxideError> {
        let template = values
            .first()
            .ok_or_else(|| crate::error::shape(self.samples.len(), 0))?;
        let dimension = template.dimension()?;
        let mut workspace = self.workspace(dimension)?;
        let mut coefficients = try_filled(
            checked_mul(
                self.mode_count(),
                dimension,
                scioxide_core::quantity!("coefficients"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("coefficients"),
        )?;
        let report = self.fit_into(values, coefficients.as_mut_slice(), &mut workspace, scope)?;
        let spline = SphericalSpline::from_parts(
            self.basis.try_clone()?,
            coefficients,
            dimension,
            zeroed(template)?,
        )?;
        Ok((SphericalSplineFit { spline, report }, workspace))
    }

    /// Fits one value per sample, writes the coefficients to `out` and returns
    /// the report.
    ///
    /// `out` holds `mode_count() · m` real components for `m` components per
    /// value, in the order of [`SphericalSpline::coefficients`](crate::SphericalSpline::coefficients).
    /// Each real component is solved with the stored factor in a child scope of
    /// `scope`. The residual of each sample is left in the workspace.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of values differs from the number of
    ///   samples, the values differ in shape, `out` has the wrong shape or
    ///   `workspace` is too small.
    /// - `InvalidInput` if the values have zero components.
    /// - `NonFinite` if a value is NaN or infinite, or an intermediate overflows.
    /// - `ResourceLimit` if the fit exceeds the memory limit of the plan.
    /// - `NumericalInstability` if a solve loses rank.
    /// - `ExecutionStopped` on cancellation or a deadline.
    pub fn fit_into<'scope, 'capability, V, O>(
        &self,
        values: &[V],
        out: &mut O,
        workspace: &mut SphericalSplineFitWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdaptiveSplineReport<S>, SciOxideError>
    where
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        workspace.valid = 0;
        let dimension = sample_dimension(values, self.samples.len())?;
        let requirements = self.workspace_requirements(dimension)?;
        scioxide_core::check_value_shape(out, requirements.coefficients())?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(crate::error::shape(
                requirements.coefficients(),
                workspace.coefficients.len(),
            ));
        }
        let columns = requirements.columns;
        let SphericalSplineFitWorkspace {
            rhs,
            solution,
            coefficients,
            query,
            residuals,
            linalg,
            ..
        } = &mut *workspace;
        let rhs = &mut rhs[..self.rows];
        let solution = &mut solution[..columns];
        let coefficients = &mut coefficients[..requirements.coefficients()];
        let residuals = &mut residuals[..self.samples.len()];
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "spherical-spline-fit", 1),
            columns as u128,
            dimension as u128,
        );
        scope.into().run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |context| {
                for component in 0..dimension {
                    context.execution().check()?;
                    rhs.fill(S::ZERO);
                    for ((value, &group), &weight) in
                        values.iter().zip(&self.groups).zip(&self.weights)
                    {
                        if group == usize::MAX {
                            continue;
                        }
                        let scaled = value.component(component)?
                            * (weight / self.group_weights[group].sqrt());
                        rhs[group] = intermediate(rhs[group] + scaled)?;
                    }
                    let report = self.factor.solve_into(
                        &*rhs,
                        VectorViewMut::from_slice(solution),
                        linalg,
                        &mut *context,
                    )?;
                    if !report.rank().is_full_rank() {
                        return Err(rank_deficient());
                    }
                    scatter(solution, dimension, component, coefficients);
                }
                Ok(())
            },
        )?;
        let (stage, rows) = query.split_at_mut(dimension);
        for (((&point, value), &weight), slot) in self
            .samples
            .iter()
            .zip(values)
            .zip(&self.weights)
            .zip(residuals.iter_mut())
        {
            self.basis
                .evaluate(point, [0, 0], coefficients, dimension, rows, stage)?;
            let mut residual = S::ZERO;
            for (component, &predicted) in stage.iter().enumerate() {
                let d = (predicted - value.component(component)?).abs() * weight.sqrt();
                residual = intermediate(residual + d * d)?;
            }
            *slot = residual;
        }
        let residual = residuals
            .iter()
            .try_fold(S::ZERO, |sum, &r| intermediate(sum + r))?;
        let [theta_knots, phi_knots] = self.basis.knots();
        let report = AdaptiveSplineReport::fixed(residual, self.penalty, self.reciprocal_condition)
            .with_geometry(
                [theta_knots.len(), phi_knots.len()],
                [2, 2],
                Some(1),
                [S::ZERO; 2],
                [S::PI, S::PI + S::PI],
            );
        scioxide_core::write_value(&*coefficients, out)?;
        workspace.valid = self.samples.len();
        Ok(report)
    }
}

/// An owned fit and the workspace holding its per-sample residuals.
type ScopedFit<S, V> = (SphericalSplineFit<S, V>, SphericalSplineFitWorkspace<S>);

/// Numerical rank loss of the spherical design.
fn rank_deficient() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::IllConditioned,
        "spherical spline design is numerically rank deficient",
    )
}

/// Weighted site rows followed by second-difference penalty rows along
/// longitude, then colatitude, of the tensor coefficients.
fn design<S: RealScalar>(
    basis: &SphereBasis<S>,
    sites: &[[S; 2]],
    group_weights: &[S],
    penalty: S,
    rows: usize,
    columns: usize,
) -> Result<Matrix<S>, SciOxideError> {
    let mut matrix = Matrix::<S>::zeros(rows, columns)?;
    let data = matrix.as_mut_slice();
    let mut local = [S::ZERO; 8];
    for (i, (&point, &weight)) in sites.iter().zip(group_weights).enumerate() {
        basis.design_row(
            point,
            weight.sqrt(),
            &mut local,
            &mut data[i * columns..(i + 1) * columns],
        )?;
    }
    if rows > sites.len() {
        let interior = basis.interior();
        let q = basis.periodic();
        let scale = penalty.sqrt();
        let stencil = difference_coefficients::<S>(2)?;
        let mut row = sites.len();
        for i in 0..interior {
            for j in 0..q {
                for (offset, &multiplier) in stencil.iter().enumerate() {
                    data[row * columns + 6 + i * q + (j + offset) % q] = scale * multiplier;
                }
                row += 1;
            }
        }
        for i in 0..interior.saturating_sub(2) {
            for j in 0..q {
                for (offset, &multiplier) in stencil.iter().enumerate() {
                    data[row * columns + 6 + (i + offset) * q + j] = scale * multiplier;
                }
                row += 1;
            }
        }
    }
    Ok(matrix)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::f64::consts::PI;
    use scioxide_core::{Complex, SciOxideErrorKind};

    const LONGITUDE: [f64; 5] = [0., PI / 2., PI, 1.5 * PI, 2. * PI];

    fn grid() -> Vec<[f64; 2]> {
        (0..=5)
            .flat_map(|i| (0..8).map(move |j| [PI * i as f64 / 5., PI * j as f64 / 4.]))
            .collect()
    }

    #[test]
    fn rectangular_samples_expand_longitude_fastest() {
        let theta = [0.5, 1.0];
        let longitude = [0.0, 2.0, 4.0];
        let samples = SphericalSamples::Rectangular {
            theta: &theta,
            longitude: &longitude,
        };
        assert_eq!(samples.count().unwrap(), 6);
        assert_eq!(samples.points().unwrap()[1], [0.5, 2.0]);
        let bad = SphericalSamples::Rectangular {
            theta: &[1.0, 0.5],
            longitude: &longitude,
        };
        assert!(bad.count().is_err());
    }

    #[test]
    fn components_are_independent_real_solves() {
        let samples = grid();
        let plan = SphericalSplinePlan::new(
            &[0., PI / 2., PI],
            &LONGITUDE,
            SphericalSamples::Scattered(&samples),
            None,
            0.25,
            SplineFitLimits::new(),
        )
        .unwrap();
        let complex: Vec<Complex<f64>> = samples
            .iter()
            .map(|p| Complex::new(p[0].cos(), (p[1]).sin() * p[0].sin()))
            .collect();
        let re: Vec<f64> = complex.iter().map(|z| z.re).collect();
        let im: Vec<f64> = complex.iter().map(|z| z.im).collect();
        let joint = plan.fit(&complex).unwrap();
        let [a, b] = [plan.fit(&re).unwrap(), plan.fit(&im).unwrap()];
        for (mode, pair) in joint.spline().coefficients().chunks_exact(2).enumerate() {
            assert_eq!(pair[0], a.spline().coefficients()[mode]);
            assert_eq!(pair[1], b.spline().coefficients()[mode]);
        }
        let total = a.report().residual() + b.report().residual();
        assert!((joint.report().residual() - total).abs() <= 1e-14 * total.max(1e-300));
    }

    #[test]
    fn fit_storage_is_checked_before_solving() {
        let samples = grid();
        let plan = SphericalSplinePlan::new(
            &[0., PI],
            &LONGITUDE,
            SphericalSamples::Scattered(&samples),
            None,
            0.0,
            SplineFitLimits::new(),
        )
        .unwrap();
        let values = vec![[1.0, 2.0]; samples.len()];
        let mut small = plan.workspace(1).unwrap();
        let mut out = vec![7.0; 2 * plan.mode_count()];
        let error = plan
            .fit_into(
                &values,
                out.as_mut_slice(),
                &mut small,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert!(out.iter().all(|&x| x == 7.0));
        assert!(small.residuals().is_empty());
        assert!(plan.workspace_requirements(0).is_err());
        let tight = SphericalSplinePlan::new(
            &[0., PI],
            &LONGITUDE,
            SphericalSamples::Scattered(&samples),
            None,
            0.0,
            SplineFitLimits::new().with_max_workspace_bytes(32 * 1024 * 1024 + 300_000),
        )
        .unwrap();
        assert_eq!(
            tight.workspace_requirements(64).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
    #[test]
    fn sorted_grouping_preserves_first_sites_and_accumulation_order() {
        let points = [
            [1.0, 2.0],
            [0.0, -0.0],
            [1.0, 2.0],
            [-0.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ];
        let weights = [1e16, 2.0, 1.0, 3.0, 0.0, 1.0];
        let grouped = group_sites(&points, &weights).unwrap();
        assert_eq!(grouped.sites, [[1.0, 2.0], [0.0, -0.0]]);
        assert_eq!(grouped.groups, [0, 1, 0, 1, usize::MAX, 0]);
        // Accumulating the small weights together first would change this sum.
        assert_eq!(grouped.weights, [1e16, 5.0]);
    }

    #[test]
    fn sorted_grouping_handles_large_distinct_and_duplicate_heavy_inputs() {
        for unique in [17, 16384] {
            let points: Vec<_> = (0..16384)
                .map(|i| {
                    let key = ((i * 7919) % 16384) % unique;
                    [key as f64 / 16384.0, 0.5]
                })
                .collect();
            let weights = vec![1.0; points.len()];
            let grouped = group_sites(&points, &weights).unwrap();
            assert_eq!(grouped.sites.len(), unique);
            let mut expected = vec![0.0; unique];
            for (point, &group) in points.iter().zip(&grouped.groups) {
                assert_eq!(*point, grouped.sites[group]);
                expected[group] += 1.0;
            }
            assert_eq!(grouped.weights, expected);
        }
    }
    #[test]
    fn shuffled_poles_and_seams_keep_component_fits() {
        let original = grid();
        let mut samples: Vec<_> = original
            .iter()
            .copied()
            .chain([
                [0.0, 12.0],
                [-0.0, -7.0],
                [PI, -3.0],
                [PI, 17.0],
                [PI / 5.0, 2.0 * PI],
                [PI / 5.0, -0.0],
            ])
            .collect();
        samples.reverse();
        let mut weights = vec![1.0; samples.len()];
        samples.push([0.25, 0.75]);
        weights.push(0.0);
        let plan = SphericalSplinePlan::new(
            &[0.0, PI],
            &LONGITUDE,
            SphericalSamples::Scattered(&samples),
            Some(&weights),
            0.0,
            SplineFitLimits::new(),
        )
        .unwrap();
        assert_eq!(plan.groups.last(), Some(&usize::MAX));
        for (i, &point) in samples.iter().enumerate() {
            if weights[i] == 0.0 {
                continue;
            }
            for (j, &other) in samples.iter().enumerate().take(i) {
                if weights[j] > 0.0 && plan.samples[i] == plan.samples[j] {
                    assert_eq!(plan.groups[i], plan.groups[j], "{point:?} vs {other:?}");
                }
            }
        }
        let values: Vec<_> = samples
            .iter()
            .map(|point| {
                let u = point[0] / PI;
                [1.0, u * u * (3.0 - 2.0 * u)]
            })
            .collect();
        let fit = plan.fit(&values).unwrap();
        assert!(fit.report().residual() < 1e-25);
        for component in 0..2 {
            let scalar = plan
                .fit(&values.iter().map(|v| v[component]).collect::<Vec<_>>())
                .unwrap();
            for (joint, &single) in fit
                .spline()
                .coefficients()
                .chunks_exact(2)
                .zip(scalar.spline().coefficients())
            {
                assert_eq!(joint[component], single);
            }
        }
    }
}
