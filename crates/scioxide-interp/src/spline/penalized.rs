//! Penalized least-squares B-spline fits with a fixed penalty (P-splines).

#![allow(
    clippy::needless_range_loop,
    reason = "band assembly indexes the band and the solution in parallel"
)]

use crate::error::invalid;
use crate::validate::{Weights, validate_count, validate_weights};
use alloc::vec::Vec;

use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, ExecutionContext, NonFiniteOrigin, RealScalar,
    SciOxideError, SolveBudgets, SolveIdentity, SolveScope, SolveState, Triangle,
};
use scioxide_linalg::{
    BandedCholeskyDecomposition, DirectSolveOptions, DirectSolveTermination, HermitianBandMatrix,
    LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut,
    RankPolicy, StructuredOperation, StructuredWorkspaceRequest,
};

use super::basis::{BSplineBasis, KnotSide};
use super::bspline::BSpline;
use crate::checked::{add, mul};
use crate::values::zeroed;

/// Stores the difference penalty `λ ‖Δ^q c‖²` on spline coefficients.
///
/// `Δ^q c` is the vector of `q`-th forward differences of the coefficients
/// `c`, for example `(Δ²c)_i = c_{i+2} − 2 c_{i+1} + c_i`, and `λ ≥ 0` is the
/// smoothing parameter. `λ = 0` gives an ordinary least-squares fit.
///
/// # Examples
///
/// ```
/// use scioxide_interp::SplinePenalty;
/// let penalty = SplinePenalty::new(0.5_f64, 2)?;
/// assert_eq!((penalty.lambda(), penalty.order()), (0.5, 2));
/// assert!(SplinePenalty::new(-1.0_f64, 2).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SplinePenalty<S: RealScalar> {
    lambda: S,
    order: usize,
}

impl<S: RealScalar> SplinePenalty<S> {
    /// Builds a penalty with smoothing parameter `lambda` and difference order `order`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `lambda` is negative, NaN or infinite,
    ///   or `order` is zero.
    pub fn new(lambda: S, order: usize) -> Result<Self, SciOxideError> {
        if !lambda.is_finite() || lambda < S::ZERO {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("lambda"),
                "spline penalty lambda must be finite and nonnegative",
            ));
        }
        if order == 0 {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("order"),
                "spline penalty order must be positive",
            ));
        }
        Ok(Self { lambda, order })
    }

    /// Returns the smoothing parameter `λ`.
    pub const fn lambda(&self) -> S {
        self.lambda
    }

    /// Returns the difference order `q`.
    pub const fn order(&self) -> usize {
        self.order
    }
}

/// Reports the objective terms and the normal-equation residual of one real
/// value component of a penalized fit.
///
/// The terms are evaluated at the computed coefficients. They describe the
/// computed solution and do not bound its forward error. See
/// [`PenalizedSplineFit`] for an example.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SplineFitDiagnostics<S: RealScalar> {
    data_fidelity: S,
    penalty: S,
    normal_residual: S,
}

impl<S: RealScalar> SplineFitDiagnostics<S> {
    const ZERO: Self = Self {
        data_fidelity: S::ZERO,
        penalty: S::ZERO,
        normal_residual: S::ZERO,
    };

    /// Returns the weighted data term `Σ_i w_i (f(x_i) − y_i)²`.
    pub const fn data_fidelity(&self) -> S {
        self.data_fidelity
    }

    /// Returns the penalty term `λ ‖Δ^q c‖²`.
    pub const fn penalty(&self) -> S {
        self.penalty
    }

    /// Returns the componentwise backward error of the normal equations
    /// `A c = b`, `max_r |A c − b|_r / (|b_r| + Σ_j |A_rj| |c_j|)`.
    ///
    /// It is `O(ε)` for a backward-stable solve.
    pub const fn normal_residual(&self) -> S {
        self.normal_residual
    }
}

/// Summarizes a penalized fit over all real value components.
///
/// The objective that the fit minimizes is [`Self::data_fidelity`] plus
/// [`Self::penalty`]. See [`PenalizedSplineFit`] for an example.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SplineFitReport<S: RealScalar> {
    data_fidelity: S,
    penalty: S,
    normal_residual: S,
}

impl<S: RealScalar> SplineFitReport<S> {
    /// Returns the data term summed over components.
    pub const fn data_fidelity(&self) -> S {
        self.data_fidelity
    }

    /// Returns the penalty term summed over components.
    pub const fn penalty(&self) -> S {
        self.penalty
    }

    /// Returns the largest normal-equation backward error over components.
    pub const fn normal_residual(&self) -> S {
        self.normal_residual
    }
}

/// Holds the result of [`PenalizedSplinePlan::fit`]: the spline, the summary
/// report and one [`SplineFitDiagnostics`] per real value component.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BSplineBasis, PenalizedSplinePlan, SplinePenalty};
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let plan = PenalizedSplinePlan::new(
///     basis,
///     vec![0.0, 0.5, 1.0],
///     None,
///     SplinePenalty::new(0.0, 1)?,
/// )?;
/// let fit = plan.fit(&[1.0, 2.0, 3.0])?;
/// assert_eq!(fit.diagnostics().len(), 1);
/// assert!(fit.report().data_fidelity() < 1e-28 && fit.diagnostics()[0].penalty() < 1e-28);
/// assert!((fit.into_spline().coefficients()[1] - 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct PenalizedSplineFit<S: RealScalar, V: AffineValue<Scale = S> = S> {
    spline: BSpline<S, V, 1>,
    report: SplineFitReport<S>,
    diagnostics: Vec<SplineFitDiagnostics<S>>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> PenalizedSplineFit<S, V> {
    /// Returns the fitted spline.
    pub const fn spline(&self) -> &BSpline<S, V, 1> {
        &self.spline
    }

    /// Returns the summary over components.
    pub const fn report(&self) -> &SplineFitReport<S> {
        &self.report
    }

    /// Returns the diagnostics of each real value component, in component order.
    pub fn diagnostics(&self) -> &[SplineFitDiagnostics<S>] {
        &self.diagnostics
    }

    /// Returns the fitted spline, consuming the result.
    pub fn into_spline(self) -> BSpline<S, V, 1> {
        self.spline
    }
}

/// Describes the scratch that [`PenalizedSplinePlan::fit_into`] needs for values
/// of a given number of real components.
///
/// Get it from [`PenalizedSplinePlan::workspace_requirements`]; see
/// [`PenalizedSplineWorkspace`] for an example.
#[derive(Debug, Clone)]
#[must_use]
pub struct PenalizedSplineRequirements {
    dimension: usize,
    coefficients: usize,
    row: usize,
    linalg: LinalgRequirements,
}

impl PenalizedSplineRequirements {
    /// Returns the number of real components per value.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Holds reusable scratch for [`PenalizedSplinePlan::fit_into`].
///
/// Create it with [`PenalizedSplinePlan::workspace`] and grow it with
/// [`Self::try_reserve`]. After a successful fit it holds the diagnostics of
/// each real value component.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext};
/// use scioxide_interp::{BSplineBasis, PenalizedSplinePlan, SplinePenalty};
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let plan = PenalizedSplinePlan::new(
///     basis,
///     vec![0.0, 0.5, 1.0],
///     None,
///     SplinePenalty::new(0.5, 1)?,
/// )?;
/// let requirements = plan.workspace_requirements(2)?;
/// let mut workspace = plan.workspace(1)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// let mut coefficients = [0.0; 4];
/// let values = [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]];
/// let report = plan.fit_into(
///     &values,
///     &mut coefficients,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(workspace.diagnostics().len(), 2);
/// assert!(report.normal_residual() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct PenalizedSplineWorkspace<S: RealScalar> {
    rhs: Vec<S>,
    solution: Vec<S>,
    row: Vec<S>,
    diagnostics: Vec<SplineFitDiagnostics<S>>,
    linalg: LinalgWorkspace<'static, S>,
}

impl<S: RealScalar> PenalizedSplineWorkspace<S> {
    /// Returns whether this workspace covers `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit` if the banded Cholesky scratch size
    ///   cannot be computed.
    pub fn check(
        &self,
        requirements: &PenalizedSplineRequirements,
    ) -> Result<Capacity, SciOxideError> {
        let linalg = self.linalg.check(&requirements.linalg)?;
        Ok(
            if linalg == Capacity::Sufficient
                && self.rhs.len() >= requirements.coefficients
                && self.solution.len() >= requirements.coefficients
                && self.row.len() >= requirements.row
                && self.diagnostics.capacity() >= requirements.dimension
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
        requirements: &PenalizedSplineRequirements,
    ) -> Result<(), SciOxideError> {
        grow(&mut self.rhs, requirements.coefficients)?;
        grow(&mut self.solution, requirements.coefficients)?;
        grow(&mut self.row, requirements.row)?;
        let additional = requirements
            .dimension
            .saturating_sub(self.diagnostics.len());
        self.diagnostics.try_reserve(additional).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("spline_fit_diagnostics"),
                requirements.dimension,
            )
        })?;
        self.linalg.try_reserve(&requirements.linalg)
    }

    /// Returns the diagnostics of each component from the last successful fit,
    /// or an empty slice.
    pub fn diagnostics(&self) -> &[SplineFitDiagnostics<S>] {
        &self.diagnostics
    }
}

/// Grow a zero-filled scratch vector to at least `length` entries.
fn grow<S: RealScalar>(values: &mut Vec<S>, length: usize) -> Result<(), SciOxideError> {
    if values.len() < length {
        values.try_reserve(length - values.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("spline_fit_scratch"),
                length,
            )
        })?;
        values.resize(length, S::ZERO);
    }
    Ok(())
}

/// Fits a B-spline to samples by penalized least squares (P-spline).
///
/// For samples `(x_i, y_i)` with weights `w_i ≥ 0` and a fixed
/// [`BSplineBasis`] `N_j`, the fit minimizes, for each real value component,
///
/// ```text
/// Σ_i w_i (y_i − f(x_i))² + λ ‖Δ^q c‖²,   f(x) = Σ_j c_j N_j(x),
/// ```
///
/// over the coefficients `c`, with the [`SplinePenalty`] `λ ‖Δ^q c‖²`. The
/// minimizer solves the normal equations `(BᵀWB + λ D_qᵀD_q) c = BᵀW y`, where
/// `B_ij = N_j(x_i)`, `W = diag(w)` and `D_q` is the `q`-th difference matrix.
/// The normal matrix is symmetric and banded, with half-bandwidth `max(p, q)`
/// for `λ > 0` and `p` otherwise.
///
/// The normal matrix depends only on the sample sites, weights and penalty, so
/// the plan assembles it from local basis rows and factors it once by banded
/// Cholesky. Each fit then solves one right-hand side per component. For `m`
/// samples, `n` controls and half-bandwidth `k`, preparation costs
/// `O(m (p² + log n) + n k²)` and stores `O(m + n (k + 1))` scalars. A
/// singular or indefinite normal matrix is an error; no diagonal shift is
/// added. The plan does not choose the knots or `λ`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     BSplineBasis, ExtrapolationPolicy, Interpolant, PenalizedSplinePlan, SplinePenalty,
/// };
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2)?;
/// let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
/// let values: Vec<[f64; 2]> = samples.iter().map(|&x| [1.0 + 2.0 * x, x]).collect();
/// let plan = PenalizedSplinePlan::new(basis, samples, None, SplinePenalty::new(0.1, 2)?)?;
/// let fit = plan.fit(&values)?;
/// assert!(fit.report().normal_residual() < 1e-14);
/// let [a, b] = fit.spline().evaluate([0.4], ExtrapolationPolicy::Reject)?;
/// assert!((a - 1.8).abs() < 1e-13 && (b - 0.4).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
/// penalties", *Statistical Science* 11(2), 1996, 89–121.
#[derive(Debug, Clone)]
#[must_use]
pub struct PenalizedSplinePlan<S: RealScalar> {
    basis: BSplineBasis<S>,
    samples: Vec<S>,
    weights: Vec<S>,
    penalty: SplinePenalty<S>,
    differences: Vec<S>,
    normal: HermitianBandMatrix<S>,
    factor: BandedCholeskyDecomposition<S>,
    diagonal_ratio: S,
}

impl<S: RealScalar> PenalizedSplinePlan<S> {
    /// Returns an upper bound on the bytes that [`Self::new`] allocates for
    /// `sample_count` samples, without allocating.
    ///
    /// The bound covers the basis, sample sites, weights, normal matrix, its
    /// Cholesky factor, the assembly row, the difference coefficients and the
    /// factorization scratch. It excludes allocator overhead and unused vector
    /// capacity. The samples and weights themselves are checked by [`Self::new`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `sample_count` is zero or the penalty order is not
    ///   below the number of basis functions.
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSplineBasis, PenalizedSplinePlan, SplinePenalty};
    /// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
    /// assert!(PenalizedSplinePlan::preparation_bytes(&basis, 3, SplinePenalty::new(0.0, 1)?)? > 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn preparation_bytes(
        basis: &BSplineBasis<S>,
        sample_count: usize,
        penalty: SplinePenalty<S>,
    ) -> Result<usize, SciOxideError> {
        validate_count(sample_count, 1, scioxide_core::argument!("sample_count"))?;
        if penalty.order >= basis.control_count() {
            return Err(invalid(
                scioxide_core::argument!("penalty"),
                "spline penalty order must be below the control count",
            ));
        }
        super::preparation::penalized_bytes::<S>(
            basis.control_count(),
            basis.degree(),
            sample_count,
            if penalty.lambda == S::ZERO {
                0
            } else {
                penalty.order
            },
        )
    }

    /// Assembles and factors the normal matrix on the calling thread.
    ///
    /// Equivalent to [`Self::build`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::build`].
    pub fn new(
        basis: BSplineBasis<S>,
        samples: Vec<S>,
        weights: Option<Vec<S>>,
        penalty: SplinePenalty<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            basis,
            samples,
            weights,
            penalty,
            ExecutionContext::sequential_native(),
        )
    }

    /// Assembles and factors the normal matrix within `scope`.
    ///
    /// Sample sites may be unsorted and repeated, and must lie in the basis
    /// domain. `None` gives unit weights. Explicit weights must be finite and
    /// nonnegative, with at least one positive. The banded Cholesky
    /// factorization counts one iteration per column in a child scope of
    /// `scope`, so a parent iteration limit bounds it; see
    /// [solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `samples` is empty.
    /// - `InvalidInput` if the penalty order is not below the number of basis
    ///   functions, a weight is negative (`OutOfRange`), or no weight is
    ///   positive.
    /// - `DimensionMismatch` if the number of weights differs from the number
    ///   of samples.
    /// - `NonFinite` if a sample site or weight is NaN or infinite.
    /// - `DomainError` (`OutsideDomain`) if a sample site is outside the basis
    ///   domain, and `DomainError` if the normal matrix is not positive definite.
    /// - `ExecutionStopped` or `ConvergenceFailure` if `scope` stops the
    ///   factorization or its iteration limit is reached.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn build<'scope, 'capability>(
        basis: BSplineBasis<S>,
        samples: Vec<S>,
        weights: Option<Vec<S>>,
        penalty: SplinePenalty<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        validate_count(samples.len(), 1, scioxide_core::argument!("samples"))?;
        let n = basis.control_count();
        if penalty.order >= n {
            return Err(invalid(
                scioxide_core::argument!("penalty"),
                "spline penalty order must be below the control count",
            ));
        }
        let weights = match weights {
            Some(values) => {
                validate_weights(
                    &values,
                    samples.len(),
                    Weights::Observed,
                    scioxide_core::argument!("weights"),
                )?;
                values
            }
            None => try_filled(samples.len(), S::ONE, scioxide_core::quantity!("weights"))?,
        };
        for &sample in &samples {
            if !sample.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "spline sample coordinates must be finite",
                ));
            }
            if !basis.domain().contains(sample) {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("samples"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "smoothing samples must lie in the spline base domain",
                ));
            }
        }
        // Lower band, column-major with the diagonal first: entry (row, column)
        // lives at column * (bandwidth + 1) + row - column.
        let degree = basis.degree();
        let active_order = if penalty.lambda == S::ZERO {
            0
        } else {
            penalty.order
        };
        let bandwidth = degree.max(active_order);
        let mut band = try_filled(
            checked_mul(n, bandwidth + 1, scioxide_core::quantity!("normal_band"))?,
            S::ZERO,
            scioxide_core::quantity!("normal_band"),
        )?;
        let mut row = try_filled(degree + 1, S::ZERO, scioxide_core::quantity!("basis_row"))?;
        for (&sample, &weight) in samples.iter().zip(&weights) {
            if weight == S::ZERO {
                continue;
            }
            let first = basis.local(sample, 0, KnotSide::Right, &mut row)?;
            for a in 0..=degree {
                if row[a] == S::ZERO {
                    continue;
                }
                for b in 0..=a {
                    if row[b] == S::ZERO {
                        continue;
                    }
                    let index = (first + b) * (bandwidth + 1) + a - b;
                    band[index] = add(band[index], mul(mul(weight, row[a])?, row[b])?)?;
                }
            }
        }
        let differences = if active_order == 0 {
            Vec::new()
        } else {
            difference_coefficients::<S>(active_order)?
        };
        // Each row of D_q holds signed binomials at start..=start + q, so
        // D_qᵀD_q is a sum of rank-one blocks.
        if penalty.lambda > S::ZERO {
            for start in 0..n - penalty.order {
                for a in 0..=penalty.order {
                    for b in 0..=a {
                        let index = (start + b) * (bandwidth + 1) + a - b;
                        band[index] = add(
                            band[index],
                            mul(mul(penalty.lambda, differences[a])?, differences[b])?,
                        )?;
                    }
                }
            }
        }
        let (low, high) = (0..n).fold((S::INFINITY, S::ZERO), |(low, high), column| {
            let diagonal = band[column * (bandwidth + 1)].abs();
            (low.min(diagonal), high.max(diagonal))
        });
        let diagonal_ratio = if high == S::ZERO { S::ZERO } else { low / high };
        let normal = HermitianBandMatrix::new(n, bandwidth, Triangle::Lower, band)?;
        let identity = SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "penalized-spline", 1),
            n as u128,
            0,
        );
        let factor = scope.into().run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |context| {
                let mut factor = BandedCholeskyDecomposition::empty();
                BandedCholeskyDecomposition::factor_into(
                    &normal,
                    &LinalgExecutionOptions::new(),
                    &mut factor,
                    &mut LinalgWorkspace::new(),
                    &mut *context,
                )?;
                Ok(factor)
            },
        )?;
        Ok(Self {
            basis,
            samples,
            weights,
            penalty,
            differences,
            normal,
            factor,
            diagonal_ratio,
        })
    }

    /// Returns the spline basis.
    pub const fn basis(&self) -> &BSplineBasis<S> {
        &self.basis
    }

    /// Returns the penalty.
    pub const fn penalty(&self) -> SplinePenalty<S> {
        self.penalty
    }

    /// Returns the ratio of the smallest to the largest absolute diagonal entry
    /// of the normal matrix.
    ///
    /// A small ratio warns of poor scaling. It is not an estimate of the
    /// condition number.
    pub const fn diagonal_ratio(&self) -> S {
        self.diagonal_ratio
    }

    /// Returns the workspace requirements for values of `dimension` real components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `dimension` is zero.
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<PenalizedSplineRequirements, SciOxideError> {
        if dimension == 0 {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("dimension"),
                "spline fit values need at least one real component",
            ));
        }
        let n = self.basis.control_count();
        Ok(PenalizedSplineRequirements {
            dimension,
            coefficients: checked_mul(n, dimension, scioxide_core::quantity!("coefficients"))?,
            row: self.basis.degree() + 1,
            linalg: BandedCholeskyDecomposition::<S>::workspace_requirements(
                &StructuredWorkspaceRequest::solve_matrix(
                    n,
                    dimension,
                    &LinalgExecutionOptions::new(),
                ),
            )?,
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
    ) -> Result<PenalizedSplineWorkspace<S>, SciOxideError> {
        let mut workspace = PenalizedSplineWorkspace {
            rhs: Vec::new(),
            solution: Vec::new(),
            row: Vec::new(),
            diagnostics: Vec::new(),
            linalg: LinalgWorkspace::new(),
        };
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Fits one value per sample and returns the spline with its diagnostics.
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
    ) -> Result<PenalizedSplineFit<S, V::Owned>, SciOxideError> {
        self.fit_scoped(values, ExecutionContext::sequential_native())
    }

    /// Owned fit within `scope`, for crate-internal adaptive fits.
    pub(crate) fn fit_scoped<'scope, 'capability, V: AffineValue<Scale = S>>(
        &self,
        values: &[V],
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PenalizedSplineFit<S, V::Owned>, SciOxideError> {
        let template = values
            .first()
            .ok_or_else(|| crate::error::shape(self.samples.len(), 0))?;
        let dimension = template.dimension()?;
        let mut workspace = self.workspace(dimension)?;
        let mut coefficients = try_filled(
            checked_mul(
                self.basis.control_count(),
                dimension,
                scioxide_core::quantity!("coefficients"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("coefficients"),
        )?;
        let report = self.fit_into(values, coefficients.as_mut_slice(), &mut workspace, scope)?;
        let mut diagnostics =
            try_with_capacity(dimension, scioxide_core::quantity!("diagnostics"))?;
        diagnostics.extend_from_slice(workspace.diagnostics());
        Ok(PenalizedSplineFit {
            spline: BSpline::from_parts(
                [self.basis.try_clone()?],
                coefficients,
                dimension,
                zeroed(template)?,
            )?,
            report,
            diagnostics,
        })
    }

    /// Fits one value per sample and writes the spline coefficients to `out`.
    ///
    /// `out` holds `control_count · m` real components for `m` components per
    /// value, in the order of [`BSpline::coefficients`]. The solve runs in a
    /// child scope of `scope`. The diagnostics of each component are left in
    /// the workspace, and the summary is returned.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of values differs from the number of
    ///   samples, the values differ in shape, `out` has the wrong shape or
    ///   `workspace` is too small.
    /// - `NonFinite` if a value is NaN or infinite, or a diagnostic overflows.
    /// - `Singular` if the banded solve fails.
    /// - `ExecutionStopped` or `ConvergenceFailure` if `scope` stops the solve
    ///   or its iteration limit is reached.
    pub fn fit_into<'scope, 'capability, V: AffineValue<Scale = S>, O>(
        &self,
        values: &[V],
        out: &mut O,
        workspace: &mut PenalizedSplineWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SplineFitReport<S>, SciOxideError>
    where
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        if values.len() != self.samples.len() {
            return Err(crate::error::shape(self.samples.len(), values.len()));
        }
        let dimension = values[0].dimension()?;
        for value in values {
            scioxide_core::check_value_shape(value, dimension)?;
            scioxide_core::validate_value(value)?;
        }
        let requirements = self.workspace_requirements(dimension)?;
        scioxide_core::check_value_shape(out, requirements.coefficients)?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(crate::error::shape(
                requirements.coefficients,
                workspace.rhs.len().min(workspace.solution.len()),
            ));
        }
        workspace.diagnostics.clear();
        let n = self.basis.control_count();
        let length = requirements.coefficients;
        let PenalizedSplineWorkspace {
            rhs,
            solution,
            row,
            diagnostics,
            linalg,
        } = workspace;
        let (rhs, solution) = (&mut rhs[..length], &mut solution[..length]);
        rhs.fill(S::ZERO);
        let degree = self.basis.degree();
        for ((&sample, &weight), value) in self.samples.iter().zip(&self.weights).zip(values) {
            if weight == S::ZERO {
                continue;
            }
            let first = self.basis.local(sample, 0, KnotSide::Right, row)?;
            for (offset, &basis) in row[..=degree].iter().enumerate() {
                let scale = mul(weight, basis)?;
                if scale == S::ZERO {
                    continue;
                }
                for c in 0..dimension {
                    let index = (first + offset) * dimension + c;
                    rhs[index] = add(rhs[index], mul(scale, value.component(c)?)?)?;
                }
            }
        }
        let identity = SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "penalized-spline-fit", 1),
            n as u128,
            dimension as u128,
        );
        let report = scope.into().run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |context| {
                self.factor.solve_matrix_into(
                    StructuredOperation::Normal,
                    MatrixView::from_slice(n, dimension, rhs)?,
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    MatrixViewMut::from_slice(n, dimension, solution)?,
                    linalg,
                    &mut *context,
                )
            },
        )?;
        if report.termination() != DirectSolveTermination::Accepted {
            return Err(SciOxideError::singular(None));
        }
        let summary = self.diagnostics(values, dimension, rhs, solution, row, diagnostics)?;
        scioxide_core::write_value(&*solution, out)?;
        Ok(summary)
    }

    /// Per-component diagnostics into `diagnostics` and their summary.
    fn diagnostics<V: AffineValue<Scale = S>>(
        &self,
        values: &[V],
        dimension: usize,
        rhs: &[S],
        solution: &[S],
        row: &mut [S],
        diagnostics: &mut Vec<SplineFitDiagnostics<S>>,
    ) -> Result<SplineFitReport<S>, SciOxideError> {
        diagnostics.resize(dimension, SplineFitDiagnostics::ZERO);
        let degree = self.basis.degree();
        for ((&sample, &weight), value) in self.samples.iter().zip(&self.weights).zip(values) {
            let first = self.basis.local(sample, 0, KnotSide::Right, row)?;
            for (c, diagnostic) in diagnostics.iter_mut().enumerate() {
                let mut predicted = S::ZERO;
                for (offset, &basis) in row[..=degree].iter().enumerate() {
                    predicted = add(
                        predicted,
                        mul(solution[(first + offset) * dimension + c], basis)?,
                    )?;
                }
                let residual = predicted - value.component(c)?;
                diagnostic.data_fidelity += weight * residual * residual;
            }
        }
        let n = self.basis.control_count();
        let order = self.penalty.order;
        let penalty_rows = if self.differences.is_empty() {
            0
        } else {
            n - order
        };
        let bandwidth = self.normal.view().bandwidth();
        let mut summary = SplineFitReport {
            data_fidelity: S::ZERO,
            penalty: S::ZERO,
            normal_residual: S::ZERO,
        };
        for (c, diagnostic) in diagnostics.iter_mut().enumerate() {
            for start in 0..penalty_rows {
                let mut difference = S::ZERO;
                for (k, &coefficient) in self.differences.iter().enumerate() {
                    difference += solution[(start + k) * dimension + c] * coefficient;
                }
                diagnostic.penalty += self.penalty.lambda * difference * difference;
            }
            // Componentwise backward error |(A c - b)_r| / (|b_r| + Σ_j |A_rj| |c_j|).
            for r in 0..n {
                let mut applied = S::ZERO;
                let mut scale = rhs[r * dimension + c].abs();
                for column in r.saturating_sub(bandwidth)..(r + bandwidth + 1).min(n) {
                    let coefficient = self.normal.view().get(r, column)?;
                    let x = solution[column * dimension + c];
                    applied += coefficient * x;
                    scale += coefficient.abs() * x.abs();
                }
                let residual = (applied - rhs[r * dimension + c]).abs();
                let normalized = if scale == S::ZERO {
                    S::ZERO
                } else {
                    residual / scale
                };
                diagnostic.normal_residual = diagnostic.normal_residual.max(normalized);
            }
            if !diagnostic.data_fidelity.is_finite()
                || !diagnostic.penalty.is_finite()
                || !diagnostic.normal_residual.is_finite()
            {
                diagnostics.clear();
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Output,
                    "spline fit diagnostics are not representable",
                ));
            }
            summary.data_fidelity += diagnostic.data_fidelity;
            summary.penalty += diagnostic.penalty;
            summary.normal_residual = summary.normal_residual.max(diagnostic.normal_residual);
        }
        if !summary.data_fidelity.is_finite() || !summary.penalty.is_finite() {
            diagnostics.clear();
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "spline fit summary is not representable",
            ));
        }
        Ok(summary)
    }
}

/// Returns the coefficients of the forward difference of order `q`.
///
/// Coefficient `k`, for `k = 0, …, q`, is `(−1)^(q−k) C(q, k)`, so that
/// `(Δ^q c)_i = Σ_k (−1)^(q−k) C(q, k) c_{i+k}`. These are the rows of the
/// difference matrix `D_q` in a [`SplinePenalty`]. The coefficients are
/// computed in the scalar type by the recurrence
/// `C(q, k + 1) = C(q, k) (q − k)/(k + 1)`, so large orders do not overflow an
/// integer type.
///
/// # Errors
///
/// - `SizeOverflow` or `AllocationFailure` if `q + 1` scalars cannot be
///   allocated.
/// - `NumericalInstability` (`Overflow`) if a coefficient overflows `S`.
///
/// # Examples
///
/// ```
/// assert_eq!(
///     scioxide_interp::difference_coefficients::<f64>(2)?,
///     [1., -2., 1.]
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn difference_coefficients<S: RealScalar>(order: usize) -> Result<Vec<S>, SciOxideError> {
    let n = scioxide_core::backend::checked_add(order, 1, scioxide_core::quantity!("differences"))?;
    let mut result = try_filled(n, S::ZERO, scioxide_core::quantity!("differences"))?;
    result[0] = if order.is_multiple_of(2) {
        S::ONE
    } else {
        -S::ONE
    };
    for k in 0..order {
        result[k + 1] = -result[k]
            * (scioxide_core::backend::scalar_from_u128_rounded::<S>((order - k) as u128)?
                / scioxide_core::backend::scalar_from_u128_rounded::<S>((k + 1) as u128)?);
        if !result[k + 1].is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "difference coefficient is not representable",
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn differences_are_signed_binomials() {
        assert_eq!(difference_coefficients::<f64>(1).unwrap(), [-1., 1.]);
        assert_eq!(
            difference_coefficients::<f64>(3).unwrap(),
            [-1., 3., -3., 1.]
        );
    }

    #[test]
    fn exact_data_and_workspace_contracts() {
        let basis = BSplineBasis::new(vec![0., 0., 1., 1.], 1).unwrap();
        let plan = PenalizedSplinePlan::new(
            basis,
            vec![0., 0.5, 1.],
            None,
            SplinePenalty::new(0., 1).unwrap(),
        )
        .unwrap();
        let fit = plan.fit(&[1.0_f64, 2.0, 3.0]).unwrap();
        assert!((fit.spline().coefficients()[0] - 1.0).abs() < 1e-15);
        assert!((fit.spline().coefficients()[1] - 3.0).abs() < 1e-15);
        assert!(fit.report().data_fidelity() < 1e-28);
        assert_eq!(fit.diagnostics().len(), 1);
        let mut small = plan.workspace(1).unwrap();
        let mut out = [7.0; 4];
        assert_eq!(
            plan.fit_into(
                &[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                &mut out,
                &mut small,
                ExecutionContext::sequential_native()
            )
            .unwrap_err()
            .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out, [7.0; 4]);
        assert!(plan.workspace_requirements(0).is_err());
    }
}

#[cfg(test)]
mod shared_differences {
    #[test]
    #[allow(clippy::unwrap_used)]
    fn coefficients_cover_zero_order_and_exceed_machine_integer_range() {
        use super::difference_coefficients;
        assert_eq!(difference_coefficients::<f64>(0).unwrap(), [1.]);
        let values = difference_coefficients::<f64>(70).unwrap();
        for (&a, &b) in values.iter().zip(values.iter().rev()) {
            assert!((a - b).abs() <= 128. * f64::EPSILON * a.abs());
        }
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(difference_coefficients::<f32>(200).is_err());
        assert!(difference_coefficients::<f64>(usize::MAX).is_err());
    }
}
