//! Optimality tolerances of the constrained, programming and verification solvers.

use scioxide_core::{GradientTolerance, RealScalar, ResidualTolerance, SciOxideError, Tolerance};

/// Holds separate tolerances for each part of a KKT optimality check.
///
/// Each tolerance has the form `absolute + relative · scale`, where the scale
/// is a magnitude of the problem. Feasibility is measured in the units of the
/// constraints, stationarity in the units of the gradient, complementarity in
/// the units of the objective, and rank in the units of the matrix
/// coefficients. [`Self::new`] sets four tolerances; the dual-feasibility,
/// pivot, reduced-cost and gap tolerances default to the feasibility, rank,
/// gradient and complementarity tolerances, and each can be set separately.
/// The [`Default`] uses the default core tolerance for every part. The linear
/// and quadratic programming solvers default to `10⁻⁸` absolute and relative,
/// or `10⁻⁶` when the machine epsilon of `S` exceeds `10⁻⁸`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimalityTolerances<S: RealScalar> {
    feasibility: scioxide_core::FeasibilityTolerance<S>,
    gradient: GradientTolerance<S>,
    complementarity: ResidualTolerance<S>,
    rank: scioxide_core::RankThreshold<S>,
    dual_feasibility: scioxide_core::FeasibilityTolerance<S>,
    pivot: ResidualTolerance<S>,
    reduced_cost: GradientTolerance<S>,
    gap: ResidualTolerance<S>,
}

/// Reference magnitudes of the normalized problem, one per tolerance.
pub(crate) struct OptimalityScales<S> {
    pub(crate) primal: S,
    pub(crate) gradient: S,
    pub(crate) complementarity: S,
    pub(crate) dual: S,
}

impl<S: RealScalar> OptimalityTolerances<S> {
    /// Creates the tolerances for feasibility, stationarity, complementarity and
    /// rank.
    pub const fn new(
        feasibility: scioxide_core::FeasibilityTolerance<S>,
        gradient: GradientTolerance<S>,
        complementarity: ResidualTolerance<S>,
        rank: scioxide_core::RankThreshold<S>,
    ) -> Self {
        Self {
            dual_feasibility: feasibility,
            pivot: ResidualTolerance::from_tolerance(rank.tolerance()),
            reduced_cost: gradient,
            gap: complementarity,
            feasibility,
            gradient,
            complementarity,
            rank,
        }
    }

    /// Returns the primal-feasibility tolerance.
    pub const fn feasibility(self) -> scioxide_core::FeasibilityTolerance<S> {
        self.feasibility
    }
    /// Returns the tolerance for the gradient of the Lagrangian.
    pub const fn gradient(self) -> GradientTolerance<S> {
        self.gradient
    }
    /// Returns the complementarity tolerance, in the units of the objective.
    pub const fn complementarity(self) -> ResidualTolerance<S> {
        self.complementarity
    }
    /// Returns the numerical-rank threshold.
    pub const fn rank(self) -> scioxide_core::RankThreshold<S> {
        self.rank
    }

    /// Sets the tolerance for the sign of inequality multipliers (dual
    /// feasibility).
    pub const fn with_dual_feasibility(
        mut self,
        tolerance: scioxide_core::FeasibilityTolerance<S>,
    ) -> Self {
        self.dual_feasibility = tolerance;
        self
    }
    /// Sets the smallest pivot magnitude the simplex method accepts.
    pub const fn with_pivot(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.pivot = tolerance;
        self
    }
    /// Sets the reduced-cost tolerance at which the simplex method stops.
    pub const fn with_reduced_cost(mut self, tolerance: GradientTolerance<S>) -> Self {
        self.reduced_cost = tolerance;
        self
    }
    /// Sets the tolerance for the duality gap of the objective and for the
    /// strict margin of an infeasibility or unboundedness certificate.
    pub const fn with_gap(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.gap = tolerance;
        self
    }
    /// Returns the dual-feasibility tolerance.
    pub const fn dual_feasibility(self) -> scioxide_core::FeasibilityTolerance<S> {
        self.dual_feasibility
    }
    /// Returns the pivot tolerance.
    pub const fn pivot(self) -> ResidualTolerance<S> {
        self.pivot
    }
    /// Returns the reduced-cost tolerance.
    pub const fn reduced_cost(self) -> GradientTolerance<S> {
        self.reduced_cost
    }
    /// Returns the gap tolerance.
    pub const fn gap(self) -> ResidualTolerance<S> {
        self.gap
    }

    pub(crate) fn programming_default() -> Self {
        let tolerance = Tolerance::from_decimal_places(8, 8);
        let tolerance = if S::EPSILON > tolerance.absolute() {
            Tolerance::from_decimal_places(6, 6)
        } else {
            tolerance
        };
        Self::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(tolerance),
            GradientTolerance::from_tolerance(tolerance),
            ResidualTolerance::from_tolerance(tolerance),
            scioxide_core::RankThreshold::from_tolerance(tolerance),
        )
    }

    pub(crate) fn satisfied(
        self,
        diagnostics: crate::OptimalityDiagnostics<S>,
        scales: OptimalityScales<S>,
    ) -> Result<bool, SciOxideError> {
        let feasible = self.feasibility.threshold(scales.primal)?;
        let gradient = self.gradient.threshold(scales.gradient)?;
        let complementarity = self.complementarity.threshold(scales.complementarity)?;
        let dual = self.dual_feasibility.threshold(scales.dual)?;
        Ok(diagnostics
            .primal_feasibility()
            .is_none_or(|r| r <= feasible)
            && diagnostics.bound_violation().is_none_or(|r| r <= feasible)
            && diagnostics.dual_feasibility().is_none_or(|r| r <= dual)
            && diagnostics.stationarity().is_some_and(|r| r <= gradient)
            && diagnostics
                .complementarity()
                .is_none_or(|r| r <= complementarity))
    }
}

impl<S: RealScalar> Default for OptimalityTolerances<S> {
    fn default() -> Self {
        Self::new(
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
