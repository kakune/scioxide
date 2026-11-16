//! Series methods and options.

use scioxide_core::{Budget, ErrorTolerance, ExecutionPolicy, RealScalar};

/// Method used by a [`SeriesSummation`](crate::SeriesSummation) to estimate the sum of a
/// series.
///
/// Levin's u transform and Wynn's epsilon algorithm are nonlinear sequence
/// transformations: they extrapolate the partial sums `Sₙ` of the series from
/// the last `order + 1` terms, and stop when successive estimates agree. Their
/// error estimates are heuristic. Euler–Maclaurin summation adds an
/// approximation of the tail `Σₖ₌ₙ^∞ f(k)` to the partial sum, using an
/// integral and derivatives that you supply, and stops when a bound on the
/// error, valid under your assumptions, meets the tolerance.
///
/// Both accelerators work well for alternating series and for partial sums
/// that converge geometrically. Use Levin's u transform for series whose terms
/// decay like a power of `n`, such as `Σ 1/n²`; Wynn's epsilon algorithm
/// accelerates these only slowly.
///
/// # References
///
/// - D. Levin, "Development of non-linear transformations for improving
///   convergence of sequences", *Int. J. Comput. Math. B* 3 (1973), 371–388.
///   <https://doi.org/10.1080/00207167308803075>
/// - P. Wynn, "On a device for computing the eₘ(Sₙ) transformation", *Math.
///   Tables Aids Comput.* 10 (1956), 91–96. <https://doi.org/10.2307/2002183>
/// - E. J. Weniger, "Nonlinear sequence transformations for the acceleration of
///   convergence and the summation of divergent series", *Comput. Phys. Rep.*
///   10 (1989), 189–371. <https://doi.org/10.1016/0167-7977(89)90011-7>
/// - NIST DLMF, §2.10(i), Euler–Maclaurin formula. <https://dlmf.nist.gov/2.10#i>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesMethod {
    /// Levin's u transform with the remainder estimate `ωₙ = (n + 1) aₙ`,
    /// where `n` is the series index of the term `aₙ`.
    ///
    /// The transform starts at the first term with increasing order, then
    /// uses the last `order + 1` terms. Terms that are zero, or too small to
    /// change the rounded partial sum, are left out; the transform then uses
    /// divided differences over the remaining indices, so series with zero
    /// terms are accelerated on their nonzero terms.
    LevinU,
    /// Wynn's epsilon algorithm on the last `order + 1` partial sums.
    ///
    /// The estimate is the last entry of the highest even column of the
    /// epsilon table. The table is truncated at the last column whose
    /// differences are above rounding level. Terms that cannot change the
    /// rounded partial sum are left out, as for `LevinU`.
    WynnEpsilon,
    /// Euler–Maclaurin summation with `order` Bernoulli corrections, using
    /// tail information from an [`EulerMaclaurinProvider`](crate::EulerMaclaurinProvider).
    ///
    /// Each term costs `order + 3` provider calls: the term, the tail
    /// integral, `order` odd derivatives and one bound on an even derivative.
    EulerMaclaurin,
}

/// Limits, tolerance and execution policy of a [`SeriesSummation`](crate::SeriesSummation).
///
/// The defaults are at most 256 terms, `order = 8`, at most 256 terms per
/// call to [`SeriesSummation::advance`](crate::SeriesSummation::advance), the default
/// [`ErrorTolerance`] (`atol = rtol = √ε`, root-mean-square norm over the
/// components), sequential execution and no limit on provider calls. The
/// orders that a method accepts are checked by
/// [`SeriesSummation::new`](crate::SeriesSummation::new). `max_terms` does not change the
/// size of the state or workspace.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::SeriesOptions;
///
/// let options = SeriesOptions::<f64>::default().with_terms_per_advance(Budget::try_from(8)?);
/// assert_eq!(options.terms_per_advance().get(), 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SeriesOptions<S: RealScalar> {
    max_terms: Budget,
    order: Budget,
    terms_per_advance: Budget,
    tolerance: ErrorTolerance<S>,
    policy: ExecutionPolicy,
    max_evaluations: Option<Budget>,
}

impl<S: RealScalar> Default for SeriesOptions<S> {
    fn default() -> Self {
        Self {
            max_terms: Budget::ONE.saturating_add(255),
            order: Budget::ONE.saturating_add(7),
            terms_per_advance: Budget::ONE.saturating_add(255),
            tolerance: ErrorTolerance::default(),
            policy: ExecutionPolicy::Sequential,
            max_evaluations: None,
        }
    }
}

impl<S: RealScalar> SeriesOptions<S> {
    /// Sets the maximum number of terms summed by one state, over all calls.
    pub const fn with_max_terms(mut self, v: Budget) -> Self {
        self.max_terms = v;
        self
    }
    /// Sets the order: the number of terms beyond the first in a Levin or Wynn
    /// transform, or the number of Euler–Maclaurin corrections.
    ///
    /// [`SeriesSummation::new`](crate::SeriesSummation::new) accepts 1 to 32 for
    /// `LevinU`, 2 to 32 for `WynnEpsilon` and 1 to 12 for `EulerMaclaurin`.
    /// Higher orders of the accelerators are more ill-conditioned.
    pub const fn with_order(mut self, v: Budget) -> Self {
        self.order = v;
        self
    }
    /// Sets the maximum number of new terms summed by one call to
    /// [`SeriesSummation::advance`](crate::SeriesSummation::advance).
    pub const fn with_terms_per_advance(mut self, v: Budget) -> Self {
        self.terms_per_advance = v;
        self
    }
    /// Sets the tolerance on the error estimate of each component, relative to
    /// the magnitude of its estimate.
    pub fn with_tolerance(mut self, v: ErrorTolerance<S>) -> Self {
        self.tolerance = v;
        self
    }
    /// Sets the execution policy for evaluating terms.
    ///
    /// A parallel policy evaluates batches of at most 32 consecutive terms at
    /// once and sums them in index order, so the result is the same as with
    /// sequential execution. When the sum converges inside a batch, up to 31
    /// terms are evaluated but not used; they count toward the provider calls
    /// and the evaluation limit. `EulerMaclaurin` always evaluates
    /// sequentially.
    pub const fn with_execution_policy(mut self, v: ExecutionPolicy) -> Self {
        self.policy = v;
        self
    }
    /// Sets or removes the limit on provider calls over all calls on one state.
    ///
    /// A call that resumes a state applies the limit of its own plan to the
    /// number of calls made on that state so far.
    pub const fn with_max_evaluations(mut self, v: Option<Budget>) -> Self {
        self.max_evaluations = v;
        self
    }
    /// Returns the maximum number of terms summed by one state.
    pub const fn max_terms(&self) -> Budget {
        self.max_terms
    }
    /// Returns the maximum number of new terms summed by one call to `advance`.
    pub const fn terms_per_advance(&self) -> Budget {
        self.terms_per_advance
    }
    /// Returns the transform order or the number of Euler–Maclaurin corrections.
    pub const fn order(&self) -> Budget {
        self.order
    }
    /// Returns the error tolerance.
    pub const fn tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }
    /// Returns the execution policy for evaluating terms.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the limit on provider calls over all calls on one state, if any.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }
}
