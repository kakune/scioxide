//! Methods, diagnostics and reports of polynomial root solves.
use super::*;
/// Selects how [`crate::polynomial_roots_with_options`] generates root
/// candidates; set it with [`crate::PolynomialRootOptions::with_method`].
///
/// Every method ends with the same polishing and backward-error check, so the
/// choice affects cost and robustness, not the acceptance test. Use `Auto`
/// unless you have a reason to choose: companion eigenvalues are the most
/// reliable for moderate degree, and Aberth–Ehrlich iteration needs only O(n)
/// memory for high degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum PolynomialRootMethod {
    /// Uses `Companion` up to degree 64 and `Aberth` above. Above degree 64
    /// there is no companion fallback: if the iteration does not converge, the
    /// report is unsuccessful.
    #[default]
    Auto,
    /// Takes the eigenvalues of the companion matrix (monomial basis) or
    /// colleague matrix (orthogonal bases) as candidates and refines them by
    /// Aberth–Ehrlich iteration. Costs O(n³) time and O(n²) memory. Falls back
    /// to `Aberth` when the coefficients cannot be rescaled without loss, or
    /// when the eigenvalue stage fails or returns a spurious zero root.
    Companion,
    /// Iterates on all roots at once with Aberth–Ehrlich corrections
    /// `zᵢ ← zᵢ − wᵢ / (1 − wᵢ Σⱼ≠ᵢ 1/(zᵢ − zⱼ))`, `wᵢ = p(zᵢ)/p′(zᵢ)`,
    /// from Newton-polygon starting points. Each sweep costs O(n²) time;
    /// memory is O(n). Converges cubically to simple roots.
    Aberth,
    /// Iterates on all roots at once with Durand–Kerner (Weierstrass)
    /// corrections `zᵢ ← zᵢ − p(zᵢ) / (aₙ Πⱼ≠ᵢ (zᵢ − zⱼ))`. Each sweep costs
    /// O(n²) time; converges quadratically to simple roots. Monomial input only.
    DurandKerner,
    /// Finds one root at a time by complex Laguerre iteration with deflation,
    /// then refines all candidates on the undeflated polynomial by
    /// Aberth–Ehrlich iteration. Monomial input only.
    Laguerre,
}

/// Names the method that generated the candidates of a solve.
///
/// Reported by [`PolynomialRootReport::effective_method`]. It differs from
/// the requested [`PolynomialRootMethod`] when `Auto` chose a method, when a
/// monomial polynomial has degree one or two, or after a fallback.
///
/// ```
/// use scioxide_rootfind::{
///     PolynomialRootAlgorithm, PolynomialRootOptions, polynomial_roots_with_options,
/// };
///
/// // A monomial quadratic is solved by the closed-form formula.
/// let options = PolynomialRootOptions::default();
/// let roots = polynomial_roots_with_options(&[-1.0_f64, 0.0, 1.0], &options)?;
/// assert_eq!(
///     roots.report().effective_method(),
///     Some(PolynomialRootAlgorithm::Analytic)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialRootAlgorithm {
    /// Closed-form roots of a monomial polynomial of degree one or two, after
    /// exact zero roots are removed.
    Analytic,
    /// Companion or colleague matrix eigenvalues, refined by Aberth–Ehrlich
    /// iteration.
    Companion,
    /// Aberth–Ehrlich iteration from Newton-polygon starting points.
    Aberth,
    /// Durand–Kerner iteration.
    DurandKerner,
    /// Laguerre iteration with deflation, refined by Aberth–Ehrlich iteration.
    Laguerre,
}
impl PolynomialRootAlgorithm {
    pub(crate) const fn from_method(method: PolynomialRootMethod) -> Option<Self> {
        match method {
            PolynomialRootMethod::Auto => None,
            PolynomialRootMethod::Companion => Some(Self::Companion),
            PolynomialRootMethod::Aberth => Some(Self::Aberth),
            PolynomialRootMethod::DurandKerner => Some(Self::DurandKerner),
            PolynomialRootMethod::Laguerre => Some(Self::Laguerre),
        }
    }
}

/// Describes the workspace of a polynomial root solve for one degree and
/// method.
///
/// Returned by [`PolynomialRootWorkspace::workspace_requirements`] and
/// passed to [`PolynomialRootWorkspace::try_reserve`] and
/// [`PolynomialRootWorkspace::check`]. The element counts below exclude the
/// eigenvalue workspace of companion methods; [`Self::required_bytes`]
/// includes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolynomialRootWorkspaceRequirements<S: RealScalar> {
    pub(crate) scalar: core::marker::PhantomData<fn() -> S>,
    pub(crate) real_elements: usize,
    pub(crate) complex_elements: usize,
    pub(crate) companion_dimension: usize,
    pub(crate) diagnostic_records: usize,
    pub(crate) evaluation_records: usize,
    pub(crate) index_elements: usize,
    pub(crate) degree: usize,
    pub(crate) method: PolynomialRootMethod,
    pub(crate) limit: usize,
    pub(crate) policy: scioxide_core::ExecutionPolicy,
}
impl<S: RealScalar> PolynomialRootWorkspaceRequirements<S> {
    /// Returns the number of stored polynomial evaluations, one per root.
    pub const fn evaluation_records(self) -> usize {
        self.evaluation_records
    }
    /// Returns the number of per-root diagnostic records, two per root.
    pub const fn diagnostic_records(self) -> usize {
        self.diagnostic_records
    }
    /// Returns the number of Newton-polygon hull indices, `n + 1`.
    pub const fn index_elements(self) -> usize {
        self.index_elements
    }
    /// Returns the number of real scalars.
    pub const fn real_elements(self) -> usize {
        self.real_elements
    }
    /// Returns the number of complex scalars.
    pub const fn complex_elements(self) -> usize {
        self.complex_elements
    }
    /// Returns the dimension of the companion matrix, or zero for methods that
    /// do not form one.
    pub const fn companion_dimension(self) -> usize {
        self.companion_dimension
    }
}

/// Reports the backward error and conditioning of one returned root.
///
/// Diagnostics are computed from the original coefficients, not from the
/// rescaled or deflated polynomial used during iteration. They are aligned
/// with the roots: diagnostic `i` describes root `i`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialRootDiagnostic<S: RealScalar> {
    pub(crate) available: bool,
    pub(crate) backward_error: Option<S>,
    pub(crate) condition: scioxide_poly::RootEstimate<S>,
    pub(crate) validated: bool,
}
impl<S: RealScalar> PolynomialRootDiagnostic<S> {
    pub(crate) const fn unevaluated() -> Self {
        Self {
            available: false,
            backward_error: None,
            condition: scioxide_poly::RootEstimate::Unavailable,
            validated: false,
        }
    }
    /// Returns whether this slot holds a computed candidate.
    ///
    /// After an early stop, slots that no method has filled hold zero and
    /// report `false`.
    pub const fn is_available(self) -> bool {
        self.available
    }
    /// Returns the normalized backward error `η(z)` of this root, or `None` if
    /// it was not evaluated.
    ///
    /// The perturbation model is
    /// [`PolynomialRootReport::backward_error_model`]; the definition is on
    /// [`crate::polynomial_roots_with_options`].
    pub const fn backward_error(self) -> Option<S> {
        self.backward_error
    }
    /// Returns the first-order absolute condition number of this root under
    /// the same coefficient perturbation model.
    ///
    /// The forward error is about this value times the backward error. A
    /// large value indicates a multiple or clustered root but does not prove
    /// multiplicity.
    pub const fn condition(self) -> scioxide_poly::RootEstimate<S> {
        self.condition
    }
    /// Returns whether this root passed the backward-error tolerance.
    pub const fn is_validated(self) -> bool {
        self.validated
    }
}

/// Reports the outcome, conditioning and work of a polynomial root solve.
///
/// [`Self::termination`] is the outcome. It is
/// [`RootTermination::Complete`] when every root passed the backward-error
/// check, and one of these otherwise:
///
/// - [`RootTermination::Unverified`]: the iteration converged, but at least
///   one root failed the check.
/// - [`RootTermination::IterationLimit`]: no start converged within its
///   iteration limit.
/// - [`RootTermination::NumericalFailure`]: overflow or another numerical
///   breakdown stopped the solve.
/// - [`RootTermination::WorkLimit`], [`RootTermination::Cancelled`],
///   [`RootTermination::TimeLimit`]: a budget, cancellation or deadline
///   stopped the solve (see [Termination](crate#termination)).
///
/// The returned roots are then the last candidates; use the per-root
/// [`PolynomialRootDiagnostic`] to decide which are usable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialRootReport<S: RealScalar> {
    pub(crate) requested_method: PolynomialRootMethod,
    pub(crate) effective_method: Option<PolynomialRootAlgorithm>,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) algorithm: scioxide_core::AlgorithmId,
    pub(crate) active_root_updates: usize,
    pub(crate) pair_interactions: usize,
    pub(crate) restarts: usize,
    pub(crate) fallback_used: bool,
    pub(crate) max_residual: Option<S>,
    pub(crate) coefficient_scale: S,
    pub(crate) max_root_condition: scioxide_poly::RootEstimate<S>,
    pub(crate) effective_degree: usize,
    pub(crate) effective_rank: usize,
    pub(crate) canonicalization_changed: bool,
    pub(crate) status: RootTermination,
    pub(crate) policy: scioxide_core::Tolerance<S>,
    pub(crate) error_model: scioxide_poly::PolynomialBackwardErrorModel,
}

impl<S: RealScalar> PolynomialRootReport<S> {
    pub(crate) fn new(
        requested_method: PolynomialRootMethod,
        policy: scioxide_core::Tolerance<S>,
        coefficient_scale: S,
        degree: usize,
        algorithm: scioxide_core::AlgorithmId,
    ) -> Self {
        Self {
            requested_method,
            effective_method: None,
            counts: scioxide_core::SolveCounts::new(),
            algorithm,
            active_root_updates: 0,
            pair_interactions: 0,
            restarts: 0,
            fallback_used: false,
            max_residual: None,
            coefficient_scale,
            max_root_condition: scioxide_poly::RootEstimate::Unavailable,
            effective_degree: degree,
            effective_rank: degree,
            canonicalization_changed: false,
            status: RootTermination::Complete,
            policy,
            error_model: scioxide_poly::PolynomialBackwardErrorModel::Componentwise,
        }
    }

    /// Returns the coefficient perturbation model of the backward errors and
    /// condition numbers: componentwise for monomial input, normwise for
    /// other bases.
    pub const fn backward_error_model(self) -> scioxide_poly::PolynomialBackwardErrorModel {
        self.error_model
    }
    /// Returns the iterations and evaluations of this solve, excluding nested
    /// eigenvalue work.
    pub const fn counts(self) -> scioxide_core::EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this solve and of its nested eigenvalue solve.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the algorithm identifier.
    pub const fn algorithm(self) -> scioxide_core::AlgorithmId {
        self.algorithm
    }
    /// Returns how far the roots can be trusted: residual-checked after a
    /// successful solve, diagnostic only otherwise.
    pub const fn trust(self) -> scioxide_core::PartialResultTrust {
        if self.status.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        }
    }
    /// Returns the requested method.
    pub const fn requested_method(self) -> PolynomialRootMethod {
        self.requested_method
    }

    /// Returns the method that generated the candidates, or `None` if the
    /// solve stopped before any method ran.
    pub const fn effective_method(self) -> Option<PolynomialRootAlgorithm> {
        self.effective_method
    }

    /// Returns the number of iterations, including Newton polishing steps and
    /// excluding nested eigenvalue iterations.
    pub const fn iterations(self) -> usize {
        self.counts.own().iterations()
    }

    /// Returns the number of single-root corrections applied by the
    /// simultaneous iteration, over all sweeps and starts.
    pub const fn active_root_updates(self) -> usize {
        self.active_root_updates
    }

    /// Returns the number of root-pair terms `1/(zᵢ − zⱼ)` or `zᵢ − zⱼ`
    /// evaluated by the simultaneous iteration.
    pub const fn pair_interactions(self) -> usize {
        self.pair_interactions
    }

    /// Returns the number of restarts after the first start.
    pub const fn restarts(self) -> usize {
        self.restarts
    }

    /// Returns whether the solve switched to Aberth–Ehrlich iteration after
    /// the requested method failed or could not be applied.
    pub const fn fallback_used(self) -> bool {
        self.fallback_used
    }

    /// Returns the largest normalized backward error over all roots, or `None`
    /// if some root was not evaluated, for example after an early stop.
    pub const fn backward_error(self) -> Option<S> {
        self.max_residual
    }

    /// Returns `maxₖ |aₖ|` of the input coefficients.
    ///
    /// This is positive infinity when finite complex components have a modulus
    /// beyond the scalar range. Coefficient significance is still classified
    /// in extended exponent range.
    pub const fn coefficient_scale(self) -> S {
        self.coefficient_scale
    }

    /// Returns the largest first-order condition number over all roots.
    pub const fn max_root_condition(self) -> scioxide_poly::RootEstimate<S> {
        self.max_root_condition
    }

    /// Returns the highest index `k` with `aₖ` above
    /// [`crate::PolynomialRootOptions::with_rank_threshold`], or zero if there
    /// is none. Without a threshold, every nonzero coefficient counts.
    ///
    /// A value below the degree suggests that the leading coefficients are
    /// noise and that some roots are large and poorly determined. The number
    /// of roots is not changed.
    pub const fn effective_degree(self) -> usize {
        self.effective_degree
    }

    /// Returns the number of coefficients above the significance threshold.
    pub const fn effective_rank(self) -> usize {
        self.effective_rank
    }

    /// Returns whether making conjugate pairs exact or moving nearly real
    /// roots onto the real axis changed any root.
    pub const fn canonicalization_changed(self) -> bool {
        self.canonicalization_changed
    }

    /// Returns the outcome of the solve.
    pub const fn termination(self) -> RootTermination {
        self.status
    }

    /// Returns whether every root passed the backward-error check.
    pub const fn is_success(self) -> bool {
        self.status.is_success()
    }
    /// Returns the backward-error tolerance used to accept roots.
    pub const fn policy(self) -> scioxide_core::Tolerance<S> {
        self.policy
    }
}

/// Holds the roots, per-root diagnostics and report of
/// [`crate::polynomial_roots_with_options`].
///
/// Roots are sorted by real part, then by imaginary part. The type
/// dereferences to `[Complex<S>]`, so `roots[i]` and `roots.iter()` work
/// directly.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PolynomialRoots<S: RealScalar> {
    pub(crate) roots: Vec<Complex<S>>,
    pub(crate) report: PolynomialRootReport<S>,
    pub(crate) diagnostics: Vec<PolynomialRootDiagnostic<S>>,
}

impl<S: RealScalar> PolynomialRoots<S> {
    /// Returns the roots in ascending order of real part, then imaginary part.
    pub fn roots(&self) -> &[Complex<S>] {
        &self.roots
    }

    /// Returns the diagnostics, aligned with [`Self::roots`].
    pub fn diagnostics(&self) -> &[PolynomialRootDiagnostic<S>] {
        &self.diagnostics
    }

    /// Returns the solve report.
    pub const fn report(&self) -> PolynomialRootReport<S> {
        self.report
    }

    /// Returns the roots, diagnostics and report by value.
    pub fn into_parts(
        self,
    ) -> (
        Vec<Complex<S>>,
        Vec<PolynomialRootDiagnostic<S>>,
        PolynomialRootReport<S>,
    ) {
        (self.roots, self.diagnostics, self.report)
    }
}

impl<S: RealScalar> Deref for PolynomialRoots<S> {
    type Target = [Complex<S>];

    fn deref(&self) -> &Self::Target {
        &self.roots
    }
}

impl<S: RealScalar> IntoIterator for PolynomialRoots<S> {
    type Item = Complex<S>;
    type IntoIter = alloc::vec::IntoIter<Complex<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.roots.into_iter()
    }
}

impl<'a, S: RealScalar> IntoIterator for &'a PolynomialRoots<S> {
    type Item = &'a Complex<S>;
    type IntoIter = core::slice::Iter<'a, Complex<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.roots.iter()
    }
}
