//! Continuation paths, options and per-point observations.
use super::*;

/// Selects how continuation moves along the solution curve.
///
/// Natural continuation fixes `λ` and solves `F(x, λ) = 0` for `x`; it fails at
/// a fold, where `∂λ/∂s = 0`. Pseudo-arclength continuation adds the
/// constraint `ṫ · D⁻² (y − ŷ) = 0`, where `ŷ` is the predicted point, `ṫ` the
/// previous unit tangent and `D` the diagonal of coordinate scales from
/// [`ContinuationOptions::with_scales`]. It passes through folds (Keller, 1977).
///
/// # References
///
/// - H. B. Keller, "Numerical solution of bifurcation and nonlinear eigenvalue
///   problems", in *Applications of Bifurcation Theory*, Academic Press, 1977.
/// - E. L. Allgower and K. Georg, *Introduction to Numerical Continuation
///   Methods*, SIAM, 2003.
#[derive(Debug, Clone, PartialEq)]
pub enum ContinuationPath<R: RealScalar> {
    /// Natural continuation through each listed parameter value in order.
    ///
    /// Each listed value is reached exactly. The list may change direction,
    /// for example `[1.0, 0.0]` goes up to 1 and back to 0.
    Parameters(Vec<R>),
    /// Pseudo-arclength continuation for this positive arclength, in the
    /// direction of the initial tangent.
    ///
    /// Arclength is measured in the metric of [`ContinuationOptions::with_scales`].
    PseudoArclength(R),
}
/// Sets step sizes, scaling, tolerances and limits for continuation.
///
/// Each step predicts along the tangent, corrects with a Newton–Krylov solve
/// configured by [`Self::with_corrector`], checks the original residual and the
/// local branch guards, and computes the new tangent. A rejected trial halves
/// the step, down to the minimum step. An accepted step grows by a factor 1.5
/// when the corrector took at most four iterations and both branch guards
/// (see [`Self::with_step_control`] and [`Self::with_correction_contraction`])
/// are within half their limits; otherwise the step size stays the same. Steps
/// never exceed the maximum step.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationOptions<R: RealScalar> {
    pub(crate) initial_step: R,
    pub(crate) minimum_step: R,
    pub(crate) maximum_step: R,
    pub(crate) max_points: Budget,
    pub(crate) max_rejections: Budget,
    pub(crate) state_scale: R,
    pub(crate) parameter_scale: R,
    pub(crate) residual: ResidualTolerance<R>,
    pub(crate) residual_scale: Option<R>,
    pub(crate) constraint_tolerance: StepTolerance<R>,
    pub(crate) corrector: NewtonKrylovOptions<R>,
    pub(crate) tangent_tolerance: ResidualTolerance<R>,
    pub(crate) tangent_iterations: Budget,
    pub(crate) tangent_restart: Budget,
    pub(crate) minimum_tangent_cosine: R,
    pub(crate) maximum_correction_ratio: R,
    pub(crate) maximum_correction_contraction: R,
    pub(crate) dense_diagnostics: bool,
    pub(crate) rank_tolerance: R,
    pub(crate) budgets: SolveBudgets,
    pub(crate) policy: scioxide_core::ExecutionPolicy,
    pub(crate) byte_limit: usize,
}
impl<R: RealScalar> ContinuationOptions<R> {
    /// Sets the work limits of the whole path, counted across all calls to
    /// [`Continuation::advance`] and [`Continuation::resume`].
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the work limits of the whole path.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }
    /// Sets the execution policy of the built-in tangent solver.
    pub const fn with_execution_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> scioxide_core::ExecutionPolicy {
        self.policy
    }
    /// Limits the workspace size in bytes; a larger requirement is a `ResourceLimit` error.
    pub const fn with_workspace_byte_limit(mut self, limit: usize) -> Self {
        self.byte_limit = limit;
        self
    }
    /// Returns the workspace size limit in bytes.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }

    /// Creates options with step sizes `minimum ≤ initial ≤ maximum` and room for
    /// `max_points` accepted points, including the corrected starting point.
    ///
    /// Defaults:
    ///
    /// - at most 12 rejected trials per accepted point;
    /// - residual test `‖F(y)‖₂ ≤ √ε · ‖F(ŷ)‖₂`, relative to the residual at
    ///   the predicted point of each correction, with no absolute part;
    /// - unit state and parameter scales;
    /// - arclength-constraint tolerance `√ε` absolute and relative;
    /// - branch guards: a tangent turn of at most 45°, a correction at most half
    ///   the predictor step, and a Jacobian change of at most 0.5;
    /// - corrector step tolerance `ε` absolute and relative;
    /// - tangent solve with relative tolerance `100ε`, 200 GMRES iterations and
    ///   restart 30.
    ///
    /// Here `ε` is the machine epsilon of `R`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a step size is not positive and finite, the step sizes
    ///   are out of order, or `max_points < 2`.
    pub fn new(
        initial: R,
        minimum: R,
        maximum: R,
        max_points: Budget,
    ) -> Result<Self, SciOxideError> {
        let positive = |v: R| v.is_finite() && v > R::ZERO;
        let argument = if max_points.get() < 2 {
            Some(scioxide_core::argument!("max_points"))
        } else if !positive(minimum) {
            Some(scioxide_core::argument!("minimum"))
        } else if !positive(maximum) {
            Some(scioxide_core::argument!("maximum"))
        } else if !positive(initial) || minimum > initial || initial > maximum {
            Some(scioxide_core::argument!("initial"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "invalid step limits or history capacity",
            ));
        }
        let residual = crate::termination::relative_tolerance();
        Ok(Self {
            budgets: SolveBudgets::unbounded(),
            policy: scioxide_core::ExecutionPolicy::Sequential,
            byte_limit: usize::MAX,
            initial_step: initial,
            minimum_step: minimum,
            maximum_step: maximum,
            max_points,
            max_rejections: Budget::ONE.saturating_add(11),
            state_scale: R::ONE,
            parameter_scale: R::ONE,
            residual,
            residual_scale: None,
            constraint_tolerance: StepTolerance::default(),
            corrector: NewtonKrylovOptions::default()
                .with_step_tolerance(StepTolerance::new(R::EPSILON, R::EPSILON)?),
            tangent_tolerance: ResidualTolerance::new(
                R::ZERO,
                R::EPSILON * R::try_from_f64_exact(100.)?,
            )?,
            tangent_iterations: Budget::ONE.saturating_add(199),
            tangent_restart: Budget::ONE.saturating_add(29),
            minimum_tangent_cosine: R::HALF.sqrt(),
            maximum_correction_ratio: R::HALF,
            maximum_correction_contraction: R::HALF,
            dense_diagnostics: false,
            rank_tolerance: R::SQRT_EPSILON,
        })
    }
    /// Sets the residual tolerance for accepting a point.
    ///
    /// The same tolerance is passed to the corrector, and continuation checks it
    /// again on a fresh evaluation of `F` at the corrected point.
    pub fn with_residual_tolerance(mut self, value: ResidualTolerance<R>) -> Self {
        self.residual = value;
        self.corrector = self.corrector.with_residual_tolerance(value);
        self
    }
    /// Sets a fixed reference `scale` for the relative residual test, instead of
    /// the residual norm at the predicted point.
    ///
    /// [`Continuation::new`] rejects a scale that is not positive and finite.
    pub fn with_residual_scale(mut self, scale: R) -> Self {
        self.corrector = self.corrector.with_residual_scale(scale);
        self.residual_scale = Some(scale);
        self
    }
    /// Sets the tolerance on the arclength constraint `ṫ · D⁻² (y − ŷ)`, in the
    /// metric of [`Self::with_scales`].
    pub fn with_constraint_tolerance(mut self, tolerance: StepTolerance<R>) -> Self {
        self.constraint_tolerance = tolerance;
        self
    }
    /// Sets the options of the Newton–Krylov corrector: forcing terms, iteration
    /// limits and derivative approximation.
    ///
    /// Continuation still checks its own residual tolerance on the corrected point.
    pub fn with_corrector(mut self, value: NewtonKrylovOptions<R>) -> Self {
        self.corrector = value;
        self
    }
    /// Sets the tolerance, iteration limit and restart length of the tangent solve.
    ///
    /// The tangent `t` solves the bordered system `[J; ṫᵀD⁻²] t = eₙ₊₁`, where `J`
    /// is the `n × (n + 1)` Jacobian of `F`. The defaults are relative tolerance
    /// `100ε`, 200 iterations and restart 30. The tolerance also applies to a
    /// solver from [`ContinuationProblem::with_linear_solver`], and continuation
    /// checks the residual of the bordered system itself. `restart` applies only
    /// to the built-in GMRES.
    ///
    /// ```
    /// use scioxide_core::{Budget, ResidualTolerance};
    /// use scioxide_rootfind::ContinuationOptions;
    /// let options = ContinuationOptions::new(0.1_f64, 1e-6, 0.2, Budget::try_from(64)?)?
    ///     .with_tangent_solver(
    ///         ResidualTolerance::new(0.0, 1e-12)?,
    ///         Budget::try_from(100)?,
    ///         Budget::try_from(20)?,
    ///     );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_tangent_solver(
        mut self,
        tolerance: ResidualTolerance<R>,
        iterations: Budget,
        restart: Budget,
    ) -> Self {
        self.tangent_tolerance = tolerance;
        self.tangent_iterations = iterations;
        self.tangent_restart = restart;
        self
    }
    /// Sets the branch guards that reject a step which may have jumped to another
    /// branch.
    ///
    /// A step is rejected when the cosine between successive tangents falls below
    /// `minimum_tangent_cosine`, or when the distance from the predicted to the
    /// corrected point exceeds `maximum_correction_ratio` times the predictor
    /// step. Both are measured in the metric of [`Self::with_scales`] and apply to
    /// natural and pseudo-arclength paths. The defaults are `cos 45°` and 0.5.
    /// These are local tests: they do not prove that the path stays on one branch.
    ///
    /// [`Continuation::new`] rejects values that are not in the open interval `(0, 1)`.
    ///
    /// ```
    /// use scioxide_core::Budget;
    /// use scioxide_rootfind::ContinuationOptions;
    /// let options = ContinuationOptions::new(0.1_f64, 1e-6, 0.2, Budget::try_from(64)?)?
    ///     .with_step_control(0.9, 0.25);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_step_control(
        mut self,
        minimum_tangent_cosine: R,
        maximum_correction_ratio: R,
    ) -> Self {
        self.minimum_tangent_cosine = minimum_tangent_cosine;
        self.maximum_correction_ratio = maximum_correction_ratio;
        self
    }
    /// Sets the limit on the change of the Jacobian along the correction.
    ///
    /// For the correction direction `d` from the predicted to the corrected point,
    /// a step is rejected unless `‖(J_new − J_old) d‖ ≤ maximum · ‖J_old d‖`, where
    /// `J_old` and `J_new` are the Jacobians of `F` at the previous and new points.
    /// This detects jumps between nearby branches with almost parallel tangents,
    /// which the tangent-angle guard cannot. It costs two Jacobian–vector products
    /// per trial, and none when the correction is zero. The default is 0.5.
    ///
    /// [`Continuation::new`] rejects values that are not in the open interval `(0, 1)`.
    ///
    /// ```
    /// use scioxide_core::Budget;
    /// use scioxide_rootfind::ContinuationOptions;
    /// let options = ContinuationOptions::new(0.1_f64, 1e-6, 0.2, Budget::try_from(64)?)?
    ///     .with_correction_contraction(0.25);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_correction_contraction(mut self, maximum: R) -> Self {
        self.maximum_correction_contraction = maximum;
        self
    }
    /// Sets the maximum number of rejected trial steps per accepted point (default 12).
    pub fn with_rejection_limit(mut self, value: Budget) -> Self {
        self.max_rejections = value;
        self
    }
    /// Sets the scales `D` of the state coordinates and of the parameter.
    ///
    /// Distances, tangents and the arclength constraint use the inner product
    /// `⟨u, v⟩ = Σᵢ uᵢ vᵢ / dᵢ²`, with `dᵢ = state` for the state coordinates and
    /// `dₙ₊₁ = parameter`. [`Continuation::new`] rejects scales that are not
    /// positive and finite.
    pub fn with_scales(mut self, state: R, parameter: R) -> Self {
        self.state_scale = state;
        self.parameter_scale = parameter;
        self
    }
    /// Computes a dense branch-point diagnostic at every accepted point.
    ///
    /// The diagnostic assembles the `n × (n + 1)` Jacobian from `n + 1`
    /// Jacobian–vector products, estimates its rank from singular values with
    /// relative threshold `threshold`, and takes the sign of the determinant of
    /// the bordered Jacobian by LU factorization. This costs O(n²) memory and
    /// O(n³) time per point; for large problems use
    /// [`ContinuationProblem::with_diagnostic`]. [`Continuation::new`] rejects
    /// a threshold that is not positive and finite.
    pub fn with_dense_diagnostics(mut self, threshold: R) -> Self {
        self.dense_diagnostics = true;
        self.rank_tolerance = threshold;
        self
    }
    pub(crate) fn scale(&self, i: usize, n: usize) -> R {
        if i == n {
            self.parameter_scale
        } else {
            self.state_scale
        }
    }
}
/// Describes one accepted point on the path.
///
/// The coordinates of the point are passed to the sink of
/// [`Continuation::advance`] and stored in [`ContinuationState`].
#[derive(Debug, Clone, Copy)]
pub struct ContinuationObservation<R: RealScalar> {
    /// Parameter at the accepted point.
    pub(crate) parameter: R,
    /// Sum of accepted pseudo-arclength steps; zero on natural paths.
    pub(crate) arclength: R,
    /// `‖F(y)‖₂` from a fresh evaluation at the accepted point.
    pub(crate) residual_norm: R,
    /// Corrector iterations for this point.
    pub(crate) corrector_iterations: usize,
    /// Rejected trial steps before this point.
    pub(crate) rejected_steps: usize,
    /// Whether the parameter component of the tangent changed sign.
    pub(crate) turning_point_candidate: bool,
    /// Branch-point diagnostic, when one was requested.
    pub(crate) diagnostic: Option<ContinuationDiagnostic<R>>,
}
impl<R: RealScalar> ContinuationObservation<R> {
    /// Returns the parameter `λ` of the accepted point.
    pub const fn parameter(&self) -> R {
        self.parameter
    }
    /// Returns the sum of accepted pseudo-arclength steps; zero on natural paths.
    pub const fn arclength(&self) -> R {
        self.arclength
    }
    /// Returns `‖F(y)‖₂`, evaluated afresh at the accepted point.
    pub const fn residual_norm(&self) -> R {
        self.residual_norm
    }
    /// Returns the number of corrector iterations for this point.
    pub const fn corrector_iterations(&self) -> usize {
        self.corrector_iterations
    }
    /// Returns the number of rejected trial steps before this point.
    pub const fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }
    /// Returns whether the parameter component of the tangent changed sign since
    /// the previous point, which indicates a fold (turning point) in between.
    pub const fn turning_point_candidate(&self) -> bool {
        self.turning_point_candidate
    }
    /// Returns the branch-point diagnostic, when one was requested.
    pub const fn diagnostic(&self) -> Option<ContinuationDiagnostic<R>> {
        self.diagnostic
    }
}
