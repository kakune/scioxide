//! Armijo backtracking line search.

#[cfg(test)]
use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

#[cfg(test)]
use crate::LineSearchFailure;
use crate::LineSearchResult;
use crate::traits::{LineSearch, ObjectiveFunction};

/// Finds a step length that satisfies the Armijo condition by backtracking.
///
/// Starting from the initial step `α₀`, the search multiplies the step by the
/// contraction factor `ρ` until the sufficient-decrease (Armijo) condition
///
/// ```text
/// f(x + α d) ≤ f(x) + c₁ α ∇f(x)ᵀd
/// ```
///
/// holds, where `x` is the current point and `d` a descent direction. A trial
/// point outside the domain of the objective, or with a non-finite value, is
/// rejected and the step contracts. The search stops unsuccessfully when the
/// next step would fall below the minimum step or the trial limit is reached.
/// Each trial costs one objective evaluation.
///
/// The default uses `α₀ = 1`, minimum step `ε` (the machine epsilon),
/// `c₁ = 10⁻⁴`, `ρ = 0.5` and at most 40 trials.
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §3.1 (Algorithm 3.1).
#[derive(Debug, Clone, Copy)]
pub struct BacktrackingLineSearch<S: RealScalar> {
    alpha_init: S,
    alpha_min: S,
    c1: S,
    roundoff_units: S,
    rho: S,
    max_iterations: scioxide_core::Budget,
}

impl<S: RealScalar> Default for BacktrackingLineSearch<S> {
    fn default() -> Self {
        let one = S::ONE;
        Self {
            alpha_init: one,
            alpha_min: <S as RealScalar>::EPSILON,
            c1: crate::globalization::sufficient_decrease(),
            roundoff_units: S::ZERO,
            rho: S::HALF,
            max_iterations: crate::globalization::TRIALS,
        }
    }
}

impl<S: RealScalar> BacktrackingLineSearch<S> {
    /// Creates a backtracking search with initial step `alpha_init`, minimum
    /// step `alpha_min`, Armijo parameter `c1`, contraction factor `rho` and
    /// at most `max_iterations` trials.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `InvalidInput` unless `0 < alpha_min ≤ alpha_init`, `0 < c1 < 1` and
    ///   `0 < rho < 1`.
    pub fn new(
        alpha_init: S,
        alpha_min: S,
        c1: S,
        rho: S,
        max_iterations: scioxide_core::Budget,
    ) -> Result<Self, SciOxideError> {
        let zero = S::ZERO;
        let one = S::ONE;
        for (name, value) in [
            ("alpha_init", alpha_init),
            ("alpha_min", alpha_min),
            ("c1", c1),
            ("rho", rho),
        ] {
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    alloc::format!("{name} must be finite"),
                ));
            }
        }
        if alpha_init <= zero || alpha_min <= zero || alpha_min > alpha_init {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("alpha_min"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "line search requires 0 < alpha_min <= alpha_init",
            ));
        }
        if c1 <= zero || c1 >= one {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("c1"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Armijo parameter c1 must satisfy 0 < c1 < 1",
            ));
        }
        if rho <= zero || rho >= one {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("rho"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "backtracking contraction rho must satisfy 0 < rho < 1",
            ));
        }
        Ok(Self {
            alpha_init,
            alpha_min,
            c1,
            roundoff_units: S::ZERO,
            rho,
            max_iterations,
        })
    }

    // L-BFGS-B uses the same objective-roundoff allowance as strong Wolfe.
    // Gradient stopping remains independent of objective representability.
    pub(crate) fn with_roundoff_allowance(mut self) -> Self {
        self.roundoff_units = S::TWO * S::TWO * S::TWO;
        self
    }

    /// Returns the initial step `α₀`.
    #[must_use]
    pub const fn initial_step(&self) -> S {
        self.alpha_init
    }

    /// Returns the minimum step.
    #[must_use]
    pub const fn minimum_step(&self) -> S {
        self.alpha_min
    }

    /// Returns the sufficient-decrease parameter `c₁`.
    #[must_use]
    pub const fn sufficient_decrease(&self) -> S {
        self.c1
    }

    /// Returns the contraction factor `ρ`.
    #[must_use]
    pub const fn contraction_factor(&self) -> S {
        self.rho
    }

    /// Returns the maximum number of trials.
    pub const fn max_iterations(&self) -> scioxide_core::Budget {
        self.max_iterations
    }

    #[allow(clippy::too_many_arguments)]
    fn search_into_kernel<F>(
        &self,
        f: &F,
        x: &[S],
        f_x: S,
        direction: &[S],
        grad_dot_dir: S,
        trial_point: &mut [S],
        scope: crate::LineSearchScope<'_>,
        initial_step: Option<S>,
    ) -> Result<LineSearchResult<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
        F: ObjectiveFunction<S> + ?Sized,
    {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if x.len() != f.dim() || x.len() != direction.len() || trial_point.len() != x.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(x.len()),
                scioxide_core::Dimensions::one(direction.len().min(trial_point.len())),
            ));
        }
        crate::utils::validate_finite_origin(x, scioxide_core::NonFiniteOrigin::Input)?;
        crate::utils::validate_finite_origin(direction, scioxide_core::NonFiniteOrigin::Input)?;
        if !grad_dot_dir.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "grad_dot_dir is non-finite",
            ));
        }
        if grad_dot_dir >= zero {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("grad_dot_dir"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "search direction is not a descent direction",
            ));
        }
        if !f_x.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "f_x must be finite",
            ));
        }

        let alpha_init =
            crate::line_search::initial_step(initial_step, self.alpha_init, self.alpha_min)?;
        let initial = scope.counts()?;
        let mut best = None;
        let outcome = crate::globalization::backtrack(
            alpha_init,
            self.alpha_min,
            self.rho,
            self.max_iterations,
            |_, alpha| {
                scope.propose()?;
                // An unrepresentable point, a point outside the domain and a
                // non-finite value all reject the trial and contract the step.
                if !crate::utils::fill_trial(trial_point, x, alpha, direction)?
                    || !f.is_in_domain(trial_point)?
                {
                    scope.reject()?;
                    return Ok(None);
                }
                let value = f.eval(trial_point)?;
                if !value.is_finite() {
                    scope.reject()?;
                    return Ok(None);
                }
                if crate::utils::armijo_satisfied(
                    value,
                    f_x,
                    self.c1,
                    alpha,
                    grad_dot_dir,
                    self.roundoff_units,
                )? {
                    return Ok(Some((alpha, value)));
                }
                if best.is_none_or(|(_, previous)| value < previous) {
                    best = Some((alpha, value));
                }
                scope.reject()?;
                Ok(None)
            },
        )?;
        let (fun, grad, rejected) = scope.statistics_since(initial)?;
        let result = match outcome {
            Ok((alpha, value)) => LineSearchResult::accepted(alpha, value, false, None, fun, grad),
            Err(reason) => {
                if let Some((step, _)) = best {
                    for (candidate, (&current, &component)) in
                        trial_point.iter_mut().zip(x.iter().zip(direction))
                    {
                        *candidate = crate::utils::trial_component(current, step, component)?;
                    }
                }
                LineSearchResult::unsuccessful(reason, best, fun, grad)
            }
        };
        Ok(result.with_rejected_trials(rejected))
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LineSearch<S> for BacktrackingLineSearch<S> {
    fn search_into<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x: &[S],
        f_x: S,
        direction: &[S],
        grad_dot_dir: S,
        derivative_strategy: scioxide_differentiate::DerivativeStrategy,
        trial_point: &mut [S],
        trial_gradient: &mut [S],
        candidate_point: &mut [S],
        candidate_gradient: &mut [S],
        finite_difference: &mut crate::DerivativeWorkspace<S>,
        scope: crate::LineSearchScope<'_>,
        initial_step: Option<S>,
    ) -> Result<LineSearchResult<S>, SciOxideError> {
        if let Some(execution) = scope.standalone() {
            return crate::line_search::standalone(self, x.len(), execution, |session| {
                self.search_into(
                    &session.observe(f) as &dyn ObjectiveFunction<S>,
                    x,
                    f_x,
                    direction,
                    grad_dot_dir,
                    derivative_strategy,
                    trial_point,
                    trial_gradient,
                    candidate_point,
                    candidate_gradient,
                    finite_difference,
                    crate::LineSearchScope::observed(session),
                    initial_step,
                )
            });
        }

        if x.len() != f.dim()
            || direction.len() != x.len()
            || trial_point.len() != x.len()
            || trial_gradient.len() != x.len()
            || candidate_point.len() != x.len()
            || candidate_gradient.len() != x.len()
            || finite_difference.input_dimension() != x.len()
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(f.dim()),
                scioxide_core::Dimensions::one(x.len()),
            ));
        }
        let result = self.search_into_kernel(
            f,
            x,
            f_x,
            direction,
            grad_dot_dir,
            candidate_point,
            scope,
            initial_step,
        )?;
        if result.has_trial() {
            trial_point.copy_from_slice(candidate_point);
        }
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::LineSearchTermination;
    use crate::line_search::testing::{Searched, Standalone};

    struct Quadratic;
    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(true)
        }
    }

    #[test]
    fn observed_search_admits_each_trial_and_never_accepts_a_denied_proposal() {
        use scioxide_core::{Budget, BudgetKind, ExecutionContext, SolveBudgets, SolveState};
        let search = BacktrackingLineSearch::<f64>::default();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::ONE);
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                crate::context::identity(
                    scioxide_core::algorithm_id!("optimize", "Test", 1),
                    scioxide_core::ProblemToken::new(105),
                    2,
                    0,
                )
                .unwrap(),
                &mut budgets,
            )
            .unwrap();
        let session =
            crate::context::SolveSession::new(&mut context, scioxide_core::argument!("context"));
        let (mut trial, mut gradient, mut candidate, mut candidate_gradient) =
            ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
        let mut derivatives = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Gradient,
            2,
            1,
        )
        .unwrap();
        let outcome = search.search_into(
            &session.observe(&Quadratic),
            &[1.0, 1.0],
            2.0,
            &[-2.0, -2.0],
            -8.0,
            scioxide_differentiate::DerivativeStrategy::default(),
            &mut trial,
            &mut gradient,
            &mut candidate,
            &mut candidate_gradient,
            &mut derivatives,
            crate::LineSearchScope::observed(&session),
            None,
        );
        assert!(matches!(
            outcome.as_ref().map_err(SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::ConvergenceFailure { .. })
        ));
        assert_eq!(trial, [0.0; 2]);
        let counts = context.counts().own();
        assert_eq!(counts.proposals(), 1);
        assert_eq!(counts.rejected(), 1);
        assert_eq!(counts.accepted(), 0);
        assert_eq!(counts.provider(scioxide_core::EvaluationRole::Objective), 1);
    }

    fn accepted_step(result: Searched<f64>) -> f64 {
        result.step().unwrap()
    }

    #[test]
    fn test_backtracking_accepts_full_step() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let x = [2.0, 2.0];
        let grad = [4.0, 4.0];
        let direction = [-4.0, -4.0];
        let grad_dot_dir = crate::utils::dot(&grad, &direction).unwrap();
        let result = ls
            .search(
                &Quadratic,
                &x,
                Quadratic.eval(&x).unwrap(),
                &direction,
                grad_dot_dir,
            )
            .unwrap();
        let alpha = accepted_step(result);
        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }

    #[test]
    fn test_backtracking_decreases_function() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let x = [2.0, 2.0];
        let grad = [4.0, 4.0];
        let direction = [-4.0, -4.0];
        let grad_dot_dir = crate::utils::dot(&grad, &direction).unwrap();
        let result = ls
            .search(
                &Quadratic,
                &x,
                Quadratic.eval(&x).unwrap(),
                &direction,
                grad_dot_dir,
            )
            .unwrap();
        let alpha = accepted_step(result);
        let x_new: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_old = Quadratic.eval(&x).unwrap();
        let f_new = Quadratic.eval(&x_new).unwrap();
        assert!(f_new < f_old);
    }

    #[test]
    fn accepted_trial_reports_exact_evaluations_and_cached_value() {
        struct OneDimensionalQuadratic;

        impl ObjectiveFunction<f64> for OneDimensionalQuadratic {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }
        }

        let result = BacktrackingLineSearch::<f64>::default()
            .search(&OneDimensionalQuadratic, &[1.0], 1.0, &[-2.0], -4.0)
            .unwrap();
        assert_eq!(result.step(), Some(0.5));
        assert_eq!(result.value(), Some(0.0));
        assert!(result.gradient().is_none());
        assert_eq!(result.directional_derivative(), None);
        assert_eq!(result.fun_evals(), 2);
        assert_eq!(result.grad_evals(), 0);
    }

    #[test]
    fn unsuccessful_search_retains_best_trial_only_as_diagnostics() {
        let line_search =
            BacktrackingLineSearch::new(1.0, f64::EPSILON, 1e-4, 0.5, scioxide_core::Budget::ONE)
                .unwrap();
        let result = line_search
            .search(&Quadratic, &[1.0, 1.0], 2.0, &[-2.0, -2.0], -8.0)
            .unwrap();

        assert!(!result.is_success());
        assert!(result.has_trial());
        assert_eq!(result.step(), Some(1.0));
        assert_eq!(result.fun_evals(), 1);
        assert!(matches!(
            result.termination(),
            LineSearchTermination::Unsuccessful(LineSearchFailure::IterationLimit)
        ));
    }

    fn objective_roundoff<S: RealScalar + NumericScalar<Real = S>>() {
        struct Offset;
        impl<S: RealScalar + NumericScalar<Real = S>> ObjectiveFunction<S> for Offset {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
                // Mathematically 2 + (x-1)^2; cancellation can make its
                // rounded value at a nonminimum one ulp smaller than at x=1.
                Ok((S::TWO + x[0] * x[0]) - S::TWO * x[0] + S::ONE)
            }
        }
        let x = S::ONE - S::EPSILON.sqrt() * S::try_from_f64_lossy(0.497).unwrap();
        let direction = S::ONE - x;
        let initial = Offset.eval(&[x]).unwrap();
        assert!(initial < S::TWO);
        let slope = -S::TWO * direction * direction;
        let strict = BacktrackingLineSearch::<S>::default();
        let rounded = strict.with_roundoff_allowance();
        let strict_result = strict
            .search(&Offset, &[x], initial, &[direction], slope)
            .unwrap();
        assert!(!strict_result.is_success() || strict_result.step() != Some(S::ONE));
        let result = rounded
            .search(&Offset, &[x], initial, &[direction], slope)
            .unwrap();
        assert!(result.is_success());
        assert_eq!(result.step(), Some(S::ONE));
        assert_eq!(result.value(), Some(S::TWO));
        assert_eq!(result.fun_evals(), 1);
        // A resolved increase still fails with the allowance enabled.
        let initial = Offset.eval(&[S::HALF]).unwrap();
        let result = rounded
            .search(&Offset, &[S::HALF], initial, &[S::TWO], -S::TWO)
            .unwrap();
        assert!(result.step().is_none_or(|step| step < S::ONE));
    }

    #[test]
    fn objective_roundoff_allowance_f32_and_f64() {
        objective_roundoff::<f32>();
        objective_roundoff::<f64>();
    }

    #[test]
    fn test_backtracking_default_params() {
        let ls = BacktrackingLineSearch::<f64>::default();
        assert!((ls.contraction_factor() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_backtracking_rejects_invalid_rho() {
        let result = BacktrackingLineSearch::new(
            1.0,
            f64::EPSILON,
            1e-4,
            1.5,
            scioxide_core::Budget::try_from(40).unwrap(),
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));

        let result = BacktrackingLineSearch::new(
            1.0,
            f64::EPSILON,
            1e-4,
            f64::NAN,
            scioxide_core::Budget::try_from(40).unwrap(),
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            })
        ));
    }

    #[test]
    fn test_backtracking_dimension_mismatch() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let result = ls.search(&Quadratic, &[1.0, 1.0], 2.0, &[-1.0], -2.0);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_backtracking_rejects_non_descent() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let result = ls.search(&Quadratic, &[1.0, 1.0], 2.0, &[1.0, 1.0], 4.0);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_backtracking_rejects_nan_grad_dot_dir() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let result = ls.search(&Quadratic, &[1.0, 1.0], 2.0, &[-1.0, -1.0], f64::NAN);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            })
        ));
    }

    #[test]
    fn test_backtracking_rejects_inf_grad_dot_dir() {
        let ls = BacktrackingLineSearch::<f64>::default();
        let result = ls.search(
            &Quadratic,
            &[1.0, 1.0],
            2.0,
            &[-1.0, -1.0],
            f64::NEG_INFINITY,
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            })
        ));
    }

    #[test]
    fn test_backtracking_rejects_invalid_alpha_bounds() {
        let budget = scioxide_core::Budget::try_from(40).unwrap();
        let result = BacktrackingLineSearch::new(0.0, f64::EPSILON, 1e-4, 0.5, budget);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));

        let result = BacktrackingLineSearch::new(1.0, f64::EPSILON, f64::NAN, 0.5, budget);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            })
        ));

        let result = BacktrackingLineSearch::new(1.0, 2.0, 1e-4, 0.5, budget);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }
}
