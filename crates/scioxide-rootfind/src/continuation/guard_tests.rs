//! Branch guards, tangent accuracy and progressive failure regressions.
use core::cell::Cell;

use super::*;
use scioxide_linalg::IterativeSolveReport;

fn r<R: RealScalar>(value: f64) -> R {
    R::try_from_f64_lossy(value).unwrap()
}
fn budget(value: usize) -> Budget {
    Budget::try_from(value).unwrap()
}
struct Hyperbola {
    stable_residual: bool,
}
impl<R: RealScalar> ResidualOperator<R> for Hyperbola {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[R], out: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        out[0] = if self.stable_residual {
            x[0].abs() - (x[1] * x[1] + r(1. / 2048.)).sqrt()
        } else {
            x[0] * x[0] - x[1] * x[1] - r(1. / 2048.)
        };
        Ok(())
    }
}
impl<R: RealScalar> JacobianOperator<R> for Hyperbola {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[R], v: &[R], out: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        out[0] = if self.stable_residual {
            x[0].signum() * v[0] - x[1] / (x[1] * x[1] + r(1. / 2048.)).sqrt() * v[1]
        } else {
            R::TWO * (x[0] * v[0] - x[1] * v[1])
        };
        Ok(())
    }
}

fn close_branches<R: RealScalar>(stable_residual: bool) {
    let curve = Hyperbola { stable_residual };
    let problem = ContinuationProblem::new(&curve).unwrap();
    for sign in [-R::ONE, R::ONE] {
        for maximum in [r(0.15), r(10.)] {
            for path in [
                ContinuationPath::PseudoArclength(r(1.5)),
                ContinuationPath::Parameters(alloc::vec![r(0.5)]),
            ] {
                let options =
                    ContinuationOptions::new(r(0.1), r(1e-6), maximum, budget(2000)).unwrap();
                let options = if stable_residual && matches!(path, ContinuationPath::Parameters(_))
                {
                    options
                } else {
                    // Squared-coordinate cancellation and constrained correction
                    // can reach the coordinate rounding floor. Natural paths
                    // with the stable residual also exercise entirely default tolerances.
                    options.with_residual_tolerance(
                        ResidualTolerance::new(r::<R>(4.) * R::EPSILON, R::ZERO).unwrap(),
                    )
                };
                let plan = Continuation::new(path, options).unwrap();
                let seed = [sign * r(0.500244140625), r(-0.499755859375)];
                let mut state = plan.state(&problem, &seed).unwrap();
                let mut workspace = plan.workspace(&problem).unwrap();
                loop {
                    let report = plan
                        .advance(
                            &problem,
                            &mut state,
                            |point, _| {
                                let expected = sign * (point[1] * point[1] + r(1. / 2048.)).sqrt();
                                assert!(sign * point[0] > R::ZERO, "switched branch: {point:?}");
                                assert!(
                                    (point[0] - expected).abs()
                                        <= r::<R>(16.) * R::SQRT_EPSILON * expected.abs(),
                                    "{point:?}"
                                );
                                Ok(())
                            },
                            &mut workspace,
                            ExecutionContext::sequential_native(),
                        )
                        .unwrap();
                    if report.termination() != RootTermination::Advanced {
                        break;
                    }
                }
                assert!(
                    state.is_success(),
                    "{:?} at {:?}",
                    state.termination(),
                    state.point()
                );
                assert!(state.point()[1] > r(0.3));
                assert!(state.history().iter().any(|h| h.rejected_steps() > 0));
            }
        }
    }
}

#[test]
fn close_branches_are_preserved_with_default_guards_and_large_maximum_steps() {
    for stable in [false, true] {
        close_branches::<f32>(stable);
        close_branches::<f64>(stable);
    }
}

struct Circle;
impl ResidualOperator<f64> for Circle {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0] + x[1] * x[1] - 1.;
        Ok(())
    }
}
impl JacobianOperator<f64> for Circle {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = 2. * (x[0] * v[0] + x[1] * v[1]);
        Ok(())
    }
}

fn circle_options() -> ContinuationOptions<f64> {
    ContinuationOptions::new(0.1, 0.03, 0.2, budget(300))
        .unwrap()
        .with_residual_tolerance(ResidualTolerance::new(1e-12, 0.).unwrap())
}

#[test]
fn tangent_iteration_failure_attempts_minimum_step_and_retains_accepted_state() {
    let problem = ContinuationProblem::new(&Circle).unwrap();
    let plan = Continuation::new(
        ContinuationPath::Parameters(alloc::vec![0.5]),
        circle_options().with_tangent_solver(
            ResidualTolerance::new(0., 1e-12).unwrap(),
            Budget::ONE,
            Budget::ONE,
        ),
    )
    .unwrap();
    let mut state = plan.state(&problem, &[1., 0.]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    plan.advance(
        &problem,
        &mut state,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let tangent = state.tangent().to_vec();
    let history = state.points.clone();
    let before = state.counts();
    let mut emitted = 0;
    let report = plan
        .advance(
            &problem,
            &mut state,
            |_, _| {
                emitted += 1;
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), RootTermination::TangentFailure);
    assert_eq!(emitted, 0);
    assert_eq!(state.point(), &[1., 0.]);
    assert_eq!(state.tangent(), tangent);
    assert_eq!(state.points, history);
    assert_eq!(state.history().len(), 1);
    assert!(
        state.counts().provider(EvaluationRole::LinearSolve)
            > before.provider(EvaluationRole::LinearSolve)
    );
    // Attempts at 0.1, 0.05 and exactly the configured 0.03 minimum.
    assert!((workspace.origin[1] - 0.03).abs() < f64::EPSILON);
    assert!(state.checkpoint().is_ok());
}

#[test]
fn geometric_controls_reject_large_turns_and_corrections() {
    let problem = ContinuationProblem::new(&Circle).unwrap();
    for (cosine, ratio) in [(0.999, 0.9), (0.1, 0.01)] {
        let options = ContinuationOptions::new(0.5, 1e-6, 0.5, budget(1000))
            .unwrap()
            .with_step_control(cosine, ratio)
            .with_residual_tolerance(ResidualTolerance::new(1e-12, 0.).unwrap());
        let plan =
            Continuation::new(ContinuationPath::Parameters(alloc::vec![0.8]), options).unwrap();
        let mut state = plan.state(&problem, &[1., 0.]).unwrap();
        let mut workspace = plan.workspace(&problem).unwrap();
        loop {
            let report = plan
                .advance(
                    &problem,
                    &mut state,
                    |point, _| {
                        assert!((point[0] - (1. - point[1] * point[1]).sqrt()).abs() < 1e-10);
                        Ok(())
                    },
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            if report.termination() != RootTermination::Advanced {
                break;
            }
        }
        assert!(state.is_success(), "{:?}", state.termination());
        assert!(state.history().iter().any(|h| h.rejected_steps() > 0));
    }
    for bad in [0., 1., -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            Continuation::new(
                ContinuationPath::PseudoArclength(1.),
                circle_options().with_correction_contraction(bad),
            )
            .is_err()
        );
        for (cosine, ratio) in [(bad, 0.5), (0.5, bad)] {
            assert!(
                Continuation::new(
                    ContinuationPath::PseudoArclength(1.),
                    circle_options().with_step_control(cosine, ratio)
                )
                .is_err()
            );
        }
    }
}

/// A provider that claims convergence after perturbing a tangent correction.
struct InexactTangent {
    error: Cell<f64>,
    requested: Cell<Option<ResidualTolerance<f64>>>,
}
impl NonlinearLinearSolver<f64> for InexactTangent {
    fn solve_into(
        &self,
        a: &crate::NonlinearLinearization<'_, f64>,
        b: &Vector<f64>,
        x: &mut Vector<f64>,
        tolerance: ResidualTolerance<f64>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let point = a.point();
        let expected = point[0] * point[0] + point[1] * point[1] - 1.;
        assert!((a.residual()[0] - expected).abs() < 1e-14);
        if b.as_slice() == [0., 1.] {
            assert_eq!(a.residual()[1], 0.);
        }
        let a = a.operator();
        let mut state = SolveState::new();
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "inexact-tangent", 1),
            2,
            0,
        );
        scope.run(&mut state, identity, SolveBudgets::unbounded(), |c| {
            let mut columns = [[0.; 2]; 2];
            for (column, direction) in columns.iter_mut().zip([[1., 0.], [0., 1.]]) {
                assert!(
                    c.record_provider(EvaluationRole::LinearOperator)?
                        .is_granted()
                );
                a.apply_into_scoped(
                    &direction,
                    column,
                    &mut [],
                    &mut scioxide_core::OperatorContext::new(c),
                )
                .and_then(scioxide_core::Invocation::require)?;
            }
            let [first, second] = columns;
            let det = first[0] * second[1] - first[1] * second[0];
            let rhs = b.as_slice();
            x.as_mut_slice()[0] = (second[1] * rhs[0] - second[0] * rhs[1]) / det;
            x.as_mut_slice()[1] = (first[0] * rhs[1] - first[1] * rhs[0]) / det;
            if rhs == [0., 1.] {
                self.requested.set(Some(tolerance));
                x.as_mut_slice()[0] += self.error.get();
            }
            Ok(IterativeSolveReport::new(
                LinearSolveTermination::Converged,
                0,
                None,
                PartialResultTrust::DiagnosticOnly,
            ))
        })
    }
}

#[test]
fn injected_tangent_tolerance_is_passed_and_independently_verified() {
    for (tolerance, expected) in [
        (1e-10, RootTermination::TangentFailure),
        (1e-3, RootTermination::Advanced),
    ] {
        let provider = InexactTangent {
            error: Cell::new(0.),
            requested: Cell::new(None),
        };
        let problem = ContinuationProblem::new(&Circle)
            .unwrap()
            .with_linear_solver(&provider);
        let requested = ResidualTolerance::new(0., tolerance).unwrap();
        let plan = Continuation::new(
            ContinuationPath::Parameters(alloc::vec![0.5]),
            circle_options().with_tangent_solver(requested, budget(10), budget(2)),
        )
        .unwrap();
        let mut state = plan.state(&problem, &[1., 0.]).unwrap();
        let mut workspace = plan.workspace(&problem).unwrap();
        plan.advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        provider.error.set(1e-5);
        let report = plan
            .advance(
                &problem,
                &mut state,
                |_, _| Ok(()),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(provider.requested.get(), Some(requested));
        assert_eq!(report.termination(), expected);
        if expected == RootTermination::TangentFailure {
            assert_eq!(state.point(), &[1., 0.]);
        }
    }
}

fn finish_circle(
    plan: &Continuation<f64>,
    problem: &ContinuationProblem<'_, f64>,
    state: &mut ContinuationState<f64>,
    workspace: &mut ContinuationWorkspace<f64>,
) {
    while plan
        .advance(
            problem,
            state,
            |_, _| Ok(()),
            workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap()
        .termination()
        == RootTermination::Advanced
    {}
    assert!(state.is_success(), "{:?}", state.termination());
}

#[test]
fn nondefault_controls_resume_exactly_and_reject_mismatched_checkpoints() {
    let problem = ContinuationProblem::new(&Circle).unwrap();
    let controls = circle_options()
        .with_step_control(0.95, 0.2)
        .with_tangent_solver(
            ResidualTolerance::new(0., 1e-11).unwrap(),
            budget(30),
            budget(2),
        );
    let plan = Continuation::new(ContinuationPath::PseudoArclength(0.5), controls.clone()).unwrap();
    let mut uninterrupted = plan.state(&problem, &[1., 0.]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    finish_circle(&plan, &problem, &mut uninterrupted, &mut workspace);
    let mut split = plan.state(&problem, &[1., 0.]).unwrap();
    for _ in 0..3 {
        plan.advance(
            &problem,
            &mut split,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    }
    let checkpoint = split.checkpoint().unwrap();
    let mut restored = plan.state(&problem, &[1., 0.]).unwrap();
    plan.resume(
        &problem,
        &checkpoint,
        &mut restored,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    finish_circle(&plan, &problem, &mut restored, &mut workspace);
    assert_eq!(restored.points, uninterrupted.points);
    assert_eq!(restored.tangent(), uninterrupted.tangent());
    assert_eq!(restored.next_step, uninterrupted.next_step);
    assert_eq!(restored.counts(), uninterrupted.counts());
    for changed in [
        controls.clone().with_step_control(0.9, 0.2),
        controls.clone().with_correction_contraction(0.25),
        controls.with_tangent_solver(
            ResidualTolerance::new(0., 1e-10).unwrap(),
            budget(30),
            budget(2),
        ),
    ] {
        let wrong = Continuation::new(plan.path().clone(), changed).unwrap();
        let mut state = wrong.state(&problem, &[1., 0.]).unwrap();
        assert!(
            wrong
                .resume(
                    &problem,
                    &checkpoint,
                    &mut state,
                    |_, _| Ok(()),
                    &mut workspace,
                    ExecutionContext::sequential_native()
                )
                .is_err()
        );
        assert_eq!(state.point(), &[1., 0.]);
        assert!(state.history().is_empty());
    }
}

#[test]
fn tangent_restart_controls_workspace_and_natural_folds_keep_their_status() {
    let problem = ContinuationProblem::new(&Circle).unwrap();
    let path = ContinuationPath::Parameters(alloc::vec![0.5]);
    let small = Continuation::new(
        path.clone(),
        circle_options().with_tangent_solver(
            ResidualTolerance::new(0., 1e-12).unwrap(),
            budget(20),
            Budget::ONE,
        ),
    )
    .unwrap();
    let large = Continuation::new(
        path,
        circle_options().with_tangent_solver(
            ResidualTolerance::new(0., 1e-12).unwrap(),
            budget(20),
            budget(2),
        ),
    )
    .unwrap();
    let small_requirements = small.workspace_requirements(&problem).unwrap();
    let large_requirements = large.workspace_requirements(&problem).unwrap();
    assert!(small_requirements.required_bytes() < large_requirements.required_bytes());
    let mut workspace = small.workspace(&problem).unwrap();
    assert_eq!(
        workspace.check(&large_requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    workspace.try_reserve(&large_requirements).unwrap();
    assert_eq!(
        workspace.check(&large_requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );

    let mut state = large
        .state(&problem, &[0., 1.])
        .unwrap()
        .with_initial_tangent(&[-1., 0.])
        .unwrap();
    large
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let report = large
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), RootTermination::TangentFailure);
    assert_eq!(state.point(), &[0., 1.]);
    assert_eq!(state.history().len(), 1);
}
