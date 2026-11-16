// Fixed numerical fixtures intentionally fail the test on setup errors.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Bounds, Budget, BudgetKind, ExecutionContext, GradientTolerance, NumericScalar, ProblemToken,
    RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveState,
    StepTolerance, Tolerance,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::*;
use scioxide_stats::*;

fn epoch<T>(
    mut budgets: SolveBudgets,
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(614),
        scioxide_core::algorithm_id!("stats", "adapter-test", 1),
        0,
        0,
    );
    let mut ctx = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    run(&mut ctx)
}
fn scalar<S: RealScalar>(v: f64) -> S {
    S::try_from_f64_lossy(v).unwrap()
}
fn close<S: RealScalar>(v: S, expected: f64, tolerance: f64) {
    assert!(
        (v.try_to_f64_exact().unwrap() - expected).abs() <= tolerance,
        "{} != {expected}",
        v.try_to_f64_exact().unwrap()
    );
}
fn ls_options<S: RealScalar>() -> LeastSquaresOptions<S> {
    let t = Tolerance::new(scalar::<S>(1e-5), scalar::<S>(1e-5)).unwrap();
    LeastSquaresOptions::default()
        .with_gradient_tolerance(GradientTolerance::from_tolerance(t))
        .with_step_tolerance(StepTolerance::from_tolerance(t))
}
fn ls_solver<S: RealScalar + NumericScalar<Real = S>>() -> LeastSquares<S> {
    LeastSquares::new(ls_options()).unwrap()
}
fn distribution<S: RealScalar + NumericScalar<Real = S>>() {
    let samples = [-1., -1., 3., 3.].map(scalar::<S>);
    let observations = samples.map(FitObservation::Exact);
    let bounds = Bounds::new(vec![
        (Some(scalar::<S>(-4.)), Some(scalar::<S>(4.))),
        (Some(scalar::<S>(0.2)), Some(scalar::<S>(5.))),
    ])
    .unwrap();
    let model = DistributionFitModel::Normal;
    let objective = PreparedFit::new(
        &model,
        &observations,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false; 2],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    let derivatives = ExactFitDerivatives {
        model,
        samples: &samples,
    };
    let initial = [scalar::<S>(0.), scalar::<S>(1.)];
    let request = FitRequest {
        objective: &objective,
        derivatives: Some(&derivatives),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &initial,
    };
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        scioxide_stats::fit_scoped(
            request,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(result.report().termination(), StatsTermination::Converged);
    let parameters = result.parameters().unwrap();
    close(parameters[0], 1., 0.002);
    close(parameters[1], 2., 0.002);
    let mut score = [scalar::<S>(0.); 2];
    derivatives.gradient(parameters, &mut score).unwrap();
    score.iter().for_each(|&g| close(g, 0., 0.004));
}
#[test]
fn distribution_f64() {
    distribution::<f64>();
}
#[test]
fn distribution_f32() {
    distribution::<f32>();
}

struct Quadratic;
impl FitObjective<f64> for Quadratic {
    type Workspace = usize;
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(core::mem::size_of::<usize>())
    }
    fn workspace(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn validate(&self, _: &[f64], _: &mut usize) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn value(&self, x: &[f64], w: &mut usize) -> Result<f64, SciOxideError> {
        *w += 1;
        Ok((x[0] - 2.).powi(2) + (x[1] - 3.).powi(2))
    }
}
impl FitDerivatives<f64> for Quadratic {
    fn capabilities(&self) -> FitDerivativeCapabilities {
        FitDerivativeCapabilities {
            gradient: true,
            ..Default::default()
        }
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = 2. * (x[0] - 2.);
        out[1] = 2. * (x[1] - 3.);
        Ok(())
    }
}
impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        FitObjective::value(self, x, &mut 0)
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        FitDerivatives::gradient(self, x, out)?;
        Ok(true)
    }
}
#[test]
fn direct_and_adapter_budgets_and_fixed_bounds() {
    let bounds = Bounds::new(vec![(Some(0.), Some(1.)), (Some(4.), Some(4.))]).unwrap();
    let initial = [0., 4.];
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &initial,
    };
    for iterations in [1, 2, 20] {
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(iterations).unwrap());
        let direct = {
            let solver = LbfgsB::new(
                LbfgsBOptions::default().with_max_iterations(Budget::try_from(iterations).unwrap()),
            );
            let mut workspace = solver.workspace(2).unwrap();
            let mut direct = scioxide_optimize::OptimizeResult::empty();
            solver
                .minimize_into(
                    &BoundedObjective::new(&Quadratic, &bounds).unwrap(),
                    &initial,
                    &mut direct,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            direct
        };
        let mut adapter = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
        let mut output = initial;
        let mut scratch = 0;
        epoch(budgets, |ctx| {
            adapter.solve(request, &mut output, &mut scratch, ctx)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(adapter.result().counts(), direct.counts());
        assert_eq!(adapter.result().termination(), direct.termination());
        if !direct.x().is_empty() {
            assert_eq!(output, direct.x());
        }
        assert_eq!(output[1], 4.);
    }
}
#[test]
fn fitting_verification_budget_preserves_output_and_reuse() {
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 0.],
    };
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
    let mut workspace = FitWorkspace::for_objective(
        &Quadratic,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    let mut output = [99.; 2];
    let report = epoch(
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        |ctx| fit_into(request, &mut solver, &mut output, &mut workspace, ctx),
    )
    .unwrap();
    assert_eq!(report.termination(), StatsTermination::BudgetExhausted);
    assert!(report.objective().is_none());
    assert_eq!(output, [99.; 2]);
    let report = epoch(SolveBudgets::unbounded(), |ctx| {
        fit_into(request, &mut solver, &mut output, &mut workspace, ctx)
    })
    .unwrap();
    assert_eq!(report.termination(), StatsTermination::Converged);
    close(output[0], 2., 1e-6);
    close(output[1], 3., 1e-6);
}
#[test]
fn sqp_fitting_and_request_constraint_translation() {
    let bounds = Bounds::new(vec![(Some(0.), Some(1.)), (Some(4.), Some(4.))]).unwrap();
    let constraints = Constraints::new(2).unwrap().bounds(bounds.clone()).unwrap();
    let mut solver = SqpFitSolver::new(Sqp::new(SqpOptions::default()), constraints).unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 4.],
    };
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        scioxide_stats::fit_scoped(
            request,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(result.report().termination(), StatsTermination::Converged);
    let p = result.parameters().unwrap();
    close(p[0], 1., 1e-6);
    close(p[1], 4., 1e-6);
    let wrong = Bounds::new(vec![(None, None); 2]).unwrap();
    assert!(
        epoch(SolveBudgets::unbounded(), |ctx| solver.solve(
            FitRequest {
                bounds: &wrong,
                ..request
            },
            &mut [0., 4.],
            &mut 0,
            ctx
        ))
        .is_err()
    );
    assert!(
        epoch(SolveBudgets::unbounded(), |ctx| solver.solve(
            FitRequest {
                integrality: &[true, false],
                ..request
            },
            &mut [0., 4.],
            &mut 0,
            ctx
        ))
        .is_err()
    );
}
fn glm<S: RealScalar + NumericScalar<Real = S>>() {
    let design = Matrix::from_vec(4, 1, vec![scalar::<S>(1.); 4]).unwrap();
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(65_536),
    )
    .unwrap();
    let response = [1., 2., 3., 2.].map(scalar::<S>);
    let objective = GlmObjective::new(&model, &response).unwrap();
    let bounds = Bounds::new(vec![(Some(scalar::<S>(-3.)), Some(scalar::<S>(3.)))]).unwrap();
    let initial = [scalar::<S>(0.)];
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 1).unwrap();
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        model.fit_scoped(
            scioxide_stats::GlmInputs::new(&response, scioxide_stats::GlmOptions::default())
                .with_solver(scioxide_stats::GlmSolver::Injected {
                    solver: &mut solver,
                    initial: &initial,
                    bounds: &bounds,
                    integrality: &[false],
                }),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(
        result.report().termination,
        scioxide_stats::StatsTermination::Converged
    );
    close(
        result.fit().unwrap().coefficients().as_slice()[0],
        2_f64.ln(),
        0.001,
    );
    let mut gradient = [scalar::<S>(0.)];
    objective
        .gradient(&[scalar::<S>(2_f64.ln())], &mut gradient)
        .unwrap();
    close(gradient[0], 0., 1e-5);
    objective.gradient(&initial, &mut gradient).unwrap();
    close(gradient[0], -4., 1e-5);
    let constraints = Constraints::new(1).unwrap().bounds(bounds.clone()).unwrap();
    let mut sqp = SqpFitSolver::new(Sqp::new(SqpOptions::default()), constraints).unwrap();
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        model.fit_scoped(
            scioxide_stats::GlmInputs::new(&response, scioxide_stats::GlmOptions::default())
                .with_solver(scioxide_stats::GlmSolver::Injected {
                    solver: &mut sqp,
                    initial: &initial,
                    bounds: &bounds,
                    integrality: &[false],
                }),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(
        result.report().termination,
        scioxide_stats::StatsTermination::Converged
    );
    close(
        result.fit().unwrap().coefficients().as_slice()[0],
        2_f64.ln(),
        0.001,
    );
}
#[test]
fn glm_f64() {
    glm::<f64>();
}
#[test]
fn glm_f32() {
    glm::<f32>();
}
fn odr<S: RealScalar + NumericScalar<Real = S>>() {
    let x = Matrix::from_vec(3, 1, [0., 1., 2.].map(scalar::<S>).to_vec()).unwrap();
    let y = Matrix::from_vec(3, 1, [1., 3., 5.].map(scalar::<S>).to_vec()).unwrap();
    let scale = Matrix::from_vec(3, 1, vec![scalar::<S>(1.); 3]).unwrap();
    let model = AffineOdrModel::new(1).unwrap();
    let objective = PreparedOdr::new(
        &model,
        x.view(),
        y.view(),
        scale.view(),
        scale.view(),
        scioxide_stats::OdrOptions::new(),
    )
    .unwrap();
    let bounds = Bounds::new(vec![(None, None); 5]).unwrap();
    let initial = [scalar::<S>(0.); 5];
    let request = FitRequest {
        objective: &objective,
        derivatives: Some(&objective),
        bounds: &bounds,
        integrality: &[false; 5],
        initial: &initial,
    };
    let mut solver = LeastSquaresFitSolver::new(ls_solver(), 5, 6).unwrap();
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        scioxide_stats::fit_scoped(
            request,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(result.report().termination(), StatsTermination::Converged);
    let p = result.parameters().unwrap();
    close(p[0], 1., 0.002);
    close(p[1], 2., 0.002);
    p[2..].iter().for_each(|&v| close(v, 0., 0.002));
    close(
        result.report().objective().unwrap(),
        2. * solver.result().cost().unwrap().try_to_f64_exact().unwrap(),
        1e-5,
    );
}
#[test]
fn odr_f64() {
    odr::<f64>();
}
#[test]
fn odr_f32() {
    odr::<f32>();
}
fn transport<S: RealScalar + NumericScalar<Real = S>>() {
    // Exact optimum: send 1/4 on cost 0, 1/2 on cost 1, 1/4 on cost 0.
    let a = [0.75, 0.25].map(scalar::<S>);
    let b = [0.25, 0.75].map(scalar::<S>);
    let costs = [0., 1., 1., 0.].map(scalar::<S>);
    let mut solver = SimplexTransportSolver::new(SimplexSolver::default(), 4, 4096);
    let mut coupling = [scalar::<S>(99.); 4];
    let mut u = [scalar::<S>(99.); 2];
    let mut v = u;
    let status = epoch(SolveBudgets::unbounded(), |ctx| {
        solver.solve(
            TransportProblem {
                source: &a,
                target: &b,
                costs: &costs,
                cost_scale: scalar::<S>(8.),
            },
            TransportCandidate {
                coupling: &mut coupling,
                source_dual: &mut u,
                target_dual: &mut v,
            },
            ctx,
        )
    })
    .unwrap();
    assert_eq!(status, StatsTermination::Converged);
    for (&x, y) in coupling.iter().zip([0.25, 0.5, 0., 0.25]) {
        close(x, y, 2e-5);
    }
    let primal = coupling
        .iter()
        .zip(costs)
        .map(|(&p, c)| p.try_to_f64_exact().unwrap() * c.try_to_f64_exact().unwrap())
        .sum::<f64>();
    let dual = a
        .iter()
        .zip(u)
        .chain(b.iter().zip(v))
        .map(|(&p, d)| p.try_to_f64_exact().unwrap() * d.try_to_f64_exact().unwrap())
        .sum::<f64>();
    assert!((primal - dual).abs() < 2e-5);
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            assert!(
                ui.try_to_f64_exact().unwrap() + vj.try_to_f64_exact().unwrap()
                    <= costs[i * 2 + j].try_to_f64_exact().unwrap() + 2e-5
            );
        }
    }
}
#[test]
fn transport_f64() {
    transport::<f64>();
}
#[test]
fn transport_f32() {
    transport::<f32>();
}

fn matching<S: RealScalar + NumericScalar<Real = S>>() {
    let observations = [1., 3.].map(|x| FitObservation::Exact(scalar::<S>(x)));
    let bounds = Bounds::new(vec![(None, None)]).unwrap();
    let model = DistributionFitModel::Exponential;
    let orders = [1];
    let scales = [scalar::<S>(1.)];
    let probabilities = [scalar::<S>(0.5)];
    for (method, expected) in [
        (
            FitMethod::Moments {
                orders: &orders,
                scales: &scales,
            },
            0.5,
        ),
        (
            FitMethod::Quantiles {
                probabilities: &probabilities,
                scales: &scales,
                convention: QuantileMethod::Linear,
            },
            2_f64.ln() / 2.,
        ),
    ] {
        let objective = PreparedFit::new(
            &model,
            &observations,
            method,
            None,
            &bounds,
            &[false],
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        )
        .unwrap();
        let initial = [scalar::<S>(0.4)];
        let request = FitRequest {
            objective: &objective,
            derivatives: None,
            bounds: &bounds,
            integrality: &[false],
            initial: &initial,
        };
        let mut solver = LeastSquaresFitSolver::new(ls_solver(), 1, 1).unwrap();
        let result = epoch(SolveBudgets::unbounded(), |ctx| {
            scioxide_stats::fit_scoped(
                request,
                &mut solver,
                scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
                ctx,
            )
        })
        .unwrap();
        assert_eq!(result.report().termination(), StatsTermination::Converged);
        close(result.parameters().unwrap()[0], expected, 0.001);
    }
}
#[test]
fn matching_f64() {
    matching::<f64>();
}
#[test]
fn matching_f32() {
    matching::<f32>();
}

fn noisy_odr<S: RealScalar + NumericScalar<Real = S>>() {
    let xs = [0., 1., 2., 3., 4.];
    let ys = [1.1, 2.9, 5.2, 6.8, 9.];
    // Independent total-least-squares solution: smaller scatter eigenvalue,
    // and its principal eigenvector, derived from the exact 2x2 covariance.
    let xbar = 2.;
    let ybar = ys.iter().sum::<f64>() / 5.;
    let xx = xs.iter().map(|x| (x - xbar) * (x - xbar)).sum::<f64>();
    let yy = ys.iter().map(|y| (y - ybar) * (y - ybar)).sum::<f64>();
    let xy = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - xbar) * (y - ybar))
        .sum::<f64>();
    let disc = ((yy - xx) * (yy - xx) + 4. * xy * xy).sqrt();
    let slope = (yy - xx + disc) / (2. * xy);
    let intercept = ybar - slope * xbar;
    let x = Matrix::from_vec(5, 1, xs.map(scalar::<S>).to_vec()).unwrap();
    let y = Matrix::from_vec(5, 1, ys.map(scalar::<S>).to_vec()).unwrap();
    let scales = Matrix::from_vec(5, 1, vec![scalar::<S>(1.); 5]).unwrap();
    let model = AffineOdrModel::new(1).unwrap();
    let objective = PreparedOdr::new(
        &model,
        x.view(),
        y.view(),
        scales.view(),
        scales.view(),
        scioxide_stats::OdrOptions::new(),
    )
    .unwrap();
    let bounds = Bounds::new(vec![(None, None); 7]).unwrap();
    let initial = [scalar::<S>(0.); 7];
    let request = FitRequest {
        objective: &objective,
        derivatives: Some(&objective),
        bounds: &bounds,
        integrality: &[false; 7],
        initial: &initial,
    };
    let mut solver = LeastSquaresFitSolver::new(ls_solver(), 7, 10).unwrap();
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        scioxide_stats::fit_scoped(
            request,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(result.report().termination(), StatsTermination::Converged);
    let p = result.parameters().unwrap();
    close(p[0], intercept, 0.002);
    close(p[1], slope, 0.002);
    close(
        result.report().objective().unwrap(),
        (xx + yy - disc) / 2.,
        0.0002,
    );
}
#[test]
fn independent_tls_f64() {
    noisy_odr::<f64>();
}
#[test]
fn independent_tls_f32() {
    noisy_odr::<f32>();
}

struct SumConstraint;
impl ConstraintFunction<f64> for SumConstraint {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 1.)
    }
    fn jacobian_into(&self, _: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out.fill(1.);
        Ok(true)
    }
}
#[test]
fn sqp_equality_fit() {
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let constraints = Constraints::new(2)
        .unwrap()
        .equality(SumConstraint)
        .unwrap();
    let mut solver = SqpFitSolver::new(Sqp::new(SqpOptions::default()), constraints).unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 0.],
    };
    let result = epoch(SolveBudgets::unbounded(), |ctx| {
        scioxide_stats::fit_scoped(
            request,
            &mut solver,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(result.report().termination(), StatsTermination::Converged);
    let p = result.parameters().unwrap();
    close(p[0], 0., 1e-5);
    close(p[1], 1., 1e-5);
    assert!(solver.result().primal_feasibility().unwrap() < 1e-5);
}

fn wasserstein<S: RealScalar + NumericScalar<Real = S>>() {
    let shape = [2, 2];
    let strides = [2, 1];
    let layout = scioxide_core::StridedLayout::new(&shape, &strides).unwrap();
    for multiplier in [1., 100.] {
        let a = [1000., 0., 1000., 2.].map(|x| scalar::<S>(x * multiplier));
        let b = [1003., 0., 1003., 2.].map(|x| scalar::<S>(x * multiplier));
        let left = BatchView::new(&a, layout, Axis::Index(0))
            .and_then(|view| view.with_coordinates(Axis::Index(1)))
            .unwrap();
        let right = BatchView::new(&b, layout, Axis::Index(0))
            .and_then(|view| view.with_coordinates(Axis::Index(1)))
            .unwrap();
        let tolerance = Tolerance::new(scalar::<S>(1e-5), scalar::<S>(1e-5)).unwrap();
        let options = TransportOptions::new()
            .with_max_pairs(4)
            .with_max_workspace_bytes(4096)
            .with_feasibility(scioxide_core::FeasibilityTolerance::from_tolerance(
                tolerance,
            ))
            .with_optimality(scioxide_core::ResidualTolerance::from_tolerance(tolerance));
        let mut workspace = TransportWorkspace::default();
        workspace.reserve(2, 2, options).unwrap();
        let mut solver = SimplexTransportSolver::new(SimplexSolver::default(), 4, 4096);
        let result = epoch(SolveBudgets::unbounded(), |ctx| {
            scioxide_stats::wasserstein_distance_nd_scoped(
                left,
                right,
                StatisticsOptions::new(),
                options,
                &mut solver,
                &mut workspace,
                ctx,
            )
        })
        .unwrap();
        assert_eq!(result.termination, StatsTermination::Converged);
        close(
            result.distance.unwrap(),
            3. * multiplier,
            0.003 * multiplier,
        );
        close(
            result.original_dual_residual().unwrap(),
            0.,
            0.003 * multiplier,
        );
        close(
            result.original_duality_gap().unwrap().unwrap(),
            0.,
            0.003 * multiplier,
        );
    }
}
#[test]
fn wasserstein_facade_f64() {
    wasserstein::<f64>();
}
#[test]
fn wasserstein_facade_f32() {
    wasserstein::<f32>();
}

#[test]
fn transport_unequal_supports_zero_mass_and_limits() {
    let source = [0., 0.25, 0.75];
    let target = [0.5, 0.5];
    let costs = [5., 7., 0., 2., 2., 0.];
    let p = TransportProblem {
        source: &source,
        target: &target,
        costs: &costs,
        cost_scale: 1.,
    };
    let mut c = [99.; 6];
    let mut u = [99.; 3];
    let mut v = [99.; 2];
    let bytes = SimplexTransportSolver::<f64>::required_model_bytes(3, 2).unwrap();
    let mut solver = SimplexTransportSolver::new(SimplexSolver::default(), 6, bytes);
    let status = epoch(SolveBudgets::unbounded(), |ctx| {
        solver.solve(
            p,
            TransportCandidate {
                coupling: &mut c,
                source_dual: &mut u,
                target_dual: &mut v,
            },
            ctx,
        )
    })
    .unwrap();
    assert_eq!(status, StatsTermination::Converged);
    assert_eq!(&c[..2], &[0., 0.]);
    close(
        c.iter().zip(costs).map(|(&p, c)| p * c).sum::<f64>(),
        0.5,
        1e-6,
    );
    for (pairs, limit, resource, required) in [
        (5, bytes, scioxide_core::quantity!("transport_pairs"), 6),
        (
            6,
            bytes - 1,
            scioxide_core::quantity!("transport_model_bytes"),
            bytes,
        ),
    ] {
        let mut too_small = SimplexTransportSolver::new(SimplexSolver::default(), pairs, limit);
        let before = c;
        let error = epoch(SolveBudgets::unbounded(), |ctx| {
            too_small.solve(
                p,
                TransportCandidate {
                    coupling: &mut c,
                    source_dual: &mut u,
                    target_dual: &mut v,
                },
                ctx,
            )
        })
        .unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource: actual, required: need, .. } if actual == resource && need == required),
            "{error:?}"
        );
        assert_eq!(c, before);
    }
    assert!(SimplexTransportSolver::<f64>::required_model_bytes(usize::MAX, 2).is_err());
    let status = epoch(
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
        |ctx| {
            solver.solve(
                p,
                TransportCandidate {
                    coupling: &mut c,
                    source_dual: &mut u,
                    target_dual: &mut v,
                },
                ctx,
            )
        },
    )
    .unwrap();
    assert_eq!(status, StatsTermination::BudgetExhausted);
}
#[test]
fn incompatible_least_squares_policy_and_shapes_fail_before_callbacks() {
    let robust =
        LeastSquares::new(ls_options::<f64>().with_loss(LeastSquaresLoss::Huber, 1.)).unwrap();
    assert!(LeastSquaresFitSolver::new(robust, 1, 1).is_err());
    let mut solver = LeastSquaresFitSolver::new(ls_solver(), 2, 2).unwrap();
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: None,
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 0.],
    };
    let mut count = 0;
    assert!(
        epoch(SolveBudgets::unbounded(), |ctx| solver.solve(
            request,
            &mut [0.; 2],
            &mut count,
            ctx
        ))
        .is_err()
    );
    assert_eq!(count, 0);
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 3).unwrap();
    assert!(
        epoch(SolveBudgets::unbounded(), |ctx| solver.solve(
            request,
            &mut [0.; 2],
            &mut count,
            ctx
        ))
        .is_err()
    );
}
#[test]
fn executor_and_cancellation_preserve_fit_contract() {
    #[cfg(feature = "std")]
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let executor = scioxide_core::SequentialExecutor;
    let cancel = scioxide_core::CancellationToken::new();
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 0.],
    };
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
    let mut workspace = FitWorkspace::for_objective(
        &Quadratic,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    for cancelled in [false, true] {
        if cancelled {
            cancel.cancel();
        }
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(614),
            scioxide_core::algorithm_id!("stats", "adapter-executor-test", 1),
            0,
            0,
        );
        let mut ctx = state
            .begin_solve(
                ExecutionContext::new(&executor).with_cancellation(&cancel),
                identity,
                &mut budgets,
            )
            .unwrap();
        let mut output = [99.; 2];
        let report = fit_into(request, &mut solver, &mut output, &mut workspace, &mut ctx).unwrap();
        if cancelled {
            assert_eq!(
                report.termination(),
                StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
            );
            assert_eq!(output, [99.; 2]);
            assert_eq!(report.counts(), scioxide_core::SolveCounts::new());
        } else {
            assert_eq!(report.termination(), StatsTermination::Converged);
            close(output[0], 2., 1e-6);
            close(output[1], 3., 1e-6);
        }
    }
}

struct ResidualProblem;
impl FitObjective<f64> for ResidualProblem {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        2
    }
    fn validate(&self, _: &[f64], _: &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn value(&self, x: &[f64], _: &mut ()) -> Result<f64, SciOxideError> {
        FitObjective::value(&Quadratic, x, &mut 0)
    }
    fn residual_count(&self) -> Option<usize> {
        Some(2)
    }
    fn residuals(&self, x: &[f64], out: &mut [f64], _: &mut ()) -> Result<(), SciOxideError> {
        out[0] = x[0] - 2.;
        out[1] = x[1] - 3.;
        Ok(())
    }
}
impl FitDerivatives<f64> for ResidualProblem {
    fn capabilities(&self) -> FitDerivativeCapabilities {
        FitDerivativeCapabilities {
            jacobian: true,
            ..Default::default()
        }
    }
    fn jacobian(&self, _: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[1., 0., 0., 1.]);
        Ok(())
    }
}
impl LeastSquaresProblem<f64> for ResidualProblem {
    fn param_dim(&self) -> usize {
        2
    }
    fn residual_dim(&self) -> usize {
        2
    }
    fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        FitObjective::residuals(self, x, out, &mut ())
    }
    fn jacobian_into(&self, _: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        out.view_mut().set(0, 0, 1.)?;
        out.view_mut().set(1, 0, 0.)?;
        out.view_mut().set(0, 1, 0.)?;
        out.view_mut().set(1, 1, 1.)?;
        Ok(true)
    }
}
#[test]
fn least_squares_direct_budget_equivalence_and_facade_overhead() {
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let initial = [-3., 0.];
    let request = FitRequest {
        objective: &ResidualProblem,
        derivatives: Some(&ResidualProblem),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &initial,
    };
    for limit in [1, 2, 100] {
        let solver =
            LeastSquares::new(ls_options().with_max_iterations(Budget::try_from(limit).unwrap()))
                .unwrap();
        let mut direct_workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::dense(2, 2))
            .unwrap();
        let mut direct = LeastSquaresResult::empty();
        solver
            .solve_into(
                &ResidualProblem,
                &initial,
                &mut direct,
                &mut direct_workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        let mut adapter = LeastSquaresFitSolver::new(solver, 2, 2).unwrap();
        let mut workspace = FitWorkspace::for_objective(
            &ResidualProblem,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
        )
        .unwrap();
        let mut output = initial;
        let report = epoch(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap()),
            |ctx| fit_into(request, &mut adapter, &mut output, &mut workspace, ctx),
        )
        .unwrap();
        assert_eq!(adapter.result().counts(), direct.counts());
        assert_eq!(adapter.result().termination(), direct.termination());
        assert_eq!(output, direct.x());
        assert_eq!(
            report
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Domain),
            1
        );
        assert_eq!(
            report
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Objective),
            1
        );
        assert_eq!(report.counts().nested(), direct.counts().total().unwrap());
    }
}
#[test]
fn sqp_direct_budget_equivalence() {
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let initial = [0., 0.];
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &initial,
    };
    let unconstrained = Constraints::new(2).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &unconstrained).unwrap();
    for limit in [1, 2, 100] {
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap());
        // The adapter forwards its remaining iteration allowance; the direct
        // solve configures the same limit through its options.
        let solver =
            Sqp::new(SqpOptions::default().with_max_iterations(Budget::try_from(limit).unwrap()));
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut direct = OptimizeResult::empty();
        solver
            .minimize_into(
                &problem,
                &initial,
                &mut direct,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        let mut adapter = SqpFitSolver::new(
            Sqp::new(SqpOptions::default()),
            Constraints::new(2).unwrap(),
        )
        .unwrap();
        let mut output = initial;
        epoch(budgets, |ctx| {
            adapter.solve(request, &mut output, &mut 0, ctx)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(adapter.result().counts(), direct.counts());
        assert_eq!(adapter.result().termination(), direct.termination());
        assert_eq!(output, direct.x());
    }
}
#[test]
fn provider_errors_and_early_budget_denial_do_not_publish_stale_candidates() {
    struct Failure;
    impl FitObjective<f64> for Failure {
        type Workspace = usize;
        fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
            Ok(8)
        }
        fn workspace(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn dimension(&self) -> usize {
            2
        }
        fn validate(&self, _: &[f64], _: &mut usize) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn value(&self, _: &[f64], _: &mut usize) -> Result<f64, SciOxideError> {
            Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "workspace"
            )))
        }
    }
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let mut adapter = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
    let mut workspace = FitWorkspace::for_objective(
        &Quadratic,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )
    .unwrap();
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &[0., 0.],
    };
    let mut output = [99.; 2];
    epoch(SolveBudgets::unbounded(), |ctx| {
        fit_into(request, &mut adapter, &mut output, &mut workspace, ctx)
    })
    .unwrap();
    output = [99.; 2];
    assert!(
        epoch(SolveBudgets::unbounded(), |ctx| fit_into(
            FitRequest {
                objective: &Failure,
                derivatives: None,
                ..request
            },
            &mut adapter,
            &mut output,
            &mut workspace,
            ctx,
        ))
        .is_err()
    );
    assert_eq!(output, [99.; 2]);
    let report = epoch(
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        |ctx| fit_into(request, &mut adapter, &mut output, &mut workspace, ctx),
    )
    .unwrap();
    assert_eq!(report.termination(), StatsTermination::BudgetExhausted);
    assert!(report.objective().is_none());
    assert_eq!(output, [99.; 2]);
}

#[test]
fn transport_direct_lp_budget_equivalence_and_cancelled_output() {
    let matrix = scioxide_sparse::CsrMatrix::from_raw(
        3,
        4,
        vec![0, 2, 4, 6],
        vec![0, 1, 2, 3, 0, 2],
        vec![1.; 6],
    )
    .unwrap();
    let lp = LpProblem::new(vec![0., 1., 1., 0.])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                matrix,
                vec![0.75, 0.25, 0.25],
                vec![LinearConstraintSense::Equal; 3],
            )
            .unwrap(),
        )
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), None); 4]).unwrap())
        .unwrap();
    for limit in [1, 2, 30] {
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap());
        // Iteration allowances are epoch-local: the pivot budget is
        // the solver's own limit in both the direct and the adapted solve.
        let simplex =
            SimplexSolver::default().with_max_iterations(Budget::try_from(limit).unwrap());
        let mut state = SolveState::new();
        let mut ctx = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(614),
                    scioxide_core::algorithm_id!("stats", "transport-direct", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = simplex.workspace(&lp).unwrap();
        let mut direct = scioxide_optimize::ProgrammingResult::empty();
        simplex
            .solve_into(&lp, &mut direct, &mut workspace, &mut ctx)
            .unwrap();
        let mut solver = SimplexTransportSolver::new(simplex, 4, 4096);
        let mut c = [99.; 4];
        let mut u = [99.; 2];
        let mut v = [99.; 2];
        let (status, counts) = epoch(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap()),
            |ctx| {
                let status = solver.solve(
                    TransportProblem {
                        source: &[0.75, 0.25],
                        target: &[0.25, 0.75],
                        costs: &[0., 1., 1., 0.],
                        cost_scale: 1.,
                    },
                    TransportCandidate {
                        coupling: &mut c,
                        source_dual: &mut u,
                        target_dual: &mut v,
                    },
                    ctx,
                )?;
                Ok((status, ctx.counts()))
            },
        )
        .unwrap();
        assert_eq!(counts.total().unwrap(), direct.counts().total().unwrap());
        if direct.is_success() {
            assert_eq!(status, StatsTermination::Converged);
        } else {
            assert_eq!(status, StatsTermination::BudgetExhausted);
        }
        if !direct.x().is_empty() {
            assert_eq!(c, direct.x());
        }
    }
    let cancel = scioxide_core::CancellationToken::new();
    cancel.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(614),
        scioxide_core::algorithm_id!("stats", "transport-stop", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancel),
            identity,
            &mut budgets,
        )
        .unwrap();
    let mut solver = SimplexTransportSolver::new(SimplexSolver::default(), 4, 4096);
    let mut c = [99.; 4];
    let mut u = [99.; 2];
    let mut v = [99.; 2];
    let status = solver
        .solve(
            TransportProblem {
                source: &[0.75, 0.25],
                target: &[0.25, 0.75],
                costs: &[0., 1., 1., 0.],
                cost_scale: 1.,
            },
            TransportCandidate {
                coupling: &mut c,
                source_dual: &mut u,
                target_dual: &mut v,
            },
            &mut ctx,
        )
        .unwrap();
    assert_eq!(
        status,
        StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
    );
    assert_eq!(c, [0.; 4]);
    assert_eq!(u, [0.; 2]);
    assert_eq!(v, [0.; 2]);
    assert_eq!(ctx.counts(), scioxide_core::SolveCounts::new());
}

#[test]
fn early_adapter_stop_does_not_reuse_last_solve_evidence() {
    let bounds = Bounds::new(vec![(None, None); 2]).unwrap();
    let initial = [0., 0.];
    let request = FitRequest {
        objective: &Quadratic,
        derivatives: Some(&Quadratic),
        bounds: &bounds,
        integrality: &[false; 2],
        initial: &initial,
    };
    let mut adapter = LbfgsBFitSolver::new(LbfgsB::default(), 2).unwrap();
    let mut candidate = initial;
    epoch(SolveBudgets::unbounded(), |ctx| {
        adapter.solve(request, &mut candidate, &mut 0, ctx)
    })
    .unwrap();
    assert!(adapter.result().is_success());
    assert!(!adapter.result().x().is_empty());
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(614),
        scioxide_core::algorithm_id!("stats", "early-stop", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    candidate = initial;
    assert_eq!(
        adapter
            .solve(request, &mut candidate, &mut 0, &mut ctx)
            .unwrap(),
        StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
    );
    assert_eq!(candidate, initial);
    assert_eq!(
        adapter.result().termination(),
        OptimizeTermination::Cancelled
    );
    assert!(adapter.result().x().is_empty());
    assert!(adapter.result().grad().is_none());
    assert_eq!(adapter.result().counts(), scioxide_core::SolveCounts::new());
    assert_eq!(
        adapter.result().trust(),
        scioxide_core::PartialResultTrust::DiagnosticOnly
    );
}
