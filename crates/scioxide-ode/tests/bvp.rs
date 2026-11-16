#![allow(clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
#[path = "../examples/support/bvp_models.rs"]
mod models;
use scioxide_core::{
    Budget, Complex, ExecutionContext, NumericScalar, ProblemToken, RealScalar, ResidualTolerance,
    SciOxideError, SolveContext, SolveIdentity,
};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;
fn sparse() -> BvpBackend {
    BvpBackend::Sparse {
        limits: Some(SparseLuLimits::new(4096, 1024 * 1024)),
    }
}
fn direct_backends() -> [BvpBackend; 2] {
    [BvpBackend::Dense, sparse()]
}
fn krylov() -> BvpBackend {
    BvpBackend::Krylov {
        iterations: Budget::try_from(1024).unwrap(),
        restart: Budget::try_from(512).unwrap(),
    }
}
fn backends() -> [BvpBackend; 3] {
    let [dense, sparse] = direct_backends();
    [dense, sparse, krylov()]
}
fn options<R: RealScalar>(backend: BvpBackend, tol: f64) -> BvpOptions<R> {
    BvpOptions::new()
        .with_max_nodes(512)
        .with_max_workspace_bytes(128 * 1024 * 1024)
        .with_backend(backend)
        .with_tolerances(
            ResidualTolerance::new(
                R::try_from_f64_lossy(tol).unwrap(),
                R::try_from_f64_lossy(tol).unwrap(),
            )
            .unwrap(),
            ResidualTolerance::new(R::try_from_f64_lossy(tol).unwrap(), R::ZERO).unwrap(),
        )
}
/// The report of a published solution.
fn report<T: NumericScalar>(solution: &BvpSolution<T>) -> BvpReport<T::Real> {
    solution.report().unwrap()
}
/// Identity of a caller-owned parent epoch.
fn parent() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "bvp-parent", 1),
        0,
        0,
    )
}
fn exponential<T: NumericScalar>(amplitude: T, backends: &[BvpBackend], tol: f64) {
    let system = models::Exponential { amplitude };
    let mesh = [T::Real::ZERO, T::Real::HALF, T::Real::ONE];
    let guess = Matrix::from_vec(3, 2, vec![amplitude; 6]).unwrap();
    for &backend in backends {
        for analytic in [true, false] {
            let problem = if analytic {
                BvpProblem::analytic(&system)
            } else {
                BvpProblem::numerical(&system)
            };
            let plan = BvpPlan::new(
                system.dimension(),
                system.parameter_dimension(),
                &mesh,
                options(backend, tol),
            )
            .unwrap();
            let solution = plan.solve(&problem, guess.view(), &[]).unwrap();
            assert_eq!(
                report(&solution).termination(),
                BvpTermination::Converged,
                "{backend:?} analytic={analytic}, residual={:?}",
                report(&solution).max_residual()
            );
            assert!(report(&solution).max_residual() <= T::Real::ONE);
            assert!(
                report(&solution).boundary_residual() <= T::Real::try_from_f64_lossy(tol).unwrap()
            );
            if tol < 1e-4 {
                assert!(report(&solution).refinements() > 0);
            }
            let mut value = [T::ZERO; 2];
            let mut derivative = [T::ZERO; 2];
            for i in 0..=40 {
                let x = T::Real::try_from_f64_lossy(i as f64 / 40.).unwrap();
                solution
                    .evaluate_with_derivative_into(x, &mut value, &mut derivative)
                    .unwrap();
                let expected = amplitude * T::from_real(x.exp());
                let bound = T::Real::try_from_f64_lossy(tol * 8.).unwrap();
                assert!(
                    (value[0] - expected).abs() < bound,
                    "value {:?} expected {:?}",
                    value[0],
                    expected
                );
                assert!((derivative[0] - expected).abs() < bound);
                assert!((derivative[1] - value[0]).abs() < bound);
            }
            assert!(
                report(&solution)
                    .accounting()
                    .total()
                    .unwrap()
                    .total_providers()
                    .unwrap()
                    > 0
            );
        }
    }
}
#[test]
fn exponential_all_scalar_domains_and_backends() {
    exponential(1f32, &backends(), 2e-3);
    exponential(Complex::new(1f32, 0.5), &backends(), 2e-3);
    // Unpreconditioned Krylov needs O(nodes) iterations per Newton step, so
    // deep refinement stays on direct backends and Krylov refines twice.
    exponential(1f64, &direct_backends(), 1e-7);
    exponential(Complex::new(1f64, 0.5), &direct_backends(), 1e-7);
    exponential(1f64, &[krylov()], 3e-5);
    exponential(Complex::new(1f64, 0.5), &[krylov()], 3e-5);
}
#[test]
fn plans_report_their_dimensions_and_validated_options() {
    let plan = BvpPlan::<f64>::new(
        2,
        1,
        &[0., 1.],
        BvpOptions::new()
            .with_max_nodes(8)
            .with_max_workspace_bytes(1 << 20),
    )
    .unwrap();
    assert_eq!((plan.dimension(), plan.parameter_dimension()), (2, 1));
    let validated = plan.options();
    assert_eq!(
        (validated.max_nodes(), validated.max_workspace_bytes()),
        (8, 1 << 20)
    );
    assert_eq!(validated.backend(), BvpBackend::Sparse { limits: None });
    // The options carry the documented default tolerances.
    let (residual, boundary) = validated.tolerances();
    assert_eq!((residual.absolute(), residual.relative()), (1e-3, 1e-3));
    assert_eq!((boundary.absolute(), boundary.relative()), (1e-3, 0.));
    let (residual, boundary) = BvpOptions::<f32>::new()
        .with_max_nodes(8)
        .with_max_workspace_bytes(1 << 20)
        .tolerances();
    assert_eq!((residual.absolute(), residual.relative()), (1e-3, 1e-3));
    assert_eq!((boundary.absolute(), boundary.relative()), (1e-3, 0.));
    let custom = options(BvpBackend::Dense, 1e-6).with_refinements(Budget::ONE);
    let plan = BvpPlan::<f64>::new(1, 0, &[0., 1.], custom.clone()).unwrap();
    assert_eq!(plan.options(), &custom);
}
#[test]
fn explicit_sparse_limits_bound_the_factorization() {
    let model = models::Exponential { amplitude: 1f64 };
    let guess = Matrix::from_vec(3, 2, vec![1.; 6]).unwrap();
    let solve = |limits| {
        let backend = BvpBackend::Sparse { limits };
        let plan = BvpPlan::new(2, 0, &[0., 0.5, 1.], options(backend, 1e-6)).unwrap();
        assert_eq!(plan.options().backend(), backend);
        plan.solve(&BvpProblem::analytic(&model), guess.view(), &[])
    };
    for limits in [None, Some(SparseLuLimits::new(4096, 1024 * 1024))] {
        let solution = solve(limits).unwrap();
        assert_eq!(report(&solution).termination(), BvpTermination::Converged);
    }
    // One factor entry cannot hold the block pattern of any mesh.
    let error = solve(Some(SparseLuLimits::new(1, 1024 * 1024))).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}
struct Bratu;
impl BvpSystem<f64> for Bratu {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[1];
        o[1] = -2. / 0.5f64.cosh().powi(2) * y[0].exp();
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[0];
        o[1] = b[0];
        Ok(())
    }
}
#[test]
fn nonlinear_bratu_matches_closed_form_and_refines() {
    let plan = BvpPlan::new(
        Bratu.dimension(),
        Bratu.parameter_dimension(),
        &[0., 0.5, 1.],
        options(BvpBackend::Dense, 1e-7),
    )
    .unwrap();
    let initial = Matrix::zeros(3, 2).unwrap();
    let s = plan
        .solve(&BvpProblem::numerical(&Bratu), initial.view(), &[])
        .unwrap();
    assert_eq!(report(&s).termination(), BvpTermination::Converged);
    let mut y = [0.; 2];
    for i in 0..=50 {
        let x = i as f64 / 50.;
        s.evaluate_into(x, &mut y).unwrap();
        let exact = 2. * (0.5f64.cosh() / (x - 0.5).cosh()).ln();
        assert!((y[0] - exact).abs() < 2e-7);
    }
    assert!(report(&s).refinements() > 0);
}
struct Eigen;
impl BvpSystem<f64> for Eigen {
    fn dimension(&self) -> usize {
        2
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[1];
        o[1] = -p[0] * p[0] * y[0];
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        p: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[0];
        o[1] = b[0];
        o[2] = a[1] - p[0];
        Ok(())
    }
}
impl BvpJacobian<f64> for Eigen {
    fn rhs_jacobian(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        j: &mut [f64],
        jp: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        j.copy_from_slice(&[0., 1., -p[0] * p[0], 0.]);
        jp.copy_from_slice(&[0., -2. * p[0] * y[0]]);
        Ok(())
    }
    fn boundary_jacobian(
        &self,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        a: &mut [f64],
        b: &mut [f64],
        p: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        a.copy_from_slice(&[1., 0., 0., 0., 0., 1.]);
        b.copy_from_slice(&[0., 0., 1., 0., 0., 0.]);
        p.copy_from_slice(&[0., 0., -1.]);
        Ok(())
    }
}
#[test]
fn normalized_eigenvalue_and_parameter_jacobians() {
    let mesh = [0., 0.25, 0.5, 0.75, 1.];
    let initial = Matrix::from_vec(
        5,
        2,
        mesh.iter()
            .flat_map(|x| [(3. * x).sin(), 3. * (3. * x).cos()])
            .collect(),
    )
    .unwrap();
    // Krylov with unknown parameters: `unknown_parameters_cover_all_scalar_domains`.
    for backend in direct_backends() {
        for analytic in [true, false] {
            let plan = BvpPlan::new(
                Eigen.dimension(),
                Eigen.parameter_dimension(),
                &mesh,
                options(backend, 1e-6),
            )
            .unwrap();
            let p = if analytic {
                BvpProblem::analytic(&Eigen)
            } else {
                BvpProblem::numerical(&Eigen)
            };
            let s = plan.solve(&p, initial.view(), &[3.]).unwrap();
            assert_eq!(
                report(&s).termination(),
                BvpTermination::Converged,
                "{backend:?} analytic={analytic}"
            );
            assert!((s.parameters()[0] - core::f64::consts::PI).abs() < 1e-6);
            let mut y = [0.; 2];
            s.evaluate_into(0.37, &mut y).unwrap();
            assert!((y[0] - (core::f64::consts::PI * 0.37).sin()).abs() < 1e-6);
        }
    }
}
struct Radial;
impl BvpSystem<f64> for Radial {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[1];
        o[1] = 6.;
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[1];
        o[1] = b[0] - 1.;
        Ok(())
    }
}
#[test]
fn singular_radial_endpoint_has_regular_dense_derivative() {
    let singular = Matrix::from_vec(2, 2, vec![0., 0., 0., -2.]).unwrap();
    let initial = Matrix::zeros(3, 2).unwrap();
    for backend in backends() {
        let plan = BvpPlan::new(
            Radial.dimension(),
            Radial.parameter_dimension(),
            &[0., 0.5, 1.],
            options(backend, 1e-6),
        )
        .unwrap()
        .with_singular(singular.view())
        .unwrap();
        let s = plan
            .solve(&BvpProblem::numerical(&Radial), initial.view(), &[])
            .unwrap();
        assert_eq!(
            report(&s).termination(),
            BvpTermination::Converged,
            "{backend:?}"
        );
        assert!(report(&s).regularity_residual() < 1e-10);
        let mut y = [0.; 2];
        let mut d = [0.; 2];
        for x in [0., 0.01, 0.25, 1.] {
            s.evaluate_with_derivative_into(x, &mut y, &mut d).unwrap();
            assert!((y[0] - x * x).abs() < 1e-8);
            assert!((y[1] - 2. * x).abs() < 1e-8);
            assert!((d[1] - 2.).abs() < 1e-8);
        }
    }
}
#[test]
fn refinement_limits_and_convergence_evidence() {
    let model = models::Exponential { amplitude: 1f64 };
    let initial = Matrix::zeros(3, 2).unwrap();
    let mesh = [0., 0.5, 1.];
    let limited = BvpOptions::new()
        .with_max_nodes(3)
        .with_max_workspace_bytes(1024 * 1024)
        .with_tolerances(
            ResidualTolerance::new(1e-9, 1e-9).unwrap(),
            ResidualTolerance::new(1e-9, 0.).unwrap(),
        );
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        limited,
    )
    .unwrap();
    let s = plan
        .solve(&BvpProblem::analytic(&model), initial.view(), &[])
        .unwrap();
    assert_eq!(report(&s).termination(), BvpTermination::NodeLimit);
    let coarse = report(&s).max_residual();
    let refined = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-9),
    )
    .unwrap()
    .solve(&BvpProblem::analytic(&model), initial.view(), &[])
    .unwrap();
    assert!(report(&refined).max_residual() < coarse);
    assert_eq!(report(&refined).termination(), BvpTermination::Converged);
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-9).with_refinements(Budget::ONE),
    )
    .unwrap();
    let s = plan
        .solve(&BvpProblem::analytic(&model), initial.view(), &[])
        .unwrap();
    assert_eq!(report(&s).termination(), BvpTermination::RefinementLimit);
}
#[test]
fn rejects_bad_mesh_shapes_storage_and_dense_coordinates() {
    let model = models::Exponential { amplitude: 1f64 };
    for mesh in [&[0., 0.][..], &[1., 0.], &[0., f64::NAN], &[0.]] {
        assert!(
            BvpPlan::<f64>::new(
                model.dimension(),
                model.parameter_dimension(),
                mesh,
                options(BvpBackend::Dense, 1e-5)
            )
            .is_err()
        );
    }
    assert!(
        BvpPlan::<f64>::new(
            model.dimension(),
            model.parameter_dimension(),
            &[0., 1.],
            BvpOptions::new()
                .with_max_nodes(3)
                .with_max_workspace_bytes(1)
        )
        .is_err()
    );
    let plan = BvpPlan::<f64>::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-5),
    )
    .unwrap();
    assert!(
        plan.solve(
            &BvpProblem::analytic(&model),
            Matrix::zeros(2, 1).unwrap().view(),
            &[]
        )
        .is_err()
    );
    assert!(
        plan.solve(
            &BvpProblem::analytic(&model),
            Matrix::zeros(2, 2).unwrap().view(),
            &[0.]
        )
        .is_err()
    );
    let s = plan
        .solve(
            &BvpProblem::analytic(&model),
            Matrix::zeros(2, 2).unwrap().view(),
            &[],
        )
        .unwrap();
    for x in [-1., 2., f64::NAN] {
        let mut out = [77.; 2];
        assert!(s.evaluate_into(x, &mut out).is_err());
        assert_eq!(out, [77.; 2]);
    }
    let mut out = [77.];
    assert!(s.evaluate_into(0.5, &mut out).is_err());
    assert_eq!(out, [77.]);
    let mut out = [77.; 2];
    assert!(
        BvpSolution::<f64>::new()
            .evaluate_into(0.5, &mut out)
            .is_err()
    );
    assert_eq!(out, [77.; 2]);
}
#[test]
fn plan_construction_validates_option_limits_and_tolerances() {
    use scioxide_core::SciOxideErrorKind;
    let model = models::Exponential { amplitude: 1f64 };
    let mesh = [0., 1.];
    let tolerance = |a: f64, r: f64| ResidualTolerance::new(a, r).unwrap();
    // The smallest admissible node limit and the mesh at that limit are accepted.
    assert!(
        BvpPlan::<f64>::new(
            model.dimension(),
            model.parameter_dimension(),
            &mesh,
            BvpOptions::new()
                .with_max_nodes(2)
                .with_max_workspace_bytes(1 << 26)
        )
        .is_ok()
    );
    for options in [
        BvpOptions::new()
            .with_max_nodes(1)
            .with_max_workspace_bytes(1 << 26),
        BvpOptions::new()
            .with_max_nodes(2)
            .with_max_workspace_bytes(0),
        BvpOptions::new()
            .with_max_nodes(64)
            .with_max_workspace_bytes(1 << 26)
            .with_tolerances(tolerance(0., 1e-3), tolerance(1e-3, 0.)),
        BvpOptions::new()
            .with_max_nodes(64)
            .with_max_workspace_bytes(1 << 26)
            .with_tolerances(tolerance(1e-3, 1e-3), tolerance(0., 0.)),
        BvpOptions::new()
            .with_max_nodes(64)
            .with_max_workspace_bytes(1 << 26)
            .with_tolerances(tolerance(1e-3, 1e-3), tolerance(1e-3, 1e-3)),
    ] {
        let Err(error) = BvpPlan::<f64>::new(
            model.dimension(),
            model.parameter_dimension(),
            &mesh,
            options,
        ) else {
            unreachable!("invalid options accepted")
        };
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
    // More initial nodes than the node limit.
    assert!(
        BvpPlan::<f64>::new(
            model.dimension(),
            model.parameter_dimension(),
            &[0., 0.5, 1.],
            BvpOptions::new()
                .with_max_nodes(2)
                .with_max_workspace_bytes(1 << 26)
        )
        .is_err()
    );
}
#[test]
fn moved_and_cloned_plans_solve_with_an_earlier_workspace() {
    fn shareable<P: Clone + Send + Sync>(_: &P) {}
    let model = models::Exponential { amplitude: 1f64 };
    let mesh = [0., 0.5, 1.];
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap();
    shareable(&plan);
    let problem = BvpProblem::analytic(&model);
    let guess = Matrix::zeros(3, 2).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let reference = plan.solve(&problem, guess.view(), &[]).unwrap();
    let clone = plan.clone();
    // Moving the plan to the heap changes its address after the workspace was created.
    let moved = Box::new(plan);
    let mut out = BvpSolution::new();
    for plan in [&*moved, &clone] {
        let published = plan
            .solve_into(
                &problem,
                guess.view(),
                &[],
                &mut out,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(published.termination(), BvpTermination::Converged);
        assert_eq!(out.report(), Some(published));
        assert_eq!(out.mesh(), reference.mesh());
        assert_eq!(out.values(), reference.values());
        assert_eq!(
            published.accounting().total().unwrap(),
            report(&reference).accounting().total().unwrap()
        );
    }
}
/// `y'' = y` with `y(0) = 1` and `y(1) = e`, failing its RHS while `fail` is set.
struct Failing(core::cell::Cell<bool>);
impl BvpSystem<f64> for Failing {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.0.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "outside the model domain",
            ));
        }
        out.copy_from_slice(&[y[1], y[0]]);
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[a[0] - 1., b[0] - 1f64.exp()]);
        Ok(())
    }
}
#[test]
fn failed_and_rejected_attempts_leave_the_published_solution_unchanged() {
    use scioxide_core::{AttemptOutcome, Capacity};
    let model = Failing(core::cell::Cell::new(false));
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 0.5, 1.],
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap();
    let problem = BvpProblem::numerical(&model);
    let requirements = plan.workspace_requirements(&problem).unwrap();
    assert_eq!(requirements.nodes(), 3);
    assert!(requirements.required_bytes() <= requirements.byte_limit());
    let mut workspace = BvpWorkspace::new();
    assert!(workspace.last_attempt().is_none());
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let solve = |guess: MatrixView<'_, f64>,
                 out: &mut BvpSolution<f64>,
                 workspace: &mut BvpWorkspace<f64>| {
        plan.solve_into(
            &problem,
            guess,
            &[],
            out,
            workspace,
            ExecutionContext::sequential(),
        )
    };
    let mut out = BvpSolution::new();
    assert!(out.report().is_none());
    let guess = Matrix::zeros(3, 2).unwrap();
    let published = solve(guess.view(), &mut out, &mut workspace).unwrap();
    assert_eq!(published.termination(), BvpTermination::Converged);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.progress(), Some(&published));
    assert!(attempt.epoch().is_some());
    assert_eq!(out.report(), Some(published));
    let snapshot = out.clone();
    let unchanged = |out: &BvpSolution<f64>| {
        assert_eq!(out.report(), snapshot.report());
        assert_eq!(out.mesh(), snapshot.mesh());
        assert_eq!(out.values(), snapshot.values());
        assert_eq!(out.derivatives(), snapshot.derivatives());
        assert_eq!(out.residuals(), snapshot.residuals());
    };
    // Preflight rejection: a guess of the wrong shape runs no callback and starts no epoch.
    assert!(
        solve(
            Matrix::zeros(2, 2).unwrap().view(),
            &mut out,
            &mut workspace
        )
        .is_err()
    );
    unchanged(&out);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
    assert!(attempt.progress().is_none());
    // Mid-solve provider failure: the epoch fails without publishing.
    model.0.set(true);
    assert!(solve(guess.view(), &mut out, &mut workspace).is_err());
    unchanged(&out);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert!(attempt.epoch().is_some());
    assert!(attempt.progress().is_none());
    assert!(attempt.counts().total().unwrap().total_providers().unwrap() > 0);
    // The same workspace serves the recovered problem again.
    model.0.set(false);
    assert_eq!(
        solve(guess.view(), &mut out, &mut workspace).unwrap(),
        published
    );
}
#[test]
fn strided_guess_and_prepared_reuse() {
    let model = models::Exponential { amplitude: 1f64 };
    let mesh = [0., 0.5, 1.];
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-3),
    )
    .unwrap();
    let values = [1., 88., 1., 88., 1., 88., 1., 88., 1., 88., 1., 88.];
    let guess = MatrixView::from_slice_with_strides(3, 2, &values, 4, 2).unwrap();
    let p = BvpProblem::analytic(&model);
    let mut workspace = plan.workspace(&p).unwrap();
    let mut solution = BvpSolution::new();
    for _ in 0..2 {
        let report = plan
            .solve_into(
                &p,
                guess,
                &[],
                &mut solution,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), BvpTermination::Converged);
        assert_eq!(solution.report(), Some(report));
    }
}

struct Broken {
    calls: core::sync::atomic::AtomicUsize,
    mode: usize,
}
impl BvpSystem<f64> for Broken {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if self.mode != 1 {
            out[0] = if self.mode == 2 { f64::INFINITY } else { 0. };
        }
        Ok(())
    }
    fn boundary(
        &self,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = if self.mode == 3 { 1. } else { 0. };
        Ok(())
    }
}
#[test]
fn incomplete_nonfinite_and_inconsistent_providers_fail_explicitly() {
    for mode in [1, 2, 3] {
        let model = Broken {
            calls: core::sync::atomic::AtomicUsize::new(0),
            mode,
        };
        let plan = BvpPlan::new(
            model.dimension(),
            model.parameter_dimension(),
            &[0., 1.],
            options(BvpBackend::Dense, 1e-4),
        )
        .unwrap();
        assert!(
            plan.solve(
                &BvpProblem::numerical(&model),
                Matrix::zeros(2, 1).unwrap().view(),
                &[]
            )
            .is_err()
        );
    }
}
#[test]
fn nested_rhs_budget_and_cancellation_prevent_callbacks() {
    use scioxide_core::{BudgetKind, CancellationToken, EvaluationRole, SolveBudgets, SolveState};
    let model = Broken {
        calls: core::sync::atomic::AtomicUsize::new(0),
        mode: 0,
    };
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-4),
    )
    .unwrap();
    let p = BvpProblem::numerical(&model);
    let mut w = plan.workspace(&p).unwrap();
    let mut out = BvpSolution::new();
    let initial = Matrix::zeros(2, 1).unwrap();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
    let mut state = SolveState::new();
    let mut c = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    assert!(
        plan.solve_into(&p, initial.view(), &[], &mut out, &mut w, &mut c)
            .is_err()
    );
    assert_eq!(model.calls.load(core::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        c.counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::RightHandSide),
        1
    );
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            parent(),
            &mut budgets,
        )
        .unwrap();
    assert!(
        plan.solve_into(&p, initial.view(), &[], &mut out, &mut w, &mut c)
            .is_err()
    );
    assert_eq!(model.calls.load(core::sync::atomic::Ordering::Relaxed), 1);
    assert!(out.report().is_none());
}
#[test]
fn resonant_singular_endpoint_rejects_incompatible_rhs() {
    let model = models::Exponential { amplitude: 1f64 };
    let singular = Matrix::identity(2).unwrap();
    let initial = Matrix::from_vec(2, 2, vec![1., 1., 1., 1.]).unwrap();
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-4),
    )
    .unwrap()
    .with_singular(singular.view())
    .unwrap();
    // Projection gives y(a)=0, but boundary values require y(a)=1. The result
    // cannot be reported converged, irrespective of whether Newton has full rank.
    match plan.solve(&BvpProblem::analytic(&model), initial.view(), &[]) {
        Ok(s) => assert_ne!(report(&s).termination(), BvpTermination::Converged),
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential_bits_and_counts() {
    use core::num::NonZeroUsize;
    use scioxide_core::ThreadExecutor;
    let model = models::Exponential {
        amplitude: Complex::new(1f64, 0.5),
    };
    let mesh = [0., 0.5, 1.];
    let initial = Matrix::from_vec(3, 2, vec![Complex::new(1., 0.5); 6]).unwrap();
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap();
    let p = BvpProblem::analytic(&model);
    let solve = |execution: ExecutionContext<'_>| {
        let mut w = plan.workspace(&p).unwrap();
        let mut solution = BvpSolution::new();
        let published = plan
            .solve_into(&p, initial.view(), &[], &mut solution, &mut w, execution)
            .unwrap();
        assert_eq!(solution.report(), Some(published));
        solution
    };
    let serial = solve(ExecutionContext::sequential());
    // Node evaluation is sequential; an injected executor changes no bits or counts.
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let parallel = solve(ExecutionContext::new(&pool));
    assert_eq!(report(&parallel).termination(), BvpTermination::Converged);
    assert_eq!(serial.values(), parallel.values());
    assert_eq!(serial.residuals(), parallel.residuals());
    assert_eq!(serial.mesh(), parallel.mesh());
    assert_eq!(
        report(&serial)
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap(),
        report(&parallel)
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap()
    );
}

#[test]
fn fixed_mesh_state_error_has_fourth_order_convergence() {
    let model = models::Exponential { amplitude: 1f64 };
    let mut errors = Vec::new();
    for nodes in [5, 9, 17] {
        let mesh: Vec<_> = (0..nodes).map(|i| i as f64 / (nodes - 1) as f64).collect();
        let initial = Matrix::zeros(nodes, 2).unwrap();
        let opts = BvpOptions::new()
            .with_max_nodes(nodes)
            .with_max_workspace_bytes(16 * 1024 * 1024)
            .with_tolerances(
                ResidualTolerance::new(1e-10, 1e-10).unwrap(),
                ResidualTolerance::new(1e-10, 0.).unwrap(),
            );
        let s = BvpPlan::new(model.dimension(), model.parameter_dimension(), &mesh, opts)
            .unwrap()
            .solve(&BvpProblem::analytic(&model), initial.view(), &[])
            .unwrap();
        assert_eq!(report(&s).termination(), BvpTermination::NodeLimit);
        let mut y = [0.; 2];
        s.evaluate_into(0.37, &mut y).unwrap();
        errors.push((y[0] - 0.37f64.exp()).abs());
    }
    assert!(errors[0] / errors[1] > 10.);
    assert!(errors[1] / errors[2] > 10.);
}
struct ParameterModel<T: NumericScalar>(T);
impl<T: NumericScalar> BvpSystem<T> for ParameterModel<T> {
    fn dimension(&self) -> usize {
        1
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: T::Real,
        _: &[T],
        p: &[T],
        o: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = p[0];
        Ok(())
    }
    fn boundary(
        &self,
        a: &[T],
        b: &[T],
        _: &[T],
        o: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[0];
        o[1] = b[0] - self.0;
        Ok(())
    }
}
impl<T: NumericScalar> BvpJacobian<T> for ParameterModel<T> {
    fn rhs_jacobian(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        y: &mut [T],
        p: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        y[0] = T::ZERO;
        p[0] = T::ONE;
        Ok(())
    }
    fn boundary_jacobian(
        &self,
        _: &[T],
        _: &[T],
        _: &[T],
        a: &mut [T],
        b: &mut [T],
        p: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        a.copy_from_slice(&[T::ONE, T::ZERO]);
        b.copy_from_slice(&[T::ZERO, T::ONE]);
        p.fill(T::ZERO);
        Ok(())
    }
}
fn unknown_parameter<T: NumericScalar>(value: T, tol: f64) {
    let model = ParameterModel(value);
    let initial = Matrix::zeros(2, 1).unwrap();
    let mesh = [T::Real::ZERO, T::Real::ONE];
    for backend in backends() {
        for analytic in [false, true] {
            let plan = BvpPlan::new(
                model.dimension(),
                model.parameter_dimension(),
                &mesh,
                options(backend, tol),
            )
            .unwrap();
            let p = if analytic {
                BvpProblem::analytic(&model)
            } else {
                BvpProblem::numerical(&model)
            };
            let s = plan.solve(&p, initial.view(), &[T::ZERO]).unwrap();
            assert_eq!(report(&s).termination(), BvpTermination::Converged);
            assert!((s.parameters()[0] - value).abs() < T::Real::try_from_f64_lossy(tol).unwrap());
            let mut y = [T::ZERO];
            s.evaluate_into(T::Real::HALF, &mut y).unwrap();
            assert!(
                (y[0] - value * T::from_real(T::Real::HALF)).abs()
                    < T::Real::try_from_f64_lossy(tol).unwrap()
            );
        }
    }
}
#[test]
fn unknown_parameters_cover_all_scalar_domains() {
    unknown_parameter(2f64, 1e-7);
    unknown_parameter(2f32, 2e-3);
    unknown_parameter(Complex::new(2f64, -0.3), 1e-7);
    unknown_parameter(Complex::new(2f32, -0.3), 2e-3);
}
struct RadialCosine;
impl BvpSystem<f64> for RadialCosine {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        x: f64,
        y: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[1];
        o[1] = -x.cos() - 2. * if x == 0. { 1. } else { x.sin() / x };
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[1];
        o[1] = b[0] - 1f64.cos();
        Ok(())
    }
}
#[test]
fn nonpolynomial_singular_solution_refines_and_is_regular() {
    let matrix = Matrix::from_vec(2, 2, vec![0., 0., 0., -2.]).unwrap();
    let guess = Matrix::from_vec(3, 2, vec![1., 0., 1., 0., 1., 0.]).unwrap();
    let plan = BvpPlan::new(
        RadialCosine.dimension(),
        RadialCosine.parameter_dimension(),
        &[0., 0.5, 1.],
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap()
    .with_singular(matrix.view())
    .unwrap();
    let s = plan
        .solve(&BvpProblem::numerical(&RadialCosine), guess.view(), &[])
        .unwrap();
    assert_eq!(report(&s).termination(), BvpTermination::Converged);
    assert!(report(&s).refinements() > 0);
    assert!(report(&s).regularity_residual() < 1e-12);
    let mut y = [0.; 2];
    for x in [0., 1e-4, 0.3, 1.] {
        s.evaluate_into(x, &mut y).unwrap();
        assert!((y[0] - x.cos()).abs() < 2e-6);
        assert!((y[1] + x.sin()).abs() < 2e-6);
    }
}

#[test]
fn newton_options_report_their_defaults_and_settings() {
    let defaults = BvpNewtonOptions::<f64>::new();
    assert_eq!(defaults.max_iterations(), Budget::try_from(100).unwrap());
    let (armijo, minimum) = defaults.backtracking();
    assert!((armijo - 1e-4).abs() <= 1e-4 * f64::EPSILON * 4.);
    assert!((minimum - f64::EPSILON.powf(2. / 3.)).abs() <= minimum * 1e-12);
    assert!((defaults.nonmonotone() - 0.85).abs() <= f64::EPSILON);
    assert!(matches!(
        defaults.forcing(),
        scioxide_rootfind::NewtonKrylovForcing::EisenstatWalker { .. }
    ));
    let forcing = scioxide_rootfind::NewtonKrylovForcing::Constant(0.25);
    let custom = defaults
        .with_max_iterations(Budget::ONE)
        .with_backtracking(0.5, 1e-3)
        .with_nonmonotone(0.)
        .with_forcing(forcing);
    assert_eq!(custom.max_iterations(), Budget::ONE);
    assert_eq!(custom.backtracking(), (0.5, 1e-3));
    assert_eq!(custom.nonmonotone(), 0.);
    assert_eq!(custom.forcing(), forcing);
    let options = BvpOptions::<f64>::new()
        .with_max_nodes(8)
        .with_max_workspace_bytes(1 << 20)
        .with_newton(custom.clone());
    assert_eq!(options.newton(), &custom);
}
#[test]
fn newton_iteration_limit_does_not_size_the_workspace() {
    let model = models::Exponential { amplitude: 1f64 };
    let options = options(BvpBackend::Dense, 1e-3).with_newton(
        BvpNewtonOptions::new().with_max_iterations(Budget::try_from(usize::MAX).unwrap()),
    );
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options,
    )
    .unwrap();
    assert!(plan.workspace(&BvpProblem::analytic(&model)).is_ok());
}

#[test]
fn newton_options_bound_the_iteration_and_validate_the_line_search() {
    let model = ScaledNonlinear(1f64);
    let mesh: Vec<f64> = (0..=10).map(|i| f64::from(i) / 10.).collect();
    let guess = Matrix::zeros(11, 2).unwrap();
    let solve = |newton: BvpNewtonOptions<f64>| {
        let plan = BvpPlan::new(
            model.dimension(),
            model.parameter_dimension(),
            &mesh,
            options(BvpBackend::Dense, 1e-8).with_newton(newton),
        )?;
        plan.solve(&BvpProblem::numerical(&model), guess.view(), &[])
            .map(|solution| report(&solution).termination())
    };
    assert_eq!(
        solve(BvpNewtonOptions::new()).unwrap(),
        BvpTermination::Converged
    );
    assert_eq!(
        solve(BvpNewtonOptions::new().with_max_iterations(Budget::ONE)).unwrap(),
        BvpTermination::Nonlinear(scioxide_rootfind::RootTermination::IterationLimit)
    );
    // A monotone search that may shorten the step only to a quarter gives up.
    assert_eq!(
        solve(
            BvpNewtonOptions::new()
                .with_nonmonotone(0.)
                .with_backtracking(0.5, 0.25)
        )
        .unwrap(),
        BvpTermination::Nonlinear(scioxide_rootfind::RootTermination::GlobalizationFailure)
    );
    for invalid in [
        BvpNewtonOptions::new().with_backtracking(1.5, 0.25),
        BvpNewtonOptions::new().with_nonmonotone(-1.),
    ] {
        assert_eq!(
            solve(invalid).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }
}

#[test]
fn workspace_reuse_checks_dimensions_backend_and_options() {
    let pmodel = ParameterModel(2f64);
    let parameter_plan = BvpPlan::new(
        pmodel.dimension(),
        pmodel.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-4),
    )
    .unwrap();
    let mut w = parameter_plan
        .workspace(&BvpProblem::analytic(&pmodel))
        .unwrap();
    let model = models::Exponential { amplitude: 1f64 };
    for backend in backends() {
        let plan = BvpPlan::new(
            model.dimension(),
            model.parameter_dimension(),
            &[0., 1.],
            options(backend, 1e-4),
        )
        .unwrap();
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                parent(),
                &mut budgets,
            )
            .unwrap();
        let mut s = BvpSolution::new();
        let r = plan
            .solve_into(
                &BvpProblem::analytic(&model),
                Matrix::zeros(2, 2).unwrap().view(),
                &[],
                &mut s,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report(&s), r);
        assert_eq!(report(&s).termination(), BvpTermination::Converged);
        assert_eq!(
            report(&s).accounting().total().unwrap(),
            context.counts().total().unwrap()
        );
    }
}

struct ScaledRadial<T: NumericScalar>(T);
impl<T: NumericScalar> BvpSystem<T> for ScaledRadial<T> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        o: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[1];
        o[1] = self.0 * T::from_real(T::Real::try_from_f64_lossy(6.)?);
        Ok(())
    }
    fn boundary(
        &self,
        a: &[T],
        b: &[T],
        _: &[T],
        o: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[1];
        o[1] = b[0] - self.0;
        Ok(())
    }
}
fn singular_domain<T: NumericScalar>(amplitude: T, tol: f64) {
    let model = ScaledRadial(amplitude);
    let matrix = Matrix::from_vec(
        2,
        2,
        vec![
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::from_real(T::Real::try_from_f64_lossy(-2.).unwrap()),
        ],
    )
    .unwrap();
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[T::Real::ZERO, T::Real::HALF, T::Real::ONE],
        options(BvpBackend::Dense, tol),
    )
    .unwrap()
    .with_singular(matrix.view())
    .unwrap();
    let s = plan
        .solve(
            &BvpProblem::numerical(&model),
            Matrix::zeros(3, 2).unwrap().view(),
            &[],
        )
        .unwrap();
    assert_eq!(report(&s).termination(), BvpTermination::Converged);
    let mut y = [T::ZERO; 2];
    let mut d = [T::ZERO; 2];
    s.evaluate_with_derivative_into(T::Real::HALF, &mut y, &mut d)
        .unwrap();
    let threshold = T::Real::try_from_f64_lossy(tol).unwrap();
    assert!(
        (y[0] - amplitude * T::from_real(T::Real::try_from_f64_lossy(0.25).unwrap())).abs()
            < threshold
    );
    assert!((y[1] - amplitude).abs() < threshold);
    assert!(
        (d[1] - amplitude * T::from_real(T::Real::try_from_f64_lossy(2.).unwrap())).abs()
            < threshold
    );
    assert!(report(&s).regularity_residual() < threshold);
}
#[test]
fn singular_endpoints_cover_real_and_complex_widths() {
    singular_domain(1f32, 2e-3);
    singular_domain(Complex::new(1f32, 0.5), 2e-3);
    singular_domain(Complex::new(1f64, 0.5), 1e-7);
}

#[test]
fn shared_subdivision_budget_preserves_assessed_mesh() {
    use scioxide_core::{AttemptOutcome, BudgetKind, SolveBudgets, SolveState};
    let model = models::Exponential { amplitude: 1f64 };
    let mesh = [0., 0.5, 1.];
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-7),
    )
    .unwrap();
    let p = BvpProblem::analytic(&model);
    let mut w = plan.workspace(&p).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Subdivision, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    let mut s = BvpSolution::new();
    let r = plan
        .solve_into(
            &p,
            Matrix::zeros(3, 2).unwrap().view(),
            &[],
            &mut s,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report(&s).termination(),
        BvpTermination::ResourceLimit(BudgetKind::Subdivision)
    );
    // A stopped attempt still publishes and records its report.
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.progress(), Some(&r));
    assert_eq!(s.mesh(), mesh);
    assert_eq!(report(&s).refinements(), 0);
    assert!(report(&s).max_residual() > 1.);
    assert_eq!(
        report(&s).accounting().total().unwrap(),
        context.counts().total().unwrap()
    );
}

#[test]
fn byte_limit_during_refinement_publishes_the_assessed_mesh() {
    use scioxide_core::AttemptOutcome;
    let model = models::Exponential { amplitude: 1f64 };
    let problem = BvpProblem::analytic(&model);
    let mesh = [0., 0.5, 1.];
    let plan_with = |bytes| {
        BvpPlan::new(
            model.dimension(),
            model.parameter_dimension(),
            &mesh,
            BvpOptions::new()
                .with_max_nodes(512)
                .with_max_workspace_bytes(bytes)
                .with_backend(BvpBackend::Dense)
                .with_tolerances(
                    ResidualTolerance::new(1e-9, 1e-9).unwrap(),
                    ResidualTolerance::new(1e-9, 0.).unwrap(),
                ),
        )
    };
    // The dense bound is independent of the limit, so the initial mesh fits exactly.
    let initial = plan_with(usize::MAX)
        .unwrap()
        .workspace_requirements(&problem)
        .unwrap()
        .required_bytes();
    assert!(plan_with(initial - 1).is_err());
    let plan = plan_with(initial).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let mut solution = BvpSolution::new();
    let report = plan
        .solve_into(
            &problem,
            Matrix::zeros(3, 2).unwrap().view(),
            &[],
            &mut solution,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), BvpTermination::MemoryLimit);
    assert!(!report.is_success());
    assert_eq!((report.nodes(), report.refinements()), (3, 0));
    assert!(report.max_residual() > 1.);
    assert!(report.workspace_bytes() <= initial);
    assert_eq!(solution.report(), Some(report));
    assert_eq!(solution.mesh(), mesh);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.progress(), Some(&report));
    // A limit that admits one refinement stops at the refined mesh.
    let five = BvpPlan::<f64>::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 0.25, 0.5, 0.75, 1.],
        BvpOptions::new()
            .with_max_nodes(512)
            .with_max_workspace_bytes(usize::MAX)
            .with_backend(BvpBackend::Dense),
    )
    .unwrap()
    .workspace_requirements(&problem)
    .unwrap()
    .required_bytes();
    assert!(five > initial);
    let report = plan_with(five)
        .unwrap()
        .solve(&problem, Matrix::zeros(3, 2).unwrap().view(), &[])
        .unwrap()
        .report()
        .unwrap();
    assert_eq!(report.termination(), BvpTermination::MemoryLimit);
    assert_eq!((report.nodes(), report.refinements()), (5, 1));
}

#[test]
fn nonfinite_initial_guess_is_an_input_failure() {
    let model = models::Exponential { amplitude: 1f64 };
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-3),
    )
    .unwrap();
    let mut guess = Matrix::zeros(2, 2).unwrap();
    guess.as_mut_slice()[0] = f64::NAN;
    let Err(error) = plan.solve(&BvpProblem::analytic(&model), guess.view(), &[]) else {
        unreachable!("invalid initial guess accepted")
    };
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
struct LargeConstant;
impl BvpSystem<f64> for LargeConstant {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = 2e307;
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        _: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = a[0];
        Ok(())
    }
}
#[test]
fn infinite_normalizer_cannot_hide_residuals() {
    let options = BvpOptions::new()
        .with_max_nodes(4)
        .with_max_workspace_bytes(1024 * 1024)
        .with_tolerances(
            ResidualTolerance::new(1.7e308, 1.).unwrap(),
            ResidualTolerance::new(1e307, 0.).unwrap(),
        );
    let plan = BvpPlan::new(
        LargeConstant.dimension(),
        LargeConstant.parameter_dimension(),
        &[0., 1.],
        options,
    )
    .unwrap();
    let guess = Matrix::from_vec(2, 1, vec![0., 2e307]).unwrap();
    let Err(error) = plan.solve(&BvpProblem::numerical(&LargeConstant), guess.view(), &[]) else {
        unreachable!("overflowing normalizer accepted")
    };
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Intermediate,
            ..
        }
    ));
}

struct NestedProvider(core::sync::atomic::AtomicUsize);
impl BvpSystem<f64> for NestedProvider {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        c.record_provider(scioxide_core::EvaluationRole::Function)?
            .require()?;
        self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        out[0] = 0.;
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = a[0];
        Ok(())
    }
}
#[test]
fn nested_provider_work_obeys_parent_budget() {
    use scioxide_core::{AttemptOutcome, BudgetKind, EvaluationRole, SolveBudgets, SolveState};
    let model = NestedProvider(core::sync::atomic::AtomicUsize::new(0));
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-3),
    )
    .unwrap();
    let problem = BvpProblem::numerical(&model);
    let mut w = plan.workspace(&problem).unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
    let mut c = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    let mut out = BvpSolution::new();
    assert!(
        plan.solve_into(
            &problem,
            Matrix::zeros(2, 1).unwrap().view(),
            &[],
            &mut out,
            &mut w,
            &mut c
        )
        .is_err()
    );
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert!(attempt.progress().is_none());
    assert!(out.report().is_none());
    assert_eq!(model.0.load(core::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(
        c.counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::RightHandSide),
        1
    );
    assert_eq!(
        c.counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        0
    );
}

struct ScaledNonlinear<S>(S);
impl<S: RealScalar + NumericScalar<Real = S>> BvpSystem<S> for ScaledNonlinear<S> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        _: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = -self.0 * (S::ONE + S::try_from_f64_lossy(0.1)? * (y[0] / self.0).sin());
        Ok(())
    }
    fn boundary(
        &self,
        a: &[S],
        b: &[S],
        _: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[a[0], b[0]]);
        Ok(())
    }
}
fn scale_invariance<S: RealScalar + NumericScalar<Real = S>>(
    scales: &[f64],
    tol: f64,
    backend: BvpBackend,
) {
    let real = |x| S::try_from_f64_lossy(x).unwrap();
    let mesh: Vec<S> = (0..=10).map(|i| real(f64::from(i) / 10.)).collect();
    let guess = Matrix::zeros(11, 2).unwrap();
    let mut reference: Option<(usize, Vec<S>)> = None;
    for &scale in scales {
        let model = ScaledNonlinear(real(scale));
        for scale_tolerance in [true, false] {
            let absolute = real(if scale_tolerance { scale * tol } else { tol });
            let opts = options(backend, tol).with_tolerances(
                ResidualTolerance::new(absolute, real(tol)).unwrap(),
                ResidualTolerance::new(absolute, S::ZERO).unwrap(),
            );
            let plan =
                BvpPlan::new(model.dimension(), model.parameter_dimension(), &mesh, opts).unwrap();
            let result = plan
                .solve(&BvpProblem::numerical(&model), guess.view(), &[])
                .unwrap();
            assert_eq!(
                report(&result).termination(),
                BvpTermination::Converged,
                "scale={scale} scaled_tolerance={scale_tolerance} residual={:?}",
                report(&result).max_residual()
            );
            assert!(report(&result).max_residual() <= S::ONE);
            let mut samples = Vec::new();
            for i in 0..=40 {
                let mut value = [S::ZERO; 2];
                result
                    .evaluate_into(real(f64::from(i) / 40.), &mut value)
                    .unwrap();
                samples.extend(value.iter().map(|&v| v / real(scale)));
            }
            if let Some((nodes, values)) = &reference {
                if scale_tolerance {
                    assert_eq!(result.mesh().len(), *nodes);
                }
                for (&actual, &expected) in samples.iter().zip(values) {
                    assert!((actual - expected).abs() <= real(8. * tol));
                }
            } else {
                reference = Some((result.mesh().len(), samples));
            }
        }
    }
}
#[test]
fn nonlinear_bvp_is_invariant_under_state_units() {
    for backend in direct_backends() {
        scale_invariance::<f64>(&[1., 1e6, 1e9], 1e-6, backend);
        scale_invariance::<f32>(&[1., 1e3], 2e-5, backend);
    }
}
struct Troesch;
impl BvpSystem<f64> for Troesch {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = 12. * (12. * y[0]).sinh();
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[a[0], b[0] - 1.]);
        Ok(())
    }
}
#[test]
fn troesch_overflow_trials_publish_typed_nonlinear_termination() {
    let mesh = [0., 0.25, 0.5, 0.75, 1.];
    let guess = Matrix::from_vec(5, 2, mesh.iter().flat_map(|&x| [x, 1.]).collect()).unwrap();
    let plan = BvpPlan::new(
        Troesch.dimension(),
        Troesch.parameter_dimension(),
        &mesh,
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap();
    let solution = plan
        .solve(&BvpProblem::numerical(&Troesch), guess.view(), &[])
        .unwrap();
    assert!(
        matches!(
            report(&solution).termination(),
            BvpTermination::Converged | BvpTermination::Nonlinear(_)
        ),
        "{:?}",
        report(&solution).termination()
    );
    assert!(solution.values().iter().all(|v| v.is_finite()));
}

struct BoundedTrial;
impl BvpSystem<f64> for BoundedTrial {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = if y[0] > 1. { f64::INFINITY } else { 0. };
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = a[0] - 2.;
        Ok(())
    }
}
impl BvpJacobian<f64> for BoundedTrial {
    fn rhs_jacobian(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state.fill(0.);
        parameters.fill(0.);
        Ok(())
    }
    fn boundary_jacobian(
        &self,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        a: &mut [f64],
        b: &mut [f64],
        p: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        a.fill(1.);
        b.fill(0.);
        p.fill(0.);
        Ok(())
    }
}
#[test]
fn exhausted_nonfinite_trials_keep_a_finite_partial_solution() {
    let plan = BvpPlan::new(
        BoundedTrial.dimension(),
        BoundedTrial.parameter_dimension(),
        &[0., 1.],
        options(BvpBackend::Dense, 1e-6),
    )
    .unwrap();
    let initial = Matrix::zeros(2, 1).unwrap();
    let result = plan
        .solve(&BvpProblem::analytic(&BoundedTrial), initial.view(), &[])
        .unwrap();
    assert!(matches!(
        report(&result).termination(),
        BvpTermination::Nonlinear(_)
    ));
    assert!(
        result
            .values()
            .iter()
            .all(|&value| value.is_finite() && value <= 1.)
    );
    assert!(report(&result).accounting().total().unwrap().rejected() > 0);
}

struct LargeStateParameter;
impl BvpSystem<f64> for LargeStateParameter {
    fn dimension(&self) -> usize {
        1
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        p: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = 1e9 * p[0].exp();
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[a[0], b[0] - 2e9]);
        Ok(())
    }
}
#[test]
fn parameter_probe_units_are_independent_of_large_states() {
    let initial = Matrix::zeros(2, 1).unwrap();
    for backend in backends() {
        let opts = options(backend, 1e-6).with_tolerances(
            ResidualTolerance::new(1e-6, 1e-6).unwrap(),
            ResidualTolerance::new(1e-3, 0.).unwrap(),
        );
        let plan = BvpPlan::new(
            LargeStateParameter.dimension(),
            LargeStateParameter.parameter_dimension(),
            &[0., 1.],
            opts,
        )
        .unwrap();
        let solution = plan
            .solve(
                &BvpProblem::numerical(&LargeStateParameter),
                initial.view(),
                &[0.],
            )
            .unwrap();
        assert_eq!(
            report(&solution).termination(),
            BvpTermination::Converged,
            "{backend:?}"
        );
        assert!((solution.parameters()[0] - 2_f64.ln()).abs() < 1e-6);
    }
}

/// Boundaries delimit Newton assemblies; the intervals between them include the
/// complete preceding trial evaluation, scaling refresh and correction verification.
struct CountedBratu {
    rhs: core::cell::Cell<usize>,
    jacobians: core::cell::Cell<usize>,
    sweeps: core::cell::RefCell<Vec<(usize, usize)>>,
}
impl BvpSystem<f64> for CountedBratu {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs.set(self.rhs.get() + 1);
        out.copy_from_slice(&[y[1], -y[0].exp()]);
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        b: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[a[0], b[0]]);
        Ok(())
    }
}
impl BvpJacobian<f64> for CountedBratu {
    fn rhs_jacobian(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        state: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        state.copy_from_slice(&[0., 1., -y[0].exp(), 0.]);
        Ok(())
    }
    fn boundary_jacobian(
        &self,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        a: &mut [f64],
        b: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        a.copy_from_slice(&[1., 0., 0., 0.]);
        b.copy_from_slice(&[0., 0., 1., 0.]);
        self.sweeps
            .borrow_mut()
            .push((self.rhs.get(), self.jacobians.get()));
        Ok(())
    }
}
#[test]
fn direct_newton_reuses_each_collocation_evaluation_and_jacobian_block() {
    for backend in direct_backends() {
        for nodes in [16, 64] {
            let model = CountedBratu {
                rhs: core::cell::Cell::new(0),
                jacobians: core::cell::Cell::new(0),
                sweeps: core::cell::RefCell::new(Vec::with_capacity(32)),
            };
            let mesh: Vec<_> = (0..nodes)
                .map(|i| f64::from(i) / f64::from(nodes - 1))
                .collect();
            let nodes = mesh.len();
            let plan = BvpPlan::new(
                model.dimension(),
                model.parameter_dimension(),
                &mesh,
                options(backend, 1e-4),
            )
            .unwrap();
            let guess =
                Matrix::from_vec(nodes, 2, (0..nodes).flat_map(|_| [0.05, 0.]).collect()).unwrap();
            let solution = plan
                .solve(&BvpProblem::analytic(&model), guess.view(), &[])
                .unwrap();
            assert_eq!(report(&solution).termination(), BvpTermination::Converged);
            assert_eq!(report(&solution).refinements(), 0);
            assert_eq!(
                report(&solution).accounting().total().unwrap().rejected(),
                0
            );
            let collocation_points = 2 * nodes - 1;
            let mut previous = (0, 0);
            let sweeps = model.sweeps.borrow();
            assert!(sweeps.len() >= 2);
            for &counts in sweeps.iter() {
                // Includes setup for the first sweep, then complete undamped iterations.
                assert_eq!(counts.0 - previous.0, 4 * nodes - 3);
                assert_eq!(counts.1 - previous.1, collocation_points + 1);
                assert!(counts.0 - previous.0 <= 2 * collocation_points);
                assert!(counts.1 - previous.1 <= 2 * collocation_points);
                previous = counts;
            }
            // The final accepted trial is assessed from the same retained evaluation.
            assert_eq!(model.rhs.get() - previous.0, 4 * nodes - 3);
            assert_eq!(
                report(&solution)
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(scioxide_core::EvaluationRole::RightHandSide),
                model.rhs.get()
            );
            assert_eq!(
                report(&solution)
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(scioxide_core::EvaluationRole::Jacobian),
                model.jacobians.get()
            );
        }
    }
}

#[test]
fn krylov_products_apply_cached_local_jacobians_without_callbacks() {
    let nodes = 32;
    let model = CountedBratu {
        rhs: core::cell::Cell::new(0),
        jacobians: core::cell::Cell::new(0),
        sweeps: core::cell::RefCell::new(Vec::with_capacity(32)),
    };
    let mesh: Vec<_> = (0..nodes)
        .map(|i| f64::from(i) / f64::from(nodes - 1))
        .collect();
    let nodes = mesh.len();
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &mesh,
        options(krylov(), 1e-4),
    )
    .unwrap();
    let guess = Matrix::from_vec(nodes, 2, (0..nodes).flat_map(|_| [0.05, 0.]).collect()).unwrap();
    let solution = plan
        .solve(&BvpProblem::analytic(&model), guess.view(), &[])
        .unwrap();
    let report = report(&solution);
    assert_eq!(report.termination(), BvpTermination::Converged);
    assert_eq!(report.refinements(), 0);
    // One boundary Jacobian per linearization: each linearization follows exactly one
    // residual sweep and evaluates 2N - 1 RHS Jacobians plus the boundary Jacobian.
    let sweeps = model.sweeps.borrow();
    assert!(sweeps.len() >= 2);
    let mut previous = (0, 0);
    for &counts in sweeps.iter() {
        assert_eq!(counts.0 - previous.0, 4 * nodes - 3);
        assert_eq!(counts.1 - previous.1, 2 * nodes);
        previous = counts;
    }
    assert_eq!(model.rhs.get() - previous.0, 4 * nodes - 3);
    assert_eq!(model.jacobians.get(), previous.1);
    // Many more Krylov products than linearizations ran without any callback.
    let products = report
        .accounting()
        .total()
        .unwrap()
        .provider(scioxide_core::EvaluationRole::LinearOperator);
    assert!(products > 4 * sweeps.len(), "{products} {}", sweeps.len());
}

#[test]
fn changing_provider_dimensions_is_a_structured_failure() {
    struct Changes(core::cell::Cell<usize>);
    impl BvpSystem<f64> for Changes {
        fn dimension(&self) -> usize {
            self.0.get()
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.0.set(3);
            out.fill(0.);
            Ok(())
        }
        fn boundary(
            &self,
            _: &[f64],
            _: &[f64],
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.);
            Ok(())
        }
    }
    let system = Changes(core::cell::Cell::new(2));
    let plan = BvpPlan::new(
        system.dimension(),
        system.parameter_dimension(),
        &[0., 1.],
        BvpOptions::new()
            .with_max_nodes(8)
            .with_max_workspace_bytes(1 << 24),
    )
    .unwrap();
    let guess = Matrix::zeros(2, 2).unwrap();
    let error = plan
        .solve(&BvpProblem::numerical(&system), guess.view(), &[])
        .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
}
