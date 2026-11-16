#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_ad::CheckpointSchedule;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, Budget, Capacity, Complex, ErrorTolerance, ExecutionContext, NormSelection,
    NumericScalar, RealScalar, RelativeTolerance, ResidualTolerance, SciOxideError, SolveContext,
    StructuralPattern, StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;

struct Constraint<S: RealScalar> {
    parameters: [S; 2],
}
impl<T: NumericScalar> DaeSystem<T> for Constraint<T::Real> {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: T::Real,
        y: &[T],
        v: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = v[0] - T::from_real(self.parameters[0]) * y[0];
        out[1] = y[1] - T::from_real(self.parameters[1]) * y[0] * y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        a: &mut MatrixViewMut<'_, T>,
        b: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, T::ZERO)?;
                b.set(i, j, T::ZERO)?;
            }
        }
        a.set(0, 0, -T::from_real(self.parameters[0]))?;
        a.set(
            1,
            0,
            -T::from_real(self.parameters[1] + self.parameters[1]) * y[0],
        )?;
        a.set(1, 1, T::ONE)?;
        b.set(0, 0, T::ONE)?;
        Ok(Some(()))
    }
    fn sparse_derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        pattern: StructuralPatternView<'_>,
        a: &mut [T],
        b: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..2 {
            for index in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[index];
                a[index] = match (row, col) {
                    (0, 0) => -T::from_real(self.parameters[0]),
                    (1, 0) => -T::from_real(self.parameters[1] + self.parameters[1]) * y[0],
                    (1, 1) => T::ONE,
                    _ => T::ZERO,
                };
                b[index] = if row == 0 && col == 0 {
                    T::ONE
                } else {
                    T::ZERO
                };
            }
        }
        Ok(Some(()))
    }
    fn derivative_product(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        a: &[T],
        b: &[T],
        out: &mut [T],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = b[0] - T::from_real(self.parameters[0]) * a[0];
        out[1] = a[1] - T::from_real(self.parameters[1] + self.parameters[1]) * y[0] * a[0];
        Ok(Some(()))
    }
}
impl<T: NumericScalar> DaeAdjointSystem<T> for Constraint<T::Real> {
    fn parameters(&self) -> &[T::Real] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        seed: &[T],
        a: &mut [T],
        b: &mut [T],
        p: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        a[0] = -T::from_real(self.parameters[0]) * seed[0]
            - (T::from_real(self.parameters[1] + self.parameters[1]) * y[0]).conj() * seed[1];
        a[1] = seed[1];
        b[0] = seed[0];
        b[1] = T::ZERO;
        p[0] = -(seed[0].conj() * y[0]).real_component();
        p[1] = -(seed[1].conj() * y[0] * y[0]).real_component();
        Ok(())
    }
}
struct AlgebraicEnergy;
impl<T: NumericScalar> AdjointObjective<T> for AlgebraicEnergy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: T::Real,
        y: &[T],
        _: &[T::Real],
        a: &mut [T],
        p: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        a.fill(T::ZERO);
        p.fill(T::Real::ZERO);
        let value = if part == AdjointObjectivePart::Terminal {
            a[1] = y[1];
            y[1].norm_sqr() * T::Real::HALF
        } else {
            T::Real::ZERO
        };
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: T::Real::ZERO,
        })
    }
}
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn backend(name: &str) -> OdeStageBackend {
    match name {
        "dense" => OdeStageBackend::Dense,
        "sparse" => OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(2, 2, vec![0, 1, 3], vec![0, 0, 1]).unwrap(),
            limits: SparseLuLimits::new(64, 128 * 1024),
        },
        _ => OdeStageBackend::Krylov {
            iterations: Budget::try_from(16).unwrap(),
            restart: Budget::try_from(2).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    }
}
fn options<S: RealScalar>(mode: AdjointMode, step: S) -> AdjointOptions<S> {
    AdjointOptions::new(mode, step)
        .unwrap()
        .with_tape_capacity(128)
        .with_max_replay_steps(20000)
        .with_max_reverse_steps(20000)
        .with_max_workspace_bytes(4 * 1024 * 1024)
}
fn primal<S: RealScalar>() -> DaeOptions<S> {
    let tol = s::<S>(1e-12).max(s::<S>(32.) * S::EPSILON);
    let residual = ResidualTolerance::new(tol, S::ZERO).unwrap();
    DaeOptions::new().with_numerics(
        scioxide_ode::DaeNumericalOptions::new()
            .with_differential_residual(residual)
            .with_algebraic_residual(residual),
    )
}

fn discrete<T: NumericScalar>(x: T, tolerance: f64) {
    let model = Constraint {
        parameters: [s::<T::Real>(-0.7), s(1.3)],
    };
    let y = [x, T::from_real(s(1.3)) * x * x];
    let v = [T::from_real(s(-0.7)) * x, T::ZERO];
    let r = 1. / (1. + 0.7_f64 / 8.);
    let energy = x.norm_sqr().into_f64().powi(2) * r.powi(16);
    for name in ["dense", "sparse", "krylov"] {
        for checkpoints in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(128, 2).unwrap()),
        ] {
            let options = options(AdjointMode::Discrete, s(0.125))
                .with_fixed_step(true)
                .with_checkpoint_policy(checkpoints);
            let plan = DaeAdjointSolver::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                2,
                backend(name),
                scioxide_ode::DaeAdjointOptions::new(options, (primal()).numerics().clone())
                    .unwrap(),
            )
            .unwrap();
            let result = plan
                .solve(&scioxide_ode::DaeAdjointProblem::new(
                    &model,
                    &AlgebraicEnergy,
                    &y,
                    &v,
                    (s(0.), s(0.5)),
                ))
                .unwrap();
            let expected = T::from_real(s(2.
                * 1.3_f64.powi(2)
                * x.norm_sqr().into_f64()
                * r.powi(16)))
                * x;
            assert!(
                (result.initial_gradient()[0] - expected).abs().into_f64() < tolerance,
                "{name}: {result:?}"
            );
            assert!(
                (result.parameter_gradient()[0].into_f64() - 1.3_f64.powi(2) * energy * r).abs()
                    < tolerance,
                "{name}: {result:?}"
            );
            assert!(
                (result.parameter_gradient()[1].into_f64() - 1.3 * energy).abs() < tolerance,
                "{name}: {result:?}"
            );
            assert_eq!(result.report().accepted_steps(), 4);
        }
    }
}
#[test]
fn implicit_discrete_recurrence_all_scalars_and_backends() {
    discrete(1.1_f64, 2e-10);
    discrete(Complex::new(1.1_f64, 0.3), 4e-10);
    discrete(1.1_f32, 3e-4);
    discrete(Complex::new(1.1_f32, 0.3), 8e-4);
}

#[test]
fn continuous_constraint_adjoint_converges_to_analytic_gradient() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    let mut errors = Vec::new();
    for step in [1. / 16., 1. / 64.] {
        let tolerance = ErrorTolerance::scalar(2e-6, 2e-6, NormSelection::Infinity).unwrap();
        let options = options(AdjointMode::Continuous, step)
            .with_fixed_step(true)
            .with_continuous_error(tolerance.clone())
            .with_quadrature_error(tolerance)
            .with_checkpoint_policy(CheckpointPolicy::StoreAll);
        let plan = DaeAdjointSolver::<f64>::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            2,
            backend("dense"),
            scioxide_ode::DaeAdjointOptions::new(options, (primal()).numerics().clone()).unwrap(),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 1.3],
                &[-0.7, 0.],
                (0., 0.5),
            ))
            .unwrap();
        let exact = (-1.4_f64).exp();
        let error = (result.initial_gradient()[0] - 2. * 1.3_f64.powi(2) * exact)
            .abs()
            .max((result.parameter_gradient()[0] - 1.3_f64.powi(2) * exact).abs());
        errors.push(error);
        assert!(result.report().errors().adjoint().is_some());
        assert!(result.report().errors().linear_residual().unwrap() < 1e-12);
    }
    assert!(errors[1] < errors[0] * 0.4, "{errors:?}");
    assert!(errors[1] < 0.01, "{errors:?}");
}

#[test]
fn initial_projection_has_a_manifold_pullback() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let dae = primal().with_numerics(
            primal()
                .numerics()
                .clone()
                .with_initialization(DaeInitialization::Project),
        );
        let plan = DaeAdjointSolver::<f64>::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            2,
            backend("dense"),
            scioxide_ode::DaeAdjointOptions::new(options(mode, 0.1), (dae).numerics().clone())
                .unwrap(),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 99.],
                &[0., 0.],
                (0., 0.),
            ))
            .unwrap();
        assert!((result.initial_gradient()[0] - 2. * 1.3 * 1.3).abs() < 1e-12);
        assert!(result.parameter_gradient()[0].abs() < 1e-12);
        assert!((result.parameter_gradient()[1] - 1.3).abs() < 1e-12);
    }
}

#[test]
fn prepared_dae_backends_execute_and_preserve_padding() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        for name in ["dense", "sparse", "krylov"] {
            let tolerance = ErrorTolerance::scalar(1e-4, 1e-4, NormSelection::Infinity).unwrap();
            let options = options(mode, 0.125)
                .with_fixed_step(true)
                .with_continuous_error(tolerance.clone())
                .with_quadrature_error(tolerance)
                .with_checkpoint_policy(CheckpointPolicy::Explicit(
                    CheckpointSchedule::binomial(128, 2).unwrap(),
                ));
            let plan = DaeAdjointSolver::<f64>::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                2,
                backend(name),
                scioxide_ode::DaeAdjointOptions::new(options, (primal()).numerics().clone())
                    .unwrap(),
            )
            .unwrap();
            let mut workspace = plan
                .workspace(&scioxide_ode::DaeAdjointProblem::new(
                    &model,
                    &AlgebraicEnergy,
                    &[1., 1.3],
                    &[-0.7, 0.],
                    (0., 0.25),
                ))
                .unwrap();
            let mut g = [99.; 3];
            let mut p = [98.; 5];
            let result = plan
                .solve_into(
                    &scioxide_ode::DaeAdjointProblem::new(
                        &model,
                        &AlgebraicEnergy,
                        &[1., 1.3],
                        &[-0.7, 0.],
                        (0., 0.25),
                    ),
                    &mut g[..1],
                    &mut p[..2],
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(result.is_success());
            assert_eq!(&g[1..], &[99., 99.]);
            assert_eq!(&p[2..], &[98.; 3]);
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_retains_gradients_counts_and_replay_order() {
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    for name in ["dense", "sparse", "krylov"] {
        let options = options(AdjointMode::Discrete, 0.125)
            .with_fixed_step(true)
            .with_checkpoint_policy(CheckpointPolicy::Explicit(
                CheckpointSchedule::binomial(128, 2).unwrap(),
            ));
        let plan = DaeAdjointSolver::<f64>::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            2,
            backend(name),
            scioxide_ode::DaeAdjointOptions::new(options, (primal()).numerics().clone()).unwrap(),
        )
        .unwrap();
        let mut workspace = plan
            .workspace(&scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 1.3],
                &[-0.7, 0.],
                (0., 0.25),
            ))
            .unwrap();
        let mut results = Vec::new();
        for execution in [
            ExecutionContext::sequential(),
            ExecutionContext::new(&executor),
        ] {
            let mut y = [0.];
            let mut p = [0.; 2];
            let report = plan
                .solve_into(
                    &scioxide_ode::DaeAdjointProblem::new(
                        &model,
                        &AlgebraicEnergy,
                        &[1., 1.3],
                        &[-0.7, 0.],
                        (0., 0.25),
                    ),
                    &mut y,
                    &mut p,
                    &mut workspace,
                    execution,
                )
                .unwrap();
            results.push((y, p, report.accounting(), report.replayed_steps()));
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
fn transposed_linear_residual_is_a_relative_option() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    let defaults = scioxide_ode::DaeAdjointOptions::new(
        options(AdjointMode::Discrete, 0.125),
        primal().numerics().clone(),
    )
    .unwrap();
    // The options carry the documented `64ε` default.
    assert_eq!(defaults.linear_residual().relative(), 64. * f64::EPSILON);
    let mut gradients = Vec::new();
    for tolerance in [
        defaults.linear_residual(),
        RelativeTolerance::new(1e-10).unwrap(),
    ] {
        let plan = DaeAdjointSolver::<f64>::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            2,
            backend("dense"),
            defaults.clone().with_linear_residual(tolerance),
        )
        .unwrap();
        assert_eq!(plan.options().linear_residual(), tolerance);
        let problem = scioxide_ode::DaeAdjointProblem::new(
            &model,
            &AlgebraicEnergy,
            &[1., 1.3],
            &[-0.7, 0.],
            (0., 0.25),
        );
        let mut workspace = plan.workspace(&problem).unwrap();
        let (mut y, mut p) = ([0.], [0.; 2]);
        let report = plan
            .solve_into(
                &problem,
                &mut y,
                &mut p,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        gradients.push((y, p));
    }
    assert!((gradients[0].0[0] - gradients[1].0[0]).abs() < 1e-8);
}

#[test]
fn inconsistent_data_is_transactional_and_another_backend_regrows_the_workspace() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    let options = options(AdjointMode::Discrete, 0.125);
    let plan = DaeAdjointSolver::<f64>::new(
        DaePartition::new(&[true, false], &[true, false]).unwrap(),
        2,
        backend("dense"),
        scioxide_ode::DaeAdjointOptions::new(options.clone(), (primal()).numerics().clone())
            .unwrap(),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::DaeAdjointProblem::new(
            &model,
            &AlgebraicEnergy,
            &[1., 9.],
            &[-0.7, 0.],
            (0., 0.25),
        ))
        .unwrap();
    let mut y = [12.];
    let mut p = [13.; 2];
    assert!(
        plan.solve_into(
            &scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 9.],
                &[-0.7, 0.],
                (0., 0.25)
            ),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!((y, p), ([12.], [13.; 2]));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().termination(), None);
    // A workspace prepared for another stage backend is replaced before the solve.
    let other = DaeAdjointSolver::<f64>::new(
        DaePartition::new(&[true, false], &[true, false]).unwrap(),
        2,
        backend("krylov"),
        scioxide_ode::DaeAdjointOptions::new(options, (primal()).numerics().clone()).unwrap(),
    )
    .unwrap();
    let requirements = other
        .workspace_requirements(&scioxide_ode::DaeAdjointProblem::new(
            &model,
            &AlgebraicEnergy,
            &[1., 1.3],
            &[-0.7, 0.],
            (0., 0.25),
        ))
        .unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let report = other
        .solve_into(
            &scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 1.3],
                &[-0.7, 0.],
                (0., 0.25),
            ),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    assert_ne!((y, p), ([12.], [13.; 2]));
}

struct AlgebraicTime;
impl DaeSystem<f64> for AlgebraicTime {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[0] - 3. * t;
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        y: &mut MatrixViewMut<'_, f64>,
        v: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        y.set(0, 0, 1.)?;
        v.set(0, 0, 0.)?;
        Ok(Some(()))
    }
}
impl DaeAdjointSystem<f64> for AlgebraicTime {
    fn parameters(&self) -> &[f64] {
        &[3.]
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        t: f64,
        _: &[f64],
        _: &[f64],
        seed: &[f64],
        y: &mut [f64],
        v: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        y[0] = seed[0];
        v[0] = 0.;
        p[0] = -t * seed[0];
        Ok(())
    }
}
struct ScalarEnergy;
impl AdjointObjective<f64> for ScalarEnergy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        y: &[f64],
        _: &[f64],
        g: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        p.fill(0.);
        let value = if part == AdjointObjectivePart::Terminal {
            g[0] = y[0];
            0.5 * y[0] * y[0]
        } else {
            g.fill(0.);
            0.
        };
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: 0.,
        })
    }
}
#[test]
fn pure_algebraic_partition_has_no_free_initial_coordinates() {
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let plan = DaeAdjointSolver::new(
            DaePartition::new(&[false], &[false]).unwrap(),
            AlgebraicTime.parameter_dimension(),
            OdeStageBackend::Dense,
            scioxide_ode::DaeAdjointOptions::new(
                options(mode, 0.125).with_fixed_step(true),
                (primal()).numerics().clone(),
            )
            .unwrap(),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &AlgebraicTime,
                &ScalarEnergy,
                &[0.],
                &[0.],
                (0., 1.),
            ))
            .unwrap();
        assert!(result.initial_gradient().is_empty());
        assert!((result.parameter_gradient()[0] - 3.).abs() < 1e-12);
    }
}

fn adaptive<T: NumericScalar>(x: T, tolerance: f64) {
    let model = Constraint {
        parameters: [s::<T::Real>(-0.7), s(1.3)],
    };
    let y = [x, T::from_real(s(1.3)) * x * x];
    let d = [T::from_real(s(-0.7)) * x, T::ZERO];
    let decay = (-1.4_f64).exp();
    let expected = T::from_real(s(2. * 1.3_f64.powi(2) * x.norm_sqr().into_f64() * decay)) * x;
    let energy = x.norm_sqr().into_f64().powi(2) * decay;
    for name in ["dense", "sparse", "krylov"] {
        for checkpoints in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(1024, 2).unwrap()),
        ] {
            let plan = DaeAdjointSolver::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                2,
                backend(name),
                scioxide_ode::DaeAdjointOptions::new(
                    options(AdjointMode::Discrete, s(0.125))
                        .with_tape_capacity(1024)
                        .with_max_replay_steps(1000000)
                        .with_checkpoint_policy(checkpoints),
                    (primal()).numerics().clone(),
                )
                .unwrap(),
            )
            .unwrap();
            let mut workspace = plan
                .workspace(&scioxide_ode::DaeAdjointProblem::new(
                    &model,
                    &AlgebraicEnergy,
                    &y,
                    &d,
                    (s(0.), s(0.5)),
                ))
                .unwrap();
            let mut gy = [T::ZERO];
            let mut gp = [T::Real::ZERO; 2];
            let report = plan
                .solve_into(
                    &scioxide_ode::DaeAdjointProblem::new(
                        &model,
                        &AlgebraicEnergy,
                        &y,
                        &d,
                        (s(0.), s(0.5)),
                    ),
                    &mut gy,
                    &mut gp,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_or_else(|failure| {
                    panic!("{name} {}: {failure:?}", core::any::type_name::<T>())
                });
            assert_eq!(report.termination(), Some(AdjointTermination::ReachedEnd));
            assert!(
                (gy[0] - expected).abs().into_f64() < tolerance,
                "{name}: {report:?}, {gy:?}"
            );
            assert!(
                (gp[0].into_f64() - 1.3_f64.powi(2) * energy).abs() < tolerance,
                "{name}: {gp:?}"
            );
            assert!(
                (gp[1].into_f64() - 1.3 * energy).abs() < tolerance,
                "{name}: {gp:?}"
            );
        }
    }
}
#[test]
fn adaptive_multistep_gradients_all_scalars_backends_and_checkpoints() {
    adaptive(1.1_f64, 2e-6);
    adaptive(Complex::new(1.1_f64, 0.3), 3e-6);
    adaptive(1.1_f32, 0.02);
    adaptive(Complex::new(1.1_f32, 0.3), 0.03);
}

// The derivative callback deliberately cannot reduce the positive-time residual.
// Both forward consumers must exhaust rejection, without committing that pair.
struct InconsistentTrial;
impl DaeSystem<f64> for InconsistentTrial {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(
        &self,
        time: f64,
        _: &[f64],
        derivative: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        output[0] = if time > 0. { 1. } else { derivative[0] };
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        state: &mut MatrixViewMut<'_, f64>,
        derivative: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        state.set(0, 0, 0.)?;
        derivative.set(0, 0, 1.)?;
        Ok(Some(()))
    }
}
impl DaeAdjointSystem<f64> for InconsistentTrial {
    fn parameters(&self) -> &[f64] {
        &[]
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        seed: &[f64],
        state: &mut [f64],
        derivative: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state.fill(0.);
        derivative.copy_from_slice(seed);
        Ok(())
    }
}
struct ZeroCost;
impl AdjointObjective<f64> for ZeroCost {
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: f64,
        _: &[f64],
        _: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        state.fill(0.);
        parameters.fill(0.);
        Ok(AdjointObjectiveValue {
            value: 0.,
            time_derivative: 0.,
        })
    }
}

#[test]
fn residual_inconsistency_is_rejected_by_forward_and_adjoint() {
    let partition = DaePartition::new(&[true], &[true]).unwrap();
    let dae_options = DaeOptions::new().with_numerics(
        scioxide_ode::DaeNumericalOptions::new()
            .with_rejection_limit(Budget::try_from(2).unwrap())
            .with_newton_limit(Budget::try_from(2).unwrap()),
    );
    let forward = (
        DaeBdf::new(partition.clone(), OdeStageBackend::Dense).unwrap(),
        dae_options.clone().with_initial_step(0.25),
    );
    let mut state = DaeState::from_slices(0., &[1.], &[0.]).unwrap();
    let mut forward_workspace = forward.0.workspace().unwrap();
    let report = forward
        .0
        .advance(
            &DaeProblem::new(&InconsistentTrial, 1.),
            &forward.1,
            &mut state,
            &mut (),
            &mut forward_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::RejectionLimit));
    assert_eq!(report.accepted_steps(), 0);
    assert_eq!(report.accounting().total().unwrap().rejected(), 2);
    assert_eq!(state.values(), [1.]);
    assert_eq!(state.derivative(), [0.]);
    let plan = DaeAdjointSolver::new(
        partition,
        InconsistentTrial.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(
            options(AdjointMode::Discrete, 0.25),
            (dae_options).numerics().clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::DaeAdjointProblem::new(
            &InconsistentTrial,
            &ZeroCost,
            &[1.],
            &[0.],
            (0., 1.),
        ))
        .unwrap();
    let mut gradient = [17.];
    let error = plan
        .solve_into(
            &scioxide_ode::DaeAdjointProblem::new(
                &InconsistentTrial,
                &ZeroCost,
                &[1.],
                &[0.],
                (0., 1.),
            ),
            &mut gradient,
            &mut [],
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.accepted_steps(), 0);
    assert_eq!(progress.accounting().total().unwrap().rejected(), 2);
    assert_eq!(progress.termination(), None);
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::NoSuccessfulCandidate,
            ..
        }
    ));
    assert_eq!(gradient, [17.]);
}

#[test]
fn assembled_adjoint_blocks_honor_the_difference_strategy() {
    use scioxide_differentiate::DerivativeStrategy;
    let model = Constraint {
        parameters: [-0.7, 1.3],
    };
    for name in ["dense", "sparse"] {
        let solve = |strategy| {
            DaeAdjointSolver::<f64>::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                2,
                backend(name),
                scioxide_ode::DaeAdjointOptions::new(
                    options(AdjointMode::Discrete, 0.125).with_fixed_step(true),
                    (primal().with_numerics(
                        primal()
                            .numerics()
                            .clone()
                            .with_derivative_strategy(strategy),
                    ))
                    .numerics()
                    .clone(),
                )
                .unwrap(),
            )
            .unwrap()
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &model,
                &AlgebraicEnergy,
                &[1., 1.3],
                &[-0.7, 0.],
                (0., 0.5),
            ))
            .unwrap()
        };
        let analytic = solve(DerivativeStrategy::Analytic);
        for strategy in [DerivativeStrategy::Forward, DerivativeStrategy::Central] {
            let numeric = solve(strategy);
            assert!(numeric.report().is_success());
            assert!((numeric.initial_gradient()[0] - analytic.initial_gradient()[0]).abs() < 2e-7);
            for (&a, &b) in numeric
                .parameter_gradient()
                .iter()
                .zip(analytic.parameter_gradient())
            {
                assert!((a - b).abs() < 2e-7);
            }
        }
    }
}

#[test]
fn rejected_nonlinear_proposals_replay_exactly_across_checkpoint_policies() {
    let model = Constraint {
        parameters: [-0.7_f64, 1.3],
    };
    let mut reference: Option<[f64; 3]> = None;
    for name in ["dense", "sparse"] {
        for policy in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(1024, 2).unwrap()),
        ] {
            let replay = matches!(policy, CheckpointPolicy::Explicit(_));
            let plan = DaeAdjointSolver::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                2,
                backend(name),
                scioxide_ode::DaeAdjointOptions::new(
                    options(AdjointMode::Discrete, 0.125)
                        .with_tape_capacity(1024)
                        .with_max_replay_steps(1000000)
                        .with_checkpoint_policy(policy),
                    (primal()).numerics().clone(),
                )
                .unwrap(),
            )
            .unwrap();
            let result = plan
                .solve(&scioxide_ode::DaeAdjointProblem::new(
                    &model,
                    &AlgebraicEnergy,
                    &[1.1, 1.3 * 1.1 * 1.1],
                    &[-0.77, 0.],
                    (0., 0.5),
                ))
                .unwrap();
            assert_eq!(
                result.report().termination(),
                Some(AdjointTermination::ReachedEnd)
            );
            assert!(result.report().accounting().total().unwrap().rejected() > 0);
            if replay {
                assert!(result.report().replayed_steps() > 0);
            }
            let gradients = [
                result.initial_gradient()[0],
                result.parameter_gradient()[0],
                result.parameter_gradient()[1],
            ];
            if let Some(reference) = reference {
                for (actual, expected) in gradients.into_iter().zip(reference) {
                    assert!((actual - expected).abs() < 1e-10);
                }
            } else {
                reference = Some(gradients);
            }
        }
    }
}

/// `x' = p₀ x, 0 = z − p₁ x`.
struct Proportional {
    parameters: [f64; 2],
}
impl DaeSystem<f64> for Proportional {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        v: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = v[0] - self.parameters[0] * y[0];
        out[1] = y[1] - self.parameters[1] * y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (i, j, state, rate) in [
            (0, 0, -self.parameters[0], 1.),
            (0, 1, 0., 0.),
            (1, 0, -self.parameters[1], 0.),
            (1, 1, 1., 0.),
        ] {
            a.set(i, j, state)?;
            b.set(i, j, rate)?;
        }
        Ok(Some(()))
    }
}
impl DaeAdjointSystem<f64> for Proportional {
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        seed: &[f64],
        state: &mut [f64],
        rate: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = -self.parameters[0] * seed[0] - self.parameters[1] * seed[1];
        state[1] = seed[1];
        rate[0] = seed[0];
        rate[1] = 0.;
        parameters[0] = -y[0] * seed[0];
        parameters[1] = -y[0] * seed[1];
        Ok(())
    }
}
/// `J = (x² + z²)/2` at the end time.
struct TerminalEnergy;
impl AdjointObjective<f64> for TerminalEnergy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        y: &[f64],
        _: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        parameters.fill(0.);
        let value = if part == AdjointObjectivePart::Terminal {
            state.copy_from_slice(y);
            0.5 * (y[0] * y[0] + y[1] * y[1])
        } else {
            state.fill(0.);
            0.
        };
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: 0.,
        })
    }
}
/// `J` from a forward solve at `rtol = atol = 10⁻¹²` with consistent initial data.
fn tight_energy(parameters: [f64; 2], x0: f64, (start, end): (f64, f64)) -> f64 {
    let tight = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
    let plan = (
        DaeBdf::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            OdeStageBackend::Dense,
        )
        .unwrap(),
        primal()
            .with_numerics(
                primal()
                    .numerics()
                    .clone()
                    .with_differential_error(tight.clone())
                    .with_algebraic_error(tight),
            )
            .with_initial_step(1e-9),
    );
    let initial = DaeState::from_slices(
        start,
        &[x0, parameters[1] * x0],
        &[parameters[0] * x0, parameters[0] * parameters[1] * x0],
    )
    .unwrap();
    let solution = plan
        .0
        .solve(
            &DaeProblem::new(&Proportional { parameters }, end),
            &plan.1,
            initial,
        )
        .unwrap();
    assert!(solution.report().is_success());
    let y = solution.state().values();
    0.5 * (y[0] * y[0] + y[1] * y[1])
}

#[test]
fn continuous_adjoint_on_adaptive_meshes_matches_forward_differences() {
    let x0 = 1.;
    let tolerance = ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity).unwrap();
    // Decaying solutions give adaptive meshes of many segments. The stationary
    // solution is accepted in one segment from 0.3 to 1.661, for which reversing the
    // recorded width from 1.661 in absolute time stops one float short of 0.3.
    for (parameters, interval, step, h) in [
        ([-0.7, 1.3], (0., 0.002), 1e-3, 1e-6),
        ([-0.7, 1.3], (0., 2.), 1e-3, 1e-4),
        ([0., 1.3], (0.3, 1.661), 2., 1e-4),
    ] {
        let plan = DaeAdjointSolver::new(
            DaePartition::new(&[true, false], &[true, false]).unwrap(),
            2,
            OdeStageBackend::Dense,
            scioxide_ode::DaeAdjointOptions::new(
                options(AdjointMode::Continuous, step)
                    .with_tape_capacity(1024)
                    .with_adjoint_error(tolerance.clone())
                    .with_quadrature_error(tolerance.clone()),
                DaeNumericalOptions::new(),
            )
            .unwrap(),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &Proportional { parameters },
                &TerminalEnergy,
                &[x0, parameters[1] * x0],
                &[parameters[0] * x0, 0.],
                interval,
            ))
            .unwrap_or_else(|failure| panic!("{interval:?}: {failure:?}"));
        let report = result.report();
        assert_eq!(report.termination(), Some(AdjointTermination::ReachedEnd));
        // Every segment of the forward mesh is reversed.
        assert!(
            report.accepted_steps() > 4 || parameters[0] == 0.,
            "{report:?}"
        );
        assert_eq!(report.reverse_steps(), report.accepted_steps());
        let central = |perturbed: &dyn Fn(f64) -> f64| (perturbed(h) - perturbed(-h)) / (2. * h);
        let differences = [
            central(&|d| tight_energy(parameters, x0 + d, interval)),
            central(&|d| tight_energy([parameters[0] + d, parameters[1]], x0, interval)),
            central(&|d| tight_energy([parameters[0], parameters[1] + d], x0, interval)),
        ];
        let gradient = [
            result.initial_gradient()[0],
            result.parameter_gradient()[0],
            result.parameter_gradient()[1],
        ];
        for (adjoint, difference) in gradient.into_iter().zip(differences) {
            assert!(
                (adjoint - difference).abs() <= 1e-6 * difference.abs().max(1.),
                "{interval:?}: {gradient:?} vs {differences:?}"
            );
        }
    }
}

trait ContinuousError<S: RealScalar> {
    fn with_continuous_error(self, error: ErrorTolerance<S>) -> Self;
}
impl<S: RealScalar> ContinuousError<S> for AdjointOptions<S> {
    fn with_continuous_error(self, error: ErrorTolerance<S>) -> Self {
        if self.mode() == AdjointMode::Continuous {
            self.with_adjoint_error(error)
        } else {
            self
        }
    }
}
