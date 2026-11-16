#![allow(clippy::unwrap_used)]

use scioxide_core::{SciOxideError, SolveContext};
use scioxide_ode::{AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue};
use scioxide_pde::{
    ConstantDiffusion, DifferentiableDiscretization, EllipticProblem, FieldData,
    PdeEllipticAdjointPlan, PdeParameterTangent, PdeSteadyAdjointOptions, SemidiscreteSystem,
    SpatialDiscretization, StructuredFdmPlan,
};

fn context<T>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "pullback", 1),
        0,
        0,
    );
    run(&mut state.begin_solve(
        scioxide_core::ExecutionContext::sequential(),
        identity,
        &mut budgets,
    )?)
}

struct Total;
impl AdjointObjective<f64> for Total {
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: f64,
        state: &[f64],
        _: &[f64],
        state_gradient: &mut [f64],
        parameter_gradient: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        state_gradient.fill(1.0);
        parameter_gradient.fill(0.0);
        Ok(AdjointObjectiveValue {
            value: state.iter().sum(),
            time_derivative: 0.0,
        })
    }
}

#[test]
fn adjoints_reject_different_operators_on_the_same_mesh() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let other = StructuredFdmPlan::new(
        plan.mesh().clone(),
        &ConstantDiffusion::new(2.0),
        plan.boundaries(),
    )?;
    let source = FieldData::zero();
    let parameters = plan.parameterize(
        &[0.0],
        &[PdeParameterTangent::new().with_source(FieldData::constant(1.0))],
    )?;
    let other_parameters = other.parameterize(
        &[0.0],
        &[PdeParameterTangent::new().with_source(FieldData::constant(1.0))],
    )?;
    let problem = EllipticProblem::new(&plan, &source);
    let different = EllipticProblem::new(&other, &source);
    let adjoint = PdeEllipticAdjointPlan::new(
        &problem,
        &parameters,
        &Total,
        PdeSteadyAdjointOptions::new(),
    )?;
    // Zero is an equilibrium of both operators, so a residual check cannot
    // detect reuse of the first operator's factorization.
    assert!(
        adjoint
            .solve(&different, &other_parameters, &[0.0; 5], &Total)
            .is_err()
    );
    assert!(
        PdeEllipticAdjointPlan::new(
            &different,
            &parameters,
            &Total,
            PdeSteadyAdjointOptions::new(),
        )
        .is_err()
    );
    assert!(
        SemidiscreteSystem::new(&other, &source)?
            .adjoint(&parameters)
            .is_err()
    );
    let cloned = plan.clone();
    let result = adjoint.solve(
        &EllipticProblem::new(&cloned, &source),
        &parameters,
        &[0.0; 5],
        &Total,
    )?;
    assert!((result.parameter_gradient()[0] - 0.3125).abs() < 1e-12);
    Ok(())
}

#[test]
fn inactive_derivatives_reject_nonfinite_time() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let parameters = plan.parameterize(&[0.0], &[PdeParameterTangent::new()])?;
    let mut scratch = vec![0.0; parameters.scratch_len()];
    for time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut out = [7.0; 5];
        assert!(
            parameters
                .residual_tangent_into(0, &[0.0; 5], time, &mut out, &mut scratch)
                .is_err()
        );
        assert_eq!(out, [7.0; 5]);
    }
    Ok(())
}

#[test]
fn derivative_callback_failure_preserves_output() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let source = FieldData::time_invariant(|point| {
        if point.point[0] > 0.5 {
            Err(SciOxideError::external("late source failure"))
        } else {
            Ok(1.0)
        }
    });
    let parameters =
        plan.parameterize(&[0.0], &[PdeParameterTangent::new().with_source(source)])?;
    let mut out = [7.0; 5];
    assert!(
        parameters
            .residual_tangent_into(
                0,
                &[0.0; 5],
                0.0,
                &mut out,
                &mut vec![0.0; parameters.scratch_len()]
            )
            .is_err()
    );
    assert_eq!(out, [7.0; 5]);
    Ok(())
}

#[test]
fn load_callback_failure_preserves_output() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let source = FieldData::time_invariant(|point| {
        if point.point[0] > 0.5 {
            Err(SciOxideError::external("late source failure"))
        } else {
            Ok(1.0)
        }
    });
    let mut out = [7.0; 5];
    assert!(
        plan.load_into(
            &source,
            0.0,
            &mut out,
            &mut vec![0.0; plan.load_workspace_len()]
        )
        .is_err()
    );
    assert_eq!(out, [7.0; 5]);
    Ok(())
}

#[test]
fn semidiscrete_pullback_commits_together() -> Result<(), SciOxideError> {
    use scioxide_ode::OdeAdjointSystem;
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let source = FieldData::zero();
    let parameters = plan.parameterize(
        &[0.0, 0.0],
        &[
            PdeParameterTangent::new().with_source(FieldData::constant(1.0)),
            PdeParameterTangent::new().with_source(FieldData::time_invariant(|_| {
                Err(SciOxideError::external("derivative failure"))
            })),
        ],
    )?;
    let system = SemidiscreteSystem::new(&plan, &source)?;
    let adjoint = system.adjoint(&parameters)?;
    let mut scratch = vec![0.0; adjoint.pullback_workspace_len()];
    let mut state_output = [7.0; 3];
    let mut parameter_output = [8.0; 2];
    assert!(
        context(|ctx| adjoint.pullback(
            0.0,
            &[0.0; 3],
            &[1.0; 3],
            &mut state_output,
            &mut parameter_output,
            &mut scratch,
            ctx
        ))
        .is_err()
    );
    assert_eq!(state_output, [7.0; 3]);
    assert_eq!(parameter_output, [8.0; 2]);
    Ok(())
}

#[test]
fn mol_pullback_stages_and_poisons_parameter_results() -> Result<(), SciOxideError> {
    use scioxide_ode::{OdeAdjointSystem, OdeSystem};
    struct Broken(bool);
    impl OdeSystem<f64> for Broken {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.0);
            Ok(())
        }
    }
    impl OdeAdjointSystem<f64> for Broken {
        fn parameters(&self) -> &[f64] {
            &[0.0]
        }
        fn is_replayable(&self) -> bool {
            true
        }
        fn pullback(
            &self,
            _: f64,
            _: &[f64],
            _: &[f64],
            state: &mut [f64],
            parameters: &mut [f64],
            _: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            state.fill(1.0);
            if self.0 {
                Ok(())
            } else {
                parameters.fill(3.0);
                Err(SciOxideError::external("late derivative failure"))
            }
        }
    }
    let plan = scioxide_pde::MethodOfLinesPlan::new(1, None, None)?;
    let mut work = plan.workspace()?;
    for incomplete in [false, true] {
        let provider = Broken(incomplete);
        let system = plan.bind_adjoint(&provider, &mut work)?;
        let mut state = [7.0];
        let mut parameters = [8.0];
        assert!(
            context(|ctx| system.pullback(
                0.0,
                &[0.0],
                &[1.0],
                &mut state,
                &mut parameters,
                &mut [],
                ctx
            ))
            .is_err()
        );
        assert_eq!(state, [7.0]);
        assert_eq!(parameters, [8.0]);
    }
    Ok(())
}

#[test]
fn automatic_step_limits_validate_provider_output_and_keep_extended_range()
-> Result<(), SciOxideError> {
    use scioxide_ode::OdeSystem;
    struct Jacobian<T>(bool, T);
    impl<T: scioxide_core::NumericScalar<Real = f64>> OdeSystem<f64, T> for Jacobian<T> {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: f64,
            _: &[T],
            out: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(T::ZERO);
            Ok(())
        }
        fn jacobian_is_constant(&self) -> bool {
            true
        }
        fn sparse_jacobian(
            &self,
            _: f64,
            _: &[T],
            _: scioxide_core::StructuralPatternView<'_>,
            out: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            if self.0 {
                out[0] = self.1;
            }
            Ok(Some(()))
        }
    }
    let pattern = scioxide_core::StructuralPattern::new(1, 1, vec![0, 1], vec![0])?;
    let plan = scioxide_pde::MethodOfLinesPlan::new(1, Some(&[0.5]), None)?
        .with_jacobian_pattern(pattern)?
        .with_step_limit(
            scioxide_pde::CflPolicy::Automatic,
            scioxide_ode::OdeLinearStability::Axes {
                real: Some(2.0),
                imaginary: None,
            },
        )?;
    let mut work = plan.workspace()?;
    let incomplete = Jacobian(false, 0.0);
    {
        let system = plan.bind(&incomplete, &mut work)?;
        assert!(context(|ctx| system.step_limit(0.0, &[0.0], ctx)).is_err());
    }
    let complete = Jacobian(true, -f64::MAX);
    let system = plan.bind(&complete, &mut work)?;
    let step = context(|ctx| system.step_limit(0.0, &[0.0], ctx))?.unwrap();
    assert!((step / (0.9 / f64::MAX) - 1.0).abs() < 1e-14);
    // Cached Jacobians must not bypass validation of subsequent requests.
    assert!(context(|ctx| system.step_limit(f64::NAN, &[0.0], ctx)).is_err());
    assert!(context(|ctx| system.step_limit(0.0, &[], ctx)).is_err());
    let complex = Jacobian(true, scioxide_core::Complex::new(-f64::MAX, f64::MAX));
    let mut complex_work = plan.workspace()?;
    let complex_system = plan.bind(&complex, &mut complex_work)?;
    let step = context(|ctx| {
        complex_system.step_limit(0.0, &[scioxide_core::Complex::new(0.0, 0.0)], ctx)
    })?
    .unwrap();
    assert!((step / ((0.9 / 2.0_f64.sqrt()) / f64::MAX) - 1.0).abs() < 1e-14);
    Ok(())
}

#[test]
fn mol_reconstruction_and_cancelled_rhs_preserve_outputs() -> Result<(), SciOxideError> {
    use scioxide_ode::OdeSystem;
    use scioxide_pde::MethodOfLinesFields;
    let cancellation = scioxide_core::CancellationToken::new();
    struct Provider<'a>(&'a scioxide_core::CancellationToken);
    impl OdeSystem<f64> for Provider<'_> {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(1.0);
            self.0.cancel();
            Ok(())
        }
    }
    struct Fields;
    impl MethodOfLinesFields<f64> for Fields {
        fn field_len(&self) -> usize {
            2
        }
        fn reconstruct_into(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 1.0;
            Err(SciOxideError::external("late reconstruction failure"))
        }
    }
    let plan = scioxide_pde::MethodOfLinesPlan::new(1, None, None)?;
    let mut work = plan.workspace()?;
    let provider = Provider(&cancellation);
    let system = plan.bind_with_fields(&provider, &Fields, &mut work)?;
    let mut field = [7.0; 2];
    assert!(system.reconstruct_into(0.0, &[0.0], &mut field).is_err());
    assert_eq!(field, [7.0; 2]);
    let mut solve_state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "rhs", 1),
        0,
        0,
    );
    let mut context = solve_state.begin_solve(
        scioxide_core::ExecutionContext::sequential().with_cancellation(&cancellation),
        identity,
        &mut budgets,
    )?;
    let mut out = [7.0];
    assert!(system.rhs(0.0, &[0.0], &mut out, &mut context).is_err());
    assert_eq!(out, [7.0]);
    Ok(())
}
