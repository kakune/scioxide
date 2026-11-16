use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_pde::{
    FieldData, ParabolicOptions, ParabolicProblem, ParabolicScheme, PdeCheckpoint,
    StructuredFdmPlan,
};

#[test]
fn cancelled_resume_preserves_the_destination_state() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let source = FieldData::zero();
    let initial = FieldData::constant(1.0);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01);
    let problem = ParabolicProblem::new(&plan, &source, 0.02)?;
    let mut accepted = problem.initial_state(&initial, &options)?;
    let mut workspace = Box::new(problem.workspace()?);
    assert!(
        problem
            .advance(
                &options,
                &mut accepted,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )?
            .is_success()
    );
    let checkpoint = PdeCheckpoint::new(&accepted)?;
    let mut target = problem.initial_state(&initial, &options)?;
    let saved = target.coordinates().to_vec();
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    assert!(
        problem
            .resume(
                &options,
                &checkpoint,
                &mut target,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancellation)
            )
            .is_err()
    );
    assert_eq!(target.time(), 0.0);
    assert_eq!(target.coordinates(), saved);
    Ok(())
}

#[test]
fn different_free_unknown_layouts_cannot_share_states() -> Result<(), SciOxideError> {
    use scioxide_pde::{BoundaryCondition, BoundaryData, ConstantDiffusion, StructuredBoundaries};
    use scioxide_spatial::{AxisSide, BoundarySet};
    let base = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let side = |which| BoundarySet::axis_side(base.mesh(), 0, which);
    let make = |reverse| {
        let (dirichlet, natural) = if reverse {
            (AxisSide::End, AxisSide::Start)
        } else {
            (AxisSide::Start, AxisSide::End)
        };
        let boundaries = StructuredBoundaries::new(
            base.mesh(),
            1,
            vec![
                BoundaryCondition::dirichlet(side(dirichlet)?, BoundaryData::zero()),
                BoundaryCondition::natural(side(natural)?, BoundaryData::zero()),
            ],
        )?;
        StructuredFdmPlan::new(
            base.mesh().clone(),
            &ConstantDiffusion::new(1.0),
            &boundaries,
        )
    };
    let left = make(false)?;
    let right = make(true)?;
    let source = FieldData::zero();
    let first = ParabolicProblem::new(&left, &source, 0.02)?;
    let other = ParabolicProblem::new(&right, &source, 0.02)?;
    let options = ParabolicOptions::new();
    let mut state = first.initial_state(&FieldData::constant(1.0), &options)?;
    let mut out = [7.0; 5];
    assert!(other.values_into(&state, &mut out).is_err());
    assert_eq!(out, [7.0; 5]);
    let mut workspace = Box::new(other.workspace()?);
    assert!(
        other
            .advance(
                &options,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(state.time(), 0.0);
    Ok(())
}

#[test]
fn fully_constrained_output_honors_cancellation_between_samples() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [2])?;
    let source = FieldData::zero();
    let problem = ParabolicProblem::new(&plan, &source, 0.03)?;
    let options =
        ParabolicOptions::new().with_output_schedule(scioxide_pde::PdeOutputSchedule::Times(vec![
            0.01, 0.02, 0.03,
        ]));
    let mut state = problem.initial_state(&source, &options)?;
    let mut workspace = Box::new(problem.workspace()?);
    let cancellation = scioxide_core::CancellationToken::new();
    let mut samples = 0;
    let mut sink = |_: scioxide_pde::PdeObservation<'_, f64, f64>| {
        samples += 1;
        cancellation.cancel();
        Ok(())
    };
    assert!(
        !problem
            .advance(
                &options,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancellation)
            )?
            .is_success()
    );
    assert_eq!(samples, 1);
    assert!((state.time() - 0.01).abs() < 1e-15);
    Ok(())
}
