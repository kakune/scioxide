//! Private checkpoint-state integrity; public continuation behavior is tested in
//! `tests/continuation.rs`.
use super::*;

fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn options<R: RealScalar>() -> ContinuationOptions<R> {
    ContinuationOptions::new(r(0.1), r(1e-6), r(0.15), budget(300))
        .unwrap()
        .with_residual_tolerance(ResidualTolerance::new(R::EPSILON * r(100.), R::ZERO).unwrap())
}
struct Fold;
impl<R: RealScalar> ResidualOperator<R> for Fold {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[R], y: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0] - x[1];
        Ok(())
    }
}
impl<R: RealScalar> JacobianOperator<R> for Fold {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        y[0] = r::<R>(2.) * x[0] * v[0] - v[1];
        Ok(())
    }
}
#[test]
fn checkpoint_integrity_and_zero_progress_are_enforced() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let plan =
        Continuation::new(ContinuationPath::Parameters(alloc::vec![2.0]), options()).unwrap();
    let mut state = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    plan.advance(
        &problem,
        &mut state,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let mut corrupted = state.checkpoint().unwrap();
    corrupted.state.point[0] = 99.0;
    assert!(
        plan.resume(
            &problem,
            &corrupted,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(state.point(), &[1.0, 1.0]);
    assert_eq!(corrupted.state.point[0], 99.0);
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        scioxide_core::AttemptKind::Resume
    );
    assert!(workspace.last_accepted().is_none());
    let tiny = Continuation::new(
        ContinuationPath::PseudoArclength(1.0),
        ContinuationOptions::new(1e-30, 1e-31, 1e-30, budget(3)).unwrap(),
    )
    .unwrap();
    let state = tiny.state(&problem, &[1.0, 1.0]).unwrap();
    assert_eq!(
        tiny.follow(&problem, state).unwrap().termination(),
        RootTermination::StepLimit
    );
}
