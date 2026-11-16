//! Prepared parameter tangents and explicit independent parameter rows.
use scioxide_core::{ExecutionContext, SciOxideError, SolveContext};
use scioxide_linalg::{MatrixView, Vector};
use scioxide_ode::{
    DormandPrince5, EnsembleParameters, OdeFn, OdeMethod, OdeOptions, OdeProblem,
    OdeSensitivitySystem, OdeSolveReport, OdeState, OdeWorkspace,
};
struct Growth;
impl OdeSensitivitySystem<f64> for Growth {
    fn dimension(&self) -> usize {
        1
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = p[0] * y[0];
        Ok(())
    }
    fn tangent_rhs(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        dy: &[f64],
        dp: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = p[0] * dy[0] + dp[0] * y[0];
        Ok(())
    }
}
type Member = (OdeState<f64>, OdeWorkspace<f64, DormandPrince5<f64>>);
fn main() -> Result<(), SciOxideError> {
    let mut binding = scioxide_ode::ForwardSensitivityWorkspace::new();
    let parameters = [0.5];
    let directions = [1.];
    let system = scioxide_ode::ForwardSensitivitySystem::bind(
        &Growth,
        &parameters,
        MatrixView::from_slice(1, 1, &directions)?,
        &mut binding,
    )?;
    let ode = (DormandPrince5::new(), OdeOptions::new());
    let solution = ode.0.solve(
        &OdeProblem::new(&system, 1.),
        &ode.1,
        OdeState::new(0., Vector::from_vec(vec![1., 0.])),
    )?;
    println!("state and dy/dp: {:?}", solution.state().values());

    let member = (DormandPrince5::new(), OdeOptions::new());
    let ensemble = scioxide_core::ExecutionPolicy::Sequential;
    let values = [-0.5, -1., -2.];
    let rates = EnsembleParameters::Independent(MatrixView::from_slice(3, 1, &values)?);
    let mut workspace = {
        let mut workspace = scioxide_ode::EnsembleWorkspace::new();
        workspace.try_reserve(&scioxide_ode::EnsembleRequirements::new(3, rates)?)?;
        workspace
    };
    let mut members: Vec<Member> = (0..3)
        .map(|_| {
            (
                OdeState::new(0., Vector::from_vec(vec![1.])),
                OdeWorkspace::new(),
            )
        })
        .collect();
    let mut outcomes: Vec<Option<Result<OdeSolveReport<f64>, SciOxideError>>> =
        (0..3).map(|_| None).collect();
    let run = |_: usize, p: &[f64], (state, ws): &mut Member, cx: &mut SolveContext<'_, '_>| {
        let rate = p[0];
        let rhs = OdeFn::new(1, move |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = rate * y[0];
            Ok(())
        })?;
        member.0.advance(
            &OdeProblem::new(&rhs, 1.),
            &member.1,
            state,
            &mut (),
            ws,
            cx,
        )
    };
    let report = scioxide_ode::solve_ensemble_into(
        rates,
        &run,
        ensemble,
        &mut members,
        &mut outcomes,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "{} of {} members succeeded",
        report.successful(),
        report.completed()
    );
    Ok(())
}
