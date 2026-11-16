//! Discrete gradients of the crate's own plans: a steady FDM Poisson problem, a
//! Crank–Nicolson heat problem, and a reverse across two explicit AMR epochs.
//!
//! The caller supplies only derivative *data* in the plans' own vocabulary; the
//! plans assemble every operator and parameter derivative themselves.
#[path = "support/amr.rs"]
mod amr;
#[path = "support/adjoint_epochs.rs"]
mod epochs;
use scioxide_core::{ResidualTolerance, SciOxideError, SolveContext};
use scioxide_ode::{
    AdjointMode, AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue, AdjointOptions,
    DaeNumericalOptions,
};
use scioxide_pde::*;
use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};

/// Parameters: diffusion scale κ, source amplitude s and left boundary value g.
const P: [f64; 3] = [1.5, 2.0, 0.5];

/// `k(x) = κ (1 + x)`; with `κ = 1` it is the tangent `∂k/∂κ`.
struct Diffusion(f64);
impl StructuredFdmCoefficients<f64, f64, 1> for Diffusion {
    fn diffusion(&self, _: usize, _: usize, x: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(self.0 * (1.0 + x[0]))
    }
}

/// `J(u) = ½ Σ m_i u_i²` with lumped-mass weights; terminal only.
struct Energy(Vec<f64>);
impl AdjointObjective<f64> for Energy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        u: &[f64],
        _: &[f64],
        gu: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        gp.fill(0.0);
        if part == AdjointObjectivePart::Running {
            gu.fill(0.0);
            return Ok(AdjointObjectiveValue {
                value: 0.0,
                time_derivative: 0.0,
            });
        }
        for ((g, m), u) in gu.iter_mut().zip(&self.0).zip(u) {
            *g = m * u;
        }
        let value = 0.5 * self.0.iter().zip(u).map(|(m, u)| m * u * u).sum::<f64>();
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: 0.0,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -(k u')' = s on [0, 1], u(0) = g, u'(1) = 0.
    let mesh = SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::uniform(0.0, 1.0, 33)?],
        [AxisTopology::Bounded],
    )?)?;
    let side = |s| BoundarySet::axis_side(&mesh, 0, s);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::dirichlet(side(AxisSide::Start)?, BoundaryData::constant(P[2])),
            BoundaryCondition::natural(side(AxisSide::End)?, BoundaryData::zero()),
        ],
    )?;
    let plan = StructuredFdmPlan::with_options(
        mesh,
        &Diffusion(P[0]),
        &boundaries,
        SpatialLimits::new(64, 1024, 1 << 20),
    )?;
    let source = FieldData::constant(P[1]);
    let unit = Diffusion(1.0);
    let parameterization = plan.parameterize(
        &P,
        &[
            PdeParameterTangent::new()
                .with_coefficients(&unit as &dyn StructuredFdmCoefficients<f64, f64, 1>),
            PdeParameterTangent::new().with_source(FieldData::constant(1.0)),
            PdeParameterTangent::new().with_boundaries(vec![
                PdeConditionTangent::new(BoundaryData::constant(1.0)),
                PdeConditionTangent::zero(),
            ]),
        ],
    )?;
    let energy = Energy(plan.lumped_mass().to_vec());

    let problem = EllipticProblem::new(&plan, &source);
    let solution = problem.solve_with_options(&(EllipticOptions::new()))?;
    let tolerance = ResidualTolerance::new(1e-10, 1e-10)?;
    let steady = PdeEllipticAdjointPlan::new(
        &problem,
        &parameterization,
        &energy,
        PdeSteadyAdjointOptions::new()
            .with_primal_tolerance(tolerance)
            .with_linear_tolerance(tolerance),
    )?
    .solve(&problem, &parameterization, solution.values(), &energy)?;
    println!(
        "steady dJ/d(kappa, source, boundary): {:?}; transpose residual {:e}",
        steady.parameter_gradient(),
        steady.report().derivative_residual()
    );

    // u_t = (k u_x)_x + s from rest on [0, 0.25] with 16 Crank–Nicolson steps.
    let heat = ParabolicProblem::new(&plan, &source, 0.25)?;
    let free = Energy(heat.system().free_mass().to_vec());
    let options = AdjointOptions::new(AdjointMode::Discrete, 0.25 / 16.0)?.with_fixed_step(true);
    let transient = PdeTransientAdjointPlan::crank_nicolson(
        &heat,
        &parameterization,
        &free,
        None,
        scioxide_ode::DaeAdjointOptions::new(
            options,
            DaeNumericalOptions::new()
                .with_initialization(scioxide_ode::DaeInitialization::Project),
        )?,
    )?
    .solve(
        &heat,
        &parameterization,
        &free,
        &vec![0.0; heat.system().free_indices().len()],
    )
    .map_err(SciOxideError::from)?;
    println!(
        "Crank–Nicolson dJ/d(kappa, source, boundary): {:?}",
        transient.parameter_gradient()
    );

    let (actual, expected) = epochs::run()?;
    println!("two-epoch gradient: {actual}; independent exact value: {expected}");
    Ok(())
}
