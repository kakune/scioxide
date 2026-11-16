//! Two fixed spatial epochs composed through conservative refinement.
use super::amr;
use scioxide_core::*;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;
use scioxide_pde::*;
use scioxide_spatial::*;
struct Source {
    p: [f64; 1],
    n: usize,
}
impl OdeSystem<f64> for Source {
    fn dimension(&self) -> usize {
        self.n
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(self.p[0]);
        Ok(())
    }
}
impl OdeAdjointSystem<f64> for Source {
    fn parameters(&self) -> &[f64] {
        &self.p
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        _: &[f64],
        seed: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        gy.fill(0.);
        gp[0] = seed.iter().sum();
        Ok(())
    }
}
struct Cost {
    weights: Vec<f64>,
    linear: bool,
}
impl AdjointObjective<f64> for Cost {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        y: &[f64],
        _: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        gp.fill(0.);
        gy.fill(0.);
        let mut value = 0.;
        if part == AdjointObjectivePart::Terminal {
            for ((out, &v), &w) in gy.iter_mut().zip(y).zip(&self.weights) {
                *out = if self.linear { w } else { w * v };
                value += if self.linear { w * v } else { 0.5 * w * v * v };
            }
        }
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: 0.,
        })
    }
}
/// Return the chained parameter derivative and an independent exact value.
pub fn run() -> Result<(f64, f64), SciOxideError> {
    let coarse = amr::tensor::<f64, 1>(2, false)?;
    let adaptation = amr::adapt(&coarse, &[0], &[])?;
    let transfer = AmrCellTransfer::new(
        &coarse,
        &adaptation.mesh,
        &adaptation.map,
        AmrOptions::new().with_limits(amr::limits()),
    )?;
    let p = 0.4;
    let initial = [p, 2. * p];
    let end_first: Vec<_> = initial.iter().map(|&v| v + 0.25 * p).collect();
    // Constant forcing makes these exact stage endpoints, independent of time method.
    let intermediate = Field::<f64, Cell, 1>::new(&coarse, end_first)?;
    let next_initial = transfer.field(&adaptation.mesh, intermediate.view())?;
    let next = Source {
        p: [p],
        n: adaptation.mesh.cell_count(),
    };
    let cost = Cost {
        weights: (0..next.n)
            .map(|i| adaptation.mesh.cell_measure(i))
            .collect::<Result<_, _>>()?,
        linear: false,
    };
    let options = AdjointOptions::new(AdjointMode::Discrete, 0.125)?
        .with_fixed_step(true)
        .with_tape_capacity(8)
        .with_max_replay_steps(128)
        .with_max_reverse_steps(128)
        .with_max_workspace_bytes(1 << 20)
        .with_checkpoint_policy(CheckpointPolicy::StoreAll);
    let later = OdeAdjointSolver::new(
        next.dimension(),
        next.parameter_dimension(),
        options.clone(),
    )?;
    let later = later.solve(&scioxide_ode::OdeAdjointProblem::new(
        &next,
        &cost,
        next_initial.values(),
        (0.25, 0.5),
    ))?;
    let fine_seed =
        Field::<f64, Cell, 1>::new(&adaptation.mesh, later.initial_gradient().to_vec())?;
    let mut incoming = vec![0.; coarse.cell_count()];
    let mut scratch = vec![0.; coarse.cell_count()];
    transfer.pullback_into(
        coarse.mesh_token(),
        fine_seed.view(),
        &mut incoming,
        &mut scratch,
        ExecutionContext::sequential_native(),
    )?;
    let first = Source {
        p: [p],
        n: coarse.cell_count(),
    };
    let linear = Cost {
        weights: incoming,
        linear: true,
    };
    let earlier = OdeAdjointSolver::new(first.dimension(), first.parameter_dimension(), options)?;
    let earlier = earlier.solve(&scioxide_ode::OdeAdjointProblem::new(
        &first,
        &linear,
        &initial,
        (0., 0.25),
    ))?;
    // Initial data depends on p: add its pullback, keeping the incoming seed passive.
    let gradient = later.parameter_gradient()[0]
        + earlier.parameter_gradient()[0]
        + earlier.initial_gradient()[0]
        + 2. * earlier.initial_gradient()[1];
    let expected = 0.5 * p * (1.5_f64.powi(2) + 2.5_f64.powi(2));
    Ok((gradient, expected))
}
