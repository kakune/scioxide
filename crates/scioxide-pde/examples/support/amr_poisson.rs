//! Analytic localized Poisson equation and its independently integrated error.
#![allow(dead_code)]
use super::amr as support;
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, Preconditioner, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{CgSolver, GmresSolver, Vector};
use scioxide_pde::*;
use scioxide_sparse::{
    AmgHierarchy, AmgLimits, AmgPreconditioner, AmgSetupOptions, CsrMatrix, DuplicatePolicy,
    ExplicitZeroPolicy, SchwarzKind, SchwarzLocalSolveView, SchwarzPlan, SchwarzSubdomainView,
    SparseBuilder, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseSymmetricLimits,
};
use scioxide_spatial::{Cell, Field};

pub struct Gaussian;
impl Gaussian {
    pub fn exact(p: [f64; 2]) -> (f64, [f64; 2]) {
        let d = [p[0] - 0.3, p[1] - 0.4];
        let value = (-80. * (d[0] * d[0] + d[1] * d[1])).exp();
        (value, d.map(|x| -160. * x * value))
    }
}
impl AmrDiffusionData<f64, f64, 2> for Gaussian {
    fn source(&self, _: usize, p: [f64; 2]) -> Result<f64, SciOxideError> {
        let radius = (p[0] - 0.3).powi(2) + (p[1] - 0.4).powi(2);
        Ok((320. - 25600. * radius) * Self::exact(p).0)
    }
}

/// Dirichlet data of the exact Gaussian on every tagged exterior face.
pub fn boundaries(
    space: &AmrSpace<f64, 2>,
    tags: &AmrBoundaryTags<2>,
) -> Result<TaggedBoundaries<f64, f64, 2>, SciOxideError> {
    TaggedBoundaries::adaptive(
        space.mesh(),
        tags,
        1,
        vec![BoundaryCondition::dirichlet(
            0,
            BoundaryData::time_invariant(|p: BoundaryPoint<f64, 2>| Ok(Gaussian::exact(p.point).0)),
        )],
    )
}

pub fn solve(
    space: &AmrSpace<f64, 2>,
    tags: &AmrBoundaryTags<2>,
    seed: &[f64],
    backend: &str,
) -> Result<(Vec<f64>, AmrIndicators<f64, 2>), SciOxideError> {
    let mesh = space.mesh();
    let k = Field::<_, Cell, 2>::new(mesh, vec![1.; mesh.cell_count()])?;
    let boundaries = boundaries(space, tags)?;
    let plan =
        AmrDiffusionPlan::with_options(space, k.view(), &boundaries, &Gaussian, support::limits())?;
    let matrix = plan.assemble_csr()?;
    let mut x = Vector::from_vec(
        space
            .independent_nodes()
            .iter()
            .map(|&i| seed.get(i).copied().unwrap_or(0.))
            .collect(),
    );
    let rhs = Vector::from_vec(plan.rhs().to_vec());
    let tolerance = ResidualTolerance::new(1e-12, 1e-12)?;
    if backend == "schwarz" {
        solve_schwarz(space, &matrix, &rhs, &mut x)?;
    } else {
        let solver = CgSolver::new(tolerance, Budget::try_from(3000)?);
        let preconditioner = if backend == "amg" {
            let constant = vec![1.; matrix.rows()];
            let hierarchy = AmgHierarchy::build(
                matrix.view(),
                scioxide_core::BlockVectorView::new(&constant, matrix.rows(), 1)?,
                AmgSetupOptions::new(),
                AmgLimits::new(12, 2_000_000, 256 * 1024 * 1024),
                scioxide_sparse::SparseSolveOptions::new(),
                ExecutionContext::sequential(),
            )?;
            Some(AmgPreconditioner::build(
                hierarchy,
                scioxide_sparse::AmgCycleOptions::new(),
                SparseSymmetricLimits::new(100_000, 0, 256 * 1024 * 1024),
                scioxide_sparse::SparseSolveOptions::new(),
                ExecutionContext::sequential(),
            )?)
        } else {
            None
        };
        let mut work = scioxide_linalg::IterativeWorkspace::new();
        let report = solver.solve_into(
            &plan,
            &rhs,
            preconditioner
                .as_ref()
                .map_or(Preconditioning::None, |p| Preconditioning::Inverse(p)),
            x.view_mut(),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        if !report.is_success() {
            return Err(failure("AMR CG did not converge"));
        }
    }
    let mut full = vec![0.; space.full_dofs()];
    let mut scratch = full.clone();
    space
        .constraints()
        .apply_into(x.as_slice(), &mut full, &mut scratch)?;
    let mut indicators = AmrIndicators::new(mesh)?;
    {
        let residual = plan.residual(&full)?;
        let estimator = AmrEstimator::with_options(space, k.view(), support::limits())?;
        let mut work = estimator.workspace()?;
        support::context(|ctx| {
            estimator.estimate_into(&residual, &mut indicators, &mut work, ctx)
        })?;
    }
    Ok((full, indicators))
}

/// Independent degree-nine Grundmann-Möller integration, separate from the
/// positive Duffy/Gauss assembly rule. Analytic derivatives supply the oracle.
pub fn errors(
    space: &AmrSpace<f64, 2>,
    field: &[f64],
    index: usize,
) -> Result<(f64, f64), SciOxideError> {
    let rule = scioxide_integrate::SimplexRule::<f64>::new(
        2,
        index,
        scioxide_integrate::GeometricOptions::default().with_max_nodes(Budget::try_from(1024)?),
    )?;
    let mut l2 = 0.;
    let mut energy = 0.;
    for cell in 0..space.mesh().cell_count() {
        let vertices = space.mesh().cell_vertices(cell)?.collect::<Vec<_>>();
        let points = vertices
            .iter()
            .map(|&v| space.mesh().point(v).map(|p| *p.as_array()))
            .collect::<Result<Vec<_>, _>>()?;
        for (bary, &weight) in rule.barycentric_nodes().chunks_exact(3).zip(rule.weights()) {
            let reference = [bary[1], bary[2]];
            let p =
                core::array::from_fn(|a| bary.iter().zip(&points).map(|(&w, p)| w * p[a]).sum());
            let (u, g) = space.evaluate(cell, reference, field)?;
            let (target, d) = Gaussian::exact(p);
            let w = weight * 2. * space.mesh().cell_measure(cell)?;
            l2 += w * (u - target).powi(2);
            energy += w * ((g[0] - d[0]).powi(2) + (g[1] - d[1]).powi(2));
        }
    }
    if l2 < 0. || energy < 0. {
        return Err(failure("independent error quadrature unresolved"));
    }
    Ok((l2.sqrt(), energy.sqrt()))
}
fn failure(message: &'static str) -> SciOxideError {
    SciOxideError::external(message)
}

// A thin borrowed composition for existing GMRES. Schwarz owns all overlap/local
// inverse mathematics; no iterative solver or preconditioner is implemented here.
struct Left<'a> {
    a: &'a CsrMatrix<f64>,
    p: &'a dyn Preconditioner<f64>,
}
impl LinearOperator<f64> for Left<'_> {
    fn rows(&self) -> usize {
        self.a.rows()
    }
    fn cols(&self) -> usize {
        self.a.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows()
            + self
                .a
                .apply_workspace_len()?
                .max(self.p.precondition_workspace_len()?))
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        work: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let (ax, scratch) = work.split_at_mut(self.rows());
        self.a
            .apply_into(input, ax, &mut scratch[..self.a.apply_workspace_len()?])?;
        self.p.precondition_into(
            ax,
            output,
            &mut scratch[..self.p.precondition_workspace_len()?],
        )
    }
}
fn solve_schwarz(
    space: &AmrSpace<f64, 2>,
    matrix: &CsrMatrix<f64>,
    rhs: &Vector<f64>,
    x: &mut Vector<f64>,
) -> Result<(), SciOxideError> {
    let mut indices = [Vec::new(), Vec::new()];
    for (i, &node) in space.independent_nodes().iter().enumerate() {
        let p = space.node_point(node)?;
        if p[0] <= 0.65 {
            indices[0].push(i);
        }
        if p[0] >= 0.35 {
            indices[1].push(i);
        }
    }
    let mut weights = [Vec::new(), Vec::new()];
    for part in 0..2 {
        for &i in &indices[part] {
            weights[part].push(if indices[1 - part].binary_search(&i).is_ok() {
                0.5
            } else {
                1.
            });
        }
    }
    let mut locals = Vec::new();
    for nodes in &indices {
        let mut coo = SparseBuilder::new(nodes.len(), nodes.len());
        for (r, &global) in nodes.iter().enumerate() {
            for k in matrix.row_ptr()[global]..matrix.row_ptr()[global + 1] {
                if let Ok(c) = nodes.binary_search(&matrix.col_indices()[k]) {
                    coo.push(r, c, matrix.values()[k])?;
                }
            }
        }
        let local = coo.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)?;
        locals.push(local);
    }
    let symbolics = locals
        .iter()
        .map(|local| {
            SparseLuSymbolic::analyze(
                local.structural_pattern(),
                scioxide_sparse::SparseOrdering::default(),
                SparseLuLimits::new(2_000_000, 256 * 1024 * 1024),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factors = locals
        .iter()
        .zip(symbolics)
        .map(|(local, symbolic)| SparseLu::factor(symbolic, local.view()))
        .collect::<Result<Vec<_>, _>>()?;
    let domains = core::array::from_fn::<_, 2, _>(|i| SchwarzSubdomainView {
        indices: &indices[i],
        weights: Some(&weights[i]),
        solve: SchwarzLocalSolveView::InverseOperator(&factors[i]),
    });
    let preconditioner = SchwarzPlan::new(matrix.rows(), SchwarzKind::Additive, &domains, None)?;
    let left = Left {
        a: matrix,
        p: &preconditioner,
    };
    let mut b = Vector::zeros(rhs.len())?;
    let mut pw = vec![0.; preconditioner.precondition_workspace_len()?];
    preconditioner.precondition_into(rhs.as_slice(), b.as_mut_slice(), &mut pw)?;
    let solver = GmresSolver::new(
        ResidualTolerance::new(1e-12, 1e-12)?,
        Budget::try_from(3000)?,
    );
    let mut work = scioxide_linalg::IterativeWorkspace::new();
    let report = solver.solve_into(
        &left,
        &b,
        Preconditioning::None,
        x.view_mut(),
        &mut work,
        ExecutionContext::sequential(),
    )?;
    if !report.is_success() {
        return Err(failure("Schwarz/GMRES did not converge"));
    }
    let mut residual = vec![0.; rhs.len()];
    let mut scratch = residual.clone();
    matrix.apply_into(x.as_slice(), &mut residual, &mut scratch)?;
    let error = residual
        .iter()
        .zip(rhs.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    if error > 1e-9 {
        return Err(failure("Schwarz solve original residual too large"));
    }
    Ok(())
}
