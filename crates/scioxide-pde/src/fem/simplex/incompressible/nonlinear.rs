use super::*;
use scioxide_core::{JacobianOperator, ResidualOperator};

/// Own reserved element and CSR tangents for successive Navier–Stokes states.
/// Construct with [`NavierStokesPlan::linearization_workspace`]. Refreshing the
/// state allocates nothing; storage is O(cells × local unknowns² + CSR entries).
pub struct NavierStokesLinearizationWorkspace<R: RealScalar> {
    storage: super::super::csr::TangentStorage<R>,
}

/// Real conservative Navier–Stokes residual over a prepared Stokes space.
/// Nonlinear iteration and globalization belong to the injected root solver.
/// Residual, JVP and adjoint scratch is `residual_workspace_len()`. Use
/// [`Self::linearize`] for an evaluated fixed-state tangent, or [`crate::PdeTangent`]
/// for products through the residual provider.
#[derive(Clone)]
pub struct NavierStokesPlan<R: RealScalar, const D: usize> {
    pub(super) stokes: StokesPlan<R, R, D>,
}
impl<R: RealScalar, const D: usize> NavierStokesPlan<R, D> {
    /// Reserve element tangents and the complete gauged CSR for [`Self::linearize`].
    /// # Errors
    /// Returns checked size and allocation failures.
    pub fn linearization_workspace(
        &self,
    ) -> Result<NavierStokesLinearizationWorkspace<R>, SciOxideError> {
        Ok(NavierStokesLinearizationWorkspace {
            storage: super::super::csr::TangentStorage::new(
                &self.stokes.assembly,
                self.stokes.matrices.len(),
                &self.stokes.assembly_extra,
            )?,
        })
    }
    /// Evaluate convection derivatives once and borrow the complete fixed-state tangent.
    /// Work is O(cells × quadrature nodes × local unknowns²). The view reuses the
    /// stored CSR for products, adjoints and numerical publication.
    /// # Errors
    /// Returns invalid state/workspace, numerical and execution failures.
    pub fn linearize<'a, 'scope, 'capability>(
        &'a self,
        point: &'a [R],
        workspace: &'a mut NavierStokesLinearizationWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SimplexLinearization<'a, R>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "navier-stokes-linearization", 1),
            |context| {
                let p = &self.stokes;
                p.check(point, point, &[], 0)?;
                let storage = &mut workspace.storage;
                if storage.matrices.len() != p.matrices.len() {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        "Navier–Stokes linearization size",
                    ));
                }
                let m = p.local();
                p.space.for_each_chunk(
                    &mut storage.matrices,
                    m * m,
                    context,
                    &|cell, matrix, _| {
                        self.local_tangent(cell, point, matrix)?;
                        for (a, b) in matrix
                            .iter_mut()
                            .zip(&p.matrices[cell * m * m..(cell + 1) * m * m])
                        {
                            *a = finite(*a + *b, scioxide_core::NonFiniteOrigin::Intermediate)?;
                        }
                        Ok(())
                    },
                )?;
                p.assembly.assemble_into(
                    &p.space,
                    &mut storage.matrix,
                    &mut storage.assembly,
                    &|cell, i, j| Ok(storage.matrices[(cell * m + i) * m + j]),
                    &p.assembly_extra,
                    context.execution(),
                )
            },
        )?;
        Ok(SimplexLinearization {
            point,
            matrix: &workspace.storage.matrix,
        })
    }
    /// Reuse the prepared space, viscosity, density, loads and pressure constraints.
    pub fn new(stokes: StokesPlan<R, R, D>) -> Self {
        Self { stokes }
    }
    /// Borrow shared geometry, fields, blocks and constraint data.
    pub fn stokes(&self) -> &StokesPlan<R, R, D> {
        &self.stokes
    }
    /// Number of nonlinear coordinates, including pressure gauge multipliers.
    pub fn unknowns(&self) -> usize {
        self.stokes.unknowns()
    }
    /// Scalar scratch for residuals, JVPs and adjoint actions.
    pub(super) fn work(&self) -> usize {
        self.unknowns() + self.stokes.work()
    }
    /// Explicit CSR tangent assembled from the same analytic element derivatives.
    /// # Errors
    /// Rejects invalid states, allocation and arithmetic failures.
    pub fn assemble_tangent_csr(&self, state: &[R]) -> Result<CsrMatrix<R>, SciOxideError> {
        self.stokes.check(state, state, &[], 0)?;
        let mut workspace = self.linearization_workspace()?;
        self.linearize(state, &mut workspace, ExecutionContext::sequential_native())?;
        Ok(workspace.storage.matrix)
    }
    /// Analytic convection derivative of one cell over its volume and exterior facet nodes.
    fn local_tangent(
        &self,
        cell: usize,
        state: &[R],
        local: &mut [R],
    ) -> Result<(), SciOxideError> {
        let p = &self.stokes;
        let m = p.local();
        let velocity = D * p.space.order.nodes(D);
        local.fill(R::ZERO);
        for (boundary, samples) in [
            (false, p.samples.cell(cell)),
            (true, p.samples.cell_facets(cell)),
        ] {
            for s in samples {
                let (u, _) = p.values(s, state, true)?;
                for i in 0..velocity {
                    for j in 0..velocity {
                        local[i * m + j] = finite(
                            local[i * m + j] + self.derivative_entry(s, &u, i, j, boundary),
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
    fn derivative_entry(
        &self,
        s: &Sample<R, D>,
        u: &[R; D],
        i: usize,
        j: usize,
        boundary: bool,
    ) -> R {
        let l = self.stokes.space.order.nodes(D);
        let values = self.stokes.basis(s);
        let a = i / l;
        let b = j / l;
        let ni = i % l;
        let nj = j % l;
        let derivative = if boundary {
            let un = u
                .iter()
                .zip(&s.normal)
                .fold(R::ZERO, |v, (&x, &n)| v + x * n);
            values[ni] * values[nj] * (if a == b { un } else { R::ZERO } + u[a] * s.normal[b])
        } else {
            let ug = u
                .iter()
                .zip(&s.gradients[ni])
                .fold(R::ZERO, |v, (&x, &g)| v + x * g);
            -values[nj] * (if a == b { ug } else { R::ZERO } + u[a] * s.gradients[ni][b])
        };
        self.stokes.density * s.weight * derivative
    }
    pub(super) fn convection(
        &self,
        x: &[R],
        direction: Option<&[R]>,
        adjoint: bool,
        out: &mut [R],
        work: &mut [R],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let p = &self.stokes;
        let n = self.unknowns();
        let m = p.local();
        let l = p.space.order.nodes(D);
        let (global, local) = work.split_at_mut(n);
        let local = &mut local[..p.mesh().cell_count() * m];
        local.fill(R::ZERO);
        p.space.partitioned(local, m, execution, &|first, part| {
            for (offset, y) in part.chunks_exact_mut(m).enumerate() {
                let cell = first + offset;
                for (boundary, samples) in [
                    (false, p.samples.cell(cell)),
                    (true, p.samples.cell_facets(cell)),
                ] {
                    for s in samples {
                        execution.check()?;
                        let (u, _) = p.values(s, x, true)?;
                        if let Some(d) = direction {
                            for i in 0..D * l {
                                let row = p.global(s.cell, i);
                                if p.prescribed(row).is_some() {
                                    continue;
                                }
                                for j in 0..D * l {
                                    let col = p.global(s.cell, j);
                                    if p.prescribed(col).is_some() {
                                        continue;
                                    }
                                    let (target, value) =
                                        if adjoint { (j, d[row]) } else { (i, d[col]) };
                                    let k = target;
                                    y[k] = finite(
                                        y[k] + self.derivative_entry(s, &u, i, j, boundary) * value,
                                        scioxide_core::NonFiniteOrigin::Intermediate,
                                    )?;
                                }
                            }
                        } else {
                            for a in 0..D {
                                for i in 0..l {
                                    let projection = if boundary {
                                        p.basis(s)[i]
                                            * u.iter()
                                                .zip(&s.normal)
                                                .fold(R::ZERO, |v, (&x, &g)| v + x * g)
                                    } else {
                                        -u.iter()
                                            .zip(&s.gradients[i])
                                            .fold(R::ZERO, |v, (&x, &g)| v + x * g)
                                    };
                                    let k = a * l + i;
                                    y[k] = finite(
                                        y[k] + p.density * s.weight * u[a] * projection,
                                        scioxide_core::NonFiniteOrigin::Intermediate,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
        p.space.scatter(
            StokesPlan::<R, R, D>::layout(),
            local,
            &mut global[..p.physical()],
            execution,
        )?;
        for i in 0..p.velocity_unknowns() {
            if p.prescribed(i).is_none() {
                out[i] = finite(
                    out[i] + global[i],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(())
    }
    fn evaluate(
        &self,
        x: &[R],
        direction: Option<&[R]>,
        adjoint: bool,
        out: &mut [R],
        scratch: &mut [R],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let p = &self.stokes;
        p.check(x, out, scratch, self.work())?;
        if let Some(d) = direction {
            p.check(d, out, &[], 0)?;
        }
        let (stage, work) = scratch.split_at_mut(self.unknowns());
        let op = IncompressibleOperator {
            plan: p,
            mass: false,
            gauged: true,
        };
        op.apply(direction.unwrap_or(x), stage, work, execution)?;
        if direction.is_none() {
            for (r, &f) in stage.iter_mut().zip(p.rhs()) {
                *r = finite(*r - f, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
        }
        self.convection(x, direction, adjoint, stage, work, execution)?;
        execution.check()?;
        out.copy_from_slice(stage);
        Ok(())
    }
}
impl<R: RealScalar, const D: usize> ResidualOperator<R> for NavierStokesPlan<R, D> {
    fn input_dimension(&self) -> usize {
        self.unknowns()
    }
    fn output_dimension(&self) -> usize {
        self.unknowns()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.work())
    }
    fn residual_into(&self, x: &[R], out: &mut [R], work: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(
            x,
            None,
            false,
            out,
            work,
            ExecutionContext::sequential_native(),
        )
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, None, false, out, work, ctx.execution())
    }
}
impl<R: RealScalar, const D: usize> JacobianOperator<R> for NavierStokesPlan<R, D> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.work())
    }
    fn jvp_into(
        &self,
        x: &[R],
        d: &[R],
        out: &mut [R],
        work: &mut [R],
    ) -> Result<(), SciOxideError> {
        self.evaluate(
            x,
            Some(d),
            false,
            out,
            work,
            ExecutionContext::sequential_native(),
        )
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        d: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(d), false, out, work, ctx.execution())
    }
}
/// Dedicated analytic adjoint of the conservative convection linearization.
impl<R: RealScalar, const D: usize> crate::PdeAdjointJacobian<R> for NavierStokesPlan<R, D> {
    fn vjp_into(
        &self,
        x: &[R],
        seed: &[R],
        out: &mut [R],
        work: &mut [R],
    ) -> Result<(), SciOxideError> {
        self.evaluate(
            x,
            Some(seed),
            true,
            out,
            work,
            ExecutionContext::sequential_native(),
        )
    }
    fn vjp_into_scoped(
        &self,
        x: &[R],
        seed: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(seed), true, out, work, ctx.execution())
    }
}
