//! Free-unknown semidiscretization of a prepared spatial plan.

use alloc::vec::Vec;
use core::cell::{RefCell, RefMut};
use core::marker::PhantomData;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    NumericScalar, RealScalar, SciOxideError, SolveContext, StructuralPattern,
    StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{OdeAdjointSystem, OdeSystem};

use crate::PdeParameterization;
use crate::boundary::FieldData;
use crate::discretization::SpatialDiscretization;
use crate::mol::{MethodOfLinesFields, MethodOfLinesOrder, MethodOfLinesPlan};
use crate::validation::{finite, invalid, shape};

struct Work<R> {
    load: Vec<R>,
    load_scratch: Vec<R>,
    load_ready: bool,
    applied: Vec<R>,
}

/// Physical residual `f(t, u) = b_f(t) − (A u)_f` of a prepared plan over its
/// free (non-Dirichlet) unknowns, with constant Jacobian `−A_ff`.
///
/// Dirichlet unknowns are eliminated from the state and reconstructed from the
/// plan's time-dependent data. Time-dependent loads re-evaluate sources and
/// boundary data at every call. When both are declared time-invariant, the first
/// successful load is reused for this system's lifetime; construct a new system
/// after changing time-invariant data captured by a callback.
/// RHS and Jacobian products reduce the prepared free rows in native arithmetic
/// with range-safe replay, without allocation; errors leave their output unchanged.
/// Bind it to [`MethodOfLinesPlan`] with its lumped mass and Jacobian pattern,
/// first order for parabolic and second order for wave problems.
///
/// ```
/// use scioxide_pde::{FieldData, MethodOfLinesOrder, SemidiscreteSystem, StructuredFdmPlan};
/// use scioxide_ode::OdeSystem;
/// let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
/// let source = FieldData::zero();
/// let system = SemidiscreteSystem::new(&plan, &source)?;
/// assert_eq!(system.dimension(), 3);
/// let adapter = system.method_of_lines(MethodOfLinesOrder::First)?;
/// assert_eq!(adapter.state_dimension(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct SemidiscreteSystem<'a, R: RealScalar, P, const D: usize> {
    plan: &'a P,
    source: &'a FieldData<R, R, D>,
    free: Vec<usize>,
    mass: Vec<R>,
    pattern: StructuralPattern,
    transpose: scioxide_sparse::SparseTraversal,
    jacobian: Vec<R>,
    bounds: crate::kernels::RowBounds<R>,
    cache_load: bool,
    work: RefCell<Work<R>>,
    marker: PhantomData<R>,
}

impl<'a, R: RealScalar, P: SpatialDiscretization<R, R, D>, const D: usize>
    SemidiscreteSystem<'a, R, P, D>
{
    /// Prepare the free-unknown system, its mass, pattern and Jacobian values.
    /// A fully constrained plan has no free dynamics and reconstructs data only.
    /// # Errors
    /// Rejects inconsistent provider shapes and nonpositive or nonfinite mass,
    /// and propagates assembly and allocation failures.
    pub fn new(plan: &'a P, source: &'a FieldData<R, R, D>) -> Result<Self, SciOxideError> {
        let unknowns = plan.unknowns();
        let constrained = plan.constrained();
        crate::discretization::validate_structure(plan)?;
        let mut free = try_with_capacity(unknowns, scioxide_core::quantity!("free_unknowns"))?;
        let mut inverse = try_filled(unknowns, None, scioxide_core::quantity!("free_unknowns"))?;
        for (i, &fixed) in constrained.iter().enumerate() {
            if !fixed {
                inverse[i] = Some(free.len());
                free.push(i);
            }
        }
        let mut mass = try_with_capacity(free.len(), scioxide_core::quantity!("free_mass"))?;
        mass.extend(free.iter().map(|&i| plan.lumped_mass()[i]));
        let csr = plan.assemble_csr()?;
        shape(csr.rows(), unknowns)?;
        shape(csr.cols(), unknowns)?;
        let mut rows = try_with_capacity(free.len() + 1, scioxide_core::quantity!("free_pattern"))?;
        // Every free row keeps its diagonal structurally, so stage matrices
        // `I - h γ J` stay factorizable when the operator diagonal vanishes.
        let capacity = csr.nnz().saturating_add(free.len());
        let mut cols = try_with_capacity(capacity, scioxide_core::quantity!("free_pattern"))?;
        let mut jacobian = try_with_capacity(capacity, scioxide_core::quantity!("free_jacobian"))?;
        rows.push(0);
        for (diagonal, &row) in free.iter().enumerate() {
            let mut pending = true;
            for k in csr.row_ptr()[row]..csr.row_ptr()[row + 1] {
                if let Some(column) = inverse[csr.col_indices()[k]] {
                    if pending && column >= diagonal {
                        pending = false;
                        if column > diagonal {
                            cols.push(diagonal);
                            jacobian.push(R::ZERO);
                        }
                    }
                    cols.push(column);
                    jacobian.push(-csr.values()[k]);
                }
            }
            if pending {
                cols.push(diagonal);
                jacobian.push(R::ZERO);
            }
            rows.push(cols.len());
        }
        let pattern = StructuralPattern::new(free.len(), free.len(), rows, cols)?;
        let transpose = scioxide_sparse::transpose_pattern(pattern.view())?;
        let vector = |quantity| try_filled(unknowns, R::ZERO, quantity);
        Ok(Self {
            plan,
            source,
            cache_load: source.time_dependence() == crate::TimeDependence::TimeInvariant
                && plan.load_time_dependence() == crate::TimeDependence::TimeInvariant,
            work: RefCell::new(Work {
                load: vector(scioxide_core::quantity!("semidiscrete_load"))?,
                load_scratch: try_filled(
                    plan.load_workspace_len(),
                    R::ZERO,
                    scioxide_core::quantity!("load_scratch"),
                )?,
                load_ready: false,
                applied: vector(scioxide_core::quantity!("semidiscrete_applied"))?,
            }),
            free,
            mass,
            bounds: crate::kernels::RowBounds::new(&jacobian, pattern.view().row_ptr()),
            pattern,
            transpose,
            jacobian,
            marker: PhantomData,
        })
    }

    /// Borrowed plan.
    pub const fn plan(&self) -> &'a P {
        self.plan
    }

    /// Plan unknown index of every free state component.
    pub fn free_indices(&self) -> &[usize] {
        &self.free
    }

    /// Positive lumped mass of the free unknowns.
    pub fn free_mass(&self) -> &[R] {
        &self.mass
    }

    /// Free-free Jacobian pattern.
    pub fn jacobian_pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }

    /// Adapter configuration with the free mass and Jacobian pattern.
    /// # Errors
    /// Propagates mass validation and allocation failures.
    pub fn method_of_lines(
        &self,
        order: MethodOfLinesOrder,
    ) -> Result<MethodOfLinesPlan<R>, SciOxideError> {
        MethodOfLinesPlan::new(self.free.len(), Some(&self.mass), None)?
            .with_order(order)
            .with_jacobian_pattern(self.pattern.view().to_owned()?)
    }

    /// Sample initial data at the free unknowns' sites, at time zero.
    /// # Errors
    /// Propagates callback and allocation failures.
    pub fn initial_state(&self, initial: &FieldData<R, R, D>) -> Result<Vec<R>, SciOxideError> {
        let mut state =
            try_with_capacity(self.free.len(), scioxide_core::quantity!("initial_state"))?;
        for &unknown in &self.free {
            state.push(initial.evaluate(self.plan.dof_point(unknown)?, R::ZERO)?);
        }
        Ok(state)
    }

    fn work(&self) -> Result<RefMut<'_, Work<R>>, SciOxideError> {
        self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "semidiscrete workspace in use",
            )
        })
    }

    fn load(&self, time: R, work: &mut Work<R>) -> Result<(), SciOxideError> {
        if !time.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "time must be finite",
            ));
        }
        if !self.cache_load || !work.load_ready {
            crate::validation::poison(&mut work.load);
            self.plan
                .load_into(self.source, time, &mut work.load, &mut work.load_scratch)?;
            crate::validation::finite_slice(&work.load, scioxide_core::NonFiniteOrigin::Callback)?;
            work.load_ready = true;
        }
        Ok(())
    }

    fn product(
        &self,
        x: &[R],
        load: Option<&[R]>,
        out: &mut [R],
        native: bool,
    ) -> Result<(), SciOxideError> {
        let pattern = self.pattern.view();
        for (i, value) in out.iter_mut().enumerate() {
            let range = pattern.row_ptr()[i]..pattern.row_ptr()[i + 1];
            let values = &self.jacobian[range.clone()];
            let columns = &pattern.col_indices()[range];
            let prior = load.map_or(R::ZERO, |load| load[self.free[i]]);
            let candidate = if native {
                crate::kernels::row(prior, values, columns, x)
            } else {
                R::NAN
            };
            *value = if candidate.is_finite() {
                candidate
            } else {
                crate::kernels::checked_row(prior, values, columns, x)?
            };
        }
        Ok(())
    }
}

impl<R: RealScalar, P: SpatialDiscretization<R, R, D>, const D: usize> OdeSystem<R>
    for SemidiscreteSystem<'_, R, P, D>
{
    fn dimension(&self) -> usize {
        self.free.len()
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn rhs(
        &self,
        t: R,
        y: &[R],
        out: &mut [R],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        shape(y.len(), self.dimension())?;
        shape(out.len(), self.dimension())?;
        let native = self.bounds.admits(y)?;
        let mut w = self.work()?;
        let w = &mut *w;
        self.load(t, w)?;
        self.product(y, Some(&w.load), &mut w.applied[..self.dimension()], native)?;
        out.copy_from_slice(&w.applied[..self.dimension()]);
        Ok(())
    }
    fn jacobian(
        &self,
        _: R,
        _: &[R],
        out: &mut MatrixViewMut<'_, R>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        shape(out.rows(), self.dimension())?;
        shape(out.cols(), self.dimension())?;
        let pattern = self.pattern.view();
        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                out.set(i, j, R::ZERO)?;
            }
            for k in pattern.row_ptr()[i]..pattern.row_ptr()[i + 1] {
                out.set(i, pattern.col_indices()[k], self.jacobian[k])?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: R,
        _: &[R],
        p: StructuralPatternView<'_>,
        out: &mut [R],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let own = self.pattern.view();
        if p.rows() != own.rows()
            || p.row_ptr() != own.row_ptr()
            || p.col_indices() != own.col_indices()
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                "semidiscrete sparse Jacobian pattern mismatch",
            ));
        }
        shape(out.len(), self.jacobian.len())?;
        out.copy_from_slice(&self.jacobian);
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        _: R,
        _: &[R],
        x: &[R],
        out: &mut [R],
        _: &mut [R],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        shape(x.len(), self.dimension())?;
        shape(out.len(), self.dimension())?;
        let native = self.bounds.admits(x)?;
        let mut w = self.work()?;
        self.product(x, None, &mut w.applied[..self.dimension()], native)?;
        out.copy_from_slice(&w.applied[..self.dimension()]);
        Ok(Some(()))
    }
}

impl<R: RealScalar, P: SpatialDiscretization<R, R, D>, const D: usize> MethodOfLinesFields<R>
    for SemidiscreteSystem<'_, R, P, D>
{
    fn field_len(&self) -> usize {
        self.plan.unknowns()
    }
    fn reconstruct_into(&self, time: R, free: &[R], full: &mut [R]) -> Result<(), SciOxideError> {
        shape(free.len(), self.dimension())?;
        shape(full.len(), self.field_len())?;
        crate::validation::finite_slice(free, scioxide_core::NonFiniteOrigin::Input)?;
        let mut w = self.work()?;
        let w = &mut *w;
        self.load(time, w)?;
        for ((out, &fixed), &value) in full.iter_mut().zip(self.plan.constrained()).zip(&w.load) {
            *out = if fixed { value } else { R::ZERO };
        }
        for (&i, &value) in self.free.iter().zip(free) {
            full[i] = value;
        }
        Ok(())
    }
}

/// Default sparse stage backend sized for a pattern.
pub(crate) fn sparse_backend(
    pattern: StructuralPatternView<'_>,
) -> Result<scioxide_ode::OdeStageBackend, SciOxideError> {
    let n = pattern.rows();
    let overflow = || SciOxideError::size_overflow(scioxide_core::quantity!("transient_lu_fill"));
    let m = n.checked_mul(n.saturating_sub(1)).ok_or_else(overflow)? / 2;
    let symbolic = scioxide_sparse::SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::FillReducing,
        None,
        scioxide_sparse::SparseSymmetricLimits::new(m, m, usize::MAX),
    )?;
    let entries = symbolic
        .cholesky_entries()
        .checked_mul(2)
        .and_then(|c| c.checked_add(n))
        .ok_or_else(overflow)?;
    Ok(scioxide_ode::OdeStageBackend::Sparse {
        pattern: pattern.to_owned()?,
        limits: scioxide_sparse::SparseLuLimits::new(entries, usize::MAX),
    })
}

/// A [`SemidiscreteSystem`] whose data depend on real parameters through a
/// [`PdeParameterization`] of its plan: an [`OdeAdjointSystem`] for the
/// discrete adjoints of `scioxide-ode`.
///
/// The pullback of `f(t, u_f) = −R_f(u, p, t)` is `f_uᵀ s = −A_ffᵀ s` for the state
/// and `f_pᵀ s = −Σ_f s_f (∂R/∂p)_f` for the parameters. The plan's mass is
/// parameter independent; bind it through [`MethodOfLinesPlan::bind_adjoint`].
/// Data callbacks must be deterministic (replayable).
pub struct SemidiscreteAdjointSystem<
    's,
    'a,
    R: RealScalar + NumericScalar<Real = R>,
    P,
    const D: usize,
> {
    system: &'s SemidiscreteSystem<'a, R, P, D>,
    parameterization: &'s PdeParameterization<R, R, D>,
    scratch: usize,
}

impl<'a, R: RealScalar + NumericScalar<Real = R>, P: SpatialDiscretization<R, R, D>, const D: usize>
    SemidiscreteSystem<'a, R, P, D>
{
    /// Bind parameter derivatives of this system's plan.
    /// # Errors
    /// Rejects a parameterization of another mesh revision or unknown count.
    pub fn adjoint<'s>(
        &'s self,
        parameterization: &'s PdeParameterization<R, R, D>,
    ) -> Result<SemidiscreteAdjointSystem<'s, 'a, R, P, D>, SciOxideError>
    where
        P: crate::DifferentiableDiscretization<R, R, D>,
    {
        parameterization.check_plan(self.plan)?;
        self.plan
            .mesh_token()
            .ensure_compatible(parameterization.mesh_token())?;
        if parameterization.unknowns() != self.plan.unknowns() {
            return Err(invalid(
                scioxide_core::argument!("parameterization"),
                "parameterization belongs to a plan of another size",
            ));
        }
        Ok(SemidiscreteAdjointSystem {
            system: self,
            parameterization,
            scratch: scioxide_core::backend::checked_add(
                scioxide_core::backend::checked_mul(
                    parameterization.unknowns(),
                    2,
                    scioxide_core::quantity!("pullback_scratch"),
                )?,
                scioxide_core::backend::checked_add(
                    parameterization.scratch_len(),
                    scioxide_core::backend::checked_add(
                        self.dimension(),
                        parameterization.len(),
                        scioxide_core::quantity!("pullback_scratch"),
                    )?,
                    scioxide_core::quantity!("pullback_scratch"),
                )?,
                scioxide_core::quantity!("pullback_scratch"),
            )?,
        })
    }

    /// `out = J_ffᵀ x = −A_ffᵀ x`.
    fn transpose_product(&self, x: &[R], out: &mut [R]) -> Result<(), SciOxideError> {
        let traversal = self.transpose.view();
        let pattern = traversal.pattern();
        for (row, value) in out.iter_mut().enumerate() {
            *value = scioxide_core::backend::standard::sum_products(
                (pattern.row_ptr()[row]..pattern.row_ptr()[row + 1]).map(|k| {
                    (
                        self.jacobian[traversal.positions()[k]],
                        x[pattern.col_indices()[k]],
                    )
                }),
                scioxide_core::Accuracy::Standard,
            )?;
        }
        Ok(())
    }
}

impl<R: RealScalar + NumericScalar<Real = R>, P: SpatialDiscretization<R, R, D>, const D: usize>
    OdeSystem<R> for SemidiscreteAdjointSystem<'_, '_, R, P, D>
{
    fn dimension(&self) -> usize {
        self.system.dimension()
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn rhs(
        &self,
        t: R,
        y: &[R],
        out: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.system.rhs(t, y, out, context)
    }
    fn jacobian(
        &self,
        t: R,
        y: &[R],
        out: &mut MatrixViewMut<'_, R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.jacobian(t, y, out, context)
    }
    fn sparse_jacobian(
        &self,
        t: R,
        y: &[R],
        p: StructuralPatternView<'_>,
        out: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.sparse_jacobian(t, y, p, out, context)
    }
    fn jacobian_product(
        &self,
        t: R,
        y: &[R],
        x: &[R],
        out: &mut [R],
        scratch: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.jacobian_product(t, y, x, out, scratch, context)
    }
}

impl<R: RealScalar + NumericScalar<Real = R>, P: SpatialDiscretization<R, R, D>, const D: usize>
    OdeAdjointSystem<R> for SemidiscreteAdjointSystem<'_, '_, R, P, D>
{
    fn parameters(&self) -> &[R] {
        self.parameterization.values()
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback_workspace_len(&self) -> usize {
        self.scratch
    }
    fn pullback(
        &self,
        time: R,
        state: &[R],
        seed: &[R],
        state_output: &mut [R],
        parameter_output: &mut [R],
        scratch: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let m = self.system.dimension();
        let n = self.parameterization.unknowns();
        shape(state.len(), m)?;
        shape(seed.len(), m)?;
        shape(state_output.len(), m)?;
        shape(parameter_output.len(), self.parameterization.len())?;
        finite(time, scioxide_core::NonFiniteOrigin::Input)?;
        crate::validation::finite_slice(state, scioxide_core::NonFiniteOrigin::Input)?;
        crate::validation::finite_slice(seed, scioxide_core::NonFiniteOrigin::Input)?;
        context.execution().check()?;
        if scratch.len() < self.pullback_workspace_len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.pullback_workspace_len()),
                scioxide_core::Dimensions::one(scratch.len()),
            ));
        }
        let (state_staged, rest) = scratch.split_at_mut(m);
        let (parameter_staged, rest) = rest.split_at_mut(self.parameterization.len());
        let (full, rest) = rest.split_at_mut(n);
        let (tangent, rest) = rest.split_at_mut(n);
        full.fill(R::ZERO);
        for (&i, &value) in self.system.free.iter().zip(state) {
            full[i] = value;
        }
        for (j, out) in parameter_staged.iter_mut().enumerate() {
            self.parameterization
                .residual_tangent_into(j, full, time, tangent, rest)?;
            *out = -scioxide_core::backend::standard::sum_products(
                self.system
                    .free
                    .iter()
                    .zip(seed)
                    .map(|(&i, &s)| (s, tangent[i])),
                scioxide_core::Accuracy::Standard,
            )?;
        }
        self.system.transpose_product(seed, state_staged)?;
        context.execution().check()?;
        state_output.copy_from_slice(state_staged);
        parameter_output.copy_from_slice(parameter_staged);
        Ok(())
    }
}

#[cfg(test)]
mod factor_tests {
    use super::*;
    #[test]
    fn symbolic_reservation_matches_diffusion_factor_fill() -> Result<(), SciOxideError> {
        fn check<const D: usize>(nodes: usize) -> Result<(), SciOxideError> {
            let plan =
                crate::StructuredFdmPlan::dirichlet_laplacian([0.0_f64; D], [1.; D], [nodes; D])?;
            let source = FieldData::zero();
            let semi = SemidiscreteSystem::new(&plan, &source)?;
            let scioxide_ode::OdeStageBackend::Sparse { pattern, limits } =
                sparse_backend(semi.jacobian_pattern())?
            else {
                unreachable!()
            };
            let mut values = semi.jacobian.iter().map(|v| -0.01 * v).collect::<Vec<_>>();
            for row in 0..pattern.view().rows() {
                for (value, &column) in values
                    [pattern.view().row_ptr()[row]..pattern.view().row_ptr()[row + 1]]
                    .iter_mut()
                    .zip(
                        &pattern.view().col_indices()
                            [pattern.view().row_ptr()[row]..pattern.view().row_ptr()[row + 1]],
                    )
                {
                    if column == row {
                        *value += semi.mass[row];
                    }
                }
            }
            let matrix = scioxide_sparse::CsrMatrix::from_pattern(pattern, values)?;
            let symbolic = scioxide_sparse::SparseLuSymbolic::analyze(
                matrix.structural_pattern(),
                scioxide_sparse::SparseOrdering::FillReducing,
                limits,
            )?;
            let factor = scioxide_sparse::SparseLu::factor_scoped(
                symbolic,
                matrix.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            assert_eq!(limits.max_factor_entries(), factor.factor_entries());
            assert_eq!(limits.max_workspace_bytes(), usize::MAX);
            Ok(())
        }
        check::<1>(81)?;
        check::<2>(101)?;
        check::<3>(11)
    }
}
