//! Prepared block factorization inverses for viscous and velocity-mass saddles.
use super::*;
use alloc::boxed::Box;
use scioxide_core::{AlgebraicScalar, OperatorContext, Preconditioner, ProductSum};
use scioxide_sparse::{
    AmgCycleOptions, AmgHierarchy, AmgLimits, AmgPreconditioner, AmgSetupOptions,
    JacobiPreconditioner, SparseCholesky, SparseOrdering, SparseProductOptions, SparseProductPlan,
    SparseProductPlanOptions, SparseSolveOptions, SparseSymmetricLimits, SparseSymmetricSymbolic,
};

/// Prepared inverse used for the positive viscous velocity block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IncompressibleVelocityInverse {
    /// Component elliptic AMG cycles with restricted translation modes.
    Amg,
    /// Sparse Cholesky on the coupled velocity block (default), subject to factor limits.
    #[default]
    SparseCholesky,
}

/// The saddle system a library preconditioner approximates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IncompressibleSaddle {
    /// The steady viscous Stokes operator; also a frozen Newton preconditioner.
    Viscous,
    /// The density-scaled velocity-mass saddle of constrained acceleration.
    VelocityMass,
}

/// Sparse setup controls for the library saddle preconditioner.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct IncompressiblePreconditionerOptions<R: RealScalar> {
    velocity: IncompressibleVelocityInverse,
    amg_setup: AmgSetupOptions<R>,
    amg_cycle: AmgCycleOptions,
    amg_limits: AmgLimits,
    factor_limits: SparseSymmetricLimits,
    max_entries: usize,
}
impl<R: RealScalar> Default for IncompressiblePreconditionerOptions<R> {
    fn default() -> Self {
        Self {
            velocity: IncompressibleVelocityInverse::SparseCholesky,
            amg_setup: AmgSetupOptions::new(),
            amg_cycle: AmgCycleOptions::new()
                .with_pre_sweeps(2)
                .with_post_sweeps(2),
            amg_limits: AmgLimits::new(32, 16_000_000, 512 << 20),
            factor_limits: SparseSymmetricLimits::new(8_000_000, 0, 1 << 30),
            max_entries: 4_000_000,
        }
    }
}
impl<R: RealScalar> IncompressiblePreconditionerOptions<R> {
    /// Sparse Cholesky velocity blocks, two-sweep AMG cycles and 4,000,000-entry blocks.
    pub fn new() -> Self {
        Self::default()
    }
    /// Viscous velocity inverse; mass saddles always use diagonal mass inversion.
    pub const fn with_velocity(mut self, velocity: IncompressibleVelocityInverse) -> Self {
        self.velocity = velocity;
        self
    }
    /// Hierarchy aggregation controls.
    pub const fn with_amg_setup(mut self, setup: AmgSetupOptions<R>) -> Self {
        self.amg_setup = setup;
        self
    }
    /// Fixed linear cycle for AMG blocks.
    pub const fn with_amg_cycle(mut self, cycle: AmgCycleOptions) -> Self {
        self.amg_cycle = cycle;
        self
    }
    /// AMG depth, expanded-product and setup-byte limits.
    pub const fn with_amg_limits(mut self, limits: AmgLimits) -> Self {
        self.amg_limits = limits;
        self
    }
    /// Sparse direct and coarse-factor limits.
    pub const fn with_factor_limits(mut self, limits: SparseSymmetricLimits) -> Self {
        self.factor_limits = limits;
        self
    }
    /// Maximum entries in each assembled block or mass Schur product.
    pub const fn with_max_entries(mut self, entries: usize) -> Self {
        self.max_entries = entries;
        self
    }
    /// Viscous velocity inverse.
    pub const fn velocity(&self) -> IncompressibleVelocityInverse {
        self.velocity
    }
    /// Hierarchy aggregation controls.
    pub const fn amg_setup(&self) -> &AmgSetupOptions<R> {
        &self.amg_setup
    }
    /// AMG cycle.
    pub const fn amg_cycle(&self) -> AmgCycleOptions {
        self.amg_cycle
    }
    /// AMG limits.
    pub const fn amg_limits(&self) -> AmgLimits {
        self.amg_limits
    }
    /// Sparse factor limits.
    pub const fn factor_limits(&self) -> SparseSymmetricLimits {
        self.factor_limits
    }
    /// Maximum assembled entries.
    pub const fn max_entries(&self) -> usize {
        self.max_entries
    }
}

/// Owned block factorization saddle preconditioner, including pressure gauges.
///
/// Applies an approximate saddle LDU inverse: a velocity prediction, a bordered
/// Schur correction and a velocity correction. Viscous systems use
/// `S ≈ M_p/μ`; velocity-mass systems use `S ≈ B diag(M_ρ)⁻¹ Bᴴ`.
/// The latter is inverted on compatible pressure coordinates with temporary
/// pins and exact constant-mode/gauge restoration. Pins change only the
/// preconditioner, never the physical operator. The inverse is generally
/// indefinite and can be nonsymmetric; use GMRES rather than methods requiring
/// a positive preconditioner.
///
/// Setup allocates; applications are transactional and allocation-free. A frozen
/// viscous preconditioner also applies to Oseen/Newton tangents; it does not
/// promise Reynolds-independent convergence.
///
/// ```
/// # fn prepare<R: scioxide_core::RealScalar, const D: usize>(
/// # plan: &scioxide_pde::StokesPlan<R, R, D>) -> Result<(), scioxide_core::SciOxideError> {
/// use scioxide_core::Preconditioner;
/// use scioxide_pde::{
///     IncompressiblePreconditioner, IncompressiblePreconditionerOptions, IncompressibleSaddle,
/// };
/// let inverse = IncompressiblePreconditioner::new(plan, IncompressibleSaddle::Viscous,
///     IncompressiblePreconditionerOptions::new())?;
/// assert_eq!(inverse.dimension(), plan.unknowns());
/// # Ok(()) }
/// ```
pub struct IncompressiblePreconditioner<V: NumericScalar> {
    velocity: BlockInverse<V>,
    pressure: BlockInverse<V>,
    free: Vec<usize>,
    velocity_count: usize,
    coupling: CsrMatrix<V>,         // free velocity rows, pressure columns
    components: Vec<Option<usize>>, // pressure vertex -> gauge, if closed
    weights: Vec<V>,
    gauge_sums: Vec<V>,
    gauge_inverse: Vec<V>, // S^-1 C for the nonsingular (mass-matrix) approximation
    gauge_denominators: Vec<V>,
    pins: Vec<usize>,
    mass: bool,
    dimension: usize,
    workspace_len: usize,
}

fn finish<V: NumericScalar>(builder: SparseBuilder<V>) -> Result<CsrMatrix<V>, SciOxideError> {
    builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
}
enum BlockInverse<V: NumericScalar> {
    Direct(Box<DirectInverse<V>>),
    Diagonal(JacobiPreconditioner<V>),
    Amg(Box<AmgPreconditioner<V>>),
    Components(ComponentInverse<V>),
}
impl<V: NumericScalar> BlockInverse<V> {
    fn provider(&self) -> &dyn Preconditioner<V> {
        match self {
            Self::Direct(p) => p.as_ref(),
            Self::Diagonal(p) => p,
            Self::Amg(p) => p.as_ref(),
            Self::Components(p) => p,
        }
    }
}
impl<V: NumericScalar> Preconditioner<V> for BlockInverse<V> {
    fn dimension(&self) -> usize {
        self.provider().dimension()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider().precondition_workspace_len()
    }
    fn precondition_into(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.provider().precondition_into(input, output, work)
    }
    fn precondition_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
        ctx: &mut OperatorContext<'_, '_, '_, V>,
    ) -> Result<(), SciOxideError> {
        self.provider()
            .precondition_into_scoped(input, output, work, ctx)
    }
}

struct ComponentInverse<V: NumericScalar> {
    blocks: Vec<BlockInverse<V>>,
    offsets: Vec<usize>,
}
impl<V: NumericScalar> Preconditioner<V> for ComponentInverse<V> {
    fn dimension(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.blocks
            .iter()
            .try_fold(0, |n, block| Ok(n.max(block.precondition_workspace_len()?)))
    }
    fn precondition_into(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
    ) -> Result<(), SciOxideError> {
        if input.len() != self.dimension()
            || output.len() != self.dimension()
            || work.len() < self.precondition_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "invalid component inverse buffers",
            ));
        }
        for (block, range) in self.blocks.iter().zip(self.offsets.windows(2)) {
            block.precondition_into(
                &input[range[0]..range[1]],
                &mut output[range[0]..range[1]],
                work,
            )?;
        }
        Ok(())
    }
    fn precondition_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
        ctx: &mut OperatorContext<'_, '_, '_, V>,
    ) -> Result<(), SciOxideError> {
        for (block, range) in self.blocks.iter().zip(self.offsets.windows(2)) {
            ctx.execution().check()?;
            block.precondition_into_scoped(
                &input[range[0]..range[1]],
                &mut output[range[0]..range[1]],
                work,
                ctx,
            )?;
        }
        Ok(())
    }
}
struct DirectInverse<V: NumericScalar>(SparseCholesky<V>);
impl<V: NumericScalar> Preconditioner<V> for DirectInverse<V> {
    fn dimension(&self) -> usize {
        self.0.rows()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.0.apply_workspace_len()
    }
    fn precondition_into(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.0.apply_into(input, output, work)
    }
    fn precondition_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
        ctx: &mut OperatorContext<'_, '_, '_, V>,
    ) -> Result<(), SciOxideError> {
        self.0.solve_into(
            input,
            scioxide_sparse::SparseOperation::Normal,
            SparseSolveOptions::new(),
            output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(work),
            &mut **ctx,
        )
    }
}
fn direct<V: NumericScalar>(
    matrix: &CsrMatrix<V>,
    limits: SparseSymmetricLimits,
    ctx: &mut SolveContext<'_, '_>,
) -> Result<BlockInverse<V>, SciOxideError> {
    let dense_entries = size(
        matrix.rows(),
        matrix.rows().saturating_sub(1),
        scioxide_core::quantity!("factor_entries"),
    )? / 2;
    let limits = limits.with_max_factor_entries(limits.max_factor_entries().min(dense_entries));
    let symbolic = SparseSymmetricSymbolic::analyze(
        matrix.structural_pattern(),
        SparseOrdering::default(),
        None,
        limits,
    )?;
    Ok(BlockInverse::Direct(Box::new(DirectInverse(
        SparseCholesky::factor_scoped(
            symbolic,
            matrix.view(),
            SparseSolveOptions::new(),
            &mut *ctx,
        )?,
    ))))
}
fn amg<V: NumericScalar>(
    matrix: CsrMatrix<V>,
    modes: &[V],
    columns: usize,
    options: IncompressiblePreconditionerOptions<V::Real>,
    ctx: &mut SolveContext<'_, '_>,
) -> Result<BlockInverse<V>, SciOxideError> {
    let hierarchy = AmgHierarchy::build(
        matrix.view(),
        BlockVectorView::new(modes, matrix.rows(), columns)?,
        options.amg_setup,
        options.amg_limits,
        SparseSolveOptions::new(),
        &mut *ctx,
    )?;
    let coarse = hierarchy.coarse_matrix().rows();
    let coarse_entries = size(
        coarse,
        coarse.saturating_sub(1),
        scioxide_core::quantity!("coarse_entries"),
    )? / 2;
    let limits = options.factor_limits.with_max_factor_entries(
        options
            .factor_limits
            .max_factor_entries()
            .min(coarse_entries),
    );
    Ok(BlockInverse::Amg(Box::new(AmgPreconditioner::build(
        hierarchy,
        options.amg_cycle,
        limits,
        SparseSolveOptions::new(),
        &mut *ctx,
    )?)))
}
impl<V: NumericScalar> IncompressiblePreconditioner<V> {
    /// Prepare the library inverse of `saddle` under sequential native execution.
    /// # Errors
    /// Returns sparse setup, resource, allocation, rank or arithmetic failures.
    pub fn new<const D: usize>(
        plan: &StokesPlan<V::Real, V, D>,
        saddle: IncompressibleSaddle,
        options: IncompressiblePreconditionerOptions<V::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(plan, saddle, options, ExecutionContext::sequential_native())
    }
    /// Prepare as one epoch of `scope`. Supplied limits cover sparse blocks,
    /// hierarchies and factors.
    /// # Errors
    /// Includes cancellation and budget exhaustion in addition to setup errors.
    pub fn build<'scope, 'capability, const D: usize>(
        plan: &StokesPlan<V::Real, V, D>,
        saddle: IncompressibleSaddle,
        options: IncompressiblePreconditionerOptions<V::Real>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "saddle-preconditioner", 2),
            |ctx| {
                Self::prepare(
                    plan,
                    saddle == IncompressibleSaddle::VelocityMass,
                    options,
                    ctx,
                )
            },
        )
    }
    fn prepare<const D: usize>(
        plan: &StokesPlan<V::Real, V, D>,
        mass: bool,
        options: IncompressiblePreconditionerOptions<V::Real>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        admit(ctx, false)?;
        if plan.entry_bound() > options.max_entries {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("saddle_entries"),
                plan.entry_bound(),
                options.max_entries,
            ));
        }
        let full = if mass {
            plan.mass_system().assemble_csr()?
        } else {
            plan.assemble_csr()?
        };
        let nv = plan.velocity_unknowns();
        let np = plan.mesh().vertex_count();
        let nf = plan.free.len();
        let ng = plan.unknowns() - nv - np;
        let mut free = try_with_capacity(nf, scioxide_core::quantity!("free"))?;
        free.extend_from_slice(&plan.free);
        let (a, coupling) = Self::blocks(&full, &free, nv, np, ctx)?;
        let mut diagonal = try_with_capacity(nf, scioxide_core::quantity!("velocity_diagonal"))?;
        for row in 0..nf {
            diagonal.push(a.get(row, row)?);
        }
        let velocity = Self::velocity_inverse(plan, &a, &free, mass, options, ctx)?;
        let mut components = try_with_capacity(np, scioxide_core::quantity!("components"))?;
        let mut weights = try_with_capacity(np, scioxide_core::quantity!("weights"))?;
        let mut gauge_sums = try_filled(ng, V::ZERO, scioxide_core::quantity!("gauge_sums"))?;
        let mut pins = try_filled(ng, usize::MAX, scioxide_core::quantity!("pins"))?;
        for (v, &c) in plan.component.iter().enumerate() {
            let gauge = plan.gauge_index[c];
            components.push(gauge);
            weights.push(V::from_real(plan.gauge_weights[v]));
            if let Some(g) = gauge {
                pins[g] = pins[g].min(v);
                gauge_sums[g] = finite(gauge_sums[g] + weights[v], NonFiniteOrigin::Intermediate)?;
            }
        }
        let pressure =
            Self::pressure_inverse(plan, &coupling, &diagonal, &pins, mass, options, ctx)?;
        let mut gauge_inverse = try_filled(np, V::ZERO, scioxide_core::quantity!("gauge_inverse"))?;
        let mut gauge_denominators =
            try_filled(ng, V::ZERO, scioxide_core::quantity!("gauge_denominators"))?;
        if !mass {
            let mut work = try_filled(
                pressure.precondition_workspace_len()?,
                V::ZERO,
                scioxide_core::quantity!("pressure_work"),
            )?;
            pressure.precondition_into(&weights, &mut gauge_inverse, &mut work)?;
            for (v, &g) in components.iter().enumerate() {
                if let Some(g) = g {
                    gauge_denominators[g] = finite(
                        gauge_denominators[g] + weights[v] * gauge_inverse[v],
                        NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        let operation_work = velocity
            .precondition_workspace_len()?
            .max(pressure.precondition_workspace_len()?);
        let workspace_len = add(
            add(
                plan.unknowns(),
                add(
                    np,
                    size(2, nf, scioxide_core::quantity!("saddle_work"))?,
                    scioxide_core::quantity!("saddle_work"),
                )?,
                scioxide_core::quantity!("saddle_work"),
            )?,
            operation_work,
            scioxide_core::quantity!("saddle_work"),
        )?;
        Ok(Self {
            velocity,
            pressure,
            free,
            velocity_count: nv,
            coupling,
            components,
            weights,
            gauge_sums,
            gauge_inverse,
            gauge_denominators,
            pins,
            mass,
            dimension: plan.unknowns(),
            workspace_len,
        })
    }
    /// Free velocity block `A_ff` and pressure coupling `Bᵀ_f` of the full operator.
    fn blocks(
        full: &CsrMatrix<V>,
        free: &[usize],
        nv: usize,
        np: usize,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(CsrMatrix<V>, CsrMatrix<V>), SciOxideError> {
        let nf = free.len();
        let mut map = try_filled(nv, usize::MAX, scioxide_core::quantity!("velocity_map"))?;
        for (i, &row) in free.iter().enumerate() {
            map[row] = i;
        }
        let mut a = SparseBuilder::with_capacity(nf, nf, full.nnz())?;
        let mut bt = SparseBuilder::with_capacity(nf, np, full.nnz())?;
        for (i, &row) in free.iter().enumerate() {
            admit(ctx, false)?;
            for k in full.row_ptr()[row]..full.row_ptr()[row + 1] {
                let col = full.col_indices()[k];
                let value = full.values()[k];
                if col < nv && map[col] != usize::MAX {
                    a.push(i, map[col], value)?;
                }
                if (nv..nv + np).contains(&col) {
                    bt.push(i, col - nv, value)?;
                }
            }
        }
        Ok((finish(a)?, finish(bt)?))
    }
    /// Velocity block inverse: Jacobi for the mass saddle, sparse Cholesky or
    /// component AMG for the viscous saddle.
    fn velocity_inverse<const D: usize>(
        plan: &StokesPlan<V::Real, V, D>,
        a: &CsrMatrix<V>,
        free: &[usize],
        mass: bool,
        options: IncompressiblePreconditionerOptions<V::Real>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<BlockInverse<V>, SciOxideError> {
        Ok(if mass {
            BlockInverse::Diagonal(JacobiPreconditioner::new(a)?)
        } else if options.velocity == IncompressibleVelocityInverse::SparseCholesky {
            direct(a, options.factor_limits, ctx)?
        } else {
            // Component elliptic inverses are spectrally equivalent to the
            // symmetric-gradient velocity form (Korn inequality). Scalar AMG
            // avoids dense system aggregates on P2 graphs. Each component's
            // constant mode is the corresponding restricted rigid translation.
            let mut blocks = try_with_capacity(D, scioxide_core::quantity!("velocity_blocks"))?;
            let mut offsets =
                try_with_capacity(D + 1, scioxide_core::quantity!("velocity_offsets"))?;
            offsets.push(0);
            for axis in 0..D {
                let start = free.partition_point(|&row| row < axis * plan.velocity_nodes());
                let end = free.partition_point(|&row| row < (axis + 1) * plan.velocity_nodes());
                let mut block =
                    SparseBuilder::with_capacity(end - start, end - start, a.nnz() / D)?;
                for row in start..end {
                    for k in a.row_ptr()[row]..a.row_ptr()[row + 1] {
                        let col = a.col_indices()[k];
                        if (start..end).contains(&col) {
                            block.push(row - start, col - start, a.values()[k])?;
                        }
                    }
                }
                let modes = try_filled(
                    end - start,
                    V::ONE,
                    scioxide_core::quantity!("velocity_modes"),
                )?;
                blocks.push(amg(finish(block)?, &modes, 1, options, ctx)?);
                offsets.push(end);
            }
            BlockInverse::Components(ComponentInverse { blocks, offsets })
        })
    }
    /// Pressure Schur inverse: pinned AMG of `B D⁻¹ Bᵀ` for the mass saddle,
    /// the viscosity-scaled pressure mass for the viscous saddle.
    fn pressure_inverse<const D: usize>(
        plan: &StokesPlan<V::Real, V, D>,
        coupling: &CsrMatrix<V>,
        diagonal: &[V],
        pins: &[usize],
        mass: bool,
        options: IncompressiblePreconditionerOptions<V::Real>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<BlockInverse<V>, SciOxideError> {
        let (nf, np, ng) = (diagonal.len(), plan.mesh().vertex_count(), pins.len());
        Ok(if mass {
            // Symmetric product of identically scaled rows avoids asymmetric rounding.
            let mut scaled = SparseBuilder::with_capacity(nf, np, coupling.nnz())?;
            for (row, &d) in diagonal.iter().enumerate() {
                let root = d.real_component().sqrt();
                if !root.is_finite() || root <= V::Real::ZERO {
                    return Err(SciOxideError::singular(None));
                }
                for k in coupling.row_ptr()[row]..coupling.row_ptr()[row + 1] {
                    scaled.push(
                        row,
                        coupling.col_indices()[k],
                        coupling.values()[k] / V::from_real(root),
                    )?;
                }
            }
            let scaled = finish(scaled)?;
            let transpose = scaled.transpose()?.to_csr()?;
            let product = SparseProductPlan::with_options(
                &transpose,
                &scaled,
                SparseProductPlanOptions::new().with_max_output_entries(options.max_entries),
            )?;
            let mut schur = product.create_output()?;
            let arithmetic = SparseProductOptions::new();
            product.evaluate_into(
                &transpose,
                &scaled,
                arithmetic,
                &mut schur,
                &mut product.workspace(arithmetic)?,
                ctx.execution(),
            )?;
            let mut pinned = SparseBuilder::with_capacity(
                np,
                np,
                add(
                    schur.nnz(),
                    ng,
                    scioxide_core::quantity!("pressure_entries"),
                )?,
            )?;
            for row in 0..np {
                if pins.contains(&row) {
                    pinned.push(row, row, V::ONE)?;
                    continue;
                }
                for k in schur.row_ptr()[row]..schur.row_ptr()[row + 1] {
                    let col = schur.col_indices()[k];
                    if !pins.contains(&col) {
                        pinned.push(row, col, schur.values()[k])?;
                    }
                }
            }
            let pinned = finish(pinned)?;
            let modes = try_filled(np, V::ONE, scioxide_core::quantity!("pressure_modes"))?;
            amg(pinned, &modes, 1, options, ctx)?
        } else {
            let mut mass_matrix = SparseBuilder::with_capacity(
                np,
                np,
                size(
                    plan.mesh().cell_count(),
                    (D + 1) * (D + 1),
                    scioxide_core::quantity!("pressure_entries"),
                )?,
            )?;
            for cell in 0..plan.mesh().cell_count() {
                let vertices = plan.mesh().cell_vertices(cell)?;
                let local = plan.element_pressure_mass(cell)?;
                for (i, &row) in vertices.iter().enumerate() {
                    for (j, &col) in vertices.iter().enumerate() {
                        mass_matrix.push(
                            row,
                            col,
                            V::from_real(local[i * (D + 1) + j] / plan.viscosity),
                        )?;
                    }
                }
            }
            direct(&finish(mass_matrix)?, options.factor_limits, ctx)?
        })
    }
    /// Remove the weighted gauge means of the mass-saddle pressure load and pin it.
    fn project_gauges(&self, pressure_rhs: &mut [V], g: &mut [V]) -> Result<(), SciOxideError> {
        for (v, &component) in self.components.iter().enumerate() {
            if let Some(k) = component {
                g[k] = finite(g[k] + pressure_rhs[v], NonFiniteOrigin::Intermediate)?;
            }
        }
        for (v, sum) in g.iter_mut().zip(&self.gauge_sums) {
            *v = finite(*v / *sum, NonFiniteOrigin::Intermediate)?;
        }
        for (v, &component) in self.components.iter().enumerate() {
            if let Some(k) = component {
                pressure_rhs[v] = finite(
                    pressure_rhs[v] - self.weights[v] * g[k],
                    NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        for &pin in &self.pins {
            pressure_rhs[pin] = V::ZERO;
        }
        Ok(())
    }
    fn apply(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if input.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(input.len()),
            ));
        }
        if output.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(output.len()),
            ));
        }
        if work.len() < self.workspace_len {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "insufficient saddle inverse scratch",
            ));
        }
        if input.iter().any(|v| !v.has_finite_components()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "nonfinite saddle inverse input",
            ));
        }
        admit(ctx, false)?;
        let np = self.components.len();
        let nv = self.velocity_count;
        let ng = self.gauge_sums.len();
        let (stage, work) = work.split_at_mut(self.dimension);
        let (pressure_rhs, work) = work.split_at_mut(np);
        let (velocity_rhs, work) = work.split_at_mut(self.free.len());
        let (velocity_initial, work) = work.split_at_mut(self.free.len());
        let (u, rest) = stage.split_at_mut(nv);
        let (p, g) = rest.split_at_mut(np);
        g.fill(V::ZERO);
        for (value, &row) in velocity_rhs.iter_mut().zip(&self.free) {
            *value = input[row];
        }
        self.velocity.precondition_into_scoped(
            velocity_rhs,
            velocity_initial,
            work,
            &mut OperatorContext::new(ctx),
        )?;
        pressure_rhs.copy_from_slice(&input[nv..nv + np]);
        for (row, &value) in velocity_initial.iter().enumerate() {
            for k in self.coupling.row_ptr()[row]..self.coupling.row_ptr()[row + 1] {
                let col = self.coupling.col_indices()[k];
                pressure_rhs[col] = finite(
                    pressure_rhs[col] - self.coupling.values()[k].conj() * value,
                    NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        if self.mass {
            self.project_gauges(pressure_rhs, g)?;
        }
        self.pressure.precondition_into_scoped(
            pressure_rhs,
            p,
            work,
            &mut OperatorContext::new(ctx),
        )?;
        pressure_rhs.fill(V::ZERO); // reuse one scalar per gauge
        for (v, &component) in self.components.iter().enumerate() {
            if let Some(k) = component {
                pressure_rhs[k] = finite(
                    pressure_rhs[k] + self.weights[v] * p[v],
                    NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        if self.mass {
            for k in 0..ng {
                pressure_rhs[k] = finite(
                    (input[nv + np + k] + pressure_rhs[k]) / self.gauge_sums[k],
                    NonFiniteOrigin::Intermediate,
                )?;
            }
            for (v, value) in p.iter_mut().enumerate() {
                *value = finite(
                    -*value + self.components[v].map_or(V::ZERO, |k| pressure_rhs[k]),
                    NonFiniteOrigin::Intermediate,
                )?;
            }
        } else {
            for k in 0..ng {
                g[k] = finite(
                    (input[nv + np + k] + pressure_rhs[k]) / self.gauge_denominators[k],
                    NonFiniteOrigin::Intermediate,
                )?;
            }
            for (v, value) in p.iter_mut().enumerate() {
                *value = finite(
                    -*value + self.components[v].map_or(V::ZERO, |k| self.gauge_inverse[v] * g[k]),
                    NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        u.copy_from_slice(&input[..nv]);
        for (row, value) in velocity_rhs.iter_mut().enumerate() {
            let mut sum = ProductSum::new();
            for k in self.coupling.row_ptr()[row]..self.coupling.row_ptr()[row + 1] {
                sum.add_product(self.coupling.values()[k], p[self.coupling.col_indices()[k]])?;
            }
            *value = sum.finish()?;
        }
        // Gather into free coordinates, then scatter after the block inverse succeeds.
        let (free_output, _) = u.split_at_mut(self.free.len());
        // Preserve the gathered result before scattering into the original coordinates.
        self.velocity.precondition_into_scoped(
            velocity_rhs,
            free_output,
            work,
            &mut OperatorContext::new(ctx),
        )?;
        velocity_rhs.copy_from_slice(free_output);
        u.copy_from_slice(&input[..nv]);
        for ((&row, &initial), &correction) in self
            .free
            .iter()
            .zip(velocity_initial.iter())
            .zip(velocity_rhs.iter())
        {
            u[row] = finite(initial - correction, NonFiniteOrigin::Intermediate)?;
        }
        ctx.execution().check()?;
        output.copy_from_slice(stage);
        Ok(())
    }
}
impl<V: NumericScalar> Preconditioner<V> for IncompressiblePreconditioner<V> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.workspace_len)
    }
    fn precondition_into(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
    ) -> Result<(), SciOxideError> {
        crate::attempt::sequential(
            scioxide_core::algorithm_id!("pde", "incompressible", 2),
            |ctx| self.apply(input, output, work, ctx),
        )
    }
    fn precondition_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        work: &mut [V],
        ctx: &mut OperatorContext<'_, '_, '_, V>,
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, work, ctx)
    }
}
