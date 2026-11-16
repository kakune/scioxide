//! Flexible Arnoldi with image QR for GMRES, LGMRES, GCROT and their block forms.
//!
//! Image QR applies the same column operations to the solution directions, so
//! that `A P = Q` with orthonormal `Q`. Then `P Qᴴ r` solves the projected
//! least-squares problem without normal equations and without applying a
//! possibly variable preconditioner again.

use crate::Preconditioning;
use crate::kernel::finite;
use crate::{LinearSolveBreakdown, LinearSolveTermination, Vector, VectorView};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Budget, LinearOperator, NumericScalar, ProblemToken, RankThreshold, RealScalar,
    ResidualTolerance, SciOxideError,
};

pub(super) type Result<T> = core::result::Result<T, SciOxideError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    Gmres,
    Lgmres,
    Gcrot,
}

impl Method {
    pub(super) const fn algorithm_id(self, block: bool) -> scioxide_core::AlgorithmId {
        match (self, block) {
            (Method::Gmres, false) => scioxide_core::algorithm_id!("linear", "GMRES", 2),
            (Method::Lgmres, false) => scioxide_core::algorithm_id!("linear", "LGMRES", 2),
            (Method::Gcrot, false) => scioxide_core::algorithm_id!("linear", "GCROT", 2),
            (Method::Gmres, true) => scioxide_core::algorithm_id!("linear", "BlockGMRES", 2),
            (Method::Lgmres, true) => scioxide_core::algorithm_id!("linear", "BlockLGMRES", 2),
            (Method::Gcrot, true) => scioxide_core::algorithm_id!("linear", "BlockGCROT", 2),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Options<R: RealScalar> {
    pub(super) method: Method,
    pub(super) tolerance: ResidualTolerance<R>,
    pub(super) max_iterations: Budget,
    pub(super) execution: crate::LinalgExecutionOptions,
    pub(super) restart: Budget,
    pub(super) rank: RankThreshold<R>,
    pub(super) history: bool,
}

impl<R: RealScalar> Options<R> {
    pub(super) fn new(method: Method, tolerance: ResidualTolerance<R>, limit: Budget) -> Self {
        Self {
            method,
            tolerance,
            max_iterations: limit,
            execution: crate::LinalgExecutionOptions::new(),
            restart: Budget::ONE.saturating_add(29),
            rank: RankThreshold::from_tolerance(scioxide_core::Tolerance::from_machine_epsilon()),
            history: false,
        }
    }

    /// Inner-iteration allowance: `max_iterations`, capped by an explicit
    /// execution budget.
    pub(super) fn limit(&self) -> Budget {
        self.execution.cap_iterations(self.max_iterations)
    }
}

pub(super) fn vectors<S: NumericScalar>(count: usize, n: usize) -> Result<Vec<Vector<S>>> {
    checked_mul(count, n, scioxide_core::quantity!("arnoldi_vector_storage"))?;
    let mut values = try_with_capacity(count, scioxide_core::quantity!("arnoldi_vectors"))?;
    for _ in 0..count {
        values.push(Vector::zeros(n)?);
    }
    Ok(values)
}

/// Project an image and carry the same operations into its solution direction.
pub(super) fn orthogonalize<S: NumericScalar>(
    image: &mut [S],
    direction: Option<&mut [S]>,
    images: &[Vector<S>],
    directions: &[Vector<S>],
    coefficients: &mut [S],
    kernels: crate::KernelContext<'_>,
) -> Result<()> {
    use crate::kernel::subspace::{self, Columns};
    subspace::orthogonalize(
        image,
        Columns::Vectors(images),
        coefficients,
        None,
        direction.map(|v| (v, Columns::Vectors(directions))),
        |_, _| {},
        kernels,
    )?;
    Ok(())
}

/// Directions `uᵢ` and images `cᵢ = A uᵢ` that LGMRES and GCROT reuse across solves.
///
/// GCROT stores orthonormal images. LGMRES stores the normalized corrections of
/// recent cycles, which may be linearly dependent. When the space is full, the
/// oldest pair is dropped. A capacity of zero stores nothing.
///
/// The operator epoch is a token you choose to identify `A`; it does not depend
/// on the right-hand side. The stored images are valid only for the operator
/// that produced them, so `solve_recycled_into` rejects a different epoch. When
/// `A` changes, call [`Self::clear`] or recompute the images with
/// [`Self::refresh`].
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Debug)]
pub struct KrylovRecycleSpace<S: NumericScalar> {
    pub(super) dimension: usize,
    pub(super) epoch: ProblemToken,
    pub(super) method: Option<Method>,
    pub(super) len: usize,
    pub(super) directions: Vec<Vector<S>>,
    pub(super) images: Vec<Vector<S>>,
    coefficients: Vec<S>,
    capacity: usize,
}
impl<S: NumericScalar> KrylovRecycleSpace<S> {
    /// Creates an empty space of dimension `dimension` for up to `capacity` pairs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `capacity · dimension` overflows `usize`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn new(dimension: usize, capacity: usize, operator_epoch: ProblemToken) -> Result<Self> {
        Ok(Self {
            dimension,
            capacity,
            epoch: operator_epoch,
            method: None,
            len: 0,
            directions: vectors(capacity, dimension)?,
            images: vectors(capacity, dimension)?,
            coefficients: try_filled(
                capacity,
                S::ZERO,
                scioxide_core::quantity!("recycle_projection"),
            )?,
        })
    }
    /// Returns the number of stored pairs; LGMRES pairs may be linearly dependent.
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Returns whether the space holds no pairs.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Returns the maximum number of pairs.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Returns the epoch that identifies the operator of the stored images.
    pub const fn operator_epoch(&self) -> ProblemToken {
        self.epoch
    }
    /// Removes all pairs and sets the epoch to `operator_epoch`.
    pub fn clear(&mut self, operator_epoch: ProblemToken) {
        self.len = 0;
        self.method = None;
        self.epoch = operator_epoch;
    }
    /// Returns the direction `uᵢ` and image `cᵢ` at `index`, or `None` if `index ≥ len()`.
    pub fn pair(&self, index: usize) -> Option<(&[S], &[S])> {
        (index < self.len).then(|| {
            (
                self.directions[index].as_slice(),
                self.images[index].as_slice(),
            )
        })
    }
    pub(super) fn copy_from(&mut self, other: &Self) -> Result<()> {
        if self.dimension != other.dimension || self.capacity() < other.len {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("recycle"),
                "incompatible recycle-space storage",
            ));
        }
        for i in 0..other.len {
            finite::copy(
                self.directions[i].as_mut_slice(),
                other.directions[i].as_slice(),
            )?;
            finite::copy(self.images[i].as_mut_slice(), other.images[i].as_slice())?;
        }
        self.len = other.len;
        self.epoch = other.epoch;
        self.method = other.method;
        Ok(())
    }
    pub(super) fn insert(
        &mut self,
        direction: &mut [S],
        image: &mut [S],
        rank: RankThreshold<S::Real>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<bool> {
        let execution = kernels.execution();
        if self.capacity() == 0 {
            return Ok(false);
        }
        let scale = finite::checked_norm(image, execution)?;
        // LGMRES retains the actual recent cycle corrections, not the GCROT
        // projections of those corrections. Dependencies are removed by the
        // cycle's augmented image QR when these candidates are used.
        if self.method != Some(Method::Lgmres) {
            orthogonalize(
                image,
                Some(direction),
                &self.images[..self.len],
                &self.directions[..self.len],
                &mut self.coefficients,
                kernels,
            )?;
        }
        let length = finite::checked_norm(image, execution)?;
        if length == S::Real::ZERO || rank.is_satisfied(length / scale, S::Real::ONE)? {
            return Ok(false);
        }
        finite::normalize(image, length, kernels.dispatch())?;
        finite::normalize(direction, length, kernels.dispatch())?;
        // Full: evict the oldest pair (FIFO) by rotating the vector handles,
        // which reuses their storage without copying vector contents.
        if self.len == self.capacity() {
            self.images[..self.capacity].rotate_left(1);
            self.directions[..self.capacity].rotate_left(1);
            self.len -= 1;
        }
        self.images[self.len].as_mut_slice().copy_from_slice(image);
        self.directions[self.len]
            .as_mut_slice()
            .copy_from_slice(direction);
        self.len += 1;
        Ok(true)
    }
}

#[derive(Debug)]
pub(super) struct Storage<S: NumericScalar> {
    pub(super) n: usize,
    pub(super) rhs: usize,
    pub(super) x: Vec<S>,
    pub(super) verified: Vec<S>,
    best: super::support::BestIterate<S>,
    best_norms: Vec<S::Real>,
    /// Rounding floor of each column.
    floors: Vec<super::safety::ResidualFloor<S::Real>>,
    /// Columns whose next judged verification compares with the rounding floor.
    floor_due: Vec<bool>,
    residual: Vec<S>,
    trial_residual: Vec<S>,
    cycle_x: Vec<S>,
    cycle_residual: Vec<S>,
    /// Cycle correction `u = Σ cᵢ dᵢ` of each column, kept for the recycle space.
    correction: Vec<S>,
    /// Image `A u = Σ cᵢ qᵢ` of [`Self::correction`], accumulated from the
    /// orthonormal images rather than from a residual difference.
    correction_image: Vec<S>,
    basis: Vec<Vector<S>>,
    directions: Vec<Vector<S>>,
    images: Vec<Vector<S>>,
    coefficients: Vec<S>,
    direction: Vector<S>,
    image: Vector<S>,
    next: Vector<S>,
    pub(super) operator_scratch: Vec<S>,
    pub(super) preconditioner_scratch: Vec<S>,
    pub(super) recycle: KrylovRecycleSpace<S>,
    pub(super) residuals: Vec<S::Real>,
    trial_norms: Vec<S::Real>,
    pub(super) references: Vec<S::Real>,
    pub(super) converged: Vec<bool>,
    pub(super) history: Vec<S::Real>,
    pub(super) verified_this_call: bool,
    pub(super) restarts: usize,
    pub(super) deflated: usize,
    pub(super) active_rank: usize,
    estimate_residual: bool,
    pub(super) generated: usize,
    pub(super) iterations: usize,
}

impl<S: NumericScalar> Storage<S> {
    /// Exact payload of [`Self::new`], including vector headers of the bases
    /// and the recycle space.
    pub(super) fn bytes(shape: &super::workspace::Shape, history: usize) -> Result<usize> {
        use super::workspace::{WORKSPACE_BYTES, bytes_of, payload, times, total};
        let (n, rhs, cycle) = (shape.cols, shape.columns, shape.cycle);
        let count = times(n, rhs)?;
        let accumulators = if shape.retained > 0 {
            times(2, count)?
        } else {
            0
        };
        let steps = checked_add(
            cycle,
            1,
            scioxide_core::quantity!("gmres_hessenberg_leading_dimension"),
        )?;
        let basis_count = times(steps, rhs)?;
        let image_count = checked_add(times(cycle, rhs)?, shape.retained, WORKSPACE_BYTES)?;
        let bases = total([
            basis_count,
            image_count,
            image_count,
            shape.retained,
            shape.retained,
        ])?;
        total([
            payload::<S>(
                total([
                    times(7, count)?,
                    accumulators,
                    times(bases, n)?,
                    image_count.max(basis_count),
                    times(3, n)?,
                    shape.operator_scratch,
                    shape.preconditioner_scratch,
                    shape.retained,
                ])?,
                total([times(4, rhs)?, history])?,
            )?,
            bytes_of::<Vector<S>>(bases)?,
            bytes_of::<super::safety::ResidualFloor<S::Real>>(rhs)?,
            bytes_of::<bool>(times(2, rhs)?)?,
        ])
    }

    /// Keep GMRES basis buffers when only the active operator dimension changes.
    pub(super) fn resize_unrecycled(
        &mut self,
        shape: &super::workspace::Shape,
        history: usize,
    ) -> Result<()> {
        fn resize<V: Clone>(v: &mut Vec<V>, n: usize, value: V) -> Result<()> {
            let additional = n.saturating_sub(v.len());
            scioxide_core::backend::try_reserve(
                v,
                additional,
                scioxide_core::quantity!("arnoldi_storage"),
            )?;
            v.resize(n, value);
            Ok(())
        }
        let n = shape.cols;
        let count = n * shape.columns;
        for v in [
            &mut self.x,
            &mut self.verified,
            &mut self.residual,
            &mut self.trial_residual,
            &mut self.cycle_x,
            &mut self.cycle_residual,
        ] {
            resize(v, count, S::ZERO)?;
        }
        self.best.solution.try_resize(count, S::ZERO)?;
        for (vectors, required) in [
            (&mut self.basis, (shape.cycle + 1) * shape.columns),
            (&mut self.directions, shape.cycle * shape.columns),
            (&mut self.images, shape.cycle * shape.columns),
        ] {
            let additional = required.saturating_sub(vectors.len());
            scioxide_core::backend::try_reserve(
                vectors,
                additional,
                scioxide_core::quantity!("arnoldi_basis"),
            )?;
            while vectors.len() < required {
                vectors.push(Vector::zeros(n)?);
            }
            for v in vectors {
                v.try_resize(n, S::ZERO)?;
            }
        }
        for v in [&mut self.direction, &mut self.image, &mut self.next] {
            v.try_resize(n, S::ZERO)?;
        }
        resize(
            &mut self.coefficients,
            self.basis.len().max(self.images.len()),
            S::ZERO,
        )?;
        resize(&mut self.operator_scratch, shape.operator_scratch, S::ZERO)?;
        resize(
            &mut self.preconditioner_scratch,
            shape.preconditioner_scratch,
            S::ZERO,
        )?;
        let additional = history.saturating_sub(self.history.len());
        scioxide_core::backend::try_reserve(
            &mut self.history,
            additional,
            scioxide_core::quantity!("arnoldi_history"),
        )?;
        self.recycle.dimension = n;
        self.n = n;
        Ok(())
    }

    pub(super) fn new(shape: &super::workspace::Shape, history: usize) -> Result<Self> {
        let (n, rhs, cycle) = (shape.cols, shape.columns, shape.cycle);
        let steps = checked_add(
            cycle,
            1,
            scioxide_core::quantity!("gmres_hessenberg_leading_dimension"),
        )?;
        let count = checked_mul(n, rhs, scioxide_core::quantity!("arnoldi_block_storage"))?;
        let basis_count = checked_mul(steps, rhs, scioxide_core::quantity!("arnoldi_block_basis"))?;
        let image_count = checked_add(
            checked_mul(cycle, rhs, scioxide_core::quantity!("arnoldi_image_basis"))?,
            shape.retained,
            scioxide_core::quantity!("arnoldi_augmented_basis"),
        )?;
        let block = || {
            try_filled(
                count,
                S::ZERO,
                scioxide_core::quantity!("arnoldi_block_storage"),
            )
        };
        Ok(Self {
            n,
            rhs,
            x: block()?,
            verified: block()?,
            best: super::support::BestIterate::new(count)?,
            best_norms: try_filled(
                rhs,
                S::Real::ZERO,
                scioxide_core::quantity!("arnoldi_best_norms"),
            )?,
            floors: try_filled(
                rhs,
                super::safety::ResidualFloor::new(S::Real::ZERO),
                scioxide_core::quantity!("arnoldi_floors"),
            )?,
            floor_due: try_filled(rhs, false, scioxide_core::quantity!("arnoldi_floor_due"))?,
            residual: block()?,
            trial_residual: block()?,
            cycle_x: block()?,
            cycle_residual: block()?,
            correction: try_filled(
                if shape.retained > 0 { count } else { 0 },
                S::ZERO,
                scioxide_core::quantity!("arnoldi_cycle_correction"),
            )?,
            correction_image: try_filled(
                if shape.retained > 0 { count } else { 0 },
                S::ZERO,
                scioxide_core::quantity!("arnoldi_cycle_correction_image"),
            )?,
            basis: vectors(basis_count, n)?,
            directions: vectors(image_count, n)?,
            images: vectors(image_count, n)?,
            coefficients: try_filled(
                image_count.max(basis_count),
                S::ZERO,
                scioxide_core::quantity!("arnoldi_projection"),
            )?,
            direction: Vector::zeros(n)?,
            image: Vector::zeros(n)?,
            next: Vector::zeros(n)?,
            operator_scratch: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("arnoldi_operator_scratch"),
            )?,
            preconditioner_scratch: try_filled(
                shape.preconditioner_scratch,
                S::ZERO,
                scioxide_core::quantity!("arnoldi_preconditioner_scratch"),
            )?,
            recycle: KrylovRecycleSpace::new(n, shape.retained, ProblemToken::new(0))?,
            residuals: try_filled(
                rhs,
                S::Real::ZERO,
                scioxide_core::quantity!("arnoldi_residual_norms"),
            )?,
            trial_norms: try_filled(
                rhs,
                S::Real::ZERO,
                scioxide_core::quantity!("arnoldi_trial_norms"),
            )?,
            references: try_filled(
                rhs,
                S::Real::ONE,
                scioxide_core::quantity!("arnoldi_reference_norms"),
            )?,
            converged: try_filled(rhs, false, scioxide_core::quantity!("arnoldi_convergence"))?,
            history: try_with_capacity(history, scioxide_core::quantity!("arnoldi_history"))?,
            verified_this_call: false,
            restarts: 0,
            deflated: 0,
            active_rank: 0,
            estimate_residual: false,
            generated: 0,
            iterations: 0,
        })
    }

    /// Residual history when the reserving solver records it.
    pub(super) fn recorded_history(&self) -> Option<&[S::Real]> {
        (self.history.capacity() > 0).then_some(self.history.as_slice())
    }

    /// Per-column true residuals of the accepted iterate.
    pub(super) fn columns(&self) -> super::workspace::KrylovColumns<'_, S::Real> {
        super::workspace::KrylovColumns {
            residuals: &self.residuals,
            references: &self.references,
            converged: &self.converged,
        }
    }

    /// Recomputes the true residual of every column and commits the block.
    ///
    /// `judge` is set at the end of a cycle and when the projected residual met
    /// the tolerance. A judged column that has not converged stagnates when the
    /// cycle left its iterate unchanged, or when its residual does not improve
    /// on the best one and the best lies within the rounding floor. Returns
    /// `None` when the evaluation limit stops a product, and whether every
    /// column has converged or stagnated otherwise.
    fn verify_block(
        &mut self,
        options: &Options<S::Real>,
        a: &dyn LinearOperator<S>,
        b: &[S],
        judge: bool,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<Option<bool>> {
        for col in 0..self.rhs {
            let range = col * self.n..(col + 1) * self.n;
            if super::safety::apply_operator_scoped(
                a,
                &self.x[range.clone()],
                finite::maximum(&self.x[range.clone()], kernels.dispatch())?,
                self.image.as_mut_slice(),
                &mut self.operator_scratch,
                context,
            )?
            .is_exhausted()
            {
                return Ok(None);
            }
            for ((out, &b), &ax) in self.trial_residual[range.clone()]
                .iter_mut()
                .zip(&b[range.clone()])
                .zip(self.image.as_slice())
            {
                *out = crate::checked::finite_candidate(b - ax)?;
            }
            self.trial_norms[col] =
                finite::checked_norm(&self.trial_residual[range], context.execution())?;
            self.floor_due[col] =
                judge && super::safety::floor_test_due(self.trial_norms[col], self.best_norms[col]);
        }
        let maximum = self
            .trial_norms
            .iter()
            .copied()
            .fold(S::Real::ZERO, RealScalar::max);
        if !self.verified_this_call {
            self.best.reset(&self.x, maximum);
            self.best_norms.copy_from_slice(&self.trial_norms);
        } else if self.best.observe(&self.x, maximum, self.iterations) {
            self.best_norms.copy_from_slice(&self.trial_norms);
        }
        // Commit only after all RHS residuals have been verified.
        self.verified.copy_from_slice(&self.x);
        self.residual.copy_from_slice(&self.trial_residual);
        self.residuals.copy_from_slice(&self.trial_norms);
        for col in 0..self.rhs {
            self.converged[col] = options
                .tolerance
                .is_satisfied(self.residuals[col], self.references[col])?;
        }
        context.commit_accepted(super::checkpoint_trust(
            self.converged.iter().all(|&value| value),
        ))?;
        self.verified_this_call = true;
        if !judge {
            return Ok(Some(false));
        }
        for col in 0..self.rhs {
            let range = col * self.n..(col + 1) * self.n;
            if self.converged[col] || self.x[range.clone()] == self.cycle_x[range.clone()] {
                continue;
            }
            if !self.floor_due[col] {
                return Ok(Some(false));
            }
            // The next basis vector and the operator image are free between steps.
            match self.floors[col].reached(
                a,
                &self.best.solution.as_slice()[range],
                self.best_norms[col].min(self.residuals[col]),
                super::safety::FloorProbe {
                    input: self.next.as_mut_slice(),
                    image: self.image.as_mut_slice(),
                    workspace: &mut self.operator_scratch,
                },
                kernels,
                context,
            )? {
                None => return Ok(None),
                Some(true) => {}
                Some(false) => return Ok(Some(false)),
            }
        }
        Ok(Some(true))
    }

    fn initial_basis(
        &mut self,
        rank: RankThreshold<S::Real>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<usize> {
        let execution = kernels.execution();
        // Block starting basis: orthonormalize the residuals of the not yet
        // converged right-hand sides. A residual that is (numerically) in the
        // span of the others is deflated, so the block Krylov space never
        // carries dependent columns.
        let mut count = 0;
        for col in 0..self.rhs {
            if self.converged[col] {
                continue;
            }
            self.next
                .as_mut_slice()
                .copy_from_slice(&self.residual[col * self.n..(col + 1) * self.n]);
            let scale = finite::checked_norm(self.next.as_slice(), execution)?;
            orthogonalize(
                self.next.as_mut_slice(),
                None,
                &self.basis[..count],
                &[],
                &mut self.coefficients,
                kernels,
            )?;
            let length = finite::checked_norm(self.next.as_slice(), execution)?;
            if length == S::Real::ZERO || rank.is_satisfied(length / scale, S::Real::ONE)? {
                self.deflated += 1;
                continue;
            }
            finite::normalize(self.next.as_mut_slice(), length, kernels.dispatch())?;
            finite::copy(self.basis[count].as_mut_slice(), self.next.as_slice())?;
            count += 1;
        }
        self.active_rank = count;
        Ok(count)
    }

    /// QR the current image/direction pair and update all RHS in the shared search space.
    ///
    fn add_image(
        &mut self,
        count: &mut usize,
        rank: RankThreshold<S::Real>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<bool> {
        let execution = kernels.execution();
        // Image QR step: orthonormalize the new image `A d` against the
        // previous images (mirroring the operations on d), so that
        // `A P = Q` with orthonormal Q. The least-squares correction over
        // the cycle is then simply `P Q^H r_cycle`, accumulated one column
        // at a time: x += d (q^H r_cycle). No Hessenberg matrix or triangular
        // solve is needed, and a variable preconditioner is harmless because
        // each stored direction is exactly the vector the operator was
        // applied to.
        let scale = finite::checked_norm(self.image.as_slice(), execution)?;
        orthogonalize(
            self.image.as_mut_slice(),
            Some(self.direction.as_mut_slice()),
            &self.images[..*count],
            &self.directions[..*count],
            &mut self.coefficients,
            kernels,
        )?;
        let length = finite::checked_norm(self.image.as_slice(), execution)?;
        if length == S::Real::ZERO || rank.is_satisfied(length / scale, S::Real::ONE)? {
            return Ok(false);
        }
        finite::normalize(self.image.as_mut_slice(), length, kernels.dispatch())?;
        finite::normalize(self.direction.as_mut_slice(), length, kernels.dispatch())?;
        finite::copy(self.images[*count].as_mut_slice(), self.image.as_slice())?;
        finite::copy(
            self.directions[*count].as_mut_slice(),
            self.direction.as_slice(),
        )?;
        for col in 0..self.rhs {
            let range = col * self.n..(col + 1) * self.n;
            let coefficient = crate::reduction::standard_dot(
                crate::FiniteVectorView::from_workspace(self.image.view()),
                crate::FiniteVectorView::from_workspace(VectorView::from_slice(
                    &self.cycle_residual[range.clone()],
                )),
                true,
                kernels.dispatch(),
            )?;
            if self.estimate_residual {
                finite::axpy(
                    &mut self.trial_residual[range.clone()],
                    -coefficient,
                    self.image.as_slice(),
                    kernels.dispatch(),
                )?;
                self.trial_norms[col] =
                    finite::checked_norm(&self.trial_residual[range.clone()], execution)?;
            }
            finite::axpy(
                &mut self.x[range.clone()],
                coefficient,
                self.direction.as_slice(),
                kernels.dispatch(),
            )?;
            if !self.correction.is_empty() {
                finite::axpy(
                    &mut self.correction[range.clone()],
                    coefficient,
                    self.direction.as_slice(),
                    kernels.dispatch(),
                )?;
                finite::axpy(
                    &mut self.correction_image[range],
                    coefficient,
                    self.image.as_slice(),
                    kernels.dispatch(),
                )?;
            }
        }
        *count += 1;
        Ok(true)
    }

    fn augment(
        &mut self,
        count: &mut usize,
        rank: RankThreshold<S::Real>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<()> {
        for i in 0..self.recycle.len {
            finite::copy(
                self.direction.as_mut_slice(),
                self.recycle.directions[i].as_slice(),
            )?;
            finite::copy(self.image.as_mut_slice(), self.recycle.images[i].as_slice())?;
            self.add_image(count, rank, kernels)?;
        }
        Ok(())
    }

    fn retain_correction(
        &mut self,
        options: &Options<S::Real>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<()> {
        // LGMRES/GCROT memory: the cycle's correction `u = Σ cᵢ dᵢ` and its
        // image `A u = Σ cᵢ qᵢ`, both accumulated with the solution updates, are
        // offered to the recycle space. Unlike `r_start − r_end`, the image
        // keeps working accuracy when the cycle changes the residual only at
        // the rounding level.
        if self.recycle.capacity() == 0 {
            return Ok(());
        }
        for col in 0..self.rhs {
            let range = col * self.n..(col + 1) * self.n;
            finite::copy(
                self.direction.as_mut_slice(),
                &self.correction[range.clone()],
            )?;
            finite::copy(self.image.as_mut_slice(), &self.correction_image[range])?;
            self.recycle.insert(
                self.direction.as_mut_slice(),
                self.image.as_mut_slice(),
                options.rank,
                kernels,
            )?;
        }
        Ok(())
    }

    fn history_sample(&mut self, options: &Options<S::Real>) {
        if options.history {
            self.history.push(
                self.residuals
                    .iter()
                    .copied()
                    .fold(S::Real::ZERO, RealScalar::max),
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub(super) fn run(
        &mut self,
        options: &Options<S::Real>,
        a: &dyn LinearOperator<S>,
        p: Preconditioning<'_, S>,
        b: &[S],
        initial: &[S],
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<LinearSolveTermination> {
        self.estimate_residual = self.rhs == 1;
        self.verified_this_call = false;
        finite::checked_norm(initial, context.execution())?;
        self.x.copy_from_slice(initial);
        self.history.clear();
        self.restarts = 0;
        self.deflated = 0;
        self.active_rank = 0;
        self.generated = 0;
        self.iterations = 0;
        for col in 0..self.rhs {
            let reference =
                finite::checked_norm(&b[col * self.n..(col + 1) * self.n], context.execution())?;
            self.floors[col] = super::safety::ResidualFloor::new(reference);
            self.references[col] = super::residual_reference(reference);
        }
        if self
            .verify_block(options, a, b, false, kernels, context)?
            .is_none()
        {
            return Err(super::evaluation_limit_error(context));
        }
        self.history_sample(options);
        let termination = self.iterate(options, a, p, b, kernels, context);
        if termination.as_ref().is_ok_and(|t| t.is_success()) {
            self.x.copy_from_slice(&self.verified);
        } else {
            self.x.copy_from_slice(self.best.solution.as_slice());
            self.residuals.copy_from_slice(&self.best_norms);
            for col in 0..self.rhs {
                self.converged[col] = options
                    .tolerance
                    .is_satisfied(self.residuals[col], self.references[col])?;
            }
        }
        if let Some(last) = self.history.last_mut() {
            *last = self
                .residuals
                .iter()
                .copied()
                .fold(S::Real::ZERO, RealScalar::max);
        }
        termination
    }

    fn iterate(
        &mut self,
        options: &Options<S::Real>,
        a: &dyn LinearOperator<S>,
        p: Preconditioning<'_, S>,
        b: &[S],
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<LinearSolveTermination> {
        let limit = super::support::iteration_limit(context);
        loop {
            if self.converged.iter().all(|&value| value) {
                return Ok(LinearSolveTermination::Converged);
            }
            if let Some(reason) = context.stop_reason()? {
                return Ok(LinearSolveTermination::ExecutionStopped(reason));
            }
            if self.iterations >= limit {
                return Ok(LinearSolveTermination::IterationLimit);
            }
            // One restart cycle. GCROT first minimizes over the recycled
            // pairs (and verifies), LGMRES appends them at the end of the
            // cycle; both then build Arnoldi `basis` vectors from the current
            // residuals, precondition each into a `direction`, and feed
            // `A * direction` through image QR. Scalar GMRES maintains its
            // projected residual and verifies convergence and cycle boundaries.
            if self.iterations > 0 {
                self.restarts += 1;
            }
            self.cycle_x.copy_from_slice(&self.verified);
            self.cycle_residual.copy_from_slice(&self.residual);
            self.trial_residual.copy_from_slice(&self.residual);
            self.correction.fill(S::ZERO);
            self.correction_image.fill(S::ZERO);
            let mut images = 0;
            if options.method == Method::Gcrot {
                self.augment(&mut images, options.rank, kernels)?;
                if images > 0 {
                    // Projection onto the recycled images is part of the
                    // cycle; stagnation is judged when the cycle ends.
                    if self
                        .verify_block(options, a, b, false, kernels, context)?
                        .is_none()
                    {
                        return Ok(LinearSolveTermination::EvaluationLimit);
                    }
                    if let Some(last) = self.history.last_mut() {
                        *last = self
                            .residuals
                            .iter()
                            .copied()
                            .fold(S::Real::ZERO, RealScalar::max);
                    }
                    if self.converged.iter().all(|&value| value) {
                        return Ok(LinearSolveTermination::Converged);
                    }
                }
            }
            let retained_images = images;
            let mut end = self.initial_basis(options.rank, kernels)?;
            let mut start = 0;
            if end == 0 {
                return Ok(LinearSolveTermination::Breakdown(
                    LinearSolveBreakdown::ArnoldiInvariant,
                ));
            }
            for step in 0..options.restart.get().min(limit) {
                if let Some(reason) = context.stop_reason()? {
                    return Ok(LinearSolveTermination::ExecutionStopped(reason));
                }
                if self.iterations >= limit || !context.record_iteration()?.is_granted() {
                    return Ok(LinearSolveTermination::IterationLimit);
                }
                self.iterations += 1;
                let mut next_end = end;
                for column in start..end {
                    if !super::support::precondition(
                        p,
                        false,
                        &self.basis[column],
                        &mut self.direction,
                        &mut self.preconditioner_scratch,
                        kernels,
                        context,
                    )? {
                        return Ok(LinearSolveTermination::EvaluationLimit);
                    }

                    if super::safety::apply_operator_scoped(
                        a,
                        self.direction.as_slice(),
                        finite::maximum(self.direction.as_slice(), kernels.dispatch())?,
                        self.image.as_mut_slice(),
                        &mut self.operator_scratch,
                        context,
                    )?
                    .is_exhausted()
                    {
                        return Ok(LinearSolveTermination::EvaluationLimit);
                    }
                    self.generated += 1;
                    // The next Arnoldi basis vector is the new image made
                    // orthogonal to the retained images and the current basis;
                    // if nothing numerically new remains, that column deflates
                    // (the Krylov space is invariant for it).
                    finite::copy(self.next.as_mut_slice(), self.image.as_slice())?;
                    let scale = finite::checked_norm(self.next.as_slice(), context.execution())?;
                    orthogonalize(
                        self.next.as_mut_slice(),
                        None,
                        &self.images[..retained_images],
                        &[],
                        &mut self.coefficients,
                        kernels,
                    )?;
                    orthogonalize(
                        self.next.as_mut_slice(),
                        None,
                        &self.basis[..next_end],
                        &[],
                        &mut self.coefficients,
                        kernels,
                    )?;
                    let length = finite::checked_norm(self.next.as_slice(), context.execution())?;
                    if length != S::Real::ZERO
                        && !options.rank.is_satisfied(length / scale, S::Real::ONE)?
                    {
                        finite::normalize(self.next.as_mut_slice(), length, kernels.dispatch())?;
                        finite::copy(self.basis[next_end].as_mut_slice(), self.next.as_slice())?;
                        next_end += 1;
                    } else {
                        self.deflated += 1;
                    }
                    self.add_image(&mut images, options.rank, kernels)?;
                }
                let exhausted = next_end == end;
                let cycle_end =
                    exhausted || step + 1 == options.restart.get() || self.iterations >= limit;
                if options.method == Method::Lgmres && cycle_end {
                    self.augment(&mut images, options.rank, kernels)?;
                }
                let candidate = self.estimate_residual
                    && options
                        .tolerance
                        .is_satisfied(self.trial_norms[0], self.references[0])?;
                let mut stagnant = false;
                if !self.estimate_residual
                    || cycle_end
                    || candidate
                    || super::safety::should_recompute_true_residual(self.iterations)
                {
                    // GMRES may leave the iterate unchanged for several steps
                    // of a cycle, so only a cycle end or a refuted projected
                    // convergence judges stagnation.
                    let judge = cycle_end || candidate;
                    let Some(judged) = self.verify_block(options, a, b, judge, kernels, context)?
                    else {
                        return Ok(LinearSolveTermination::EvaluationLimit);
                    };
                    stagnant = judged;
                    self.history_sample(options);
                } else if options.history {
                    self.history.push(self.trial_norms[0]);
                }
                if self.converged.iter().all(|&value| value) {
                    self.retain_correction(options, kernels)?;
                    return Ok(LinearSolveTermination::Converged);
                }
                if exhausted && images == retained_images {
                    return Ok(LinearSolveTermination::Breakdown(
                        LinearSolveBreakdown::ArnoldiInvariant,
                    ));
                }
                if stagnant {
                    return Ok(LinearSolveTermination::Stagnation);
                }
                if cycle_end || candidate {
                    self.retain_correction(options, kernels)?;
                    break;
                }
                start = end;
                end = next_end;
            }
        }
    }
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes for KrylovRecycleSpace<S> {
    fn storage_bytes(&self) -> Result<usize> {
        use crate::workspace::accounting::sum;
        let vectors = self
            .directions
            .iter()
            .chain(&self.images)
            .try_fold(0usize, |total, vector| {
                sum([total, vector.storage_bytes()?])
            })?;
        sum([
            vectors,
            self.directions.storage_bytes()?,
            self.images.storage_bytes()?,
            self.coefficients.storage_bytes()?,
        ])
    }
}
impl<S: NumericScalar> KrylovRecycleSpace<S> {
    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        n: usize,
        count: usize,
    ) -> Result<()> {
        for vectors in [&self.directions, &self.images] {
            plan.buffer(vectors, count)?;
            for index in 0..count {
                plan.capacity::<S>(vectors.get(index).map_or(0, Vector::capacity), n)?;
            }
        }
        plan.buffer(&self.coefficients, count)
    }
    pub(crate) fn configure_refresh(&mut self, n: usize, count: usize) -> Result<()> {
        for vectors in [&mut self.directions, &mut self.images] {
            scioxide_core::backend::try_reserve(
                vectors,
                count.saturating_sub(vectors.len()),
                scioxide_core::quantity!("recycle_vectors"),
            )?;
            while vectors.len() < count {
                vectors.push(Vector::empty());
            }
            for vector in &mut vectors[..count] {
                vector.try_resize(n, S::ZERO)?;
            }
        }
        crate::workspace::accounting::resize(&mut self.coefficients, count, S::ZERO)?;
        self.dimension = n;
        self.capacity = count;
        self.len = 0;
        Ok(())
    }
    pub(crate) const fn empty() -> Self {
        Self {
            dimension: 0,
            capacity: 0,
            epoch: ProblemToken::new(0),
            method: None,
            len: 0,
            directions: Vec::new(),
            images: Vec::new(),
            coefficients: Vec::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::kernel::finite;
    use scioxide_core::Complex;

    #[test]
    fn image_updates_preserve_representable_product_cancellation() {
        let mut real = [-f64::MAX];
        finite::axpy(
            &mut real,
            2.,
            &[f64::MAX],
            crate::KernelContext::sequential().dispatch(),
        )
        .unwrap();
        assert_eq!(real, [f64::MAX]);
        let mut complex = [Complex::new(0., -f64::MAX)];
        finite::axpy(
            &mut complex,
            Complex::new(2., 2.),
            &[Complex::new(f64::MAX / 2., f64::MAX / 2.)],
            crate::KernelContext::sequential().dispatch(),
        )
        .unwrap();
        assert_eq!(complex, [Complex::new(0., f64::MAX)]);
    }
}
