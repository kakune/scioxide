//! Workspace for Newton–Krylov solves.
use super::*;

/// Forward-difference probe point and residual-operator scratch.
pub(crate) struct DifferenceStorage<T: NumericScalar> {
    pub(crate) point: Vec<T>,
    pub(crate) residual_scratch: Vec<T>,
}
/// Stores the iterates, Krylov basis and scratch of a [`NewtonKrylovSolver`].
///
/// The size is `O(n m)` for `n` unknowns and GMRES restart length `m`; no Jacobian
/// matrix is stored. A workspace prepared for one solver and problem serves any
/// number of solves. See
/// [Reusing workspaces](crate#reusing-workspaces).
pub struct NewtonKrylovWorkspace<T: NumericScalar> {
    pub(crate) scaled: Vector<T>,
    pub(crate) scaled_trial: Vector<T>,
    pub(crate) row_scales: Vec<T::Real>,
    pub(crate) variable_scales: Vec<T::Real>,
    pub(crate) n: usize,
    pub(crate) residual_scratch: Vec<T>,
    pub(crate) product_scratch: Vec<T>,
    pub(crate) x: Vector<T>,
    pub(crate) f: Vector<T>,
    pub(crate) trial: Vector<T>,
    pub(crate) trial_f: Vector<T>,
    pub(crate) rhs: Vector<T>,
    pub(crate) direction: Vector<T>,
    pub(crate) product: Vector<T>,
    pub(crate) step: Vector<T>,
    pub(crate) linear: Option<IterativeWorkspace<T>>,
    pub(crate) differences: Option<RefCell<DifferenceStorage<T>>>,
    pub(crate) history: Vec<NewtonKrylovObservation<T::Real>>,
    pub(crate) shell: crate::attempt::Shell<ResidualReport<T::Real>>,
    pub(crate) accepted: Option<ResidualReport<T::Real>>,
}

/// Describes the workspace that a [`NewtonKrylovSolver`] needs for one problem.
///
/// Returned by [`NewtonKrylovSolver::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct NewtonKrylovWorkspaceRequirements<T: NumericScalar> {
    pub(crate) n: usize,
    pub(crate) residual_scratch: usize,
    pub(crate) product_scratch: usize,
    pub(crate) numerical: bool,
    pub(crate) linear: Option<scioxide_linalg::IterativeRequirements<T>>,
    pub(crate) history: usize,
    pub(crate) bytes: usize,
    pub(crate) limit: usize,
}
impl<T: NumericScalar> NewtonKrylovWorkspaceRequirements<T> {
    /// Returns the workspace size in bytes, including the GMRES storage.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the number of unknowns.
    pub const fn dimension(&self) -> usize {
        self.n
    }
    pub(crate) fn new(
        problem: &NewtonKrylovProblem<'_, T>,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        problem.validate()?;
        let n = problem.dimension();
        let operator = Linearization {
            problem,
            point: &[],
            residual: &[],
            row_scales: &[],
            variable_scales: &[],
            differences: None,
        };
        let linear = if problem.linear_solver.is_none() {
            let mut request = IterativeWorkspaceRequest::operator(&operator)?;
            if let Some(preconditioner) = problem.preconditioner {
                request = request.with_preconditioner(Preconditioning::Inverse(preconditioner))?;
            }
            Some(
                options
                    .linear::<T>(options.forcing.initial())?
                    .workspace_requirements(&request)?,
            )
        } else {
            None
        };
        let residual_scratch = problem.residual.residual_workspace_len()?.max(
            problem
                .scaling
                .map_or(Ok(0), NonlinearScaling::scaling_workspace_len)?,
        );
        let product_scratch = operator.apply_workspace_len()?;
        Self::sized(
            n,
            residual_scratch,
            product_scratch,
            problem.jacobian.is_none(),
            linear,
            options,
        )
    }
}
impl<T: NumericScalar> Default for NewtonKrylovWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> NewtonKrylovWorkspace<T> {
    pub(crate) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        use crate::workspace::{buffer, sum};
        let differences = self.differences.as_ref().map_or(Ok(0), |d| {
            let d = d.borrow();
            sum([buffer(&d.point), buffer(&d.residual_scratch)])
        });
        sum([
            buffer(&self.residual_scratch),
            buffer(&self.product_scratch),
            self.scaled.retained_bytes(),
            self.scaled_trial.retained_bytes(),
            buffer(&self.row_scales),
            buffer(&self.variable_scales),
            self.x.retained_bytes(),
            self.f.retained_bytes(),
            self.trial.retained_bytes(),
            self.trial_f.retained_bytes(),
            self.rhs.retained_bytes(),
            self.direction.retained_bytes(),
            self.product.retained_bytes(),
            self.step.retained_bytes(),
            buffer(&self.history),
            differences,
            self.linear.as_ref().map_or(Ok(0), |w| w.retained_bytes()),
        ])
    }

    /// Creates an empty workspace without allocating. Call
    /// [`try_reserve`](Self::try_reserve) before solving.
    pub fn new() -> Self {
        Self {
            scaled: Vector::empty(),
            scaled_trial: Vector::empty(),
            row_scales: Vec::new(),
            variable_scales: Vec::new(),
            n: 0,
            residual_scratch: Vec::new(),
            product_scratch: Vec::new(),
            x: Vector::empty(),
            f: Vector::empty(),
            trial: Vector::empty(),
            trial_f: Vector::empty(),
            rhs: Vector::empty(),
            direction: Vector::empty(),
            product: Vector::empty(),
            step: Vector::empty(),
            linear: None,
            differences: None,
            history: Vec::new(),
            shell: crate::attempt::Shell::new(),
            accepted: None,
        }
    }
    /// Returns whether this workspace fits `r`. Tolerances do not affect the
    /// result.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the storage already held, or the storage `r` needs,
    /// exceeds the byte limit in `r`; `SizeOverflow` if a size does not fit in
    /// `usize`.
    pub fn check(
        &self,
        r: &NewtonKrylovWorkspaceRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = self.retained_bytes()?.max(r.bytes);
        if bytes > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                r.limit,
            ));
        }
        let difference = !r.numerical
            || self.differences.as_ref().is_some_and(|d| {
                let d = d.borrow();
                d.point.len() >= r.n && d.residual_scratch.len() >= r.residual_scratch
            });
        let linear = match &r.linear {
            Some(req) => self
                .linear
                .as_ref()
                .map(|w| w.check(req))
                .transpose()?
                .is_some_and(|c| c == scioxide_core::Capacity::Sufficient),
            None => true,
        };
        Ok(
            if self.n == r.n
                && self.residual_scratch.len() >= r.residual_scratch
                && self.product_scratch.len() >= r.product_scratch
                && self.history.capacity() >= r.history
                && difference
                && linear
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Grows this workspace to fit `r`, if needed. Growing clears the record
    /// returned by [`last_attempt`](Self::last_attempt).
    ///
    /// # Errors
    ///
    /// `ResourceLimit` as for [`check`](Self::check), and `SizeOverflow` or
    /// `AllocationFailure` if the storage cannot be allocated.
    pub fn try_reserve(
        &mut self,
        r: &NewtonKrylovWorkspaceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        use scioxide_core::backend::{checked_add, checked_mul};
        let q = scioxide_core::quantity!("newton_krylov_workspace");
        let vector_bytes = checked_mul(r.n, core::mem::size_of::<T>(), q)?;
        let mut projected = 0;
        for v in [
            &self.x,
            &self.f,
            &self.trial,
            &self.trial_f,
            &self.rhs,
            &self.direction,
            &self.product,
            &self.step,
            &self.scaled,
            &self.scaled_trial,
        ] {
            projected = checked_add(projected, v.retained_bytes()?.max(vector_bytes), q)?;
        }
        for (capacity, required, size) in [
            (
                self.row_scales.capacity(),
                r.n,
                core::mem::size_of::<T::Real>(),
            ),
            (
                self.variable_scales.capacity(),
                r.n,
                core::mem::size_of::<T::Real>(),
            ),
            (
                self.residual_scratch.capacity(),
                r.residual_scratch,
                core::mem::size_of::<T>(),
            ),
            (
                self.product_scratch.capacity(),
                r.product_scratch,
                core::mem::size_of::<T>(),
            ),
            (
                self.history.capacity(),
                r.history,
                core::mem::size_of::<NewtonKrylovObservation<T::Real>>(),
            ),
        ] {
            projected = checked_add(projected, checked_mul(capacity.max(required), size, q)?, q)?;
        }
        if r.numerical || self.differences.is_some() {
            let (point, scratch) = self.differences.as_ref().map_or((0, 0), |d| {
                let d = d.borrow();
                (d.point.capacity(), d.residual_scratch.capacity())
            });
            let count = checked_add(
                point.max(if r.numerical { r.n } else { 0 }),
                scratch.max(if r.numerical { r.residual_scratch } else { 0 }),
                q,
            )?;
            projected = checked_add(
                projected,
                checked_mul(count, core::mem::size_of::<T>(), q)?,
                q,
            )?;
        }
        let linear = self
            .linear
            .as_ref()
            .map_or(Ok(0), |w| w.retained_bytes())?
            .max(r.linear.as_ref().map_or(0, |req| req.required_bytes()));
        projected = checked_add(projected, linear, q)?;
        if projected > r.limit {
            return Err(SciOxideError::resource_limit(q, projected, r.limit));
        }
        self.n = 0;
        self.shell.clear();
        self.accepted = None;
        for v in [
            &mut self.x,
            &mut self.f,
            &mut self.trial,
            &mut self.trial_f,
            &mut self.rhs,
            &mut self.direction,
            &mut self.product,
            &mut self.step,
            &mut self.scaled,
            &mut self.scaled_trial,
        ] {
            v.try_resize(r.n, T::ZERO)?;
        }
        fn resize<V: Clone>(v: &mut Vec<V>, n: usize, value: V) -> Result<(), SciOxideError> {
            scioxide_core::backend::try_reserve(
                v,
                n.saturating_sub(v.len()),
                scioxide_core::quantity!("newton_storage"),
            )?;
            v.resize(n, value);
            Ok(())
        }
        resize(&mut self.row_scales, r.n, T::Real::ONE)?;
        resize(&mut self.variable_scales, r.n, T::Real::ONE)?;
        resize(&mut self.residual_scratch, r.residual_scratch, T::ZERO)?;
        resize(&mut self.product_scratch, r.product_scratch, T::ZERO)?;
        if r.numerical {
            let d = self.differences.get_or_insert_with(|| {
                RefCell::new(DifferenceStorage {
                    point: Vec::new(),
                    residual_scratch: Vec::new(),
                })
            });
            let d = d.get_mut();
            resize(&mut d.point, r.n, T::ZERO)?;
            resize(&mut d.residual_scratch, r.residual_scratch, T::ZERO)?;
        }
        if let Some(req) = &r.linear {
            self.linear
                .get_or_insert_with(IterativeWorkspace::new)
                .try_reserve(req)?;
        }
        let additional_history = r.history.saturating_sub(self.history.len());
        scioxide_core::backend::try_reserve(
            &mut self.history,
            additional_history,
            scioxide_core::quantity!("newton_history"),
        )?;
        self.n = r.n;
        Ok(())
    }
    pub(crate) fn allocate(
        r: &NewtonKrylovWorkspaceRequirements<T>,
    ) -> Result<Self, SciOxideError> {
        let mut linear = r.linear.as_ref().map(|_| IterativeWorkspace::new());
        if let (Some(workspace), Some(requirements)) = (&mut linear, &r.linear) {
            workspace.try_reserve(requirements)?;
        }
        let differences = if r.numerical {
            Some(RefCell::new(DifferenceStorage {
                point: scioxide_core::backend::try_filled(
                    r.n,
                    T::ZERO,
                    scioxide_core::quantity!("difference_point"),
                )?,
                residual_scratch: scioxide_core::backend::try_filled(
                    r.residual_scratch,
                    T::ZERO,
                    scioxide_core::quantity!("difference_scratch"),
                )?,
            }))
        } else {
            None
        };
        Ok(Self {
            scaled: Vector::zeros(r.n)?,
            scaled_trial: Vector::zeros(r.n)?,
            row_scales: scioxide_core::backend::try_filled(
                r.n,
                T::Real::ONE,
                scioxide_core::quantity!("residual_scales"),
            )?,
            variable_scales: scioxide_core::backend::try_filled(
                r.n,
                T::Real::ONE,
                scioxide_core::quantity!("variable_scales"),
            )?,
            n: r.n,
            residual_scratch: scioxide_core::backend::try_filled(
                r.residual_scratch,
                T::ZERO,
                scioxide_core::quantity!("residual_scratch"),
            )?,
            product_scratch: scioxide_core::backend::try_filled(
                r.product_scratch,
                T::ZERO,
                scioxide_core::quantity!("product_scratch"),
            )?,
            x: Vector::zeros(r.n)?,
            f: Vector::zeros(r.n)?,
            trial: Vector::zeros(r.n)?,
            trial_f: Vector::zeros(r.n)?,
            rhs: Vector::zeros(r.n)?,
            direction: Vector::zeros(r.n)?,
            product: Vector::zeros(r.n)?,
            step: Vector::zeros(r.n)?,
            linear,
            differences,
            history: scioxide_core::backend::try_with_capacity(
                r.history,
                scioxide_core::quantity!("history"),
            )?,
            shell: crate::attempt::Shell::new(),
            accepted: None,
        })
    }
    /// Returns the record of the most recent `solve_into` call, including a call
    /// rejected during input validation.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<ResidualReport<T::Real>>> {
        self.shell.last_attempt()
    }
    /// Returns the last point accepted by the most recent call, or `None` if that
    /// call accepted no point.
    pub fn last_accepted(&self) -> Option<&[T]> {
        self.shell.accepted(self.x.as_slice())
    }
}
crate::attempt::attempted!(
    NewtonKrylovWorkspace<T>,
    [T: NumericScalar],
    ResidualReport<T::Real>
);

impl<T: NumericScalar> NewtonKrylovWorkspaceRequirements<T> {
    pub(crate) fn augmented(
        n: usize,
        residual_scratch: usize,
        product_scratch: usize,
        injected: bool,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let operator = crate::residual::LinearShape::<T>::new(n, product_scratch);
        let linear = if injected {
            None
        } else {
            Some(
                options
                    .linear::<T>(options.forcing.initial())?
                    .workspace_requirements(&IterativeWorkspaceRequest::operator(&operator)?)?,
            )
        };
        Self::sized(n, residual_scratch, product_scratch, false, linear, options)
    }

    /// Size nonlinear vectors, callback scratch, optional difference storage,
    /// history and nested Krylov storage, and enforce the byte limit.
    fn sized(
        n: usize,
        residual_scratch: usize,
        product_scratch: usize,
        numerical: bool,
        linear: Option<scioxide_linalg::IterativeRequirements<T>>,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        use scioxide_core::backend::{checked_add, checked_mul};
        let history = if options.history {
            options.iterations.get()
        } else {
            0
        };
        let q = scioxide_core::quantity!("newton_krylov_workspace");
        let mut scalars = checked_add(
            checked_add(checked_mul(n, 12, q)?, residual_scratch, q)?,
            product_scratch,
            q,
        )?;
        if numerical {
            scalars = checked_add(scalars, checked_add(n, residual_scratch, q)?, q)?;
        }
        let bytes = checked_add(
            checked_add(
                checked_mul(scalars, core::mem::size_of::<T>(), q)?,
                checked_mul(
                    history,
                    core::mem::size_of::<NewtonKrylovObservation<T::Real>>(),
                    q,
                )?,
                q,
            )?,
            linear.as_ref().map_or(0, |r| r.required_bytes()),
            q,
        )?;
        if bytes > options.byte_limit {
            return Err(SciOxideError::resource_limit(q, bytes, options.byte_limit));
        }
        Ok(Self {
            n,
            residual_scratch,
            product_scratch,
            numerical,
            linear,
            history,
            bytes,
            limit: options.byte_limit,
        })
    }
}
