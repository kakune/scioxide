//! Workspace for fixed-point solves.
use super::*;

pub(crate) struct AndersonStorage<T: NumericScalar> {
    pub(crate) matrix: Matrix<T>,
    pub(crate) rhs: Vector<T>,
    pub(crate) coefficients: Vector<T>,
    pub(crate) factor: HouseholderDecomposition<T>,
    pub(crate) scratch: LinalgWorkspace<'static, T>,
}

/// Stores the iterates, history and scratch of a [`FixedPointSolver`].
///
/// A workspace prepared for one solver and problem shape serves any number of
/// solves; each solve starts with an empty history. Its size is
/// `O(n m)` for `n` unknowns and `m` history pairs, plus the scratch that the
/// residual operator requests. See
/// [Reusing workspaces](crate#reusing-workspaces).
pub struct FixedPointWorkspace<T: NumericScalar> {
    pub(crate) method: FixedPointMethod,
    pub(crate) dimension: usize,
    pub(crate) capacity: usize,
    pub(crate) x: Vector<T>,
    pub(crate) f: Vector<T>,
    pub(crate) trial: Vector<T>,
    pub(crate) trial_f: Vector<T>,
    pub(crate) step: Vector<T>,
    pub(crate) difference: Vector<T>,
    pub(crate) work: Vector<T>,
    pub(crate) row: Vector<T>,
    pub(crate) diagonal: Vector<T>,
    pub(crate) first: Vec<T>,
    pub(crate) second: Vec<T>,
    pub(crate) scratch: Vec<T>,
    pub(crate) used: usize,
    pub(crate) anderson: Option<AndersonStorage<T>>,
    pub(crate) diagnostics: FixedPointDiagnostics,
    pub(crate) shell: crate::attempt::Shell<FixedPointDiagnostics>,
    pub(crate) accepted: bool,
}

/// Describes the workspace that a [`FixedPointSolver`] needs for one problem.
///
/// Returned by [`FixedPointSolver::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct FixedPointWorkspaceRequirements<T: NumericScalar> {
    pub(crate) method: FixedPointMethod,
    pub(crate) dimension: usize,
    pub(crate) capacity: usize,
    pub(crate) scratch: usize,
    pub(crate) rank: RankPolicy<T::Real>,
    pub(crate) bytes: usize,
    pub(crate) limit: usize,
    pub(crate) policy: ExecutionPolicy,
    pub(crate) anderson: Option<scioxide_linalg::LinalgRequirements>,
}
impl<T: NumericScalar> FixedPointWorkspaceRequirements<T> {
    /// Returns the workspace size in bytes, including the Anderson QR storage.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn new(
        method: FixedPointMethod,
        dimension: usize,
        scratch: usize,
        options: &FixedPointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        use scioxide_core::backend::{checked_add, checked_mul};
        let q = scioxide_core::quantity!("fixed_point_workspace");
        let capacity = options.history.min(dimension);
        let history = if matches!(
            method,
            FixedPointMethod::Anderson
                | FixedPointMethod::BroydenFirst
                | FixedPointMethod::BroydenSecond
        ) {
            checked_mul(dimension, capacity, q)?
        } else {
            0
        };
        let scalars = checked_add(
            checked_add(
                checked_mul(
                    dimension,
                    if matches!(
                        method,
                        FixedPointMethod::DiagonalBroyden | FixedPointMethod::ExcitingMixing
                    ) {
                        9
                    } else {
                        8
                    },
                    q,
                )?,
                checked_mul(history, 2, q)?,
                q,
            )?,
            scratch,
            q,
        )?;
        let mut bytes = checked_mul(scalars, core::mem::size_of::<T>(), q)?;
        let mut anderson = None;
        if method == FixedPointMethod::Anderson && capacity > 0 {
            let rows = checked_add(dimension, capacity, q)?;
            let execution = scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(options.policy);
            let factor = scioxide_linalg::HouseholderWorkspaceRequest::factor(
                rows,
                capacity,
                HouseholderOrientation::Qr,
                &execution,
            );
            let solve = scioxide_linalg::HouseholderWorkspaceRequest::solve(
                rows,
                rows,
                HouseholderOrientation::Qr,
                &execution,
            );
            let scratch = HouseholderDecomposition::<T>::workspace_requirements(&factor)?.max(
                &HouseholderDecomposition::<T>::workspace_requirements(&solve)?,
            );
            anderson = Some(scratch);
            let matrix_and_vectors = checked_add(
                checked_mul(rows, capacity, q)?,
                checked_add(rows, capacity, q)?,
                q,
            )?;
            bytes = checked_add(
                bytes,
                checked_mul(matrix_and_vectors, core::mem::size_of::<T>(), q)?,
                q,
            )?;
            bytes = checked_add(
                bytes,
                HouseholderDecomposition::<T>::factor_storage_bytes(
                    rows,
                    capacity,
                    HouseholderOrientation::Qr,
                )?,
                q,
            )?;
            bytes = checked_add(bytes, scratch.required_bytes::<T>()?, q)?;
        }
        if bytes > options.byte_limit {
            return Err(SciOxideError::resource_limit(q, bytes, options.byte_limit));
        }
        Ok(Self {
            method,
            dimension,
            capacity,
            scratch,
            rank: options.rank,
            bytes,
            limit: options.byte_limit,
            policy: options.policy,
            anderson,
        })
    }
}
impl<T: NumericScalar> FixedPointWorkspace<T> {
    pub(crate) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        use crate::workspace::{buffer, sum};
        let anderson = self.anderson.as_ref().map_or(Ok(0), |a| {
            sum([
                a.matrix.retained_bytes(),
                a.rhs.retained_bytes(),
                a.coefficients.retained_bytes(),
                a.factor.retained_bytes(),
                a.scratch.retained_bytes(),
            ])
        });
        sum([
            self.x.retained_bytes(),
            self.f.retained_bytes(),
            self.trial.retained_bytes(),
            self.trial_f.retained_bytes(),
            self.step.retained_bytes(),
            self.difference.retained_bytes(),
            self.work.retained_bytes(),
            self.row.retained_bytes(),
            self.diagonal.retained_bytes(),
            buffer(&self.first),
            buffer(&self.second),
            buffer(&self.scratch),
            anderson,
        ])
    }

    /// Creates an empty workspace without allocating. Call
    /// [`try_reserve`](Self::try_reserve) before solving.
    pub fn new() -> Self {
        Self {
            method: FixedPointMethod::Picard,
            dimension: 0,
            capacity: 0,
            x: Vector::empty(),
            f: Vector::empty(),
            trial: Vector::empty(),
            trial_f: Vector::empty(),
            step: Vector::empty(),
            difference: Vector::empty(),
            work: Vector::empty(),
            row: Vector::empty(),
            diagonal: Vector::empty(),
            first: Vec::new(),
            second: Vec::new(),
            scratch: Vec::new(),
            used: 0,
            anderson: None,
            diagnostics: FixedPointDiagnostics::default(),
            shell: crate::attempt::Shell::new(),
            accepted: false,
        }
    }
    /// Returns whether this workspace fits `r`.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the storage already held, or the storage `r` needs,
    /// exceeds the byte limit in `r`.
    pub fn check(
        &self,
        r: &FixedPointWorkspaceRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = self.retained_bytes()?.max(r.bytes);
        if bytes > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                r.limit,
            ));
        }
        let anderson = match &r.anderson {
            Some(req) => self
                .anderson
                .as_ref()
                .map(|a| a.scratch.check(req))
                .transpose()?
                .is_some_and(|c| c == scioxide_core::Capacity::Sufficient),
            None => true,
        };
        Ok(
            if self.method == r.method
                && self.dimension == r.dimension
                && self.capacity == r.capacity
                && self.scratch.len() >= r.scratch
                && anderson
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
    /// `AllocationFailure` if the storage cannot be allocated. On error, the
    /// workspace is unchanged.
    pub fn try_reserve(
        &mut self,
        r: &FixedPointWorkspaceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::NeedsGrowth {
            let mut prepared = Self::allocate(r)?;
            prepared.shell = self.shell.retained();
            *self = prepared;
        }
        Ok(())
    }
    pub(crate) fn allocate(
        requirements: &FixedPointWorkspaceRequirements<T>,
    ) -> Result<Self, SciOxideError> {
        let dimension = requirements.dimension;
        let capacity = requirements.capacity;
        let has_history = matches!(
            requirements.method,
            FixedPointMethod::Anderson
                | FixedPointMethod::BroydenFirst
                | FixedPointMethod::BroydenSecond
        );
        let length = if has_history {
            dimension.checked_mul(capacity).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("fixed_point_history"))
            })?
        } else {
            0
        };
        let anderson = if requirements.method == FixedPointMethod::Anderson && capacity > 0 {
            let rows = dimension.checked_add(capacity).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("fixed_point_qr"))
            })?;
            let matrix = Matrix::zeros(rows, capacity)?;
            let mut factor = HouseholderDecomposition::empty();
            let mut scratch = LinalgWorkspace::new();
            HouseholderDecomposition::factor_into(
                &matrix,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(requirements.policy),
                &mut factor,
                &mut scratch,
                scioxide_core::ExecutionContext::sequential(),
            )?;
            let _ = factor.analyze_rank(requirements.rank)?;
            scratch.try_reserve(
                &scioxide_linalg::HouseholderDecomposition::<T>::workspace_requirements(
                    &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                        rows,
                        rows,
                        scioxide_linalg::HouseholderOrientation::Qr,
                        &scioxide_linalg::LinalgExecutionOptions::new()
                            .with_execution_policy(requirements.policy),
                    ),
                )?,
            )?;
            Some(AndersonStorage {
                matrix,
                factor,
                scratch,
                rhs: Vector::zeros(rows)?,
                coefficients: Vector::zeros(capacity)?,
            })
        } else {
            None
        };
        Ok(FixedPointWorkspace {
            method: requirements.method,
            shell: crate::attempt::Shell::new(),
            accepted: false,
            dimension,
            capacity,
            x: Vector::zeros(dimension)?,
            f: Vector::zeros(dimension)?,
            trial: Vector::zeros(dimension)?,
            trial_f: Vector::zeros(dimension)?,
            step: Vector::zeros(dimension)?,
            difference: Vector::zeros(dimension)?,
            work: Vector::zeros(dimension)?,
            row: Vector::zeros(dimension)?,
            diagonal: Vector::zeros(
                if matches!(
                    requirements.method,
                    FixedPointMethod::DiagonalBroyden | FixedPointMethod::ExcitingMixing
                ) {
                    dimension
                } else {
                    0
                },
            )?,
            first: scioxide_core::backend::try_filled(
                length,
                T::ZERO,
                scioxide_core::quantity!("fixed_point_history"),
            )?,
            second: scioxide_core::backend::try_filled(
                length,
                T::ZERO,
                scioxide_core::quantity!("fixed_point_history"),
            )?,
            scratch: scioxide_core::backend::try_filled(
                requirements.scratch,
                T::ZERO,
                scioxide_core::quantity!("fixed_point_scratch"),
            )?,
            used: 0,
            anderson,
            diagnostics: FixedPointDiagnostics::default(),
        })
    }
    /// Returns the record of the most recent `solve_into` call, including a call
    /// rejected during input validation.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<FixedPointDiagnostics>> {
        self.shell.last_attempt()
    }
    /// Returns the last point accepted by the most recent call, or `None` if that
    /// call accepted no point.
    pub fn last_accepted(&self) -> Option<&[T]> {
        self.shell.accepted(self.x.as_slice())
    }
}
crate::attempt::attempted!(FixedPointWorkspace<T>, [T: NumericScalar], FixedPointDiagnostics);
impl<T: NumericScalar> Default for FixedPointWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
