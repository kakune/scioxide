//! Reusable storage for penalty and augmented-Lagrangian solvers.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_reserve};
use scioxide_core::{ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::LinalgWorkspace;

use crate::constraints::{ConstraintProfile, ConstraintsWorkspace};
use crate::result::OptimizeResult;
use crate::unconstrained::FirstOrderStorage;
use scioxide_linalg::{Matrix, Vector};

use crate::storage::resize_with_quantity as resize;

/// Reusable storage for penalty and augmented-Lagrangian solvers.
#[derive(Debug)]
#[must_use]
pub(crate) struct ConstrainedStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    pub(crate) accepted_snapshot: crate::constrained::snapshot::Snapshot<S>,
    pub(crate) trial_snapshot: crate::constrained::snapshot::Snapshot<S>,
    pub(crate) accepted_diagnostics: crate::OptimalityDiagnostics<S>,
    pub(crate) accepted_value: Option<S>,
    pub(crate) accepted_iterations: usize,
    pub(crate) dimension: usize,
    pub(crate) equality_count: usize,
    pub(crate) inequality_count: usize,
    pub(crate) x: Vec<S>,
    pub(crate) objective_gradient: Vec<S>,
    pub(crate) trial_gradient: Vec<S>,
    pub(crate) constraint_values: Vec<S>,
    pub(crate) trial_constraint_values: Vec<S>,
    pub(crate) multipliers: Vec<S>,
    pub(crate) certificate_multipliers: Vec<S>,
    pub(crate) active_constraints: Vec<usize>,
    pub(crate) factor_rhs: Vector<S>,
    pub(crate) factor_solution: Vector<S>,
    pub(crate) merit_scratch: Vec<S>,
    pub(crate) constraint_jacobian: Matrix<S>,
    pub(crate) constraints: ConstraintsWorkspace<S>,
    pub(crate) objective_hessian: Matrix<S>,
    pub(crate) kkt: Matrix<S>,
    pub(crate) certificate_qr: scioxide_linalg::HouseholderDecomposition<S>,
    pub(crate) certificate_work: LinalgWorkspace<'static, S>,
    pub(crate) certificate_candidates: Vec<usize>,
    pub(crate) certificate_provider: Option<crate::DenseKktProvider<S>>,
    pub(crate) first_order: FirstOrderStorage<S>,
    pub(crate) inner_result: OptimizeResult<S>,
    pub(crate) penalty_parameter: S,
}

impl<S> ConstrainedStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Create empty storage without allocating.
    pub(crate) fn new() -> Self {
        Self {
            accepted_snapshot: crate::constrained::snapshot::Snapshot::empty(),
            trial_snapshot: crate::constrained::snapshot::Snapshot::empty(),
            accepted_diagnostics: crate::OptimalityDiagnostics::unavailable(),
            accepted_value: None,
            accepted_iterations: 0,
            dimension: 0,
            equality_count: 0,
            inequality_count: 0,
            x: Vec::new(),
            objective_gradient: Vec::new(),
            trial_gradient: Vec::new(),
            constraint_values: Vec::new(),
            trial_constraint_values: Vec::new(),
            multipliers: Vec::new(),
            certificate_multipliers: Vec::new(),
            active_constraints: Vec::new(),
            factor_rhs: Vector::empty(),
            factor_solution: Vector::empty(),
            merit_scratch: Vec::new(),
            constraint_jacobian: Matrix::empty(),
            constraints: ConstraintsWorkspace::empty(),
            objective_hessian: Matrix::empty(),
            kkt: Matrix::empty(),
            certificate_qr: scioxide_linalg::HouseholderDecomposition::empty(),
            certificate_work: LinalgWorkspace::new(),
            certificate_candidates: Vec::new(),
            certificate_provider: None,
            first_order: FirstOrderStorage::new(),
            inner_result: OptimizeResult::empty(),
            penalty_parameter: <S as scioxide_core::AlgebraicScalar>::ZERO,
        }
    }

    /// Whether every buffer was prepared for exactly this shape.
    pub(crate) fn fits(&self, dimension: usize, equalities: usize, inequalities: usize) -> bool {
        self.certificate_provider.is_some()
            && self.dimension == dimension
            && self.equality_count == equalities
            && self.inequality_count == inequalities
    }

    pub(crate) fn try_reserve_nested_bfgs(
        &mut self,
        dimension: usize,
        equalities: usize,
        inequalities: usize,
    ) -> Result<(), SciOxideError> {
        self.try_reserve_solver_storage_impl(dimension, equalities, inequalities)?;
        self.accepted_snapshot
            .reserve(dimension, equalities, inequalities)?;
        self.trial_snapshot
            .reserve(dimension, equalities, inequalities)?;
        self.constraints.try_reserve(
            dimension,
            ConstraintProfile::from_counts(equalities, inequalities),
        )?;
        // Record the shape only after every buffer is prepared.
        self.dimension = dimension;
        self.equality_count = equalities;
        self.inequality_count = inequalities;
        Ok(())
    }

    fn try_reserve_solver_storage_impl(
        &mut self,
        dimension: usize,
        equalities: usize,
        inequalities: usize,
    ) -> Result<(), SciOxideError> {
        let constraints = checked_add(
            equalities,
            inequalities,
            scioxide_core::quantity!("constraint_count"),
        )?;
        let kkt_order = checked_add(
            dimension,
            constraints,
            scioxide_core::quantity!("kkt_order"),
        )?;
        for (values, context) in [
            (
                &mut self.x,
                scioxide_core::quantity!("constrained_current_point"),
            ),
            (
                &mut self.objective_gradient,
                scioxide_core::quantity!("constrained_objective_gradient"),
            ),
            (
                &mut self.trial_gradient,
                scioxide_core::quantity!("constrained_trial_gradient"),
            ),
        ] {
            resize(
                values,
                dimension,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                context,
            )?;
        }
        for (values, context) in [
            (
                &mut self.constraint_values,
                scioxide_core::quantity!("constraint_values"),
            ),
            (
                &mut self.trial_constraint_values,
                scioxide_core::quantity!("trial_constraint_values"),
            ),
            (
                &mut self.multipliers,
                scioxide_core::quantity!("constraint_multipliers"),
            ),
            (
                &mut self.certificate_multipliers,
                scioxide_core::quantity!("certificate_multipliers"),
            ),
        ] {
            resize(
                values,
                constraints,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                context,
            )?;
        }
        resize(
            &mut self.merit_scratch,
            constraints.max(dimension),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("constraint_merit_scratch"),
        )?;
        if constraints > self.active_constraints.len() {
            let additional = constraints - self.active_constraints.len();
            try_reserve(
                &mut self.active_constraints,
                additional,
                scioxide_core::quantity!("active_constraint_indices"),
            )?;
        }
        self.active_constraints.clear();
        self.factor_rhs
            .try_resize(kkt_order, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.factor_solution
            .try_resize(kkt_order, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.constraint_jacobian
            .try_resize_flat(constraints, dimension)?;
        self.objective_hessian
            .try_resize_flat(dimension, constraints)?;
        self.kkt.try_resize_flat(kkt_order, kkt_order)?;
        self.certificate_qr.try_reserve(
            dimension,
            constraints,
            scioxide_linalg::HouseholderOrientation::Qr,
        )?;
        self.certificate_work.try_reserve(
            &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::factor(
                    dimension,
                    constraints,
                    scioxide_linalg::HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?,
        )?;
        crate::storage::resize(&mut self.certificate_candidates, constraints, 0)?;
        if self.certificate_provider.is_none()
            || self.dimension != dimension
            || self.equality_count != equalities
            || self.inequality_count != inequalities
        {
            self.certificate_provider = Some(crate::DenseKktProvider::new(kkt_order)?);
        }
        self.first_order.try_reserve_bfgs(dimension)?;
        self.inner_result.try_reserve(dimension)?;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.accepted_value = None;
        self.accepted_diagnostics = crate::OptimalityDiagnostics::unavailable();
        self.accepted_iterations = 0;
        self.first_order.reset();
        self.penalty_parameter = <S as scioxide_core::AlgebraicScalar>::ZERO;
        self.active_constraints.clear();
    }
}

impl<S> Default for ConstrainedStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn default() -> Self {
        Self::new()
    }
}
