//! Reusable storage for Newton and trust-region minimizers.

use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::{ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{CholeskyDecomposition, LinalgWorkspace};

use scioxide_linalg::{Matrix, Vector};

use crate::storage::resize_with_quantity as resize;

/// Storage shape of a Newton or trust-region solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SecondOrderShape {
    /// Number of variables.
    pub(crate) dimension: usize,
    /// Dense Hessian and its finite-difference workspace.
    pub(crate) dense: bool,
    /// Modified-Newton candidate Hessian and LDL factor storage.
    pub(crate) regularize: bool,
    /// Steihaug-CG Krylov vectors.
    pub(crate) cg: bool,
    /// Dense LDL inertia check of a stationary point.
    pub(crate) second_order: bool,
}

/// Reusable storage for Newton and trust-region solvers.
#[derive(Debug)]
#[must_use]
pub(crate) struct SecondOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    pub(crate) accepted_value: Option<S>,
    pub(crate) accepted_gradient: bool,
    pub(crate) accepted_iterations: usize,
    pub(crate) dimension: usize,
    pub(crate) x: Vec<S>,
    pub(crate) trial_x: Vec<S>,
    pub(crate) gradient: Vec<S>,
    pub(crate) trial_gradient: Vec<S>,
    pub(crate) direction: Vec<S>,
    pub(crate) rhs: Vec<S>,
    pub(crate) cg_residual: Vec<S>,
    pub(crate) cg_direction: Vec<S>,
    pub(crate) cg_product: Vec<S>,
    pub(crate) model_product: Vec<S>,
    pub(crate) gemv_scratch: Vec<S>,
    pub(crate) factor_rhs: Vector<S>,
    pub(crate) factor_solution: Vector<S>,
    pub(crate) hessian: Matrix<S>,
    pub(crate) candidate_hessian: Matrix<S>,
    pub(crate) gradient_difference: Option<DerivativeWorkspace<S>>,
    pub(crate) hessian_difference: Option<DerivativeWorkspace<S>>,
    pub(crate) cholesky_factor: CholeskyDecomposition<S>,
    pub(crate) cholesky_workspace: LinalgWorkspace<'static, S>,
    pub(crate) ldl_factor: scioxide_linalg::LdlDecomposition<S>,
    pub(crate) ldl_workspace: scioxide_linalg::LinalgWorkspace<'static, S>,
    pub(crate) newton: crate::NewtonDiagnostics<S>,
    shape: Option<SecondOrderShape>,
}

impl<S> SecondOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Create empty storage without allocating.
    pub(crate) fn new() -> Self {
        Self {
            accepted_value: None,
            accepted_gradient: false,
            accepted_iterations: 0,
            dimension: 0,
            x: Vec::new(),
            trial_x: Vec::new(),
            gradient: Vec::new(),
            trial_gradient: Vec::new(),
            direction: Vec::new(),
            rhs: Vec::new(),
            cg_residual: Vec::new(),
            cg_direction: Vec::new(),
            cg_product: Vec::new(),
            model_product: Vec::new(),
            gemv_scratch: Vec::new(),
            factor_rhs: Vector::empty(),
            factor_solution: Vector::empty(),
            hessian: Matrix::empty(),
            candidate_hessian: Matrix::empty(),
            gradient_difference: None,
            hessian_difference: None,
            cholesky_factor: CholeskyDecomposition::empty(),
            cholesky_workspace: LinalgWorkspace::new(),
            ldl_factor: scioxide_linalg::LdlDecomposition::empty(),
            ldl_workspace: scioxide_linalg::LinalgWorkspace::new(),
            newton: crate::NewtonDiagnostics::default(),
            shape: None,
        }
    }

    /// Whether the storage was prepared with exactly this shape.
    pub(crate) fn fits(&self, shape: SecondOrderShape) -> bool {
        self.shape == Some(shape)
    }

    /// Replace the storage with every buffer and factor output of `shape`.
    ///
    /// On failure the existing storage is unchanged.
    pub(crate) fn prepare(&mut self, shape: SecondOrderShape) -> Result<(), SciOxideError> {
        let mut candidate = Self::new();
        candidate.try_reserve_model(shape.dimension, shape.regularize, shape.dense)?;
        if shape.cg {
            for (values, quantity) in [
                (
                    &mut candidate.cg_residual,
                    scioxide_core::quantity!("newton_cg_residual"),
                ),
                (
                    &mut candidate.cg_direction,
                    scioxide_core::quantity!("newton_cg_direction"),
                ),
                (
                    &mut candidate.cg_product,
                    scioxide_core::quantity!("newton_cg_product"),
                ),
            ] {
                resize(
                    values,
                    shape.dimension,
                    <S as scioxide_core::AlgebraicScalar>::ZERO,
                    quantity,
                )?;
            }
        }
        if shape.dense && !shape.regularize {
            candidate.cholesky_factor.try_reserve(shape.dimension)?;
        }
        if shape.second_order {
            candidate.ldl_factor.try_reserve(shape.dimension)?;
            candidate.ldl_workspace.try_reserve(
                &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::LdlWorkspaceRequest::factor(
                        shape.dimension,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
        }
        candidate.shape = Some(shape);
        *self = candidate;
        Ok(())
    }

    /// Reserve all second-order storage for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns a structured size, allocation, or execution-policy error.
    pub(crate) fn try_reserve(
        &mut self,
        dimension: usize,
        regularize: bool,
    ) -> Result<(), SciOxideError> {
        self.try_reserve_model(dimension, regularize, true)
    }

    pub(crate) fn try_reserve_model(
        &mut self,
        dimension: usize,
        regularize: bool,
        dense: bool,
    ) -> Result<(), SciOxideError> {
        for (values, context) in [
            (
                &mut self.x,
                scioxide_core::quantity!("second_order_current_point"),
            ),
            (
                &mut self.trial_x,
                scioxide_core::quantity!("second_order_trial_point"),
            ),
            (
                &mut self.gradient,
                scioxide_core::quantity!("second_order_gradient"),
            ),
            (
                &mut self.trial_gradient,
                scioxide_core::quantity!("second_order_trial_gradient"),
            ),
            (
                &mut self.direction,
                scioxide_core::quantity!("second_order_direction"),
            ),
            (
                &mut self.rhs,
                scioxide_core::quantity!("second_order_right_hand_side"),
            ),
            (
                &mut self.model_product,
                scioxide_core::quantity!("trust_region_model_product"),
            ),
            (
                &mut self.gemv_scratch,
                scioxide_core::quantity!("second_order_gemv_scratch"),
            ),
        ] {
            resize(
                values,
                dimension,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                context,
            )?;
        }
        if dense {
            self.hessian.try_resize_flat(dimension, dimension)?;
        }
        if regularize {
            self.candidate_hessian
                .try_resize_flat(dimension, dimension)?;
        }
        self.factor_rhs
            .try_resize(dimension, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.factor_solution
            .try_resize(dimension, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        if regularize {
            self.ldl_factor.try_reserve(dimension)?;
            self.ldl_workspace.try_reserve(
                &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::LdlWorkspaceRequest::factor(
                        dimension,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
            self.ldl_workspace.try_reserve(
                &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                        dimension,
                        1,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
        } else if dense {
            self.cholesky_workspace.try_reserve(
                &scioxide_linalg::CholeskyDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::CholeskyWorkspaceRequest::factor(
                        dimension,
                        &scioxide_linalg::LinalgExecutionOptions::new()
                            .with_execution_policy(ExecutionPolicy::Sequential),
                    ),
                )?,
            )?;
            self.cholesky_workspace.try_reserve(
                &scioxide_linalg::CholeskyDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::CholeskyWorkspaceRequest::solve(
                        dimension,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
        }
        if self
            .gradient_difference
            .as_ref()
            .is_none_or(|workspace| workspace.input_dimension() != dimension)
        {
            self.gradient_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Gradient,
                dimension,
                1,
            )?);
        }
        if dense
            && self
                .hessian_difference
                .as_ref()
                .is_none_or(|workspace| workspace.input_dimension() != dimension)
        {
            self.hessian_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Hessian,
                dimension,
                1,
            )?);
        }
        self.dimension = dimension;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.accepted_value = None;
        self.accepted_gradient = false;
        self.accepted_iterations = 0;
        self.newton = crate::NewtonDiagnostics::default();
    }
}

impl<S> crate::attempt::Accepted<S> for SecondOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        Some(crate::attempt::AcceptedState {
            point: &self.x,
            value: self.accepted_value?,
            gradient: self.accepted_gradient.then_some(self.gradient.as_slice()),
            iterations: self.accepted_iterations,
        })
    }
}

impl<S> Default for SecondOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn default() -> Self {
        Self::new()
    }
}
