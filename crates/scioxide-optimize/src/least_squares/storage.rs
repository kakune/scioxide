//! Reusable storage for nonlinear least-squares solves.

use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::backend::checked_add;
use scioxide_core::{ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};

use scioxide_linalg::{Matrix, Vector};

use crate::storage::resize_with_quantity as resize;

/// Reusable storage for Gauss-Newton and Levenberg-Marquardt.
#[derive(Debug)]
#[must_use]
pub(crate) struct LeastSquaresStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    pub(crate) accepted_value: Option<scioxide_core::Scaled<S>>,
    pub(crate) accepted_gradient: bool,
    pub(crate) accepted_iterations: usize,
    pub(crate) parameter_dimension: usize,
    pub(crate) residual_dimension: usize,
    pub(crate) x: Vec<S>,
    pub(crate) trial_x: Vec<S>,
    pub(crate) residual: Vec<S>,
    pub(crate) trial_residual: Vec<S>,
    pub(crate) gradient: Vec<S>,
    pub(crate) gradient_norm: Option<scioxide_core::Scaled<S>>,
    pub(crate) row_weights: Vec<scioxide_core::Scaled<S>>,
    pub(crate) parameter_scales: Vec<S>,
    pub(crate) column_norms: Vec<scioxide_core::Scaled<S>>,
    pub(crate) current_column_norms: Vec<scioxide_core::ScaledNorm<S>>,
    pub(crate) orthogonality: S,
    pub(crate) radius: Option<scioxide_core::Scaled<S>>,
    pub(crate) curvature: Vec<scioxide_core::Scaled<S>>,
    pub(crate) native_weights: Vec<S>,
    pub(crate) native_curvature: Vec<S>,
    pub(crate) damping_diagonal: Vec<S>,
    pub(crate) factor_scales: Vec<S>,
    pub(crate) model_jacobian: Matrix<S>,
    pub(crate) weighted_residual: Vec<S>,
    pub(crate) model_scale: scioxide_core::Scaled<S>,
    pub(crate) metric_scale: scioxide_core::Scaled<S>,
    pub(crate) step: Vec<S>,
    pub(crate) model_product: Vec<S>,
    pub(crate) residual_rhs: Vector<S>,
    pub(crate) factor_solution: Vector<S>,
    pub(crate) jacobian: Matrix<S>,
    pub(crate) finite_difference: Option<DerivativeWorkspace<S>>,
    pub(crate) qr_factor: scioxide_linalg::HouseholderDecomposition<S>,
    pub(crate) qr_workspace: scioxide_linalg::LinalgWorkspace<'static, S>,
    pub(crate) use_svd: bool,
    pub(crate) svd_factor: scioxide_linalg::ThinSvdDecomposition<S>,
    pub(crate) svd_workspace: scioxide_linalg::LinalgWorkspace<'static, S>,
    pub(crate) rank: Option<scioxide_linalg::RankReport<S>>,
    pub(crate) last_gain: Option<scioxide_core::Scaled<S>>,
    pub(crate) inner_termination: Option<scioxide_linalg::LinearSolveTermination>,
    pub(crate) condition: Option<scioxide_linalg::ConditionEstimate<S>>,
    pub(crate) lsmr_workspace: Option<scioxide_linalg::IterativeWorkspace<S>>,
    pub(crate) operator_scratch: Vec<S>,
    pub(crate) operator_shape: (usize, usize, usize),
    pub(crate) damping: S,
    shape: Option<LeastSquaresShape>,
}

/// Storage shape of a nonlinear least-squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeastSquaresShape {
    pub(crate) parameters: usize,
    pub(crate) residuals: usize,
    /// Provider scratch of the Jacobian operator; absent for a dense Jacobian.
    pub(crate) operator_scratch: Option<usize>,
}

impl<S> LeastSquaresStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Create empty storage without allocating.
    pub(crate) fn new() -> Self {
        Self {
            accepted_value: None,
            accepted_gradient: false,
            accepted_iterations: 0,
            parameter_dimension: 0,
            residual_dimension: 0,
            x: Vec::new(),
            trial_x: Vec::new(),
            residual: Vec::new(),
            trial_residual: Vec::new(),
            gradient: Vec::new(),
            gradient_norm: None,
            row_weights: Vec::new(),
            parameter_scales: Vec::new(),
            column_norms: Vec::new(),
            current_column_norms: Vec::new(),
            orthogonality: <S as scioxide_core::AlgebraicScalar>::ZERO,
            radius: None,
            curvature: Vec::new(),
            native_weights: Vec::new(),
            native_curvature: Vec::new(),
            damping_diagonal: Vec::new(),
            factor_scales: Vec::new(),
            model_jacobian: Matrix::empty(),
            weighted_residual: Vec::new(),
            model_scale: scioxide_core::Scaled::ONE,
            metric_scale: scioxide_core::Scaled::ONE,
            step: Vec::new(),
            model_product: Vec::new(),
            residual_rhs: Vector::empty(),
            factor_solution: Vector::empty(),
            jacobian: Matrix::empty(),
            finite_difference: None,
            qr_factor: scioxide_linalg::HouseholderDecomposition::empty(),
            qr_workspace: scioxide_linalg::LinalgWorkspace::new(),
            use_svd: true,
            svd_factor: scioxide_linalg::ThinSvdDecomposition::empty(),
            svd_workspace: scioxide_linalg::LinalgWorkspace::new(),
            rank: None,
            last_gain: None,
            inner_termination: None,
            condition: None,
            lsmr_workspace: None,
            operator_scratch: Vec::new(),
            operator_shape: (0, 0, 0),
            damping: <S as scioxide_core::AlgebraicScalar>::ZERO,
            shape: None,
        }
    }

    /// Whether the storage was prepared with exactly this shape.
    pub(crate) fn fits(&self, shape: LeastSquaresShape) -> bool {
        self.shape == Some(shape)
    }

    /// Replace the storage with every buffer and dense factor output of `shape`.
    ///
    /// On failure the existing storage is kept.
    pub(crate) fn prepare(&mut self, shape: LeastSquaresShape) -> Result<(), SciOxideError> {
        let (parameters, residuals) = (shape.parameters, shape.residuals);
        let dense = shape.operator_scratch.is_none();
        let mut candidate = Self::new();
        candidate.try_reserve(parameters, residuals, dense)?;
        if let Some(provider) = shape.operator_scratch {
            let scratch = provider
                .checked_add(parameters.max(residuals))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("operator_scratch"))
                })?;
            resize(
                &mut candidate.operator_scratch,
                scratch,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.operator_scratch"),
            )?;
            let rows = checked_add(
                parameters,
                residuals,
                scioxide_core::quantity!("augmented_rows"),
            )?;
            candidate
                .residual_rhs
                .try_resize(rows, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        } else if parameters > 0 && residuals > 0 {
            // Factor outputs swap with their staging: warm both sides of each kernel.
            let execution = scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential);
            for _ in 0..2 {
                if residuals >= parameters {
                    scioxide_linalg::HouseholderDecomposition::factor_into(
                        &candidate.jacobian,
                        scioxide_linalg::HouseholderOrientation::Qr,
                        scioxide_linalg::HouseholderPivoting::Column,
                        &execution,
                        &mut candidate.qr_factor,
                        &mut candidate.qr_workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )?;
                }
                scioxide_linalg::ThinSvdDecomposition::compute_into(
                    &candidate.jacobian,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    &mut candidate.svd_factor,
                    &mut candidate.svd_workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )?;
            }
        }
        candidate.shape = Some(shape);
        *self = candidate;
        Ok(())
    }

    /// Reserve storage for a problem with `parameters` and `residuals`.
    ///
    /// # Errors
    ///
    /// Returns a structured size, allocation, or execution-policy error.
    pub(crate) fn try_reserve(
        &mut self,
        parameters: usize,
        residuals: usize,
        dense: bool,
    ) -> Result<(), SciOxideError> {
        for (values, context) in [
            (
                &mut self.x,
                scioxide_core::quantity!("least_squares_current_parameters"),
            ),
            (
                &mut self.trial_x,
                scioxide_core::quantity!("least_squares_trial_parameters"),
            ),
            (
                &mut self.gradient,
                scioxide_core::quantity!("least_squares_gradient"),
            ),
            (
                &mut self.step,
                scioxide_core::quantity!("least_squares_step"),
            ),
        ] {
            resize(
                values,
                parameters,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                context,
            )?;
        }
        resize(
            &mut self.residual,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("least_squares_residual"),
        )?;
        resize(
            &mut self.trial_residual,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("least_squares_trial_residual"),
        )?;
        resize(
            &mut self.damping_diagonal,
            parameters,
            <S as scioxide_core::AlgebraicScalar>::ONE,
            scioxide_core::quantity!("damping_diagonal"),
        )?;
        resize(
            &mut self.factor_scales,
            parameters,
            <S as scioxide_core::AlgebraicScalar>::ONE,
            scioxide_core::quantity!("factor_scales"),
        )?;
        resize(
            &mut self.current_column_norms,
            parameters,
            scioxide_core::ScaledNorm::new(),
            scioxide_core::quantity!("current_column_norms"),
        )?;
        resize(
            &mut self.column_norms,
            parameters,
            scioxide_core::Scaled::ZERO,
            scioxide_core::quantity!("column_norms"),
        )?;
        resize(
            &mut self.curvature,
            residuals,
            scioxide_core::Scaled::ONE,
            scioxide_core::quantity!("curvature"),
        )?;
        resize(
            &mut self.parameter_scales,
            parameters,
            <S as scioxide_core::AlgebraicScalar>::ONE,
            scioxide_core::quantity!("parameter_scales"),
        )?;
        resize(
            &mut self.row_weights,
            residuals,
            scioxide_core::Scaled::ZERO,
            scioxide_core::quantity!("residual_weights"),
        )?;
        resize(
            &mut self.weighted_residual,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("weighted_residual"),
        )?;
        resize(
            &mut self.native_weights,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ONE,
            scioxide_core::quantity!("native_weights"),
        )?;
        resize(
            &mut self.native_curvature,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ONE,
            scioxide_core::quantity!("native_curvature"),
        )?;
        if dense {
            self.model_jacobian.try_resize_flat(residuals, parameters)?;
            self.jacobian.try_resize_flat(residuals, parameters)?;
        }
        self.factor_solution
            .try_resize(parameters, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.residual_rhs
            .try_resize(residuals, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        resize(
            &mut self.model_product,
            residuals,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("least_squares_model_residual"),
        )?;
        if dense {
            self.qr_workspace.try_reserve(
                &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::HouseholderWorkspaceRequest::factor(
                        residuals,
                        parameters,
                        scioxide_linalg::HouseholderOrientation::Qr,
                        &scioxide_linalg::LinalgExecutionOptions::new()
                            .with_execution_policy(ExecutionPolicy::Sequential),
                    ),
                )?,
            )?;
            // Wide systems use the SVD path and have no strict QR solve job.
            if residuals >= parameters {
                self.qr_workspace.try_reserve(
                    &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                        &scioxide_linalg::HouseholderWorkspaceRequest::damped_solve(
                            residuals,
                            parameters,
                            &scioxide_linalg::LinalgExecutionOptions::new(),
                        ),
                    )?,
                )?;
            }
        }
        if dense {
            self.svd_workspace.try_reserve(
                &scioxide_linalg::ThinSvdDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::SvdWorkspaceRequest::new(
                        residuals,
                        parameters,
                        &(scioxide_linalg::LinalgExecutionOptions::new()),
                    ),
                )?,
            )?;
        }
        if dense
            && self.finite_difference.as_ref().is_none_or(|workspace| {
                workspace.input_dimension() != parameters
                    || workspace.output_dimension() != residuals
            })
        {
            self.finite_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Jacobian,
                parameters,
                residuals,
            )?);
        }
        self.parameter_dimension = parameters;
        self.residual_dimension = residuals;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.accepted_value = None;
        self.column_norms.fill(scioxide_core::Scaled::ZERO);
        self.radius = None;
        self.gradient_norm = None;
        self.rank = None;
        self.condition = None;
        self.last_gain = None;
        self.inner_termination = None;
        self.accepted_gradient = false;
        self.accepted_iterations = 0;
        self.damping = <S as scioxide_core::AlgebraicScalar>::ZERO;
    }
}

impl<S> Default for LeastSquaresStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn default() -> Self {
        Self::new()
    }
}
