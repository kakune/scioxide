//! Reusable storage for first-order local minimizers.

use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::backend::checked_mul;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use scioxide_linalg::Matrix;

use crate::storage::resize_with_quantity as resize;

/// Reusable storage for gradient descent, nonlinear CG, BFGS, and L-BFGS.
#[derive(Debug)]
#[must_use]
pub(crate) struct FirstOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    pub(crate) globalization: crate::GlobalizationDiagnostics<S>,
    pub(crate) update_workspace: crate::HessianUpdateWorkspace<S>,
    pub(crate) accepted_value: Option<S>,
    pub(crate) accepted_gradient: bool,
    pub(crate) accepted_iterations: usize,
    pub(crate) dimension: usize,
    pub(crate) history_capacity: usize,
    pub(crate) x: Vec<S>,
    pub(crate) trial_x: Vec<S>,
    pub(crate) direction: Vec<S>,
    pub(crate) gradient: Vec<S>,
    pub(crate) trial_gradient: Vec<S>,
    pub(crate) s: Vec<S>,
    pub(crate) y: Vec<S>,
    pub(crate) q: Vec<S>,
    pub(crate) r: Vec<S>,
    pub(crate) hessian_product: Vec<S>,
    pub(crate) gemv_scratch: Vec<S>,
    pub(crate) inverse_hessian: Matrix<S>,
    pub(crate) history_s: Vec<S>,
    pub(crate) history_y: Vec<S>,
    pub(crate) history_curvature: Vec<scioxide_core::Scaled<S>>,
    pub(crate) history_alpha: Vec<scioxide_core::Scaled<S>>,
    // Physical ring-slot indexing keeps eviction O(m), without moving vectors.
    pub(crate) history_sy: Vec<S>,
    pub(crate) history_ss: Vec<S>,
    pub(crate) history_yy: Vec<S>,
    pub(crate) history_t: Vec<S>,
    pub(crate) history_b: Vec<S>,
    pub(crate) history_cache_valid: bool,
    pub(crate) history_len: usize,
    pub(crate) history_start: usize,
    pub(crate) finite_difference: Option<DerivativeWorkspace<S>>,
}

impl<S> FirstOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Create empty storage without allocating.
    pub(crate) fn new() -> Self {
        Self {
            globalization: crate::GlobalizationDiagnostics::default(),
            update_workspace: crate::HessianUpdateWorkspace::new(),
            accepted_value: None,
            accepted_gradient: false,
            accepted_iterations: 0,
            dimension: 0,
            history_capacity: 0,
            x: Vec::new(),
            trial_x: Vec::new(),
            direction: Vec::new(),
            gradient: Vec::new(),
            trial_gradient: Vec::new(),
            s: Vec::new(),
            y: Vec::new(),
            q: Vec::new(),
            r: Vec::new(),
            hessian_product: Vec::new(),
            gemv_scratch: Vec::new(),
            inverse_hessian: Matrix::empty(),
            history_s: Vec::new(),
            history_y: Vec::new(),
            history_curvature: Vec::new(),
            history_alpha: Vec::new(),
            history_sy: Vec::new(),
            history_ss: Vec::new(),
            history_yy: Vec::new(),
            history_t: Vec::new(),
            history_b: Vec::new(),
            history_cache_valid: true,
            history_len: 0,
            history_start: 0,
            finite_difference: None,
        }
    }

    /// Reserve storage for `dimension` variables and bounded L-BFGS `history`.
    ///
    /// Dense BFGS storage is deliberately excluded. If growth fails, the
    /// existing workspace remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns a structured size or allocation error.
    pub(crate) fn try_reserve(
        &mut self,
        dimension: usize,
        history: usize,
    ) -> Result<(), SciOxideError> {
        if self.dimension == dimension
            && self.history_capacity == history
            && self.inverse_hessian.rows() == 0
        {
            self.history_len = 0;
            self.history_start = 0;
            return Ok(());
        }
        let mut candidate = Self::new();
        candidate.try_reserve_storage(dimension, history, false)?;
        *self = candidate;
        Ok(())
    }

    /// Whether the storage already has exactly this shape.
    pub(crate) fn fits(&self, dimension: usize, history: usize, dense: bool) -> bool {
        self.dimension == dimension
            && self.history_capacity == history
            && self.inverse_hessian.rows() == if dense { dimension } else { 0 }
            && self.x.len() == dimension
            && self
                .finite_difference
                .as_ref()
                .is_some_and(|workspace| workspace.input_dimension() == dimension)
    }

    pub(crate) fn try_reserve_bfgs(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if self.dimension == dimension
            && self.history_capacity == 0
            && self.inverse_hessian.rows() == dimension
            && self.inverse_hessian.cols() == dimension
        {
            self.history_len = 0;
            self.history_start = 0;
            return Ok(());
        }
        let mut candidate = Self::new();
        candidate.try_reserve_storage(dimension, 0, true)?;
        *self = candidate;
        Ok(())
    }

    fn try_reserve_storage(
        &mut self,
        dimension: usize,
        history: usize,
        dense_inverse_hessian: bool,
    ) -> Result<(), SciOxideError> {
        let history_values = checked_mul(
            dimension,
            history,
            scioxide_core::quantity!("l_bfgs_history_storage"),
        )?;
        for (values, context) in [
            (
                &mut self.x,
                scioxide_core::quantity!("first_order_current_point"),
            ),
            (
                &mut self.trial_x,
                scioxide_core::quantity!("first_order_trial_point"),
            ),
            (
                &mut self.direction,
                scioxide_core::quantity!("first_order_direction"),
            ),
            (
                &mut self.gradient,
                scioxide_core::quantity!("first_order_gradient"),
            ),
            (
                &mut self.trial_gradient,
                scioxide_core::quantity!("first_order_trial_gradient"),
            ),
            (&mut self.s, scioxide_core::quantity!("first_order_bfgs_s")),
            (&mut self.y, scioxide_core::quantity!("first_order_bfgs_y")),
        ] {
            resize(
                values,
                dimension,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                context,
            )?;
        }
        for (buffer, needed, context) in [
            (
                &mut self.q,
                history > 0,
                scioxide_core::quantity!("l_bfgs_backward_pass"),
            ),
            (
                &mut self.r,
                history > 0,
                scioxide_core::quantity!("l_bfgs_forward_pass"),
            ),
            (
                &mut self.hessian_product,
                dense_inverse_hessian,
                scioxide_core::quantity!("bfgs_hessian_product"),
            ),
        ] {
            if needed {
                resize(
                    buffer,
                    dimension,
                    <S as scioxide_core::AlgebraicScalar>::ZERO,
                    context,
                )?;
            }
        }
        if dense_inverse_hessian {
            resize(
                &mut self.gemv_scratch,
                checked_mul(4, dimension, scioxide_core::quantity!("bfgs_product_lanes"))?,
                S::ZERO,
                scioxide_core::quantity!("bfgs_product_lanes"),
            )?;
            self.inverse_hessian.try_resize_flat(dimension, dimension)?;
            self.update_workspace.try_reserve(dimension)?;
        }
        resize(
            &mut self.history_s,
            history_values,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("l_bfgs_s_history"),
        )?;
        resize(
            &mut self.history_y,
            history_values,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("l_bfgs_y_history"),
        )?;
        resize(
            &mut self.history_curvature,
            history,
            scioxide_core::Scaled::ZERO,
            scioxide_core::quantity!("l_bfgs_curvature_history"),
        )?;
        resize(
            &mut self.history_alpha,
            history,
            scioxide_core::Scaled::ZERO,
            scioxide_core::quantity!("l_bfgs_alpha_history"),
        )?;
        let square = checked_mul(history, history, scioxide_core::quantity!("history_gram"))?;
        for (buffer, len) in [
            (&mut self.history_sy, square),
            (&mut self.history_yy, square),
            (&mut self.history_t, history),
            (&mut self.history_b, history),
        ] {
            resize(
                buffer,
                len,
                S::ZERO,
                scioxide_core::quantity!("history_gram"),
            )?;
        }
        self.history_cache_valid = true;
        if self
            .finite_difference
            .as_ref()
            .is_none_or(|workspace| workspace.input_dimension() != dimension)
        {
            self.finite_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Gradient,
                dimension,
                1,
            )?);
        }
        self.dimension = dimension;
        self.history_capacity = history;
        self.history_len = 0;
        self.history_start = 0;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.globalization = crate::GlobalizationDiagnostics::default();
        self.accepted_value = None;
        self.accepted_gradient = false;
        self.accepted_iterations = 0;
        self.history_len = 0;
        self.history_start = 0;
    }
}

impl<S> crate::attempt::Accepted<S> for FirstOrderStorage<S>
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

impl<S> Default for FirstOrderStorage<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn default() -> Self {
        Self::new()
    }
}
