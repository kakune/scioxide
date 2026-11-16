//! Explicit same-point observations. Only completed fields become reusable.

use crate::context::Observation;
use crate::{Constraints, ConstraintsWorkspace, ObjectiveFunction};
use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;

#[derive(Debug)]
pub(crate) struct Snapshot<S: RealScalar + NumericScalar<Real = S>> {
    pub point: Vec<S>,
    pub value: Option<S>,
    pub values: Vec<S>,
    pub gradient: Vec<S>,
    pub jacobian: Matrix<S>,
    pub original_value: Option<S>,
    pub original_values: Vec<S>,
    pub objective_scale: S,
    pub row_scales: Vec<S>,
    values_valid: bool,
    gradient_valid: bool,
    jacobian_valid: bool,
    constraints: ConstraintsWorkspace<S>,
    finite_difference: Option<DerivativeWorkspace<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Snapshot<S> {
    pub(crate) fn empty() -> Self {
        Self {
            point: Vec::new(),
            value: None,
            values: Vec::new(),
            gradient: Vec::new(),
            jacobian: Matrix::empty(),
            original_value: None,
            original_values: Vec::new(),
            objective_scale: <S as scioxide_core::AlgebraicScalar>::ZERO,
            row_scales: Vec::new(),
            values_valid: false,
            gradient_valid: false,
            jacobian_valid: false,
            constraints: ConstraintsWorkspace::empty(),
            finite_difference: None,
        }
    }

    pub(crate) fn reserve(
        &mut self,
        n: usize,
        equalities: usize,
        inequalities: usize,
    ) -> Result<(), SciOxideError> {
        let m = scioxide_core::backend::checked_add(
            equalities,
            inequalities,
            scioxide_core::quantity!("snapshot_constraints"),
        )?;
        for (buffer, len) in [
            (&mut self.point, n),
            (&mut self.gradient, n),
            (&mut self.values, m),
            (&mut self.original_values, m),
            (&mut self.row_scales, m),
        ] {
            if len > buffer.len() {
                scioxide_core::backend::try_reserve(
                    buffer,
                    len - buffer.len(),
                    scioxide_core::quantity!("constrained_snapshot"),
                )?;
            }
            buffer.resize(len, <S as scioxide_core::AlgebraicScalar>::ZERO);
        }
        self.jacobian.try_resize_flat(m, n)?;
        self.constraints.try_reserve(
            n,
            crate::ConstraintProfile::from_counts(equalities, inequalities),
        )?;
        if self
            .finite_difference
            .as_ref()
            .is_none_or(|w| w.input_dimension() != n)
        {
            self.finite_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Gradient,
                n,
                1,
            )?);
        }
        self.invalidate();
        Ok(())
    }

    pub(crate) fn invalidate(&mut self) {
        self.value = None;
        self.original_value = None;
        self.values_valid = false;
        self.gradient_valid = false;
        self.jacobian_valid = false;
    }

    pub(crate) fn at(&mut self, point: &[S]) {
        // Equality alone would incorrectly reuse a callback at a different signed zero.
        if self
            .point
            .iter()
            .zip(point)
            .any(|(&a, &b)| a != b || a.is_sign_negative() != b.is_sign_negative())
        {
            self.invalidate();
            self.point.copy_from_slice(point);
        }
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.point.copy_from_slice(&other.point);
        self.value = other.value;
        self.values.copy_from_slice(&other.values);
        self.gradient.copy_from_slice(&other.gradient);
        self.jacobian
            .as_mut_slice()
            .copy_from_slice(other.jacobian.as_slice());
        self.values_valid = other.values_valid;
        self.gradient_valid = other.gradient_valid;
        self.jacobian_valid = other.jacobian_valid;
        self.original_value = other.original_value;
        self.original_values.copy_from_slice(&other.original_values);
        self.objective_scale = other.objective_scale;
        self.row_scales.copy_from_slice(&other.row_scales);
    }

    pub(crate) fn objective<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        session: &dyn Observation,
    ) -> Result<S, SciOxideError> {
        let value = self.load_objective(f, session)?;
        crate::utils::validate_finite(&[value])?;
        Ok(value)
    }

    pub(crate) fn constraints(
        &mut self,
        constraints: Option<&Constraints<S>>,
        session: &dyn Observation,
    ) -> Result<(), SciOxideError> {
        self.load_constraints(constraints, session, false)
            .map(|_| ())
    }

    /// Observe a trial for an inner search. `false` reports a non-finite
    /// constraint or (unless `restoration`) objective value, which rejects the
    /// trial; nothing non-finite is stored.
    pub(crate) fn trial<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&Constraints<S>>,
        restoration: bool,
        session: &dyn Observation,
    ) -> Result<bool, SciOxideError> {
        Ok(self.load_constraints(constraints, session, true)?
            && (restoration || self.load_objective(f, session)?.is_finite()))
    }

    /// Normalized objective; a non-finite value is returned but not stored.
    fn load_objective<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        session: &dyn Observation,
    ) -> Result<S, SciOxideError> {
        if let Some(value) = self.value {
            session.cache_hit()?;
            return Ok(value);
        }
        let value = crate::context::observe(f, session).eval(&self.point)?;
        if !value.is_finite() {
            return Ok(value);
        }
        self.original_value = Some(value);
        let scale = if self.objective_scale > <S as scioxide_core::AlgebraicScalar>::ZERO {
            self.objective_scale
        } else {
            <S as scioxide_core::AlgebraicScalar>::ONE
        };
        let normalized = scioxide_core::checked_float_quotient(value, scale)?;
        self.value = Some(normalized);
        Ok(normalized)
    }

    /// Evaluate and normalize constraint values. A `trial` with a non-finite
    /// value returns `false` without storing it; otherwise values are checked.
    fn load_constraints(
        &mut self,
        constraints: Option<&Constraints<S>>,
        session: &dyn Observation,
        trial: bool,
    ) -> Result<bool, SciOxideError> {
        if self.values_valid {
            session.cache_hit()?;
            return Ok(true);
        }
        if let Some(constraints) = constraints {
            let (eq, ineq) = self
                .original_values
                .split_at_mut(constraints.profile().equality_count());
            if trial {
                if !constraints.evaluate_trial_observed(
                    &self.point,
                    eq,
                    ineq,
                    &mut self.constraints,
                    Some(session),
                )? {
                    return Ok(false);
                }
            } else {
                constraints.evaluate_observed(
                    &self.point,
                    eq,
                    ineq,
                    &mut self.constraints,
                    Some(session),
                )?;
            }
        }
        for ((out, &value), &scale) in self
            .values
            .iter_mut()
            .zip(&self.original_values)
            .zip(&self.row_scales)
        {
            *out = scioxide_core::checked_float_quotient(value, scale)?;
        }
        self.values_valid = true;
        Ok(true)
    }

    pub(crate) fn derivatives<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&Constraints<S>>,
        strategy: DerivativeStrategy,
        restoration: bool,
        session: &dyn Observation,
    ) -> Result<(), SciOxideError> {
        self.constraints(constraints, session)?;
        let new_gradient = !restoration && !self.gradient_valid;
        let new_jacobian = !self.jacobian_valid;
        if !restoration && !self.gradient_valid {
            self.objective(f, session)?;
            let value = self.original_value.ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("f"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing original objective",
                )
            })?;
            let finite_difference = crate::derivatives::prepared(&mut self.finite_difference)?;
            crate::utils::get_gradient_into(
                &crate::context::observe(f, session),
                &self.point,
                value,
                strategy,
                &mut self.gradient,
                finite_difference,
                session.execution()?,
            )?;
            self.gradient_valid = true;
        }
        if !self.jacobian_valid {
            if let Some(constraints) = constraints {
                let (eq, ineq) = self
                    .original_values
                    .split_at(constraints.profile().equality_count());
                let _ = constraints.jacobian_observed(
                    &self.point,
                    eq,
                    ineq,
                    strategy,
                    &mut self.jacobian,
                    &mut self.constraints,
                    Some(session),
                )?;
            }
            self.jacobian_valid = true;
        }
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        if self.objective_scale == zero {
            let maximum = self
                .gradient
                .iter()
                .fold(zero, |a, &b| a.max(NumericScalar::abs(b)));
            self.objective_scale = if maximum > zero { maximum } else { one };
            for (i, scale) in self.row_scales.iter_mut().enumerate() {
                let maximum = self.jacobian.as_slice()
                    [i * self.point.len()..(i + 1) * self.point.len()]
                    .iter()
                    .fold(zero, |a, &b| a.max(NumericScalar::abs(b)));
                let magnitude = if maximum > zero {
                    maximum
                } else {
                    NumericScalar::abs(self.original_values[i])
                };
                *scale = if magnitude > zero { magnitude } else { one };
            }
        }
        self.value = self
            .original_value
            .map(|v| scioxide_core::checked_float_quotient(v, self.objective_scale))
            .transpose()?;
        if new_gradient {
            for value in &mut self.gradient {
                *value = scioxide_core::checked_float_quotient(*value, self.objective_scale)?;
            }
        }
        for (i, &scale) in self.row_scales.iter().enumerate() {
            self.values[i] = scioxide_core::checked_float_quotient(self.original_values[i], scale)?;
            if new_jacobian {
                for value in &mut self.jacobian.as_mut_slice()
                    [i * self.point.len()..(i + 1) * self.point.len()]
                {
                    *value = scioxide_core::checked_float_quotient(*value, scale)?;
                }
            }
        }
        Ok(())
    }
}
