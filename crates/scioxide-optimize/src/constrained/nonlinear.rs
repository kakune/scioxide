//! Shared bounded nonlinear observations and inward numerical derivatives.
use super::sqp::{invalid, resize};
use crate::{ObjectiveFunction, context::SolveSession};
use alloc::vec::Vec;

use crate::derivatives::Differences;
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;
#[derive(Debug)]
pub(super) struct Observation<S: RealScalar + NumericScalar<Real = S>> {
    pub(super) x: Vec<S>,
    pub(super) value: Option<S>,
    pub(super) gradient: Vec<S>,
    pub(super) values: Vec<S>,
    pub(super) jacobian: Matrix<S>,
    pub(super) derivatives: bool,
    pub(super) fixed_derivatives: bool,
    pub(super) original_value: Option<S>,
    pub(super) original_gradient: Vec<S>,
    pub(super) objective_scale: S,
    pub(super) row_scales: Vec<S>,
    pub(super) original_values: Vec<S>,
    pub(super) original_multipliers: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Observation<S> {
    pub(super) fn new() -> Self {
        Self {
            x: Vec::new(),
            value: None,
            gradient: Vec::new(),
            values: Vec::new(),
            jacobian: Matrix::empty(),
            derivatives: false,
            fixed_derivatives: true,
            original_value: None,
            original_gradient: Vec::new(),
            objective_scale: <S as scioxide_core::AlgebraicScalar>::ZERO,
            row_scales: Vec::new(),
            original_values: Vec::new(),
            original_multipliers: Vec::new(),
        }
    }
    pub(super) fn reserve(&mut self, n: usize, m: usize) -> Result<(), SciOxideError> {
        resize(&mut self.x, n, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        resize(
            &mut self.gradient,
            n,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        resize(
            &mut self.values,
            m,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        self.jacobian.try_resize_flat(m, n).and_then(|()| {
            resize(
                &mut self.original_gradient,
                n,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
            )?;
            resize(
                &mut self.original_values,
                m,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
            )?;
            resize(
                &mut self.original_multipliers,
                m,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
            )?;
            resize(
                &mut self.row_scales,
                m,
                <S as scioxide_core::AlgebraicScalar>::ONE,
            )
        })
    }
    pub(super) fn at(&mut self, x: &[S]) {
        self.x.copy_from_slice(x);
        self.value = None;
        self.derivatives = false;
        self.fixed_derivatives = true;
    }
    pub(super) fn copy_from(&mut self, s: &Self) {
        self.x.copy_from_slice(&s.x);
        self.value = s.value;
        self.gradient.copy_from_slice(&s.gradient);
        self.values.copy_from_slice(&s.values);
        self.jacobian
            .as_mut_slice()
            .copy_from_slice(s.jacobian.as_slice());
        self.derivatives = s.derivatives;
        self.fixed_derivatives = s.fixed_derivatives;
        self.inherit_scales(s);
        self.original_value = s.original_value;
        self.original_gradient.copy_from_slice(&s.original_gradient);
        self.original_values.copy_from_slice(&s.original_values);
    }

    pub(super) fn inherit_scales(&mut self, source: &Self) {
        self.objective_scale = source.objective_scale;
        self.row_scales.copy_from_slice(&source.row_scales);
    }

    /// Frozen row and objective normalization. Zero derivatives use a nonzero
    /// residual magnitude; a completely zero observation uses the unit scale.
    fn normalize_derivatives(&mut self) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        if self.objective_scale == zero {
            let maximum = self
                .gradient
                .iter()
                .fold(zero, |a, &b| a.max(NumericScalar::abs(b)));
            self.objective_scale = if maximum > zero {
                maximum
            } else {
                self.original_value
                    .map_or(one, |v| NumericScalar::abs(v).max(one))
            };
            for (i, scale) in self.row_scales.iter_mut().enumerate() {
                let maximum = self.jacobian.as_slice()[i * self.x.len()..(i + 1) * self.x.len()]
                    .iter()
                    .fold(zero, |a, &b| a.max(NumericScalar::abs(b)));
                *scale = if maximum > zero {
                    maximum
                } else {
                    let value = NumericScalar::abs(self.original_values[i]);
                    if value > zero { value } else { one }
                };
            }
        }
        self.value = self
            .original_value
            .map(|v| scioxide_core::checked_float_quotient(v, self.objective_scale))
            .transpose()?;
        self.original_gradient.copy_from_slice(&self.gradient);
        for g in &mut self.gradient {
            *g = scioxide_core::checked_float_quotient(*g, self.objective_scale)?;
        }
        for (i, &scale) in self.row_scales.iter().enumerate() {
            self.values[i] = scioxide_core::checked_float_quotient(self.original_values[i], scale)?;
            for j in &mut self.jacobian.as_mut_slice()[i * self.x.len()..(i + 1) * self.x.len()] {
                *j = scioxide_core::checked_float_quotient(*j, scale)?;
            }
        }
        Ok(())
    }
}
/// Why an observation was rejected.
enum Rejection {
    Domain,
    NonFinite,
}
impl<S: RealScalar + NumericScalar<Real = S>> Observation<S> {
    /// Evaluate a trial. `false` rejects it: the point lies outside the
    /// objective domain or an objective or constraint value is non-finite.
    pub(super) fn evaluate<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&crate::Constraints<S>>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok(self.observe(f, constraints, session)?.is_ok())
    }
    /// Evaluate the start, which must lie in the domain with finite values.
    pub(super) fn evaluate_start<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&crate::Constraints<S>>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        match self.observe(f, constraints, session)? {
            Ok(()) => Ok(()),
            Err(Rejection::Domain) => Err(crate::validation::invalid(
                scioxide_core::argument!("x0"),
                "initial point is outside the objective domain",
            )),
            Err(Rejection::NonFinite) => Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "objective or constraint returned a non-finite value at the start",
            )),
        }
    }
    fn observe<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&crate::Constraints<S>>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<Result<(), Rejection>, SciOxideError> {
        self.value = None;
        self.derivatives = false;
        if !session.invoke(EvaluationRole::Domain, || f.is_in_domain(&self.x))? {
            return Ok(Err(Rejection::Domain));
        }
        let value = session.invoke(EvaluationRole::Objective, || f.eval(&self.x))?;
        if !value.is_finite() {
            return Ok(Err(Rejection::NonFinite));
        }
        self.original_value = Some(value);
        let scale = if self.objective_scale > <S as scioxide_core::AlgebraicScalar>::ZERO {
            self.objective_scale
        } else {
            <S as scioxide_core::AlgebraicScalar>::ONE
        };
        self.value = Some(scioxide_core::checked_float_quotient(value, scale)?);
        if let Some(c) = constraints {
            for (i, function) in c.nonlinear_functions().enumerate() {
                let value =
                    session.invoke(EvaluationRole::Constraint, || function.eval(&self.x))?;
                if !value.is_finite() {
                    self.value = None;
                    return Ok(Err(Rejection::NonFinite));
                }
                self.original_values[i] = value;
                self.values[i] = scioxide_core::checked_float_quotient(value, self.row_scales[i])?;
            }
        }
        Ok(Ok(()))
    }
    pub(super) fn derivatives<F: ObjectiveFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        constraints: Option<&crate::Constraints<S>>,
        d: &mut Differences<S>,
        strategy: DerivativeStrategy,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let value = self.original_value.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("f"),
                "missing objective observation",
            )
        })?;
        self.fixed_derivatives = d.gradient(
            &self.x,
            value,
            strategy,
            &mut self.gradient,
            |analytic, out| {
                session.invoke(EvaluationRole::Gradient, || {
                    if analytic {
                        f.gradient_into(&self.x, out)
                    } else {
                        f.automatic_gradient_into(&self.x, out)
                    }
                })
            },
            |x| session.invoke(EvaluationRole::Objective, || f.eval(x)),
            session.execution()?,
        )?;
        if let Some(c) = constraints {
            let n = self.x.len();
            for (i, function) in c.nonlinear_functions().enumerate() {
                self.fixed_derivatives &= d.gradient(
                    &self.x,
                    self.original_values[i],
                    strategy,
                    &mut self.jacobian.as_mut_slice()[i * n..(i + 1) * n],
                    |analytic, out| {
                        session.invoke(EvaluationRole::ConstraintJacobian, || {
                            if analytic {
                                function.jacobian_into(&self.x, out)
                            } else {
                                function.automatic_jacobian_into(&self.x, out)
                            }
                        })
                    },
                    |x| session.invoke(EvaluationRole::Constraint, || function.eval(x)),
                    session.execution()?,
                )?;
            }
        }
        self.normalize_derivatives()?;
        self.derivatives = true;
        Ok(())
    }
}
