//! Rules the caller supplies with their own derivative products.

use alloc::sync::Arc;
use alloc::vec::Vec;

use scioxide_core::{Capability, SciOxideError};

use super::call::{Call, Rule};
use super::word;
use super::{TapeValue, Var};
use crate::rules::Class;

/// A vector function `y = f(x)` with its own pullback and pushforward, recorded as
/// one call statement by [`Var::external`].
///
/// A rule is a pure function of its inputs: the tape keeps the `Arc` and the input
/// and output values and hands them back to every product, so the rule holds no
/// per-call state. A rule generic in `V` also serves `Tape<Dual<S>>`. Generic code
/// that only needs scalar rules uses [`Real::map_scalar`](crate::Real::map_scalar)
/// and [`Real::combine`](crate::Real::combine) instead, which work in every mode.
pub trait ExternalRule<V: TapeValue>: Send + Sync {
    /// Writes `f(inputs)` to `outputs`.
    ///
    /// # Errors
    ///
    /// Any error of the rule, returned unchanged by [`Var::external`].
    fn evaluate(&self, inputs: &[V], outputs: &mut [V]) -> Result<(), SciOxideError>;

    /// Adds `Jᵀ·output_adjoints` to `input_adjoints`, where `J = ∂f/∂x` at `inputs`.
    ///
    /// # Errors
    ///
    /// Any error of the rule, returned unchanged by the product.
    fn pullback(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_adjoints: &[V],
        input_adjoints: &mut [V],
    ) -> Result<(), SciOxideError>;

    /// Writes `J·input_tangents` to `output_tangents`.
    ///
    /// # Errors
    ///
    /// The default returns `CapabilityUnavailable`, which tangent products of a
    /// recording with this rule report.
    fn pushforward(
        &self,
        inputs: &[V],
        outputs: &[V],
        input_tangents: &[V],
        output_tangents: &mut [V],
    ) -> Result<(), SciOxideError> {
        let _ = (inputs, outputs, input_tangents, output_tangents);
        Err(SciOxideError::capability_unavailable(
            Capability::Operation("external rule pushforward"),
        ))
    }
}

impl<'t, V: TapeValue> Var<'t, V> {
    /// Applies `rule` to `inputs` and writes the results to `outputs`.
    ///
    /// With passive inputs the call only evaluates, in `inputs + outputs` values of
    /// scratch allocated on every such call, and the outputs are passive. Otherwise
    /// it records one call statement; recording allocates nothing on a tape that has
    /// recorded the same function before.
    ///
    /// # Errors
    ///
    /// - The errors of [`ExternalRule::evaluate`], unchanged.
    /// - `AllocationFailure` if the evaluation scratch of a first recording or of
    ///   a call with passive inputs cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use scioxide_ad::{ExternalRule, Real, Var, grad};
    /// use scioxide_core::SciOxideError;
    ///
    /// /// `y = x₀·x₁`.
    /// struct Product;
    /// impl ExternalRule<f64> for Product {
    ///     fn evaluate(&self, x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    ///         y[0] = x[0] * x[1];
    ///         Ok(())
    ///     }
    ///     fn pullback(&self, x: &[f64], _: &[f64], w: &[f64], g: &mut [f64]) -> Result<(), SciOxideError> {
    ///         g[0] += w[0] * x[1];
    ///         g[1] += w[0] * x[0];
    ///         Ok(())
    ///     }
    /// }
    /// let rule: Arc<dyn ExternalRule<f64>> = Arc::new(Product);
    /// let g = grad(|x| { let mut y = [Var::constant(0.0)]; Var::external(&rule, x, &mut y)?; Ok(y[0]) }, &[3.0, 4.0])?;
    /// assert_eq!(g.as_slice(), [4.0, 3.0]);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    pub fn external(
        rule: &Arc<dyn ExternalRule<V>>,
        inputs: &[Self],
        outputs: &mut [Self],
    ) -> Result<(), SciOxideError> {
        let recorder = inputs.iter().find_map(|input| input.recorder());
        let (n, m) = (inputs.len(), outputs.len());
        let evaluate = |scratch: &mut [V]| -> Result<(), SciOxideError> {
            let (values, results) = scratch.split_at_mut(n);
            for (value, input) in values.iter_mut().zip(inputs) {
                *value = input.value();
            }
            rule.evaluate(values, results)
        };
        let Some(recorder) = recorder else {
            let mut scratch = owned_scratch(n + m)?;
            evaluate(&mut scratch)?;
            for (output, &value) in outputs.iter_mut().zip(&scratch[n..]) {
                *output = Self::passive_value(value);
            }
            return Ok(());
        };
        if inputs
            .iter()
            .filter_map(|input| input.recorder())
            .any(|other| !core::ptr::eq(other, recorder))
        {
            recorder.fault(super::recorder::MIXED);
            for output in outputs.iter_mut() {
                *output =
                    Self::passive_value(V::constant(<V::Scalar as scioxide_core::RealScalar>::NAN));
            }
            return Ok(());
        }
        let saved = recorder.full_position();
        let publish = |scratch: &mut [V], outputs: &mut [Self]| -> Result<(), SciOxideError> {
            evaluate(scratch)?;
            scratch.iter().for_each(|&value| recorder.push_full(value));
            for (output, &value) in outputs.iter_mut().zip(&scratch[n..]) {
                *output = Self::passive_value(value);
            }
            Ok(())
        };
        recorder.with_scratch(n + m, |scratch| match scratch {
            Some(scratch) => publish(scratch, outputs),
            None => publish(&mut owned_scratch(n + m)?, outputs),
        })?;
        let index = recorder.push_call(Call {
            rule: Rule::External(Arc::clone(rule)),
            inputs: n,
            outputs: m,
            saved,
        });
        let arguments = inputs.iter().map(|input| (input.slot(), None));
        recorder.extended(Some(index), Class::Nonlinear, n, arguments);
        for output in outputs.iter_mut() {
            let slot = recorder.push_word(word::CALL_OUTPUT);
            *output = Self::active(output.value(), slot, recorder);
        }
        Ok(())
    }
}

/// Allocates evaluation scratch on a first recording or for passive inputs.
fn owned_scratch<V: TapeValue>(len: usize) -> Result<Vec<V>, SciOxideError> {
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(len).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("external_rule_scratch"), len)
    })?;
    scratch.resize(len, V::zero());
    Ok(scratch)
}
