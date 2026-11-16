//! Call statements: rules the tape applies as one statement.

use alloc::sync::Arc;

use scioxide_core::SciOxideError;

use super::TapeValue;
use super::external::ExternalRule;
use crate::real::sealed::Token;
use crate::structured::{Factor, subtract_product};

/// One entry of the call table of a recording.
///
/// The inputs and outputs of the call are saved in the full partial column at
/// `saved`, inputs first.
pub(crate) struct Call<V: TapeValue> {
    pub(crate) rule: Rule<V>,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) saved: usize,
}

/// The rule of a call statement.
pub(crate) enum Rule<V: TapeValue> {
    External(Arc<dyn ExternalRule<V>>),
    /// `X = A⁻¹·B`: inputs are `A` (`n²`) then `B`, outputs `X`.
    Solve(Factor<V::Scalar>),
    /// A checkpointed loop: inputs are the initial state then the parameters,
    /// outputs the final state.
    Loop {
        body: Arc<dyn super::loops::ErasedLoop<V>>,
        steps: usize,
        schedule: crate::CheckpointSchedule,
    },
}

fn scratch<V: Copy>(values: &[V]) -> Result<alloc::vec::Vec<V>, SciOxideError> {
    scioxide_core::backend::try_copy_slice(values, scioxide_core::quantity!("solve_scratch"))
}

impl<V: TapeValue> Call<V> {
    /// Values the call keeps in the full partial column.
    #[inline]
    pub(crate) const fn saved_len(&self) -> usize {
        self.inputs + self.outputs
    }

    /// Adds `Jᵀ·output_adjoints` to `input_adjoints`.
    pub(crate) fn pullback(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_adjoints: &[V],
        input_adjoints: &mut [V],
    ) -> Result<(), SciOxideError> {
        match &self.rule {
            Rule::External(rule) => rule.pullback(inputs, outputs, output_adjoints, input_adjoints),
            Rule::Solve(factor) => {
                // λ = A⁻ᵀ·x̄, b̄ += λ, Ā −= λ·xᵀ.
                let n = factor.order();
                let (a, _) = inputs.split_at(n * n);
                let columns = outputs.len() / n.max(1);
                let mut lambda = scratch(output_adjoints)?;
                V::solve_factored(Token(()), factor, a, &mut lambda, true)?;
                let (a_bar, b_bar) = input_adjoints.split_at_mut(n * n);
                for (target, &l) in b_bar.iter_mut().zip(&lambda) {
                    *target += l;
                }
                for i in 0..n {
                    for k in 0..n {
                        let mut sum = a_bar[i * n + k];
                        for column in 0..columns {
                            sum -= lambda[i * columns + column] * outputs[k * columns + column];
                        }
                        a_bar[i * n + k] = sum;
                    }
                }
                Ok(())
            }
            Rule::Loop {
                body,
                steps,
                schedule,
            } => super::loops::pullback(
                body,
                *steps,
                schedule,
                inputs,
                output_adjoints,
                input_adjoints,
            ),
        }
    }

    /// Writes `J·input_tangents` to `output_tangents`.
    pub(crate) fn pushforward(
        &self,
        inputs: &[V],
        outputs: &[V],
        input_tangents: &[V],
        output_tangents: &mut [V],
    ) -> Result<(), SciOxideError> {
        match &self.rule {
            Rule::External(rule) => {
                rule.pushforward(inputs, outputs, input_tangents, output_tangents)
            }
            Rule::Solve(factor) => {
                // ẋ = A⁻¹·(ḃ − Ȧ·x).
                let n = factor.order();
                let (a, _) = inputs.split_at(n * n);
                let (a_dot, b_dot) = input_tangents.split_at(n * n);
                output_tangents.copy_from_slice(b_dot);
                subtract_product(output_tangents, a_dot, outputs, n, false);
                V::solve_factored(Token(()), factor, a, output_tangents, false)
            }
            Rule::Loop { body, steps, .. } => {
                super::loops::pushforward(body, *steps, inputs, input_tangents, output_tangents)
            }
        }
    }

    /// Whether products of the call need state owned by a tape.
    pub(crate) const fn needs_tape(&self) -> bool {
        match self.rule {
            Rule::External(_) | Rule::Solve(_) => false,
            Rule::Loop { .. } => true,
        }
    }
}
