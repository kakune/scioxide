//! Products of one recording at one point.

use scioxide_core::{
    ColoringSchedule, ExecutionContext, NonFiniteOrigin, RealScalar, SciOxideError,
    StructuralPattern,
};

use super::operator::JacobianOperator;
use super::tape::{Taken, Tape};
use super::{TapeValue, pattern};
use crate::{JacobianMode, Mask, NonFinitePolicy, Real};

/// The recording of a vector function `y = f(x)` at one point, for any number of
/// derivative products there.
///
/// It borrows its [`Tape`] mutably. `vjp_into` and `jvp_into` are one sweep over
/// the recording, costing a small constant times the evaluation; `jacobian_into`
/// runs one sweep per output or per input, and `sparse_jacobian_into` one per
/// color. Products allocate nothing after their first use on the tape, except a
/// reverse sweep through the call statement of [`solve_into`](crate::solve_into),
/// whose transposed solve allocates its scratch. Outputs are unchanged when a
/// product fails.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, Tape, TapeOptions};
/// use scioxide_core::ExecutionContext;
///
/// let cx = ExecutionContext::sequential();
/// let mut tape = Tape::<f64>::new(TapeOptions::new());
/// let mut linear = tape.linearize(|x, y| { y.push(x[0] * x[1]); y.push(x[0].exp()); Ok(()) }, &[2.0, 3.0], cx)?;
/// let mut jacobian = [0.0; 4];
/// linear.jacobian_into(&mut jacobian, cx)?;
/// assert_eq!(jacobian, [3.0, 2.0, 2.0_f64.exp(), 0.0]);
/// let mut pullback = [0.0; 2];
/// linear.vjp_into(&[1.0, 0.0], &mut pullback, cx)?;
/// assert_eq!(pullback, [3.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct Linearization<'a, V: TapeValue> {
    tape: &'a mut Tape<V>,
}

impl<'a, V: TapeValue> Linearization<'a, V> {
    pub(crate) fn new(tape: &'a mut Tape<V>) -> Self {
        Self { tape }
    }

    /// Returns the number of inputs `n`.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.tape.point_len
    }

    /// Returns the number of outputs `m`.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.tape.output_slots.len()
    }

    /// Returns the output values `f(x)`.
    #[must_use]
    pub fn values(&self) -> &[V] {
        &self.tape.output_values
    }

    /// Writes the reverse product `Jᵀ·cotangent` (one entry per input) to `out`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `cotangent` has not `m` or `out` not `n` entries.
    /// - `NonFinite` (origin `Output`) under [`NonFinitePolicy::Error`].
    /// - `ExecutionStopped` if `context` is cancelled; the errors of external rules.
    pub fn vjp_into(
        &mut self,
        cotangent: &[V],
        out: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.tape.vjp_into(cotangent, out, context)
    }

    /// Writes the tangent product `J·direction` (one entry per output) to `out`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `direction` has not `n` or `out` not `m` entries.
    /// - `NonFinite` (origin `Output`) under [`NonFinitePolicy::Error`].
    /// - `CapabilityUnavailable` if an external rule has no pushforward;
    ///   `ExecutionStopped` if `context` is cancelled.
    pub fn jvp_into(
        &mut self,
        direction: &[V],
        out: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.tape.jvp_into(direction, out, context)
    }

    /// Writes the dense Jacobian `J` (row-major, outputs × inputs) to `out`.
    ///
    /// [`JacobianMode::Auto`] runs one reverse sweep per output when `m < n` and one
    /// tangent sweep per input otherwise.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out.len() != m·n`.
    /// - `AllocationFailure` if the staging buffer cannot grow on first use.
    /// - The errors of [`Linearization::vjp_into`] or [`Linearization::jvp_into`].
    pub fn jacobian_into(
        &mut self,
        out: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let (m, n) = (self.outputs(), self.inputs());
        let len = m
            .checked_mul(n)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("jacobian")))?;
        crate::support::check_len(out.len(), len)?;
        let tape = &mut *self.tape;
        let mut stage = tape.take_buffer(Taken::Stage);
        let outcome = jacobian(tape, &mut stage, m, n, context);
        if outcome.is_ok() {
            out.copy_from_slice(&stage[..len]);
        }
        tape.restore(stage, Taken::Stage);
        outcome
    }

    /// Returns the structural pattern of `J` (outputs × inputs): entry `(i, j)` is
    /// present if output `i` depends on input `j` through recorded statements.
    ///
    /// A pattern is valid for the points that take the same branches.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if the pattern cannot be stored.
    pub fn jacobian_pattern(&self) -> Result<StructuralPattern, SciOxideError> {
        pattern::jacobian(
            &self.tape.frozen(),
            self.tape.point_len,
            &self.tape.output_slots,
        )
    }

    /// Returns the structural pattern of the Hessian of the single output
    /// (inputs × inputs), by nonlinear interaction domains (Walther 2008).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` unless there is exactly one output.
    /// - `AllocationFailure` if the pattern cannot be stored.
    ///
    /// # References
    ///
    /// A. Walther, "Computing sparse Hessians with automatic differentiation",
    /// *ACM Trans. Math. Softw.* 34(1), 2008.
    pub fn hessian_pattern(&self) -> Result<StructuralPattern, SciOxideError> {
        crate::support::check_len(self.outputs(), 1)?;
        pattern::hessian(
            &self.tape.frozen(),
            self.tape.point_len,
            self.tape.output_slots[0],
        )
    }
}

fn jacobian<V: TapeValue>(
    tape: &mut Tape<V>,
    stage: &mut alloc::vec::Vec<V>,
    m: usize,
    n: usize,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    tape.grow_taken(
        stage,
        Taken::Stage,
        m * n,
        scioxide_core::quantity!("jacobian"),
    )?;
    let reverse = match tape.options.jacobian_mode() {
        JacobianMode::Auto => m < n,
        JacobianMode::Forward => false,
        JacobianMode::Reverse => true,
    };
    if reverse {
        for row in 0..m {
            // A sweep shorter than 1,024 words never checks, so check per sweep.
            context.check()?;
            tape.clean_work();
            let slot = tape.output_slots[row] as usize;
            if let Some(target) = tape.work.get_mut(slot) {
                *target += V::one();
            }
            tape.sweep_reverse(context)?;
            stage[row * n..(row + 1) * n].copy_from_slice(&tape.work[1..=n]);
            tape.clear_inputs(n);
        }
    } else {
        for column in 0..n {
            context.check()?;
            tape.clean = false;
            tape.work[..=n].fill(V::zero());
            tape.work[column + 1] = V::one();
            tape.sweep_tangent(context)?;
            for row in 0..m {
                let slot = tape.output_slots[row] as usize;
                stage[row * n + column] = tape.work.get(slot).copied().unwrap_or_else(V::zero);
            }
        }
    }
    tape.check_results(&stage[..m * n])
}

impl<S: RealScalar> Linearization<'_, S> {
    /// Writes the entries of a sparse `J` in the CSR order of `coloring`'s pattern,
    /// with one tangent sweep per color.
    ///
    /// `coloring` comes from `scioxide_sparse::derivative_coloring` applied to
    /// [`Linearization::jacobian_pattern`] or to any pattern that contains it. The
    /// values equal those of a dense Jacobian built by tangent sweeps
    /// ([`JacobianMode::Forward`](crate::JacobianMode::Forward)) bit for bit; a dense
    /// Jacobian from reverse sweeps adds the contributions to an entry in another
    /// order and agrees to rounding.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `coloring`, `UnsupportedConfiguration`) unless
    ///   the coloring is a derivative-column coloring
    ///   (`ColoringKind::DerivativeColumns`).
    /// - `DimensionMismatch` if the pattern is not outputs × inputs or `values` has
    ///   not one entry per pattern entry.
    /// - The errors of [`Linearization::jvp_into`].
    pub fn sparse_jacobian_into(
        &mut self,
        coloring: &ColoringSchedule,
        values: &mut [S],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        crate::support::check_coloring(coloring)?;
        let pattern = coloring.pattern();
        let (m, n) = (self.outputs(), self.inputs());
        if pattern.rows() != m || pattern.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                [m, n],
                [pattern.rows(), pattern.cols()],
            ));
        }
        let entries = pattern.col_indices().len();
        crate::support::check_len(values.len(), entries)?;
        let tape = &mut *self.tape;
        let mut stage = tape.take_buffer(Taken::Stage);
        let outcome = (|| {
            tape.grow_taken(
                &mut stage,
                Taken::Stage,
                entries,
                scioxide_core::quantity!("sparse_jacobian"),
            )?;
            let colors = coloring.colors();
            for (color, members) in coloring.groups().enumerate() {
                context.check()?;
                tape.clean = false;
                tape.work[..=n].fill(S::ZERO);
                for &column in members {
                    tape.work[column + 1] = S::ONE;
                }
                tape.sweep_tangent(context)?;
                for row in 0..m {
                    let slot = tape.output_slots[row] as usize;
                    let derivative = tape.work.get(slot).copied().unwrap_or(S::ZERO);
                    let range = pattern.row_ptr()[row]..pattern.row_ptr()[row + 1];
                    for entry in range {
                        if colors[pattern.col_indices()[entry]] == color {
                            stage[entry] = derivative;
                        }
                    }
                }
            }
            if tape.options.non_finite_policy() == NonFinitePolicy::Error
                && stage[..entries].iter().any(|v| !Real::is_finite(*v).all())
            {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Output,
                    "a derivative is not finite",
                ));
            }
            Ok(())
        })();
        if outcome.is_ok() {
            values.copy_from_slice(&stage[..entries]);
        }
        tape.restore(stage, Taken::Stage);
        outcome
    }

    /// Returns `J` as a matrix-free operator for iterative solvers: `apply_into` is a
    /// tangent sweep and `apply_adjoint_into` a reverse sweep in the caller's
    /// workspace, so applications allocate nothing, except a reverse sweep through
    /// the call statement of [`solve_into`](crate::solve_into).
    ///
    /// # Errors
    ///
    /// `CapabilityUnavailable` if the recording contains a checkpointed loop, whose
    /// pullback records its steps on a tape of its own.
    pub fn jacobian_operator(&self) -> Result<JacobianOperator<'_, S>, SciOxideError> {
        JacobianOperator::new(self.tape)
    }
}
