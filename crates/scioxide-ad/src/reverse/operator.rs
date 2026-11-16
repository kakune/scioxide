//! Matrix-free derivative operators over recordings.

use core::cell::Cell;

use scioxide_core::{
    AdjointOperator, Capability, ExecutionContext, LinearOperator, NonFiniteOrigin, RealScalar,
    SciOxideError,
};

use super::recorder::{Columns, Counts};
use super::tape::Tape;
use super::{Var, sweep};
use crate::Dual;

fn non_finite(origin: NonFiniteOrigin) -> SciOxideError {
    SciOxideError::non_finite(origin, "a derivative operator met a non-finite value")
}

/// The Jacobian `J` of a [`Linearization`](crate::Linearization) as a
/// [`LinearOperator`] and [`AdjointOperator`].
///
/// `apply_into` is a tangent sweep and `apply_adjoint_into` a reverse sweep over
/// the shared recording. The buffers come from the caller's workspace
/// (`apply_workspace_len` covers one value per statement word and the scratch of
/// call statements), so applications allocate nothing, except that
/// `apply_adjoint_into` through the call statement of
/// [`solve_into`](crate::solve_into) allocates the scratch of its transposed solve.
/// Inputs and results must be finite (`NonFinite` otherwise), as the core operator
/// protocol requires.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Tape, TapeOptions};
/// use scioxide_core::{AdjointOperator, ExecutionContext, LinearOperator};
/// // y = (x₀x₁, x₀ + x₁) at (2, 3): J = [[3, 2], [1, 1]].
/// let mut tape = Tape::<f64>::new(TapeOptions::new());
/// let cx = ExecutionContext::sequential();
/// let linear = tape.linearize(
///     |x, y| {
///         y.extend([x[0] * x[1], x[0] + x[1]]);
///         Ok(())
///     },
///     &[2.0, 3.0],
///     cx,
/// )?;
/// let j = linear.jacobian_operator()?;
/// let mut work = vec![0.0; j.apply_workspace_len()?];
/// let mut out = [0.0; 2];
/// j.apply_into(&[1.0, 0.0], &mut out, &mut work)?;
/// assert_eq!(out, [3.0, 1.0]);
/// j.apply_adjoint_into(&[1.0, 0.0], &mut out, &mut work)?;
/// assert_eq!(out, [3.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct JacobianOperator<'a, S: RealScalar> {
    tape: &'a Tape<S>,
}

impl<'a, S: RealScalar> JacobianOperator<'a, S> {
    pub(crate) fn new(tape: &'a Tape<S>) -> Result<Self, SciOxideError> {
        let frozen = tape.frozen();
        if (0..frozen.calls.len())
            .any(|index| frozen.with_call(index, |call| call.is_some_and(|c| c.needs_tape())))
        {
            return Err(SciOxideError::capability_unavailable(
                Capability::Operation("operator over a checkpointed loop"),
            ));
        }
        Ok(Self { tape })
    }

    fn workspace_len(&self) -> Result<usize, SciOxideError> {
        self.tape
            .counts
            .words
            .checked_add(2 * self.tape.counts.scratch)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("operator_workspace"))
            })
    }

    fn split<'w>(
        &self,
        workspace: &'w mut [S],
    ) -> Result<(&'w mut [S], &'w mut [S]), SciOxideError> {
        crate::support::check_len(workspace.len(), self.workspace_len()?)?;
        Ok(workspace.split_at_mut(self.tape.counts.words))
    }
}

impl<S: RealScalar> LinearOperator<S> for JacobianOperator<'_, S> {
    fn rows(&self) -> usize {
        self.tape.output_slots.len()
    }

    fn cols(&self) -> usize {
        self.tape.point_len
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.workspace_len()
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::support::check_len(input.len(), self.cols())?;
        crate::support::check_len(output.len(), self.rows())?;
        let (work, call_work) = self.split(workspace)?;
        if input.iter().any(|v| !v.is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Input));
        }
        work[0] = S::ZERO;
        work[1..=input.len()].copy_from_slice(input);
        let frozen = self.tape.frozen();
        sweep::tangent(
            &frozen,
            Counts::default(),
            self.tape.counts.words,
            work,
            call_work,
            ExecutionContext::sequential(),
        )?;
        let value = |row: usize| {
            work.get(self.tape.output_slots[row] as usize)
                .copied()
                .unwrap_or(S::ZERO)
        };
        if (0..output.len()).any(|row| !value(row).is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Output));
        }
        for (row, target) in output.iter_mut().enumerate() {
            *target = value(row);
        }
        Ok(())
    }
}

impl<S: RealScalar> AdjointOperator<S> for JacobianOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.workspace_len()
    }

    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::support::check_len(input.len(), self.rows())?;
        crate::support::check_len(output.len(), self.cols())?;
        let (work, call_work) = self.split(workspace)?;
        if input.iter().any(|v| !v.is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Input));
        }
        work.fill(S::ZERO);
        for (&slot, &seed) in self.tape.output_slots.iter().zip(input) {
            if let Some(target) = work.get_mut(slot as usize) {
                *target += seed;
            }
        }
        let frozen = self.tape.frozen();
        sweep::reverse(
            &frozen,
            self.tape.counts,
            0,
            work,
            call_work,
            ExecutionContext::sequential(),
        )?;
        let gradient = &work[1..=output.len()];
        if gradient.iter().any(|v| !v.is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Output));
        }
        output.copy_from_slice(gradient);
        Ok(())
    }
}

/// The Hessian `H` of a scalar function as a [`LinearOperator`].
///
/// Each application records the function on its `Tape<Dual<S>>` with the
/// direction as seeds and sweeps it once (forward over reverse), so `H·v` costs a
/// small constant times a gradient. The tape is held in a cell and taken for each
/// application; the control flow depends on the point only, so the capacity found
/// at construction is final and applications allocate nothing.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Tape, TapeOptions};
/// use scioxide_core::{ExecutionContext, LinearOperator};
/// // f = x₀²x₁ at (1, 2): H = [[2x₁, 2x₀], [2x₀, 0]] = [[4, 2], [2, 0]].
/// let mut tape = Tape::<Dual<f64>>::new(TapeOptions::new());
/// let cx = ExecutionContext::sequential();
/// let h = tape.hessian_operator(|x| Ok(x[0] * x[0] * x[1]), &[1.0, 2.0], cx)?;
/// let mut out = [0.0; 2];
/// h.apply_into(&[1.0, 0.0], &mut out, &mut [])?;
/// assert_eq!(out, [4.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct HessianOperator<'a, S: RealScalar, F> {
    tape: Cell<Option<&'a mut Tape<Dual<S>>>>,
    function: F,
    point: &'a [S],
    context: ExecutionContext<'a>,
}

impl<S: RealScalar, F> core::fmt::Debug for HessianOperator<'_, S, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HessianOperator")
            .field("order", &self.point.len())
            .finish_non_exhaustive()
    }
}

impl<'a, S: RealScalar, F> HessianOperator<'a, S, F>
where
    F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
{
    pub(crate) fn new(
        tape: &'a mut Tape<Dual<S>>,
        function: F,
        point: &'a [S],
        context: ExecutionContext<'a>,
    ) -> Result<Self, SciOxideError> {
        let operator = Self {
            tape: Cell::new(Some(tape)),
            function,
            point,
            context,
        };
        // One application at the zero direction sizes the tape.
        operator.with_tape(|tape| {
            tape.hvp_seeded(&operator.function, point, |_| S::ZERO, |_, _| {}, context)
        })?;
        Ok(operator)
    }

    fn with_tape<T>(
        &self,
        function: impl FnOnce(&mut Tape<Dual<S>>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        // Restore the exclusive borrow on both normal return and unwinding.
        struct Restore<'c, 't, S: RealScalar> {
            cell: &'c Cell<Option<&'t mut Tape<Dual<S>>>>,
            tape: Option<&'t mut Tape<Dual<S>>>,
        }
        impl<S: RealScalar> Drop for Restore<'_, '_, S> {
            fn drop(&mut self) {
                self.cell.set(self.tape.take());
            }
        }
        let tape = self.tape.take().ok_or_else(|| {
            SciOxideError::accounting_invariant("the Hessian operator tape is in use", None)
        })?;
        let mut restore = Restore {
            cell: &self.tape,
            tape: Some(tape),
        };
        match restore.tape.as_deref_mut() {
            Some(tape) => function(tape),
            None => Err(SciOxideError::accounting_invariant(
                "missing borrowed Hessian tape",
                None,
            )),
        }
    }
}

impl<S: RealScalar, F> LinearOperator<S> for HessianOperator<'_, S, F>
where
    F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
{
    fn rows(&self) -> usize {
        self.point.len()
    }

    fn cols(&self) -> usize {
        self.point.len()
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::support::check_len(input.len(), self.cols())?;
        crate::support::check_len(output.len(), self.rows())?;
        crate::support::check_len(workspace.len(), 0)?;
        if input.iter().any(|v| !v.is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Input));
        }
        let mut finite = true;
        self.with_tape(|tape| {
            tape.hvp_seeded(
                &self.function,
                self.point,
                |k| input[k],
                |_, product: &[Dual<S>]| {
                    finite = product.iter().all(|v| v.derivative().is_finite());
                    if finite {
                        for (target, value) in output.iter_mut().zip(product) {
                            *target = value.derivative();
                        }
                    }
                },
                self.context,
            )
        })?;
        if finite {
            Ok(())
        } else {
            Err(non_finite(NonFiniteOrigin::Output))
        }
    }
}
