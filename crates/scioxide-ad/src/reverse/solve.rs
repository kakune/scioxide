//! Structured statements recorded by tape variables: inner products and solves.

use alloc::vec::Vec;

use scioxide_core::{RealScalar, SciOxideError};

use super::call::{Call, Rule};
use super::recorder::{MIXED, Recorder};
use super::word;
use super::{TapeValue, Var};
use crate::Real;
use crate::real::sealed::{Linear, Token};
use crate::rules::Class;
use crate::structured::Factor;

/// The common recorder of the active values, or `Err(())` for two recorders.
pub(crate) fn common<'t, V: TapeValue>(
    values: impl Iterator<Item = Var<'t, V>>,
) -> Result<Option<&'t Recorder<'t, V>>, ()> {
    let mut found: Option<&'t Recorder<'t, V>> = None;
    for value in values {
        if let Some(r) = value.recorder() {
            match found {
                None => found = Some(r),
                Some(q) if core::ptr::eq(q, r) => {}
                Some(q) => {
                    q.fault(MIXED);
                    r.fault(MIXED);
                    return Err(());
                }
            }
        }
    }
    Ok(found)
}

fn values<V: TapeValue>(vars: &[Var<'_, V>]) -> Result<Vec<V>, SciOxideError> {
    let mut out = Vec::new();
    out.try_reserve_exact(vars.len()).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("solve_scratch"), vars.len())
    })?;
    out.extend(vars.iter().map(|v| v.value()));
    Ok(out)
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Linear<S> for Var<'_, V> {
    fn dot<I: Iterator<Item = (Self, Self)> + Clone>(_: Token, pairs: I) -> Self {
        if pairs.clone().next().is_none() {
            return Var::constant(S::ZERO);
        }
        let value = pairs.clone().fold(V::constant(S::NEG_ZERO), |sum, (a, b)| {
            sum + a.value() * b.value()
        });
        let recorder = common(pairs.clone().flat_map(|(a, b)| [a, b]));
        let Ok(recorder) = recorder else {
            return Var::constant(S::NAN);
        };
        let Some(r) = recorder else {
            return Var::lift(value);
        };
        let count = pairs.clone().count();
        let lefts = pairs.clone().map(|(a, b)| (a.slot(), Some(b.value())));
        let rights = pairs.map(|(a, b)| (b.slot(), Some(a.value())));
        let slot = r.extended(None, Class::Bilinear, 2 * count, lefts.chain(rights));
        Var::active(value, slot, r)
    }

    /// A solve on tape variables nested in a forward type (`Dual<Var>`, `Jet<Var>`):
    /// one call statement per component, as [`Linear::solve`].
    fn solve_factored(
        token: Token,
        factor: &Factor<S>,
        matrix: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        if transpose {
            // Only a tape pullback solves with `Aᵀ`, and its values are never `Var`.
            return Err(SciOxideError::accounting_invariant(
                "a transposed solve never runs on tape variables",
                None,
            ));
        }
        let copy =
            scioxide_core::backend::try_copy_slice(rhs, scioxide_core::quantity!("solve_scratch"))?;
        <Self as Linear<S>>::solve(token, matrix, &copy, rhs, factor.order())
    }

    fn solve(
        token: Token,
        matrix: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        order: usize,
    ) -> Result<(), SciOxideError> {
        let a = values(matrix)?;
        let mut x = values(rhs)?;
        let factor = Factor::new(&a, order)?;
        V::solve_factored(token, &factor, &a, &mut x, false)?;
        let recorder = common(matrix.iter().chain(rhs).copied());
        let Ok(recorder) = recorder else {
            out.fill(Var::constant(S::NAN));
            return Ok(());
        };
        let Some(r) = recorder else {
            for (target, &value) in out.iter_mut().zip(&x) {
                *target = Var::lift(value);
            }
            return Ok(());
        };
        let saved = r.full_position();
        a.iter()
            .copied()
            .chain(rhs.iter().map(|v| v.value()))
            .chain(x.iter().copied())
            .for_each(|v| r.push_full(v));
        let inputs = matrix.len() + rhs.len();
        let index = r.push_call(Call {
            rule: Rule::Solve(factor),
            inputs,
            outputs: x.len(),
            saved,
        });
        let arguments = matrix.iter().chain(rhs).map(|v| (v.slot(), None));
        r.extended(Some(index), Class::Nonlinear, inputs, arguments);
        for (target, &value) in out.iter_mut().zip(&x) {
            let slot = r.push_word(word::CALL_OUTPUT);
            *target = Var::active(value, slot, r);
        }
        Ok(())
    }
}
