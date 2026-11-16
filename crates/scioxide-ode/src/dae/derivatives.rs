//! Strategy-selected residual derivatives, sharing the stiff difference rule.
use super::{DaeNumericalOptions, DaePartition, DaeSystem, evaluate};
use crate::OdeStageBackend;
use crate::safety::finite;
use crate::stiff::jacobian::{self, Differences, SparseColumns};
use crate::termination::{Fault, Tunnel, admit};
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, EvaluationRole, NumericScalar, RealScalar, SciOxideError, SolveContext,
    StructuralPatternView,
};
use scioxide_differentiate::{DerivativeCandidate, DifferenceScheme};
use scioxide_linalg::Matrix;

/// A residual linearization and its partitioned difference policies.
#[derive(Clone, Copy)]
pub(crate) struct Point<'a, T: NumericScalar, P: ?Sized> {
    pub(crate) problem: &'a P,
    pub(crate) partition: &'a DaePartition,
    pub(crate) options: &'a DaeNumericalOptions<T::Real>,
    pub(crate) time: T::Real,
    pub(crate) state: &'a [T],
    pub(crate) derivative: &'a [T],
    pub(crate) base: Option<&'a [T]>,
}

impl<T: NumericScalar, P: DaeSystem<T> + ?Sized> Point<'_, T, P> {
    fn floors(&self, mut visit: impl FnMut(usize, T::Real, T::Real)) -> Result<(), SciOxideError> {
        let mut differential = 0;
        let mut algebraic = 0;
        let count = self.partition.differential();
        for (i, &role) in self.partition.variables().iter().enumerate() {
            let (state, derivative) = if role {
                let state = self
                    .options
                    .differential_error()
                    .absolute()
                    .component(differential, count)?;
                let rate = self
                    .options
                    .initial_derivative_error()
                    .absolute()
                    .component(differential, count)?;
                differential += 1;
                (state, rate)
            } else {
                let state = self
                    .options
                    .algebraic_error()
                    .absolute()
                    .component(algebraic, self.state.len() - count)?;
                algebraic += 1;
                (state, T::Real::ONE)
            };
            visit(i, floor(state), floor(derivative));
        }
        Ok(())
    }
}
fn floor<S: RealScalar>(value: S) -> S {
    if value > S::ZERO { value } else { S::ONE }
}

/// Prepared linear-size residual sampling storage and optional sparse coloring.
pub(crate) struct Blocks<T: NumericScalar> {
    differences: Differences<T>,
    base: Vec<T>,
    state_floors: Vec<T::Real>,
    derivative_floors: Vec<T::Real>,
    columns: Option<SparseColumns>,
}
impl<T: NumericScalar> Blocks<T> {
    pub(crate) fn new(n: usize, backend: &OdeStageBackend) -> Result<Self, SciOxideError> {
        let filled = |value| {
            scioxide_core::backend::try_filled(
                n,
                value,
                scioxide_core::quantity!("dae_difference_storage"),
            )
        };
        Ok(Self {
            differences: Differences::new(n)?,
            base: scioxide_core::backend::try_filled(
                n,
                T::ZERO,
                scioxide_core::quantity!("dae_difference_base"),
            )?,
            state_floors: filled(T::Real::ONE)?,
            derivative_floors: filled(T::Real::ONE)?,
            columns: match backend {
                OdeStageBackend::Sparse { pattern, .. } => Some(SparseColumns::new(pattern)?),
                _ => None,
            },
        })
    }

    pub(crate) fn acquire<P: DaeSystem<T> + ?Sized>(
        &mut self,
        point: Point<'_, T, P>,
        mut output: Output<'_, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        // Selection propagates only `SciOxideError`; denials cross it in the tunnel.
        let tunnel = Tunnel::default();
        let selected = point.options.derivative_strategy().select(|candidate| {
            output.fill(T::from_real(T::Real::NAN));
            match candidate {
                DerivativeCandidate::Analytic => {
                    admit(cx, EvaluationRole::Jacobian).map_err(|fault| tunnel.carry(fault))?;
                    let available = match &mut output {
                        Output::Dense(y, v) => point.problem.derivatives(
                            point.time,
                            point.state,
                            point.derivative,
                            &mut y.view_mut(),
                            &mut v.view_mut(),
                            cx,
                        )?,
                        Output::Sparse(pattern, y, v) => point.problem.sparse_derivatives(
                            point.time,
                            point.state,
                            point.derivative,
                            *pattern,
                            y,
                            v,
                            cx,
                        )?,
                    };
                    if available.is_some() {
                        output.validate()?;
                    }
                    Ok(available)
                }
                DerivativeCandidate::AutomaticDifferentiation => Ok(None),
                DerivativeCandidate::FiniteDifference(scheme) => {
                    point.floors(|i, y, v| {
                        self.state_floors[i] = y;
                        self.derivative_floors[i] = v;
                    })?;
                    if let Some(base) = point.base {
                        self.base.copy_from_slice(base);
                    } else {
                        evaluate(
                            point.problem,
                            point.time,
                            point.state,
                            point.derivative,
                            &mut self.base,
                            cx,
                        )
                        .map_err(|fault| tunnel.carry(fault))?;
                    }
                    for rate in [false, true] {
                        let (state, floors) = if rate {
                            (point.derivative, &self.derivative_floors)
                        } else {
                            (point.state, &self.state_floors)
                        };
                        let dense = matches!(output, Output::Dense(..));
                        let mut sample = |group: &[usize]| {
                            jacobian::group(
                                state,
                                &self.base,
                                floors,
                                scheme,
                                group,
                                0..state.len(),
                                &mut self.differences,
                                cx,
                                |values, out, _, cx| {
                                    let (y, v) = if rate {
                                        (point.state, values)
                                    } else {
                                        (values, point.derivative)
                                    };
                                    evaluate(
                                        point.problem,
                                        point.time,
                                        y,
                                        v,
                                        out.as_mut_slice(),
                                        cx,
                                    )
                                    .map_err(|fault| tunnel.carry(fault))
                                },
                                |column, numerator, offset, replace_zeros| {
                                    match &mut output {
                                        Output::Dense(y, v) => {
                                            let matrix = if rate { v } else { y };
                                            for (row, &value) in numerator.iter().enumerate() {
                                                let entry = &mut matrix.as_mut_slice()
                                                    [row * state.len() + column];
                                                if !replace_zeros || *entry == T::ZERO {
                                                    *entry = value / T::from_real(offset);
                                                }
                                            }
                                        }
                                        Output::Sparse(_, y, v) => {
                                            let columns = self
                                                .columns
                                                .as_ref()
                                                .ok_or_else(missing_pattern)?;
                                            let values = if rate { v } else { y };
                                            for &k in &columns.entries
                                                [columns.starts[column]..columns.starts[column + 1]]
                                            {
                                                if !replace_zeros || values[k] == T::ZERO {
                                                    values[k] = numerator[columns.rows[k]]
                                                        / T::from_real(offset);
                                                }
                                            }
                                        }
                                    }
                                    Ok(())
                                },
                            )?;
                            Ok::<_, SciOxideError>(())
                        };
                        if dense {
                            for column in 0..state.len() {
                                sample(core::slice::from_ref(&column))?;
                            }
                        } else {
                            let columns = self.columns.as_ref().ok_or_else(missing_pattern)?;
                            for group in columns.coloring.groups() {
                                sample(group)?;
                            }
                        }
                    }
                    output.validate()?;
                    Ok(Some(()))
                }
            }
        });
        let _ = tunnel.restore(selected)?;
        Ok(())
    }
}
fn missing_pattern() -> SciOxideError {
    crate::safety::invalid(
        scioxide_core::argument!("workspace"),
        "missing residual derivative coloring",
    )
}

pub(crate) enum Output<'a, T: NumericScalar> {
    Dense(&'a mut Matrix<T>, &'a mut Matrix<T>),
    Sparse(StructuralPatternView<'a>, &'a mut [T], &'a mut [T]),
}
impl<T: NumericScalar> Output<'_, T> {
    fn fill(&mut self, value: T) {
        match self {
            Self::Dense(y, v) => {
                y.as_mut_slice().fill(value);
                v.as_mut_slice().fill(value);
            }
            Self::Sparse(_, y, v) => {
                y.fill(value);
                v.fill(value);
            }
        }
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        let (y, v) = match self {
            Self::Dense(y, v) => (y.as_slice(), v.as_slice()),
            Self::Sparse(_, y, v) => (&**y, &**v),
        };
        finite(y, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(v, scioxide_core::NonFiniteOrigin::Callback)
    }
}

/// Difference a joint residual direction inside operator work that is already counted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product<T: NumericScalar, P: DaeSystem<T> + ?Sized>(
    point: Point<'_, T, P>,
    initialization: bool,
    y_direction: &[T],
    v_direction: &[T],
    output: &mut [T],
    scratch: &mut [T],
    scheme: DifferenceScheme,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), Fault> {
    let n = point.state.len();
    let (y, rest) = scratch.split_at_mut(n);
    let (v, rest) = rest.split_at_mut(n);
    let (forward, backward) = rest.split_at_mut(n);
    let mut magnitude = T::Real::ZERO;
    let mut direction = T::Real::ZERO;
    point.floors(|i, yf, vf| {
        if initialization && point.partition.variables()[i] {
            magnitude = magnitude.max(jacobian::scale(point.derivative[i], vf));
            direction = direction.max(v_direction[i].abs());
        } else {
            magnitude = magnitude.max(jacobian::scale(point.state[i], yf));
            direction = direction.max(y_direction[i].abs());
        }
    })?;
    // A joint residual product needs a resolvable signal across all equations.
    // Use the infinity norm of the Newton unknown and its direction; a tiny
    // algebraic coordinate must not erase increments of unit-sized components.
    let length = direction / magnitude;
    if length == T::Real::ZERO {
        output.fill(T::ZERO);
        return Ok(());
    }
    let step = jacobian::direction_step::<T::Real>(scheme, length)?;
    let mut sample = |step, out: &mut [T], cx: &mut SolveContext<'_, '_>| {
        for i in 0..n {
            y[i] = crate::safety::state_increment(point.state[i], step, y_direction[i])?;
            v[i] = crate::safety::state_increment(point.derivative[i], step, v_direction[i])?;
        }
        evaluate(point.problem, point.time, y, v, out, cx)
    };
    sample(step, forward, cx)?;
    let (reference, width) = if scheme == DifferenceScheme::Central {
        sample(-step, backward, cx)?;
        (&*backward, step + step)
    } else {
        (
            point.base.ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("workspace"),
                    "missing residual product base",
                )
            })?,
            step,
        )
    };
    for ((out, &plus), &minus) in output.iter_mut().zip(forward.iter()).zip(reference) {
        *out = crate::safety::state_difference(plus, minus)? / T::from_real(width);
    }
    Ok(finite(output, scioxide_core::NonFiniteOrigin::Callback)?)
}
