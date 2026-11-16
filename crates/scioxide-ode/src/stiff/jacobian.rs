//! Jacobian evaluation following [`DerivativeStrategy`] on every stage backend.
//!
//! Sources are tried in the strategy's order and a source returning `Ok(None)` falls
//! through; provider errors propagate. Dense storage accepts analytic, automatic and
//! finite-difference Jacobians. Sparse storage accepts analytic values and colored
//! finite differences; `OdeSystem` exposes no sparse automatic derivative, so that source
//! is unavailable there. Operators accept analytic or finite-difference products.
//!
//! There is one finite-difference rule. A perturbation moves the real coordinates of a
//! group of structurally independent columns (one column per group when dense, a
//! derivative coloring of the pattern when sparse), reuses `f(y)` as the forward base,
//! and divides by the representable offset. Steps are `√ε·sⱼ` for forward and `∛ε·sⱼ` for
//! central differences with the component scale `sⱼ = max(|yⱼ|, aⱼ)`, where `aⱼ` is the
//! component's absolute error tolerance (one when it is zero). Flooring by the
//! tolerance rather than by one keeps perturbations of tiny components, such as
//! Robertson's intermediate species, within their own magnitude.
//! Native complex states are differentiated along real coordinates, which requires a
//! holomorphic right-hand side. Unresolved zero entries are retried once at the
//! maximum component scale, preserving quotients resolved at the local scale.
//! Directional products use one step `δ` along `v` with
//! `δ·|vⱼ| ≤ √ε·sⱼ` (or `∛ε·sⱼ`) in every component.
use crate::safety::{finite, invalid};
use crate::stepper::StepRequest;
use crate::traits::{OdeSystem, evaluate_rhs_staged_slice};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    ColoringSchedule, EvaluationRole, NumericScalar, RealScalar, SciOxideError, SolveContext,
    StructuralPattern,
};
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy, DifferenceScheme};
use scioxide_linalg::{Matrix, Vector};

/// Reusable perturbation storage for finite differences of an `n`-dimensional system.
#[derive(Debug)]
pub(crate) struct Differences<T: NumericScalar> {
    perturbed: Vector<T>,
    forward: Vector<T>,
    backward: Vector<T>,
    candidate: Vector<T>,
    offsets: Vec<T::Real>,
}

impl<T: NumericScalar> Differences<T> {
    pub(crate) fn new(n: usize) -> Result<Self, SciOxideError> {
        Self::rectangular(n, n)
    }
    pub(crate) fn rectangular(inputs: usize, outputs: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            perturbed: Vector::zeros(inputs)?,
            forward: Vector::zeros(outputs)?,
            backward: Vector::zeros(outputs)?,
            candidate: Vector::zeros(outputs)?,
            offsets: try_filled(
                inputs,
                T::Real::ZERO,
                scioxide_core::quantity!("difference_offsets"),
            )?,
        })
    }
}

/// Column structure of a sparse Jacobian for colored differences.
#[derive(Debug)]
pub(crate) struct SparseColumns {
    pub(crate) pattern: StructuralPattern,
    pub(crate) coloring: ColoringSchedule,
    /// Entries of column `j` are `entries[starts[j]..starts[j + 1]]` (CSR positions).
    pub(crate) starts: Vec<usize>,
    pub(crate) entries: Vec<usize>,
    pub(crate) rows: Vec<usize>,
}

impl SparseColumns {
    pub(crate) fn new(pattern: &StructuralPattern) -> Result<Self, SciOxideError> {
        let p = pattern.view();
        let (n, nnz) = (p.cols(), p.col_indices().len());
        let mut starts = try_filled(n + 1, 0usize, scioxide_core::quantity!("jacobian_columns"))?;
        let mut rows = try_filled(nnz, 0usize, scioxide_core::quantity!("jacobian_rows"))?;
        for (row, bounds) in p.row_ptr().windows(2).enumerate() {
            rows[bounds[0]..bounds[1]].fill(row);
            for &column in &p.col_indices()[bounds[0]..bounds[1]] {
                starts[column + 1] += 1;
            }
        }
        for column in 0..n {
            starts[column + 1] += starts[column];
        }
        let mut next = starts.clone();
        let mut entries = try_filled(nnz, 0usize, scioxide_core::quantity!("jacobian_entries"))?;
        for (k, &column) in p.col_indices().iter().enumerate() {
            entries[next[column]] = k;
            next[column] += 1;
        }
        Ok(Self {
            pattern: pattern.clone(),
            coloring: scioxide_sparse::derivative_coloring(p)?,
            starts,
            entries,
            rows,
        })
    }

    /// Number of structural entries.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// `y += J·x` for CSR values on this pattern, accumulated by row.
    pub(crate) fn multiply<T: NumericScalar>(
        &self,
        values: &[T],
        input: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        let p = self.pattern.view();
        for (row, bounds) in p.row_ptr().windows(2).enumerate() {
            output[row] = (bounds[0]..bounds[1]).fold(T::ZERO, |sum, k| {
                sum + values[k] * input[p.col_indices()[k]]
            });
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}

pub(crate) fn nominal<S: RealScalar>(scheme: DifferenceScheme) -> S {
    match scheme {
        DifferenceScheme::Central => S::CBRT_EPSILON,
        _ => S::SQRT_EPSILON,
    }
}

/// Validate the common directional finite-difference increment.
pub(crate) fn direction_step<S: RealScalar>(
    scheme: DifferenceScheme,
    length: S,
) -> Result<S, SciOxideError> {
    let step = nominal::<S>(scheme) / length;
    if !step.is_finite() || step <= S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "finite-difference direction step is not representable",
        ));
    }
    Ok(step)
}

/// Component scale `max(|y|, floor)` of one difference step.
pub(crate) fn scale<T: NumericScalar>(value: T, floor: T::Real) -> T::Real {
    value.abs().max(floor)
}

pub(crate) fn shifted<T: NumericScalar>(value: T, step: T::Real) -> (T, T::Real) {
    let real = value.real_component() + step;
    (
        T::from_components(real, value.imaginary_component()),
        real - value.real_component(),
    )
}

/// Difference one group of independent columns and hand each column's quotient
/// numerator and representable offset to `scatter`.
#[allow(clippy::too_many_arguments)]
fn difference_group<S, T, P>(
    system: &P,
    time: S,
    state: &[T],
    base: &[T],
    floors: &[S],
    scheme: DifferenceScheme,
    group: &[usize],
    work: &mut Differences<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
    scatter: impl FnMut(usize, &[T], S, bool) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    self::group(
        state,
        base,
        floors,
        scheme,
        group,
        0..state.len(),
        work,
        cx,
        |state, output, candidate, cx| {
            evaluate_rhs_staged_slice(system, time, state, output, candidate, request, cx)
        },
        scatter,
    )
}

/// Difference independent columns using an evaluation the caller has already counted.
/// Zero quotients retry at the maximum scale within `retry_domain`; parameter
/// coordinates may use their own domain rather than inheriting state units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn group<T: NumericScalar>(
    state: &[T],
    base: &[T],
    floors: &[T::Real],
    scheme: DifferenceScheme,
    columns: &[usize],
    retry_domain: core::ops::Range<usize>,
    work: &mut Differences<T>,
    cx: &mut SolveContext<'_, '_>,
    mut evaluate: impl FnMut(
        &[T],
        &mut Vector<T>,
        &mut Vector<T>,
        &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>,
    mut scatter: impl FnMut(usize, &[T], T::Real, bool) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let central = matches!(scheme, DifferenceScheme::Central);
    let global = state[retry_domain.clone()]
        .iter()
        .zip(&floors[retry_domain])
        .fold(T::Real::ZERO, |maximum, (&value, &floor)| {
            maximum.max(scale(value, floor))
        });
    let can_retry = columns
        .iter()
        .any(|&column| scale(state[column], floors[column]) < global);
    for retry in [false, true] {
        work.perturbed.as_mut_slice().copy_from_slice(state);
        for &column in columns {
            let size = if retry {
                global
            } else {
                scale(state[column], floors[column])
            };
            let (value, offset) = shifted(state[column], nominal::<T::Real>(scheme) * size);
            work.perturbed.as_mut_slice()[column] = value;
            work.offsets[column] = offset;
        }
        evaluate(
            work.perturbed.as_slice(),
            &mut work.forward,
            &mut work.candidate,
            cx,
        )?;

        if central {
            for &column in columns {
                let size = if retry {
                    global
                } else {
                    scale(state[column], floors[column])
                };
                let (value, offset) = shifted(state[column], -nominal::<T::Real>(scheme) * size);
                work.perturbed.as_mut_slice()[column] = value;
                work.offsets[column] -= offset;
            }
            evaluate(
                work.perturbed.as_slice(),
                &mut work.backward,
                &mut work.candidate,
                cx,
            )?;
        }
        let reference = if central {
            work.backward.as_slice()
        } else {
            base
        };
        for (difference, &other) in work.forward.as_mut_slice().iter_mut().zip(reference) {
            *difference = crate::safety::state_difference(*difference, other)?;
        }
        for &column in columns {
            let offset = work.offsets[column];
            if !offset.is_finite() || offset <= T::Real::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "finite-difference step is not representable",
                ));
            }
            scatter(column, work.forward.as_slice(), offset, retry)?;
        }
        // A tiny coordinate can disappear inside a residual containing much larger
        // terms (e.g. y0 + y1 + y2 - 1). Retry unresolved zero entries at the
        // system scale, preserving all quotients resolved at the local scale.
        if retry
            || !can_retry
            || work
                .forward
                .as_slice()
                .iter()
                .all(|value| *value != T::ZERO)
        {
            break;
        }
    }
    Ok(())
}

/// Acquire a dense Jacobian at `(time, state)` with `f(state) = base`.
/// Returns the selected derivative source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense<S, T, P>(
    system: &P,
    strategy: DerivativeStrategy,
    time: S,
    state: &[T],
    base: &[T],
    floors: &[S],
    output: &mut Matrix<T>,
    work: &mut Differences<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<DerivativeCandidate, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    let n = state.len();
    let (source, ()) = strategy.select(|candidate| {
        output.as_mut_slice().fill(T::ZERO);
        let available = match candidate {
            DerivativeCandidate::Analytic => {
                crate::stepper::invoke(EvaluationRole::Jacobian, request, cx, |cx| {
                    system.jacobian(time, state, &mut output.view_mut(), cx)
                })?
                .is_some()
            }
            DerivativeCandidate::AutomaticDifferentiation => {
                crate::stepper::invoke(EvaluationRole::Jacobian, request, cx, |cx| {
                    system.automatic_jacobian(time, state, &mut output.view_mut(), cx)
                })?
                .is_some()
            }
            DerivativeCandidate::FiniteDifference(scheme) => {
                for column in 0..n {
                    difference_group(
                        system,
                        time,
                        state,
                        base,
                        floors,
                        scheme,
                        core::slice::from_ref(&column),
                        work,
                        request,
                        cx,
                        |column, numerator, offset, replace_zeros| {
                            for (row, &value) in numerator.iter().enumerate() {
                                let entry = &mut output.as_mut_slice()[row * n + column];
                                if !replace_zeros || *entry == T::ZERO {
                                    *entry = value / T::from_real(offset);
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                true
            }
        };
        crate::traits::validate_provider_matrix(output, n)?;
        Ok(available.then_some(()))
    })?;
    Ok(source)
}

/// Acquire sparse Jacobian values on `columns.pattern`.
/// Returns the selected derivative source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse<S, T, P>(
    system: &P,
    strategy: DerivativeStrategy,
    time: S,
    state: &[T],
    base: &[T],
    floors: &[S],
    columns: &SparseColumns,
    values: &mut [T],
    work: &mut Differences<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<DerivativeCandidate, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    let (source, ()) = strategy.select(|candidate| {
        values.fill(T::from_real(S::NAN));
        let available = match candidate {
            DerivativeCandidate::Analytic => {
                crate::stepper::invoke(EvaluationRole::Jacobian, request, cx, |cx| {
                    system.sparse_jacobian(time, state, columns.pattern.view(), values, cx)
                })?
                .is_some()
            }
            DerivativeCandidate::AutomaticDifferentiation => false,
            DerivativeCandidate::FiniteDifference(scheme) => {
                for group in columns.coloring.groups() {
                    difference_group(
                        system,
                        time,
                        state,
                        base,
                        floors,
                        scheme,
                        group,
                        work,
                        request,
                        cx,
                        |column, numerator, offset, replace_zeros| {
                            for &k in
                                &columns.entries[columns.starts[column]..columns.starts[column + 1]]
                            {
                                if !replace_zeros || values[k] == T::ZERO {
                                    values[k] = numerator[columns.rows[k]] / T::from_real(offset);
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                true
            }
        };
        if available {
            finite(values, scioxide_core::NonFiniteOrigin::Callback)?;
        }
        Ok(available.then_some(()))
    })?;
    Ok(source)
}

/// Evaluate `f` inside nested product work, counted once in `cx`.
fn rhs_into<S, T, P>(
    system: &P,
    time: S,
    state: &[T],
    output: &mut [T],
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    output.fill(T::from_real(S::NAN));
    crate::stepper::nested_rhs(system, time, state, output, request, cx)?;
    finite(output, scioxide_core::NonFiniteOrigin::Callback)
}

/// Linearization point and source memory of one operator Jacobian block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProductPoint<'a, S, T> {
    pub(crate) time: S,
    pub(crate) state: &'a [T],
    pub(crate) base: &'a [T],
    pub(crate) floors: &'a [S],
}

/// `J·direction` at `point` through the remembered or newly selected source.
/// `scratch` holds the provider scratch followed by `3n` difference values. The
/// product runs inside operator work already counted in `cx`: difference RHS
/// evaluations are counted there and analytic products charge their own work.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product<S, T, P>(
    system: &P,
    strategy: DerivativeStrategy,
    source: &core::cell::Cell<Option<DerivativeCandidate>>,
    point: ProductPoint<'_, S, T>,
    direction: &[T],
    output: &mut [T],
    scratch: &mut [T],
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    let n = point.state.len();
    let Some(split) = scratch.len().checked_sub(3 * n) else {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            "Jacobian product scratch is too small",
        ));
    };
    let (provider, differences) = scratch.split_at_mut(split);
    let mut evaluate = |candidate: DerivativeCandidate| -> Result<Option<()>, SciOxideError> {
        match candidate {
            DerivativeCandidate::Analytic => {
                output.fill(T::from_real(S::NAN));
                let available = request.trace.provider(system.jacobian_product(
                    point.time,
                    point.state,
                    direction,
                    output,
                    provider,
                    cx,
                ))?;
                if available.is_some() {
                    finite(output, scioxide_core::NonFiniteOrigin::Callback)?;
                }
                Ok(available)
            }
            DerivativeCandidate::AutomaticDifferentiation => Ok(None),
            DerivativeCandidate::FiniteDifference(scheme) => {
                let (perturbed, rest) = differences.split_at_mut(n);
                let (forward, backward) = rest.split_at_mut(n);
                let length = direction
                    .iter()
                    .zip(point.state.iter().zip(point.floors))
                    .fold(S::ZERO, |maximum, (&v, (&y, &floor))| {
                        maximum.max(v.abs() / scale(y, floor))
                    });
                if length == S::ZERO {
                    output.fill(T::ZERO);
                    return Ok(Some(()));
                }
                let step = direction_step::<S>(scheme, length)?;
                for ((value, &y), &v) in perturbed.iter_mut().zip(point.state).zip(direction) {
                    *value = crate::safety::state_increment(y, step, v)?;
                }
                rhs_into(system, point.time, perturbed, forward, request, cx)?;

                let (reference, width) = if matches!(scheme, DifferenceScheme::Central) {
                    for ((value, &y), &v) in perturbed.iter_mut().zip(point.state).zip(direction) {
                        *value = crate::safety::state_increment(y, -step, v)?;
                    }
                    rhs_into(system, point.time, perturbed, backward, request, cx)?;

                    (&*backward, step + step)
                } else {
                    (point.base, step)
                };
                for ((out, &plus), &minus) in output.iter_mut().zip(&*forward).zip(reference) {
                    *out = crate::safety::state_difference(plus, minus)? / T::from_real(width);
                }
                finite(output, scioxide_core::NonFiniteOrigin::Callback)?;
                Ok(Some(()))
            }
        }
    };
    match source.get() {
        Some(candidate) => {
            evaluate(candidate)?.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("system"),
                    "a selected Jacobian product source became unavailable",
                )
            })?;
        }
        None => source.set(Some(strategy.select(&mut evaluate)?.0)),
    }
    Ok(())
}

/// Acquire a partial time derivative at fixed state through the remembered or newly
/// selected source, so an unavailable analytic callback is asked once per epoch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn time_derivative<S, T, P>(
    system: &P,
    strategy: DerivativeStrategy,
    source: &mut Option<DerivativeCandidate>,
    time: S,
    state: &[T],
    base: &[T],
    output: &mut Vector<T>,
    forward: &mut Vector<T>,
    backward: &mut Vector<T>,
    candidate: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    let mut evaluate = |selected: DerivativeCandidate| -> Result<Option<()>, SciOxideError> {
        output.as_mut_slice().fill(T::from_real(S::NAN));
        let available = match selected {
            DerivativeCandidate::Analytic => {
                crate::stepper::invoke(EvaluationRole::Jacobian, request, cx, |cx| {
                    system.time_derivative(time, state, output.as_mut_slice(), cx)
                })?
            }
            DerivativeCandidate::AutomaticDifferentiation => None,
            DerivativeCandidate::FiniteDifference(scheme) => {
                let step = nominal::<S>(scheme) * NumericScalar::abs(time).max(S::ONE);
                let (plus, offset) = shifted(time, step);
                crate::safety::checked_step_endpoint(time, step)?;
                evaluate_rhs_staged_slice(system, plus, state, forward, candidate, request, cx)?;

                let (reference, width) = if scheme == DifferenceScheme::Central {
                    let (minus, negative) = shifted(time, -step);
                    crate::safety::checked_step_endpoint(time, -step)?;
                    evaluate_rhs_staged_slice(
                        system, minus, state, backward, candidate, request, cx,
                    )?;

                    (backward.as_slice(), offset - negative)
                } else {
                    (base, offset)
                };
                for ((out, &plus), &minus) in output
                    .as_mut_slice()
                    .iter_mut()
                    .zip(forward.as_slice())
                    .zip(reference)
                {
                    *out = crate::safety::state_difference(plus, minus)? / T::from_real(width);
                }
                Some(())
            }
        };
        if available.is_some() {
            finite(output.as_slice(), scioxide_core::NonFiniteOrigin::Callback)?;
        }
        Ok(available)
    };
    match *source {
        Some(selected) => {
            evaluate(selected)?.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("system"),
                    "a selected time-derivative source became unavailable",
                )
            })?;
        }
        None => *source = Some(strategy.select(&mut evaluate)?.0),
    }
    Ok(())
}
