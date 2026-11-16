//! Shared helper functions for PDE solvers.
//!
//! Shared elliptic orchestration over prepared spatial plans.

use alloc::vec::Vec;
use scioxide_linalg::Preconditioning;

use crate::elliptic::EllipticMatrix;
use scioxide_core::backend::checked_mul;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{checked_float_quotient, checked_sum, checked_sum_of_products};
use scioxide_linalg::{IterativeSolveReport, Vector};

use crate::result::PdeDiagnostics;

/// Prepared elliptic system: the eliminated operator, its optional compact
/// bands, the structural constant modes and their mass weights.
pub(crate) struct EllipticSystem<'a, S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) gauge_argument: scioxide_core::Argument,
    pub(crate) matrix: &'a scioxide_sparse::CsrMatrix<S>,
    pub(crate) bands: Option<EllipticMatrix<S>>,
    pub(crate) constant_components: Option<usize>,
    pub(crate) weights: &'a [S],
    pub(crate) direct_limits: Option<scioxide_sparse::SparseLuLimits>,
}

/// Shared elliptic orchestration over the eliminated physical operator.
///
/// The solve lifecycle (`self`, which owns the gauge and inner solver) is named
/// in configuration errors.
pub(crate) fn solve_elliptic_system<S: RealScalar + NumericScalar<Real = S>>(
    system: &EllipticSystem<'_, S>,
    rhs: &Vector<S>,
    supplied: Option<(
        scioxide_core::BlockVectorView<'_, S>,
        scioxide_core::BlockVectorView<'_, S>,
    )>,
    opts: &crate::result::PdeKernelOptions<S>,
    injected: Option<&dyn scioxide_linalg::IterativeLinearSolver<S>>,
    workspace: &mut crate::elliptic::EllipticStorage<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(Vec<S>, PdeDiagnostics<S>), SciOxideError> {
    use scioxide_core::{BlockVectorView, LinearOperator};
    use scioxide_linalg::BorderedSystem;
    let gauge_argument = system.gauge_argument;
    let matrix = system.matrix;
    let n = matrix.rows();
    let constant_mode = system.constant_components.is_some();
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "an elliptic plan needs at least one unknown",
        ));
    }
    if let crate::EllipticGauge::WeightedMean(target) = opts.gauge {
        crate::spatial::finite_callback(target)?;
        if supplied.is_some() {
            return Err(SciOxideError::invalid_input(
                gauge_argument,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "supplied bases require the orthogonal gauge",
            ));
        }
    }
    let known = constant_mode || supplied.is_some();
    if !known && !matches!(opts.gauge, crate::EllipticGauge::Unspecified) {
        return Err(SciOxideError::invalid_input(
            gauge_argument,
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "the assembled problem has no declared or detected nullspace",
        ));
    }
    if known && matches!(opts.gauge, crate::EllipticGauge::Unspecified) {
        return Err(SciOxideError::invalid_input(
            gauge_argument,
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "a known nullspace requires an explicit gauge",
        ));
    }
    let components = system.constant_components.unwrap_or(1);
    let block = n / components;
    let constant = if known && supplied.is_none() {
        let mut values = try_filled(
            checked_mul(
                n,
                components,
                scioxide_core::quantity!("constant_nullspace"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("constant_nullspace"),
        )?;
        let unit = checked_float_quotient(S::ONE, S::try_from_usize_exact(block)?.sqrt())?;
        for component in 0..components {
            let column = &mut values[component * n..(component + 1) * n];
            column[component * block..(component + 1) * block].fill(unit);
        }
        values
    } else {
        Vec::new()
    };
    let bases = match supplied {
        Some(bases) => Some(bases),
        None if constant_mode => {
            let basis = BlockVectorView::new(&constant, n, components)?;
            Some((basis, basis))
        }
        None => None,
    };
    let plan = if let Some((left, right)) = bases {
        // Frobenius norm is an explicit upper bound on the spectral norm.
        let scale = scioxide_core::checked_scaled_norm(matrix.values())?;
        Some(BorderedSystem::build(
            matrix,
            left,
            right,
            scale,
            &scioxide_linalg::BorderedSystemOptions::new()
                .with_validation_tolerance(opts.compatibility_tolerance)
                .with_compatibility_tolerance(opts.compatibility_tolerance),
            &mut *context,
        )?)
    } else {
        None
    };
    let targets = try_filled(
        bases.map_or(0, |(left, _)| left.right_hand_sides()),
        S::ZERO,
        scioxide_core::quantity!("gauge_coordinates"),
    )?;
    let prepared = plan
        .as_ref()
        .map(|p| -> Result<_, SciOxideError> {
            let mut output = Vector::from_vec(try_filled(
                p.augmented().rows(),
                S::ZERO,
                scioxide_core::quantity!("bordered_load"),
            )?);
            p.prepare_rhs_into(
                rhs,
                targets.as_slice(),
                opts.compatibility_policy,
                output.view_mut(),
                &mut scioxide_linalg::LinalgWorkspace::new(),
                &mut *context,
            )?;
            Ok(output)
        })
        .transpose()?;
    let load = prepared.as_ref().unwrap_or(rhs);
    let (mut solution, mut stats, inner_termination) = if let Some(solver) = injected {
        let augmented = plan.as_ref().map(BorderedSystem::augmented);
        let operator: &dyn LinearOperator<S> = augmented
            .as_ref()
            .map_or(matrix as &dyn LinearOperator<S>, |a| a);
        let mut x = Vector::from_vec(try_filled(
            operator.cols(),
            S::ZERO,
            scioxide_core::quantity!("elliptic_solution"),
        )?);
        let report = solver.solve_into(
            operator,
            load.view(),
            Preconditioning::None,
            x.view_mut(),
            &mut workspace.iterative,
            scioxide_linalg::LinalgExecutionContext::from(&mut *context)
                .with_iteration_budget(opts.max_linear_iterations()),
        )?;
        let residual = linear_solve_residual(&report)?;
        let stats = PdeDiagnostics::from_linear_solve(
            solver.algorithm_id().method(),
            report.termination(),
            report.iterations(),
            residual,
            None,
            scioxide_core::argument!("workspace"),
        )?;
        (
            leading_solution_values(&x, n, operator.rows())?,
            stats,
            report.termination(),
        )
    } else {
        let (x, solver_name) = if let Some((left, right)) = supplied {
            (
                solve_supplied_bordered(
                    matrix,
                    left,
                    right,
                    load.as_slice(),
                    system.direct_limits,
                    context,
                )?,
                "bordered-sparse-lu",
            )
        } else if let Some(bands) = &system.bands {
            let cyclic =
                !constant_mode && (bands.corner_upper != S::ZERO || bands.corner_lower != S::ZERO);
            (
                crate::elliptic::direct(
                    bands,
                    &load.as_slice()[..n],
                    system.constant_components,
                    &mut workspace.factor,
                    context,
                )?
                .into_vec(),
                if cyclic {
                    "cyclic-tridiagonal-direct"
                } else {
                    "tridiagonal-direct"
                },
            )
        } else {
            (
                crate::elliptic::sparse_direct(
                    matrix,
                    &load.as_slice()[..n],
                    system.constant_components,
                    system.direct_limits,
                    context,
                )?,
                "sparse-lu-direct",
            )
        };
        let stats = PdeDiagnostics::from_linear_solve(
            solver_name,
            scioxide_linalg::LinearSolveTermination::Converged,
            0,
            S::ZERO,
            None,
            scioxide_core::argument!("workspace"),
        )?
        .with_direct_factorization();
        (x, stats, scioxide_linalg::LinearSolveTermination::Converged)
    };
    if let Some(plan) = plan {
        let mut weighted_gauge = None;
        if let crate::EllipticGauge::WeightedMean(target) = opts.gauge {
            if !constant_mode || supplied.is_some() {
                return Err(SciOxideError::invalid_input(
                    gauge_argument,
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "weighted mean requires the structurally detected constant mode",
                ));
            }
            let target = crate::spatial::finite_callback(target)?;
            let mut defect = S::ZERO;
            for (values, weights) in solution
                .chunks_exact_mut(block)
                .zip(system.weights.chunks_exact(block))
            {
                let weights = normalized_weights(weights)?;
                let mean = checked_sum_of_products(&weights, values)?;
                let shift = scioxide_core::checked_difference(target, mean)?;
                for value in values.iter_mut() {
                    *value = checked_sum(&[*value, shift])?;
                }
                let actual = checked_sum_of_products(&weights, values)?;
                defect = defect.max(<S as NumericScalar>::abs(
                    scioxide_core::checked_difference(actual, target)?,
                ));
            }
            weighted_gauge = Some(defect);
        }
        let diagnostics = plan.diagnose_scoped(
            rhs,
            load,
            solution.as_slice(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            &mut *context,
        )?;
        let gauge_defect = weighted_gauge.unwrap_or(diagnostics.gauge_defect());
        if inner_termination.is_success() {
            let gauge_scale = match opts.gauge {
                crate::EllipticGauge::WeightedMean(target) => <S as NumericScalar>::abs(target),
                _ => S::ZERO,
            };
            if !opts.tolerance().is_residual_ok(gauge_defect, gauge_scale)? {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "the returned solution violates the requested gauge",
                ));
            }
        }
        stats = stats.with_nullspace(
            diagnostics,
            if supplied.is_some() {
                crate::PdeNullspaceSource::Supplied
            } else {
                crate::PdeNullspaceSource::StructuralConstant
            },
            gauge_defect,
        );
    }
    if injected.is_none() || (known && inner_termination.is_success()) {
        let residual = match &system.bands {
            Some(bands) => bands.residual(&load.as_slice()[..n], &solution)?,
            None => crate::elliptic::sparse_residual(matrix, &load.as_slice()[..n], &solution)?,
        };
        if !opts
            .tolerance()
            .is_residual_ok(residual.raw_norm(), residual.scaling().denominator())?
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "physical elliptic equations fail the requested backward-error tolerance",
            ));
        }
        stats = stats.with_linear_residual(residual);
        if known
            && inner_termination == scioxide_linalg::LinearSolveTermination::LeastSquaresConverged
        {
            stats = stats.with_completed();
        }
    }
    Ok((solution, stats))
}

/// Copy the physical unknowns from a possibly augmented solver result.
pub(crate) fn leading_solution_values<S: NumericScalar>(
    solution: &Vector<S>,
    physical_dimension: usize,
    system_dimension: usize,
) -> Result<Vec<S>, SciOxideError> {
    if solution.len() != system_dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(system_dimension),
            scioxide_core::Dimensions::one(solution.len()),
        ));
    }
    let values = solution
        .as_slice()
        .get(..physical_dimension)
        .ok_or_else(|| {
            SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(physical_dimension),
                scioxide_core::Dimensions::one(solution.len()),
            )
        })?;
    let mut result = try_with_capacity(
        physical_dimension,
        scioxide_core::quantity!("elliptic_solution_values"),
    )?;
    result.extend_from_slice(values);
    Ok(result)
}

/// Extract the mandatory diagnostic used by PDE inner-solve reporting.
///
/// A missing residual is reported against the solve lifecycle (`self`), which
/// owns the inner solver.
pub(crate) fn linear_solve_residual<R: RealScalar>(
    report: &IterativeSolveReport<R>,
) -> Result<R, SciOxideError> {
    if report.diagnostic_residual_norm().is_none()
        && let scioxide_linalg::LinearSolveTermination::ExecutionStopped(reason) =
            report.termination()
    {
        return Err(SciOxideError::execution_stopped(reason));
    }
    report.diagnostic_residual_norm().ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "inner solver omitted residual diagnostics",
        )
    })
}

fn normalized_weights<S: RealScalar + NumericScalar<Real = S>>(
    weights: &[S],
) -> Result<Vec<S>, SciOxideError> {
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "weight is non-finite",
        ));
    }
    let mut absolute_weights = try_with_capacity(
        weights.len(),
        scioxide_core::quantity!("normalized_weights"),
    )?;
    absolute_weights.extend(
        weights
            .iter()
            .map(|&weight| <S as NumericScalar>::abs(weight)),
    );
    let norm = checked_sum(&absolute_weights)?;
    if norm == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "weight vector has zero norm",
        ));
    }

    let mut normalized = try_with_capacity(
        weights.len(),
        scioxide_core::quantity!("normalized_weights"),
    )?;
    for &weight in weights {
        normalized.push(checked_float_quotient(weight, norm)?);
    }
    Ok(normalized)
}

/// Sparse bordered solve; natural columns retain the path plus its small dense border.
fn solve_supplied_bordered<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &scioxide_sparse::CsrMatrix<S>,
    left: scioxide_core::BlockVectorView<'_, S>,
    right: scioxide_core::BlockVectorView<'_, S>,
    rhs: &[S],
    limits: Option<scioxide_sparse::SparseLuLimits>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<Vec<S>, SciOxideError> {
    use scioxide_sparse::{
        DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseLu, SparseLuLimits,
        SparseLuSymbolic, SparseOperation, SparseOrdering, SparseSolveOptions,
    };
    let n = matrix.rows();
    let k = left.right_hand_sides();
    let dimension =
        scioxide_core::backend::checked_add(n, k, scioxide_core::quantity!("bordered_dimension"))?;
    let width =
        scioxide_core::backend::checked_add(k, 3, scioxide_core::quantity!("bordered_width"))?;
    let entries = scioxide_core::backend::checked_product(
        &[n, width, 8],
        scioxide_core::quantity!("bordered_entries"),
    )?;
    let bytes = scioxide_core::backend::checked_product(
        &[entries, 128],
        scioxide_core::quantity!("bordered_bytes"),
    )?;
    let border_entries = scioxide_core::backend::checked_product(
        &[2, n, k],
        scioxide_core::quantity!("border_entries"),
    )?;
    let capacity = scioxide_core::backend::checked_add(
        matrix.nnz(),
        border_entries,
        scioxide_core::quantity!("bordered_capacity"),
    )?;
    let mut builder = SparseBuilder::with_capacity(dimension, dimension, capacity)?;
    for (i, j, v) in matrix.iter() {
        builder.push(i, j, v)?;
    }
    for col in 0..k {
        for (row, (&l, &r)) in left.column(col)?.iter().zip(right.column(col)?).enumerate() {
            builder.push(row, n + col, l)?;
            builder.push(n + col, row, r)?;
        }
    }
    let bordered = builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Drop)?;
    let limits = limits.unwrap_or_else(|| SparseLuLimits::new(entries, bytes));
    let symbolic = SparseLuSymbolic::analyze(
        bordered.structural_pattern(),
        SparseOrdering::Natural,
        limits,
    )?;
    let factor = SparseLu::factor_scoped(
        symbolic,
        bordered.view(),
        SparseSolveOptions::new(),
        &mut *context,
    )?;
    let mut solution = try_filled(
        dimension,
        S::ZERO,
        scioxide_core::quantity!("bordered_solution"),
    )?;
    factor.solve_into(
        rhs,
        SparseOperation::Normal,
        SparseSolveOptions::new(),
        &mut solution,
        &mut factor.workspace(1)?,
        &mut *context,
    )?;
    solution.truncate(n);
    Ok(solution)
}
