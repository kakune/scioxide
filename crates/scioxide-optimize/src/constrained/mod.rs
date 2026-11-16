//! Solvers for smooth nonlinear programs with equality, inequality and bound
//! constraints.
//!
//! Every solver minimizes a [`ConstrainedProblem`](crate::ConstrainedProblem):
//! an objective, its [`Constraints`](crate::Constraints) and, optionally, a
//! [`LagrangianHessian`]. The problem carries the constraints, so one
//! configured solver can solve many problems.
//!
//! | Solver | Method | Derivatives |
//! |---|---|---|
//! | [`Sqp`] | sequential quadratic programming with an ℓ₁ merit line search | first |
//! | [`InteriorPoint`] | primal-dual log-barrier method with a merit line search | first, optional Lagrangian Hessian |
//! | [`AugmentedLagrangian`] | method of multipliers with BFGS subproblems | first |
//! | [`PenaltyMethod`] | quadratic penalty with BFGS subproblems | first |
//! | [`Cobyla`] | linear interpolation in a trust region | none |
//! | [`Cobyqa`] | quadratic interpolation in a trust region | none |

mod augmented_lagrangian;
mod cobyla;
pub use cobyla::{
    Cobyla, CobylaDiagnostics, CobylaOptions, CobylaWorkspace, CobylaWorkspaceRequirements,
};
mod cobyqa;
mod derivative_free;
pub use cobyqa::{
    Cobyqa, CobyqaDiagnostics, CobyqaOptions, CobyqaWorkspace, CobyqaWorkspaceRequirements,
};
mod interior_point;
pub use interior_point::{
    InteriorPoint, InteriorPointDiagnostics, InteriorPointOptions, InteriorPointState,
    InteriorPointWorkspace, InteriorPointWorkspaceRequirements, LagrangianHessian,
};
mod nonlinear;
mod outer;
mod penalty;
pub(crate) mod snapshot;
pub(crate) mod sqp;
mod storage;

pub use augmented_lagrangian::{
    AugmentedLagrangian, AugmentedLagrangianOptions, AugmentedLagrangianWorkspace,
    AugmentedLagrangianWorkspaceRequirements,
};
pub use penalty::{
    PenaltyMethod, PenaltyMethodWorkspace, PenaltyMethodWorkspaceRequirements, PenaltyOptions,
};
pub use sqp::{
    DenseSqpSubproblem, Sqp, SqpDiagnostics, SqpOptions, SqpState, SqpSubproblem,
    SqpSubproblemObjective, SqpSubproblemProvider, SqpSubproblemReport, SqpWorkspace,
    SqpWorkspaceRequirements,
};

use crate::KktProvider;
use scioxide_core::backend::checked_add;
use scioxide_core::{ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};

use crate::result::{OptimizeResult, OptimizeTermination};
use crate::utils::{commit_optimize_result, norm2};
pub(crate) use storage::ConstrainedStorage;

fn select_certificate_constraints<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut ConstrainedStorage<S>,
    activation: S,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
        LinalgExecutionOptions, RankPolicy,
    };
    workspace.active_constraints.clear();
    let mut count = 0;
    for row in 0..workspace.constraint_values.len() {
        if row < workspace.equality_count
            || NumericScalar::abs(workspace.constraint_values[row]) <= activation
        {
            workspace.certificate_candidates[count] = row;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(());
    }
    let n = workspace.dimension;
    workspace.objective_hessian.try_resize_flat(n, count)?;
    for (column, &row) in workspace.certificate_candidates[..count].iter().enumerate() {
        let coefficients = &workspace.constraint_jacobian.as_slice()[row * n..(row + 1) * n];
        let norm = norm2(coefficients)?;
        for (i, &value) in coefficients.iter().enumerate() {
            workspace.objective_hessian.as_mut_slice()[i * count + column] =
                if norm == <S as scioxide_core::AlgebraicScalar>::ZERO {
                    norm
                } else {
                    scioxide_core::checked_float_quotient(value, norm)?
                };
        }
    }
    session.invoke(scioxide_core::EvaluationRole::Factorization, || {
        HouseholderDecomposition::factor_into(
            &workspace.objective_hessian,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut workspace.certificate_qr,
            &mut workspace.certificate_work,
            session.execution()?,
        )
    })?;
    let rank = workspace
        .certificate_qr
        .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
        .rank();
    workspace.active_constraints.extend(
        workspace.certificate_qr.permutation()[..rank]
            .iter()
            .map(|&column| workspace.certificate_candidates[column]),
    );
    Ok(())
}

fn solve_certificate_multipliers<S>(
    workspace: &mut ConstrainedStorage<S>,
    activation: S,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    select_certificate_constraints(workspace, activation, session)?;
    workspace.multipliers.fill(zero);
    loop {
        let active = workspace.active_constraints.len();
        if active == 0 {
            return Ok(());
        }
        let order = checked_add(
            workspace.dimension,
            active,
            scioxide_core::quantity!("constraint_certificate_kkt_order"),
        )?;
        workspace.kkt.try_resize_flat(order, order)?;
        workspace.factor_rhs.try_resize(order, zero)?;
        workspace.factor_solution.try_resize(order, zero)?;
        let n = workspace.dimension;
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut workspace.kkt, order)?;
        assembly.diagonal(0, n, |_| Ok(one))?;
        assembly.coupling(n, 0, active, n, |i, j| {
            Ok(workspace.constraint_jacobian.as_slice()[workspace.active_constraints[i] * n + j])
        })?;
        for (rhs, &g) in workspace.factor_rhs.as_mut_slice()[..n]
            .iter_mut()
            .zip(&workspace.objective_gradient)
        {
            *rhs = -g;
        }
        workspace.factor_rhs.as_mut_slice()[n..].fill(zero);
        let provider = workspace.certificate_provider.as_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "unprepared certificate provider",
            )
        })?;
        session.invoke(scioxide_core::EvaluationRole::Factorization, || {
            provider.factor(workspace.kkt.view(), session.execution()?)
        })?;
        session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
            provider.solve_into(
                workspace.factor_rhs.as_slice(),
                workspace.factor_solution.as_mut_slice(),
                session.execution()?,
            )
        })?;
        // Near-zero equality equations need an absolute rounding allowance;
        // the independently checked original-model certificate remains decisive.
        let scale = workspace
            .factor_rhs
            .as_slice()
            .iter()
            .chain(workspace.factor_solution.as_slice())
            .fold(one, |a, &v| a.max(NumericScalar::abs(v)));
        let floor = scioxide_core::checked_float_product(
            S::EPSILON * S::try_from_usize_exact(order)? * S::try_from_usize_exact(16)?,
            scale,
        )?
        .min(activation);
        crate::kkt::verify_residual_with_floor(
            workspace.kkt.view(),
            workspace.factor_rhs.as_slice(),
            workspace.factor_solution.as_slice(),
            floor,
        )?;
        let most_negative = workspace
            .active_constraints
            .iter()
            .enumerate()
            .filter(|(_, row)| **row >= workspace.equality_count)
            .fold(None, |minimum, (active_row, &constraint_row)| {
                let multiplier =
                    workspace.factor_solution.as_slice()[workspace.dimension + active_row];
                match minimum {
                    Some((_, _, best)) if best <= multiplier => minimum,
                    _ => Some((active_row, constraint_row, multiplier)),
                }
            });
        if let Some((active_row, _, multiplier)) = most_negative
            && multiplier < -activation
        {
            workspace.active_constraints.remove(active_row);
            continue;
        }
        for (active_row, &constraint_row) in workspace.active_constraints.iter().enumerate() {
            workspace.multipliers[constraint_row] =
                workspace.factor_solution.as_slice()[workspace.dimension + active_row];
        }
        return Ok(());
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_constrained_without_gradient<S>(
    workspace: &mut ConstrainedStorage<S>,
    output: &mut OptimizeResult<S>,
    value: S,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    commit_optimize_result(
        output,
        &workspace.x,
        value,
        None,
        iterations,
        termination,
        message,
    )?;

    Ok(())
}
