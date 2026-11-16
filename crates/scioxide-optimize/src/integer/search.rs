//! Best-bound search over the model, relaxation and cut kernels.
use super::*;
use crate::context::SolveSession;
use crate::{LinearConstraintSense as Sense, ProgrammingCertificate};

use model::Row;
use validation::{candidate, distance, global_bound, result};

type Session<'a, 'b, 'c> = SolveSession<'a, 'b, 'c>;

fn push<S: RealScalar>(queue: &mut Vec<MipNode<S>>, node: MipNode<S>) {
    queue.push(node);
    let mut child = queue.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if compare(&queue[parent], &queue[child]) != core::cmp::Ordering::Greater {
            break;
        }
        queue.swap(parent, child);
        child = parent;
    }
}
fn pop<S: RealScalar>(queue: &mut Vec<MipNode<S>>) -> Option<MipNode<S>> {
    if queue.is_empty() {
        return None;
    }
    let result = queue.swap_remove(0);
    let mut parent = 0;
    while parent * 2 + 1 < queue.len() {
        let left = parent * 2 + 1;
        let right = left + 1;
        let child = if right < queue.len() && compare(&queue[right], &queue[left]).is_lt() {
            right
        } else {
            left
        };
        if !compare(&queue[child], &queue[parent]).is_lt() {
            break;
        }
        queue.swap(parent, child);
        parent = child;
    }
    Some(result)
}

fn prunable<S: RealScalar + NumericScalar<Real = S>>(
    incumbent: Option<S>,
    bound: Option<S>,
    t: scioxide_core::ResidualTolerance<S>,
) -> Result<bool, SciOxideError> {
    let (Some(c), Some(b)) = (incumbent, bound) else {
        return Ok(false);
    };
    Ok(checked_difference(c, b)?
        <= t.threshold(
            NumericScalar::abs(c)
                .max(NumericScalar::abs(b))
                .max(<S as scioxide_core::AlgebraicScalar>::ONE),
        )?)
}
fn close<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    bound: Option<S>,
) {
    if let Some(b) = bound {
        s.closed_bound = Some(s.closed_bound.map_or(b, |old| old.min(b)));
    }
}
fn contradictory<S: RealScalar>(bounds: &[(Option<S>, Option<S>)]) -> bool {
    bounds
        .iter()
        .any(|&(l, u)| matches!((l,u),(Some(l),Some(u)) if l>u))
}
fn tighten<S: RealScalar + NumericScalar<Real = S>>(
    bounds: &mut [(Option<S>, Option<S>)],
    row: &Row<S>,
) {
    if let [(j, _)] = row.entries.as_slice() {
        let (l, u) = &mut bounds[*j];
        match row.sense {
            Sense::GreaterEqual => *l = Some(l.map_or(row.rhs, |l| l.max(row.rhs))),
            Sense::LessEqual => *u = Some(u.map_or(row.rhs, |u| u.min(row.rhs))),
            Sense::Equal => {
                *l = Some(row.rhs);
                *u = Some(row.rhs);
            }
        }
    }
}
fn path_bounds<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    path: Option<usize>,
) -> Result<(), SciOxideError> {
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("model"), "missing model"))?;
    crate::storage::resize(&mut s.bounds_scratch, model.bounds.len(), (None, None))?;
    s.bounds_scratch.copy_from_slice(&model.bounds);
    let mut cursor = path;
    while let Some(i) = cursor {
        let link = &s.row_links[i];
        if link.row.provenance.is_none() {
            tighten(&mut s.bounds_scratch, &link.row);
        }
        cursor = link.parent;
    }
    Ok(())
}
fn set_path_bounds<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    path: Option<usize>,
    p: &mut LpProblem<S>,
) -> Result<(), SciOxideError> {
    path_bounds(s, path)?;
    p.update_bounds(&s.bounds_scratch)
}
fn node_lp<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    path: Option<usize>,
) -> Result<Option<LpProblem<S>>, SciOxideError> {
    path_bounds(s, path)?;
    if contradictory(&s.bounds_scratch) {
        return Ok(None);
    }
    s.next_cut_ids.clear();
    let mut cursor = path;
    while let Some(i) = cursor {
        let link = &s.row_links[i];
        if link.row.provenance.is_some() {
            scioxide_core::backend::try_reserve(
                &mut s.next_cut_ids,
                1,
                scioxide_core::quantity!("mip_active_cuts"),
            )?;
            s.next_cut_ids.push(i);
        }
        cursor = link.parent;
    }
    s.next_cut_ids.sort_unstable();
    let changed = s.cut_ids != s.next_cut_ids;
    let mut p = if changed || s.lp.is_none() {
        let model = s
            .model
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("model"), "missing model"))?;
        let p = model.lp(s.next_cut_ids.iter().map(|&i| &s.row_links[i].row))?;
        if changed {
            s.relaxation = None;
            s.verifier = None;
        }
        core::mem::swap(&mut s.cut_ids, &mut s.next_cut_ids);
        p
    } else {
        s.lp.take()
            .ok_or_else(|| invalid(scioxide_core::argument!("lp"), "missing retained LP"))?
    };
    p.update_bounds(&s.bounds_scratch)?;
    Ok(Some(p))
}

fn admit<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    extra: usize,
    limit: usize,
) -> Result<bool, SciOxideError> {
    let bytes = checked_add(
        s.bytes(),
        extra,
        scioxide_core::quantity!("mip_storage_admission"),
    )?;
    if bytes > limit {
        return Ok(false);
    }
    s.stats.peak_search_bytes = s.stats.peak_search_bytes.max(bytes);
    Ok(true)
}

pub(super) fn run<S, R>(
    solver: &BranchAndBound<S>,
    p: &MipProblem<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    lp: &mut R,
    session: &Session<'_, '_, '_>,
) -> Result<MipResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    if let Err(e) = session.check() {
        return match session.stopped() {
            Some(stop) => result(s, stop, "search stopped before preparation"),
            None => Err(e),
        };
    }
    // Bound presolve and its staging before copying the original matrix.
    let n = p.relaxation().dimension();
    let rows = p.relaxation().constraints().map_or(0, |c| c.len());
    let entries = p
        .relaxation()
        .constraints()
        .map_or(0, |c| match c.coefficients() {
            crate::CoefficientMatrix::Dense(a) => a.as_slice().len(),
            crate::CoefficientMatrix::Sparse(a) => a.nnz(),
        });
    let initial = scioxide_core::backend::checked_product(
        &[
            checked_add(
                checked_add(n, rows, scioxide_core::quantity!("mip_model_size"))?,
                checked_add(entries, 1, scioxide_core::quantity!("mip_model_size"))?,
                scioxide_core::quantity!("mip_model_size"),
            )?,
            256,
        ],
        scioxide_core::quantity!("mip_model_size"),
    )?;
    if !admit(s, initial, solver.limits.max_search_bytes)? {
        return result(s, OptimizeTermination::MemoryLimit, "presolve storage cap");
    }
    s.model = Some(model::Model::prepare(p, solver.presolve)?);
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing prepared model"))?;
    if model.infeasible {
        return result(
            s,
            OptimizeTermination::Infeasible,
            "root model contradiction",
        );
    }
    if model.objective.is_empty() {
        s.incumbent = candidate(
            p,
            model.map.reconstruct(&[])?,
            false,
            solver.relaxation_tolerances.feasibility(),
        )?;
        return result(
            s,
            if s.incumbent.is_some() {
                OptimizeTermination::Converged
            } else {
                OptimizeTermination::Stagnation
            },
            "fully fixed original model checked",
        );
    }
    scioxide_core::backend::try_reserve(
        &mut s.costs,
        model.objective.len(),
        scioxide_core::quantity!("mip_pseudocosts"),
    )?;
    s.costs
        .extend((0..model.objective.len()).map(|_| Pseudocost {
            sums: [<S as scioxide_core::AlgebraicScalar>::ZERO; 2],
            counts: [0; 2],
        }));
    s.active = Some(MipNode {
        id: 0,
        depth: 0,
        lower_bound: None,
        rows: None,
        branch: None,
    });
    s.next_id = 1;
    loop {
        if s.active.is_none() {
            if s.queue.is_empty() {
                return result(
                    s,
                    if s.incumbent.is_some() {
                        OptimizeTermination::Converged
                    } else {
                        OptimizeTermination::Infeasible
                    },
                    "search frontier exhausted",
                );
            }
            s.active = pop(&mut s.queue);
        }
        if gap_met(s, solver.relaxation_tolerances.gap())? {
            return result(
                s,
                OptimizeTermination::Converged,
                "verified numerical global gap met",
            );
        }
        if s.stats.nodes >= solver.max_nodes.get() {
            return result(
                s,
                OptimizeTermination::NodeLimit,
                "tree node budget reached",
            );
        }
        // A tree iteration belongs to the MIP scope; LP/probe iterations belong
        // to their child scopes and must not consume this allowance.
        if let Some(stop) = session.iteration()? {
            return result(s, stop, "parent tree-iteration limit");
        }
        if let Err(e) = session.node() {
            return match session.stopped() {
                Some(stop) => result(s, stop, "parent resource limit"),
                None => Err(e),
            };
        }
        s.stats.nodes = checked_add(s.stats.nodes, 1, scioxide_core::quantity!("mip_nodes"))?;
        let mut node = s
            .active
            .take()
            .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing active node"))?;
        let outcome = process(solver, p, s, &mut node, lp, session);
        match outcome {
            Ok(Some((stop, message))) => {
                s.active = Some(node);
                return result(s, stop, message);
            }
            Ok(None) => {}
            Err(e) => {
                s.active = Some(node);
                return match session.stopped() {
                    Some(stop) => result(s, stop, "nested resource stop"),
                    None => Err(e),
                };
            }
        }
    }
}
fn compare<S: RealScalar>(a: &MipNode<S>, b: &MipNode<S>) -> core::cmp::Ordering {
    match (a.lower_bound, b.lower_bound) {
        (None, None) => core::cmp::Ordering::Equal,
        (None, Some(_)) => core::cmp::Ordering::Less,
        (Some(_), None) => core::cmp::Ordering::Greater,
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(core::cmp::Ordering::Equal),
    }
    .then(a.id.cmp(&b.id))
}
fn gap_met<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &MipSearchState<S, W>,
    t: scioxide_core::ResidualTolerance<S>,
) -> Result<bool, SciOxideError> {
    prunable(s.incumbent.as_ref().map(|c| c.fun), global_bound(s), t)
}

#[allow(clippy::too_many_arguments)]
fn solve_lp<S, R>(
    p: &LpProblem<S>,
    lp: &mut R,
    iterations: Budget,
    output: &mut crate::ProgrammingResult<S>,
    solver: &BranchAndBound<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    let workspace_bound = lp.workspace_bytes(p)?;
    if workspace_bound > solver.limits.max_relaxation_bytes {
        output.termination = OptimizeTermination::MemoryLimit;
        return Ok(());
    }
    let staging = scioxide_core::backend::checked_product(
        &[
            checked_add(
                workspace_bound,
                contracts::model_payload_bound(p)?,
                scioxide_core::quantity!("lp_staging"),
            )?,
            4,
        ],
        scioxide_core::quantity!("lp_staging"),
    )?;
    if !admit(s, staging, solver.limits.max_search_bytes)? {
        output.termination = OptimizeTermination::MemoryLimit;
        return Ok(());
    }
    s.stats.relaxations = checked_add(
        s.stats.relaxations,
        1,
        scioxide_core::quantity!("mip_lp_count"),
    )?;
    let identity = crate::context::local_identity(
        scioxide_core::algorithm_id!("optimize", "MipRelaxation", 1),
        p.dimension(),
        &(s.stats.relaxations, iterations),
    );
    if s.relaxation.is_none() {
        s.relaxation = Some(lp.workspace(p)?);
    }
    if s.verifier.is_none() {
        s.verifier = Some(crate::VerificationWorkspace::for_lp(
            &crate::LpVerificationProblem::from_problem(p),
        )?);
    }
    let warm = s.relaxation.as_mut().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("workspace"),
            "missing relaxation workspace",
        )
    })?;
    let verification = s.verifier.as_mut().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("workspace"),
            "missing verifier workspace",
        )
    })?;
    let mut child = scioxide_core::SolveState::new();
    let outcome = session.with_child(
        &mut child,
        identity,
        scioxide_core::SolveBudgets::unbounded()
            .with_limit(scioxide_core::BudgetKind::Iteration, iterations),
        |ctx| {
            lp.solve_into(
                p,
                warm,
                output,
                iterations,
                solver.limits.max_relaxation_bytes,
                ctx,
            )?;
            let r = &mut *output;
            if r.x.len() != p.dimension()
                || r.x.iter().any(|v| !v.is_finite())
                || !r.fun.is_finite()
            {
                return Err(invalid(
                    scioxide_core::argument!("lp"),
                    "relaxation returned invalid values",
                ));
            }
            let fun = crate::utils::dot(p.objective(), &r.x)?;
            if NumericScalar::abs(checked_difference(fun, r.fun)?)
                > solver
                    .relaxation_tolerances
                    .gap()
                    .threshold(NumericScalar::abs(fun).max(NumericScalar::abs(r.fun)))?
            {
                return Err(invalid(
                    scioxide_core::argument!("lp"),
                    "relaxation objective disagrees with its point",
                ));
            }
            r.fun = fun;
            r.verification = None;
            let applicable = matches!(
                (r.termination, r.certificate()),
                (
                    OptimizeTermination::Converged,
                    Some(ProgrammingCertificate::Optimality { .. })
                ) | (
                    OptimizeTermination::Infeasible,
                    Some(ProgrammingCertificate::Infeasibility(_))
                ) | (
                    OptimizeTermination::Unbounded,
                    Some(ProgrammingCertificate::Unboundedness { .. })
                )
            );
            if applicable {
                let view = crate::LpVerificationProblem::from_problem(p);
                let evidence = r.certificate().ok_or_else(|| {
                    invalid(scioxide_core::argument!("lp"), "missing LP evidence")
                })?;
                let verifier = SolveSession::new(ctx, scioxide_core::argument!("ctx"));
                r.verification = Some(crate::verification::kernel::interrupted(
                    crate::verification::kernel::lp_session(
                        &view,
                        evidence,
                        solver.relaxation_tolerances,
                        verification,
                        Some(&verifier),
                    ),
                    &verifier,
                )?);
            }
            if !r.verification.as_ref().is_some_and(|v| v.is_accepted()) {
                r.termination = match r.termination {
                    OptimizeTermination::Converged => OptimizeTermination::Stagnation,
                    OptimizeTermination::Infeasible => OptimizeTermination::SuspectedInfeasible,
                    OptimizeTermination::Unbounded => OptimizeTermination::SuspectedUnbounded,
                    other => other,
                };
                if let Some(stop) = r.verification.as_ref().and_then(|v| v.termination) {
                    r.termination = stop;
                }
            }
            Ok(())
        },
    );
    // Context accounting includes admitted work even if the provider fails.
    s.stats.iterations = session.accounting()?.nested().iterations();
    match outcome {
        Ok(()) => {
            let r = output;
            if lp.reused_basis(warm) {
                s.stats.basis_reuses = checked_add(
                    s.stats.basis_reuses,
                    1,
                    scioxide_core::quantity!("mip_basis_reuse"),
                )?;
            }
            if matches!(
                r.termination,
                OptimizeTermination::IterationLimit
                    | OptimizeTermination::EvaluationLimit
                    | OptimizeTermination::ProposalLimit
                    | OptimizeTermination::Cancelled
                    | OptimizeTermination::TimeLimit
            ) {
                session.inherit_child_stop(Some(r.termination))?;
            }
            Ok(())
        }
        Err(e) => {
            session.record_execution_stop(&e);
            for stop in [
                OptimizeTermination::EvaluationLimit,
                OptimizeTermination::IterationLimit,
                OptimizeTermination::ProposalLimit,
            ] {
                session.inherit_child_stop(Some(stop))?;
            }
            Err(e)
        }
    }
}

fn empty_candidate<S: RealScalar + NumericScalar<Real = S>>() -> Candidate<S> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    Candidate {
        x: Vec::new(),
        fun: z,
        primal: z,
        bounds: z,
        integrality: z,
    }
}
fn accept_candidate<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if s.incumbent.as_ref().is_none_or(|c| s.candidate.fun < c.fun) {
        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
        if let Some(old) = &mut s.incumbent {
            core::mem::swap(old, &mut s.candidate);
        } else {
            let mut empty = empty_candidate();
            crate::storage::resize(
                &mut empty.x,
                s.candidate.x.len(),
                <S as scioxide_core::AlgebraicScalar>::ZERO,
            )?;
            s.incumbent = Some(core::mem::replace(&mut s.candidate, empty));
        }
    }
    Ok(())
}

fn accept<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &mut MipSearchState<S, W>,
    value: Option<Candidate<S>>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if let Some(c) = value
        && s.incumbent.as_ref().is_none_or(|old| c.fun < old.fun)
    {
        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
        s.incumbent = Some(c);
    }
    Ok(())
}
fn branch_row<S: RealScalar + NumericScalar<Real = S>>(
    j: usize,
    up: bool,
    value: S,
) -> Result<Row<S>, SciOxideError> {
    let mut entries = capacity(1, scioxide_core::quantity!("mip_branch"))?;
    entries.push((j, <S as scioxide_core::AlgebraicScalar>::ONE));
    Ok(Row {
        provenance: None,
        entries,
        rhs: if up { value.ceil() } else { value.floor() },
        sense: if up {
            Sense::GreaterEqual
        } else {
            Sense::LessEqual
        },
    })
}
fn progressing<'a, S: 'a + RealScalar + NumericScalar<Real = S>>(
    node: impl Iterator<Item = &'a Row<S>>,
    j: usize,
    value: S,
) -> bool {
    value.floor() < value.ceil()
        && !node.into_iter().any(|r| {
            matches!(r.entries.as_slice(),[(k,a)] if *k==j&&*a==<S as scioxide_core::AlgebraicScalar>::ONE)
                && match r.sense {
                    Sense::LessEqual => r.rhs <= value.floor(),
                    Sense::GreaterEqual => r.rhs >= value.ceil(),
                    Sense::Equal => true,
                }
        })
}

fn process<S, R>(
    solver: &BranchAndBound<S>,
    original: &MipProblem<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    node: &mut MipNode<S>,
    lp: &mut R,
    session: &Session<'_, '_, '_>,
) -> Result<Option<(OptimizeTermination, &'static str)>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    if prunable(
        s.incumbent.as_ref().map(|c| c.fun),
        node.lower_bound,
        solver.relaxation_tolerances.gap(),
    )? {
        close(s, node.lower_bound);
        return Ok(None);
    }
    let Some(mut p) = node_lp(s, node.rows)? else {
        return Ok(None);
    };
    let mut relaxation = core::mem::replace(&mut s.lp_output, crate::ProgrammingResult::empty());
    let outcome = process_body(
        solver,
        original,
        s,
        node,
        lp,
        session,
        &mut p,
        &mut relaxation,
    );
    s.lp = Some(p);
    s.lp_output = relaxation;
    outcome
}

#[allow(clippy::too_many_arguments)]
fn process_body<S, R>(
    solver: &BranchAndBound<S>,
    original: &MipProblem<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    node: &mut MipNode<S>,
    lp: &mut R,
    session: &Session<'_, '_, '_>,
    p: &mut LpProblem<S>,
    relaxation: &mut crate::ProgrammingResult<S>,
) -> Result<Option<(OptimizeTermination, &'static str)>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    // Include node LP staging, validation vectors and copied node rows before work.
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
    let staging = checked_add(
        model.bytes(),
        scioxide_core::backend::checked_product(
            &[node.bytes(), 4],
            scioxide_core::quantity!("mip_staging"),
        )?,
        scioxide_core::quantity!("mip_staging"),
    )?;
    if !admit(s, staging, solver.limits.max_search_bytes)? {
        return Ok(Some((OptimizeTermination::MemoryLimit, "node staging cap")));
    }
    if relaxation_bytes(p)? > solver.limits.max_relaxation_bytes {
        return Ok(Some((OptimizeTermination::MemoryLimit, "LP workspace cap")));
    }
    solve_lp(
        p,
        lp,
        solver.max_lp_iterations,
        relaxation,
        solver,
        s,
        session,
    )?;
    if let Some(stop) = session.stopped() {
        return Ok(Some((stop, "parent resource stop")));
    }
    if relaxation.termination == OptimizeTermination::MemoryLimit {
        return Ok(Some((
            OptimizeTermination::MemoryLimit,
            "relaxation storage admission",
        )));
    }
    // Store a clearly diagnostic original point for limits without an incumbent.
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
    if s.partial.is_none() {
        let x = model.map.reconstruct(&relaxation.x)?;
        let mut integrality = z;
        for &j in original.integer_variables() {
            integrality = integrality.max(distance(x[j])?);
        }
        s.partial = Some(Candidate {
            fun: crate::utils::dot(original.relaxation().objective(), &x)?,
            x,
            primal: relaxation.primal_feasibility().unwrap_or(z),
            bounds: relaxation.bound_violation().unwrap_or(z),
            integrality,
        });
    }
    if relaxation.termination == OptimizeTermination::Infeasible {
        return Ok(None);
    }
    if relaxation.termination == OptimizeTermination::Unbounded {
        let model = s
            .model
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
        let valid_base = candidate(
            original,
            model.map.reconstruct(&relaxation.x)?,
            false,
            solver.relaxation_tolerances.feasibility(),
        )?;
        let integer_ray = matches!(relaxation.certificate(),Some(ProgrammingCertificate::Unboundedness{direction,..}) if model.integer.iter().all(|&j|direction[j].round()==direction[j]));
        if valid_base.is_some() && integer_ray {
            accept(s, valid_base, session)?;
            return Ok(Some((
                OptimizeTermination::Unbounded,
                "integer-feasible recession witness",
            )));
        }
        return Ok(Some((
            OptimizeTermination::SuspectedUnbounded,
            "unbounded LP lacks an integer recession witness",
        )));
    }
    if !relaxation.is_success() {
        return Ok(Some((relaxation.termination, "unresolved node relaxation")));
    }
    let Some(bound) = relaxation.verification_report().and_then(|v| v.dual_bound) else {
        return Ok(Some((
            OptimizeTermination::Stagnation,
            "LP supplies no verified bound",
        )));
    };
    let offset = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?
        .map
        .offset;
    let bound = scioxide_core::checked_sum(&[bound, offset])?;
    node.lower_bound = Some(node.lower_bound.map_or(bound, |b| b.max(bound)));
    if let Some((j, up, step, parent)) = node.branch {
        update_cost(&mut s.costs[j], up, step, (bound - parent).max(z))?;
    }
    for round in 0..=solver.limits.cut_rounds {
        let model = s
            .model
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
        // Nearest-integer rounding is a heuristic even when values are far away.
        let mut candidate = core::mem::replace(&mut s.candidate, empty_candidate());
        let valid = validation::node_candidate_into(
            original,
            model,
            s.rows(node.rows),
            &relaxation.x,
            &mut candidate,
            solver.relaxation_tolerances.feasibility(),
        )?;
        s.candidate = candidate;
        if valid {
            accept_candidate(s, session)?;
        }
        if prunable(
            s.incumbent.as_ref().map(|c| c.fun),
            node.lower_bound,
            solver.relaxation_tolerances.gap(),
        )? {
            close(s, node.lower_bound);
            return Ok(None);
        }
        if round == solver.limits.cut_rounds {
            break;
        }
        let cut_staging = checked_add(
            scioxide_core::backend::checked_product(
                &[
                    s.model
                        .as_ref()
                        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?
                        .bytes(),
                    4,
                ],
                scioxide_core::quantity!("cut_staging"),
            )?,
            scioxide_core::backend::checked_product(
                &[node.bytes(), 4],
                scioxide_core::quantity!("cut_staging"),
            )?,
            scioxide_core::quantity!("cut_staging"),
        )?;
        if !admit(s, cut_staging, solver.limits.max_search_bytes)? {
            break;
        }
        let model = s
            .model
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
        let mut cut_workspace = core::mem::take(&mut s.cut_workspace);
        let separation = cuts::separate_with_workspace(
            model,
            &relaxation.x,
            s.rows(node.rows),
            solver.limits.max_cuts.saturating_sub(s.stats.cuts),
            solver
                .limits
                .max_cut_nonzeros
                .saturating_sub(s.stats.cut_nonzeros),
            solver.relaxation_tolerances.feasibility(),
            &mut cut_workspace,
        );
        s.cut_workspace = cut_workspace;
        let mut new = separation?;
        if new.is_empty() {
            s.cut_workspace.output = new;
            break;
        }
        let count = new.len();
        let nonzeros = new.iter().map(|r| r.entries.len()).sum::<usize>();
        let bytes = new.iter().map(Row::bytes).sum::<usize>();
        if !admit(
            s,
            checked_add(
                checked_add(staging, bytes, scioxide_core::quantity!("mip_cut_staging"))?,
                contracts::growth_bytes(&s.row_links, count)?,
                scioxide_core::quantity!("mip_cut_staging"),
            )?,
            solver.limits.max_search_bytes,
        )? {
            break;
        }
        contracts::reserve(
            &mut s.row_links,
            count,
            scioxide_core::quantity!("mip_cut_paths"),
        )?;
        for row in new.drain(..) {
            node.rows = s.append_row(node.rows, row)?;
        }
        s.cut_workspace.output = new;
        s.stats.cuts += count;
        s.stats.cut_nonzeros += nonzeros;
        *p = node_lp(s, node.rows)?.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("node"),
                "cut contradicted node bounds",
            )
        })?;
        if relaxation_bytes(p)? > solver.limits.max_relaxation_bytes {
            return Ok(Some((
                OptimizeTermination::MemoryLimit,
                "cut LP workspace cap",
            )));
        }
        solve_lp(
            p,
            lp,
            solver.max_lp_iterations,
            relaxation,
            solver,
            s,
            session,
        )?;
        if let Some(stop) = session.stopped() {
            return Ok(Some((stop, "parent cut resource stop")));
        }
        if relaxation.termination == OptimizeTermination::Infeasible {
            return Ok(None);
        }
        if !relaxation.is_success() {
            return Ok(Some((relaxation.termination, "unresolved cut relaxation")));
        }
        if let Some(b) = relaxation.verification_report().and_then(|v| v.dual_bound) {
            let b = scioxide_core::checked_sum(&[b, offset])?;
            node.lower_bound = Some(node.lower_bound.map_or(b, |old| old.max(b)));
        } else {
            return Ok(Some((
                OptimizeTermination::Stagnation,
                "cut relaxation lacks bound",
            )));
        }
    }
    if solver.limits.repair {
        repair(solver, original, s, node, &relaxation.x, lp, session, p)?;
    }
    if let Some(stop) = session.stopped() {
        return Ok(Some((stop, "parent heuristic resource stop")));
    }
    if prunable(
        s.incumbent.as_ref().map(|c| c.fun),
        node.lower_bound,
        solver.relaxation_tolerances.gap(),
    )? {
        close(s, node.lower_bound);
        return Ok(None);
    }
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
    let mut vars = core::mem::take(&mut s.vars);
    vars.clear();
    scioxide_core::backend::try_reserve(
        &mut vars,
        model.integer.len(),
        scioxide_core::quantity!("mip_branch_candidates"),
    )?;
    for &j in &model.integer {
        if progressing(s.rows(node.rows), j, relaxation.x[j]) {
            vars.push(j);
        }
    }
    if vars.is_empty() {
        return Ok(Some((
            OptimizeTermination::Stagnation,
            "no progressing exact integer branch",
        )));
    }
    if node.depth >= solver.limits.max_depth {
        return Ok(Some((
            OptimizeTermination::DepthLimit,
            "maximum branch depth",
        )));
    }
    // Both children must fit before either is committed to the frontier.
    if s.queue.len().saturating_add(2) > solver.limits.max_open_nodes {
        return Ok(Some((
            OptimizeTermination::MemoryLimit,
            "open-node capacity",
        )));
    }
    let required = checked_add(
        staging,
        checked_add(
            contracts::growth_bytes(&s.queue, 2)?,
            checked_add(
                contracts::growth_bytes(&s.row_links, 2)?,
                2 * core::mem::size_of::<(usize, S)>(),
                scioxide_core::quantity!("mip_child_capacity"),
            )?,
            scioxide_core::quantity!("mip_child_capacity"),
        )?,
        scioxide_core::quantity!("mip_child_capacity"),
    )?;
    if !admit(s, required, solver.limits.max_search_bytes)? {
        return Ok(Some((
            OptimizeTermination::MemoryLimit,
            "child payload capacity",
        )));
    }
    let selection = select(solver, s, node, &relaxation.x, &mut vars, lp, session, p);
    s.vars = vars;
    let j = selection?;
    if let Some(stop) = session.stopped() {
        return Ok(Some((stop, "parent strong-branch resource stop")));
    }
    let parent = node.lower_bound.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("state"),
            "branch without verified bound",
        )
    })?;
    contracts::reserve(&mut s.queue, 2, scioxide_core::quantity!("mip_frontier"))?;
    let branches = [
        branch_row(j, false, relaxation.x[j])?,
        branch_row(j, true, relaxation.x[j])?,
    ];
    contracts::reserve(
        &mut s.row_links,
        2,
        scioxide_core::quantity!("mip_path_arena"),
    )?;
    let next_id = checked_add(s.next_id, 2, scioxide_core::quantity!("mip_node_ids"))?;
    for (k, row) in branches.into_iter().enumerate() {
        let up = k == 1;
        let rows = s.append_row(node.rows, row)?;
        let step = if up {
            relaxation.x[j].ceil() - relaxation.x[j]
        } else {
            relaxation.x[j] - relaxation.x[j].floor()
        };
        push(
            &mut s.queue,
            MipNode {
                id: s.next_id + k,
                depth: node.depth + 1,
                lower_bound: node.lower_bound,
                rows,
                branch: Some((j, up, step, parent)),
            },
        );
    }
    s.next_id = next_id;

    Ok(None)
}

fn update_cost<S: RealScalar>(
    c: &mut Pseudocost<S>,
    up: bool,
    step: S,
    gain: S,
) -> Result<(), SciOxideError> {
    let k = usize::from(up);
    let v = scioxide_core::checked_float_quotient(gain, step)?;
    c.sums[k] = scioxide_core::checked_sum(&[c.sums[k], v])?;
    c.counts[k] = checked_add(
        c.counts[k],
        1,
        scioxide_core::quantity!("mip_pseudocost_count"),
    )?;
    Ok(())
}
fn score<S: RealScalar>(c: &Pseudocost<S>, x: S) -> Result<S, SciOxideError> {
    let down = x - x.floor();
    let up = x.ceil() - x;
    let mean = |k: usize| -> Result<S, SciOxideError> {
        if c.counts[k] == 0 {
            Ok(S::ONE)
        } else {
            Ok(c.sums[k] / S::try_from_usize_exact(c.counts[k])?)
        }
    };
    // Minimum directional predicted gain, deterministic and scale-free weights.
    Ok((mean(0)? * down).min(mean(1)? * up))
}
#[allow(clippy::too_many_arguments)]
fn select<S, R>(
    solver: &BranchAndBound<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    node: &MipNode<S>,
    x: &[S],
    vars: &mut [usize],
    lp: &mut R,
    session: &Session<'_, '_, '_>,
    p: &mut LpProblem<S>,
) -> Result<usize, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    vars.sort_unstable_by(|&a, &b| {
        let da = (x[a] - x[a].floor()).min(x[a].ceil() - x[a]);
        let db = (x[b] - x[b].floor()).min(x[b].ceil() - x[b]);
        db.partial_cmp(&da)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if solver.branching == MipBranching::MostFractional {
        return Ok(vars[0]);
    }
    if solver.branching == MipBranching::Reliability && solver.limits.strong_iterations > 0 {
        for &j in vars.iter().take(solver.limits.strong_candidates) {
            if solver.integrality_tolerance.is_satisfied(distance(x[j])?)? {
                continue;
            }
            for up in [false, true] {
                if s.costs[j].counts[usize::from(up)] >= solver.limits.reliability {
                    continue;
                }
                set_path_bounds(s, node.rows, p)?;
                let offset = s
                    .model
                    .as_ref()
                    .ok_or_else(|| invalid(scioxide_core::argument!("model"), "missing model"))?
                    .map
                    .offset;
                let bound = branch_row(j, up, x[j])?;
                tighten(&mut s.bounds_scratch, &bound);
                if contradictory(&s.bounds_scratch) {
                    continue;
                }
                p.update_bounds(&s.bounds_scratch)?;
                let mut r =
                    core::mem::replace(&mut s.probe_output, crate::ProgrammingResult::empty());
                s.stats.probes =
                    checked_add(s.stats.probes, 1, scioxide_core::quantity!("strong_probes"))?;
                let outcome = solve_lp(
                    p,
                    lp,
                    Budget::try_from(solver.limits.strong_iterations)?,
                    &mut r,
                    solver,
                    s,
                    session,
                );
                set_path_bounds(s, node.rows, p)?;
                outcome?;
                if session.stopped().is_some() {
                    s.probe_output = r;
                    return Ok(j);
                }
                if r.is_success()
                    && let Some(b) = r.verification_report().and_then(|v| v.dual_bound)
                {
                    let b = scioxide_core::checked_sum(&[b, offset])?;
                    let step = if up {
                        x[j].ceil() - x[j]
                    } else {
                        x[j] - x[j].floor()
                    };
                    update_cost(
                        &mut s.costs[j],
                        up,
                        step,
                        (b - node.lower_bound.unwrap_or(b))
                            .max(<S as scioxide_core::AlgebraicScalar>::ZERO),
                    )?;
                }
                s.probe_output = r;
            }
        }
    }
    let mut chosen = vars[0];
    let mut best = score(&s.costs[chosen], x[chosen])?;
    for &j in &vars[1..] {
        let v = score(&s.costs[j], x[j])?;
        if v > best || (v == best && j < chosen) {
            chosen = j;
            best = v;
        }
    }
    Ok(chosen)
}
#[allow(clippy::too_many_arguments)]
fn repair<S, R>(
    solver: &BranchAndBound<S>,
    original: &MipProblem<S>,
    s: &mut MipSearchState<S, R::Workspace>,
    node: &MipNode<S>,
    x: &[S],
    lp: &mut R,
    session: &Session<'_, '_, '_>,
    p: &mut LpProblem<S>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: MipRelaxationSolver<S>,
{
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
    if model.integer.is_empty() {
        return Ok(());
    }
    let staging = checked_add(
        scioxide_core::backend::checked_product(
            &[model.bytes(), 4],
            scioxide_core::quantity!("repair_staging"),
        )?,
        scioxide_core::backend::checked_product(
            &[node.bytes(), 4],
            scioxide_core::quantity!("repair_staging"),
        )?,
        scioxide_core::quantity!("repair_staging"),
    )?;
    if !admit(s, staging, solver.limits.max_search_bytes)? {
        return Ok(());
    }
    set_path_bounds(s, node.rows, p)?;
    let model = s
        .model
        .as_ref()
        .ok_or_else(|| invalid(scioxide_core::argument!("model"), "missing model"))?;
    for &j in &model.integer {
        let v = x[j].round();
        let (l, u) = s.bounds_scratch[j];
        if l.is_some_and(|l| v < l) || u.is_some_and(|u| v > u) {
            return Ok(());
        }
        s.bounds_scratch[j] = (Some(v), Some(v));
    }
    if contradictory(&s.bounds_scratch) {
        return Ok(());
    }
    p.update_bounds(&s.bounds_scratch)?;
    let mut r = core::mem::replace(&mut s.probe_output, crate::ProgrammingResult::empty());
    s.stats.repairs = checked_add(s.stats.repairs, 1, scioxide_core::quantity!("mip_repairs"))?;
    let outcome = solve_lp(p, lp, solver.max_lp_iterations, &mut r, solver, s, session);
    set_path_bounds(s, node.rows, p)?;
    outcome?;
    if session.stopped().is_some() {
        s.probe_output = r;
        return Ok(());
    }
    if r.is_success() {
        let model = s
            .model
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("state"), "missing model"))?;
        let mut candidate = core::mem::replace(&mut s.candidate, empty_candidate());
        let valid = validation::node_candidate_into(
            original,
            model,
            s.rows(node.rows),
            &r.x,
            &mut candidate,
            solver.relaxation_tolerances.feasibility(),
        )?;
        s.candidate = candidate;
        if valid {
            accept_candidate(s, session)?;
        }
    }
    s.probe_output = r;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn frontier_orders_unknown_then_best_bound_then_insertion() {
        let mut queue = Vec::new();
        for (id, bound) in [
            (5, Some(2.)),
            (3, Some(-1.)),
            (4, None),
            (2, Some(-1.)),
            (1, None),
        ] {
            push(
                &mut queue,
                MipNode {
                    id,
                    lower_bound: bound,
                    depth: 0,
                    rows: None,
                    branch: None,
                },
            );
        }
        let ids: Vec<_> = core::iter::from_fn(|| pop(&mut queue))
            .map(|n| n.id)
            .collect();
        assert_eq!(ids, vec![1, 4, 2, 3, 5]);
    }

    #[test]
    fn pruning_uses_absolute_relative_and_unit_scale_gap_thresholds() {
        for (incumbent, t) in [
            (
                0.0_f64,
                scioxide_core::ResidualTolerance::new(0.125, 0.).unwrap(),
            ),
            (
                8.,
                scioxide_core::ResidualTolerance::new(0., 0.125).unwrap(),
            ),
            (
                0.25,
                scioxide_core::ResidualTolerance::new(0., 0.125).unwrap(),
            ),
        ] {
            let threshold = t.threshold(incumbent.abs().max(1.)).unwrap();
            for (gap, expected) in [
                (threshold * 0.5, true),
                (threshold, true),
                (threshold * 2., false),
            ] {
                assert_eq!(
                    prunable(Some(incumbent), Some(incumbent - gap), t).unwrap(),
                    expected
                );
            }
        }
        assert!(
            !prunable(
                None,
                Some(0.),
                scioxide_core::ResidualTolerance::<f64>::new(0., 0.).unwrap()
            )
            .unwrap()
        );
    }
}
