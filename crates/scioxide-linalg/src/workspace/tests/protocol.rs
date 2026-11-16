//! Storage reservation and solve-slot behavior, independent of numerical families.

use super::super::*;

use scioxide_core::{
    AttemptOutcome, ExecutionContext, ProblemToken, SciOxideErrorKind, SolveBudgets, SolveIdentity,
};

fn requirements() -> LinalgRequirements {
    LinalgRequirements {
        scalars: [1; 8],
        reals: [1; 2],
        indices: 1,
        accumulators: 1,
        extended: 1,
        scaled: 1,
        exponents: 1,
        ..Default::default()
    }
}

#[test]
fn byte_limits_include_retained_owned_capacity_and_compose_by_minimum() -> Result<(), SciOxideError>
{
    let first = requirements();
    let bytes = first.required_bytes::<f64>()?;
    let exact = LinalgRequirements {
        byte_limit: Some(bytes),
        ..first
    };
    let short = LinalgRequirements {
        byte_limit: Some(bytes - 1),
        ..first
    };
    assert_eq!(exact.max(&short).byte_limit, short.byte_limit);
    let mut workspace = LinalgWorkspace::<f64>::new();
    workspace.try_reserve(&exact)?;
    let retained = workspace.retained_bytes()?;
    assert!(retained >= bytes);
    for request in [
        short,
        LinalgRequirements {
            byte_limit: Some(0),
            ..Default::default()
        },
    ] {
        assert_eq!(
            workspace.check(&request).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            workspace.reservation_bytes(&request).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            workspace.try_reserve(&request).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(workspace.retained_bytes()?, retained);
    }
    assert_eq!(
        LinalgWorkspace::<f64>::new().check(&LinalgRequirements::new(Some(0)))?,
        Capacity::Sufficient
    );
    Ok(())
}

#[test]
fn contiguous_scalar_storage_repartitions_without_owning_or_growing() -> Result<(), SciOxideError> {
    let mut scalars = [0.0; 16];
    let mut workspace = LinalgWorkspace::from_scalar_slice(&mut scalars);
    let request = LinalgRequirements::default().with_scalar(ScalarRegion::Candidate, 4);
    workspace.try_reserve(&request)?;
    assert_eq!(workspace.check(&request)?, Capacity::Sufficient);
    assert_eq!(workspace.retained_bytes()?, 0);
    let unavailable = request.with_real(RealRegion::Direct, 1);
    assert_eq!(
        workspace.try_reserve(&unavailable).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(workspace.check(&request)?, Capacity::Sufficient);
    Ok(())
}

#[test]
fn solve_slots_replace_attempts_and_restore_transaction_depth() -> Result<(), SciOxideError> {
    let mut workspace = LinalgWorkspace::<f64>::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "workspace", 1),
        0,
        0,
    );
    assert!(workspace.last_attempt().is_none());
    workspace.begin_attempt();
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    workspace.run_scope(
        ExecutionContext::sequential().into(),
        identity,
        SolveBudgets::unbounded(),
        |_, _, solve| {
            assert!(solve.record_iteration()?.is_granted());
            Ok(())
        },
    )?;
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert!(attempt.epoch().is_some());
    assert_eq!(attempt.counts().own().iterations(), 1);
    let error = workspace
        .run_scope(
            ExecutionContext::sequential().into(),
            identity,
            SolveBudgets::unbounded(),
            |_, _, solve| {
                assert!(solve.record_iteration()?.is_granted());
                Err::<(), _>(SciOxideError::singular(None))
            },
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::Singular);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert_eq!(workspace.scope_depth, 0);
    workspace.begin_attempt();
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
    assert_eq!(attempt.counts(), scioxide_core::SolveCounts::new());
    Ok(())
}

#[test]
fn contiguous_scalar_storage_serves_successive_operations_of_any_fitting_shape()
-> Result<(), SciOxideError> {
    let mut scalars = [0.0; 12];
    let mut workspace = LinalgWorkspace::from_scalar_slice(&mut scalars);
    let first = LinalgRequirements::default().with_scalar(ScalarRegion::Candidate, 10);
    let second = LinalgRequirements::default().with_scalar(ScalarRegion::Direct, 12);
    // A fresh slice that fits needs no growth.
    assert_eq!(workspace.check(&second)?, Capacity::Sufficient);
    (|| {
        workspace.try_reserve(&first)?;
        workspace.try_reserve(&second)?;
        workspace.try_reserve(&first)
    })()?;
    assert_eq!(workspace.check(&first.max(&second))?, Capacity::NeedsGrowth);
    assert_eq!(
        workspace
            .try_reserve(&first.max(&second))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    Ok(())
}
