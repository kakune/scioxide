//! Reservation is explicit, fallible, and accounts for all retained staging.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Capacity, ExecutionContext, NdView, SciOxideErrorKind, StridedLayout};
use scioxide_spatial::*;

#[test]
fn typed_staging_growth_reuse_and_retained_limits() {
    let options = TransformOptions::new();
    let requests = [
        TransformWorkspaceRequest::apply(8, options),
        TransformWorkspaceRequest::euler(8, options),
        TransformWorkspaceRequest::rotations(8, options),
    ];
    let mut workspace = TransformWorkspace::<f64>::new();
    let mut total = 0;
    for request in requests {
        let requirements = TransformWorkspace::<f64>::workspace_requirements(request).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::NeedsGrowth
        );
        workspace.try_reserve(&requirements).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
        total += requirements.bytes::<f64>().unwrap();
    }
    let limited = options.with_max_workspace_bytes(total);
    let requirements = TransformWorkspace::<f64>::workspace_requirements(
        TransformWorkspaceRequest::rotations(4, limited),
    )
    .unwrap();
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let too_small = TransformWorkspace::<f64>::workspace_requirements(
        TransformWorkspaceRequest::apply(1, options.with_max_workspace_bytes(total - 1)),
    )
    .unwrap();
    assert_eq!(
        workspace.check(&too_small).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        workspace.try_reserve(&too_small).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    // A small active request cannot hide storage left by earlier operations.
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let larger = TransformWorkspace::<f64>::workspace_requirements(
        TransformWorkspaceRequest::apply(16, options),
    )
    .unwrap();
    assert_eq!(workspace.check(&larger).unwrap(), Capacity::NeedsGrowth);
    workspace.try_reserve(&larger).unwrap();
    assert_eq!(workspace.check(&larger).unwrap(), Capacity::Sufficient);
}

#[test]
fn requirement_overflow_and_byte_limits_are_structured() {
    let options = TransformOptions::new();
    for request in [
        TransformWorkspaceRequest::apply(usize::MAX, options),
        TransformWorkspaceRequest::euler(usize::MAX, options),
        TransformWorkspaceRequest::rotations(usize::MAX, options),
    ] {
        assert_eq!(
            TransformWorkspace::<f64>::workspace_requirements(request)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    }
    let zero = options.with_max_workspace_bytes(0);
    for request in [
        TransformWorkspaceRequest::apply(1, zero),
        TransformWorkspaceRequest::euler(1, zero),
        TransformWorkspaceRequest::rotations(1, zero),
    ] {
        assert_eq!(
            TransformWorkspace::<f64>::workspace_requirements(request)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
    // Requirements may be inspected with another scalar width; check uses the workspace's width.
    let request = TransformWorkspaceRequest::apply(1, options.with_max_workspace_bytes(12));
    let requirements = TransformWorkspace::<f32>::workspace_requirements(request).unwrap();
    assert_eq!(requirements.bytes::<f32>().unwrap(), 12);
    assert_eq!(
        TransformWorkspace::<f64>::new()
            .check(&requirements)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn execution_rejects_missing_or_over_limit_staging_without_publishing() {
    let shape = [2, 3];
    let strides = [3, 1];
    let points = PointSet::new(
        NdView::new(&[0.0; 6], StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
    )
    .unwrap();
    let options = TransformOptions::new();
    let mut output = [[42.0; 3]; 2];
    let mut workspace = TransformWorkspace::new();
    let error = Rotation::identity()
        .apply_batch_into(
            &points,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(output, [[42.0; 3]; 2]);
    workspace
        .try_reserve(
            &TransformWorkspace::<f64>::workspace_requirements(TransformWorkspaceRequest::apply(
                16, options,
            ))
            .unwrap(),
        )
        .unwrap();
    let error = Rotation::identity()
        .apply_batch_into(
            &points,
            options.with_max_workspace_bytes(48),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(output, [[42.0; 3]; 2]);
    Rotation::identity()
        .apply_batch_into(
            &points,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [[0.0; 3]; 2]);
}
