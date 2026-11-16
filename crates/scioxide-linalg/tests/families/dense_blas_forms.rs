//! Canonical dense/BLAS forms preserve numerical and transactional contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::Accuracy;
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_core::{
    ExecutionContext, ProblemToken, SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity,
    SolveState,
};
use scioxide_linalg::{
    DenseWorkspaceRequest, LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace,
    MatrixSide, MatrixView, MatrixViewMut, StructuredOperation, VectorView, VectorViewMut,
    givens_rotation_scoped, modified_givens_rotation_scoped,
};

#[test]
fn named_rotation_preserves_sign_encoding_and_scaled_inputs() -> Result<(), SciOxideError> {
    for (a, b) in [
        (3.0_f64, 4.0),
        (-4.0, 3.0),
        (3.0, -3.0),
        (0.0, 0.0),
        (0.0, -2.0),
        (2.0, 0.0),
        (3e-300, 4e-300),
        (3e300, 4e300),
    ] {
        let rotation = givens_rotation_scoped(
            a,
            b,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ExecutionContext::sequential(),
        )?;
        let scale = a.abs().max(b.abs());
        if scale == 0.0 {
            assert_eq!(
                (
                    rotation.r(),
                    rotation.z(),
                    rotation.rotation().cosine(),
                    rotation.rotation().sine()
                ),
                (0.0, 0.0, 1.0, 0.0)
            );
            continue;
        }
        let (x, y) = (a / scale, b / scale);
        assert!(
            (rotation.rotation().cosine() * x + rotation.rotation().sine() * y
                - rotation.r() / scale)
                .abs()
                < 1e-14
        );
        assert!((-rotation.rotation().sine() * x + rotation.rotation().cosine() * y).abs() < 1e-14);
        assert_eq!(
            rotation.r().is_sign_negative(),
            (if a.abs() > b.abs() { a } else { b }).is_sign_negative()
        );
        let z = if a.abs() > b.abs() {
            rotation.rotation().sine()
        } else if rotation.rotation().cosine() != 0.0 {
            1.0 / rotation.rotation().cosine()
        } else {
            1.0
        };
        assert!((rotation.z() - z).abs() <= 1e-14 * z.abs().max(1.0));
    }
    assert_eq!(
        givens_rotation_scoped(
            f64::NAN,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert!(
        givens_rotation_scoped(
            f64::MAX,
            f64::MAX,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn reductions_and_generators_accept_parent_and_reborrowed_contexts() -> Result<(), SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(1099),
        scioxide_core::algorithm_id!("tests", "dense_forms", 1),
        0,
        0,
    );
    let mut parent = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let values = VectorView::from_slice(&[3.0_f64, 4.0]);
    assert!(
        (values.norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), &mut parent)? - 5.0)
            .abs()
            < 1e-14
    );
    let mut context = LinalgExecutionContext::from(&mut parent);
    assert!(
        (values.dotu_scoped(
            values,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            context.reborrow()
        )? - 25.0)
            .abs()
            < 1e-14
    );
    assert!(
        (givens_rotation_scoped(
            3.0_f64,
            4.0,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            context.reborrow()
        )?
        .r() - 5.0)
            .abs()
            < 1e-14
    );
    let modified = modified_givens_rotation_scoped(
        1.0_f64,
        1.0,
        3.0,
        4.0,
        &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
        context.reborrow(),
    )?;
    assert!(modified.d1().is_finite() && modified.d2().is_finite());
    Ok(())
}

#[test]
fn owned_triangular_forms_apply_alpha_on_both_sides() -> Result<(), SciOxideError> {
    let triangular = MatrixView::from_slice(2, 2, &[2.0_f64, 1.0, f64::NAN, 3.0])?;
    let general = MatrixView::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0])?;
    let options = LinalgExecutionOptions::new();
    for side in [MatrixSide::Left, MatrixSide::Right] {
        for operation in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let product = triangular.triangular_mul_mat(
                general,
                side,
                Triangle::Upper,
                DiagonalKind::Explicit,
                operation,
                2.5,
            )?;
            let solved = triangular.triangular_solve_mat(
                product.view(),
                side,
                Triangle::Upper,
                DiagonalKind::Explicit,
                operation,
                0.4,
            )?;
            for (actual, expected) in solved
                .as_slice()
                .iter()
                .zip(general.try_to_matrix()?.as_slice())
            {
                assert!((actual - expected).abs() < 1e-13);
            }
            let mut inplace = general.try_to_matrix()?;
            triangular.triangular_mul_mat_in_place(
                side,
                Triangle::Upper,
                DiagonalKind::Explicit,
                operation,
                2.5,
                &options,
                inplace.view_mut(),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )?;
            for (actual, expected) in inplace.as_slice().iter().zip(product.as_slice()) {
                assert!((actual - expected).abs() < 1e-13);
            }
        }
    }
    Ok(())
}

#[test]
fn warmed_in_place_forms_preserve_padding_and_execute() -> Result<(), SciOxideError> {
    let options = LinalgExecutionOptions::new();
    let mut workspace =
        LinalgWorkspace::<f64>::workspace(&DenseWorkspaceRequest::vector_pair(2, &options))?;
    let mut first = [3.0, 101.0, 4.0];
    let mut second = [4.0, 202.0, -3.0];
    (|| {
        let rotation = givens_rotation_scoped(
            3.0,
            4.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ExecutionContext::sequential(),
        )?;
        VectorViewMut::rotate_in_place(
            rotation.rotation(),
            &options,
            VectorViewMut::from_slice_with_stride(&mut first, 2, 2)?,
            VectorViewMut::from_slice_with_stride(&mut second, 2, 2)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        VectorViewMut::swap_in_place(
            &options,
            VectorViewMut::from_slice_with_stride(&mut first, 2, 2)?,
            VectorViewMut::from_slice_with_stride(&mut second, 2, 2)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )
    })()?;
    for (actual, expected) in first.iter().zip([0.0, 101.0, -5.0]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    for (actual, expected) in second.iter().zip([5.0, 202.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    for (cosine, sine) in [(f64::NAN, 0.0), (1.0, f64::INFINITY)] {
        assert_eq!(
            scioxide_linalg::GivensRotation::new(cosine, sine)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    Ok(())
}

#[test]
fn rank_update_failure_keeps_all_matrix_storage() -> Result<(), SciOxideError> {
    let options = LinalgExecutionOptions::new();
    let mut data = [1.0_f64, 99.0, 2.0, 88.0, 3.0, 77.0, 4.0];
    let before = data;
    let mut scratch = [];
    let result = MatrixViewMut::rank_one_update_in_place(
        1.0,
        VectorView::from_slice(&[2.0, 3.0]),
        VectorView::from_slice(&[4.0, 5.0]),
        &options,
        MatrixViewMut::from_slice_with_strides(2, 2, &mut data, 4, 2)?,
        &mut LinalgWorkspace::from_scalar_slice(&mut scratch),
        ExecutionContext::sequential(),
    );
    assert_eq!(
        result.unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(data, before);
    Ok(())
}

#[test]
fn prepared_rank_updates_and_triangular_solves_execute() -> Result<(), SciOxideError> {
    let options = LinalgExecutionOptions::new();
    let triangular = MatrixView::from_slice(2, 2, &[2.0_f64, 0.0, 1.0, 3.0])?;
    let x = VectorView::from_slice(&[1.0_f64, 2.0]);
    let mut data = [1.0_f64, 2.0, 3.0, 4.0];
    let mut workspace = LinalgWorkspace::<f64>::workspace(
        &DenseWorkspaceRequest::triangular_solve(2, 2, 2, &options),
    )?;
    workspace.try_reserve(&LinalgWorkspace::<f64>::workspace_requirements(
        &DenseWorkspaceRequest::vector(4, &options),
    )?)?;
    (|| {
        MatrixViewMut::rank_one_update_in_place(
            0.5,
            x,
            x,
            &options,
            MatrixViewMut::from_slice(2, 2, &mut data)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        triangular.triangular_solve_mat_in_place(
            MatrixSide::Left,
            Triangle::Lower,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            1.0,
            &options,
            MatrixViewMut::from_slice(2, 2, &mut data)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )
    })()?;
    for (actual, expected) in data.iter().zip([0.75, 1.5, 13.0 / 12.0, 1.5]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    Ok(())
}

#[test]
fn plain_level1_kernels_equal_their_scoped_forms() -> Result<(), SciOxideError> {
    let options = LinalgExecutionOptions::default();
    let close = |actual: f64, expected: f64| {
        assert!((actual - expected).abs() <= 4.0 * f64::EPSILON * expected.abs());
    };
    let x = VectorView::from_slice(&[3.0_f64, 4.0]);
    let norm = x.norm2()?;
    let scoped = x.norm2_scoped(&options, ExecutionContext::sequential_native())?;
    assert_eq!(norm.to_bits(), scoped.to_bits());
    close(norm, 5.0);
    let dot = x.dotu(x)?;
    let scoped = x.dotu_scoped(x, &options, ExecutionContext::sequential_native())?;
    assert_eq!(dot.to_bits(), scoped.to_bits());
    close(dot, 25.0);
    let dot = x.dotc_left(x)?;
    let scoped = x.dotc_left_scoped(x, &options, ExecutionContext::sequential_native())?;
    assert_eq!(dot.to_bits(), scoped.to_bits());
    close(dot, 25.0);
    let asum = x.asum()?;
    let scoped = x.asum_scoped(&options, ExecutionContext::sequential_native())?;
    assert_eq!(asum.to_bits(), scoped.to_bits());
    close(asum, 7.0);
    assert_eq!(x.iamax()?, Some(1));
    assert_eq!(
        x.iamax_scoped(&options, ExecutionContext::sequential_native())?,
        Some(1)
    );

    // conj(3 + 4i) (1 − 2i) = −5 − 10i.
    let (z, w) = (
        [scioxide_core::Complex::new(3.0_f64, 4.0)],
        [scioxide_core::Complex::new(1.0, -2.0)],
    );
    let (z, w) = (VectorView::from_slice(&z), VectorView::from_slice(&w));
    let inner = z.dotc_left(w)?;
    let scoped = z.dotc_left_scoped(w, &options, ExecutionContext::sequential_native())?;
    assert_eq!(
        (inner.re.to_bits(), inner.im.to_bits()),
        (scoped.re.to_bits(), scoped.im.to_bits())
    );
    close(inner.re, -5.0);
    close(inner.im, -10.0);

    let y = VectorView::from_slice(&[3.0_f32, 4.0]);
    let wide = y.dot_wide(y)?;
    let scoped = y.dot_wide_scoped(y, &options, ExecutionContext::sequential_native())?;
    assert_eq!(wide.to_bits(), scoped.to_bits());
    close(wide, 25.0);
    let plus = y.dot_plus(y, 1.0)?;
    let scoped = y.dot_plus_scoped(y, 1.0, &options, ExecutionContext::sequential_native())?;
    assert_eq!(plus.to_bits(), scoped.to_bits());
    assert!((plus - 26.0).abs() <= 4.0 * f32::EPSILON * 26.0);

    let rotation = scioxide_linalg::givens_rotation(3.0_f64, 4.0)?;
    let scoped = givens_rotation_scoped(3.0, 4.0, &options, ExecutionContext::sequential_native())?;
    let bits = |r: scioxide_linalg::GivensRotationResult<f64>| {
        [r.r(), r.z(), r.rotation().cosine(), r.rotation().sine()].map(f64::to_bits)
    };
    assert_eq!(bits(rotation), bits(scoped));
    close(rotation.r(), 5.0);

    let modified = scioxide_linalg::modified_givens_rotation(1.0_f64, 1.0, 3.0, 4.0)?;
    let scoped = modified_givens_rotation_scoped(
        1.0,
        1.0,
        3.0,
        4.0,
        &options,
        ExecutionContext::sequential_native(),
    )?;
    // `Debug` prints each float as its shortest round-trip decimal, so equal output means equal bits.
    assert_eq!(format!("{modified:?}"), format!("{scoped:?}"));
    // √d₁′ x₁′ = ‖(3, 4)‖₂ = 5.
    close(modified.d1().sqrt() * modified.x1(), 5.0);
    Ok(())
}
