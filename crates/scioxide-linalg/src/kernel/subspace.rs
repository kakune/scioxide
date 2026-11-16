//! Shared block kernels for restarted spectral recurrences. Vector updates,
//! norms and bounds come from [`super::finite`].

use super::finite;
use crate::KernelContext;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

/// Combine stored columns with a small factor column, including scalar promotion.
/// Range scaling is shared with vector updates; cached operator images need not
/// have unit magnitude. The output is private staging storage.
pub(crate) fn combine<S: NumericScalar, T: NumericScalar>(
    output: &mut [T],
    basis: &[S],
    coefficients: impl Iterator<Item = T>,
    promote: impl Fn(S) -> T,
    dispatch: scioxide_core::backend::KernelDispatch,
) -> Result<(), SciOxideError> {
    output.fill(T::ZERO);
    if output.is_empty() {
        return Ok(());
    }
    for (column, coefficient) in basis.chunks_exact(output.len()).zip(coefficients) {
        let bound = column.iter().fold(T::Real::ZERO, |bound, &v| {
            bound.max(crate::kernel::standard::magnitude(promote(v)))
        });
        let affine = super::standard::Affine::new(
            coefficient,
            bound,
            T::ONE,
            finite::maximum(output, dispatch)?,
        )?;
        if affine.is_unscaled() {
            for (out, &value) in output.iter_mut().zip(column) {
                *out = super::standard::fused_update(coefficient, promote(value), *out);
            }
        } else {
            for (out, &value) in output.iter_mut().zip(column) {
                *out = crate::checked::finite_candidate(affine.apply(promote(value), *out))?;
            }
        }
    }
    Ok(())
}

pub(crate) use super::level2::Columns;

/// Scratch for a same-scalar Ritz block, including provider and range fallbacks.
pub(crate) fn block_requirements<S: NumericScalar>(
    rows: usize,
    shared: usize,
    cols: usize,
    options: &crate::LinalgExecutionOptions,
) -> Result<crate::LinalgRequirements, SciOxideError> {
    Ok(crate::LinalgWorkspace::<S>::workspace_requirements(
        &crate::DenseWorkspaceRequest::product(rows, shared, cols, options).with_provider_packing(),
    )?
    .max(&crate::LinalgWorkspace::<S>::workspace_requirements(
        &crate::DenseWorkspaceRequest::product(cols, shared, rows, options).with_provider_packing(),
    )?))
}

/// Transform packed basis columns into a private Ritz block. Gather only the
/// small selected factor; GEMM shares range proofs and packed basis panels across
/// all output columns. The enclosing action owns verification and publication.
pub(crate) fn combine_block<S: NumericScalar>(
    output: crate::MatrixViewMut<'_, S>,
    basis: &[S],
    coefficient: impl Fn(usize, usize) -> S,
    scratch: &mut [S],
    workspace: &mut crate::LinalgWorkspace<'_, S>,
    policy: scioxide_core::ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let rows = output.rows();
    let cols = output.cols();
    if rows == 0 || cols == 0 {
        return context.execution().check();
    }
    let shared = basis.len() / rows;
    let factor = &mut scratch[..shared * cols];
    for (row, entries) in factor.chunks_mut(cols).enumerate() {
        for (col, entry) in entries.iter_mut().enumerate() {
            *entry = coefficient(row, col);
        }
    }
    let left = crate::FiniteMatrixView::from_validated(
        crate::MatrixView::from_slice(shared, rows, basis)?.transpose(),
        finite::maximum(basis, context.dispatch())?,
    );
    let right = crate::FiniteMatrixView::from_workspace(crate::MatrixView::from_slice(
        shared, cols, factor,
    )?);
    // Packed retained columns are row-major when transposed. Compute C^T =
    // F^T Q^T so the long vector dimension fills native GEMM column tiles.
    let (left, right, output) = if output.row_stride == 1 {
        (right.transpose(), left.transpose(), output.transpose())
    } else {
        (left, right, output)
    };
    super::gemm_workspace_with_context(
        super::WorkspaceGemmProblem {
            left,
            right,
            output,
            config: super::GemmConfig::ordinary(),
        },
        policy,
        workspace,
        context,
    )
}

/// Classical Gram--Schmidt with the Daniel--Gragg--Kaufman--Stewart norm-loss
/// test: repeat the projection when the first pass loses more than half the
/// squared norm. Accumulate both passes into the recurrence and apply identical
/// transformations to the optional metric image and paired image/direction.
/// B norms use the supplied B image; positivity/dependence policy belongs to
/// the caller. All coefficient scratch is reserved before entering the kernel.
pub(crate) fn orthogonalize<S: NumericScalar>(
    vector: &mut [S],
    basis: Columns<'_, S>,
    coefficients: &mut [S],
    mut metric: Option<(&mut [S], Columns<'_, S>)>,
    mut paired: Option<(&mut [S], Columns<'_, S>)>,
    mut accumulate: impl FnMut(usize, S),
    cx: KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    cx.execution().check()?;
    let measure = |vector: &[S], metric: Option<&[S]>| {
        metric.map_or_else(
            || finite::norm(vector),
            |image| {
                Ok(finite::dot(vector, image, cx.dispatch())?
                    .real_component()
                    .abs()
                    .sqrt())
            },
        )
    };
    let before = measure(vector, metric.as_ref().map(|(image, _)| &**image))?;
    // Bounds are exact under every reduction order, so they use native leaves.
    let basis_maximum = basis.maximum(cx.dispatch())?;
    let metric_maximum = metric
        .as_ref()
        .map(|(_, columns)| columns.maximum(cx.dispatch()))
        .transpose()?;
    let paired_maximum = paired
        .as_ref()
        .map(|(_, columns)| columns.maximum(cx.dispatch()))
        .transpose()?;
    // Cached metric recurrences retain the original four-lane coefficient
    // reduction. A different native reduction tree can change their convergence
    // and explicit-verification schedule even when residual accuracy is unchanged.
    let projection_dispatch = if metric.is_some() {
        scioxide_core::backend::KernelDispatch::for_backend(
            scioxide_core::backend::KernelBackend::Scalar,
            cx.execution().processor(),
        )?
    } else {
        cx.dispatch()
    };
    let coefficients = &mut coefficients[..basis.iter().len()];
    let mut after = before;
    for pass in 0..2 {
        let image = metric.as_ref().map_or(&*vector, |(image, _)| &**image);
        super::level2::project_columns(
            basis,
            basis_maximum,
            image,
            finite::maximum(image, cx.dispatch())?,
            coefficients,
            projection_dispatch,
        )?;
        super::level2::subtract_columns(basis, basis_maximum, coefficients, vector)?;
        if let Some((image, columns)) = &mut metric {
            super::level2::subtract_columns(
                *columns,
                metric_maximum.unwrap_or(S::Real::ZERO),
                coefficients,
                image,
            )?;
        }
        if let Some((image, columns)) = &mut paired {
            super::level2::subtract_columns(
                *columns,
                paired_maximum.unwrap_or(S::Real::ZERO),
                coefficients,
                image,
            )?;
        }
        for (i, &coefficient) in coefficients.iter().enumerate() {
            accumulate(i, coefficient);
        }
        after = measure(vector, metric.as_ref().map(|(image, _)| &**image))?;
        if pass == 0 && after >= S::Real::HALF.sqrt() * before {
            break;
        }
    }
    Ok(after)
}

/// Euclidean action projection over packed columns.
pub(crate) fn project<S: NumericScalar>(
    vector: &mut [S],
    basis: &[S],
    coefficients: &mut [S],
    accumulate: impl FnMut(usize, S),
    cx: KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    let rows = vector.len();
    orthogonalize(
        vector,
        Columns::Packed(basis, rows),
        coefficients,
        None,
        None,
        accumulate,
        cx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;
    use alloc::vec::Vec;
    use scioxide_core::Complex;

    #[test]
    fn block_ritz_rotation_preserves_complex_orthogonality_and_images() {
        use crate::{Accuracy, LinalgExecutionOptions, Matrix};
        use scioxide_core::ExecutionPolicy;
        let n = 65;
        let z = |re, im| Complex::new(re, im);
        let mut basis = vec![z(0., 0.); 2 * n];
        basis[0] = z(0.6, 0.);
        basis[n - 1] = z(0., 0.8);
        basis[n] = z(0.8, 0.);
        basis[2 * n - 1] = z(0., -0.6);
        let images: Vec<_> = basis
            .iter()
            .enumerate()
            .map(|(i, &x)| x * if i % n == 0 { 2. } else { 5. })
            .collect();
        // Swap the selected columns and rotate their phases.
        let factor = [z(0., 0.), z(0., 1.), z(0., -1.), z(0., 0.)];
        for accuracy in [Accuracy::Standard, Accuracy::Exact] {
            let options = LinalgExecutionOptions::new().with_accuracy(accuracy);
            let mut workspace = crate::LinalgWorkspace::new();
            workspace
                .try_reserve(&block_requirements::<Complex<f64>>(n, 2, 2, &options).unwrap())
                .unwrap();
            let mut scratch = [z(0., 0.); 4];
            let mut vectors = Matrix::from_vec(n, 2, vec![z(0., 0.); 2 * n]).unwrap();
            let mut transformed = Matrix::from_vec(n, 2, vec![z(0., 0.); 2 * n]).unwrap();
            for (output, input) in [(&mut vectors, &basis), (&mut transformed, &images)] {
                combine_block(
                    output.view_mut(),
                    input,
                    |i, j| factor[i * 2 + j],
                    &mut scratch,
                    &mut workspace,
                    ExecutionPolicy::Sequential,
                    KernelContext::from(crate::kernel::tests::test_context())
                        .with_accuracy(accuracy),
                )
                .unwrap();
            }
            for column in 0..2 {
                let v = vectors.view().col(column).unwrap();
                let norm = v
                    .norm2_scoped(
                        &options,
                        crate::LinalgExecutionContext::unscoped(KernelContext::sequential()),
                    )
                    .unwrap();
                assert!((norm - 1.).abs() < 1e-14);
                for row in 0..n {
                    let expected = basis[(1 - column) * n + row]
                        * if column == 0 { z(0., -1.) } else { z(0., 1.) };
                    assert!((vectors.get_element(row, column) - expected).norm() < 1e-14);
                    let scale = if row == 0 { 2. } else { 5. };
                    assert!(
                        (transformed.get_element(row, column) - expected * scale).norm() < 1e-14
                    );
                }
            }
            assert!(
                vectors
                    .view()
                    .col(0)
                    .unwrap()
                    .dotc_left_scoped(
                        vectors.view().col(1).unwrap(),
                        &options,
                        crate::LinalgExecutionContext::unscoped(KernelContext::sequential())
                    )
                    .unwrap()
                    .norm()
                    < 1e-14
            );
        }
    }

    #[test]
    fn block_ritz_extreme_products_preserve_cancellation_and_reject_overflow() {
        let mut workspace = crate::LinalgWorkspace::<f64>::new();
        workspace
            .try_reserve(
                &block_requirements::<f64>(1, 2, 2, &crate::LinalgExecutionOptions::new()).unwrap(),
            )
            .unwrap();
        let mut scratch = [0.; 4];
        let mut output = [0.; 2];
        combine_block(
            crate::MatrixViewMut::from_slice(1, 2, &mut output).unwrap(),
            &[f64::MAX, -f64::MAX],
            |_, j| if j == 0 { 2. } else { f64::MIN_POSITIVE },
            &mut scratch,
            &mut workspace,
            scioxide_core::ExecutionPolicy::Sequential,
            KernelContext::sequential(),
        )
        .unwrap();
        assert_eq!(output, [0., 0.]);
        combine_block(
            crate::MatrixViewMut::from_slice(1, 2, &mut output).unwrap(),
            &[f64::MAX, f64::MIN_POSITIVE],
            |i, j| [[f64::MIN_POSITIVE, 0.], [f64::MAX, 1.]][i][j],
            &mut scratch,
            &mut workspace,
            scioxide_core::ExecutionPolicy::Sequential,
            KernelContext::sequential(),
        )
        .unwrap();
        assert!((output[0] - 8.).abs() <= 8. * f64::EPSILON);
        assert_eq!(output[1], f64::MIN_POSITIVE);
        assert!(matches!(
            combine_block(
                crate::MatrixViewMut::from_slice(1, 2, &mut output).unwrap(),
                &[f64::MAX, f64::MAX],
                |_, _| 2.,
                &mut scratch,
                &mut workspace,
                scioxide_core::ExecutionPolicy::Sequential,
                KernelContext::sequential()
            )
            .unwrap_err()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }

    #[test]
    fn dgks_accumulates_coefficients_and_preserves_paired_images() {
        let q = [0.6_f64, 0.8, 0.0];
        let mut vector = [0.6, 0.8, 1e-10];
        let mut image = [1.2, 2.4, 4e-10];
        let aq = [1.2, 2.4, 0.0];
        let mut scratch = [0.0];
        let mut coefficient = 0.0;
        let mut passes = 0;
        let length = orthogonalize(
            &mut vector,
            Columns::Packed(&q, 3),
            &mut scratch,
            None,
            Some((&mut image, Columns::Packed(&aq, 3))),
            |_, c| {
                coefficient += c;
                passes += 1;
            },
            KernelContext::sequential(),
        )
        .unwrap();
        assert_eq!(passes, 2);
        assert!((coefficient - 1.).abs() < 1e-15);
        assert!((length - 1e-10).abs() < 1e-24);
        for ((v, image), factor) in vector.into_iter().zip(image).zip([2., 3., 4.]) {
            assert!((image - factor * v).abs() < 1e-15);
        }
        let vectors = [Vector::from_vec(q.to_vec())];
        let mut orthogonal = [0., 0., 1.];
        let mut passes = 0;
        orthogonalize(
            &mut orthogonal,
            Columns::Vectors(&vectors),
            &mut scratch,
            None,
            None,
            |_, _| passes += 1,
            KernelContext::sequential(),
        )
        .unwrap();
        assert_eq!(passes, 1);
    }

    #[test]
    fn complex_mass_projection_carries_the_same_transform_to_both_images() {
        let z = |r, i| Complex::new(r, i);
        let q = [z(0., 0.5), z(0., 0.)];
        let bq = [z(0., 2.), z(0., 0.)];
        let aq = [z(0., 1.5), z(0., 0.)];
        let mut vector = [z(1., 1.), z(0.5, -0.25)];
        let mut bv = [z(4., 4.), z(1., -0.5)];
        let mut av = [z(3., 3.), z(2.5, -1.25)];
        let mut scratch = [z(0., 0.)];
        orthogonalize(
            &mut vector,
            Columns::Packed(&q, 2),
            &mut scratch,
            Some((&mut bv, Columns::Packed(&bq, 2))),
            Some((&mut av, Columns::Packed(&aq, 2))),
            |_, _| {},
            KernelContext::sequential(),
        )
        .unwrap();
        assert!(vector[0].norm() < 1e-15);
        for i in 0..2 {
            assert!((bv[i] - vector[i] * [4., 2.][i]).norm() < 1e-15);
            assert!((av[i] - vector[i] * [3., 5.][i]).norm() < 1e-15);
        }
    }
}
