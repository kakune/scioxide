//! Closure adapters for linear operators, adjoints and preconditioners.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AdjointOperator, BlockOperator, BlockVectorView, BlockVectorViewMut, ClosureOperator,
    ClosurePreconditioner, Complex64, LinearOperator, NonFiniteOrigin, NumericScalar,
    Preconditioner, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};

// A = [[1, 2, 0], [0, 1, 3]] (2 × 3) as an operator with its transpose.
fn operator<S: NumericScalar>() -> impl AdjointOperator<S> + BlockOperator<S> {
    let two = S::ONE + S::ONE;
    let three = two + S::ONE;
    ClosureOperator::<S, _>::new(2, 3, move |x, y| {
        y[0] = x[0] + two * x[1];
        y[1] = x[1] + three * x[2];
        Ok(())
    })
    .with_adjoint(move |x, y| {
        y[0] = x[0];
        y[1] = two * x[0] + x[1];
        y[2] = three * x[1];
        Ok(())
    })
}

fn applications<S: NumericScalar>(value: impl Fn(f64) -> S) {
    let a = operator::<S>();
    let input = [1.0, 2.0, 3.0].map(&value);
    let mut output = [S::ZERO; 3];
    let mut workspace = [S::ZERO; 3];
    assert_eq!(a.apply_workspace_len().unwrap(), 2);
    assert_eq!(a.apply_adjoint_workspace_len().unwrap(), 3);
    {
        a.apply_into(&input, &mut output[..2], &mut workspace[..2])
            .unwrap();
    };
    assert_eq!(output[..2], [5.0, 11.0].map(&value));
    {
        a.apply_adjoint_into(&input[..2], &mut output, &mut workspace)
            .unwrap();
    };
    assert_eq!(output, [1.0, 4.0, 6.0].map(&value));
    let m = ClosurePreconditioner::<S, _>::new(2, |r, z| {
        z.copy_from_slice(r);
        z[1] = z[1] + z[1];
        Ok(())
    });
    assert_eq!(m.precondition_workspace_len().unwrap(), 2);
    {
        m.precondition_into(&input[..2], &mut output[..2], &mut workspace[..2])
            .unwrap();
    };
    assert_eq!(output[..2], [1.0, 4.0].map(&value));
}

#[test]
fn closures_without_annotations_apply_for_every_scalar() {
    applications::<f32>(|x| x as f32);
    applications::<f64>(|x| x);
    applications::<Complex64>(|x| Complex64::new(x, -x));
    let copy = ClosureOperator::<f64, _>::new(2, 2, |x, y| {
        y.copy_from_slice(x);
        Ok(())
    });
    let mut y = [0.0; 2];
    copy.apply_into(&[1.0, 2.0], &mut y, &mut [0.0; 2]).unwrap();
    assert_eq!(y, [1.0, 2.0]);
}

#[test]
fn wrong_lengths_are_dimension_mismatches_and_leave_output_unchanged() {
    let a = operator::<f64>();
    let m = ClosurePreconditioner::<f64, _>::new(2, |r, z| {
        z.copy_from_slice(r);
        Ok(())
    });
    let sentinel = [77.0; 4];
    {
        for (input, output, workspace) in [(2, 2, 2), (3, 3, 2), (3, 2, 3), (3, 2, 1)] {
            let mut y = sentinel;
            let error = a
                .apply_into(
                    &[1.0; 4][..input],
                    &mut y[..output],
                    &mut [0.0; 4][..workspace],
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(y, sentinel);
        }
        for (input, output, workspace) in [(3, 3, 3), (2, 2, 3), (2, 3, 2), (2, 3, 4)] {
            let mut y = sentinel;
            let error = a
                .apply_adjoint_into(
                    &[1.0; 4][..input],
                    &mut y[..output],
                    &mut [0.0; 4][..workspace],
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(y, sentinel);
        }
        for (input, output, workspace) in [(1, 2, 2), (2, 3, 2), (2, 2, 1)] {
            let mut z = sentinel;
            let error = m
                .precondition_into(
                    &[1.0; 4][..input],
                    &mut z[..output],
                    &mut [0.0; 4][..workspace],
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(z, sentinel);
        }
    };
}

fn failing(value: f64) -> impl Fn(&[f64], &mut [f64]) -> Result<(), SciOxideError> {
    move |_, output| {
        output.fill(value);
        if value == 0.0 {
            Err(SciOxideError::external("model failed"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn closure_errors_and_non_finite_values_leave_output_unchanged() {
    for value in [0.0, f64::NAN, f64::INFINITY] {
        let a = ClosureOperator::new(1, 1, failing(value)).with_adjoint(failing(value));
        let m = ClosurePreconditioner::new(1, failing(value));
        let errors = {
            let mut y = [77.0];
            let errors = [
                a.apply_into(&[1.0], &mut y, &mut [0.0]).unwrap_err(),
                a.apply_adjoint_into(&[1.0], &mut y, &mut [0.0])
                    .unwrap_err(),
                m.precondition_into(&[1.0], &mut y, &mut [0.0]).unwrap_err(),
            ];
            assert_eq!(y, [77.0]);
            errors
        };
        for (index, error) in errors.into_iter().enumerate() {
            if value == 0.0 {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::External { detail } if detail == "model failed"
                ));
            } else {
                let expected = if index < 2 {
                    "operator callback returned a non-finite component"
                } else {
                    "preconditioner callback returned a non-finite component"
                };
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Callback,
                        detail,
                    } if detail == expected
                ));
            }
        }
    }
}

#[test]
fn blocks_use_the_default_column_loop() {
    let a = operator::<f64>();
    let input = [1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
    let mut output = [0.0; 4];
    let mut workspace = [0.0; 6];
    assert_eq!(a.apply_block_workspace_len(2).unwrap(), 6);
    {
        let input = BlockVectorView::new(&input, 3, 2).unwrap();
        let mut output = BlockVectorViewMut::new(&mut output, 2, 2).unwrap();
        a.apply_block_into(input, &mut output, &mut workspace)
            .unwrap();
    };
    assert_eq!(output, [5.0, 11.0, 2.0, 1.0]);
}

#[test]
fn adapters_coerce_to_trait_objects() {
    let a = operator::<f64>();
    let m = ClosurePreconditioner::<f64, _>::new(2, |r, z| {
        z.copy_from_slice(r);
        Ok(())
    });
    let linear: &dyn LinearOperator<f64> = &a;
    let adjoint: &dyn AdjointOperator<f64> = &a;
    let preconditioner: &dyn Preconditioner<f64> = &m;
    assert_eq!((linear.rows(), linear.cols()), (2, 3));
    assert_eq!((adjoint.rows(), adjoint.cols()), (2, 3));
    assert_eq!(preconditioner.dimension(), 2);
    let mut y = [0.0; 3];
    adjoint
        .apply_adjoint_into(&[1.0, 1.0], &mut y, &mut [0.0; 3])
        .unwrap();
    assert_eq!(y, [1.0, 3.0, 3.0]);
}
