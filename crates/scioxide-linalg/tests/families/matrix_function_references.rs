//! Independent dense non-normal references, including real Schur pairs.
#![allow(clippy::unwrap_used)]
use crate::support::function_reference as reference;
use scioxide_core::{Complex, ExecutionContext, NumericScalar};
use scioxide_linalg::{Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan};

fn check<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let mut count = 0;
    for fields in reference::records::<S>(include_str!("../data/matrix_functions.tsv")) {
        assert_eq!(fields.len(), 9);
        let n = fields[4].parse().unwrap();
        let a = Matrix::<S>::from_vec(n, n, reference::values(fields[5])).unwrap();
        let function = match fields[3] {
            "exp" => MatrixFunction::Exp,
            "log" => MatrixFunction::Log,
            "sqrt" => MatrixFunction::Sqrt,
            _ => unreachable!(),
        };
        let expected = reference::expected::<S>(fields[6], fields[7]);
        let plan =
            MatrixFunctionPlan::new(&a, function, &MatrixFunctionOptions::default()).unwrap();
        let mut workspace = plan
            .workspace(&scioxide_linalg::MatrixFunctionWorkspaceRequest::Evaluate)
            .unwrap();
        let mut output = Matrix::<S>::zeros(n, n).unwrap();
        for context in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            {
                plan.evaluate_into(output.view_mut(), &mut workspace, context)
                    .unwrap();
            };
            reference::assert_close(output.as_slice(), &expected, n, fields[8], fields[2]);
        }
        count += 1;
    }
    assert_eq!(count, 15);
}

#[test]
fn independent_dense_matrix_functions() {
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
