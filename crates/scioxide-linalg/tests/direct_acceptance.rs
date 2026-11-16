//! Direct-style acceptance gates owned accessors and atomic borrowed publication.
#![allow(clippy::unwrap_used)]

use scioxide_core::{DiagonalKind, ExecutionContext, ResidualTolerance, Tolerance, Triangle};
use scioxide_linalg::*;

#[test]
fn factor_rejection_hides_owned_solutions_and_preserves_vector_and_matrix_outputs() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0., 0., 4.]).unwrap();
    let rhs = [1., 4.];
    let rhs_matrix = Matrix::from_vec(2, 2, vec![1., 2., 4., 8.]).unwrap();
    let accepted = DirectSolveOptions::new(RankPolicy::Exact);
    let rejected =
        accepted.with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.9).unwrap());
    macro_rules! check {
        ($factor:expr $(, $operation:expr)?) => {{
            let factor = $factor;
            let result = factor.solve_with_options($($operation,)? &rhs[..], &rejected).unwrap();
            assert_eq!(result.report().termination(), DirectSolveTermination::IllConditioned);
            assert!(result.solution().is_none());
            assert!(result.into_solution().is_none());
            let result = factor.solve_matrix_with_options($($operation,)? &rhs_matrix, &rejected).unwrap();
            assert!(result.solution().is_none());
            assert!(result.into_solution().is_none());

            let mut workspace = LinalgWorkspace::new();
            let mut vector = [91., 92.];
            let report = factor.solve_into(
                $($operation,)? &rhs[..], &rejected, VectorViewMut::from_slice(&mut vector),
                &mut workspace, ExecutionContext::sequential(),
            ).unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::IllConditioned);
            assert_eq!(vector, [91., 92.]);
            let mut matrix = Matrix::from_vec(2, 2, vec![91.; 4]).unwrap();
            let report = factor.solve_matrix_into(
                $($operation,)? &rhs_matrix, &rejected, matrix.view_mut(),
                &mut workspace, ExecutionContext::sequential(),
            ).unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::IllConditioned);
            assert_eq!(matrix.as_slice(), &[91.; 4]);

            let report = factor.solve_into(
                $($operation,)? &rhs[..], &accepted, VectorViewMut::from_slice(&mut vector),
                &mut workspace, ExecutionContext::sequential(),
            ).unwrap();
            assert!(report.termination().is_success());
            assert!(vector.iter().all(|x| (x - 1.).abs() < 1e-14));
        }};
    }
    check!(LuDecomposition::factor(&a).unwrap());
    check!(
        CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None).unwrap()
    );
    check!(LdlDecomposition::factor(&a, HermitianInput::Exact).unwrap());
    check!(
        HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column
        )
        .unwrap()
    );
    let band = BandedMatrix::new(2, 2, 0, 0, 1, vec![1., 4.]).unwrap();
    check!(
        BandedLuDecomposition::factor(&band).unwrap(),
        StructuredOperation::Normal
    );
    let hermitian = HermitianBandMatrix::new(2, 0, Triangle::Lower, vec![1., 4.]).unwrap();
    check!(
        BandedCholeskyDecomposition::factor(&hermitian).unwrap(),
        StructuredOperation::Normal
    );
    let tridiagonal = TridiagonalMatrix::new(vec![0.], vec![1., 4.], vec![0.]).unwrap();
    check!(
        TridiagonalLuDecomposition::factor(&tridiagonal).unwrap(),
        StructuredOperation::Normal
    );
    check!(
        TriangularPlan::from_dense(a.view(), Triangle::Upper, DiagonalKind::Explicit).unwrap(),
        StructuredOperation::Normal
    );
}

#[test]
fn matrix_equation_rejection_preserves_outputs_and_consuming_accessors() {
    let a = Matrix::from_vec(2, 2, vec![-2.0_f64, 0.3, 0.2, -3.]).unwrap();
    let b = Matrix::identity(2).unwrap();
    let rhs = Matrix::from_vec(2, 2, vec![1., 0.2, 0.2, 2.]).unwrap();
    let strict =
        MatrixEquationOptions::new().with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    macro_rules! check {
        ($build:expr) => {{
            let build = $build;
            let plan = build(strict).unwrap();
            let result = plan.solve(&rhs).unwrap();
            assert_eq!(
                result.report().termination(),
                DirectSolveTermination::ResidualRejected
            );
            assert!(result.report().residual() > 0.);
            assert!(result.solution().is_none());
            let (solution, report) = result.into_parts();
            assert!(solution.is_none());
            assert_eq!(
                report.termination(),
                DirectSolveTermination::ResidualRejected
            );

            let mut output = Matrix::from_vec(2, 2, vec![79.; 4]).unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            let report = plan
                .solve_into(
                    &rhs,
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                DirectSolveTermination::ResidualRejected
            );
            assert_eq!(output.as_slice(), &[79.; 4]);
            let plan = build(MatrixEquationOptions::new()).unwrap();
            let report = plan
                .solve_into(
                    &rhs,
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.termination().is_success());
            let (solution, report) = plan.solve(&rhs).unwrap().into_parts();
            assert!(report.termination().is_success());
            assert!(
                solution
                    .unwrap()
                    .as_slice()
                    .iter()
                    .zip(output.as_slice())
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
        }};
    }
    check!(|options| SylvesterPlan::new(&a, &b, SylvesterSign::Plus, options));
    check!(|options| LyapunovPlan::new(&a, LyapunovKind::Continuous, options));
    check!(|options| LyapunovPlan::new(&a, LyapunovKind::Discrete, options));
}

#[test]
fn composite_rejection_retains_diagnostics_and_allocation_free_outputs() {
    let a = Matrix::<f64>::identity(2).unwrap();
    let finite = a.view().try_finite().unwrap();
    let bad = Matrix::from_vec(2, 2, vec![1., 0., 0., 2.]).unwrap();
    let bad = bad.view().try_finite().unwrap();
    let options = CompositeSolveOptions::new().with_tolerance(Tolerance::new(0., 0.).unwrap());
    let kron = KroneckerOperator::new(&finite, &finite).unwrap();
    let mut workspace = scioxide_linalg::LinalgWorkspace::<f64>::workspace(
        &scioxide_linalg::OperatorWorkspaceRequest::kronecker_solve(
            &kron,
            [&finite, &bad],
            &options.execution_options(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut output = [73.; 4];
    let report = kron
        .solve_into(
            [&finite, &bad],
            &[1.; 4][..],
            &options,
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert!((report.backward_error() - 1. / 3.).abs() < 1e-15);
    assert_eq!(output, [73.; 4]);
    let report = kron
        .solve_into(
            [&finite, &finite],
            &[1.; 4][..],
            &options,
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.termination().is_success());
    assert_eq!(output, [1.; 4]);

    let blocks: [&(dyn scioxide_core::AdjointOperator<f64> + Sync); 2] = [&finite, &finite];
    let block = BlockDiagonalOperator::new(&blocks).unwrap();
    let mut workspace = scioxide_linalg::LinalgWorkspace::<f64>::workspace(
        &scioxide_linalg::OperatorWorkspaceRequest::block_diagonal_solve(
            &block,
            &[&finite, &bad],
            &options.execution_options(),
        )
        .unwrap(),
    )
    .unwrap();
    let report = block
        .solve_into(
            &[&finite, &bad],
            &[1.; 4][..],
            &options,
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert!((report.backward_error() - 1. / 3.).abs() < 1e-15);
    assert_eq!(output, [1.; 4]);
    let report = block
        .solve_into(
            &[&finite, &finite],
            &[2.; 4][..],
            &options,
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.termination().is_success());
    assert_eq!(output, [2.; 4]);
}

#[test]
fn empty_equations_accept_zero_tolerance() {
    let empty = Matrix::<f64>::zeros(0, 0).unwrap();
    let options =
        MatrixEquationOptions::new().with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    let plan = SylvesterPlan::new(&empty, &empty, SylvesterSign::Plus, options).unwrap();
    let result = plan.solve(&empty).unwrap();
    assert!(result.report().termination().is_success());
    assert!(result.solution().unwrap().as_slice().is_empty());
    assert!(result.into_parts().0.is_some());
}

#[test]
fn nested_breakdown_can_be_matched_exhaustively_by_consumers() {
    fn category(reason: LinearSolveBreakdown) -> usize {
        match reason {
            LinearSolveBreakdown::RecurrenceDenominator => 0,
            LinearSolveBreakdown::NonPositiveCurvature => 1,
            LinearSolveBreakdown::ArnoldiInvariant => 2,
            LinearSolveBreakdown::Preconditioner => 3,
        }
    }
    assert_eq!(category(LinearSolveBreakdown::Preconditioner), 3);
}

#[test]
fn circulant_residual_rejection_preserves_output_without_a_gko_path() {
    let column = [2.0_f64, 0.1875, -0.25];
    let matrix = StructuredMatrixView::circulant(&column).unwrap();
    let options = StructuredSolveOptions::new().with_tolerance(Tolerance::new(0., 0.).unwrap());
    let plan = StructuredSolvePlan::new(matrix, &options).unwrap();
    let request = StructuredWorkspaceRequest::solve(3, &LinalgExecutionOptions::new());
    let mut workspace = plan.workspace(&request).unwrap();
    let mut output = [81.; 3];
    let rhs = [4., -6. / 11., 14. / 13.];
    let report = plan
        .solve_into(
            StructuredOperation::Normal,
            &rhs[..],
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert!(report.backward_error() > 0.);
    assert_eq!(report.circulant_solves(), 0);
    assert_eq!(output, [81.; 3]);
    let report = plan
        .solve_into(
            StructuredOperation::Normal,
            &[0.; 3][..],
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.termination().is_success());
    assert_eq!(output, [0.; 3]);
}
