//! Public error kinds, structured details, borrowed extents and publication.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::*;
use scioxide_sparse::*;
use scioxide_sparse::{GraphPlan, SchwarzOptions, SparseProductPlanOptions};
mod amg_support;
const CX: ExecutionContext<'static> = ExecutionContext::sequential();
fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4., -1., -1., 4.],
    )
    .unwrap()
}
fn kind<T>(result: Result<T, SciOxideError>, expected: SciOxideErrorKind) -> SciOxideError {
    let Err(error) = result else {
        panic!("expected an error")
    };
    assert_eq!(error.kind(), expected, "{error}");
    error
}
fn short(result: Result<(), SciOxideError>, output: &[f64]) {
    let _ = kind(result, SciOxideErrorKind::DimensionMismatch);
    assert!(output.iter().all(|v| *v == 19.));
}
fn resource<T>(result: Result<T, SciOxideError>) {
    let e = kind(result, SciOxideErrorKind::ResourceLimit);
    assert!(
        matches!(e.view(),SciOxideErrorView::ResourceLimit {required,limit,..} if required>limit)
    );
}
fn limited<T>(
    budget_kind: BudgetKind,
    call: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(budget_kind, Budget::ONE);
    let id = SolveIdentity::new(
        ProblemToken::new(1755),
        algorithm_id!("sparse", "classification", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(CX, id, &mut budgets).unwrap();
    call(&mut context)
}

#[test]
fn storage_triangular_and_jacobi_share_exact_vectors_and_minimum_scratch() {
    let a = matrix();
    let csc = a.to_csc().unwrap();
    let coo = a.to_coo().unwrap();
    let bsr = BsrMatrix::from_sparse(a.view(), 2, 2).unwrap();
    let dia = DiaMatrix::from_sparse(a.view()).unwrap();
    let view = a.view();
    let coo_view = coo.view();
    for op in [
        &a as &dyn LinearOperator<f64>,
        &csc,
        &coo_view,
        &bsr,
        &dia,
        &view,
    ] {
        let required = op.apply_workspace_len().unwrap();
        let mut out = [19.; 2];
        short(
            op.apply_into(&[1.; 2], &mut out, &mut vec![0.; required - 1]),
            &out,
        );
        short(
            op.apply_into(&[1.], &mut out, &mut vec![0.; required]),
            &out,
        );
        let mut long_output = [19.; 3];
        short(
            op.apply_into(&[1.; 2], &mut long_output, &mut vec![0.; required]),
            &long_output,
        );
        let mut work = vec![71.; required + 3];
        op.apply_into(&[1.; 2], &mut out, &mut work).unwrap();
        assert_eq!(out, [3.; 2]);
        assert_eq!(&work[required..], &[71.; 3]);
    }
    let triangle = SparseTriangular::new(
        CsrMatrix::from_raw(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![4., -1., 4.]).unwrap(),
        Triangle::Lower,
        DiagonalKind::Explicit,
    )
    .unwrap();
    let jacobi = JacobiPreconditioner::new(&a).unwrap();
    let mut out = [19.; 2];
    short(
        triangle.solve_into(
            &[4., 3.],
            scioxide_sparse::SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 1]),
            scioxide_core::ExecutionContext::sequential(),
        ),
        &out,
    );
    short(
        triangle.solve_into(
            &[4.],
            scioxide_sparse::SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 2]),
            scioxide_core::ExecutionContext::sequential(),
        ),
        &out,
    );
    let mut work = [71.; 7];
    triangle
        .solve_into(
            &[4., 3.],
            scioxide_sparse::SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut work),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out, [1.; 2]);
    assert_eq!(&work[4..], &[71.; 3]);
    out.fill(19.);
    short(
        jacobi.precondition_into(&[1.; 2], &mut out, &mut [0.; 1]),
        &out,
    );
    short(
        jacobi.precondition_into(&[1.], &mut out, &mut [0.; 2]),
        &out,
    );
    jacobi
        .precondition_into(&[1.; 2], &mut out, &mut work)
        .unwrap();
    assert_eq!(out, [0.25; 2]);
    assert_eq!(&work[4..], &[71.; 3]);
    IdentityPreconditioner::new(2)
        .precondition_into(&[1.; 2], &mut out, &mut work)
        .unwrap();
    let mut bad = a.clone();
    let e = kind(bad.set_value(0, f64::NAN), SciOxideErrorKind::NonFinite);
    assert!(matches!(
        e.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn products_and_arithmetic_classify_dimensions_and_limits() {
    let a = matrix();
    let rectangular = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![0], vec![1.]).unwrap();
    let _ = kind(
        a.add_with_options(
            &rectangular,
            scioxide_sparse::SparseArithmeticOptions::new(),
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    let _ = kind(
        SparseProductPlan::new(&a, &rectangular),
        SciOxideErrorKind::DimensionMismatch,
    );
    resource(SparseProductPlan::with_options(
        &a,
        &a,
        SparseProductPlanOptions::new().with_max_output_entries(0),
    ));
    let plan = SparseProductPlan::new(&a, &a).unwrap();
    let mut out = plan.create_output::<f64>().unwrap();
    out.copy_values_from(&vec![19.; out.values().len()])
        .unwrap();
    let _ = kind(
        plan.evaluate_into(
            &a,
            &a,
            Default::default(),
            &mut out,
            &mut scioxide_sparse::SparseProductWorkspace::default(),
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    assert!(out.values().iter().all(|v| *v == 19.));
    let _ = kind(
        plan.evaluate_into(
            &rectangular,
            &a,
            Default::default(),
            &mut out,
            &mut plan.workspace(Default::default()).unwrap(),
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    assert!(out.values().iter().all(|v| *v == 19.));
    // Larger retained capacity from another product serves the plan unchanged.
    let larger = scioxide_sparse::SparseProductRequirements::sparse_dense(
        a.view(),
        SparseOperation::Normal,
        8,
    )
    .unwrap();
    let mut work = plan.workspace(Default::default()).unwrap();
    work.try_reserve(&larger).unwrap();
    plan.evaluate_into(&a, &a, Default::default(), &mut out, &mut work, CX)
        .unwrap();
    assert_eq!(work.check(&larger).unwrap(), Capacity::Sufficient);
    // Same shape, wrong pattern remains malformed input and names the operand.
    let diagonal = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.; 2]).unwrap();
    let before = out.values().to_vec();
    let e = kind(
        plan.evaluate_into(&diagonal, &a, Default::default(), &mut out, &mut work, CX),
        SciOxideErrorKind::InvalidInput,
    );
    assert!(
        matches!(e.view(),SciOxideErrorView::InvalidInput{argument,..} if argument==argument!("left"))
    );
    assert_eq!(out.values(), before);
}

#[test]
fn direct_factors_limits_geometry_work_budgets_and_rollback() {
    let a = matrix();
    let pattern = a.structural_pattern();
    let lu_limits = SparseLuLimits::new(8, usize::MAX);
    let sym_limits = SparseSymmetricLimits::new(8, 8, usize::MAX);
    let qr_limits = SparseQrLimits::new(16, 8, usize::MAX);
    let inc_limits = IncompleteLimits::new(8, usize::MAX);
    resource(SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        lu_limits.with_max_workspace_bytes(0),
    ));
    resource(SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        None,
        sym_limits.with_max_workspace_bytes(0),
    ));
    resource(SparseQrSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        qr_limits.with_max_workspace_bytes(0),
    ));
    resource(IncompleteSymbolic::analyze(
        pattern,
        IncompleteMethod::Ilu { level: 0 },
        1,
        inc_limits.with_max_workspace_bytes(0),
    ));
    let lu_plan = SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        lu_limits,
    )
    .unwrap();
    let sym_plan = SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        None,
        sym_limits,
    )
    .unwrap();
    let qr_plan = SparseQrSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        qr_limits,
    )
    .unwrap();
    let inc_plan =
        IncompleteSymbolic::analyze(pattern, IncompleteMethod::Ilu { level: 0 }, 1, inc_limits)
            .unwrap();
    let options = IncompleteOptions::new().with_max_off_diagonal(2);
    let mut lu = SparseLu::factor_scoped(
        lu_plan.clone(),
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        CX,
    )
    .unwrap();
    let mut chol = SparseCholesky::factor_scoped(
        sym_plan.clone(),
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        CX,
    )
    .unwrap();
    let mut ldl = SparseLdl::factor_scoped(
        sym_plan.clone(),
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        CX,
    )
    .unwrap();
    let mut qr = SparseQr::factor_scoped(
        qr_plan.clone(),
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        CX,
    )
    .unwrap();
    let mut inc = IncompleteFactor::factor_scoped(
        inc_plan.clone(),
        a.view(),
        options,
        scioxide_sparse::SparseSolveOptions::new(),
        CX,
    )
    .unwrap();
    let mut lu_work = (lu_plan).workspace(()).unwrap();
    let mut sym_work = (sym_plan).workspace(()).unwrap();
    let mut qr_work = qr_plan.workspace(()).unwrap();
    let threshold = RankThreshold::new(0., 1e-12).unwrap();
    let mut qr_solve_work =
        SparseQrMinimumNormPlan::build(qr.clone(), threshold, SparseSolveOptions::new(), CX)
            .unwrap()
            .workspace(1)
            .unwrap();
    let mut inc_work = inc_plan.workspace(()).unwrap();
    let mut out = [19.; 2];
    macro_rules! check_solve {
        ($factor:expr) => {{
            let factor = &$factor;
            let n = factor.workspace_requirements(1).unwrap().scalar_len();
            short(
                factor.solve_into(
                    &[3.; 2],
                    SparseOperation::Normal,
                    scioxide_sparse::SparseSolveOptions::new(),
                    &mut out,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut vec![0.; n - 1]),
                    CX,
                ),
                &out,
            );
            short(
                factor.solve_into(
                    &[3.],
                    SparseOperation::Normal,
                    scioxide_sparse::SparseSolveOptions::new(),
                    &mut out,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut vec![0.; n]),
                    CX,
                ),
                &out,
            );
            let mut work = vec![71.; n + 3];
            factor
                .solve_into(
                    &[3.; 2],
                    SparseOperation::Normal,
                    scioxide_sparse::SparseSolveOptions::new(),
                    &mut out,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut work),
                    CX,
                )
                .unwrap();
            assert!(out.iter().all(|v| (*v - 1.).abs() < 1e-13));
            assert_eq!(&work[n..], &[71.; 3]);
            out.fill(19.);
        }};
    }
    check_solve!(lu);
    check_solve!(chol);
    check_solve!(ldl);
    check_solve!(inc);
    macro_rules! check_qr {
        () => {{
            let plan = SparseQrMinimumNormPlan::build(
                qr.clone(),
                threshold,
                SparseSolveOptions::new(),
                CX,
            )
            .unwrap();
            let n = plan.workspace_requirements(1).unwrap().scalar_len();
            short(
                plan.solve_into(
                    &[3.; 2],
                    SparseOperation::Normal,
                    SparseSolveOptions::new(),
                    &mut out,
                    &mut SparseSolveWorkspace::borrowed(&mut vec![0.; n - 1]),
                    CX,
                )
                .map(drop),
                &out,
            );
            plan.solve_into(
                &[3.; 2],
                SparseOperation::Normal,
                SparseSolveOptions::new(),
                &mut out,
                &mut qr_solve_work,
                CX,
            )
            .unwrap();
            assert!(out.iter().all(|v| (*v - 1.).abs() < 1e-13));
            out.fill(19.);
        }};
    }
    check_qr!();
    short(
        qr.apply_q_into(
            &[1.; 2],
            SparseOperation::Normal,
            SparseSolveOptions::new(),
            &mut out,
            &mut SparseSolveWorkspace::borrowed(&mut [0.; 1]),
            CX,
        ),
        &out,
    );
    short(
        qr.apply_q_into(
            &[1.],
            SparseOperation::Normal,
            SparseSolveOptions::new(),
            &mut out,
            &mut SparseSolveWorkspace::borrowed(&mut [0.; 2]),
            CX,
        ),
        &out,
    );
    let _ = kind(
        qr.apply_q_into(
            &[f64::NAN, 1.],
            SparseOperation::Normal,
            SparseSolveOptions::new(),
            &mut out,
            &mut qr.workspace(1).unwrap(),
            CX,
        ),
        SciOxideErrorKind::NonFinite,
    );
    // A two-column block admits two iterations against a one-iteration allowance.
    let mut q_out = [19.; 4];
    let _ = kind(
        qr.apply_q_block_into(
            BlockVectorView::new(&[1.; 4], 2, 2).unwrap(),
            SparseOperation::Normal,
            SparseSolveOptions::new().with_budgets(
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
            ),
            &mut BlockVectorViewMut::new(&mut q_out, 2, 2).unwrap(),
            &mut qr.workspace(2).unwrap(),
            CX,
        ),
        SciOxideErrorKind::ConvergenceFailure,
    );
    assert_eq!(q_out, [19.; 4]);
    assert_eq!(out, [19.; 2]);
    macro_rules! stopped {
        ($call:expr) => {
            kind(
                limited(BudgetKind::Iteration, $call),
                SciOxideErrorKind::ConvergenceFailure,
            )
        };
    }
    let _ = stopped!(|cx| lu.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut lu_work,
        cx
    ));
    let _ = stopped!(|cx| chol.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut sym_work,
        cx
    ));
    let _ = stopped!(|cx| ldl.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut sym_work,
        cx
    ));
    let _ = stopped!(|cx| qr.refactor(
        a.view(),
        SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut qr_work,
        cx
    ));
    let _ = stopped!(|cx| inc.refactor(
        a.view(),
        SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut inc_work,
        cx
    ));
    check_solve!(lu);
    check_solve!(chol);
    check_solve!(ldl);
    check_solve!(inc);
    check_qr!();
    let rect = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![0], vec![1.]).unwrap();
    let _ = kind(
        lu.refactor(
            rect.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut lu_work,
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    let _ = kind(
        chol.refactor(
            rect.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut sym_work,
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    let _ = kind(
        ldl.refactor(
            rect.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut sym_work,
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    let _ = kind(
        qr.refactor(rect.view(), SparseSolveOptions::new(), &mut qr_work, CX),
        SciOxideErrorKind::DimensionMismatch,
    );
    let _ = kind(
        inc.refactor(
            rect.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut inc_work,
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    check_solve!(lu);
    check_solve!(chol);
    check_solve!(ldl);
    check_solve!(inc);
    check_qr!();
}

#[test]
fn amg_schwarz_and_graph_errors_are_classified() {
    // A local allowance of one unit of `budget`.
    let one = |budget| {
        SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(budget, Budget::ONE))
    };
    let a = amg_support::grid::<f64>(8, 1, 1.);
    let p = amg_support::preconditioner(&a, AmgCycleKind::V);
    let mut out = [19.; 8];
    short(
        p.apply_into(
            &[1.; 8],
            SparseSolveOptions::new(),
            &mut out,
            &mut SparseSolveWorkspace::borrowed(&mut []),
            CX,
        ),
        &out,
    );
    let mut work = vec![0.; p.precondition_workspace_len().unwrap() + 3];
    short(
        p.apply_into(
            &[1.; 7],
            SparseSolveOptions::new(),
            &mut out,
            &mut SparseSolveWorkspace::borrowed(&mut work),
            CX,
        ),
        &out,
    );
    // Each right-hand side admits one iteration, so two exceed an allowance of one.
    let mut block_out = [19.; 16];
    let mut block_work = vec![0.; 2 * p.precondition_workspace_len().unwrap()];
    let _ = kind(
        p.apply_block_into(
            BlockVectorView::new(&[1.; 16], 8, 2).unwrap(),
            one(BudgetKind::Iteration),
            &mut BlockVectorViewMut::new(&mut block_out, 8, 2).unwrap(),
            &mut SparseSolveWorkspace::borrowed(&mut block_work),
            CX,
        ),
        SciOxideErrorKind::ConvergenceFailure,
    );
    assert_eq!(block_out, [19.; 16]);
    let constant = BlockVectorView::new(&[1.; 8], 8, 1).unwrap();
    let build = |matrix: &CsrMatrix<f64>, modes, limits| {
        AmgHierarchy::build(
            matrix.view(),
            modes,
            AmgSetupOptions::new(),
            limits,
            SparseSolveOptions::new(),
            CX,
        )
    };
    resource(build(
        &a,
        constant,
        amg_support::limits().with_max_setup_bytes(0),
    ));
    let mut bad = a.clone();
    bad.set_value(0, -1.).unwrap();
    let e = kind(
        build(&bad, constant, amg_support::limits()),
        SciOxideErrorKind::InvalidInput,
    );
    assert!(
        matches!(e.view(),SciOxideErrorView::InvalidInput{argument,..} if argument==argument!("matrix"))
    );
    let _ = kind(
        build(
            &a,
            BlockVectorView::new(&[1.; 7], 7, 1).unwrap(),
            amg_support::limits(),
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    let local = IdentityPreconditioner::new(1);
    let domains = [
        SchwarzSubdomainView {
            indices: &[0],
            weights: Some(&[1.]),
            solve: SchwarzLocalSolveView::Preconditioner(&local),
        },
        SchwarzSubdomainView {
            indices: &[1],
            weights: Some(&[1.]),
            solve: SchwarzLocalSolveView::Preconditioner(&local),
        },
    ];
    let limits = SchwarzOptions::new();
    resource(SchwarzPlan::with_options(
        2,
        SchwarzKind::Additive,
        &domains,
        None,
        limits.with_max_plan_bytes(0),
    ));
    // Non-finite weights are non-finite input; finite weights outside [0, 1] are invalid.
    for (weight, expected) in [
        (f64::NAN, SciOxideErrorKind::NonFinite),
        (f64::INFINITY, SciOxideErrorKind::NonFinite),
        (2., SciOxideErrorKind::InvalidInput),
    ] {
        let weights = [weight];
        let domains = [domains[0], {
            let mut domain = domains[1];
            domain.weights = Some(&weights);
            domain
        }];
        let _ = kind(
            SchwarzPlan::with_options(2, SchwarzKind::Additive, &domains, None, limits),
            expected,
        );
    }
    let schwarz =
        SchwarzPlan::with_options(2, SchwarzKind::Additive, &domains, None, limits).unwrap();
    let mut out = [19.; 2];
    let mut work = vec![0.; schwarz.precondition_workspace_len().unwrap() + 3];
    short(schwarz.precondition_into(&[1.; 2], &mut out, &mut []), &out);
    short(schwarz.precondition_into(&[1.], &mut out, &mut work), &out);
    // Two right-hand sides exceed one iteration; each local solve records one
    // provider evaluation, so two exceed one evaluation.
    let mut block_out = [19.; 4];
    let mut block_work = vec![0.; 2 * schwarz.precondition_workspace_len().unwrap()];
    for budget in [BudgetKind::Iteration, BudgetKind::Evaluation] {
        let _ = kind(
            schwarz.apply_block_into(
                BlockVectorView::new(&[1.; 4], 2, 2).unwrap(),
                one(budget),
                &mut BlockVectorViewMut::new(&mut block_out, 2, 2).unwrap(),
                &mut SparseSolveWorkspace::borrowed(&mut block_work),
                CX,
            ),
            SciOxideErrorKind::ConvergenceFailure,
        );
        assert_eq!(block_out, [19.; 4]);
    }
    let graph = GraphPlan::new(a.view(), GraphDirection::Directed).unwrap();
    let mut workspace = GraphWorkspace::new();
    let _ = kind(
        graph.shortest_paths_scoped(
            GraphPathMethod::BellmanFord { source: 0 },
            GraphExecutionOptions::new(),
            &mut workspace,
            CX,
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
    workspace
        .try_reserve(
            &graph
                .workspace_requirements(GraphWorkspaceRequest::new(
                    GraphPathMethod::BellmanFord { source: 0 },
                    GraphExecutionOptions::new(),
                ))
                .unwrap(),
        )
        .unwrap();
    let _ = kind(
        limited(BudgetKind::Iteration, |cx| {
            graph
                .shortest_paths_scoped(
                    GraphPathMethod::BellmanFord { source: 0 },
                    GraphExecutionOptions::new().with_budgets(
                        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
                    ),
                    &mut workspace,
                    cx,
                )
                .map(|_| ())
        }),
        SciOxideErrorKind::ConvergenceFailure,
    );
}
