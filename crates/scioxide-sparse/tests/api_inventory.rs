//! The authoritative scioxide-sparse root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test
//! fails when this list and the `pub use` items of `src/lib.rs` differ. Exported
//! plans are asserted `Send + Sync + Clone`, and the canonical construction,
//! requirements and execution forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_sparse::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    AmgCycle, AmgCycleKind, AmgCycleOptions, AmgDiagnostics, AmgHierarchy, AmgLevelView, AmgLimits,
    AmgPreconditioner, AmgSetupOptions, AmgSetupTermination, AmgSmootherView, AssignmentObjective,
    AssignmentOutcome, AssignmentPlan, AssignmentResiduals, AssignmentView, AssignmentWorkspace,
    AssignmentWorkspaceRequirements, BipartiteMatching, BsrMatrix, CapacityEdge, CooMatrix,
    CscMatrix, CsrMatrix, DiaMatrix, DuplicatePolicy, ExplicitZeroPolicy, FlowView, FlowWorkspace,
    FlowWorkspaceRequirements, ForestEdge, FullAssignment, GraphComponents, GraphDirection,
    GraphExecutionOptions, GraphPathMethod, GraphPathResult, GraphPathView, GraphPlan,
    GraphTraversal, GraphWorkspace, GraphWorkspaceRequest, GraphWorkspaceRequirements,
    IdentityPreconditioner, IncompleteDiagnostics, IncompleteFactor, IncompleteLimits,
    IncompleteMethod, IncompleteOptions, IncompletePivotPolicy, IncompleteRequirements,
    IncompleteSymbolic, IncompleteWorkspace, JacobiPreconditioner, LaplacianDegree,
    LaplacianNormalization, LaplacianPlan, LdlInertia, LdlPivotBlock, MatchingView,
    MatchingWorkspace, MatchingWorkspaceRequirements, MaxFlowPlan, MaximumFlow,
    MinimumSpanningForest, NegativeCycle, PathOutcome, SchwarzKind, SchwarzLocalSolveView,
    SchwarzOperatorView, SchwarzOptions, SchwarzPlan, SchwarzSubdomainView, SparseArithmeticOptions,
    SparseBuilder, SparseCholesky, SparseEntries, SparseLdl, SparseLu, SparseLuLimits,
    SparseLuRequirements, SparseLuSymbolic, SparseLuWorkspace, SparseOperation, SparseOperator,
    SparseOrdering, SparseProductOptions, SparseProductPlan, SparseProductPlanOptions,
    SparseProductRequirements, SparseProductWorkspace, SparsePruning, SparseQr, SparseQrLimits,
    SparseQrMinimumNormPlan, SparseQrRequirements, SparseQrSolveReport, SparseQrSymbolic,
    SparseQrWorkspace, SparseSolveOptions, SparseSolveRequirements, SparseSolveWorkspace,
    SparseStorageRequirements, SparseSymmetricLimits, SparseSymmetricRequirements,
    SparseSymmetricSymbolic, SparseSymmetricWorkspace, SparseTraversal, SparseTraversalView,
    SparseTriangular, SparseValueRequirements, SparseValueWorkspace, SparseView,
    approximate_minimum_degree, connected_components, derivative_coloring, distance_one_coloring,
    laplacian, least_squares, maximum_bipartite_matching, maximum_bipartite_matching_scoped,
    maximum_bipartite_matching_with_options, maximum_flow, minimum_cost_assignment,
    reverse_cuthill_mckee, shortest_paths, solve, sparse_block_diagonal, sparse_block_grid,
    sparse_dense_into, sparse_diagonals, transpose_pattern,
}

// Cross-crate implementation helpers are reachable only under `backend`.
#[allow(unused_imports)]
use scioxide_sparse::backend::hermitian::{block_values, determinant, solve_block};

/// Asserts the plan auto traits for each listed instantiation and
/// records the plan names for the completeness test.
macro_rules! plans {
    ($($name:ident $(<$($argument:tt),*>)?),* $(,)?) => {
        const _: fn() = || {
            fn plan<T: Send + Sync + Clone>() {}
            $(plan::<$name $(<$($argument),*>)?>();)*
        };
        const PLANS: &[&str] = &[$(stringify!($name)),*];
    };
}

plans! {
    AssignmentPlan, GraphPlan<f64>, LaplacianPlan<f64>, MaxFlowPlan, SchwarzPlan<'static, f64>,
    SparseProductPlan, SparseQrMinimumNormPlan<f64>,
}

/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("\npub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn inventory_lists_every_root_export_once() {
    let listed: BTreeSet<_> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    let root = root_exports(include_str!("../src/lib.rs"));
    let unlisted: Vec<_> = root.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&root).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "update tests/api_inventory.rs\nunlisted: {unlisted:?}\nstale: {stale:?}"
    );
}

#[test]
fn every_exported_plan_is_send_sync_clone() {
    let exported: BTreeSet<_> = INVENTORY
        .iter()
        .copied()
        .filter(|name| name.ends_with("Plan"))
        .collect();
    let asserted: BTreeSet<_> = PLANS.iter().copied().collect();
    assert_eq!(
        exported, asserted,
        "update plans! in tests/api_inventory.rs"
    );
}

#[test]
fn root_exports_read_single_and_braced_statements() {
    let source = "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\nuse c::Hidden;\npub(crate) use d::Internal;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}

// One construction, requirements and execution form per operation. Scoped forms
// accept an `ExecutionContext` (root epoch) or `&mut SolveContext` (child epoch).
const _: fn() = || {
    use scioxide_core::{
        BlockVectorView, BlockVectorViewMut, Capacity, ExecutionContext, SciOxideError,
        SolveContext, StructuralPatternView,
    };
    type Cx = ExecutionContext<'static>;
    type Rhs = BlockVectorView<'static, f64>;
    type Out = BlockVectorViewMut<'static, f64>;
    type Ws = SparseSolveWorkspace<'static, f64>;
    type View = SparseView<'static, f64>;

    // Preconditioners share the factor solve workspace and scoped forms.
    let _: fn(&JacobiPreconditioner<f64>, usize) -> Result<SparseSolveRequirements, SciOxideError> =
        JacobiPreconditioner::workspace_requirements;
    let _: fn(&JacobiPreconditioner<f64>, usize) -> Result<Ws, SciOxideError> =
        JacobiPreconditioner::workspace;
    let _: fn(
        &JacobiPreconditioner<f64>,
        &[f64],
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = JacobiPreconditioner::apply_into;
    let _: fn(
        &IdentityPreconditioner<f64>,
        &[f64],
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = IdentityPreconditioner::apply_into;
    let _: fn(
        &AmgCycle<'static, f64>,
        &[f64],
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = AmgCycle::apply_into;
    let _: fn(
        &AmgPreconditioner<f64>,
        Rhs,
        SparseSolveOptions,
        &mut Out,
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = AmgPreconditioner::apply_block_into;
    let _: fn(
        &SchwarzPlan<'static, f64>,
        &[f64],
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SchwarzPlan::apply_into;
    let _ = |p: &SchwarzPlan<'_, f64>,
             rhs: &[f64],
             out: &mut [f64],
             ws: &mut SparseSolveWorkspace<'_, f64>,
             cx: &mut SolveContext<'_, '_>| {
        p.apply_into(rhs, SparseSolveOptions::new(), out, ws, cx)
    };

    // Preparation validates configuration once.
    let _: fn(
        View,
        Rhs,
        AmgSetupOptions<f64>,
        AmgLimits,
        SparseSolveOptions,
        Cx,
    ) -> Result<AmgHierarchy<f64>, SciOxideError> = AmgHierarchy::build;
    let _: fn(
        AmgHierarchy<f64>,
        AmgCycleOptions,
        SparseSymmetricLimits,
        SparseSolveOptions,
        Cx,
    ) -> Result<AmgPreconditioner<f64>, SciOxideError> = AmgPreconditioner::build;
    let _: fn(
        StructuralPatternView<'static>,
        IncompleteMethod,
        usize,
        IncompleteLimits,
    ) -> Result<IncompleteSymbolic, SciOxideError> = IncompleteSymbolic::analyze;
    let _: fn(
        IncompleteSymbolic,
        View,
        IncompleteOptions<f64>,
        SparseSolveOptions,
        Cx,
    ) -> Result<IncompleteFactor<f64>, SciOxideError> = IncompleteFactor::factor_scoped;
    let _: fn(
        &mut IncompleteFactor<f64>,
        View,
        SparseSolveOptions,
        &mut IncompleteWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = IncompleteFactor::refactor;
    let _: fn(&IncompleteSymbolic, ()) -> Result<IncompleteRequirements, SciOxideError> =
        IncompleteSymbolic::workspace_requirements::<f64>;
    let _: fn(&mut IncompleteWorkspace<f64>, &IncompleteRequirements) -> Result<(), SciOxideError> =
        IncompleteWorkspace::try_reserve;
    let _: fn(
        &IncompleteWorkspace<f64>,
        &IncompleteRequirements,
    ) -> Result<Capacity, SciOxideError> = IncompleteWorkspace::check;
    let _: fn(
        &IncompleteFactor<f64>,
        &[f64],
        SparseOperation,
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = IncompleteFactor::solve_into;

    // Products: the operation is positional and options carry arithmetic and policy.
    let _: fn(
        View,
        SparseOperation,
        Rhs,
        SparseProductOptions,
        &mut Out,
        &mut SparseProductWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = sparse_dense_into;
    let _: fn(View, SparseOperation, usize) -> Result<SparseProductRequirements, SciOxideError> =
        SparseProductRequirements::sparse_dense;
    let _: fn(&SparseProductPlan, SparseProductOptions) -> SparseProductRequirements =
        SparseProductPlan::workspace_requirements;
    let _: fn(
        &SparseProductPlan,
        &CsrMatrix<f64>,
        &CsrMatrix<f64>,
        SparseProductOptions,
        &mut CsrMatrix<f64>,
        &mut SparseProductWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = SparseProductPlan::evaluate_into;
    let _: fn(
        &mut SparseProductWorkspace<f64>,
        &SparseProductRequirements,
    ) -> Result<(), SciOxideError> = SparseProductWorkspace::try_reserve;
    let _: fn(
        &SparseProductWorkspace<f64>,
        &SparseProductRequirements,
    ) -> Result<Capacity, SciOxideError> = SparseProductWorkspace::check;

    // Direct factors and triangular solves.
    let _: fn(
        SparseLuSymbolic,
        View,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseLu<f64>, SciOxideError> = SparseLu::factor_scoped;
    let _: fn(
        &mut SparseLu<f64>,
        View,
        SparseSolveOptions,
        &mut SparseLuWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = SparseLu::refactor;
    let _: fn(
        &SparseLdl<f64>,
        Rhs,
        SparseOperation,
        SparseSolveOptions,
        &mut Out,
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SparseLdl::solve_block_into;
    let _: fn(
        &SparseTriangular<f64>,
        &[f64],
        SparseOperation,
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SparseTriangular::solve_into;

    // QR: one scoped factorization and preparation; Q actions and minimum-norm
    // solves share the factor solve workspace.
    let _: fn(&SparseQrSymbolic, ()) -> Result<SparseQrRequirements, SciOxideError> =
        SparseQrSymbolic::workspace_requirements::<f64>;
    let _: fn(&SparseQrSymbolic, ()) -> Result<SparseQrWorkspace<f64>, SciOxideError> =
        SparseQrSymbolic::workspace::<f64>;
    let _: fn(
        SparseQrSymbolic,
        View,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseQr<f64>, SciOxideError> = SparseQr::factor_scoped;
    let _: fn(
        &mut SparseQr<f64>,
        View,
        SparseSolveOptions,
        &mut SparseQrWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = SparseQr::refactor;
    let _: fn(&SparseQr<f64>, usize) -> Result<SparseSolveRequirements, SciOxideError> =
        SparseQr::workspace_requirements;
    let _: fn(
        &SparseQr<f64>,
        &[f64],
        SparseOperation,
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SparseQr::apply_q_into;
    let _: fn(
        &SparseQr<f64>,
        Rhs,
        SparseOperation,
        SparseSolveOptions,
        &mut Out,
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SparseQr::apply_q_block_into;
    let _: fn(
        SparseQr<f64>,
        scioxide_core::RankThreshold<f64>,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseQrMinimumNormPlan<f64>, SciOxideError> = SparseQrMinimumNormPlan::build;
    let _: fn(
        &SparseQrMinimumNormPlan<f64>,
        usize,
    ) -> Result<SparseSolveRequirements, SciOxideError> =
        SparseQrMinimumNormPlan::workspace_requirements;
    let _: fn(
        &SparseQrMinimumNormPlan<f64>,
        &[f64],
        SparseOperation,
        SparseSolveOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<SparseQrSolveReport<f64>, SciOxideError> = SparseQrMinimumNormPlan::solve_into;
    let _: fn(
        &SparseQrMinimumNormPlan<f64>,
        Rhs,
        SparseOperation,
        SparseSolveOptions,
        &mut Out,
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = SparseQrMinimumNormPlan::solve_block_into;
};

// one-shot, owned and scoped forms: one-shot functions and `new`
// constructors take the problem data, owned forms take no options, workspace or
// context, and `_scoped` forms take the options, workspace and scope.
const _: fn() = || {
    use scioxide_core::{SciOxideError, StructuralPatternView, Tolerance};
    type Cx = ExecutionContext<'static>;
    type View = SparseView<'static, f64>;
    type Pattern = StructuralPatternView<'static>;
    type Csr = CsrMatrix<f64>;
    type Owned = Result<Vec<f64>, SciOxideError>;
    use scioxide_core::ExecutionContext;

    // Construction and products.
    let _: fn(usize, usize, &[usize], &[usize], &[f64]) -> Result<Csr, SciOxideError> =
        CsrMatrix::from_triplets;
    let _: fn(usize) -> Result<Csr, SciOxideError> = CsrMatrix::identity;
    let _: fn(
        SparseBuilder<f64>,
        DuplicatePolicy,
        ExplicitZeroPolicy,
    ) -> Result<CooMatrix<f64>, SciOxideError> = SparseBuilder::finish_coo;
    let _: fn(
        SparseBuilder<f64>,
        DuplicatePolicy,
        ExplicitZeroPolicy,
    ) -> Result<Csr, SciOxideError> = SparseBuilder::finish_csr;
    let _: fn(
        SparseBuilder<f64>,
        DuplicatePolicy,
        ExplicitZeroPolicy,
    ) -> Result<CscMatrix<f64>, SciOxideError> = SparseBuilder::finish_csc;
    let _: fn(usize, usize, &[isize], &[&[f64]]) -> Result<Csr, SciOxideError> = sparse_diagonals;
    let _: fn(&[usize], &[usize], &[Option<View>]) -> Result<Csr, SciOxideError> =
        sparse_block_grid;
    let _: fn(&[View]) -> Result<Csr, SciOxideError> = sparse_block_diagonal;
    let _: fn(View, &[f64]) -> Owned = SparseView::apply;
    let _: fn(View, &[f64]) -> Owned = SparseView::apply_transpose;
    let _: fn(View, &[f64]) -> Owned = SparseView::apply_adjoint;
    let _: fn(SparseOperator<'static, f64>, &[f64]) -> Owned = SparseOperator::apply;
    let _: fn(SparseOperator<'static, f64>, &[f64]) -> Owned = SparseOperator::apply_transpose;
    let _: fn(SparseOperator<'static, f64>, &[f64]) -> Owned = SparseOperator::apply_adjoint;
    let _: fn(&Csr, &[f64]) -> Owned = CsrMatrix::apply;
    let _: fn(&Csr, &[f64]) -> Owned = CsrMatrix::apply_adjoint;
    let _: fn(&CscMatrix<f64>, &[f64]) -> Owned = CscMatrix::apply;
    let _: fn(&CscMatrix<f64>, &[f64]) -> Owned = CscMatrix::apply_adjoint;
    let _: fn(&BsrMatrix<f64>, &[f64]) -> Owned = BsrMatrix::apply;
    let _: fn(&BsrMatrix<f64>, &[f64]) -> Owned = BsrMatrix::apply_adjoint;
    let _: fn(&DiaMatrix<f64>, &[f64]) -> Owned = DiaMatrix::apply;
    let _: fn(&DiaMatrix<f64>, &[f64]) -> Owned = DiaMatrix::apply_adjoint;
    let _: fn(View, f64) -> Result<Csr, SciOxideError> = SparseView::scaled;
    let _: fn(View, f64, SparseArithmeticOptions) -> Result<Csr, SciOxideError> =
        SparseView::scaled_with_options;
    let _: fn(&Csr, &Csr) -> Result<Csr, SciOxideError> = CsrMatrix::add;
    let _: fn(&Csr, &Csr, SparseArithmeticOptions) -> Result<Csr, SciOxideError> =
        CsrMatrix::add_with_options;
    let _: fn(&Csr, &Csr) -> Result<Csr, SciOxideError> = CsrMatrix::subtract;
    let _: fn(&Csr, &Csr, SparseArithmeticOptions) -> Result<Csr, SciOxideError> =
        CsrMatrix::subtract_with_options;
    let _: fn(&Csr, &Csr) -> Result<Csr, SciOxideError> = CsrMatrix::hadamard;
    let _: fn(&Csr, &Csr, SparseArithmeticOptions) -> Result<Csr, SciOxideError> =
        CsrMatrix::hadamard_with_options;
    let _: fn(&Csr, &Csr) -> Result<Csr, SciOxideError> = CsrMatrix::multiply;
    let _: fn() -> SparseArithmeticOptions = SparseArithmeticOptions::new;
    let _: fn(SparseArithmeticOptions, ExplicitZeroPolicy) -> SparseArithmeticOptions =
        SparseArithmeticOptions::with_zero_policy;
    let _: fn(&Csr, &Csr) -> Result<SparseProductPlan, SciOxideError> = SparseProductPlan::new;
    let _: fn(&Csr, &Csr, SparseProductPlanOptions) -> Result<SparseProductPlan, SciOxideError> =
        SparseProductPlan::with_options;
    let _: fn() -> SparseProductPlanOptions = SparseProductPlanOptions::new;
    let _: fn(SparseProductPlanOptions, usize) -> SparseProductPlanOptions =
        SparseProductPlanOptions::with_max_output_entries;
    let _: fn(SparseProductPlanOptions) -> usize = SparseProductPlanOptions::max_output_entries;

    // Direct solvers.
    let _: fn(&Csr, &[f64]) -> Owned = solve;
    let _: fn(&Csr, &[f64]) -> Owned = least_squares;
    let _: fn(&Csr) -> Result<SparseLu<f64>, SciOxideError> = SparseLu::new;
    let _: fn(SparseLuSymbolic, View) -> Result<SparseLu<f64>, SciOxideError> = SparseLu::factor;
    let _: fn(
        &mut SparseLuWorkspace<f64>,
        View,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseLu<f64>, SciOxideError> = SparseLuWorkspace::factor_scoped;
    let _: fn(&Csr) -> Result<SparseCholesky<f64>, SciOxideError> = SparseCholesky::new;
    let _: fn(SparseSymmetricSymbolic, View) -> Result<SparseCholesky<f64>, SciOxideError> =
        SparseCholesky::factor;
    let _: fn(
        SparseSymmetricSymbolic,
        View,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseCholesky<f64>, SciOxideError> = SparseCholesky::factor_scoped;
    let _: fn(&Csr) -> Result<SparseLdl<f64>, SciOxideError> = SparseLdl::new;
    let _: fn(SparseSymmetricSymbolic, View) -> Result<SparseLdl<f64>, SciOxideError> =
        SparseLdl::factor;
    let _: fn(
        SparseSymmetricSymbolic,
        View,
        SparseSolveOptions,
        Cx,
    ) -> Result<SparseLdl<f64>, SciOxideError> = SparseLdl::factor_scoped;
    let _: fn(SparseQrSymbolic, View) -> Result<SparseQr<f64>, SciOxideError> = SparseQr::factor;
    let _: fn(&Csr) -> Result<SparseQrMinimumNormPlan<f64>, SciOxideError> =
        SparseQrMinimumNormPlan::new;
    type Qr = Result<(Vec<f64>, SparseQrSolveReport<f64>), SciOxideError>;
    let _: fn(&SparseQrMinimumNormPlan<f64>, &[f64]) -> Qr = SparseQrMinimumNormPlan::solve;
    let _: fn(&SparseQrMinimumNormPlan<f64>, &[f64]) -> Qr =
        SparseQrMinimumNormPlan::solve_transpose;
    let _: fn(&SparseQrMinimumNormPlan<f64>, &[f64]) -> Qr = SparseQrMinimumNormPlan::solve_adjoint;
    let _: fn(&SparseLu<f64>, &[f64]) -> Owned = SparseLu::solve;
    let _: fn(&SparseLu<f64>, &[f64]) -> Owned = SparseLu::solve_transpose;
    let _: fn(&SparseLu<f64>, &[f64]) -> Owned = SparseLu::solve_adjoint;
    let _: fn(&SparseCholesky<f64>, &[f64]) -> Owned = SparseCholesky::solve;
    let _: fn(&SparseCholesky<f64>, &[f64]) -> Owned = SparseCholesky::solve_transpose;
    let _: fn(&SparseCholesky<f64>, &[f64]) -> Owned = SparseCholesky::solve_adjoint;
    let _: fn(&SparseLdl<f64>, &[f64]) -> Owned = SparseLdl::solve;
    let _: fn(&SparseLdl<f64>, &[f64]) -> Owned = SparseLdl::solve_transpose;
    let _: fn(&SparseLdl<f64>, &[f64]) -> Owned = SparseLdl::solve_adjoint;
    let _: fn(&IncompleteFactor<f64>, &[f64]) -> Owned = IncompleteFactor::solve;
    let _: fn(&IncompleteFactor<f64>, &[f64]) -> Owned = IncompleteFactor::solve_transpose;
    let _: fn(&IncompleteFactor<f64>, &[f64]) -> Owned = IncompleteFactor::solve_adjoint;
    let _: fn(&SparseTriangular<f64>, &[f64]) -> Owned = SparseTriangular::solve;
    let _: fn(&SparseTriangular<f64>, &[f64]) -> Owned = SparseTriangular::solve_transpose;
    let _: fn(&SparseTriangular<f64>, &[f64]) -> Owned = SparseTriangular::solve_adjoint;

    // Preconditioners.
    let _: fn(&Csr) -> Result<JacobiPreconditioner<f64>, SciOxideError> = JacobiPreconditioner::new;
    let _: fn(&Csr) -> Result<IncompleteFactor<f64>, SciOxideError> = IncompleteFactor::ilu0;
    let _: fn(&Csr) -> Result<IncompleteFactor<f64>, SciOxideError> = IncompleteFactor::ic0;
    let _: fn(
        IncompleteSymbolic,
        View,
        IncompleteOptions<f64>,
    ) -> Result<IncompleteFactor<f64>, SciOxideError> = IncompleteFactor::factor;
    let _: fn(&Csr) -> Result<AmgPreconditioner<f64>, SciOxideError> = AmgPreconditioner::new;
    let _: fn() -> AmgLimits = AmgLimits::default;
    let _: fn() -> SchwarzOptions<f64> = SchwarzOptions::new;
    let _: fn(SchwarzOptions<f64>, Tolerance<f64>) -> SchwarzOptions<f64> =
        SchwarzOptions::with_partition_tolerance;
    let _: fn(SchwarzOptions<f64>, usize) -> SchwarzOptions<f64> =
        SchwarzOptions::with_max_plan_bytes;
    let _: fn(SchwarzOptions<f64>, usize) -> SchwarzOptions<f64> =
        SchwarzOptions::with_max_workspace_bytes;
    let _: fn(SchwarzOptions<f64>) -> Tolerance<f64> = SchwarzOptions::partition_tolerance;
    let _: fn(SchwarzOptions<f64>) -> usize = SchwarzOptions::max_plan_bytes;
    let _: fn(SchwarzOptions<f64>) -> usize = SchwarzOptions::max_workspace_bytes;
    let _: fn(
        usize,
        SchwarzKind,
        &'static [SchwarzSubdomainView<'static, f64>],
        Option<SchwarzOperatorView<'static, f64>>,
    ) -> Result<SchwarzPlan<'static, f64>, SciOxideError> = SchwarzPlan::new;
    let _: fn(
        usize,
        SchwarzKind,
        &'static [SchwarzSubdomainView<'static, f64>],
        Option<SchwarzOperatorView<'static, f64>>,
        SchwarzOptions<f64>,
    ) -> Result<SchwarzPlan<'static, f64>, SciOxideError> = SchwarzPlan::with_options;
    let _: fn(&IdentityPreconditioner<f64>, &[f64]) -> Owned = IdentityPreconditioner::apply;
    let _: fn(&JacobiPreconditioner<f64>, &[f64]) -> Owned = JacobiPreconditioner::apply;
    let _: fn(&AmgCycle<'static, f64>, &[f64]) -> Owned = AmgCycle::apply;
    let _: fn(&AmgPreconditioner<f64>, &[f64]) -> Owned = AmgPreconditioner::apply;
    let _: fn(&SchwarzPlan<'static, f64>, &[f64]) -> Owned = SchwarzPlan::apply;

    // Graphs.
    let _: fn(Pattern) -> Result<GraphComponents, SciOxideError> = connected_components;
    let _: fn(View, usize) -> Result<GraphPathResult<f64>, SciOxideError> = shortest_paths;
    let _: fn(View) -> Result<Csr, SciOxideError> = laplacian;
    let _: fn(Pattern) -> Result<BipartiteMatching, SciOxideError> = maximum_bipartite_matching;
    let _: fn(Pattern, GraphExecutionOptions) -> Result<BipartiteMatching, SciOxideError> =
        maximum_bipartite_matching_with_options;
    let _: for<'a> fn(
        Pattern,
        GraphExecutionOptions,
        &'a mut MatchingWorkspace,
        Cx,
    ) -> Result<MatchingView<'a>, SciOxideError> = maximum_bipartite_matching_scoped;
    let _: fn(usize, &[CapacityEdge], usize, usize) -> Result<MaximumFlow, SciOxideError> =
        maximum_flow;
    let _: fn(&Csr) -> Result<FullAssignment<f64>, SciOxideError> = minimum_cost_assignment;
    let _: fn(View, GraphDirection) -> Result<GraphPlan<f64>, SciOxideError> = GraphPlan::new;
    let _: fn(Pattern, GraphDirection) -> Result<GraphPlan<f64>, SciOxideError> =
        GraphPlan::from_pattern;
    let _: fn(&GraphPlan<f64>, usize) -> Result<GraphTraversal, SciOxideError> =
        GraphPlan::breadth_first;
    let _: fn(&GraphPlan<f64>, usize) -> Result<GraphTraversal, SciOxideError> =
        GraphPlan::depth_first;
    let _: fn(&GraphPlan<f64>) -> Result<GraphComponents, SciOxideError> =
        GraphPlan::weak_components;
    let _: fn(&GraphPlan<f64>) -> Result<GraphComponents, SciOxideError> =
        GraphPlan::strong_components;
    let _: fn(&GraphPlan<f64>) -> Result<MinimumSpanningForest<f64>, SciOxideError> =
        GraphPlan::minimum_spanning_forest;
    let _: fn(
        &GraphPlan<f64>,
        GraphPathMethod,
    ) -> Result<PathOutcome<GraphPathResult<f64>>, SciOxideError> = GraphPlan::shortest_paths;
    let _: for<'w> fn(
        &GraphPlan<f64>,
        GraphPathMethod,
        GraphExecutionOptions,
        &'w mut GraphWorkspace<f64>,
        Cx,
    )
        -> Result<PathOutcome<GraphPathView<'w, f64>, &'w [usize]>, SciOxideError> =
        GraphPlan::shortest_paths_scoped;
    let _: for<'a> fn(&'a GraphPathResult<f64>) -> GraphPathView<'a, f64> =
        GraphPathResult::as_view;
    let _: fn(Pattern) -> Result<MatchingWorkspaceRequirements, SciOxideError> =
        MatchingWorkspace::workspace_requirements;
    let _: fn(Pattern) -> Result<MatchingWorkspace, SciOxideError> = MatchingWorkspace::workspace;
    let _: fn(&MatchingView<'static>, Pattern) -> bool = MatchingView::verify;
    let _: fn(usize, &[CapacityEdge]) -> Result<MaxFlowPlan, SciOxideError> = MaxFlowPlan::new;
    let _: fn(&MaxFlowPlan) -> usize = MaxFlowPlan::vertex_count;
    let _: fn(&MaxFlowPlan) -> &[CapacityEdge] = MaxFlowPlan::edges;
    let _: fn(&MaxFlowPlan, usize, usize) -> Result<MaximumFlow, SciOxideError> =
        MaxFlowPlan::maximum_flow;
    let _: for<'a> fn(
        &MaxFlowPlan,
        usize,
        usize,
        GraphExecutionOptions,
        &'a mut FlowWorkspace,
        Cx,
    ) -> Result<FlowView<'a>, SciOxideError> = MaxFlowPlan::maximum_flow_scoped;
    let _: fn(&FlowView<'static>, &MaxFlowPlan) -> Result<bool, SciOxideError> = FlowView::verify;
    let _: fn(
        &AssignmentPlan,
        &[f64],
        AssignmentObjective,
    ) -> Result<AssignmentOutcome<FullAssignment<f64>>, SciOxideError> = AssignmentPlan::solve;
    let _: for<'a> fn(
        &AssignmentPlan,
        &[f64],
        AssignmentObjective,
        GraphExecutionOptions,
        &'a mut AssignmentWorkspace<f64>,
        Cx,
    ) -> Result<AssignmentOutcome<AssignmentView<'a, f64>>, SciOxideError> =
        AssignmentPlan::solve_scoped;
};
