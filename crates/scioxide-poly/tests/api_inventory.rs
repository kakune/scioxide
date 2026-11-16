//! The authoritative scioxide-poly root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test
//! fails when this list and the `pub use` items of `src/lib.rs` differ. Exported
//! plans are asserted `Send + Sync + Clone`, and the requirements, workspace and
//! borrowed-scratch forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_poly::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    PolynomialDivision, Vandermonde, vandermonde, PolynomialFit, PolynomialFitPlan,
    PolynomialFitReport, PolynomialFitWorkspace, PolynomialFitRequirements,
    // Exact algebra over caller-provided fields.
    ExactCoefficient, ExactField, ExactPolynomial, ExactRationalFunction,
    ExactPartialFractionPlan, ExactSubresultants, exact_discriminant, exact_subresultants,
    // Exact partial fractions; returned by `ExactPartialFractionPlan::decompose`.
    ExactPartialFractionTerm, ExactPartialFractions,
    // Prepared approximate bases, policy and their diagnostics.
    ApproxPolynomial, BarycentricBasis, BernsteinBasis, MonomialBasis, NewtonBasis,
    PolynomialApproximationPolicy, PolynomialBasis, PolynomialCoefficientReport, PolynomialConditionReport,
    PolynomialConversion, PolynomialEvaluation, PolynomialScratchRequirements,
    // Root linearization and checked root evaluation.
    PolynomialBackwardErrorModel, PolynomialRootBasis, PolynomialRootBasisRef,
    CheckedBasisEvaluation, checked_basis_evaluation, polynomial_root_matrix_into,
    // Classical orthogonal families and prepared bases. Operators and recurrences
    // are return types of plan and family methods.
    BasisNormalization, JacobiRecurrence, OrthogonalAntiderivative, OrthogonalBasis,
    OrthogonalBasisPlan, OrthogonalBasisSpec, OrthogonalDerivative, OrthogonalFamily,
    OrthogonalTransform,
    // Ordinary root diagnostics. Reports, records and disks are return types.
    RootCluster, RootConditionReport, RootDiagnostic, RootDiagnostics,
    RootDiagnosticsRequirements, RootDiagnosticsStatus, RootDiagnosticsWorkspace,
    RootDiskEstimate, RootEstimate, RootEstimateFailure, RootRadiusReport,
    polynomial_root_diagnostics, polynomial_root_diagnostics_into, polynomial_root_radii,
    // Approximate GCD, rational functions and simplification results.
    ApproxRationalFunction, ApproximateGcd, ApproximateGcdReport,
    ApproximateRationalEvaluation, ApproximateRationalSimplification, approximate_gcd,
    // Checked-factor partial fractions. Factor, term and report are return types.
    ApproxPartialFractionPlan, ApproxPartialFractions, PartialFractionFactor,
    PartialFractionReport, PartialFractionRequirements, PartialFractionTerm,
    PartialFractionWorkspace,
    // Numerical elimination: method values and their fixed-size workspaces.
    ApproximateElimination, SylvesterDiscriminant, DiscriminantWorkspace, SylvesterElimination,
    EliminationReport, EliminationWorkspace, approximate_discriminant, approximate_resultant,
    approximate_subresultant,
    // Coefficient operations and their growable workspaces.
    ExactPolynomialProductTree, ExactPolynomialWorkspace, ExactPolynomialWorkspaceRequirements,
    exact_compose_into, exact_multiply_into, exact_taylor_shift_into, PolynomialExecutionOptions, PolynomialOperationPlan, PolynomialOperationRequest,
    PolynomialProductAlgorithm, PolynomialWorkspace, PolynomialWorkspaceRequirements,
    polynomial_composition_condition, polynomial_product_condition,
    // Sparse multivariate polynomials and tensor Bernstein nets.
    ApproxSparsePolynomial, BernsteinBounds, ExactSparseView, ExactSparsePolynomial,
    Monomial, MonomialOrder, SparseEvaluationPlan, SparseExpansionLimits, SparseTerms, SparseTermIter, TensorBernstein,
    TensorBernsteinView,
}

// The documented implementation API consumed by interpolation and root finding.
#[allow(unused_imports)]
use scioxide_poly::backend::{
    polynomial_basis::{
        barycentric_weights, barycentric_weights_into, confluent_divided_differences_into,
        linear_difference_ratio, linear_multiply_difference, newton_value_and_derivative_from_iter,
        real_divided_differences_into,
    },
    polynomial_iteration::{WeierstrassCorrection, weierstrass_correction},
};

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

// The one checked point evaluation, shared with root iteration.
const _: fn(
    &[f64],
    scioxide_core::Complex<f64>,
    PolynomialRootBasisRef<'_, f64>,
) -> Result<CheckedBasisEvaluation<f64>, scioxide_core::SciOxideError> =
    checked_basis_evaluation::<f64>;

plans! {
    PolynomialFitPlan<f64>, ApproxPartialFractionPlan<f64>, ExactPartialFractionPlan<Q>,
    OrthogonalBasisPlan<f64>, PolynomialOperationPlan<f64>, SparseEvaluationPlan<'static, f64>,
}

// Complex plans, the prepared exact product tree, and the builder, method values
// and view that hold only validated settings share the plan auto traits.
const _: fn() = || {
    fn plan<T: Send + Sync + Clone>() {}
    type C = scioxide_core::Complex<f64>;
    plan::<PolynomialOperationPlan<C>>();
    plan::<PolynomialFitPlan<C>>();
    plan::<PolynomialFitPlan<C, OrthogonalBasisPlan<f64>>>();
    plan::<ApproxPartialFractionPlan<C>>();
    plan::<SparseEvaluationPlan<'static, C>>();
    plan::<ExactPolynomialProductTree<Q>>();
    plan::<Vandermonde<f64>>();
    plan::<SylvesterElimination<f64>>();
    plan::<SylvesterElimination<C>>();
    plan::<SylvesterDiscriminant<f64>>();
    plan::<ExactSparseView<'static, Q>>();
};

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

// One requirements protocol per growable workspace: requirements from the plan
// (or, for plan-less root analysis, the workspace type), then `try_reserve`,
// `check` and `workspace`. Plan-fixed elimination workspaces come from
// `workspace()` alone, and borrowed scratch reports its minimum length.
const _: fn() = || {
    use scioxide_core::{Capacity, ExecutionContext, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Monic = ApproxPolynomial<f64, MonomialBasis<f64>>;
    type Orthogonal = ApproxPolynomial<f64, OrthogonalBasis<f64>>;
    type Policy = PolynomialApproximationPolicy<f64>;
    type Complex = scioxide_core::Complex<f64>;

    // One-shot constructors take the data; their configured forms add the policy.
    let _: fn(Vec<f64>) -> Result<Monic, SciOxideError> = Monic::monomial;
    let _: fn(Vec<f64>, Policy) -> Result<Monic, SciOxideError> = Monic::monomial_with_options;
    let _: fn(&[f64]) -> Result<Monic, SciOxideError> = Monic::from_roots;
    let _: fn(&[f64], Policy) -> Result<Monic, SciOxideError> = Monic::from_roots_with_options;
    let _: fn(&[f64], &[Complex]) -> Result<Monic, SciOxideError> = Monic::from_conjugate_pairs;
    let _: fn(&[f64], &[Complex], Policy) -> Result<Monic, SciOxideError> =
        Monic::from_conjugate_pairs_with_options;
    type Newton = ApproxPolynomial<f64, NewtonBasis<f64>>;
    let _: fn(Vec<f64>, Vec<f64>) -> Result<Newton, SciOxideError> = Newton::newton;
    let _: fn(Vec<f64>, Vec<f64>, Policy) -> Result<Newton, SciOxideError> =
        Newton::newton_with_options;
    let _: fn(Vec<f64>, Vec<f64>) -> Result<Newton, SciOxideError> = Newton::interpolate_newton;
    let _: fn(Vec<f64>, Vec<f64>, Policy) -> Result<Newton, SciOxideError> =
        Newton::interpolate_newton_with_options;
    type Barycentric = ApproxPolynomial<f64, BarycentricBasis<f64>>;
    let _: fn(Vec<f64>, Vec<f64>) -> Result<Barycentric, SciOxideError> =
        Barycentric::interpolate_barycentric;
    let _: fn(Vec<f64>, Vec<f64>, Policy) -> Result<Barycentric, SciOxideError> =
        Barycentric::interpolate_barycentric_with_options;
    type Bernstein = ApproxPolynomial<f64, BernsteinBasis<f64>>;
    let _: fn(Vec<f64>, f64, f64) -> Result<Bernstein, SciOxideError> = Bernstein::bernstein;
    let _: fn(Vec<f64>, f64, f64, Policy) -> Result<Bernstein, SciOxideError> =
        Bernstein::bernstein_with_options;
    let _: fn(Vec<f64>, OrthogonalBasisSpec<f64>) -> Result<Orthogonal, SciOxideError> =
        Orthogonal::orthogonal;
    let _: fn(Vec<f64>, OrthogonalBasisPlan<f64>, Policy) -> Result<Orthogonal, SciOxideError> =
        Orthogonal::orthogonal_with_plan;
    let _: fn(
        &[usize],
        Vec<scioxide_core::Interval<f64>>,
        usize,
        Vec<f64>,
    ) -> Result<TensorBernstein<f64>, SciOxideError> = TensorBernstein::new;
    let _: fn(
        &[usize],
        Vec<scioxide_core::Interval<f64>>,
        usize,
        Vec<f64>,
        PolynomialExecutionOptions,
    ) -> Result<TensorBernstein<f64>, SciOxideError> = TensorBernstein::with_options;
    type Sparse = ApproxSparsePolynomial<f64>;
    let _: fn(&Sparse, &[Sparse], usize) -> Result<Sparse, SciOxideError> = Sparse::substitute;
    let _: fn(&Sparse, &[Sparse], usize, SparseExpansionLimits) -> Result<Sparse, SciOxideError> =
        Sparse::substitute_with_options;

    // Point and batch evaluation of a polynomial in any basis.
    let _: fn(&Bernstein, f64) -> Result<f64, SciOxideError> = Bernstein::evaluate;
    let _: fn(&Bernstein, &[f64], &mut [f64], &mut [f64], Cx) -> Result<(), SciOxideError> =
        Bernstein::evaluate_batch_into;
    let _: fn(
        &Bernstein,
        &[f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
        Cx,
    ) -> Result<(), SciOxideError> = Bernstein::evaluate_batch_with_derivative_into;

    type Fit = PolynomialFitPlan<f64>;
    type FitWs = PolynomialFitWorkspace<f64>;
    let _: fn(&Fit, ()) -> Result<PolynomialFitRequirements, SciOxideError> =
        Fit::workspace_requirements;
    let _: fn(&Fit, ()) -> Result<FitWs, SciOxideError> = Fit::workspace;
    let _: fn() -> FitWs = FitWs::new;
    let _: fn(&mut FitWs, &PolynomialFitRequirements) -> Result<(), SciOxideError> =
        FitWs::try_reserve;
    let _: fn(&FitWs, &PolynomialFitRequirements) -> Result<Capacity, SciOxideError> = FitWs::check;
    let _: fn(
        &Fit,
        &[f64],
        &mut [f64],
        &mut FitWs,
        Cx,
    ) -> Result<PolynomialFitReport<f64>, SciOxideError> = Fit::fit_into;

    // Prepared floating coefficient operations.
    type Req = PolynomialWorkspaceRequirements<f64>;
    type Ws = PolynomialWorkspace<f64>;
    let _: fn(
        &[PolynomialOperationRequest],
    ) -> Result<PolynomialOperationPlan<f64>, SciOxideError> = PolynomialOperationPlan::new;
    let _: for<'a> fn(&'a PolynomialOperationPlan<f64>) -> &'a [PolynomialOperationRequest] =
        PolynomialOperationPlan::requests;
    let _: fn(
        &PolynomialOperationPlan<f64>,
        &[PolynomialOperationRequest],
    ) -> Result<Req, SciOxideError> = PolynomialOperationPlan::workspace_requirements;
    let _: fn(
        &PolynomialOperationPlan<f64>,
        &[PolynomialOperationRequest],
    ) -> Result<Ws, SciOxideError> = PolynomialOperationPlan::workspace;
    let _: fn() -> Ws = PolynomialWorkspace::new;
    let _: fn(&mut Ws, &Req) -> Result<(), SciOxideError> = PolynomialWorkspace::try_reserve;
    let _: fn(&Ws, &Req) -> Result<Capacity, SciOxideError> = PolynomialWorkspace::check;
    let _: fn(Req) -> Result<usize, SciOxideError> = Req::required_bytes;
    let _: fn(
        &PolynomialOperationPlan<f64>,
        &[f64],
        &[f64],
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<PolynomialProductAlgorithm, SciOxideError> = PolynomialOperationPlan::multiply_into;
    let _: fn(
        &PolynomialOperationPlan<f64>,
        &[f64],
        &[f64],
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = PolynomialOperationPlan::compose_into;

    // Plan-less exact coefficient operations: the request is the capacity.
    type ExactReq = ExactPolynomialWorkspaceRequirements;
    type ExactWs = ExactPolynomialWorkspace<Q>;
    let _: fn(usize) -> Result<ExactReq, SciOxideError> = ExactWs::workspace_requirements;
    let _: fn(usize) -> Result<ExactWs, SciOxideError> = ExactWs::workspace;
    let _: fn(&[Q], &[Q], &mut [Q], &mut ExactWs, Cx) -> Result<(), SciOxideError> =
        exact_multiply_into::<Q>;
    let _: fn(&[Q], &[Q], &mut [Q], &mut ExactWs, Cx) -> Result<(), SciOxideError> =
        exact_compose_into::<Q>;
    let _: fn(&[Q], Q, &mut [Q], &mut ExactWs, Cx) -> Result<(), SciOxideError> =
        exact_taylor_shift_into::<Q>;
    let _: fn(&mut ExactWs, &ExactReq) -> Result<(), SciOxideError> =
        ExactPolynomialWorkspace::try_reserve;
    let _: fn(&ExactWs, &ExactReq) -> Result<Capacity, SciOxideError> =
        ExactPolynomialWorkspace::check;

    // Partial fractions: the request is the maximum numerator length, and the
    // same requirements reserve the publishing output.
    type FractionReq = PartialFractionRequirements<f64>;
    type FractionWs = PartialFractionWorkspace<f64>;
    let _: fn(&ApproxPartialFractionPlan<f64>, usize) -> Result<FractionReq, SciOxideError> =
        ApproxPartialFractionPlan::workspace_requirements;
    let _: fn(&ApproxPartialFractionPlan<f64>, usize) -> Result<FractionWs, SciOxideError> =
        ApproxPartialFractionPlan::workspace;
    let _: fn(&mut FractionWs, &FractionReq) -> Result<(), SciOxideError> =
        PartialFractionWorkspace::try_reserve;
    let _: fn(&FractionWs, &FractionReq) -> Result<Capacity, SciOxideError> =
        PartialFractionWorkspace::check;
    let _: fn(&mut ApproxPartialFractions<f64>, &FractionReq) -> Result<(), SciOxideError> =
        ApproxPartialFractions::try_reserve;
    let _: fn(
        &ApproxPartialFractionPlan<f64>,
        &[f64],
        &mut ApproxPartialFractions<f64>,
        &mut FractionWs,
        Cx,
    ) -> Result<(), SciOxideError> = ApproxPartialFractionPlan::decompose_into;
    let _: fn(&Monic, &[(&Monic, usize)]) -> Result<ApproxPartialFractionPlan<f64>, SciOxideError> =
        ApproxPartialFractionPlan::new;

    // Plan-less root diagnostics: the request is the degree.
    type RootReq = RootDiagnosticsRequirements<f64>;
    type RootWs = RootDiagnosticsWorkspace<f64>;
    let _: fn(usize) -> Result<RootReq, SciOxideError> = RootWs::workspace_requirements;
    let _: fn(usize) -> Result<RootWs, SciOxideError> = RootWs::workspace;
    let _: fn(&mut RootWs, &RootReq) -> Result<(), SciOxideError> = RootWs::try_reserve;
    let _: fn(&RootWs, &RootReq) -> Result<Capacity, SciOxideError> = RootWs::check;
    let _: fn(&mut RootDiagnostics<f64>, &RootReq) -> Result<(), SciOxideError> =
        RootDiagnostics::try_reserve;
    let _: fn(
        &[f64],
        &[scioxide_core::Complex<f64>],
        &mut RootDiagnostics<f64>,
        &mut RootWs,
        Cx,
    ) -> Result<(), SciOxideError> = polynomial_root_diagnostics_into::<f64>;

    // Elimination method values: fixed-size workspaces, owned forms and scoped
    // forms that take the workspace and then the scope.
    type Elimination = SylvesterElimination<f64>;
    type Discriminant = SylvesterDiscriminant<f64>;
    type Value = Result<ApproximateElimination<f64>, SciOxideError>;
    let _: fn(usize, usize, usize, Policy) -> Result<Elimination, SciOxideError> = Elimination::new;
    let _: fn(usize, Policy) -> Result<Discriminant, SciOxideError> = Discriminant::new;
    let _: fn(&Elimination) -> Result<EliminationWorkspace<f64>, SciOxideError> =
        Elimination::workspace;
    let _: fn(&Discriminant) -> Result<DiscriminantWorkspace<f64>, SciOxideError> =
        Discriminant::workspace;
    let _: fn(&Elimination, &[f64], &[f64], usize) -> Value = Elimination::coefficient;
    let _: fn(&Elimination, &[f64], &[f64], usize, &mut EliminationWorkspace<f64>, Cx) -> Value =
        Elimination::coefficient_scoped;
    let _: fn(&Discriminant, &[f64]) -> Value = Discriminant::compute;
    let _: fn(&Discriminant, &[f64], &mut DiscriminantWorkspace<f64>, Cx) -> Value =
        Discriminant::compute_scoped;
    let _: for<'a> fn(
        &'a EliminationWorkspace<f64>,
    ) -> Option<&'a scioxide_core::AttemptRecord<()>> = EliminationWorkspace::last_attempt;
    let _: for<'a> fn(
        &'a DiscriminantWorkspace<f64>,
    ) -> Option<&'a scioxide_core::AttemptRecord<()>> = DiscriminantWorkspace::last_attempt;
    let _: fn(&OrthogonalTransform<f64>) -> usize =
        <OrthogonalTransform<f64> as scioxide_core::LinearOperator<f64>>::rows;
    let _: fn(&OrthogonalTransform<f64>) -> Result<usize, SciOxideError> =
        <OrthogonalTransform<f64> as scioxide_core::LinearOperator<f64>>::apply_workspace_len;
    let _: fn(&OrthogonalDerivative<'static, f64>) -> usize =
        <OrthogonalDerivative<'static, f64> as scioxide_core::LinearOperator<f64>>::cols;
    let _: fn(&OrthogonalDerivative<'static, f64>) -> Result<usize, SciOxideError> =
        <OrthogonalDerivative<'static, f64> as scioxide_core::LinearOperator<f64>>::apply_workspace_len;

    // Borrowed scratch reports its minimum length.
    type Scratch = Result<PolynomialScratchRequirements, SciOxideError>;
    let _: fn(&Monic, usize) -> Scratch = ApproxPolynomial::workspace_requirements;
    let _: fn(&ApproxSparsePolynomial<f64>, &[f64]) -> Result<f64, SciOxideError> =
        ApproxSparsePolynomial::evaluate;
    let _: fn(
        &SparseEvaluationPlan<'static, f64>,
        &[f64],
        &mut [f64],
    ) -> Result<f64, SciOxideError> = SparseEvaluationPlan::evaluate;
    let _: fn(&TensorBernsteinView<'static, f64>, usize) -> Scratch =
        TensorBernsteinView::workspace_requirements;
    let _: fn(&SparseEvaluationPlan<'static, f64>, usize) -> Scratch =
        SparseEvaluationPlan::workspace_requirements;
    let _: fn(&ExactSparseView<'static, Q>, usize) -> Scratch =
        ExactSparseView::workspace_requirements;
    let _: fn(&OrthogonalBasisPlan<f64>, usize) -> Scratch =
        OrthogonalBasisPlan::workspace_requirements;
    let _: fn(&OrthogonalAntiderivative<f64>, ()) -> Scratch =
        OrthogonalAntiderivative::workspace_requirements;
    let _: fn(PolynomialScratchRequirements) -> usize = PolynomialScratchRequirements::scratch_len;

    // Borrowed-output forms take the execution context last and publish atomically.
    let _: fn(
        &[f64],
        PolynomialRootBasisRef<'_, f64>,
        &mut scioxide_linalg::Matrix<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = polynomial_root_matrix_into::<f64>;
    let _: fn(
        &OrthogonalAntiderivative<f64>,
        &[f64],
        (f64, f64),
        &mut [f64],
        &mut [f64],
        Cx,
    ) -> Result<(), SciOxideError> = OrthogonalAntiderivative::apply_into::<f64>;
    let _: fn(
        &TensorBernsteinView<'static, f64>,
        &[f64],
        &mut [f64],
        &mut [f64],
        Cx,
    ) -> Result<(), SciOxideError> = TensorBernsteinView::evaluate_into;
    let _: fn(
        &ApproxSparsePolynomial<f64>,
        scioxide_core::NdViewMut<'_, '_, f64>,
        &[usize],
        Cx,
    ) -> Result<(), SciOxideError> = ApproxSparsePolynomial::to_tensor_into;
};
