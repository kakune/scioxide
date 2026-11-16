//! The authoritative scioxide-validated root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test
//! fails when this list and the `pub use` items of `src/lib.rs` differ. Exported
//! plans are asserted `Send + Sync + Clone`, and each family's requirements,
//! workspace and execution forms are pinned by signature.
#![allow(clippy::type_complexity)]
use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Capacity, Complex, ErrorTolerance, ExecutionContext,
    ExtendedInterval, SciOxideError, StepTolerance,
};
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_validated::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    solve, roots, polynomial_roots, integrate,
    // Shared configuration, requests and provider protocols.
    ProofInconclusive, RefinementLimit, ProofRequest, ValidatedOptions, IntervalBox, IntervalJacobianOperator, IntervalLinearOperator,
    IntervalResidualOperator, IntervalScalarFunction,
    // Verified linear systems.
    DenseIntervalOperator, LinearBounds, LinearCertificate, LinearCertificateView,
    LinearCertificateViewMut, LinearInconclusive, LinearProofFailure, LinearRejection, LinearVerificationReport, LinearVerificationStatus,
    LuIntervalInverse, SparseIntervalOperator, LinearVerifier, LinearRequirements,
    LinearWorkspace, interval_enclosure, point_enclosure,
    // Certified quadrature.
    QuadratureVerifier, QuadratureAssumption, QuadratureCertificate,
    QuadratureCertificateView, QuadratureGenerationReport,
    QuadratureInconclusive, QuadratureProofFailure, QuadratureObligation, QuadratureOracle, QuadraturePanel,
    QuadraturePiece, QuadratureQuery, QuadratureRejection, QuadratureRequest,
    QuadratureRequirements, QuadratureTermination, QuadratureVerificationStatus,
    QuadratureWorkspace,
    // Root proofs.
    PolynomialIsolationReport, PolynomialRootCertificate, PolynomialRootCover, PolynomialRootPlan,
    PolynomialVerificationReport, SystemRootCertificate,
    SystemRootCertificateView, SystemRootCertificateViewMut, RootClaim, RootDisk, RootInconclusive, RootProofFailure, RootIsolationTermination,
    RootIsolation, RootIsolationNode, RootIsolationReport, RootIsolationView, RootRequirements,
    RootVerificationReport, RootVerificationStatus, RootWorkspace, ScalarRootCertificate,
    ScalarRootVerifier, SystemRootVerifier,
    // Affine arithmetic and elementary enclosures.
    AffineBinary, AffineContext, AffineFallback, AffineForm, AffineOptions, AffineProjection, AffineReport,
    AffineState, AffineSymbol, AffineTerm, AffineUnary, AffineView, AffineViewMut,
    AffineWorkspace, ElementaryEnclosure, EnclosureStatus,
}
// Types defined at the crate root itself.
#[allow(unused_imports)]
use scioxide_validated::{DecoratedInterval, Decoration};

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

plans! { PolynomialRootPlan<f64> }

// Complex plans, options and requests share the same auto traits.
const _: fn() = || {
    fn value<T: Send + Sync + Clone>() {}
    type C = Complex<f64>;
    value::<LinearVerifier<'static, f64>>();
    value::<SystemRootVerifier<'static, f64>>();
    value::<ScalarRootVerifier<'static, f64>>();
    value::<QuadratureVerifier<'static, f64>>();
    value::<LinearVerifier<'static, C>>();
    value::<SystemRootVerifier<'static, C>>();
    value::<PolynomialRootPlan<C>>();
    value::<QuadratureVerifier<'static, C>>();
    value::<ValidatedOptions>();
    value::<AffineOptions>();
    value::<ProofRequest>();
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

type R<T> = Result<T, SciOxideError>;
type Cx = ExecutionContext<'static>;

// One requirements protocol per family workspace: plans report requirements for a
// request, workspaces reserve and check them, and plans build reserved workspaces.
const _: fn() = || {
    let _: fn(&LinearVerifier<'static, f64>, usize) -> R<LinearRequirements> =
        LinearVerifier::workspace_requirements;
    let _: fn(&LinearVerifier<'static, f64>, usize) -> R<LinearWorkspace<f64>> =
        LinearVerifier::workspace;
    let _: fn(&mut LinearWorkspace<f64>, &LinearRequirements) -> R<()> =
        LinearWorkspace::try_reserve;
    let _: fn(&LinearWorkspace<f64>, &LinearRequirements) -> R<Capacity> = LinearWorkspace::check;
    let _: fn(&ScalarRootVerifier<'static, f64>, ProofRequest) -> R<RootRequirements> =
        ScalarRootVerifier::workspace_requirements;
    let _: fn(&SystemRootVerifier<'static, f64>, ProofRequest) -> R<RootRequirements> =
        SystemRootVerifier::workspace_requirements;
    let _: fn(&PolynomialRootPlan<f64>, ProofRequest) -> R<RootWorkspace<f64>> =
        PolynomialRootPlan::workspace;
    let _: fn(&mut RootWorkspace<f64>, &RootRequirements) -> R<()> = RootWorkspace::try_reserve;
    let _: fn(&RootWorkspace<f64>, &RootRequirements) -> R<Capacity> = RootWorkspace::check;
    let _: fn(&QuadratureVerifier<'static, f64>, ProofRequest) -> R<QuadratureRequirements> =
        QuadratureVerifier::workspace_requirements;
    let _: fn(&QuadratureVerifier<'static, f64>, ProofRequest) -> R<QuadratureWorkspace<f64>> =
        QuadratureVerifier::workspace;
    let _: fn(&mut QuadratureWorkspace<f64>, &QuadratureRequirements) -> R<()> =
        QuadratureWorkspace::try_reserve;
    let _: fn(&QuadratureWorkspace<f64>, &QuadratureRequirements) -> R<Capacity> =
        QuadratureWorkspace::check;
    let _: fn(&AffineContext) -> R<AffineWorkspace<f64>> = AffineContext::workspace::<f64>;
};

// Component projections change the scalar domain and require no scratch workspace.
const _: fn() = || {
    let _: fn(&AffineForm<Complex<f64>>, AffineProjection) -> R<AffineForm<f64>> =
        AffineForm::project;
    let _: fn(
        &AffineView<'static, Complex<f64>>,
        AffineProjection,
        AffineViewMut<'_, f64>,
    ) -> R<AffineReport<f64>> = AffineView::project_into;
    let _: fn(&AffineForm<f32>, AffineProjection) -> R<AffineForm<f32>> = AffineForm::project;
};

// Owned forms allocate and run sequentially; prepared forms take a workspace and a
// scope last and never a caller identity, execution argument or byte budget.
const _: fn() = || {
    type Linear = LinearVerifier<'static, f64>;
    type Report = LinearVerificationReport<f64>;
    let _: fn(&Linear, &[IntervalBox<f64>]) -> R<(Report, Option<LinearCertificate<f64>>)> =
        Linear::solve;
    let _: fn(
        &Linear,
        &[IntervalBox<f64>],
        LinearCertificateViewMut<'_, f64>,
        &mut LinearWorkspace<f64>,
        Cx,
    ) -> R<Report> = Linear::solve_into;
    let _: fn(&Linear, &[IntervalBox<f64>], LinearCertificateView<'_, f64>) -> R<Report> =
        Linear::verify;
    let _: fn(
        &Linear,
        &[IntervalBox<f64>],
        LinearCertificateView<'_, f64>,
        &mut LinearWorkspace<f64>,
        Cx,
    ) -> R<Report> = Linear::verify_scoped;
    let _: fn(&Linear) -> R<Report> = Linear::bounds;
    let _: fn(
        &Linear,
        BlockVectorView<'_, IntervalBox<f64>>,
        &mut BlockVectorViewMut<'_, f64>,
        &mut BlockVectorViewMut<'_, Complex<f64>>,
        &mut [Report],
        &mut LinearWorkspace<f64>,
        Cx,
    ) -> R<()> = Linear::solve_block_into;
    let _: fn(
        &scioxide_linalg::LuDecomposition<f64>,
        ValidatedOptions,
    ) -> R<LuIntervalInverse<'static, f64>> = LuIntervalInverse::from_dense;
    let _: fn(usize) -> R<LinearCertificate<f64>> = LinearCertificate::new;

    type Scalar = ScalarRootVerifier<'static, f64>;
    type Root = RootVerificationReport<f64>;
    let _: fn(&Scalar, ScalarRootCertificate<f64>) -> R<Root> = Scalar::verify;
    let _: fn(&Scalar, ScalarRootCertificate<f64>, Cx) -> R<Root> = Scalar::verify_scoped;
    let _: fn(
        &Scalar,
        ExtendedInterval<f64>,
        StepTolerance<f64>,
        usize,
    ) -> R<(RootIsolationReport, RootIsolation<f64>)> = Scalar::isolate;
    let _: fn(
        &Scalar,
        ExtendedInterval<f64>,
        StepTolerance<f64>,
        &mut RootIsolation<f64>,
        &mut RootWorkspace<f64>,
        Cx,
    ) -> R<RootIsolationReport> = Scalar::isolate_into;
    let _: fn(
        &Scalar,
        &[ScalarRootCertificate<f64>],
        &mut [Root],
        &mut RootWorkspace<f64>,
        Cx,
    ) -> R<()> = Scalar::verify_batch_into;
    type System = SystemRootVerifier<'static, f64>;
    let _: fn(&System, SystemRootCertificateView<'_, f64>) -> R<Root> = System::verify;
    let _: fn(
        &System,
        &[IntervalBox<f64>],
        StepTolerance<f64>,
        usize,
    ) -> R<(RootIsolationReport, RootIsolation<f64>)> = System::isolate;
    let _: fn(
        &System,
        &[IntervalBox<f64>],
        StepTolerance<f64>,
        &mut RootIsolation<f64>,
        &mut RootWorkspace<f64>,
        Cx,
    ) -> R<RootIsolationReport> = System::isolate_into;
    let _: fn(&System, &[IntervalBox<f64>], RootIsolationView<'_, f64>) -> R<RootIsolationReport> =
        System::verify_isolation;
    type Polynomial = PolynomialRootPlan<f64>;
    type Count = PolynomialVerificationReport;
    let _: fn(&Polynomial, PolynomialRootCertificate<f64>) -> R<Count> = Polynomial::verify;
    let _: fn(&Polynomial, &[PolynomialRootCertificate<f64>]) -> R<Count> =
        Polynomial::verify_cover;
    let _: fn(
        &Polynomial,
        &[PolynomialRootCertificate<f64>],
        &mut [Count],
        &mut RootWorkspace<f64>,
        Cx,
    ) -> R<()> = Polynomial::verify_batch_into;
    let _: fn(
        &Polynomial,
        &[RootDisk<f64>],
    ) -> R<(PolynomialIsolationReport, PolynomialRootCover<f64>)> = Polynomial::isolate;
    let _: fn(
        &Polynomial,
        &[RootDisk<f64>],
        &mut PolynomialRootCover<f64>,
        &mut RootWorkspace<f64>,
        Cx,
    ) -> R<PolynomialIsolationReport> = Polynomial::isolate_into;
    let _: fn(usize, usize) -> R<RootIsolation<f64>> = RootIsolation::new;
    let _: fn(usize) -> R<PolynomialRootCover<f64>> = PolynomialRootCover::new;

    type Quadrature = QuadratureVerifier<'static, f64>;
    let _: fn(
        &Quadrature,
        &ErrorTolerance<f64>,
        usize,
    ) -> R<(
        QuadratureGenerationReport,
        Option<QuadratureCertificate<f64>>,
    )> = Quadrature::generate;
    let _: fn(
        &Quadrature,
        &ErrorTolerance<f64>,
        &mut QuadratureCertificate<f64>,
        &mut QuadratureWorkspace<f64>,
        Cx,
    ) -> R<QuadratureGenerationReport> = Quadrature::generate_into;
    let _: fn(&Quadrature, QuadratureCertificateView<'_, f64>) -> R<QuadratureVerificationStatus> =
        Quadrature::verify;
    let _: fn(
        &Quadrature,
        QuadratureCertificateView<'_, f64>,
        &mut QuadratureWorkspace<f64>,
        Cx,
    ) -> R<QuadratureVerificationStatus> = Quadrature::verify_scoped;
    let _: fn(
        &Quadrature,
        &[QuadratureCertificateView<'_, f64>],
        &mut [QuadratureVerificationStatus],
        &mut QuadratureWorkspace<f64>,
        Cx,
    ) -> R<()> = Quadrature::verify_batch_into;
    let _: fn(usize, usize) -> R<QuadratureCertificate<f64>> = QuadratureCertificate::new;
    // Variable-length outputs track their published length behind an infallible view.
    let _: fn(&RootIsolation<f64>) -> RootIsolationView<'_, f64> = RootIsolation::view;
    let _: fn(&PolynomialRootCover<f64>) -> &[PolynomialRootCertificate<f64>] =
        PolynomialRootCover::view;
    let _: fn(&QuadratureCertificate<f64>) -> QuadratureCertificateView<'_, f64> =
        QuadratureCertificate::view;
};

#[test]
fn caller_report_slots_require_no_proof() {
    let root = RootVerificationReport::<f64>::not_run();
    assert_eq!(root.status(), RootVerificationStatus::NotRun);
    assert!(root.evidence().is_none() && root.defect().is_none());
    let count = PolynomialVerificationReport::not_run();
    assert_eq!(count.status(), RootVerificationStatus::NotRun);
    assert!(count.count().is_none() && !count.is_proved());
    let linear = LinearVerificationReport::<f64>::not_run();
    assert_eq!(linear.status(), LinearVerificationStatus::NotRun);
    assert!(
        linear.bounds().is_none() && linear.residual().is_none() && linear.correction().is_none()
    );
    assert!(!root.is_proved() && !linear.is_verified());
    let statuses = [QuadratureVerificationStatus::NotRun; 2];
    assert!(
        statuses
            .iter()
            .all(|status| *status == QuadratureVerificationStatus::NotRun)
    );
}

// Plain-data entry points and owned convenience forms retain their public signatures.
const _: fn() = || {
    type Callback = fn(DecoratedInterval<f64>) -> R<DecoratedInterval<f64>>;
    let _: fn(scioxide_linalg::MatrixView<'static, f64>, &[f64]) -> R<LinearCertificate<f64>> =
        solve;
    let _: fn(Callback, Callback, f64, f64) -> R<Vec<scioxide_core::Interval<f64>>> = roots;
    let _: fn(&[f64], &[Complex<f64>]) -> R<PolynomialRootCover<f64>> = polynomial_roots;
    let _: fn(Callback, Callback, f64, f64) -> R<scioxide_core::Interval<f64>> = integrate;
    let _: fn(f64, f64) -> R<DecoratedInterval<f64>> = DecoratedInterval::new;
    let _: fn(f64) -> R<DecoratedInterval<f64>> = DecoratedInterval::point;
    let _: fn(
        &LinearVerifier<'static, f64>,
        &[IntervalBox<f64>],
        &[f64],
    ) -> R<(
        LinearVerificationReport<f64>,
        Option<LinearCertificate<f64>>,
    )> = LinearVerifier::propose;
    let _: fn(
        &'static [QuadraturePiece<f64>],
        usize,
        &'static (dyn QuadratureOracle<f64> + Sync),
        ValidatedOptions,
    ) -> R<QuadratureVerifier<'static, f64>> = QuadratureVerifier::new;
    fn regions(tree: &RootIsolation<f64>) -> impl Iterator<Item = &[IntervalBox<f64>]> + '_ {
        tree.isolated_regions()
    }
    let _ = regions;
};
