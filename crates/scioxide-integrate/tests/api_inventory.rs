//! Compiles the reviewed integration API and checks its complete root inventory.
use std::collections::BTreeSet;
macro_rules! inventory { ($($name:ident),* $(,)?) => { #[allow(unused_imports)] use scioxide_integrate::{$($name),*}; const INVENTORY: &[&str] = &[$(stringify!($name)),*]; }; }
inventory! { AdaptiveIntegrator, AdaptiveRule, AdaptiveWorkspace, ClenshawCurtis, ContourPlan, CoordinateMap, CubaturePlan, CubatureWorkspace, DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint, DoubleExponentialWorkspace, EndpointWeight, ErrorEstimate, ErrorReliability, EulerMaclaurinProvider, FixedRule, FixedRuleNode, FixedRuleWorkspace, FourierDomain, FourierTailOptions, FourierTailPlan, FourierTailWorkspace, FourierWeight, GaussKronrod, GaussKronrodRule, GaussianDomain, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport, GeometricOptions, GeometricWorkspace, IdentityMap, Integral, IntegrationErrorComponents, IntegrationReport, IntegrationRequirements, IntegrationTermination, IteratedDomainPlan, IteratedOptions, IteratedWorkspace, JvpQuery, LebedevRule, LevelOptions, MiserOptions, MiserStrata, MiserTrainer, MiserWorkspace, MonteCarloChild, MonteCarloIntegrator, MonteCarloOptions, MonteCarloProposal, MonteCarloStatistics, MonteCarloWorkspace, NewtonCotes, NewtonCotesEndpoints, OscillatoryGeometry, OscillatoryMethod, OscillatoryOptions, OscillatoryPhase, OscillatoryPhaseInverse, OscillatoryQuadraturePlan, OscillatoryRegion, OscillatoryWorkspace, ProductOptions, ProductRule, ProductWorkspace, ProposalUniforms, QuadratureSensitivityPlan, QuadratureSensitivityWorkspace, RandomStreams, Romberg, RombergWorkspace, SAMPLE_BLOCK, SampledAxisWorkspace, SampledRombergPlan, SampledSimpsonPlan, SampledTrapezoidPlan, SampledWorkspace, SensitivityDiscretization, SensitivityErrorComponents, SensitivityIntegrand, SensitivityMode, SensitivityOptions, SensitivityPartition, SensitivityReport, SensitivityRequirements, SensitivitySample, SeriesError, SeriesMethod, SeriesOptions, SeriesProblem, SeriesReport, SeriesRequirements, SeriesResult, SeriesState, SeriesSummation, SeriesTermination, SeriesTerms, SeriesWorkspace, SimplexMeasure, SimplexRule, SparseGridOptions, SparseGridPlan, SparseGridWorkspace, SphereProductRule, SubdivisionOptions, TailDiagnostics, TailStatus, TrainingReport, UniformProposal, UniformSource, VarianceBasis, VegasMap, VegasOptions, VegasTrainer, VegasWorkspace, VjpQuery, WeightedQuadratureOptions, WeightedQuadraturePlan, WeightedQuadratureWorkspace, cumulative_trapezoid, integrate, integrate_box, integrate_fourier, integrate_iterated, integrate_monte_carlo, integrate_piecewise, integrate_unbounded, simpson, sum_series, trapezoid }
macro_rules! plans { ($($name:ident<$scalar:ty>),* $(,)?) => { const _: fn() = || { fn shared<T: Send+Sync+Clone>() {} $(shared::<$name<$scalar>>();)* }; const PLANS: &[&str] = &[$(stringify!($name)),*]; }; }
plans! { ContourPlan<f64>, CubaturePlan<f64>, FourierTailPlan<f64>, GaussianRulePlan<f64>, IteratedDomainPlan<f64>, OscillatoryQuadraturePlan<f64>, QuadratureSensitivityPlan<f64>, SampledRombergPlan<f64>, SampledSimpsonPlan<f64>, SampledTrapezoidPlan<f64>, SparseGridPlan<f64>, WeightedQuadraturePlan<f64> }
const _: fn() = || {
    use scioxide_core::Complex;
    fn shareable<T: Send + Sync + Clone>() {}
    shareable::<GaussianRulePlan<f64>>();
    shareable::<GaussianRuleOptions<f64>>();
    shareable::<GaussKronrod<f64>>();
    shareable::<ClenshawCurtis<f64>>();
    shareable::<NewtonCotes<f64>>();
    shareable::<AdaptiveIntegrator<f64>>();
    shareable::<SubdivisionOptions<f64>>();
    shareable::<Romberg<f64>>();
    shareable::<LevelOptions<f64>>();
    shareable::<DoubleExponentialIntegrator<f64>>();
    shareable::<WeightedQuadraturePlan<f64>>();
    shareable::<FourierTailPlan<f64>>();
    shareable::<ContourPlan<f64>>();
    shareable::<OscillatoryQuadraturePlan<f64>>();
    shareable::<OscillatoryGeometry<f64>>();
    shareable::<CubaturePlan<f64>>();
    shareable::<IteratedDomainPlan<f64>>();
    shareable::<SparseGridPlan<f64>>();
    shareable::<SparseGridOptions<f64>>();
    shareable::<ProductRule<f64, GaussKronrod<f64>>>();
    shareable::<ProductOptions>();
    shareable::<SimplexRule<f64>>();
    shareable::<SphereProductRule<f64>>();
    shareable::<LebedevRule<f64>>();
    shareable::<GeometricOptions>();
    shareable::<MonteCarloIntegrator<f64>>();
    shareable::<MonteCarloOptions>();
    shareable::<MonteCarloStatistics<f64>>();
    shareable::<ProposalUniforms<'static, f64>>();
    shareable::<VegasTrainer<f64>>();
    shareable::<VegasOptions<f64>>();
    shareable::<MiserTrainer<f64>>();
    shareable::<MiserOptions>();
    shareable::<WeightedQuadratureOptions<f32>>();
    shareable::<FourierTailOptions<f64>>();
    shareable::<OscillatoryOptions<f64>>();
    shareable::<SeriesSummation<f64>>();
    shareable::<SeriesOptions<f64>>();
    shareable::<SeriesSummation<Complex<f32>>>();
    shareable::<QuadratureSensitivityPlan<f64>>();
    shareable::<SensitivityOptions<f64>>();
    shareable::<SensitivityPartition<f64>>();
    shareable::<SampledTrapezoidPlan<f64>>();
    shareable::<SampledSimpsonPlan<f64>>();
    shareable::<SampledRombergPlan<f64>>();
    shareable::<Integral<f64>>();
};
/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("pub use ")
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
    let source = "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\nuse c::Hidden;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}

#[allow(clippy::type_complexity)]
const _: fn() = || {
    use scioxide_core::{ExtendedInterval, Interval, SciOxideError};
    use scioxide_poly::OrthogonalFamily;
    type F1 = fn(f64) -> Result<f64, SciOxideError>;
    type Fnd = fn(&[f64]) -> Result<f64, SciOxideError>;
    type Bounds = fn(&[f64]) -> Result<Interval<f64>, SciOxideError>;
    type Term = fn(usize) -> Result<f64, SciOxideError>;
    type Answer = Result<Integral<f64>, SciOxideError>;
    let _: fn(F1, f64, f64) -> Answer = integrate::<f64, F1, f64>;
    let _: fn(F1, f64, f64, &[f64]) -> Answer = integrate_piecewise::<f64, F1, f64>;
    let _: fn(F1, f64, f64, f64, FourierWeight) -> Answer = integrate_fourier::<f64, F1, f64>;
    let _: fn(F1, ExtendedInterval<f64>) -> Answer = integrate_unbounded::<f64, F1, f64>;
    let _: fn(Fnd, &[f64], &[f64]) -> Answer = integrate_box::<f64, Fnd, f64>;
    let _: fn(Fnd, Bounds, usize) -> Answer = integrate_iterated::<f64, Fnd, Bounds, f64>;
    let _: fn(
        Fnd,
        &[f64],
        &[f64],
        &scioxide_random::Philox4x64,
    ) -> Result<MonteCarloStatistics<f64>, SciOxideError> =
        integrate_monte_carlo::<f64, Fnd, scioxide_random::Philox4x64>;
    let _: fn(&[f64], &[f64]) -> Result<f64, SciOxideError> = trapezoid::<f64, f64>;
    let _: fn(&[f64], &[f64]) -> Result<f64, SciOxideError> = simpson::<f64, f64>;
    let _: fn(&[f64], &[f64]) -> Result<Vec<f64>, SciOxideError> = cumulative_trapezoid::<f64, f64>;
    let _: fn(Term, usize) -> Result<SeriesResult<f64>, SciOxideError> = sum_series::<f64, Term>;
    let _: fn(usize) -> Result<GaussianRulePlan<f64>, SciOxideError> =
        GaussianRulePlan::gauss_legendre;
    let _: fn(OrthogonalFamily<f64>, usize) -> Result<GaussianRulePlan<f64>, SciOxideError> =
        GaussianRulePlan::new;
    let _: fn(
        OrthogonalFamily<f64>,
        usize,
        GaussianRuleVariant,
        GaussianRuleOptions<f64>,
    ) -> Result<GaussianRulePlan<f64>, SciOxideError> = GaussianRulePlan::with_options;
};
