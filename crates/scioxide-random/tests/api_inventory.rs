//! The authoritative scioxide-random root census.
//!
//! Every crate-root export is imported here as an external consumer, and a
//! test fails when this list and the `pub use` items of `src/lib.rs` differ.
//! Every exported plan and the sampling policies are asserted
//! `Send + Sync + Clone` at compile time.
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_random::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    AdaptiveRejectionSampler, AdaptiveSamplingWorkspace, AdvanceOnError, Affine, Alpha, Anglit,
    Argus, AsymmetricLaplace, Beta, BetaBinomial, BetaNegativeBinomial, BetaPrime, Binomial,
    BinomialSampler, BitGenerator, Boltzmann, BoundedDensity, BoundedDensityFn, BoundedSubstream, Bradford,
    BrownianBridgePlan, Burr12, Cauchy, Cdf, CenteredDiscrepancy, ChiSquared, Cmrg, ConcaveDensity, ConcaveDensityFn,
    ContinuousLaw, Cosine, CountDistribution, CountSupport, CrystalBall, Density, DensityTransform,
    DirectionNumberSet, Dirichlet, DirichletMultinomial,
    DiscrepancyMethod, DiscreteLaplace, DiscreteUniform, Distribution, DoubleGamma,
    DoubleParetoLogNormal, DoubleWeibull, Entropy, Exponential, ExponentialPower, ExponentialTransform,
    ExponentiallyModifiedNormal, ExponentiatedWeibull, FatigueLife, FisherF,
    FisherNoncentralHypergeometric, Gamma, GaussHypergeometric, GeneralizedExponential,
    GeneralizedExtremeValue, GeneralizedGamma, GeneralizedHalfLogistic, GeneralizedHyperbolic,
    GeneralizedInverseGaussian, GeneralizedLogistic, GeneralizedNormal, GeneralizedPareto,
    GeneratorAlgorithm, Geometric, GeometricDiscrepancy, Gfsr4, Gompertz, GumbelRight, Histogram,
    HyperbolicSecant, Hypergeometric, InverseGaussian, InverseSurvival, InverseWeibull,
    InverseWishart, InversionOptions, InversionReport, InversionSampler, IrwinHall, JohnsonSB,
    JohnsonSU, JonesFaddySkewT, JumpableRng, Kappa4, Kolmogorov, KolmogorovRequirements,
    KolmogorovSmirnovOneSided, KolmogorovSmirnovTwoSided, KolmogorovWorkspace,
    LatinHypercubeOptions, LatinHypercubeReport, LatinHypercubeTermination, LawOptions, Lcg64, LevyStable, LogCdf, LogCoordinateLaw, LogDensity,
    LogMass, LogNormal, LogSeries, LogSurvival, LogTailsFn, LogTransformed, Logistic, Mass, MatrixNormal, MatrixT, Mean,
    Mielke, Mixture, Moyal, Mrg, Mrg32k3a, Mt19937, Multinomial, MultivariateHypergeometric,
    MultivariateNormal, MultivariateNormalOptions, MultivariateProbabilityOptions,
    MultivariateProbabilityPlan, MultivariateProbabilityReport, MultivariateProbabilityTermination,
    MultivariateProbabilityWorkspace, MultivariateStudentT, NegativeBinomial, NoncentralChiSquared,
    NoncentralF, NoncentralStudentT, Normal, NormalInverseGamma, NormalInverseGaussian,
    OrderStatistic, OrthogonalGroup, OutputPolicy, Pareto, Pcg64, Pearson3, Philox4x64, Poisson,
    PoissonBinomial, PoissonDisk, PoissonDiskOptions, PoissonDiskReport, PoissonDiskTermination, PoissonDiskWorkspace, PoissonSampler,
    PowerNormal, QmcCheckpoint, QmcFamily, QmcMultinomial, QmcMultivariateNormal,
    QmcPlan, QmcPosition, QmcReplicateEstimate, QmcReplicateOptions, QmcReplicatePlan, QmcRandomization, QmcScramble, QmcSequence,
    QmcTransportWorkspace, Quantile, QuasiRng, RandomCorrelation, RandomTable, Ranlux,
    RanluxLuxury, Ranlxd, RanlxdLuxury, Ranlxs, RanlxsLuxury, RatioUniformsBounds,
    RatioUniformsSampler, ReciprocalInverseGaussian, RelativisticBreitWigner, Rice, Rng, Rollback,
    SamplingDiagnostics, SamplingPolicy, SamplingRequest, SamplingRequirements, SamplingWorkspace,
    SeedableRng, SerializableState, SignedCountSupport, Skellam, SkellamSampler, SkewCauchy,
    SkewNormal, SkipAhead, SnapshotGenerator, SobolDirection, SpecialOrthogonalGroup, SplitMix64,
    SplittableRng, StatePolicy, StreamIdentity, StreamPartitioner, StudentT, StudentizedRange,
    Support, Survival, Taus113, Taus2, TestStatisticResult, Threefry4x64, Transformed,
    TransformedDensitySampler, Trapezoid, Truncated, TruncatedPareto, TukeyLambda, Uniform,
    UniformDirection, UnitaryGroup, UniversalSample, UniversalSamplingOptions, UserDensity,
    Variance, VonMises, VonMisesFisher, VonMisesLine, WalleniusNoncentralHypergeometric,
    WeibullMin, WeightedSampler, Wishart, WrappedCauchy, Xoshiro128StarStar, Xoshiro256StarStar,
    YuleSimon, Zipf, Zipfian, choose, decode_u32, decode_u64, fill_uniforms, discrepancy, geometric_discrepancy,
    invert_log_probability, invert_log_probability_scoped, invert_log_probability_with_options,
    latin_hypercube, latin_hypercube_into, latin_hypercube_with_options, log_interval_probability,
    log_interval_probability_scoped, log_interval_probability_with_options, philox4x64_block,
    poisson_disk, sample_indices, seed_sequence, shuffle, shuffle_in_place, threefry4x64_block, uniform_index,
}

/// Asserts the plan auto traits for each listed instantiation and
/// records the plan names for the completeness test.
macro_rules! plans {
    ($($name:ident $(<$argument:ty>)?),* $(,)?) => {
        const _: fn() = || {
            fn plan<T: Send + Sync + Clone>() {}
            $(plan::<$name $(<$argument>)?>();)*
        };
        const PLANS: &[&str] = &[$(stringify!($name)),*];
    };
}

plans! {
    BrownianBridgePlan<f32>, BrownianBridgePlan<f64>, MultivariateProbabilityPlan, QmcPlan,
    QmcReplicatePlan,
}

// Other reusable values that callers share across threads.
const _: fn() = || {
    fn shared<T: Send + Sync + Clone>() {}
    shared::<PoissonDisk<f32>>();
    shared::<PoissonDisk<f64>>();
    shared::<SamplingPolicy<AdvanceOnError>>();
    shared::<SamplingPolicy<Rollback>>();
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

// Constructor policy remains an explicit part of the reviewed public surface.
#[allow(clippy::type_complexity)]
const _: fn() = || {
    use scioxide_core::SciOxideError;
    use scioxide_linalg::Matrix;
    let _: fn(f64) -> Result<Alpha<f64>, SciOxideError> = Alpha::new;
    let _: fn(f64, LawOptions<f64>) -> Result<Alpha<f64>, SciOxideError> = Alpha::with_options;
    let _: fn() -> Result<Cosine<f64>, SciOxideError> = Cosine::new;
    let _: fn(f64) -> Result<Poisson<f64>, SciOxideError> = Poisson::new;
    let _: fn() -> Result<Kolmogorov<f64>, SciOxideError> = Kolmogorov::new;
    let _ = |mean: &[f64],
             covariance: &Matrix<f64>|
     -> Result<MultivariateNormal<f64>, SciOxideError> {
        MultivariateNormal::<f64>::new(mean, covariance)
    };
};
