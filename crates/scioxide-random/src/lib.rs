//! Random number generation: generators, probability distributions, quasi-random sequences and sampling.
//!
//! The crate provides seeded pseudo-random number generators with skip-ahead,
//! numbered streams and portable checkpoints; continuous, discrete,
//! circular, multivariate and matrix probability distributions with densities, distribution functions, quantiles, moments and
//! samplers; quasi-random (low-discrepancy) sequences and space-filling
//! designs; and shuffling, subset selection and weighted choice. Results are
//! `f32` or `f64`. No generator is cryptographically secure.
//!
//! # Quick start
//!
//! ```
//! use scioxide_random::{Distribution, Normal, Quantile};
//! use scioxide_random::{SeedableRng, Xoshiro256StarStar};
//!
//! let law = Normal::<f64>::standard();
//! // Standard normal quantile: Φ⁻¹(0.975) = 1.959963984540054
//! assert!((law.quantile(0.975)? - 1.959_963_984_540_054).abs() < 1e-12);
//! let mut rng = Xoshiro256StarStar::from_seed(42);
//! let draws = law.sample_n(&mut rng, 10_000)?;
//! // The sample mean has standard error 0.01; allow five of them.
//! let mean = draws.iter().sum::<f64>() / 10_000.0;
//! assert!(mean.abs() < 0.05);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Task | Entry points | Method |
//! |---|---|---|
//! | Raw random words | [`BitGenerator`], one of the [generators](#generators) | |
//! | Uniform floats in `[0, 1)` or `(0, 1)` | [`Rng::next_float`], [`Rng::next_open_float`], [`decode_u64`], [`fill_uniforms`] | top 53 bits for `f64`, 24 bits for `f32` |
//! | Seeds for several generators | [`seed_sequence`] | SplitMix64 mixing of a master seed |
//! | Streams for parallel work | [`SplittableRng`], [`JumpableRng`], [`StreamPartitioner`], [`BoundedSubstream`] | stream selection or jump-ahead, with bounded draws per stream |
//! | Draws from a named distribution | the [distribution types](#distribution-families), [`Distribution`], [`CountDistribution`] | per family, for example Ziggurat for normal and exponential, Marsaglia–Tsang for gamma, BTPE for binomial, PTRS for Poisson |
//! | Density, CDF, quantile, moments | [`Density`], [`LogDensity`], [`Cdf`], [`Survival`], [`Quantile`], [`InverseSurvival`], [`Mean`], [`Variance`], [`Entropy`] | closed forms, special functions or quadrature, per family |
//! | Shifted, scaled, truncated or transformed laws, mixtures | [`Affine`], [`Truncated`], [`OrderStatistic`], [`Transformed`], [`LogTransformed`], [`Mixture`], [`Histogram`] | |
//! | Draws from a density you supply | [`AdaptiveRejectionSampler`], [`TransformedDensitySampler`], [`RatioUniformsSampler`], [`InversionSampler`] | adaptive rejection for log-concave densities, transformed density rejection, ratio of uniforms, numerical inversion |
//! | Density given by closures | [`LogTailsFn`], [`ConcaveDensityFn`], [`BoundedDensityFn`] | adapters for the samplers above |
//! | Root of a monotone log-probability, probability of an interval | [`invert_log_probability`], [`log_interval_probability`] | bracketing and bisection; tails in log space |
//! | Weighted choice from a fixed table | [`WeightedSampler`] | alias method, O(1) per draw |
//! | Shuffles, subsets, permutations, single choices | [`shuffle`], [`shuffle_in_place`], [`sample_indices`], [`choose`], [`uniform_index`] | Fisher–Yates; Floyd's algorithm for small subsets |
//! | Quasi-random points | [`QmcPlan`], [`QmcSequence`] | Sobol (Joe–Kuo directions), Halton, reverse Halton, Niederreiter base 2, with optional scrambling |
//! | Quasi-Monte Carlo integration with an error estimate | [`QmcReplicatePlan`] | independently randomized replicates |
//! | Gaussian paths and transformed QMC points | [`BrownianBridgePlan`], [`QmcMultivariateNormal`], [`QmcMultinomial`] | Brownian bridge construction; inverse-CDF transforms |
//! | Space-filling designs and their quality | [`latin_hypercube`], [`poisson_disk`], [`PoissonDisk`], [`discrepancy`], [`geometric_discrepancy`] | Latin hypercube sampling; Poisson disk sampling; L² discrepancies |
//! | Multivariate normal and t probabilities of a box | `rectangle` of [`MultivariateNormal`] and [`MultivariateStudentT`], [`MultivariateProbabilityPlan`] | randomized quasi-Monte Carlo with a replicate error estimate |
//!
//! Use [`Xoshiro256StarStar`] for a single stream, [`Pcg64`] or [`Philox4x64`]
//! for numbered streams, and [`Mt19937`] to reproduce published streams.
//!
//! # Generators
//!
//! | Type | Algorithm | Output | Period |
//! |---|---|---|---|
//! | [`SplitMix64`] | SplitMix (Steele, Lea and Flood; Vigna) | 64 bits | 2⁶⁴ |
//! | [`Pcg64`] | PCG-XSL-RR, 128-bit LCG with permuted output (O'Neill) | 64 bits | 2¹²⁸ per stream, 2¹²⁷ streams |
//! | [`Xoshiro256StarStar`] | xoshiro256** (Blackman and Vigna) | 64 bits | 2²⁵⁶ − 1 |
//! | [`Xoshiro128StarStar`] | xoshiro128** (Blackman and Vigna) | 32 bits | 2¹²⁸ − 1 |
//! | [`Mt19937`] | Mersenne Twister (Matsumoto and Nishimura) | 32 bits | 2¹⁹⁹³⁷ − 1 |
//! | [`Lcg64`] | linear congruential generator, Knuth's constants | 64 bits | 2⁶⁴ |
//! | [`Ranlux`], [`Ranlxs`], [`Ranlxd`] | subtract-with-carry with luxury levels (Lüscher) | 24 or 48 bits | about 10¹⁷¹ |
//! | [`Taus2`], [`Taus113`] | combined Tausworthe (L'Ecuyer) | 32 bits | about 2⁸⁸, 2¹¹³ |
//! | [`Gfsr4`] | four-tap generalized feedback shift register (Ziff) | 32 bits | 2⁹⁶⁸⁹ − 1 |
//! | [`Mrg`], [`Cmrg`] | multiple recursive and combined multiple recursive (L'Ecuyer) | 31 bits | about 2¹⁵⁵, 2¹⁸⁵ |
//! | [`Mrg32k3a`] | combined multiple recursive generator with streams and substreams (L'Ecuyer) | 32 bits | about 2¹⁹¹ |
//! | [`Philox4x64`], [`Threefry4x64`] | counter-based Philox4x64-10 and Threefry4x64-20 (Salmon et al.) | 64 bits | counter space |
//!
//! Each generator documents its own movement operations: [`SkipAhead`] to
//! advance by `n` outputs, [`JumpableRng`] for fixed long jumps,
//! [`SplittableRng`] for numbered streams, and [`SerializableState`] for
//! portable checkpoints. [`philox4x64_block`] and [`threefry4x64_block`]
//! evaluate the counter-based functions directly.
//!
//! # Seeding and reproducibility
//!
//! Seed generators with [`SeedableRng::from_seed`]. [`seed_sequence`] derives
//! several seeds from one master seed. [`SeedableRng::seed_from_entropy`] takes
//! an explicit [`scioxide_core::EntropySource`]; the crate never reads entropy
//! from the operating system or global state.
//!
//! For a given seed and sequence of calls, a generator returns the same
//! integers, floats, bytes, skip and jump results and split streams when the
//! algorithm and decoder protocol versions match. The same rule applies to
//! [`seed_sequence`] and the quasi-random families. Checkpoints reject unsupported
//! protocol versions. Samples from distributions are not covered:
//! sampling algorithms and their rounding may change between releases. Integer
//! and count samplers read [`BitGenerator::next_uniform_u64`] (or its 32-bit
//! form), and continuous samplers read [`Rng::next_float`]. For generators
//! whose native words do not span all 32 or 64 bits (RANLUX, RANLXS, RANLXD,
//! MRG, CMRG, MRG32k3a), a uniform word combines several native words, while a
//! float read decodes one native word.
//!
//! [`SerializableState`] writes a versioned, little-endian checkpoint with the
//! algorithm, the [`StreamIdentity`], the position and a CRC-32C checksum.
//! Restoring validates the checkpoint and requires the same stream identity
//! before replacing the generator.
//!
//! # Uniform variates
//!
//! Every generator implements [`Rng`] for `f32` and `f64`: a float read maps
//! one native word to a float in `[0, 1)` and caches nothing between calls.
//! Full-width words go through [`decode_u64`] or [`decode_u32`]; RANLUX
//! words are left-aligned first, and the `f64` reads of MRG, CMRG and MRG32k3a
//! divide the word by the modulus `m₁` (`m₁ + 1` for MRG32k3a). [`Rng::next_open_float`]
//! returns a float in `(0, 1)`, as needed before taking a logarithm; it rejects
//! zeros and fails with `ConvergenceFailure` after 1,024 of them in a row.
//!
//! You can implement [`Rng`] for your own source of uniforms. Distributions and
//! samplers then check each value it returns: a NaN or infinity is a
//! `NonFinite` error with callback origin, and a finite value outside `[0, 1)`
//! is `InvalidInput`. An overridden open-uniform provider is also checked and
//! must return a value strictly between 0 and 1. Values from a built-in generator are in range by
//! construction and are not checked again.
//!
//! # Probability laws
//!
//! A distribution implements one trait per operation it supports:
//! [`Density`], [`LogDensity`], [`Cdf`], [`LogCdf`], [`Survival`],
//! [`LogSurvival`], [`Quantile`], [`InverseSurvival`], [`Mean`], [`Variance`]
//! and [`Entropy`], with [`Support`] for the support interval. Discrete laws
//! use [`Mass`] and [`LogMass`] with integer arguments, and draw exact integer
//! counts through [`CountDistribution`]. [`Binomial`], [`Poisson`] and
//! [`Skellam`] draw through their prepared samplers ([`BinomialSampler`],
//! [`PoissonSampler`], [`SkellamSampler`]), which compute their constants
//! once; keep the sampler for a loop of single draws.
//!
//! - Queries reject NaN and infinite arguments. A zero probability has
//!   logarithm `−∞`.
//! - Upper tails are computed directly, not as `1 − F(x)`, so small tail
//!   probabilities keep their relative accuracy.
//! - Continuous quantiles return the left end of an interval where the CDF is
//!   constant; inverse survival functions return the right end. Discrete
//!   quantiles return the smallest `k` with `F(k) ≥ p`.
//! - The quantile at `p = 0` or `p = 1` is the end of the support, which may be
//!   infinite for a continuous law. For a discrete law with unbounded support
//!   it is `NotImplemented`.
//!
//! A law implements an operation only if it computes it. Cauchy has no
//! [`Mean`], and a mixture has no [`Entropy`]:
//!
//! ```compile_fail
//! use scioxide_random::{Cauchy, Mean};
//! fn requires_mean<D: Mean<f64>>(_: D) {}
//! requires_mean(Cauchy::new(0.0, 1.0).unwrap());
//! ```
//!
//! ```compile_fail
//! use scioxide_random::{Entropy, Mixture, Normal};
//! fn requires_entropy<D: Entropy<f64>>(_: D) {}
//! requires_entropy(Mixture::new(vec![Normal::new(0.0, 1.0).unwrap()], &[1.0]).unwrap());
//! ```
//!
//! Every law draws with `sample` (one sample), `sample_n` (`n` samples in an
//! owned `Vec`) and `sample_n_into` (a slice you provide). Vector- and
//! matrix-valued laws draw rows of `sample_width()` elements, and also
//! `sample_indexed_into`, which gives each row its own stream, so the result
//! does not depend on the number of threads. Matrix samples are stored row by
//! row in the law's `shape()`.
//!
//! Combinators build new laws from existing ones. A Rayleigh law with scale 1
//! is a Weibull law with shape 2 scaled by √2:
//!
//! ```
//! use scioxide_random::{Affine, Cdf, Distribution, SeedableRng, SplitMix64, WeibullMin};
//!
//! let rayleigh = Affine::new(WeibullMin::<f64>::new(2.0)?, 0.0, 2.0_f64.sqrt())?;
//! // Rayleigh CDF: F(x) = 1 − exp(−x²/2)
//! assert!((rayleigh.cdf(1.0)? - (1.0 - (-0.5_f64).exp())).abs() < 1e-13);
//! let mut rng = SplitMix64::from_seed(42);
//! assert!(rayleigh.sample(&mut rng)? >= 0.0);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! Directional laws draw angles and unit vectors:
//!
//! ```
//! use scioxide_random::{Density, Distribution, SeedableRng, SplitMix64, VonMises, VonMisesFisher};
//! # fn main() -> Result<(), scioxide_core::SciOxideError> {
//! let angular = VonMises::new(0.0_f64, 10.0)?;
//! let mut rng = SplitMix64::from_seed(541);
//! let theta = angular.sample(&mut rng)?;
//! let directional = VonMisesFisher::new(&[1.0_f64, 0.0, 0.0], 10.0)?;
//! let direction = directional.sample(&mut rng)?;
//! assert!(angular.pdf(theta)? > 0.0);
//! assert!(directional.contains_support(&direction)?);
//! # Ok(())
//! # }
//! ```
//!
//! # Distribution families
//!
//! Each type has its own definition, parameterization and example.
//!
//! - **Normal and related:** [`Normal`], [`LogNormal`], [`PowerNormal`],
//!   [`GeneralizedNormal`], [`ExponentiallyModifiedNormal`], [`DoubleParetoLogNormal`],
//!   [`FatigueLife`], [`Alpha`], [`Rice`], [`JohnsonSB`], [`JohnsonSU`].
//! - **Gamma, beta and exponential:** [`Gamma`], [`DoubleGamma`], [`GeneralizedGamma`],
//!   [`Pearson3`], [`ChiSquared`], [`Beta`], [`BetaPrime`], [`Burr12`], [`Mielke`],
//!   [`GaussHypergeometric`], [`Exponential`], [`ExponentialPower`],
//!   [`GeneralizedExponential`], [`InverseGaussian`], [`ReciprocalInverseGaussian`].
//! - **Weibull, extreme value and heavy tails:** [`WeibullMin`], [`DoubleWeibull`],
//!   [`ExponentiatedWeibull`], [`InverseWeibull`], [`GeneralizedExtremeValue`],
//!   [`GumbelRight`], [`Gompertz`], [`Kappa4`], [`Pareto`], [`GeneralizedPareto`],
//!   [`TruncatedPareto`], [`Cauchy`], [`StudentT`], [`LevyStable`], [`Moyal`],
//!   [`RelativisticBreitWigner`].
//! - **Logistic, Laplace, skewed and hyperbolic:** [`Logistic`], [`GeneralizedLogistic`],
//!   [`GeneralizedHalfLogistic`], [`HyperbolicSecant`], [`AsymmetricLaplace`],
//!   [`SkewNormal`], [`SkewCauchy`], [`JonesFaddySkewT`], [`CrystalBall`],
//!   [`TukeyLambda`], [`GeneralizedHyperbolic`], [`GeneralizedInverseGaussian`],
//!   [`NormalInverseGaussian`].
//! - **Bounded:** [`Uniform`], [`Trapezoid`], [`Anglit`], [`Argus`], [`Bradford`],
//!   [`Cosine`], [`IrwinHall`].
//! - **Discrete:** [`Binomial`], [`Poisson`], [`PoissonBinomial`], [`Skellam`],
//!   [`NegativeBinomial`], [`Geometric`], [`LogSeries`], [`BetaBinomial`],
//!   [`BetaNegativeBinomial`], [`YuleSimon`], [`Zipf`], [`Zipfian`], [`Boltzmann`],
//!   [`DiscreteLaplace`], [`DiscreteUniform`], [`Hypergeometric`],
//!   [`FisherNoncentralHypergeometric`], [`WalleniusNoncentralHypergeometric`].
//! - **Circular and directional:** [`VonMises`], [`VonMisesLine`], [`WrappedCauchy`],
//!   [`VonMisesFisher`], [`UniformDirection`].
//! - **Test statistics:** [`FisherF`], [`NoncentralChiSquared`], [`NoncentralF`],
//!   [`NoncentralStudentT`], [`Kolmogorov`], [`KolmogorovSmirnovOneSided`],
//!   [`KolmogorovSmirnovTwoSided`], [`StudentizedRange`].
//! - **Multivariate:** [`MultivariateNormal`], [`MultivariateStudentT`], [`Multinomial`],
//!   [`Dirichlet`], [`DirichletMultinomial`], [`MultivariateHypergeometric`],
//!   [`RandomTable`], [`NormalInverseGamma`].
//! - **Random matrices:** [`Wishart`], [`InverseWishart`], [`MatrixNormal`],
//!   [`MatrixT`], [`OrthogonalGroup`], [`SpecialOrthogonalGroup`], [`UnitaryGroup`],
//!   [`RandomCorrelation`].
//! - **Combinators:** [`Affine`], [`Truncated`], [`OrderStatistic`], [`Mixture`],
//!   [`Transformed`], [`LogTransformed`], [`Histogram`].
//!
//! # Quasi-random sequences
//!
//! [`QmcPlan`] prepares a Sobol, Halton, reverse Halton or Niederreiter base-2
//! sequence, optionally scrambled or randomly shifted. A [`QmcSequence`] cursor
//! skips to any index in O(1) and exports its position as portable bytes that
//! do not depend on the output type:
//!
//! ```
//! use scioxide_random::{QmcPlan, QmcRandomization, QuasiRng, SeedableRng, SplitMix64};
//! let plan = QmcPlan::sobol(32)?.with_randomization(
//!     QmcRandomization::LinearMatrixShift,
//!     &mut SplitMix64::from_seed(42),
//! )?;
//! let mut cursor = plan.sequence::<f64>()?;
//! cursor.skip_to((1_u64 << 32) + 17)?;
//! let mut point = [0.0; 32];
//! cursor.next_point_into(&mut point)?;
//! assert!(point.iter().all(|x| (0.0..1.0).contains(x)));
//! let recovered = QmcPlan::import_state(&cursor.export_state()?)?;
//! let _narrow = recovered.sequence::<f32>()?;
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Batch sampling
//!
//! Owned forms (`sample_n`, `points`, [`sample_indices`], [`shuffle`],
//! [`latin_hypercube`], [`poisson_disk`], `apply`) take no policy and no
//! workspace. They return their result only on success; on error they return
//! nothing and the generator stays advanced by the words it consumed. To
//! restore it, take a [`SnapshotGenerator::snapshot`] before the call.
//!
//! Forms that fill a slice you provide (`sample_n_into`, [`shuffle_in_place`],
//! [`QuasiRng::points_into`]) take a [`SamplingPolicy`] and a
//! [`SamplingWorkspace`]. The policy combines two independent choices:
//!
//! - [`OutputPolicy::Atomic`] writes the batch only if every draw succeeds; on
//!   error the output is unchanged. [`OutputPolicy::Incremental`] writes each
//!   completed sample (or row) as it goes, so completed samples stay written
//!   after an error.
//! - [`AdvanceOnError`] leaves the generator where it stopped. [`Rollback`]
//!   restores the generator to its state before the call; it requires a
//!   [`SnapshotGenerator`].
//!
//! A plain fill of a slice is a loop over `sample`. Rejection samplers consume
//! a varying number of words per draw. A successful batch consumes the same
//! words, in the same order, as the same number of single draws. Rejected
//! proposals consume words too.
//!
//! # Reusing plans and workspaces
//!
//! Four plans precompute what repeated calls share: [`QmcPlan`] (direction
//! numbers, bases and scramble matrices), [`QmcReplicatePlan`] (`R`
//! independently randomized copies of a sequence), [`BrownianBridgePlan`]
//! (the bridge weights of a time grid) and [`MultivariateProbabilityPlan`]
//! (the pivoted factor of a covariance). [`WeightedSampler`], [`PoissonDisk`],
//! the prepared count samplers and the distribution types are immutable after
//! construction too; every draw takes the generator as a separate argument.
//!
//! Atomic batches stage their output in a [`SamplingWorkspace`]. Size it with
//! [`SamplingPolicy::workspace_requirements`] or a law's own
//! `workspace_requirements` ([`SamplingRequirements`],
//! [`KolmogorovRequirements`]) and reserve it with `try_reserve`, or call
//! `workspace(request)` to do both. A sufficient workspace is reused without
//! allocation, as described in [Plans and
//! workspaces](scioxide_core#plans-and-workspaces).
//!
//! # Work limits
//!
//! Distributions that need iteration, quadrature or a numerical search use
//! [`LawOptions`]: `new(parameters…)` takes the default options and
//! `with_options(parameters…, options)` sets them; closed-form laws take no
//! options. The options set the tolerance, a work limit per operation
//! ([`LawOptions::work_budget`], default 65,536), a limit on random words per
//! draw ([`LawOptions::sample_word_budget`], default 65,536) and a byte limit
//! on storage the law allocates ([`LawOptions::max_bytes`], default 64 MiB).
//! Every rejection loop or stepwise search within one draw stops after 1,024
//! candidates. Exhausting a work or word limit is a `ConvergenceFailure`;
//! exceeding the byte limit is a `ResourceLimit`.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: an invalid parameter or size, such as zero dimension or
//!   an empty weight table, a finite value outside `[0, 1)` from a custom
//!   [`Rng`], or a corrupted or mismatched checkpoint.
//! - `DomainError`: a parameter outside the domain of the family, such as a
//!   nonpositive scale, or a probability outside `[0, 1]` given to a quantile.
//! - `NonFinite`: a NaN or infinite parameter or argument, or a non-finite
//!   value from a custom [`Rng`].
//! - `NumericalInstability`: a result outside the range of the output type,
//!   such as a sample `μ + σz` that overflows.
//! - `ConvergenceFailure`: a rejection loop exceeded 1,024 candidates, or an
//!   operation exhausted a [work limit](#work-limits).
//! - `OutOfBounds`: a [`BoundedSubstream`] has no draws left. The substream
//!   does not advance.
//! - `NotImplemented`: an operation that the parameters put out of reach, such
//!   as a quantile at `p = 1` for a discrete law with unbounded support.
//! - `ResourceLimit`, `SizeOverflow`, `AllocationFailure`: storage exceeds
//!   [`LawOptions::max_bytes`] or cannot be allocated.
//!
//! Each item lists its own conditions. On error, a caller-buffer batch leaves
//! the generator and the output as its [`SamplingPolicy`] says (see [Batch
//! sampling](#batch-sampling)); every other function may have advanced the
//! generator.
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide crates this crate depends
//!   on.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. The
//! API is the same.
//!
//! # References
//!
//! - L. Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986.
//! - W. Hörmann, J. Leydold and G. Derflinger, *Automatic Nonuniform Random
//!   Variate Generation*, Springer, 2004.
//! - D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
//!   Addison-Wesley, 1997, §3.2–3.4.
//! - P. L'Ecuyer, "Good parameters and implementations for combined multiple
//!   recursive random number generators", *Operations Research* 47(1), 1999,
//!   159–164.
//! - J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random
//!   numbers: as easy as 1, 2, 3", *Proc. SC11*, 2011.
//! - D. Blackman and S. Vigna, "Scrambled linear pseudorandom number
//!   generators", *ACM Trans. Math. Softw.* 47(4), 2021.
//! - S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better
//!   two-dimensional projections", *SIAM J. Sci. Comput.* 30(5), 2008,
//!   2635–2654.
//! - J. Dick and F. Pillichshammer, *Digital Nets and Sequences*, Cambridge
//!   University Press, 2010.
//! - NIST Digital Library of Mathematical Functions, <https://dlmf.nist.gov>.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg_attr(all(test, not(feature = "std")), macro_use)]
extern crate alloc;

#[cfg(test)]
extern crate std;

mod distribution;
mod float;
mod prng;
mod quasi;
mod sampling;
mod seed;
mod state;
mod stream;
mod traits;
mod transaction;
mod util;
mod workspace;

pub use distribution::LawOptions;
pub use distribution::{Alpha, Anglit, Argus, BetaPrime, Bradford, JohnsonSB, Mielke, Trapezoid};
pub use distribution::{
    AsymmetricLaplace, Cosine, DoubleGamma, DoubleWeibull, GeneralizedLogistic, GeneralizedNormal,
    GumbelRight, HyperbolicSecant,
};
pub use distribution::{
    Beta, BetaBinomial, BetaNegativeBinomial, Binomial, BinomialSampler, Boltzmann, Burr12, Cauchy,
    ChiSquared, CrystalBall, DiscreteLaplace, DiscreteUniform, Exponential, FisherF,
    FisherNoncentralHypergeometric, Gamma, GeneralizedExtremeValue, GeneralizedHyperbolic,
    GeneralizedInverseGaussian, GeneralizedPareto, Geometric, Hypergeometric, InverseWeibull,
    JonesFaddySkewT, Kolmogorov, KolmogorovRequirements, KolmogorovSmirnovOneSided,
    KolmogorovSmirnovTwoSided, KolmogorovWorkspace, LevyStable, LogNormal, LogSeries, Logistic,
    Moyal, Multinomial, MultivariateNormal, MultivariateNormalOptions, NegativeBinomial,
    NoncentralChiSquared, NoncentralF, NoncentralStudentT, Normal, NormalInverseGaussian, Pareto,
    Poisson, PoissonBinomial, PoissonSampler, RelativisticBreitWigner, Rice, Skellam,
    SkellamSampler, SkewCauchy, SkewNormal, StudentT, StudentizedRange, TestStatisticResult,
    Uniform, WalleniusNoncentralHypergeometric, YuleSimon, Zipf, Zipfian,
};
pub use distribution::{
    DoubleParetoLogNormal, ExponentiallyModifiedNormal, ExponentiatedWeibull,
    GeneralizedExponential, GeneralizedHalfLogistic, JohnsonSU, PowerNormal, TruncatedPareto,
};
pub use distribution::{
    ExponentialPower, FatigueLife, GeneralizedGamma, Gompertz, InverseGaussian, Pearson3,
    ReciprocalInverseGaussian, WeibullMin,
};
pub use distribution::{GaussHypergeometric, IrwinHall};
pub use distribution::{Kappa4, TukeyLambda};
pub use distribution::{UniformDirection, VonMises, VonMisesFisher, VonMisesLine, WrappedCauchy};
pub use float::{decode_u32, decode_u64, fill_uniforms};
pub use prng::{
    Cmrg, Gfsr4, Lcg64, Mrg, Mrg32k3a, Mt19937, Pcg64, Philox4x64, Ranlux, RanluxLuxury, Ranlxd,
    RanlxdLuxury, Ranlxs, RanlxsLuxury, SplitMix64, Taus2, Taus113, Threefry4x64,
    Xoshiro128StarStar, Xoshiro256StarStar, philox4x64_block, threefry4x64_block,
};
pub use quasi::{
    BrownianBridgePlan, CenteredDiscrepancy, DirectionNumberSet, DiscrepancyMethod,
    GeometricDiscrepancy, LatinHypercubeOptions, LatinHypercubeReport, LatinHypercubeTermination,
    PoissonDisk, PoissonDiskOptions, PoissonDiskReport, PoissonDiskTermination,
    PoissonDiskWorkspace, QmcCheckpoint, QmcFamily, QmcMultinomial, QmcMultivariateNormal, QmcPlan,
    QmcPosition, QmcRandomization, QmcReplicateEstimate, QmcReplicateOptions, QmcReplicatePlan,
    QmcScramble, QmcSequence, QmcTransportWorkspace, SobolDirection, discrepancy,
    geometric_discrepancy, latin_hypercube, latin_hypercube_into, latin_hypercube_with_options,
    poisson_disk,
};
pub use sampling::{
    WeightedSampler, choose, sample_indices, shuffle, shuffle_in_place, uniform_index,
};
pub use seed::seed_sequence;
pub use state::{GeneratorAlgorithm, StreamIdentity};
pub use stream::{BoundedSubstream, StreamPartitioner};
pub use traits::{
    BitGenerator, Cdf, CountDistribution, Density, Distribution, Entropy, InverseSurvival,
    JumpableRng, LogCdf, LogDensity, LogMass, LogSurvival, Mass, Mean, Quantile, QuasiRng, Rng,
    SeedableRng, SerializableState, SkipAhead, SplittableRng, Survival, Variance,
};
pub use workspace::{SamplingRequest, SamplingRequirements, SamplingWorkspace};

pub use transaction::{
    AdvanceOnError, OutputPolicy, Rollback, SamplingPolicy, SnapshotGenerator, StatePolicy,
};

#[cfg(test)]
#[path = "../tests/support/errors.rs"]
mod error_assertions;
mod law;
pub use law::{
    Support, log_interval_probability, log_interval_probability_scoped,
    log_interval_probability_with_options,
};
mod inversion;
pub use inversion::{
    InversionOptions, InversionReport, invert_log_probability, invert_log_probability_scoped,
    invert_log_probability_with_options,
};
mod histogram;
pub use histogram::Histogram;
mod transforms;
pub use transforms::{Affine, ContinuousLaw, OrderStatistic, Truncated};
mod mixture;
pub use mixture::Mixture;
mod nonlinear;
pub use nonlinear::{ExponentialTransform, LogCoordinateLaw, LogTransformed, Transformed};
pub use traits::{CountSupport, SignedCountSupport};

pub use distribution::{Dirichlet, DirichletMultinomial, MultivariateHypergeometric, RandomTable};
pub use distribution::{InverseWishart, MatrixNormal, MatrixT, Wishart};
pub use distribution::{
    MultivariateProbabilityOptions, MultivariateProbabilityPlan, MultivariateProbabilityReport,
    MultivariateProbabilityTermination, MultivariateProbabilityWorkspace,
};
pub use distribution::{MultivariateStudentT, NormalInverseGamma};
pub use distribution::{OrthogonalGroup, RandomCorrelation, SpecialOrthogonalGroup, UnitaryGroup};

mod universal;
pub use universal::{
    AdaptiveRejectionSampler, AdaptiveSamplingWorkspace, BoundedDensity, BoundedDensityFn,
    ConcaveDensity, ConcaveDensityFn, DensityTransform, InversionSampler, LogTailsFn,
    RatioUniformsBounds, RatioUniformsSampler, SamplingDiagnostics, TransformedDensitySampler,
    UniversalSample, UniversalSamplingOptions, UserDensity,
};
