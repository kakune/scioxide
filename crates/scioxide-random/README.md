# scioxide-random

Random number generation for Rust.
The crate provides seeded pseudo-random number generators with skip-ahead, numbered streams and portable checkpoints; continuous, discrete, circular, multivariate and matrix probability distributions with densities, distribution functions, quantiles, moments and samplers; quasi-random sequences and space-filling designs; and shuffling, subset selection and weighted choice.
No generator is cryptographically secure, and no seed is read from the operating system.

## Main entry points

- Generators: `SplitMix64`, `Pcg64`, `Xoshiro256StarStar`, `Xoshiro128StarStar`, `Mt19937`, `Lcg64`, `Ranlux`, `Ranlxs`, `Ranlxd`, `Taus2`, `Taus113`, `Gfsr4`, `Mrg`, `Cmrg`, `Mrg32k3a`, `Philox4x64`, `Threefry4x64`.
- Generator traits and streams: `BitGenerator`, `Rng`, `SeedableRng`, `seed_sequence`, `SkipAhead`, `JumpableRng`, `SplittableRng`, `SerializableState`, `StreamPartitioner`, `BoundedSubstream`.
- Distributions: `Normal`, `Gamma`, `Beta`, `StudentT`, `Binomial`, `Poisson`, `MultivariateNormal`, `Dirichlet`, `Wishart`, `VonMises` and many more, with the traits `Distribution`, `CountDistribution`, `Density`, `Cdf`, `Survival`, `Quantile`, `Mean`, `Variance`, `Entropy` and options `LawOptions` (`new` uses the defaults, `with_options` sets them).
- Combinators and custom densities: `Affine`, `Truncated`, `OrderStatistic`, `Mixture`, `Transformed`, `Histogram`, `AdaptiveRejectionSampler`, `RatioUniformsSampler`, `InversionSampler`, and the closure adapters `LogTailsFn`, `ConcaveDensityFn`, `BoundedDensityFn`.
- Quasi-random and designs: `QmcPlan`, `QmcReplicatePlan`, `BrownianBridgePlan`, `latin_hypercube`, `poisson_disk`, `PoissonDisk`, `discrepancy`.
- Sampling: `shuffle`, `shuffle_in_place`, `choose`, `sample_indices`, `uniform_index`, `WeightedSampler`.
- Batches: `sample_n` returns an owned `Vec`; `sample_n_into` fills a slice under a `SamplingPolicy` (`OutputPolicy` with `AdvanceOnError` or `Rollback`) and a `SamplingWorkspace`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-random) for a quick start, a guide to choosing a generator or method, and the rules for reproducibility, batches and errors.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
