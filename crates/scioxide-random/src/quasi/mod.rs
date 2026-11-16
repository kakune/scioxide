//! Quasi-random sequences, space-filling designs and QMC diagnostics.
//!
//! [`QmcPlan`] prepares Sobol (Joe–Kuo or custom direction numbers), Halton,
//! reverse Halton and Niederreiter base-2 sequences, with optional
//! randomization, and hands out [`QmcSequence`] cursors and [`QmcCheckpoint`]s.
//! [`QmcReplicatePlan`] estimates means from independent randomized
//! replicates, and [`BrownianBridgePlan`] maps normal coordinates to Brownian
//! paths in bridge order.
//! [`latin_hypercube`], [`poisson_disk`] and [`PoissonDisk`] build designs, and
//! [`discrepancy`] and [`geometric_discrepancy`] measure point sets.
//! [`QmcMultivariateNormal`] and [`QmcMultinomial`] transport QMC points to laws.

mod bridge;
mod checkpoint;
mod direction_numbers;
mod discrepancy;
mod halton;
mod niederreiter;
mod plan;
mod scramble;
pub use discrepancy::{
    CenteredDiscrepancy, DiscrepancyMethod, GeometricDiscrepancy, discrepancy,
    geometric_discrepancy,
};

pub use bridge::BrownianBridgePlan;
pub use checkpoint::QmcCheckpoint;
pub use direction_numbers::{DirectionNumberSet, SobolDirection};
pub use plan::{QmcFamily, QmcPlan, QmcPosition, QmcRandomization, QmcScramble, QmcSequence};
mod design;
pub use design::{
    LatinHypercubeOptions, LatinHypercubeReport, LatinHypercubeTermination, PoissonDisk,
    PoissonDiskOptions, PoissonDiskReport, PoissonDiskTermination, PoissonDiskWorkspace,
    latin_hypercube, latin_hypercube_into, latin_hypercube_with_options, poisson_disk,
};
mod transport;
pub use transport::{QmcMultinomial, QmcMultivariateNormal, QmcTransportWorkspace};
mod replicate;
pub(crate) use replicate::replicate_moments;
pub use replicate::{QmcReplicateEstimate, QmcReplicateOptions, QmcReplicatePlan};

#[cfg(test)]
mod kernel_tests;
