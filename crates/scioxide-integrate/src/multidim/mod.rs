//! Integration in several dimensions.
//!
//! | Plan | Domain | Method |
//! |---|---|---|
//! | [`CubaturePlan`] | box | adaptive G7K15 for `d = 1`, Genz–Malik degree 7/5 for `d ≥ 2` |
//! | [`SparseGridPlan`] | box | dimension-adaptive Smolyak quadrature, Clenshaw–Curtis |
//! | [`ProductRule`] | box | tensor product of a fixed one-dimensional rule |
//! | [`IteratedDomainPlan`] | variable limits | iterated adaptive G7K15 |
//! | [`SimplexRule`] | simplex | positive interior rules or Grundmann–Möller |
//! | [`SphereProductRule`] | sphere surface | Gauss–Legendre in `cos θ`, trapezoidal in `φ` |
//! | [`LebedevRule`] | sphere surface | Lebedev–Laikov |
//! | [`MonteCarloIntegrator`] | box | importance, replicated, randomized QMC and stratified sampling |
//! | [`VegasTrainer`] | box | VEGAS adaptive importance sampling |
//! | [`MiserTrainer`] | box | MISER recursive stratified sampling |

mod breakpoints;
mod cubature;
mod iterated;
mod sampling;
pub use iterated::{
    CoordinateMap, IdentityMap, IteratedDomainPlan, IteratedOptions, IteratedWorkspace,
    integrate_iterated,
};
mod geometric;
mod positive_simplex_data;
// Preserve the source decimals in the independently transcribed Lebedev table.
#[allow(clippy::excessive_precision)]
mod lebedev_data;
mod product_rule;
mod sparse_grid;
mod stochastic;

pub use stochastic::{
    MiserOptions, MiserStrata, MiserTrainer, MiserWorkspace, MonteCarloChild, MonteCarloIntegrator,
    MonteCarloOptions, MonteCarloProposal, MonteCarloStatistics, MonteCarloWorkspace,
    ProposalUniforms, RandomStreams, SAMPLE_BLOCK, TrainingReport, UniformProposal, UniformSource,
    VarianceBasis, VegasMap, VegasOptions, VegasTrainer, VegasWorkspace, integrate_monte_carlo,
};

pub use cubature::{CubaturePlan, CubatureWorkspace, integrate_box};
pub use geometric::{
    GeometricOptions, GeometricWorkspace, LebedevRule, SimplexMeasure, SimplexRule,
    SphereProductRule,
};
pub use product_rule::{ProductOptions, ProductRule, ProductWorkspace};
pub use sparse_grid::{SparseGridOptions, SparseGridPlan, SparseGridWorkspace};
