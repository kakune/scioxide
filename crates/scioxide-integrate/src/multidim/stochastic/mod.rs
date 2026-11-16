//! Monte Carlo integration over finite boxes: importance sampling, replicated
//! randomized QMC, VEGAS maps and MISER strata.
//!
//! Plans are immutable. [`MonteCarloIntegrator::integrate_into`] evaluates an estimate
//! from a [`UniformSource`]; [`VegasTrainer::train_into`] and [`MiserTrainer::train_into`]
//! produce a [`VegasMap`] or [`MiserStrata`] that a later evaluation uses.
//!
//! # Schedule independence
//!
//! Work is split into children: the single run of an independent estimate, the
//! replicates of a replicate mean, the strata of a stratified plan, the
//! iterations of VEGAS training and the pilot visits of MISER training. Child `c`
//! draws `n_c` observations in `⌈n_c / B⌉` blocks of [`SAMPLE_BLOCK`] = `B`
//! observations, numbered after the blocks of children `0..c`. Random block `b`
//! draws from `root.substream(stream, b)`; QMC replicates index their points
//! directly. Each block is summarized sequentially and the summaries are merged
//! in block order, children in index order, so the result does not depend on the
//! execution policy or the number of workers.

mod kernel;
mod miser;
mod options;
mod plan;
mod proposal;
mod source;
mod statistics;
mod training;
mod vegas;
mod workspace;

pub use miser::{MiserOptions, MiserStrata, MiserTrainer, MiserWorkspace};
pub use options::MonteCarloOptions;
pub use plan::{MonteCarloIntegrator, integrate_monte_carlo};
pub use proposal::{MonteCarloProposal, ProposalUniforms, UniformProposal};
pub use source::{RandomStreams, UniformSource};
pub use statistics::{MonteCarloStatistics, VarianceBasis};
pub use training::TrainingReport;
pub use vegas::{VegasMap, VegasOptions, VegasTrainer, VegasWorkspace};
pub use workspace::{MonteCarloChild, MonteCarloWorkspace};

use source::Source;

/// Number of observations in one sample block of a Monte Carlo plan or training run.
///
/// Observations are drawn in blocks of this size. Each block draws its uniforms
/// from its own substream of the [`RandomStreams`] root, and blocks are the unit
/// of parallel work. [`RandomStreams`] describes how blocks are numbered.
pub const SAMPLE_BLOCK: usize = 1024;

/// Builds an `InvalidInput` error.
fn invalid(
    argument: scioxide_core::Argument,
    reason: scioxide_core::InvalidInputReason,
    detail: &'static str,
) -> scioxide_core::SciOxideError {
    crate::kernel::invalid(argument, reason, detail)
}

#[cfg(test)]
mod tests;
