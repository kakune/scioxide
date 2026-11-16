//! Distributions of angles and of directions on the unit sphere.
use crate::law::{
    argument, constant as number, exhausted, finite, invalid, log1mexp, nonnegative, probability,
};
use crate::{Density, Distribution, Entropy, LogDensity, Mean, Rng, Variance};
use core::f64::consts::{PI, TAU};
use scioxide_core::{Budget, RealScalar, SciOxideError};

mod angular;
mod directional;
pub use angular::{VonMises, VonMisesLine, WrappedCauchy};
pub use directional::{UniformDirection, VonMisesFisher};

fn scaled_bessel(order: f64, x: f64, left: &mut usize) -> Result<f64, SciOxideError> {
    let budget = core::num::NonZeroUsize::new(*left).ok_or_else(exhausted)?;
    let result = scioxide_special::bessel_i_scaled_with_options(
        order,
        x,
        scioxide_special::BesselOptions::default().with_iteration_budget(Budget::from(budget)),
    )?;
    *left = left.saturating_sub(result.counts().iterations());
    result.try_value()
}

// Preserve the shared scalar transaction contract while admitting family storage.
macro_rules! scalar_batches {
    () => {
        fn sample_n_into<R: Rng<S>>(
            &self,
            rng: &mut R,
            policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
            output: &mut [S],
            workspace: &mut crate::SamplingWorkspace<S>,
        ) -> Result<(), SciOxideError> {
            crate::workspace::sample_scalars(
                rng,
                policy,
                output,
                S::ZERO,
                workspace,
                self.options().max_bytes(),
                |rng| self.sample(rng),
            )
        }
        fn sample_n<R: Rng<S>>(
            &self,
            rng: &mut R,
            n: usize,
        ) -> Result<alloc::vec::Vec<S>, SciOxideError> {
            self.options().admit::<S>(n)?;
            crate::workspace::sample_owned(rng, n, S::ZERO, |rng| self.sample(rng))
        }
    };
}
use scalar_batches;
