//! Object-safe law adapter for fixture-driven sampling checks.
use scioxide_core::RealScalar;
use scioxide_random::{ContinuousLaw, Distribution, InverseSurvival, Quantile, SplitMix64};

pub trait CheckedLaw<S: RealScalar>: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S> {
    fn draw(&self, rng: &mut SplitMix64) -> Result<S, scioxide_core::SciOxideError>;
}
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S> + Distribution<S>>
    CheckedLaw<S> for D
{
    fn draw(&self, rng: &mut SplitMix64) -> Result<S, scioxide_core::SciOxideError> {
        self.sample(rng)
    }
}
