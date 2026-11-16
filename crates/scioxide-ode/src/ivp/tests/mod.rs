//! Driver contracts that need crate-private steppers, direct stepping or the internal
//! initial-step estimator.
#![allow(clippy::unwrap_used)]

mod custom_steppers;
mod direct_steps;
mod event_localization;
mod initial_step;
mod nonlinear_orders;
mod step_limits;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};

use crate::OdeSystem;
use crate::stepper::Stepper;

/// `y' = -y` in every component.
struct Decay(usize);

impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Decay {
    fn dimension(&self) -> usize {
        self.0
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.iter_mut().zip(y).for_each(|(out, y)| *out = -*y);
        Ok(())
    }
}

/// Method storage reserved and begun for `dimension` components, as the driver does.
fn prepared<S, T, M>(method: &M, dimension: usize) -> M::Workspace
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: Stepper<S, T>,
{
    let mut workspace = M::Workspace::default();
    method.reserve(dimension, false, &mut workspace).unwrap();
    method.begin(dimension, false, &mut workspace).unwrap();
    workspace
}
