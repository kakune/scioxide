//! Bin masses and weighted means accumulated about a shifted origin.
use crate::arithmetic::{
    RangeSum, center_difference_sum, divided_center, divided_center_scaled, native, promote, shift,
};
use scioxide_core::backend::scalar_from_u128_rounded;
use scioxide_core::{RealScalar, Scaled, ScaledAccumulator, SciOxideError};

/// Bin mass: an integer count stored as `u128`, or a sum of mass weights.
#[derive(Clone, Copy)]
pub(super) struct Mass {
    integer: u128,
    relative: RangeSum,
}
impl Mass {
    pub(super) fn zero() -> Self {
        Self {
            integer: 0,
            relative: RangeSum::ZERO,
        }
    }
    pub(super) fn count(count: usize) -> Self {
        Self {
            integer: count as u128,
            relative: RangeSum::ZERO,
        }
    }
    pub(super) fn relative(sum: RangeSum) -> Self {
        Self {
            integer: 0,
            relative: sum,
        }
    }
    #[inline]
    pub(super) fn add(&mut self, count: usize, weight: Option<f64>) -> Result<(), SciOxideError> {
        if let Some(weight) = weight {
            self.relative.add_product(weight, 1.0)?;
        } else {
            // There are at most usize::MAX records of at most usize::MAX weight.
            self.integer += count as u128;
        }
        Ok(())
    }
    pub(super) fn native(self, relative: bool) -> Result<Option<f64>, SciOxideError> {
        if !relative {
            return Ok(Some(scalar_from_u128_rounded(self.integer)?));
        }
        self.relative.native()
    }
    pub(super) fn scaled(self, relative: bool) -> Result<Scaled<f64>, SciOxideError> {
        if relative {
            self.relative.total()
        } else {
            Scaled::from_value(scalar_from_u128_rounded(self.integer)?)
        }
    }
    pub(super) fn requested_value<S: RealScalar>(self, relative: bool) -> Result<S, SciOxideError> {
        if relative {
            self.relative.total()?.round_to()
        } else {
            scalar_from_u128_rounded(self.integer)
        }
    }
    pub(super) fn value<S: RealScalar>(self, relative: bool) -> Result<Option<S>, SciOxideError> {
        if relative {
            Ok(self.relative.total()?.try_value_to().ok())
        } else {
            Ok(scalar_from_u128_rounded(self.integer).ok())
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Mean<S> {
    origin: Option<S>,
    shifted: RangeSum,
}
impl<S: RealScalar> Mean<S> {
    pub(super) fn empty() -> Self {
        Self {
            origin: None,
            shifted: RangeSum::ZERO,
        }
    }
    #[inline]
    pub(super) fn add(
        &mut self,
        value: S,
        weight: f64,
        old_mass: Mass,
        relative: bool,
    ) -> Result<(), SciOxideError> {
        let Some(origin) = self.origin else {
            self.origin = Some(value);
            return Ok(());
        };
        // A small origin avoids manufacturing huge cancellation when a tiny
        // observation arrives between extreme opposite values. Rebase both
        // compensated parts before changing the authoritative origin.
        if value.abs() < origin.abs() {
            if let Some(mass) = old_mass.native(relative)? {
                self.add_difference(origin, value, mass)?;
            } else {
                self.add_extended_difference(origin, value, old_mass.scaled(relative)?)?;
            }
            self.origin = Some(value);
        } else {
            self.add_difference(value, origin, weight)?;
        }
        Ok(())
    }
    #[inline]
    fn add_difference(&mut self, value: S, origin: S, weight: f64) -> Result<(), SciOxideError> {
        let (x, base) = (value.into_f64(), origin.into_f64());
        if native(x, 64) && native(base, 64) {
            for part in shift(x, base) {
                self.shifted.add_product(part, weight)?;
            }
            Ok(())
        } else {
            self.add_extended_difference(value, origin, Scaled::from_value(weight)?)
        }
    }
    fn add_extended_difference(
        &mut self,
        value: S,
        origin: S,
        weight: Scaled<f64>,
    ) -> Result<(), SciOxideError> {
        let difference = center_difference_sum(
            ScaledAccumulator::from_value(origin.into_f64())?,
            ScaledAccumulator::from_value(value.into_f64())?,
        )?;
        for part in difference.parts() {
            self.shifted.add_scaled(part.mul(weight)?)?;
        }
        Ok(())
    }
    pub(super) fn value(self, mass: Mass, relative: bool) -> Result<Option<S>, SciOxideError> {
        let Some(origin) = self.origin else {
            return Ok(None);
        };
        if let RangeSum::Native(numerator) = self.shifted
            && native(origin.into_f64(), 64)
            && let Some(mass) = mass.native(relative)?
        {
            let center = divided_center(origin.into_f64(), numerator.parts(), [mass, 0.0])?;
            return promote(center)?.total()?.round_to().map(Some);
        }
        let center = divided_center_scaled(
            ScaledAccumulator::from_value(origin.into_f64())?,
            self.shifted.extended()?,
            ScaledAccumulator::from_scaled(mass.scaled(relative)?),
        )?;
        center.total()?.round_to().map(Some)
    }
}
