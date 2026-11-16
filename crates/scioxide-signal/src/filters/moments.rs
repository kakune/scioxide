//! Running window mean and variance for the Wiener filter.
use scioxide_core::backend::two_sum;
use scioxide_core::{RealScalar, Scaled, SciOxideError};

use super::{local_moments, scalar};

#[derive(Clone, Copy, Debug)]
struct Sum<S: RealScalar> {
    parts: [S; 2],
    error: S,
}
impl<S: RealScalar> Sum<S> {
    const ZERO: Self = Self {
        parts: [S::ZERO; 2],
        error: S::ZERO,
    };

    #[inline]
    fn add(&mut self, value: S) {
        let (hi, residual) = two_sum(self.parts[0], value);
        let lo = self.parts[1] + residual;
        // The leading addition is error-free. Only the low-part addition
        // loses information. EPSILON (2u) and inflation bound its rounding,
        // including gradual underflow. A normal-sized absolute floor avoids
        // creating subnormal bound state for exact updates. Source admission
        // and the rebuild period bound all sum parts below overflow.
        self.error =
            (self.error + S::EPSILON * lo.abs() + S::MIN_POSITIVE) * (S::ONE + S::TWO * S::EPSILON);
        self.parts = [hi, lo];
    }

    #[inline]
    fn total(self) -> S {
        self.parts[0] + self.parts[1]
    }
}

#[derive(Clone, Copy, Debug)]
struct Native<S: RealScalar> {
    origin: S,
    first: Sum<S>,
    squares: Sum<S>,
    inverse: S,
    limit: S,
    mean: S,
    variance: S,
    nonzero: usize,
    extent: S,
}

impl<S: RealScalar> Native<S> {
    fn rebuild(
        samples: impl Iterator<Item = S> + Clone,
        count: usize,
    ) -> Result<Option<Self>, SciOxideError> {
        let count_scalar = scalar::<S>(count)?;
        let mut state = Self {
            origin: S::ZERO,
            first: Sum::ZERO,
            squares: Sum::ZERO,
            inverse: S::ONE / count_scalar,
            // Reserve headroom for differences, squares, and at most three
            // window widths of additions before a periodic rebuild.
            limit: (S::MAX / count_scalar).sqrt() / S::try_from_f64_lossy(16.)?,
            mean: S::ZERO,
            variance: S::ZERO,
            nonzero: 0,
            extent: S::ZERO,
        };
        let first = samples.clone().next().unwrap_or(S::ZERO);
        let mut shift = Sum::ZERO;
        for value in samples.clone() {
            if value.abs() > state.limit {
                return Ok(None);
            }
            shift.add(value - first);
        }
        // Center the retained coordinate at a working-precision mean. Choosing
        // the first observation would needlessly cancel Q against T^2/w on
        // smooth ramps, forcing a full reduction for almost every output.
        state.origin = first + shift.total() * state.inverse;
        for value in samples {
            if !state.change(value, false) {
                return Ok(None);
            }
        }
        Ok(state.reliable().then_some(state))
    }

    #[inline]
    fn change(&mut self, value: S, remove: bool) -> bool {
        if value.abs() > self.limit {
            return false;
        }
        let d = value - self.origin;
        let q = d * d;
        // Subnormal differences/squares need extended exponents. In particular
        // a nonzero difference must never disappear through squared underflow.
        if d != S::ZERO && (d.abs() < S::MIN_POSITIVE || q < S::MIN_POSITIVE) {
            return false;
        }
        if d != S::ZERO {
            if remove {
                self.nonzero -= 1;
            } else {
                self.nonzero += 1;
            }
        }
        self.extent = self.extent.max(d.abs());
        let sign = if remove { -S::ONE } else { S::ONE };
        self.first.add(sign * d);
        self.squares.add(sign * q);
        true
    }

    #[inline]
    fn reliable(&mut self) -> bool {
        let eight = S::TWO * S::TWO * S::TWO;
        let budget = eight * eight * S::EPSILON;
        if self.nonzero == 0 {
            self.first = Sum::ZERO;
            self.squares = Sum::ZERO;
            self.mean = self.origin;
            self.variance = S::ZERO;
            return true;
        }
        let sum = self.first.total();
        let q = self.squares.total();
        let contrast = sum * self.inverse;
        let correction = sum * contrast;
        let energy = q - correction;
        let first_error = (S::TWO * sum.abs() + self.first.error) * self.first.error * self.inverse;
        // Difference and square rounding, the centering correction, inverse
        // count, and final subtraction are covered by 8 eps (Q+C). Inflate
        // once more to cover evaluating the bound itself. This is the same
        // fixed-origin contraction bound used by stats rolling dispersions.
        let error = S::TWO
            * (self.squares.error
                + first_error
                + eight * S::EPSILON * (q.abs() + self.squares.error + correction + first_error));
        self.mean = self.origin + contrast;
        self.variance = energy * self.inverse;
        let mean_scale = self.origin.abs() + self.extent + contrast.abs();
        let mean_error = self.first.error * self.inverse + eight * S::EPSILON * mean_scale;
        self.mean.is_finite()
            && self.variance.is_finite()
            && error.is_finite()
            && q >= S::ZERO
            && energy > S::ZERO
            && error <= budget * energy
            && mean_error <= budget * mean_scale
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct State<S: RealScalar> {
    native: Option<Native<S>>,
    age: usize,
}
impl<S: RealScalar> State<S> {
    pub(super) const EMPTY: Self = Self {
        native: None,
        age: 0,
    };

    pub(super) fn evaluate(
        &mut self,
        samples: impl Iterator<Item = S> + Clone,
        count: usize,
        change: Option<(S, S)>,
    ) -> Result<Value<S>, SciOxideError> {
        if let (Some(native), Some((old, new))) = (&mut self.native, change)
            && self.age < count
        {
            self.age += 1;
            if native.change(old, true) && native.change(new, false) && native.reliable() {
                return Ok(Value::Native {
                    mean: native.mean,
                    variance: native.variance,
                });
            }
        }
        self.native = Native::rebuild(samples.clone(), count)?;
        self.age = 0;
        if let Some(native) = self.native {
            Ok(Value::Native {
                mean: native.mean,
                variance: native.variance,
            })
        } else {
            let (mean, variance) = local_moments(samples, count)?;
            Ok(Value::Extended { mean, variance })
        }
    }
}

pub(super) enum Value<S: RealScalar> {
    Native {
        mean: S,
        variance: S,
    },
    Extended {
        mean: Scaled<S>,
        variance: Scaled<S>,
    },
}
impl<S: RealScalar> Value<S> {
    pub(super) fn wiener(self, center: S, noise: S) -> Result<S, SciOxideError> {
        match self {
            Self::Native { mean, variance } => {
                if variance == S::ZERO && center == mean {
                    return Ok(center);
                }
                let result = if variance <= noise {
                    mean
                } else {
                    center + (mean - center) * (noise / variance)
                };
                if result.is_finite() {
                    return Ok(result);
                }
                // A finite convex candidate can have an overflowing contrast.
                Self::Extended {
                    mean: Scaled::from_value(mean)?,
                    variance: Scaled::from_value(variance)?,
                }
                .wiener(center, noise)
            }
            Self::Extended { mean, variance } => {
                let noise = Scaled::from_value(noise)?;
                if variance.abs_le(noise) {
                    return mean.round();
                }
                let center = Scaled::from_value(center)?;
                center
                    .add(mean.sub(center)?.mul(noise)?.div(variance)?)?
                    .round()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::unwrap_used)]
    fn smooth_windows_reuse_native_state_between_rebuilds() {
        let mut state = State::<f64>::EMPTY;
        let mut rebuilds = 0;
        for i in 0..1024 {
            let samples = (0..17).map(|j| (i + j) as f64 / 32.);
            let change = (i > 0).then(|| ((i - 1) as f64 / 32., (i + 16) as f64 / 32.));
            let value = state.evaluate(samples, 17, change).unwrap();
            assert!(matches!(value, Value::Native { .. }));
            if state.age == 0 {
                rebuilds += 1;
            }
        }
        assert!(rebuilds < 256);
    }
}
