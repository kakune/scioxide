//! Compensated sinc derivatives for `|z| ≥ 1`. Keep trigonometric tails until after the
//! cancelling derivative numerators, then round once to `f64`.
use scioxide_core::RealScalar;
use scioxide_core::backend::ScalarKernelError;

#[derive(Clone, Copy)]
struct Pair(f64, f64);

impl Pair {
    const PI: Self = Self(core::f64::consts::PI, 1.224_646_799_147_353_2e-16);

    fn add(self, rhs: Self) -> Self {
        let high = self.0 + rhs.0;
        let b = high - self.0;
        let low = (self.0 - (high - b)) + (rhs.0 - b) + self.1 + rhs.1;
        let sum = high + low;
        Self(sum, low - (sum - high))
    }

    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }

    fn mul(self, rhs: Self) -> Self {
        let high = self.0 * rhs.0;
        let low = RealScalar::mul_add(self.0, rhs.0, -high)
            + self.0 * rhs.1
            + self.1 * rhs.0
            + self.1 * rhs.1;
        let sum = high + low;
        Self(sum, low - (sum - high))
    }

    fn div(self, rhs: Self) -> Self {
        let high = self.0 / rhs.0;
        let rest = self.add(rhs.mul(Self(high, 0.0)).neg());
        Self(high, 0.0).add(Self((rest.0 + rest.1) / rhs.0, 0.0))
    }

    fn value(self) -> f64 {
        self.0 + self.1
    }
}

fn sin_cos(angle: Pair, quadrant: i32) -> (Pair, Pair) {
    let square = angle.mul(angle).neg();
    let mut sine = angle;
    let mut cosine = Pair(1.0, 0.0);
    let mut s = sine;
    let mut c = cosine;
    // On |angle| <= pi/4, the omitted tails are below 2^-130.
    for k in 1..=18 {
        s = s.mul(square).div(Pair(f64::from(2 * k * (2 * k + 1)), 0.0));
        c = c.mul(square).div(Pair(f64::from((2 * k - 1) * 2 * k), 0.0));
        sine = sine.add(s);
        cosine = cosine.add(c);
    }
    match quadrant & 3 {
        0 => (sine, cosine),
        1 => (cosine, sine.neg()),
        2 => (sine.neg(), cosine.neg()),
        _ => (cosine.neg(), sine),
    }
}

pub(super) fn derivatives(x: f64, normalized: bool) -> Result<(f64, f64), ScalarKernelError> {
    // For huge arguments, the 1/x corrections cannot cause significant
    // cancellation except near a trigonometric zero. The compensated numerator
    // path still applies there; sequential division preserves derivative range.
    let (s, c) = if normalized {
        // Above 2^52 every binary64 argument is integral. Handle those exact
        // phases without attempting an integer quadrant conversion.
        if x.abs() >= 4_503_599_627_370_496.0 {
            let (s, c) = scioxide_core::backend::real_sin_cos_pi(x)?;
            (Pair(s, 0.0), Pair(c, 0.0))
        } else {
            let twice = RealScalar::round(2.0 * x);
            let reduced = Pair(x - twice * 0.5, 0.0).mul(Pair::PI);
            sin_cos(reduced, (twice % 4.0) as i32)
        }
    } else {
        let (quadrant, high, low) = scioxide_core::backend::real_reduce_radians(x)?;
        sin_cos(Pair(high, low), quadrant)
    };
    let reciprocal = Pair(1.0 / x, 0.0);
    let scale = if normalized { Pair::PI } else { Pair(1.0, 0.0) };
    // Avoid overflowing pi*x. Refine 1/x so the cancelling numerator also
    // includes its division residual.
    let inverse = reciprocal.add(Pair(RealScalar::mul_add(-reciprocal.0, x, 1.0) / x, 0.0));
    let sinc = s.mul(inverse).div(scale);
    let first = c.add(sinc.neg()).mul(inverse);
    let second = scale
        .mul(s)
        .neg()
        .add(first.mul(Pair(-2.0, 0.0)))
        .mul(inverse);
    Ok((first.value(), second.value()))
}
