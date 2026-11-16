// Ring-only bounded integers for allocation tests.
use scioxide_poly::ExactCoefficient;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Integer(pub(crate) i64);

impl core::ops::Add for Integer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl core::ops::Sub for Integer {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl core::ops::Mul for Integer {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl core::ops::Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[allow(clippy::unwrap_used)]
impl ExactCoefficient for Integer {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn from_usize(value: usize) -> Self {
        Self(i64::try_from(value).unwrap())
    }
}
