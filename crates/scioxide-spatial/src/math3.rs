//! Native-scalar fixed-size geometry arithmetic; evaluation order is intentional.
use scioxide_core::RealScalar;

// Exact small integer constant built by binary doubling. It is infallible
// and exact for both f32 and f64, unlike a fallible scalar conversion at
// every call site.
pub(crate) fn n<S: RealScalar>(mut value: u16) -> S {
    let mut result = S::ZERO;
    let mut place = S::ONE;
    while value != 0 {
        if value & 1 != 0 {
            result += place;
        }
        value >>= 1;
        place = place + place;
    }
    result
}
// Fixed-width geometry contraction, deliberately retaining native scalar
// rounding. Rotation operands are normalized; spline arithmetic is checked at
// preparation/evaluation boundaries. This does not promise the extended-range
// behavior of core::checked_sum_of_products for arbitrary reductions.
pub(crate) fn dot<S: RealScalar>(a: [S; 3], b: [S; 3]) -> S {
    a.iter().zip(b).map(|(&x, y)| x * y).sum()
}
pub(crate) fn cross<S: RealScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn norm<S: RealScalar>(a: [S; 3]) -> S {
    a[0].hypot(a[1]).hypot(a[2])
}
pub(crate) fn add<S: RealScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    core::array::from_fn(|i| a[i] + b[i])
}
pub(crate) fn scale<S: RealScalar>(a: [S; 3], b: S) -> [S; 3] {
    a.map(|x| x * b)
}

pub(crate) type Mat<S> = [[S; 3]; 3];
pub(crate) fn sub<S: RealScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}
pub(crate) fn mul<S: RealScalar>(a: Mat<S>, b: [S; 3]) -> [S; 3] {
    a.map(|row| dot(row, b))
}
pub(crate) fn product<S: RealScalar>(a: Mat<S>, b: Mat<S>) -> Mat<S> {
    core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

pub(crate) fn transpose<S: Copy>(a: Mat<S>) -> Mat<S> {
    core::array::from_fn(|i| core::array::from_fn(|j| a[j][i]))
}
