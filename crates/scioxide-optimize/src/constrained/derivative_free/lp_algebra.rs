//! Givens rotations and reductions for Powell’s active-set LP.
use scioxide_core::RealScalar as Scalar;
pub(crate) fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    // Two-dimensional LPs benefit from unrolled reductions. Keep Scalar::sum
    // so its identity and accumulation order also apply to signed zeros.
    if let ([a0, a1], [b0, b1]) = (a, b) {
        return [*a0 * *b0, *a1 * *b1].into_iter().sum();
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn dot_abs<F: Scalar>(a: &[F], b: &[F]) -> F {
    if let ([a0, a1], [b0, b1]) = (a, b) {
        return [a0.abs() * b0.abs(), a1.abs() * b1.abs()].into_iter().sum();
    }
    a.iter().zip(b).map(|(&x, &y)| x.abs() * y.abs()).sum()
}

pub(crate) fn dot_pair<F: Scalar>(a: &[F], b: &[F]) -> (F, F) {
    if a.len() == 2 && b.len() == 2 {
        return (dot(a, b), dot_abs(a, b));
    }
    // Scalar::sum starts floating-point reductions at negative zero. Reusing
    // its identity preserves the signs of empty and all-negative-zero sums.
    let identity: F = core::iter::empty().sum();
    a.iter()
        .zip(b)
        .fold((identity, identity), |(signed, absolute), (&x, &y)| {
            (signed + x * y, absolute + x.abs() * y.abs())
        })
}

pub(crate) fn col<F>(a: &[F], r: usize, j: usize) -> &[F] {
    &a[j * r..(j + 1) * r]
}

pub(crate) fn fsign<F: Scalar>(a: F, b: F) -> F {
    if b >= F::ZERO { a.abs() } else { -a.abs() }
}

pub(crate) fn hypotenuse<F: Scalar>(x0: F, x1: F) -> F {
    let (a, b) = (x0.abs(), x1.abs());
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if hi <= F::ZERO {
        F::ZERO
    } else {
        let t = lo / hi;
        hi * (t * t + F::ONE).sqrt()
    }
}

pub(crate) fn isminor<F: Scalar>(x: F, refv: F) -> bool {
    let sensitivity = F::ONE / F::TEN;
    let two = F::TWO;
    let refa = refv.abs() + sensitivity * x.abs();
    let refb = refv.abs() + two * sensitivity * x.abs();
    refv.abs() >= refa || refa >= refb
}

pub(crate) fn planerot<F: Scalar>(x0: F, x1: F) -> (F, F) {
    let zero = F::ZERO;
    let one = F::ONE;
    let eps = F::EPSILON;
    if x0.is_nan() || x1.is_nan() {
        return (one, zero);
    }
    if x0.is_infinite() && x1.is_infinite() {
        let r = one / (one + one).sqrt();
        return (fsign(r, x0), fsign(r, x1));
    }
    if x0.abs() <= zero && x1.abs() <= zero {
        return (one, zero);
    }
    if x1.abs() <= eps * x0.abs() {
        return (fsign(one, x0), zero);
    }
    if x0.abs() <= eps * x1.abs() {
        return (zero, fsign(one, x1));
    }
    if x0.abs() > x1.abs() {
        let t = x1 / x0;
        let u = fsign((one + t * t).sqrt(), x0);
        (one / u, t / u)
    } else {
        let t = x0 / x1;
        let u = fsign((one + t * t).sqrt(), x1);
        (t / u, one / u)
    }
}
