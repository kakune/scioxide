//! Shared SIMD radix codelets, inlined into the caller's ISA leaf.
//!
//! These register-only butterflies follow the scalar codelets' arithmetic
//! order. Architecture-specific leaves own memory access and complex products.

use scioxide_core::{Complex, RealScalar};

/// Register arithmetic for complex pairs in one scalar precision.
///
/// Every unsafe method requires the implementing lane's ISA to be available
/// (AVX2/FMA on x86, NEON on aarch64) and initialized register operands.
/// Addition, subtraction and multiplication operate componentwise without
/// contracting operations; rotation multiplies complex pairs by an exact ±i.
pub(super) trait CodeletLane: RealScalar {
    type Vector: Copy;

    unsafe fn splat(value: Self) -> Self::Vector;
    unsafe fn vadd(left: Self::Vector, right: Self::Vector) -> Self::Vector;
    unsafe fn vsub(left: Self::Vector, right: Self::Vector) -> Self::Vector;
    unsafe fn vmul(left: Self::Vector, right: Self::Vector) -> Self::Vector;
    /// Multiply by -i (forward) or +i (inverse).
    unsafe fn rotate<const INVERSE: bool>(value: Self::Vector) -> Self::Vector;
}

/// Apply one radix butterfly to initialized complex registers.
///
/// # Safety
/// The lane ISA must be available. `RADIX` must be 2, 3, 4, 5 or 7;
/// odd radices require at least `RADIX` forward roots in `roots`.
#[inline(always)]
pub(super) unsafe fn codelet<L: CodeletLane, const RADIX: usize, const INVERSE: bool>(
    values: &mut [L::Vector; 7],
    roots: &[Complex<L>],
) {
    // SAFETY: the caller proves the lane ISA, initialized registers and enough
    // roots; each selected radix uses fixed in-bounds root indices.
    unsafe {
        match RADIX {
            2 => {
                let a = values[0];
                let b = values[1];
                values[0] = L::vadd(a, b);
                values[1] = L::vsub(a, b);
            }
            3 => {
                let pair = L::vadd(values[1], values[2]);
                let difference = L::vsub(values[1], values[2]);
                let base = L::vsub(values[0], L::vmul(pair, L::splat(L::HALF)));
                let rotated = L::rotate::<INVERSE>(L::vmul(difference, L::splat(-roots[1].im)));
                values[0] = L::vadd(values[0], pair);
                values[1] = L::vadd(base, rotated);
                values[2] = L::vsub(base, rotated);
            }
            4 => {
                let even_sum = L::vadd(values[0], values[2]);
                let even_difference = L::vsub(values[0], values[2]);
                let odd_sum = L::vadd(values[1], values[3]);
                let odd_difference = L::vsub(values[1], values[3]);
                let rotated = L::rotate::<INVERSE>(odd_difference);
                values[0] = L::vadd(even_sum, odd_sum);
                values[1] = L::vadd(even_difference, rotated);
                values[2] = L::vsub(even_sum, odd_sum);
                values[3] = L::vsub(even_difference, rotated);
            }
            5 => radix5::<L, INVERSE>(values, roots),
            7 => radix7::<L, INVERSE>(values, roots),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn radix5<L: CodeletLane, const INVERSE: bool>(
    values: &mut [L::Vector; 7],
    roots: &[Complex<L>],
) {
    // SAFETY: the caller proves the lane ISA, initialized registers and five
    // roots. All arithmetic stays in registers with fixed in-bounds indices.
    unsafe {
        let x0 = values[0];
        let c1 = L::splat(roots[1].re);
        let c2 = L::splat(roots[2].re);
        let s1 = L::splat(-roots[1].im);
        let s2 = L::splat(-roots[2].im);
        let p1 = L::vadd(values[1], values[4]);
        let p2 = L::vadd(values[2], values[3]);
        let d1 = L::vsub(values[1], values[4]);
        let d2 = L::vsub(values[2], values[3]);
        // Complete each independent output pair before starting the next to
        // limit live registers without changing any sum's association.
        values[0] = L::vadd(L::vadd(x0, p1), p2);
        let base1 = L::vadd(L::vadd(x0, L::vmul(p1, c1)), L::vmul(p2, c2));
        let rot1 = L::rotate::<INVERSE>(L::vadd(L::vmul(d1, s1), L::vmul(d2, s2)));
        values[1] = L::vadd(base1, rot1);
        values[4] = L::vsub(base1, rot1);
        let base2 = L::vadd(L::vadd(x0, L::vmul(p1, c2)), L::vmul(p2, c1));
        let rot2 = L::rotate::<INVERSE>(L::vsub(L::vmul(d1, s2), L::vmul(d2, s1)));
        values[2] = L::vadd(base2, rot2);
        values[3] = L::vsub(base2, rot2);
    }
}

#[inline(always)]
unsafe fn radix7<L: CodeletLane, const INVERSE: bool>(
    values: &mut [L::Vector; 7],
    roots: &[Complex<L>],
) {
    // SAFETY: the caller proves the lane ISA, initialized registers and seven
    // roots. All arithmetic stays in registers with fixed in-bounds indices.
    unsafe {
        let x0 = values[0];
        let p1 = L::vadd(values[1], values[6]);
        let p2 = L::vadd(values[2], values[5]);
        let p3 = L::vadd(values[3], values[4]);
        let d1 = L::vsub(values[1], values[6]);
        let d2 = L::vsub(values[2], values[5]);
        let d3 = L::vsub(values[3], values[4]);
        let [c1, c2, c3] = [
            L::splat(roots[1].re),
            L::splat(roots[2].re),
            L::splat(roots[3].re),
        ];
        let [s1, s2, s3] = [
            L::splat(-roots[1].im),
            L::splat(-roots[2].im),
            L::splat(-roots[3].im),
        ];
        let b1 = L::vadd(
            L::vadd(L::vadd(x0, L::vmul(p1, c1)), L::vmul(p2, c2)),
            L::vmul(p3, c3),
        );
        let b2 = L::vadd(
            L::vadd(L::vadd(x0, L::vmul(p1, c2)), L::vmul(p2, c3)),
            L::vmul(p3, c1),
        );
        let b3 = L::vadd(
            L::vadd(L::vadd(x0, L::vmul(p1, c3)), L::vmul(p2, c1)),
            L::vmul(p3, c2),
        );
        let r1 = L::rotate::<INVERSE>(L::vadd(
            L::vadd(L::vmul(d1, s1), L::vmul(d2, s2)),
            L::vmul(d3, s3),
        ));
        let r2 = L::rotate::<INVERSE>(L::vsub(
            L::vsub(L::vmul(d1, s2), L::vmul(d2, s3)),
            L::vmul(d3, s1),
        ));
        let r3 = L::rotate::<INVERSE>(L::vadd(
            L::vsub(L::vmul(d1, s3), L::vmul(d2, s1)),
            L::vmul(d3, s2),
        ));
        values[0] = L::vadd(L::vadd(L::vadd(x0, p1), p2), p3);
        values[1] = L::vadd(b1, r1);
        values[6] = L::vsub(b1, r1);
        values[2] = L::vadd(b2, r2);
        values[5] = L::vsub(b2, r2);
        values[3] = L::vadd(b3, r3);
        values[4] = L::vsub(b3, r3);
    }
}
