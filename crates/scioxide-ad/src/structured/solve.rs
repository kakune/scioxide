//! Linear solves that factor the base matrix once and apply the factors to every
//! derivative component.

use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use crate::real::sealed::{self, Token};
use crate::{BaseReal, Dual, Lanes, Real};

fn buffer<T: Clone>(len: usize, fill: T) -> Result<Vec<T>, SciOxideError> {
    scioxide_core::backend::try_filled(len, fill, scioxide_core::quantity!("solve_scratch"))
}

/// LU factors with partial pivoting of the base values of a matrix, one
/// factorization per lane (`P·A = L·U`, `L` unit lower triangular).
#[derive(Debug, Clone)]
pub struct Factor<S> {
    order: usize,
    lanes: usize,
    lu: Vec<S>,
    pivots: Vec<usize>,
}

impl<S: RealScalar> Factor<S> {
    /// Factors the base values of the `order × order` row-major `matrix`.
    pub(crate) fn new<R: Real<S>>(matrix: &[R], order: usize) -> Result<Self, SciOxideError> {
        let (n, lanes) = (order, R::LANES);
        let size = scioxide_core::backend::checked_product(
            &[lanes, n, n],
            scioxide_core::quantity!("solve_scratch"),
        )?;
        let pivot_count = scioxide_core::backend::checked_mul(
            lanes,
            n,
            scioxide_core::quantity!("solve_scratch"),
        )?;
        let mut lu = buffer(size, S::ZERO)?;
        let mut pivots = buffer(pivot_count, 0)?;
        for lane in 0..lanes {
            let a = &mut lu[lane * n * n..(lane + 1) * n * n];
            for (target, entry) in a.iter_mut().zip(matrix) {
                *target = entry.base().lane(lane).unwrap_or(S::NAN);
            }
            let pivot_row = &mut pivots[lane * n..(lane + 1) * n];
            for k in 0..n {
                let (p, magnitude) = (k..n).fold((k, -S::ONE), |best, i| {
                    let m = NumericScalar::abs(a[i * n + k]);
                    if m > best.1 { (i, m) } else { best }
                });
                if magnitude <= S::ZERO || !magnitude.is_finite() {
                    return Err(SciOxideError::singular(Some(k)));
                }
                pivot_row[k] = p;
                if p != k {
                    for j in 0..n {
                        a.swap(k * n + j, p * n + j);
                    }
                }
                let pivot = a[k * n + k];
                for i in k + 1..n {
                    let l = a[i * n + k] / pivot;
                    a[i * n + k] = l;
                    for j in k + 1..n {
                        let update = l * a[k * n + j];
                        a[i * n + j] -= update;
                    }
                }
            }
        }
        Ok(Self {
            order,
            lanes,
            lu,
            pivots,
        })
    }

    /// Returns the order `n`.
    pub(crate) const fn order(&self) -> usize {
        self.order
    }

    /// Solves in place with the factors of one lane, in `T` arithmetic.
    fn substitute<T: Real<S>>(&self, lane: usize, x: &mut [T], transpose: bool) {
        let n = self.order;
        let columns = x.len() / n.max(1);
        let lu = &self.lu[lane * n * n..(lane + 1) * n * n];
        let pivots = &self.pivots[lane * n..(lane + 1) * n];
        let row_update = |x: &mut [T], target: usize, source: usize, factor: S| {
            for column in 0..columns {
                let update = x[source * columns + column] * factor;
                x[target * columns + column] -= update;
            }
        };
        if transpose {
            // Aᵀ = Uᵀ·Lᵀ·P: forward with Uᵀ, backward with Lᵀ, then the inverse
            // row interchanges.
            for k in 0..n {
                for j in 0..k {
                    row_update(x, k, j, lu[j * n + k]);
                }
                for column in 0..columns {
                    x[k * columns + column] /= lu[k * n + k];
                }
            }
            for k in (0..n).rev() {
                for j in k + 1..n {
                    row_update(x, k, j, lu[j * n + k]);
                }
            }
            for k in (0..n).rev() {
                if pivots[k] != k {
                    for column in 0..columns {
                        x.swap(k * columns + column, pivots[k] * columns + column);
                    }
                }
            }
        } else {
            // `new` applies each interchange to whole rows, earlier multipliers
            // included, so `L` belongs to `P·A`: permute first, then substitute.
            for (k, &p) in pivots.iter().enumerate() {
                if p != k {
                    for column in 0..columns {
                        x.swap(k * columns + column, p * columns + column);
                    }
                }
            }
            for k in 0..n {
                for i in k + 1..n {
                    row_update(x, i, k, lu[i * n + k]);
                }
            }
            for k in (0..n).rev() {
                for j in k + 1..n {
                    row_update(x, k, j, lu[k * n + j]);
                }
                for column in 0..columns {
                    x[k * columns + column] /= lu[k * n + k];
                }
            }
        }
    }

    /// Solves in place for base values, one lane at a time.
    pub(crate) fn solve_base<B: BaseReal<S>>(
        &self,
        rhs: &mut [B],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        if self.lanes == 1 {
            self.substitute(0, rhs, transpose);
            return Ok(());
        }
        let len = rhs.len();
        let size = scioxide_core::backend::checked_mul(
            self.lanes,
            len,
            scioxide_core::quantity!("solve_scratch"),
        )?;
        let mut lanes = buffer(size, S::ZERO)?;
        for (lane, x) in lanes.chunks_exact_mut(len.max(1)).enumerate() {
            for (target, value) in x.iter_mut().zip(rhs.iter()) {
                *target = value.lane(lane).unwrap_or(S::NAN);
            }
            self.substitute(lane, x, transpose);
        }
        for (k, value) in rhs.iter_mut().enumerate() {
            *value = B::from_fn(|lane| lanes[lane * len + k]);
        }
        Ok(())
    }
}

impl<S: RealScalar> sealed::Linear<S> for S {
    fn solve_factored(
        _: Token,
        factor: &Factor<S>,
        _: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        factor.solve_base(rhs, transpose)
    }
}

impl<S: RealScalar, const W: usize> sealed::Linear<S> for Lanes<S, W> {
    fn solve_factored(
        _: Token,
        factor: &Factor<S>,
        _: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        factor.solve_base(rhs, transpose)
    }
}

/// Subtracts `B·X` (or `Bᵀ·X`) from `rhs` for an `n × n` matrix `b`.
pub(crate) fn subtract_product<S: RealScalar, V: Real<S>>(
    rhs: &mut [V],
    b: &[V],
    x: &[V],
    n: usize,
    transpose: bool,
) {
    let columns = rhs.len() / n.max(1);
    for i in 0..n {
        for column in 0..columns {
            let mut value = rhs[i * columns + column];
            for k in 0..n {
                let entry = if transpose {
                    b[k * n + i]
                } else {
                    b[i * n + k]
                };
                value -= entry * x[k * columns + column];
            }
            rhs[i * columns + column] = value;
        }
    }
}

impl<S: RealScalar, V: Real<S>> sealed::Linear<S> for Dual<V> {
    fn solve_factored(
        token: Token,
        factor: &Factor<S>,
        matrix: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        let n = factor.order();
        let zero = V::zero();
        let mut values = buffer(matrix.len(), zero)?;
        let mut tangents = buffer(matrix.len(), zero)?;
        for ((v, d), entry) in values.iter_mut().zip(tangents.iter_mut()).zip(matrix) {
            *v = entry.value();
            *d = entry.derivative();
        }
        let mut x = buffer(rhs.len(), zero)?;
        let mut x_dot = buffer(rhs.len(), zero)?;
        for ((v, d), entry) in x.iter_mut().zip(x_dot.iter_mut()).zip(rhs.iter()) {
            *v = entry.value();
            *d = entry.derivative();
        }
        // x = A⁻¹b, then ẋ = A⁻¹(ḃ − Ȧ·x).
        V::solve_factored(token, factor, &values, &mut x, transpose)?;
        subtract_product(&mut x_dot, &tangents, &x, n, transpose);
        V::solve_factored(token, factor, &values, &mut x_dot, transpose)?;
        for (target, (&v, &d)) in rhs.iter_mut().zip(x.iter().zip(&x_dot)) {
            *target = Dual::new(v, d);
        }
        Ok(())
    }
}

/// Solves `A·X = B` for an `order × order` row-major matrix `A` and `columns`
/// right-hand sides `B` (row-major `order × columns`) into `out`.
///
/// The base matrix is factored once per lane (LU with partial pivoting) and the
/// factors are applied to every derivative component, so no elimination is
/// differentiated: for duals `ẋ = A⁻¹(ḃ − Ȧ·x)`, for jets
/// `xₖ = A₀⁻¹(bₖ − Σⱼ Aⱼ·xₖ₋ⱼ)`, and on a tape one call statement with pullback
/// `λ = A⁻ᵀ·x̄`, `b̄ += λ`, `Ā −= λ·xᵀ`. Cost, in floating-point operations: `⅔n³`
/// per lane for the factors, and per column and derivative component `2n²` for the
/// substitutions plus `2n²` for the right-hand side `ḃ − Ȧ·x` (a `Jet`
/// coefficient `k` forms `k` such products); the tape's pullback forms
/// `Ā −= λ·xᵀ`, another `2n²` per column. The call allocates its scratch
/// fallibly: a scalar copy of the matrix and the pivots, and for derivative number
/// types the base values and the derivative components of the solution.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length.
/// - `Singular` if a pivot of the base matrix is zero or not finite in some lane.
/// - `SizeOverflow` if a matrix or lane workspace size overflows `usize`.
/// - `AllocationFailure` if the scratch cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Real, solve_into};
/// // d/dt of A⁻¹b for A = [[2, 0], [0, 4]] + t·I and b = [2, 4] at t = 0.
/// let a = [Dual::new(2.0, 1.0), Dual::new(0.0, 0.0), Dual::new(0.0, 0.0), Dual::new(4.0, 1.0)];
/// let b = [Dual::new(2.0, 0.0), Dual::new(4.0, 0.0)];
/// let mut x = [Dual::constant(0.0); 2];
/// solve_into(&a, 2, &b, 1, &mut x)?;
/// assert_eq!((x[0].value(), x[1].value()), (1.0, 1.0));
/// assert_eq!((x[0].derivative(), x[1].derivative()), (-0.5, -0.25));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §3.4.
pub fn solve_into<S: RealScalar, R: Real<S>>(
    matrix: &[R],
    order: usize,
    rhs: &[R],
    columns: usize,
    out: &mut [R],
) -> Result<(), SciOxideError> {
    let square = order
        .checked_mul(order)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("matrix")))?;
    let block = order
        .checked_mul(columns)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("rhs")))?;
    crate::support::check_len(matrix.len(), square)?;
    crate::support::check_len(rhs.len(), block)?;
    crate::support::check_len(out.len(), block)?;
    <R as sealed::Linear<S>>::solve(Token(()), matrix, rhs, out, order)
}

impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Linear<S> for crate::MultiDual<V, N> {
    fn solve_factored(
        token: Token,
        factor: &Factor<S>,
        matrix: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        let n = factor.order();
        let zero = V::zero();
        let mut values = buffer(matrix.len(), zero)?;
        for (v, entry) in values.iter_mut().zip(matrix) {
            *v = entry.value();
        }
        let mut x = buffer(rhs.len(), zero)?;
        for (v, entry) in x.iter_mut().zip(rhs.iter()) {
            *v = entry.value();
        }
        V::solve_factored(token, factor, &values, &mut x, transpose)?;
        let tangent_len = scioxide_core::backend::checked_mul(
            N,
            rhs.len(),
            scioxide_core::quantity!("solve_scratch"),
        )?;
        let mut tangents = buffer(tangent_len, zero)?;
        let mut slope = buffer(matrix.len(), zero)?;
        for direction in 0..N {
            for (t, entry) in slope.iter_mut().zip(matrix) {
                *t = entry.gradient()[direction];
            }
            let x_dot = &mut tangents[direction * rhs.len()..(direction + 1) * rhs.len()];
            for (t, entry) in x_dot.iter_mut().zip(rhs.iter()) {
                *t = entry.gradient()[direction];
            }
            subtract_product(x_dot, &slope, &x, n, transpose);
            V::solve_factored(token, factor, &values, x_dot, transpose)?;
        }
        for (k, target) in rhs.iter_mut().enumerate() {
            let gradient = core::array::from_fn(|direction| tangents[direction * x.len() + k]);
            *target = crate::MultiDual::new(x[k], gradient);
        }
        Ok(())
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Linear<S> for crate::Jet<V, N> {
    fn solve_factored(
        token: Token,
        factor: &Factor<S>,
        matrix: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), SciOxideError> {
        // xₖ = A₀⁻¹·(bₖ − Σⱼ₌₁..ₖ Aⱼ·xₖ₋ⱼ).
        let n = factor.order();
        let len = rhs.len();
        let zero = V::zero();
        let matrix_len = scioxide_core::backend::checked_mul(
            N,
            matrix.len(),
            scioxide_core::quantity!("solve_scratch"),
        )?;
        let solution_len =
            scioxide_core::backend::checked_mul(N, len, scioxide_core::quantity!("solve_scratch"))?;
        let mut a = buffer(matrix_len, zero)?;
        for (index, entry) in matrix.iter().enumerate() {
            for (k, &c) in entry.coefficients().iter().enumerate() {
                a[k * matrix.len() + index] = c;
            }
        }
        let mut x = buffer(solution_len, zero)?;
        for (index, entry) in rhs.iter().enumerate() {
            for (k, &c) in entry.coefficients().iter().enumerate() {
                x[k * len + index] = c;
            }
        }
        let size = matrix.len();
        for k in 0..N {
            let (done, rest) = x.split_at_mut(k * len);
            let current = &mut rest[..len];
            for j in 1..=k {
                let a_j = &a[j * size..(j + 1) * size];
                subtract_product(
                    current,
                    a_j,
                    &done[(k - j) * len..(k - j + 1) * len],
                    n,
                    transpose,
                );
            }
            V::solve_factored(token, factor, &a[..size], current, transpose)?;
        }
        for (index, target) in rhs.iter_mut().enumerate() {
            *target = crate::Jet::new(core::array::from_fn(|k| x[k * len + index]));
        }
        Ok(())
    }
}
