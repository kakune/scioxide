//! Compact frontal LU for bordered block-bidiagonal systems.
#![allow(clippy::needless_range_loop)]
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, standard, try_filled, try_reserve};
use scioxide_core::{AlgebraicScalar, ExecutionContext, NumericScalar, SciOxideError};

#[derive(Clone, Copy, Debug)]
struct Shape {
    n: usize,
    k: usize,
    nodes: usize,
    width: usize,
    border: usize,
    columns: usize,
    dimension: usize,
    panel: usize,
}
impl Shape {
    fn new(n: usize, k: usize, nodes: usize) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("bordered_bidiagonal_storage");
        if nodes < 2 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("shape"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "at least two node blocks required",
            ));
        }
        let border = checked_add(n, k, q)?;
        let width = checked_add(checked_mul(2, n, q)?, k, q)?;
        Ok(Self {
            n,
            k,
            nodes,
            width,
            border,
            columns: checked_add(width, n, q)?,
            dimension: checked_add(checked_mul(nodes, n, q)?, k, q)?,
            panel: checked_mul(checked_mul(2, n, q)?, width, q)?,
        })
    }
    fn entries(self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("bordered_bidiagonal_entries");
        checked_add(
            checked_mul(self.nodes - 1, self.panel, q)?,
            checked_mul(self.border, self.border, q)?,
            q,
        )
    }
    fn scalars(self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("bordered_bidiagonal_storage");
        checked_add(
            self.entries()?,
            checked_add(
                checked_mul(self.width, self.columns, q)?,
                checked_mul(self.border, self.width, q)?,
                q,
            )?,
            q,
        )
    }
}

/// Prepared partial-pivoted frontal LU with linear storage in the node count.
///
/// There are `nodes*n+k` unknowns: node-major state blocks followed by parameters.
/// The first `(nodes-1)*n` rows couple adjacent nodes and parameters; the final
/// `n+k` rows couple the two endpoints and parameters. Each input row stores
/// `2*n+k` values in that order, including explicit zeros.
///
/// Elimination pivots across the current interval and all retained border rows.
/// No individual interval block is required to be invertible. A zero state order is valid, including the empty system. The final border
/// is solved with dense partial pivoting. This backend mutates scratch on failure;
/// owners provide resource admission, accounting and transactional publication.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::backend::bordered_bidiagonal::BorderedBidiagonalFactor;
/// let mut factor = BorderedBidiagonalFactor::<f64>::new(1, 0, 2)?;
/// factor.factor(&[-1.0, 1.0, 1.0, 0.0], ExecutionContext::sequential())?;
/// let mut rhs = [1.0, 0.0];
/// factor.solve_in_place(&mut rhs, ExecutionContext::sequential())?;
/// assert_eq!(rhs, [0.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct BorderedBidiagonalFactor<T: NumericScalar> {
    shape: Shape,
    panels: Vec<T>,
    tail: Vec<T>,
    pivots: Vec<usize>,
    front: Vec<T>,
    carry: Vec<T>,
    valid: bool,
}
impl<T: NumericScalar> BorderedBidiagonalFactor<T> {
    /// Stored L/U slots, including structural zeros; checked without allocating.
    /// # Errors
    /// Returns invalid shape or size overflow.
    pub fn required_entries(n: usize, k: usize, nodes: usize) -> Result<usize, SciOxideError> {
        Shape::new(n, k, nodes)?.entries()
    }
    /// Logical factor and scratch payload bytes, excluding allocator bookkeeping.
    /// # Errors
    /// Returns invalid shape or size overflow.
    pub fn required_bytes(n: usize, k: usize, nodes: usize) -> Result<usize, SciOxideError> {
        let s = Shape::new(n, k, nodes)?;
        let q = scioxide_core::quantity!("bordered_bidiagonal_bytes");
        checked_add(
            checked_mul(s.scalars()?, core::mem::size_of::<T>(), q)?,
            checked_mul(s.dimension, core::mem::size_of::<usize>(), q)?,
            q,
        )
    }
    /// Allocate prepared factor and scratch storage; no matrix is evaluated.
    /// # Errors
    /// Returns invalid shape, overflow or allocation failure.
    pub fn new(n: usize, k: usize, nodes: usize) -> Result<Self, SciOxideError> {
        let s = Shape::new(n, k, nodes)?;
        let q = scioxide_core::quantity!("bordered_bidiagonal_storage");
        let _ = s.scalars()?;
        Ok(Self {
            shape: s,
            panels: try_filled((nodes - 1) * s.panel, T::ZERO, q)?,
            tail: try_filled(s.border * s.border, T::ZERO, q)?,
            pivots: try_filled(s.dimension, 0, q)?,
            front: try_filled(s.width * s.columns, T::ZERO, q)?,
            carry: try_filled(s.border * s.width, T::ZERO, q)?,
            valid: false,
        })
    }
    /// State block order and parameter count.
    pub const fn block_shape(&self) -> (usize, usize) {
        (self.shape.n, self.shape.k)
    }
    /// Retained payload, including unused capacity from earlier meshes.
    pub fn retained_bytes(&self) -> usize {
        (self.panels.capacity()
            + self.tail.capacity()
            + self.front.capacity()
            + self.carry.capacity())
            * core::mem::size_of::<T>()
            + self.pivots.capacity() * core::mem::size_of::<usize>()
    }
    /// Resize active node storage without shrinking capacity; invalidates the factor.
    /// # Errors
    /// Returns invalid shape, overflow or allocation failure; grown capacity may remain.
    pub fn resize(&mut self, nodes: usize) -> Result<(), SciOxideError> {
        let s = Shape::new(self.shape.n, self.shape.k, nodes)?;
        let _ = s.scalars()?;
        self.valid = false;
        resize(&mut self.panels, (nodes - 1) * s.panel, T::ZERO)?;
        resize(&mut self.pivots, s.dimension, 0)?;
        self.shape = s;
        Ok(())
    }
    /// Factor packed block rows in Standard arithmetic, checking each pivot and stop.
    /// # Errors
    /// Returns shape, non-finite input, singular pivot, arithmetic or execution errors.
    pub fn factor(&mut self, values: &[T], cx: ExecutionContext<'_>) -> Result<(), SciOxideError> {
        self.valid = false;
        let s = self.shape;
        let (n, w, b, c) = (s.n, s.width, s.border, s.columns);
        check_len(values.len(), s.dimension * w)?;
        finite(values, scioxide_core::NonFiniteOrigin::Input)?;
        self.carry.copy_from_slice(&values[(s.nodes - 1) * n * w..]);
        for i in 0..s.nodes - 1 {
            self.front.fill(T::ZERO);
            for row in 0..n {
                let source = &values[(i * n + row) * w..(i * n + row + 1) * w];
                let target = &mut self.front[row * c..(row + 1) * c];
                target[..2 * n].copy_from_slice(&source[..2 * n]);
                target[3 * n..].copy_from_slice(&source[2 * n..]);
            }
            for row in 0..b {
                let source = &self.carry[row * w..(row + 1) * w];
                let target = &mut self.front[(n + row) * c..(n + row + 1) * c];
                target[..n].copy_from_slice(&source[..n]);
                if i + 2 == s.nodes {
                    // The next node is the terminal border node. Use one column
                    // before elimination, avoiding artificial split-column growth.
                    target[n..2 * n].copy_from_slice(&source[n..2 * n]);
                    target[3 * n..].copy_from_slice(&source[2 * n..]);
                } else {
                    target[2 * n..].copy_from_slice(&source[n..]);
                }
            }
            if n == 2 && s.k == 0 {
                eliminate::<T, 4, 6, 2>(
                    &mut self.front,
                    w,
                    c,
                    n,
                    &mut self.pivots[i * n..(i + 1) * n],
                    i * n,
                    cx,
                )?;
            } else {
                eliminate::<T, 0, 0, 0>(
                    &mut self.front,
                    w,
                    c,
                    n,
                    &mut self.pivots[i * n..(i + 1) * n],
                    i * n,
                    cx,
                )?;
            }
            let panel = &mut self.panels[i * s.panel..(i + 1) * s.panel];
            for row in 0..w {
                panel[row * n..(row + 1) * n].copy_from_slice(&self.front[row * c..row * c + n]);
            }
            for row in 0..n {
                panel[w * n + row * w..w * n + (row + 1) * w]
                    .copy_from_slice(&self.front[row * c + n..(row + 1) * c]);
            }
            for row in 0..b {
                self.carry[row * w..(row + 1) * w]
                    .copy_from_slice(&self.front[(n + row) * c + n..(n + row + 1) * c]);
            }
        }
        for row in 0..b {
            for col in 0..b {
                self.tail[row * b + col] = if col < n {
                    self.carry[row * w + col] + self.carry[row * w + n + col]
                } else {
                    self.carry[row * w + n + col]
                };
            }
        }
        finite(&self.tail, scioxide_core::NonFiniteOrigin::Intermediate)?;
        if b == 2 {
            eliminate::<T, 2, 2, 2>(
                &mut self.tail,
                b,
                b,
                b,
                &mut self.pivots[(s.nodes - 1) * n..],
                (s.nodes - 1) * n,
                cx,
            )?;
        } else {
            eliminate::<T, 0, 0, 0>(
                &mut self.tail,
                b,
                b,
                b,
                &mut self.pivots[(s.nodes - 1) * n..],
                (s.nodes - 1) * n,
                cx,
            )?;
        }
        self.valid = true;
        Ok(())
    }
    /// Solve in place using retained scratch. Output is unspecified on failure.
    /// # Errors
    /// Returns missing factor, shape, non-finite input, arithmetic or execution errors.
    pub fn solve_in_place(
        &mut self,
        rhs: &mut [T],
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if !self.valid {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "a successful factorization is required before solving",
            ));
        }
        let s = self.shape;
        let (n, w, b) = (s.n, s.width, s.border);
        check_len(rhs.len(), s.dimension)?;
        finite(rhs, scioxide_core::NonFiniteOrigin::Input)?;
        self.carry[..b].copy_from_slice(&rhs[(s.nodes - 1) * n..]);
        for i in 0..s.nodes - 1 {
            cx.check()?;
            self.front[..n].copy_from_slice(&rhs[i * n..(i + 1) * n]);
            self.front[n..w].copy_from_slice(&self.carry[..b]);
            let panel = &self.panels[i * s.panel..(i + 1) * s.panel];
            for col in 0..n {
                self.front.swap(col, self.pivots[i * n + col]);
            }
            for col in 0..n {
                for row in col + 1..w {
                    self.front[row] =
                        update(self.front[row], panel[row * n + col], self.front[col])?;
                }
            }
            rhs[i * n..(i + 1) * n].copy_from_slice(&self.front[..n]);
            self.carry[..b].copy_from_slice(&self.front[n..w]);
        }
        let tail = &mut rhs[(s.nodes - 1) * n..];
        tail.copy_from_slice(&self.carry[..b]);
        for col in 0..b {
            cx.check()?;
            tail.swap(col, self.pivots[(s.nodes - 1) * n + col]);
        }
        for col in 0..b {
            for row in col + 1..b {
                tail[row] = update(tail[row], self.tail[row * b + col], tail[col])?;
            }
        }
        for row in (0..b).rev() {
            tail[row] = standard::substitute_row(
                tail[row],
                self.tail[row * b + row],
                (row + 1..b).map(|col| (self.tail[row * b + col], tail[col])),
            )?;
        }
        for i in (0..s.nodes - 1).rev() {
            cx.check()?;
            let panel = &self.panels[i * s.panel..(i + 1) * s.panel];
            for row in (0..n).rev() {
                rhs[i * n + row] = standard::substitute_row(
                    rhs[i * n + row],
                    panel[row * n + row],
                    (row + 1..n + w).map(|col| {
                        if col < n {
                            (panel[row * n + col], rhs[i * n + col])
                        } else {
                            let t = col - n;
                            let global = if t < n {
                                (i + 1) * n + t
                            } else if t < 2 * n {
                                (s.nodes - 1) * n + t - n
                            } else {
                                s.nodes * n + t - 2 * n
                            };
                            (panel[w * n + row * w + t], rhs[global])
                        }
                    }),
                )?;
            }
        }
        finite(rhs, scioxide_core::NonFiniteOrigin::Intermediate)
    }
}
fn resize<V: Copy>(v: &mut Vec<V>, len: usize, value: V) -> Result<(), SciOxideError> {
    try_reserve(
        v,
        len.saturating_sub(v.len()),
        scioxide_core::quantity!("bordered_bidiagonal_storage"),
    )?;
    v.resize(len, value);
    Ok(())
}
fn check_len(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch([expected], [actual]))
    }
}
fn finite<T: NumericScalar>(
    v: &[T],
    origin: scioxide_core::NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if v.iter().all(|x| x.has_finite_components()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "non-finite bordered block values",
        ))
    }
}
#[inline(always)]
fn update<T: NumericScalar>(a: T, b: T, c: T) -> Result<T, SciOxideError> {
    let p = b * c;
    let out = a - p;
    if !standard::product_needs_retry(b, c, p) && out.has_finite_components() {
        Ok(out)
    } else {
        standard::sum_products(
            [(a, T::ONE), (-b, c)].into_iter(),
            scioxide_core::Accuracy::Standard,
        )
    }
}
#[inline(always)]
fn eliminate<T: NumericScalar, const ROWS: usize, const COLUMNS: usize, const PIVOTS: usize>(
    matrix: &mut [T],
    rows: usize,
    columns: usize,
    pivots: usize,
    permutation: &mut [usize],
    offset: usize,
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let rows = if ROWS == 0 { rows } else { ROWS };
    let columns = if COLUMNS == 0 { columns } else { COLUMNS };
    let pivots = if PIVOTS == 0 { pivots } else { PIVOTS };
    for col in 0..pivots {
        cx.check()?;
        let mut chosen = col;
        let mut largest = standard::magnitude(matrix[col * columns + col]);
        for row in col + 1..rows {
            let size = standard::magnitude(matrix[row * columns + col]);
            if size > largest {
                largest = size;
                chosen = row;
            }
        }
        if largest == T::Real::ZERO {
            return Err(SciOxideError::singular(Some(offset + col)));
        }
        permutation[col] = chosen;
        if chosen != col {
            for j in 0..columns {
                matrix.swap(col * columns + j, chosen * columns + j);
            }
        }
        let diagonal = matrix[col * columns + col];
        for row in col + 1..rows {
            let multiplier = standard::factor_quotient(matrix[row * columns + col], diagonal)?;
            matrix[row * columns + col] = multiplier;
            if multiplier != T::ZERO {
                for j in col + 1..columns {
                    matrix[row * columns + j] = update(
                        matrix[row * columns + j],
                        multiplier,
                        matrix[col * columns + j],
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{Complex32, Complex64, RealScalar};
    fn v<T: NumericScalar>(x: f64) -> T {
        T::from_real(T::Real::try_from_f64_lossy(x).unwrap())
    }
    fn qualify<T: NumericScalar>() {
        let cx = ExecutionContext::sequential();
        for n in [1, 2, 3] {
            for k in [0, 1, 2] {
                for nodes in [2, 3, 8] {
                    for one_sided in [0, 1, 2] {
                        let w = 2 * n + k;
                        let dim = nodes * n + k;
                        let mut values = vec![T::ZERO; dim * w];
                        for row in 0..(nodes - 1) * n {
                            let r = row % n;
                            for col in 0..n {
                                values[row * w + col] = if one_sided == 1 {
                                    T::ZERO
                                } else if r == col {
                                    v::<T>(-3.)
                                } else {
                                    v::<T>(0.1)
                                };
                                values[row * w + n + col] = if one_sided == 2 {
                                    T::ZERO
                                } else if r == col {
                                    v::<T>(2.)
                                } else {
                                    v::<T>(0.05)
                                };
                            }
                            for p in 0..k {
                                values[row * w + 2 * n + p] = v::<T>(0.01);
                            }
                        }
                        for row in 0..n + k {
                            let offset = ((nodes - 1) * n + row) * w;
                            if row < n {
                                values[offset + row] = T::ONE;
                                values[offset + n + row] = T::ONE;
                            } else {
                                values[offset + 2 * n + row - n] = v::<T>(3.);
                            }
                        }
                        let expected: Vec<T> = (0..dim)
                            .map(|i| {
                                T::from_components(
                                    (v::<T>(1.)
                                        + T::from_real(T::Real::try_from_usize_exact(i).unwrap())
                                            * v::<T>(0.02))
                                    .real_component(),
                                    v::<T>(0.1).real_component(),
                                )
                            })
                            .collect();
                        let mut rhs = vec![T::ZERO; dim];
                        for row in 0..dim {
                            let a = if row < (nodes - 1) * n {
                                row / n * n
                            } else {
                                0
                            };
                            let b = if row < (nodes - 1) * n {
                                a + n
                            } else {
                                (nodes - 1) * n
                            };
                            for (col, global) in
                                (a..a + n).chain(b..b + n).chain(nodes * n..dim).enumerate()
                            {
                                rhs[row] = rhs[row] + values[row * w + col] * expected[global];
                            }
                        }
                        let mut factor = BorderedBidiagonalFactor::new(n, k, nodes).unwrap();
                        factor.factor(&values, cx).unwrap();
                        factor.solve_in_place(&mut rhs, cx).unwrap();
                        for (&actual, &wanted) in rhs.iter().zip(&expected) {
                            assert!(
                                (actual - wanted).abs()
                                    <= v::<T>(512.).abs()
                                        * T::Real::EPSILON
                                        * (T::Real::ONE + wanted.abs()),
                                "n={n} k={k} nodes={nodes} one_sided={one_sided}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn frontal_pivots_cover_border_rows_parameters_and_complex_states() {
        qualify::<f32>();
        qualify::<f64>();
        qualify::<Complex32>();
        qualify::<Complex64>();
    }
    #[test]
    fn empty_shapes_failures_and_retained_capacity_are_explicit() {
        let cx = ExecutionContext::sequential();
        let mut empty = BorderedBidiagonalFactor::<f64>::new(0, 0, 2).unwrap();
        empty.factor(&[], cx).unwrap();
        empty.solve_in_place(&mut [], cx).unwrap();
        let mut factor = BorderedBidiagonalFactor::<f64>::new(1, 0, 2).unwrap();
        assert!(factor.solve_in_place(&mut [0.; 2], cx).is_err());
        assert!(factor.factor(&[0.; 4], cx).is_err());
        assert!(factor.factor(&[f64::NAN; 4], cx).is_err());
        assert!(factor.factor(&[1.; 3], cx).is_err());
        factor.resize(8).unwrap();
        let retained = factor.retained_bytes();
        factor.resize(2).unwrap();
        assert_eq!(factor.retained_bytes(), retained);
        factor.factor(&[-1., 1., 1., 0.], cx).unwrap();
        let mut rhs = [1., 0.];
        factor.solve_in_place(&mut rhs, cx).unwrap();
        assert_eq!(rhs, [0., 1.]);
    }
    #[test]
    fn wide_and_tiny_frames_preserve_substitution() {
        let cx = ExecutionContext::sequential();
        for scale in [1e-200, 1., 1e200] {
            let mut factor = BorderedBidiagonalFactor::<f64>::new(1, 0, 2).unwrap();
            factor.factor(&[-scale, scale, scale, 0.], cx).unwrap();
            let mut rhs = [scale, 0.];
            factor.solve_in_place(&mut rhs, cx).unwrap();
            assert!(rhs[0].abs() < 1e-14 && (rhs[1] - 1.).abs() < 1e-14);
        }
        let mut parameter = BorderedBidiagonalFactor::<f64>::new(0, 1, 2).unwrap();
        parameter.factor(&[2.], cx).unwrap();
        let mut rhs = [4.];
        parameter.solve_in_place(&mut rhs, cx).unwrap();
        assert_eq!(rhs, [2.]);
        assert!(BorderedBidiagonalFactor::<f64>::required_bytes(1, usize::MAX, 2).is_err());
    }
    #[test]
    fn coincident_terminal_columns_do_not_create_artificial_overflow() {
        let m = f64::MAX * 0.75;
        let values = [
            1.,
            0.,
            m,
            0.,
            0.,
            0.,
            0.,
            1.,
            1.,
            1.,
            0.75 * m,
            0.,
            -1.,
            1.,
            -0.75 * m,
            1.,
        ];
        let mut factor = BorderedBidiagonalFactor::<f64>::new(2, 0, 2).unwrap();
        factor
            .factor(&values, ExecutionContext::sequential())
            .unwrap();
        let mut rhs = [2., 1., 2.75, 0.25];
        factor
            .solve_in_place(&mut rhs, ExecutionContext::sequential())
            .unwrap();
        assert!((rhs[0] - 1.).abs() < 1e-14 && (rhs[1] - 1.).abs() < 1e-14);
        assert!((rhs[2] * m - 1.).abs() < 1e-14 && (rhs[3] - 1.).abs() < 1e-14);
    }
}
