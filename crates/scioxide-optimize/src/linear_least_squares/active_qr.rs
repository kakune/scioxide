//! Lawson–Hanson's transformed full matrix: no explicit Q and no normal matrix.
//! The passive columns occupy a triangular prefix. Column insertion uses one
//! Householder reflection, deletion chases the resulting bulge with rotations.
use super::*;
use scioxide_core::backend::{standard, try_filled};

pub(super) struct ActiveQr<S: RealScalar + NumericScalar<Real = S>> {
    matrix: Matrix<S>,
    permutation: Vec<usize>,
    reflector: Vec<S>,
    products: Vec<S>,
    count: usize,
    exponent: i32,
    usable: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>> ActiveQr<S> {
    /// Retry a stalled updated solve against the original residual, once.
    pub(super) fn invalidate(&mut self) -> bool {
        core::mem::replace(&mut self.usable, false)
    }
    pub(super) fn new(m: usize, n: usize) -> Result<Self, SciOxideError> {
        let width = n
            .checked_add(1)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("active_qr")))?;
        Ok(Self {
            matrix: Matrix::zeros(m, width)?,
            permutation: try_filled(n, 0, scioxide_core::quantity!("active_qr.permutation"))?,
            reflector: try_filled(m, S::ZERO, scioxide_core::quantity!("active_qr.reflector"))?,
            products: try_filled(
                width,
                S::ZERO,
                scioxide_core::quantity!("active_qr.products"),
            )?,
            count: 0,
            exponent: 0,
            usable: true,
        })
    }
    pub(super) fn reset(&mut self, a: MatrixView<'_, S>, b: &[S]) -> Result<(), SciOxideError> {
        self.count = 0;
        self.usable = true;
        let n = a.cols();
        let mut maximum = b.iter().fold(S::ZERO, |v, &x| v.max(NumericScalar::abs(x)));
        for i in 0..a.rows() {
            for j in 0..n {
                maximum = maximum.max(NumericScalar::abs(a.get(i, j)?));
            }
        }
        self.exponent = standard::exponent(maximum)?;
        for (j, index) in self.permutation.iter_mut().enumerate() {
            *index = j;
        }
        for (i, row) in self
            .matrix
            .as_mut_slice()
            .chunks_exact_mut(n + 1)
            .enumerate()
        {
            for (j, value) in row.iter_mut().enumerate() {
                let original = if j == n { b[i] } else { a.get(i, j)? };
                *value = original.scalbn(-self.exponent);
                // Exceptional dynamic range retains the original QR/SVD path.
                if original != S::ZERO && NumericScalar::abs(*value) < S::MIN_POSITIVE.sqrt() {
                    self.usable = false;
                }
            }
        }
        Ok(())
    }
    fn swap(&mut self, first: usize, second: usize) {
        if first != second {
            let stride = self.matrix.cols();
            for row in self.matrix.as_mut_slice().chunks_exact_mut(stride) {
                row.swap(first, second);
            }
            self.permutation.swap(first, second);
        }
    }
    fn insert(&mut self, position: usize) -> Result<(), SciOxideError> {
        let k = self.count;
        self.swap(k, position);
        let (m, stride) = (self.matrix.rows(), self.matrix.cols());
        let data = self.matrix.as_mut_slice();
        let v = &mut self.reflector[k..m];
        let mut maximum = S::ZERO;
        for (value, row) in v.iter_mut().zip(data[k * stride..].chunks_exact(stride)) {
            *value = row[k];
            maximum = maximum.max(NumericScalar::abs(*value));
        }
        if maximum != S::ZERO {
            let exponent = standard::exponent(maximum)?;
            for value in v.iter_mut() {
                *value = value.scalbn(-exponent);
            }
            let norm = standard::dot(v.iter().map(|&x| (x, x))).sqrt();
            let beta = -norm.copysign(v[0]);
            let denominator = v[0] - beta;
            let tau = -denominator / beta;
            v[0] = S::ONE;
            for value in &mut v[1..] {
                *value /= denominator;
            }
            let products = &mut self.products[k + 1..stride];
            products.fill(S::ZERO);
            for (&v, row) in v.iter().zip(data[k * stride..].chunks_exact(stride)) {
                for (dot, &x) in products.iter_mut().zip(&row[k + 1..]) {
                    *dot += v * x;
                }
            }
            for dot in products.iter_mut() {
                *dot *= tau;
            }
            for (&v, row) in v.iter().zip(data[k * stride..].chunks_exact_mut(stride)) {
                for (x, &dot) in row[k + 1..].iter_mut().zip(products.iter()) {
                    *x -= v * dot;
                }
            }
            data[k * stride + k] = beta.scalbn(exponent);
        }
        for row in data[(k + 1) * stride..].chunks_exact_mut(stride) {
            row[k] = S::ZERO;
        }
        self.count += 1;
        Ok(())
    }
    fn delete(&mut self, position: usize) -> Result<(), SciOxideError> {
        let stride = self.matrix.cols();
        // Keep the surviving triangular columns in order; the removed column
        // remains transformed alongside every other inactive column.
        for j in position..self.count - 1 {
            self.swap(j, j + 1);
        }
        self.count -= 1;
        let data = self.matrix.as_mut_slice();
        for j in position..self.count {
            let a = data[j * stride + j];
            let b = data[(j + 1) * stride + j];
            let maximum = NumericScalar::abs(a).max(NumericScalar::abs(b));
            if maximum != S::ZERO {
                let exponent = standard::exponent(maximum)?;
                let a = a.scalbn(-exponent);
                let b = b.scalbn(-exponent);
                let norm = a.hypot(b);
                let c = a / norm;
                let s = b / norm;
                let (top, bottom) = data.split_at_mut((j + 1) * stride);
                for (x, y) in top[j * stride..].iter_mut().zip(&mut bottom[..stride]) {
                    let (a, b) = (*x, *y);
                    *x = c * a + s * b;
                    *y = c * b - s * a;
                }
            }
            data[(j + 1) * stride + j] = S::ZERO;
        }
        Ok(())
    }
    /// Synchronize the passive prefix and return its triangular factor in
    /// original units. Inconvenient geometry/range uses the prepared fallback.
    pub(super) fn synchronize(
        &mut self,
        passive: &[bool],
        r: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        if !self.usable || passive.iter().filter(|&&p| p).count() > self.matrix.rows() {
            return Ok(false);
        }
        for position in (0..self.count).rev() {
            if !passive[self.permutation[position]] {
                self.delete(position)?;
            }
        }
        for position in self.count..self.permutation.len() {
            if passive[self.permutation[position]] {
                self.insert(position)?;
            }
        }
        let (m, k, stride) = (self.matrix.rows(), self.count, self.matrix.cols());
        r.try_resize_flat(m, k)?;
        r.as_mut_slice().fill(S::ZERO);
        let multiplier = S::ONE.scalbn(self.exponent);
        for i in 0..k {
            for j in i..k {
                let value = self.matrix.as_slice()[i * stride + j];
                let restored = value * multiplier;
                r.as_mut_slice()[i * k + j] = if multiplier.is_finite()
                    && restored.is_finite()
                    && (value == S::ZERO || NumericScalar::abs(restored) >= S::MIN_POSITIVE)
                {
                    restored
                } else {
                    standard::restore(value, self.exponent)?
                };
            }
        }
        Ok(true)
    }
    pub(super) fn step(&mut self, x: &[S], step: &mut [S]) -> Result<(), SciOxideError> {
        let (k, stride) = (self.count, self.matrix.cols());
        let data = self.matrix.as_slice();
        for i in (0..k).rev() {
            let row = &data[i * stride..(i + 1) * stride];
            let products = self.permutation.iter().enumerate().map(|(j, &original)| {
                let value = if j > i && j < k {
                    self.products[j]
                } else {
                    x[original]
                };
                (if j == i { S::ZERO } else { row[j] }, value)
            });
            let rhs = row[stride - 1];
            let residual = rhs - standard::dot(products.clone());
            let target = standard::finish_substitution(rhs, row[i], residual, products, None)?;
            self.products[i] = target;
        }
        for (j, &original) in self.permutation[..k].iter().enumerate() {
            step[original] = super::engine::add(self.products[j], -S::ONE, x[original])?;
        }
        finite(step)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn insertion_deletion_and_reentry_solve_the_original_free_system() {
        let a = Matrix::from_vec(
            5,
            4,
            alloc::vec![
                2., 1., 0., 1., 1., 3., 1., 0., 0., 1., 4., 1., 1., 0., 1., 5., 2., -1., 1., 2.,
            ],
        )
        .unwrap();
        let b = [1., -2., 3., 4., -1.];
        let mut qr = ActiveQr::new(5, 4).unwrap();
        let mut r = Matrix::zeros(5, 4).unwrap();
        qr.reset(a.view(), &b).unwrap();
        for passive in [
            [false, false, true, false],
            [true, false, true, false],
            [true, true, true, true],
            [true, false, true, true],
            [false, false, true, true],
            [false, true, true, true],
            [false; 4],
            [true; 4],
        ] {
            assert!(qr.synchronize(&passive, &mut r).unwrap());
            let x = [0.5, -0.5, 0.25, -0.25];
            let mut step = [0.; 4];
            qr.step(&x, &mut step).unwrap();
            for j in 0..4 {
                if !passive[j] {
                    assert_eq!(step[j], 0.);
                    continue;
                }
                let gradient: f64 = (0..5)
                    .map(|i| {
                        let residual = (0..4)
                            .map(|k| a.view().get(i, k).unwrap() * (x[k] + step[k]))
                            .sum::<f64>()
                            - b[i];
                        a.view().get(i, j).unwrap() * residual
                    })
                    .sum();
                assert!(gradient.abs() < 1e-12, "{passive:?}: {gradient}");
            }
        }
    }
}
