//! Reused small-system factors for approximate coordinates, never topology.
use crate::tessellation::storage::Storage;
use crate::tessellation::{Result, unstable};
use scioxide_core::SciOxideError;
use scioxide_core::{ExecutionContext, RealScalar};
use scioxide_linalg::{LinalgWorkspace, LuDecomposition, Matrix, Vector};

/// Small factors live on the stack. Higher orders and arithmetic breakdown use
/// the ordinary equilibrated factorization, with all scratch it keeps charged.
/// Every system first tries the small path again: one breakdown never moves
/// later, well-conditioned systems to the general factorization.
pub(super) struct LinearFactor {
    small: SmallLu,
    // Inline optional metadata avoids allocating a box for the general path.
    // Its owned buffers are charged and allocated only on first use.
    general: Option<GeneralFactor>,
    // Which factor holds the current system.
    current: Current,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Current {
    Small,
    General,
}

struct GeneralFactor {
    factor: LuDecomposition<f64>,
    workspace: LinalgWorkspace<'static, f64>,
}

// The exact systems solved here are nonsingular; a rounded breakdown is a
// numerical failure of the approximate construction.
fn numerical(error: SciOxideError) -> SciOxideError {
    if error.kind() == scioxide_core::SciOxideErrorKind::Singular {
        unstable("rounded small system is numerically singular")
    } else {
        error
    }
}

impl LinearFactor {
    pub(super) fn new(d: usize, storage: &Storage) -> Result<Self> {
        Ok(Self {
            small: SmallLu {
                dimension: d,
                entries: [0.; 9],
                permutation: [0; 3],
                rows: [0; 3],
                columns: [0; 3],
            },
            general: if d > 3 {
                Some(Self::general(d, storage)?)
            } else {
                None
            },
            current: Current::General,
        })
    }

    fn general(d: usize, storage: &Storage) -> Result<GeneralFactor> {
        Ok(GeneralFactor {
            factor: LuDecomposition::empty(),
            workspace: storage.linalg(d)?,
        })
    }

    fn factor_general(&mut self, matrix: &Matrix<f64>, storage: &Storage) -> Result<()> {
        self.current = Current::General;
        let general = match &mut self.general {
            Some(general) => general,
            None => self.general.insert(Self::general(matrix.rows(), storage)?),
        };
        LuDecomposition::factor_into(
            matrix,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut general.factor,
            &mut general.workspace,
            ExecutionContext::sequential(),
        )
        .map_err(numerical)
    }

    pub(super) fn factor(&mut self, matrix: &Matrix<f64>, storage: &Storage) -> Result<()> {
        if matrix.rows() <= 3 && self.small.factor(matrix.as_slice()).is_ok() {
            self.current = Current::Small;
            return Ok(());
        }
        self.factor_general(matrix, storage)
    }

    pub(super) fn solve(
        &mut self,
        matrix: &Matrix<f64>,
        rhs: &Vector<f64>,
        output: &mut Vector<f64>,
        storage: &Storage,
    ) -> Result<()> {
        if self.current == Current::Small {
            if self
                .small
                .solve(rhs.as_slice(), output.as_mut_slice())
                .is_ok()
            {
                return Ok(());
            }
            self.factor_general(matrix, storage)?;
        }
        let general = self
            .general
            .as_mut()
            .ok_or_else(|| unstable("missing general factor"))?;
        let report = general
            .factor
            .solve_into(
                rhs,
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                output.view_mut(),
                &mut general.workspace,
                ExecutionContext::sequential(),
            )
            .map_err(numerical)?;
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(unstable("rounded small system is numerically singular"))
        }
    }

    pub(super) fn inverse(
        &mut self,
        matrix: &Matrix<f64>,
        output: &mut Matrix<f64>,
        storage: &Storage,
    ) -> Result<()> {
        if self.current == Current::Small {
            if self.small.inverse(output.as_mut_slice()).is_ok() {
                return Ok(());
            }
            self.factor_general(matrix, storage)?;
        }
        let general = self
            .general
            .as_mut()
            .ok_or_else(|| unstable("missing general factor"))?;
        general
            .factor
            .inverse_into(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut general.workspace,
                ExecutionContext::sequential(),
            )
            .map_err(numerical)
    }
}

/// Partial-pivoting LU of a binary row/column-equilibrated matrix of order 1–3.
/// Approximate solves are checked by the caller's conditioning/certificate rule.
struct SmallLu {
    dimension: usize,
    entries: [f64; 9],
    permutation: [usize; 3],
    rows: [i32; 3],
    columns: [i32; 3],
}

fn power(maximum: f64) -> i32 {
    match maximum.decompose() {
        scioxide_core::FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } => -(i32::from(exponent) + 63 - significand.leading_zeros() as i32),
        _ => 0,
    }
}

impl SmallLu {
    fn factor(&mut self, matrix: &[f64]) -> Result<()> {
        let n = self.dimension;
        self.entries[..n * n].copy_from_slice(matrix);
        for i in 0..n {
            self.permutation[i] = i;
            let row = &mut self.entries[i * n..(i + 1) * n];
            let maximum = row.iter().fold(0_f64, |m, x| m.max(x.abs()));
            self.rows[i] = power(maximum);
            for value in row {
                *value = value.scalbn(self.rows[i]);
            }
        }
        for j in 0..n {
            let maximum = (0..n).fold(0_f64, |m, i| m.max(self.entries[i * n + j].abs()));
            self.columns[j] = power(maximum);
            for i in 0..n {
                self.entries[i * n + j] = self.entries[i * n + j].scalbn(self.columns[j]);
            }
        }
        for j in 0..n {
            let mut pivot = j;
            for i in j + 1..n {
                if self.entries[i * n + j].abs() > self.entries[pivot * n + j].abs() {
                    pivot = i;
                }
            }
            if self.entries[pivot * n + j] == 0. {
                return Err(SciOxideError::singular(None));
            }
            if pivot != j {
                for k in 0..n {
                    self.entries.swap(j * n + k, pivot * n + k);
                }
                self.permutation.swap(j, pivot);
            }
            for i in j + 1..n {
                let multiplier = self.entries[i * n + j] / self.entries[j * n + j];
                self.entries[i * n + j] = multiplier;
                for k in j + 1..n {
                    self.entries[i * n + k] -= multiplier * self.entries[j * n + k];
                }
            }
        }
        if self.entries[..n * n].iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(unstable("small-system factorization overflow"))
        }
    }

    fn solve(&self, rhs: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.dimension;
        for i in 0..n {
            let p = self.permutation[i];
            let mut value = rhs[p].scalbn(self.rows[p]);
            for (j, &x) in output[..i].iter().enumerate() {
                value -= self.entries[i * n + j] * x;
            }
            output[i] = value;
        }
        for i in (0..n).rev() {
            let mut value = output[i];
            for (j, &x) in output.iter().enumerate().take(n).skip(i + 1) {
                value -= self.entries[i * n + j] * x;
            }
            output[i] = value / self.entries[i * n + i];
        }
        for (value, &exponent) in output.iter_mut().zip(&self.columns) {
            *value = value.scalbn(exponent);
        }
        if output.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(unstable("small-system solve overflow"))
        }
    }

    fn inverse(&self, output: &mut [f64]) -> Result<()> {
        let n = self.dimension;
        let mut rhs = [0.; 3];
        let mut solution = [0.; 3];
        for j in 0..n {
            rhs[..n].fill(0.);
            rhs[j] = 1.;
            self.solve(&rhs[..n], &mut solution[..n])?;
            for i in 0..n {
                output[i * n + j] = solution[i];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn pivoted_solves_and_inverses_preserve_scaled_systems() {
        // Leading zero pivots force row exchanges. Dyadic inputs and known
        // solutions make the test independent of another rounded solver.
        for n in 1..=4 {
            for scale in [1., 2_f64.powi(900), 2_f64.powi(-900)] {
                let storage = Storage::new(usize::MAX, 0).unwrap();
                let mut a = Matrix::zeros(n, n).unwrap();
                let dense: &[f64] = match n {
                    1 => &[2.],
                    2 => &[0., 2., 3., 4.],
                    3 => &[0., 2., 1., 2., -1., 1., 1., 1., 3.],
                    _ => &[
                        0., 2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2., 2., 0., 0., 0.,
                    ],
                };
                for (a, &value) in a.as_mut_slice().iter_mut().zip(dense) {
                    *a = value * scale;
                }
                let expected: Vec<_> = (0..n).map(|i| i as f64 - 0.5).collect();
                let rhs = Vector::from_vec(
                    dense
                        .chunks_exact(n)
                        .map(|row| {
                            row.iter().zip(&expected).map(|(&a, &x)| a * x).sum::<f64>() * scale
                        })
                        .collect(),
                );
                let mut output = Vector::zeros(n).unwrap();
                let mut factor = LinearFactor::new(n, &storage).unwrap();
                factor.factor(&a, &storage).unwrap();
                factor.solve(&a, &rhs, &mut output, &storage).unwrap();
                for (&actual, &expected) in output.as_slice().iter().zip(&expected) {
                    assert!((actual - expected).abs() <= 8. * f64::EPSILON * expected.abs());
                }
                let mut inverse = Matrix::zeros(n, n).unwrap();
                factor.inverse(&a, &mut inverse, &storage).unwrap();
                for i in 0..n {
                    for j in 0..n {
                        let product: f64 = (0..n)
                            .map(|k| a.as_slice()[i * n + k] * inverse.as_slice()[k * n + j])
                            .sum();
                        assert!((product - f64::from(i == j)).abs() <= 8. * f64::EPSILON);
                    }
                }
            }
        }
    }

    #[test]
    fn singular_and_unrepresentable_systems_fail_explicitly() {
        let storage = Storage::new(usize::MAX, 0).unwrap();
        let mut factor = LinearFactor::new(2, &storage).unwrap();
        let singular = Matrix::from_vec(2, 2, vec![1., 2., 2., 4.]).unwrap();
        // General factorization may retain a deficient factor; solving must
        // still reject the inconsistent equation rather than publish a NaN.
        match factor.factor(&singular, &storage) {
            Ok(()) => {
                let error = factor
                    .solve(
                        &singular,
                        &Vector::from_vec(vec![1., 0.]),
                        &mut Vector::zeros(2).unwrap(),
                        &storage,
                    )
                    .unwrap_err();
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::NumericalInstability
                );
            }
            Err(error) => {
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::NumericalInstability
                );
            }
        }
        assert!(factor.current == Current::General);
        // A breakdown does not move later well-conditioned systems off the
        // small path.
        let identity = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap();
        factor.factor(&identity, &storage).unwrap();
        assert!(factor.current == Current::Small);
        let mut factor = LinearFactor::new(1, &storage).unwrap();
        let a = Matrix::from_vec(1, 1, vec![f64::from_bits(1)]).unwrap();
        factor.factor(&a, &storage).unwrap();
        assert!(
            factor
                .solve(
                    &a,
                    &Vector::from_vec(vec![1.]),
                    &mut Vector::zeros(1).unwrap(),
                    &storage
                )
                .is_err()
        );
    }
}
