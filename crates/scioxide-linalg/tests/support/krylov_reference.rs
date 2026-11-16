//! Matrix-free copies of the independent SciPy oracle's deterministic problems.
use scioxide_core::{AdjointOperator, LinearOperator, SciOxideError};
use scioxide_linalg::{IterativeSolveReport, IterativeSolveResult, LinearSolveTermination, Vector};

pub struct Fixture {
    pub problem: &'static str,
    pub order: usize,
    pub method: &'static str,
    reference_iterations: usize,
}

pub fn fixtures() -> impl Iterator<Item = Fixture> {
    include_str!("../data/krylov_counts.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split('\t');
            Fixture {
                problem: fields.next().unwrap(),
                order: fields.next().unwrap().parse().unwrap(),
                method: fields.next().unwrap(),
                reference_iterations: fields.next().unwrap().parse().unwrap(),
            }
        })
}

impl Fixture {
    pub fn rhs(&self) -> Vector<f64> {
        Vector::from_vec(vec![1.; self.rows()])
    }

    pub fn check(&self, result: &IterativeSolveResult<f64>) {
        self.check_report(result.solution().as_slice(), result.report());
    }

    pub fn check_report(&self, solution: &[f64], report: IterativeSolveReport<f64>) {
        let mut ax = vec![0.; self.rows()];
        self.apply_into(solution, &mut ax, &mut []).unwrap();
        let residual = ax.iter().fold(0_f64, |norm, ax| norm.hypot(1. - ax));
        assert_eq!(
            report.termination(),
            LinearSolveTermination::Converged,
            "{}: {report:?}",
            self.method
        );
        assert!(
            residual <= 1e-8 * (self.rows() as f64).sqrt(),
            "{}: {residual}",
            self.method
        );
        assert!(
            report.iterations() * 5 <= self.reference_iterations * 6,
            "{}: {} iterations vs unrestarted reference {}",
            self.method,
            report.iterations(),
            self.reference_iterations
        );
        assert!(
            (report.diagnostic_residual_norm().unwrap() - residual).abs()
                <= 1e-12 * (self.rows() as f64).sqrt()
        );
        if let Some(diagnostics) = report.krylov() {
            assert_eq!(
                diagnostics.restarts(),
                0,
                "{}: {diagnostics:?}",
                self.method
            );
        }
    }

    fn stencil(&self, x: &[f64], y: &mut [f64], adjoint: bool) {
        if self.problem == "least_squares" {
            if adjoint {
                for (i, out) in y.iter_mut().enumerate() {
                    *out = (i + 1) as f64 / self.order as f64 * (x[i] + x[self.order + i]);
                }
            } else {
                for (i, out) in y.iter_mut().enumerate() {
                    let j = i % self.order;
                    *out = (j + 1) as f64 / self.order as f64 * x[j];
                }
            }
        } else if self.problem == "poisson" {
            let side = (self.order as f64).sqrt() as usize;
            for (i, out) in y.iter_mut().enumerate() {
                *out = 4. * x[i];
                if i % side > 0 {
                    *out -= x[i - 1];
                }
                if i % side + 1 < side {
                    *out -= x[i + 1];
                }
                if i >= side {
                    *out -= x[i - side];
                }
                if i + side < self.order {
                    *out -= x[i + side];
                }
            }
        } else {
            let (diagonal, left, right) = if self.problem == "indefinite" {
                (1.99, -1., -1.)
            } else if adjoint {
                (2.001, -0.99, -1.01)
            } else {
                (2.001, -1.01, -0.99)
            };
            for (i, out) in y.iter_mut().enumerate() {
                *out = diagonal * x[i];
                if i > 0 {
                    *out += left * x[i - 1];
                }
                if i + 1 < self.order {
                    *out += right * x[i + 1];
                }
            }
        }
    }
}

impl LinearOperator<f64> for Fixture {
    fn rows(&self) -> usize {
        self.order
            * if self.problem == "least_squares" {
                2
            } else {
                1
            }
    }
    fn cols(&self) -> usize {
        self.order
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.stencil(x, y, false);
        Ok(())
    }
}
impl AdjointOperator<f64> for Fixture {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.stencil(x, y, true);
        Ok(())
    }
}
