//! Vector-valued, matrix-valued and random-group probability laws.
use super::rows::{RowLaw, Scratch, vector_sampling};
use crate::law::{constant as number, finite, invalid, logadd, positive};
use crate::{
    BitGenerator, Density, Distribution, Gamma, LogDensity, LogMass, Mass, Mean, Normal, Variance,
};
use alloc::vec::Vec;
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, MatrixView};

/// Radial family; every stored scalar degree is finite and positive.
#[derive(Debug, Clone, Copy)]
pub(super) enum DegreesOfFreedom {
    StudentT(f64),
    Gaussian,
}

mod counts;
mod elliptical;
mod groups;
mod matrix;
mod multinomial;
mod normal;
mod normal_inverse_gamma;
mod probability;
mod simplex;
pub use counts::{DirichletMultinomial, MultivariateHypergeometric, RandomTable};
pub use elliptical::MultivariateStudentT;
pub use groups::{OrthogonalGroup, RandomCorrelation, SpecialOrthogonalGroup, UnitaryGroup};
pub use matrix::{InverseWishart, MatrixNormal, MatrixT, Wishart};
pub use multinomial::Multinomial;
pub use normal::{MultivariateNormal, MultivariateNormalOptions};
pub use normal_inverse_gamma::NormalInverseGamma;
pub use probability::{
    MultivariateProbabilityOptions, MultivariateProbabilityPlan, MultivariateProbabilityReport,
    MultivariateProbabilityTermination, MultivariateProbabilityWorkspace,
};
pub use simplex::Dirichlet;

impl<S: RealScalar> super::LawOptions<S> {
    pub(super) fn steps(self, steps: usize) -> Result<(), SciOxideError> {
        if steps > self.work.get() {
            return Err(super::count_math::exhausted());
        }
        Ok(())
    }
}
fn size(n: usize, expected: usize) -> Result<(), SciOxideError> {
    if n != expected {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(expected),
            scioxide_core::Dimensions::one(n),
        ));
    }
    Ok(())
}
fn total(counts: &[usize], argument: scioxide_core::Argument) -> Result<usize, SciOxideError> {
    counts.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| invalid(argument, "count total overflows"))
    })
}
fn real_count(n: usize) -> Result<f64, SciOxideError> {
    super::count_math::analytic_count(n)
}
fn values<S: RealScalar>(input: &[S]) -> Result<Vec<f64>, SciOxideError> {
    let mut out = scioxide_core::backend::try_with_capacity(
        input.len(),
        scioxide_core::quantity!("multivariate_parameters"),
    )?;
    for &x in input {
        crate::law::argument(x, "x")?;
        out.push(x.into_f64());
    }
    Ok(out)
}
pub(super) fn converted<S: RealScalar>(
    input: impl IntoIterator<Item = f64>,
) -> Result<Vec<S>, SciOxideError> {
    let iter = input.into_iter();
    let mut out = scioxide_core::backend::try_with_capacity(
        iter.size_hint().0,
        scioxide_core::quantity!("multivariate_result"),
    )?;
    for x in iter {
        scioxide_core::backend::try_reserve(
            &mut out,
            1,
            scioxide_core::quantity!("multivariate_result"),
        )?;
        out.push(finite(number(x)?)?);
    }
    Ok(out)
}
fn covariance<S: RealScalar + NumericScalar<Real = S>>(
    p: &[f64],
    scale: f64,
) -> Result<Matrix<S>, SciOxideError> {
    Matrix::from_vec(
        p.len(),
        p.len(),
        converted(p.iter().enumerate().flat_map(|(i, &pi)| {
            p.iter()
                .enumerate()
                .map(move |(j, &pj)| scale * pi * (if i == j { 1.0 - pi } else { -pj }))
        }))?,
    )
}

// A conservative cancellation admission check, not an accuracy certificate.
// Special-function error contracts still apply to the individual terms.
fn conditioned_number<S: RealScalar>(value: f64, magnitude: f64) -> Result<S, SciOxideError> {
    let tolerance = if S::EPSILON.into_f64() > f64::EPSILON {
        2e-6
    } else {
        5e-12
    };
    if !value.is_finite() || 8.0 * f64::EPSILON * magnitude > tolerance * (1.0 + value.abs()) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "cancellation exceeds the supported scalar accuracy",
        ));
    }
    number(value)
}
// Sum log-density terms and reject the result if the terms' total magnitude
// implies more cancellation than the target scalar can tolerate.
fn log_terms<S: RealScalar>(terms: &[f64]) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::RangeAccumulator::new();
    let mut magnitude = 0.0;
    for &term in terms {
        sum.add(term)?;
        magnitude += term.abs();
    }
    conditioned_number(sum.finish()?, magnitude)
}

// Draws charge the caller's per-draw word allowance.
fn normal_matrix<R: crate::Rng<f64>>(
    rng: &mut R,
    rows: usize,
    cols: usize,
) -> Result<Matrix<f64>, SciOxideError> {
    let mut out = Matrix::zeros(rows, cols)?;
    let normal = Normal::<f64>::standard();
    for v in out.as_mut_slice() {
        *v = normal.sample(rng)?;
    }
    Ok(out)
}

#[cfg(test)]
mod contracts;
#[cfg(test)]
mod reference;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Multinomial, QmcMultinomial, SeedableRng, SplitMix64};
    fn close(x: f64, y: f64) {
        assert!((x - y).abs() < 2e-11 * (1.0 + y.abs()), "{x} != {y}");
    }
    #[test]
    fn simplex_and_count_normalization() {
        let d = Dirichlet::with_options(
            &[1.0, 1.0, 1.0],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        close(d.pdf(&[0.2, 0.3, 0.5]).unwrap(), 2.0);
        let dm = DirichletMultinomial::with_options(
            4,
            &[1.0, 1.0, 1.0],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let mn = Multinomial::with_options(
            4,
            &[0.2, 0.3, 0.5],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let hg = MultivariateHypergeometric::<f64>::with_options(
            &[2, 3, 4],
            4,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
        for i in 0..=4 {
            for j in 0..=4 - i {
                let k = [i, j, 4 - i - j];
                close(dm.pmf(&k).unwrap(), 1.0 / 15.0);
                a += dm.pmf(&k).unwrap();
                b += mn.pmf(&k).unwrap();
                c += hg.pmf(&k).unwrap();
            }
        }
        close(a, 1.0);
        close(b, 1.0);
        close(c, 1.0);
    }
    #[test]
    fn table_exact_enumeration() {
        let t = RandomTable::<f64>::with_options(
            &[2, 2],
            &[2, 2],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        for (k, p) in [(0, 1.0 / 6.0), (1, 2.0 / 3.0), (2, 1.0 / 6.0)] {
            close(t.pmf(&[k, 2 - k, 2 - k, k]).unwrap(), p);
        }
        assert!(t.contains_support(&[2, 0, 0, 2]).unwrap());
        assert!(!t.contains_support(&[2, 0, 1, 1]).unwrap());
        let mut rng = SplitMix64::from_seed(542);
        for _ in 0..1000 {
            assert!(t.contains_support(&t.sample(&mut rng).unwrap()).unwrap());
        }
    }
    #[test]
    fn small_concentration_sampling_and_f32() {
        let mut rng = SplitMix64::from_seed(542);
        for alpha in [[1e-4, 1e-4, 1e-4], [1.0, 2.0, 3.0]] {
            let d = Dirichlet::<f32>::with_options(
                &alpha,
                crate::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap();
            for _ in 0..100 {
                let x = d.sample(&mut rng).unwrap();
                assert!(d.contains_support(&x).unwrap());
            }
        }
        let dm = DirichletMultinomial::<f32>::with_options(
            20,
            &[0.1, 2.0, 0.5],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        for _ in 0..100 {
            assert!(dm.contains_support(&dm.sample(&mut rng).unwrap()).unwrap());
        }
    }
    #[test]
    fn impossible_categories_and_transport_domain() {
        let m = Multinomial::with_options(
            10,
            &[1.0, 1e-100],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert!(m.log_pmf(&[9, 1]).unwrap().is_finite());
        assert!(QmcMultinomial::new(&m).is_err());
        let m = Multinomial::with_options(
            10,
            &[0.0, 1.0, 0.0],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert_eq!(m.sample(&mut SplitMix64::from_seed(1)).unwrap(), [0, 10, 0]);
        assert_eq!(m.pmf(&[1, 9, 0]).unwrap(), 0.0);
        assert!(
            RandomTable::<f32>::with_options(
                &[1],
                &[2],
                crate::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
            )
            .is_err()
        );
        assert!(
            Dirichlet::with_options(
                &[0.0, 1.0],
                crate::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
            )
            .is_err()
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod matrix_tests {
    use super::*;
    use crate::{ChiSquared, LogDensity, SeedableRng, SplitMix64, StudentT};
    fn close(x: f64, y: f64, tol: f64) {
        assert!((x - y).abs() < tol * (1.0 + y.abs()), "{x} != {y}");
    }
    #[test]
    fn scalar_reductions() {
        let mean = Matrix::from_vec(1, 1, vec![0.0]).unwrap();
        let scale = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        let x = Matrix::from_vec(1, 1, vec![1.5]).unwrap();
        let wish = Wishart::with_options(
            5.0,
            &scale,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        close(
            wish.log_pdf(x.view()).unwrap(),
            ChiSquared::new(5.0).unwrap().log_pdf(1.5).unwrap(),
            1e-12,
        );
        let inv = InverseWishart::with_options(
            5.0,
            &scale,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let expected =
            Gamma::new(2.5, 2.0).unwrap().log_pdf(1.0 / 1.5).unwrap() - 2.0 * 1.5_f64.ln();
        close(inv.log_pdf(x.view()).unwrap(), expected, 1e-12);
        let mn = MatrixNormal::with_options(
            &mean,
            &scale,
            &scale,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        close(
            mn.log_pdf(x.view()).unwrap(),
            Normal::new(0.0, 1.0).unwrap().log_pdf(1.5).unwrap(),
            1e-12,
        );
        let mt = MatrixT::with_options(
            &mean,
            &scale,
            &scale,
            5.0,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        close(
            mt.log_pdf(x.view()).unwrap(),
            StudentT::new(5.0)
                .unwrap()
                .log_pdf(1.5 * 5.0_f64.sqrt())
                .unwrap()
                + 0.5 * 5.0_f64.ln(),
            1e-12,
        );
        let vt = MultivariateStudentT::with_options(
            &[0.0],
            &scale,
            5.0,
            Default::default(),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        close(
            vt.log_pdf(&[1.5]).unwrap(),
            StudentT::new(5.0).unwrap().log_pdf(1.5).unwrap(),
            1e-12,
        );
    }
    #[test]
    fn matrix_samples_and_moments() {
        let scale = Matrix::from_vec(2, 2, vec![1.0, 0.3, 0.3, 2.0]).unwrap();
        let wish = Wishart::with_options(
            8.0,
            &scale,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let inv = InverseWishart::with_options(
            8.0,
            &scale,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let mut rng = SplitMix64::from_seed(542);
        let (mut wm, mut im) = (0.0, 0.0);
        for _ in 0..10000 {
            let w = wish.sample(&mut rng).unwrap();
            let i = inv.sample(&mut rng).unwrap();
            wm += w[0] / 10000.0;
            im += i[0] / 10000.0;
            assert!(
                wish.contains_support(MatrixView::from_slice(2, 2, &w).unwrap())
                    .unwrap()
            );
        }
        close(wm, 8.0, 0.025);
        close(im, 0.2, 0.015);
    }
    #[test]
    fn group_constraints_and_spectrum() {
        let mut rng = SplitMix64::from_seed(542);
        let group = SpecialOrthogonalGroup::<f64>::with_options(
            4,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let q = Matrix::from_vec(4, 4, group.sample(&mut rng).unwrap()).unwrap();
        let gram = q
            .view()
            .mul_mat(&q.view().transpose().try_to_matrix().unwrap())
            .unwrap();
        for i in 0..4 {
            for j in 0..4 {
                close(
                    gram.view().get(i, j).unwrap(),
                    if i == j { 1.0 } else { 0.0 },
                    1e-13,
                );
            }
        }
        close(
            scioxide_linalg::LuDecomposition::factor(q.view())
                .and_then(|factor| factor.determinant())
                .unwrap(),
            1.0,
            1e-13,
        );
        let unitary = UnitaryGroup::<f32>::with_options(
            3,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let mut re = 0.0;
                let mut im = 0.0;
                for k in 0..3 {
                    let a = unitary[i * 3 + k];
                    let b = unitary[j * 3 + k];
                    re += a.re * b.re + a.im * b.im;
                    im += a.im * b.re - a.re * b.im;
                }
                close(re as f64, if i == j { 1.0 } else { 0.0 }, 1e-6);
                close(im as f64, 0.0, 1e-6);
            }
        }
        let correlation = RandomCorrelation::with_options(
            &[0.0, 0.5, 1.0, 2.5],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        for _ in 0..50 {
            let c = Matrix::from_vec(4, 4, correlation.sample(&mut rng).unwrap()).unwrap();
            for i in 0..4 {
                close(c.view().get(i, i).unwrap(), 1.0, 1e-12);
            }
            let e = scioxide_linalg::HermitianEigenDecomposition::decompose(
                &c,
                scioxide_linalg::HermitianInput::Exact,
            )
            .unwrap();
            let mut values = e.eigenvalues().to_vec();
            values.sort_by(f64::total_cmp);
            for (a, b) in values.into_iter().zip([0.0, 0.5, 1.0, 2.5]) {
                close(a, b, 1e-11);
            }
        }
    }
    #[test]
    fn t_singular_support_and_moments() {
        let shape = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let t = MultivariateStudentT::with_options(
            &[0.0, 0.0],
            &shape,
            0.5,
            Default::default(),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert_eq!(t.rank(), 1);
        assert!(t.mean().is_err());
        assert!(t.log_pdf(&[0.0, 0.0]).is_err());
        assert_eq!(
            t.log_pseudo_density(
                &[0.0, 1.0],
                scioxide_core::Tolerance::from_machine_epsilon()
            )
            .unwrap(),
            f64::NEG_INFINITY
        );
        let ig = NormalInverseGamma::with_options(
            0.0,
            2.0,
            3.0,
            4.0,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let lp = ig.log_pdf((0.0, 2.0)).unwrap();
        let expected = Normal::new(0.0, 1.0).unwrap().log_pdf(0.0).unwrap()
            + Gamma::new(3.0, 0.25).unwrap().log_pdf(0.5).unwrap()
            - 2.0 * 2.0_f64.ln();
        close(lp, expected, 1e-12);
    }
}

/// Small row-major matrix products in scratch storage. Products accumulate
/// before rounding.
fn row_product(
    left: MatrixView<'_, f64>,
    right: MatrixView<'_, f64>,
    out: &mut [f64],
) -> Result<(), SciOxideError> {
    size(left.cols(), right.rows())?;
    size(out.len(), left.rows() * right.cols())?;
    for i in 0..left.rows() {
        for j in 0..right.cols() {
            let mut sum = scioxide_core::RangeAccumulator::new();
            for k in 0..left.cols() {
                sum.add_product(left.get(i, k)?, right.get(k, j)?)?;
            }
            out[i * right.cols() + j] = sum.finish()?;
        }
    }
    Ok(())
}
fn row_lower_solve(
    lower: MatrixView<'_, f64>,
    rhs: &mut [f64],
    cols: usize,
) -> Result<(), SciOxideError> {
    size(rhs.len(), lower.rows() * cols)?;
    for i in 0..lower.rows() {
        for j in 0..cols {
            let mut sum = scioxide_core::RangeAccumulator::new();
            sum.add(rhs[i * cols + j])?;
            for k in 0..i {
                sum.add_product(-lower.get(i, k)?, rhs[k * cols + j])?;
            }
            rhs[i * cols + j] = finite(sum.finish()? / lower.get(i, i)?)?;
        }
    }
    Ok(())
}
