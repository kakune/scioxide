//! KKT provider selection and factor reuse: a new matrix invalidates the
//! factorization, a new right-hand side reuses it.
use super::{DenseKktProvider, KktProvider};
use crate::context::SolveSession;
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, RankPolicy};

enum ResidualCheck {
    Componentwise,
    Normwise,
}
impl ResidualCheck {
    fn verify<S: RealScalar + NumericScalar<Real = S>>(
        &self,
        matrix: &Matrix<S>,
        rhs: &[S],
        solution: &[S],
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Componentwise => super::verify_residual(matrix.view(), rhs, solution),
            Self::Normwise => matrix
                .view()
                .verify_solution_normwise(
                    rhs,
                    solution,
                    scioxide_core::Tolerance::new(
                        <S as scioxide_core::AlgebraicScalar>::ZERO,
                        S::SQRT_EPSILON,
                    )?,
                )
                .map(|_| ()),
        }
    }
}

pub(crate) struct Factor<'a, S: RealScalar + NumericScalar<Real = S>> {
    injected: Option<&'a mut (dyn KktProvider<S> + Send)>,
    dense: Option<DenseKktProvider<S>>,
    ready: bool,
    singular: bool,
    residual_check: ResidualCheck,
    staging: alloc::vec::Vec<S>,
    residual: alloc::vec::Vec<S>,
    correction: alloc::vec::Vec<S>,
}

impl<'a, S: RealScalar + NumericScalar<Real = S>> Factor<'a, S> {
    pub(crate) fn new(
        provider: Option<&'a mut (dyn KktProvider<S> + Send)>,
        maximum: usize,
        rank_policy: Option<RankPolicy<S>>,
    ) -> Result<Self, SciOxideError> {
        let dense = if provider.is_none() {
            let mut dense = DenseKktProvider::new(maximum)?;
            dense.use_rank_policy(rank_policy);
            Some(dense)
        } else {
            None
        };
        Ok(Self {
            injected: provider,
            dense,
            ready: false,
            singular: false,
            residual_check: ResidualCheck::Componentwise,
            residual: scioxide_core::backend::try_filled(
                maximum,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("kkt_residual"),
            )?,
            correction: scioxide_core::backend::try_filled(
                maximum,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("kkt_correction"),
            )?,
            staging: scioxide_core::backend::try_filled(
                maximum,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("kkt_staging"),
            )?,
        })
    }
    pub(crate) fn with_normwise_residual(mut self) -> Self {
        self.residual_check = ResidualCheck::Normwise;
        self
    }
    pub(crate) fn invalidate(&mut self) {
        self.ready = false;
    }
    /// Drop the built-in dense provider when a lent provider serves every call.
    pub(crate) fn release_builtin(&mut self) {
        self.dense = None;
        self.ready = false;
    }
    /// Factor with `lent`, when supplied, in place of the owned providers.
    pub(crate) fn factor_lent(
        &mut self,
        lent: Option<&mut dyn KktProvider<S>>,
        matrix: &Matrix<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<scioxide_sparse::LdlInertia, SciOxideError> {
        self.ready = false;
        let builtin = lent.is_none() && self.injected.is_none();
        let provider: &mut dyn KktProvider<S> = match (lent, &mut self.injected, &mut self.dense) {
            (Some(provider), _, _) => provider,
            (None, Some(provider), _) => &mut **provider,
            (None, None, Some(dense)) => dense,
            _ => return Err(SciOxideError::singular(None)),
        };
        let inertia = session.invoke(EvaluationRole::Factorization, || {
            provider.factor(matrix.view(), session.execution()?)
        })?;
        super::validate_inertia(inertia, matrix.rows())?;
        self.singular = inertia.zero() != 0
            || (builtin && self.dense.as_ref().is_some_and(|dense| dense.singular));
        self.ready = true;
        Ok(inertia)
    }
    pub(crate) fn solve(
        &mut self,
        matrix: &Matrix<S>,
        rhs: &[S],
        output: &mut [S],
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.solve_lent(None, matrix, rhs, output, session)
    }
    /// Solve with `lent`, when supplied, in place of the owned providers.
    pub(crate) fn solve_lent(
        &mut self,
        mut lent: Option<&mut dyn KktProvider<S>>,
        matrix: &Matrix<S>,
        rhs: &[S],
        output: &mut [S],
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = matrix.rows();
        if rhs.len() != n || output.len() != n || n > self.staging.len() {
            return Err(SciOxideError::dimension_mismatch(
                [n, n],
                [rhs.len(), output.len()],
            ));
        }
        if !self.ready {
            self.factor_lent(super::lend(&mut lent), matrix, session)?;
        }
        let verify = lent.is_some() || self.injected.is_some() || !self.singular;
        let provider: &mut dyn KktProvider<S> = match (lent, &mut self.injected, &mut self.dense) {
            (Some(provider), _, _) => provider,
            (None, Some(provider), _) => &mut **provider,
            (None, None, Some(dense)) => dense,
            _ => return Err(SciOxideError::singular(None)),
        };
        session.invoke(EvaluationRole::LinearSolve, || {
            provider.solve_into(rhs, &mut self.staging[..n], session.execution()?)
        })?;
        crate::utils::validate_finite(&self.staging[..n])?;
        if verify {
            // Pivoted elimination is backward stable in a normwise sense; a
            // few same-factor corrections also resolve near-zero KKT rows
            // before demanding the stronger componentwise acceptance check.
            for attempt in 0..=3 {
                match self.residual_check.verify(matrix, rhs, &self.staging[..n]) {
                    Ok(()) => break,
                    Err(error)
                        if attempt < 3
                            && matches!(
                                error.view(),
                                scioxide_core::SciOxideErrorView::NumericalInstability {
                                    issue: scioxide_core::NumericalIssue::Breakdown,
                                    ..
                                }
                            ) => {}
                    Err(error) => return Err(error),
                }
                matrix.view().residual_into(
                    rhs,
                    &self.staging[..n],
                    &mut self.residual[..n],
                    &mut self.correction[..n],
                )?;
                session.invoke(EvaluationRole::LinearSolve, || {
                    provider.solve_into(
                        &self.residual[..n],
                        &mut self.correction[..n],
                        session.execution()?,
                    )
                })?;
                crate::utils::validate_finite(&self.correction[..n])?;
                for (value, &correction) in self.staging[..n].iter_mut().zip(&self.correction[..n])
                {
                    *value = scioxide_core::checked_sum(&[*value, correction])?;
                }
                session.cache_hit()?;
            }
        }
        output.copy_from_slice(&self.staging[..n]);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::MatrixView;
    use scioxide_sparse::LdlInertia;

    struct Counting<'a> {
        dense: DenseKktProvider<f64>,
        factors: &'a AtomicUsize,
        solves: &'a AtomicUsize,
        inaccurate: bool,
    }
    impl KktProvider<f64> for Counting<'_> {
        fn factor(
            &mut self,
            a: MatrixView<'_, f64>,
            e: ExecutionContext<'_>,
        ) -> Result<LdlInertia, SciOxideError> {
            self.factors.fetch_add(1, Ordering::Relaxed);
            self.dense.factor(a, e)
        }
        fn solve_into(
            &mut self,
            b: &[f64],
            x: &mut [f64],
            e: ExecutionContext<'_>,
        ) -> Result<(), SciOxideError> {
            self.solves.fetch_add(1, Ordering::Relaxed);
            if self.inaccurate {
                x.fill(f64::NAN);
                Ok(())
            } else {
                self.dense.solve_into(b, x, e)
            }
        }
    }
    fn session<T>(run: impl FnOnce(&SolveSession<'_, '_, '_>) -> Result<T, SciOxideError>) -> T {
        crate::context::sequential_with_budgets(
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(1975),
                scioxide_core::algorithm_id!("optimize", "KktTest", 1),
                2,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |context| {
                run(&SolveSession::new(
                    context,
                    scioxide_core::argument!("context"),
                ))
            },
        )
        .unwrap()
    }
    #[test]
    fn rhs_reuses_factors_and_same_shape_value_changes_invalidate() {
        let factors = AtomicUsize::new(0);
        let solves = AtomicUsize::new(0);
        let mut provider = Counting {
            dense: DenseKktProvider::new(1).unwrap(),
            factors: &factors,
            solves: &solves,
            inaccurate: false,
        };
        let mut factor = Factor::new(Some(&mut provider), 1, None).unwrap();
        let mut a = Matrix::from_vec(1, 1, alloc::vec![2.0]).unwrap();
        let mut x = [99.0];
        session(|session| {
            factor.solve(&a, &[4.0], &mut x, session)?;
            assert!((x[0] - 2.0).abs() < 1e-14);
            factor.solve(&a, &[6.0], &mut x, session)?;
            assert!((x[0] - 3.0).abs() < 1e-14);
            assert_eq!(
                (
                    factors.load(Ordering::Relaxed),
                    solves.load(Ordering::Relaxed)
                ),
                (1, 2)
            );
            a.as_mut_slice()[0] = 4.0;
            factor.invalidate();
            factor.solve(&a, &[4.0], &mut x, session)?;
            assert!((x[0] - 1.0).abs() < 1e-14);
            assert_eq!(
                (
                    factors.load(Ordering::Relaxed),
                    solves.load(Ordering::Relaxed)
                ),
                (2, 3)
            );
            let counts = session.counts()?;
            assert_eq!(counts.provider(EvaluationRole::Factorization), 2);
            assert_eq!(counts.provider(EvaluationRole::LinearSolve), 3);
            Ok(())
        });
    }
    #[test]
    fn residual_corrections_reuse_a_single_factor_and_charge_each_solve() {
        struct OnceInexact {
            dense: DenseKktProvider<f64>,
            calls: usize,
        }
        impl KktProvider<f64> for OnceInexact {
            fn factor(
                &mut self,
                a: MatrixView<'_, f64>,
                e: ExecutionContext<'_>,
            ) -> Result<LdlInertia, SciOxideError> {
                self.dense.factor(a, e)
            }
            fn solve_into(
                &mut self,
                b: &[f64],
                x: &mut [f64],
                e: ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                self.dense.solve_into(b, x, e)?;
                if self.calls == 0 {
                    x[0] += 1e-4;
                }
                self.calls += 1;
                Ok(())
            }
        }
        let mut provider = OnceInexact {
            dense: DenseKktProvider::new(1).unwrap(),
            calls: 0,
        };
        let mut factor = Factor::new(Some(&mut provider), 1, None).unwrap();
        let a = Matrix::from_vec(1, 1, alloc::vec![2.0]).unwrap();
        let mut output = [99.0];
        session(|session| {
            factor.solve(&a, &[4.0], &mut output, session)?;
            assert!((output[0] - 2.0).abs() < 1e-14);
            assert_eq!(session.counts()?.provider(EvaluationRole::Factorization), 1);
            assert_eq!(session.counts()?.provider(EvaluationRole::LinearSolve), 2);
            Ok(())
        });
    }

    #[test]
    fn bad_injected_solution_never_publishes() {
        let factors = AtomicUsize::new(0);
        let solves = AtomicUsize::new(0);
        let mut provider = Counting {
            dense: DenseKktProvider::new(1).unwrap(),
            factors: &factors,
            solves: &solves,
            inaccurate: true,
        };
        let mut factor = Factor::new(Some(&mut provider), 1, None).unwrap();
        let a = Matrix::from_vec(1, 1, alloc::vec![2.0]).unwrap();
        let mut x = [99.0];
        session(|session| {
            assert!(factor.solve(&a, &[4.0], &mut x, session).is_err());
            assert_eq!(x, [99.0]);
            Ok(())
        });
    }
    #[test]
    fn singular_models_retain_minimum_norm_and_nullspace_residuals() {
        let a = Matrix::from_vec(2, 2, alloc::vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut factor = Factor::new(None, 2, Some(RankPolicy::DimensionScaledEpsilon)).unwrap();
        let mut x = [99.0; 2];
        session(|session| {
            factor.solve(&a, &[2.0, 2.0], &mut x, session)?;
            assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
            factor.solve(&a, &[3.0, 1.0], &mut x, session)?;
            assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
            assert_eq!(session.counts()?.provider(EvaluationRole::Factorization), 1);
            Ok(())
        });
    }
}
