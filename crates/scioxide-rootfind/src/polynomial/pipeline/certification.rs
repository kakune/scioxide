//! Polishing, backward-error checks and ordering of polynomial roots.
use super::*;
pub(crate) fn residual_ok<S: RealScalar>(
    error: S,
    options: &PolynomialRootOptions<S>,
) -> Result<bool, SciOxideError> {
    options
        .backward_error_tolerance
        .tolerance()
        .is_residual_ok(error, S::ONE)
}
pub(crate) fn assess<S: RealScalar>(
    coefficients: &[impl NumericScalar<Real = S>],
    basis: PolynomialRootBasisRef<'_, S>,
    root: Complex<S>,
    options: &PolynomialRootOptions<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<PolynomialRootDiagnostic<S>, PolynomialFailure> {
    let e = invoke(session, scioxide_core::EvaluationRole::Function, || {
        checked_basis_evaluation(coefficients, root, basis)
    })?;
    diagnostic(e, options)
}

fn diagnostic<S: RealScalar>(
    e: scioxide_poly::CheckedBasisEvaluation<S>,
    options: &PolynomialRootOptions<S>,
) -> Result<PolynomialRootDiagnostic<S>, PolynomialFailure> {
    Ok(PolynomialRootDiagnostic {
        available: true,
        backward_error: Some(e.normalized_residual()),
        condition: e.root_condition()?.absolute(),
        validated: residual_ok(e.normalized_residual(), options)?,
    })
}
pub(crate) fn finalize<S: RealScalar>(
    coefficients: &[impl NumericScalar<Real = S>],
    p: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
    options: &PolynomialRootOptions<S>,
    stats: &mut PolynomialRootReport<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), PolynomialFailure>
where
    Complex<S>: NumericScalar<Real = S>,
{
    // Iteration used a possibly scaled/deflated polynomial. Only original-input
    // evaluations from this phase can supply final diagnostics.
    w.evaluations.fill(None);
    if options.polish {
        for i in p.zeros..w.candidates.len() {
            for _ in 0..10.min(options.max_iterations.get()) {
                iteration(session)?;
                let root = w.candidates[i];
                let e = match w.evaluations[i] {
                    Some(e) => e,
                    None => invoke(session, scioxide_core::EvaluationRole::Function, || {
                        checked_basis_evaluation(coefficients, root, p.basis)
                    })?,
                };
                w.evaluations[i] = Some(e);
                if e.normalized_residual().is_zero() {
                    break;
                }
                let Some(correction) = e.newton_correction()? else {
                    break;
                };
                let proposal = root - correction;
                if !proposal.re.is_finite() || !proposal.im.is_finite() {
                    break;
                }
                let d = if proposal == root {
                    e
                } else {
                    invoke(session, scioxide_core::EvaluationRole::Function, || {
                        checked_basis_evaluation(coefficients, proposal, p.basis)
                    })?
                };
                if d.normalized_residual() > e.normalized_residual() {
                    break;
                }
                w.candidates[i] = proposal;
                w.evaluations[i] = Some(d);
                if options
                    .step_tolerance
                    .tolerance()
                    .is_residual_ok((proposal - root).norm(), S::ONE + proposal.norm())?
                {
                    break;
                }
            }
        }
    }
    for i in 0..w.candidates.len() {
        // An earlier pair transaction may already have assessed this exact point.
        if w.scratch_diagnostics[i].backward_error.is_some() {
            continue;
        }
        admitted(session.check_stop()?)?;
        let root = w.candidates[i];
        let e = match w.evaluations[i] {
            Some(e) => e,
            None => invoke(session, scioxide_core::EvaluationRole::Function, || {
                checked_basis_evaluation(coefficients, root, p.basis)
            })?,
        };
        w.evaluations[i] = Some(e);
        w.scratch_diagnostics[i] = diagnostic(e, options)?;
        if !p.real || root.im.is_zero() {
            continue;
        }
        // A first-order roundoff displacement is capped by sqrt(epsilon): a
        // singular/multiple-root estimate never licenses an arbitrary projection.
        let cap = S::SQRT_EPSILON * (S::ONE + root.norm());
        let eight = S::try_from_usize_exact(8)?;
        let floor = eight * S::EPSILON * (S::ONE + root.norm());
        let threshold = w.scratch_diagnostics[i]
            .condition
            .value()
            .map_or(floor, |condition| {
                floor.max((eight * S::EPSILON * condition).min(cap))
            });
        if root.im.abs() <= threshold {
            let proposal = Complex::new(root.re, S::ZERO);
            let d = invoke(session, scioxide_core::EvaluationRole::Function, || {
                checked_basis_evaluation(coefficients, proposal, p.basis)
            })?;
            let error = d.residual_with_roundoff();
            if error.is_finite() && residual_ok(error, options)? {
                let assessment = diagnostic(d, options)?;
                w.candidates[i] = proposal;
                w.scratch_diagnostics[i] = assessment;
                stats.canonicalization_changed = true;
            }
            // Each root receives one canonicalization proposal. A rejected real
            // projection keeps the already assessed point, without a pairing trial.
            w.evaluations[i] = None;
            continue;
        }
        if root.im < S::ZERO {
            continue;
        }
        let partner = w
            .candidates
            .iter()
            .enumerate()
            // A completed assessment with no cached state marks a consumed
            // canonicalization trial, so a root cannot be paired twice.
            .filter(|(j, other)| {
                other.im < S::ZERO
                    && (w.scratch_diagnostics[*j].backward_error.is_none()
                        || w.evaluations[*j].is_some())
            })
            .min_by(|(_, a), (_, b)| complex_order_distance(**a, **b, root.conj()))
            .map(|(j, _)| j);
        if let Some(j) = partner {
            let partner = w.candidates[j];
            if (partner - root.conj()).norm() > threshold {
                continue;
            }
            let proposal = Complex::new(
                root.re * S::HALF + partner.re * S::HALF,
                root.im * S::HALF - partner.im * S::HALF,
            );
            let first = if proposal == root {
                e
            } else {
                invoke(session, scioxide_core::EvaluationRole::Function, || {
                    checked_basis_evaluation(coefficients, proposal, p.basis)
                })?
            };
            let error = first.residual_with_roundoff();
            if error.is_finite() && residual_ok(error, options)? {
                // Real coefficients imply identical residual and absolute
                // conditioning for conjugate points, including native bases.
                let assessment = diagnostic(first, options)?;
                stats.canonicalization_changed |= root != proposal || partner != proposal.conj();
                w.candidates[i] = proposal;
                w.candidates[j] = proposal.conj();
                w.scratch_diagnostics[i] = assessment;
                w.scratch_diagnostics[j] = assessment;
                w.evaluations[i] = None;
                w.evaluations[j] = None;
            }
        }
    }
    Ok(())
}
pub(crate) fn complex_order_distance<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    target: Complex<S>,
) -> Ordering {
    (a - target)
        .norm()
        .partial_cmp(&(b - target).norm())
        .unwrap_or(Ordering::Equal)
}
pub(crate) fn sort_with_diagnostics<S: RealScalar>(
    roots: &mut [Complex<S>],
    diagnostics: &mut [PolynomialRootDiagnostic<S>],
) {
    // In-place paired heapsort avoids an allocation in stable slice sorting.
    pub(crate) fn less<S: RealScalar>(a: Complex<S>, b: Complex<S>) -> bool {
        a.re < b.re || (a.re == b.re && a.im < b.im)
    }
    pub(crate) fn sift<S: RealScalar>(
        r: &mut [Complex<S>],
        d: &mut [PolynomialRootDiagnostic<S>],
        mut parent: usize,
        end: usize,
    ) {
        while parent < end / 2 {
            let mut child = 2 * parent + 1;
            if child + 1 < end && less(r[child], r[child + 1]) {
                child += 1;
            }
            if !less(r[parent], r[child]) {
                break;
            }
            r.swap(parent, child);
            d.swap(parent, child);
            parent = child;
        }
    }
    for i in (0..roots.len() / 2).rev() {
        sift(roots, diagnostics, i, roots.len());
    }
    for end in (1..roots.len()).rev() {
        roots.swap(0, end);
        diagnostics.swap(0, end);
        sift(roots, diagnostics, 0, end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets, SolveState};

    fn finish(
        coefficients: &[f64],
        candidates: &[Complex<f64>],
        options: PolynomialRootOptions<f64>,
    ) -> (PolynomialRootWorkspace<f64>, PolynomialRootReport<f64>) {
        let degree = candidates.len();
        let mut workspace = PolynomialRootWorkspace::workspace(
            degree,
            &options.clone().with_method(PolynomialRootMethod::Aberth),
        )
        .unwrap();
        workspace.candidates.copy_from_slice(candidates);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(2011),
            scioxide_core::algorithm_id!("test", "polynomial-finalization", 1),
            degree as u128,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let session = crate::context::Session::new(&mut context);
        let mut report = PolynomialRootReport::new(
            PolynomialRootMethod::Aberth,
            options.backward_error_tolerance.tolerance(),
            1.0,
            degree,
            session.algorithm().unwrap(),
        );
        finalize(
            coefficients,
            Prepared {
                zeros: 0,
                degree,
                scale: 1.0,
                basis: PolynomialRootBasisRef::Monomial,
                real: true,
                companion_safe: true,
            },
            &mut workspace,
            &options,
            &mut report,
            &session,
        )
        .unwrap();
        report.counts = session.accounting().unwrap();
        for (&root, d) in workspace
            .candidates
            .iter()
            .zip(&workspace.scratch_diagnostics)
        {
            let expected =
                checked_basis_evaluation(coefficients, root, PolynomialRootBasisRef::Monomial)
                    .unwrap();
            assert_eq!(d.backward_error, Some(expected.normalized_residual()));
            assert_eq!(d.condition, expected.root_condition().unwrap().absolute());
        }
        (workspace, report)
    }

    #[test]
    fn finalization_reuses_exact_polishing_evaluations_and_conjugate_symmetry() {
        for polish in [false, true] {
            let options = PolynomialRootOptions::default().with_polish(polish);
            let (_, report) = finish(
                &[-1.0, 0.0, 1.0],
                &[Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)],
                options.clone(),
            );
            assert_eq!(report.counts().provider(EvaluationRole::Function), 2);
            let (_, report) = finish(
                &[2.0, -2.0, 1.0],
                &[Complex::new(1.0, 1.0), Complex::new(1.0, -1.0)],
                options,
            );
            assert_eq!(
                report.counts().provider(EvaluationRole::Function),
                if polish { 2 } else { 1 }
            );
        }
    }

    #[test]
    fn real_projection_has_one_checked_trial_whether_accepted_or_rejected() {
        let roots = [
            Complex::new(-1.0, f64::EPSILON),
            Complex::new(1.0, -f64::EPSILON),
        ];
        for accepted in [false, true] {
            let mut options = PolynomialRootOptions::default().with_polish(false);
            if !accepted {
                options = options.with_backward_error_tolerance(
                    scioxide_core::ResidualTolerance::from_tolerance(
                        scioxide_core::Tolerance::new(0.0, 0.0).unwrap(),
                    ),
                );
            }
            let (workspace, report) = finish(&[-1.0, 0.0, 1.0], &roots, options);
            assert_eq!(report.counts().provider(EvaluationRole::Function), 4);
            assert_eq!(report.canonicalization_changed, accepted);
            if accepted {
                assert!(workspace.candidates.iter().all(|r| r.im == 0.0));
            } else {
                assert_eq!(workspace.candidates, roots);
            }
        }
    }

    #[test]
    fn perturbed_conjugates_share_one_trial_and_diagnostics_in_either_order() {
        for reverse in [false, true] {
            let mut roots = [
                Complex::new(1.0 + 4.0 * f64::EPSILON, 1.0),
                Complex::new(1.0, -1.0),
            ];
            if reverse {
                roots.reverse();
            }
            let (workspace, report) = finish(
                &[2.0, -2.0, 1.0],
                &roots,
                PolynomialRootOptions::default().with_polish(false),
            );
            assert!(report.canonicalization_changed);
            assert_eq!(
                report.counts().provider(EvaluationRole::Function),
                if reverse { 3 } else { 2 }
            );
            assert_eq!(workspace.candidates[0], workspace.candidates[1].conj());
        }
    }

    #[test]
    fn adaptive_polishing_retains_each_accepted_evaluation_for_the_next_step() {
        let (workspace, report) = finish(
            &[-1.0, 0.0, 1.0],
            &[Complex::new(-1.1, 0.0), Complex::new(1.1, 0.0)],
            PolynomialRootOptions::default(),
        );
        let count = report.counts().provider(EvaluationRole::Function);
        assert!(report.iterations() > 4);
        assert!(count <= report.iterations() + 2);
        assert!(count < 2 * report.iterations());
        assert!(workspace.scratch_diagnostics.iter().all(|d| d.validated));
    }
}
