#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex, EvaluationRole, ExecutionContext, ProblemToken,
    SolveBudgets, SolveState,
};
use scioxide_poly::{ApproxPolynomial, PolynomialApproximationPolicy};
use scioxide_rootfind::{
    PolynomialRootMethod as Method, PolynomialRootOptions, PolynomialRootWorkspace, RootTermination,
};

#[test]
fn interrupted_finalization_retains_only_exact_candidate_diagnostics() {
    let coefficients = [-1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0];
    let options = PolynomialRootOptions::default().with_method(Method::Aberth);
    let full = scioxide_rootfind::polynomial_roots_with_options(&coefficients, &options).unwrap();
    assert!(full.report().termination().is_success());
    let total = full.report().counts().provider(EvaluationRole::Function);
    for remaining in 1..=full.len() {
        let limit = total - remaining;
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let limited = PolynomialRootOptions::default()
            .with_method(Method::Aberth)
            .with_budgets(budgets);
        let mut workspace = PolynomialRootWorkspace::workspace(5, &limited).unwrap();
        let mut output = [Complex::new(99.0, 99.0); 5];
        let report = scioxide_rootfind::polynomial_roots_into(
            &coefficients,
            &limited,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(report.counts().provider(EvaluationRole::Function), limit);
        for &point in &output {
            assert!(full.iter().any(|&root| (point - root).norm() <= 1e-12));
        }
        for (&point, diagnostic) in output.iter().zip(workspace.diagnostics()) {
            assert!(diagnostic.is_available());
            if let Some(residual) = diagnostic.backward_error() {
                assert_eq!(
                    residual,
                    scioxide_poly::checked_basis_evaluation(
                        &coefficients,
                        point,
                        scioxide_poly::PolynomialRootBasisRef::Monomial
                    )
                    .unwrap()
                    .normalized_residual()
                );
            }
        }
    }
}

fn reconstruction(c: &[Complex<f64>], roots: &[Complex<f64>]) -> f64 {
    let mut product = vec![*c.last().unwrap()];
    for &root in roots {
        let mut next = vec![Complex::new(0.0, 0.0); product.len() + 1];
        for (i, &value) in product.iter().enumerate() {
            next[i] -= root * value;
            next[i + 1] += value;
        }
        product = next;
    }
    product
        .iter()
        .zip(c)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
        / c.iter().map(|c| c.norm()).fold(0.0, f64::max)
}
fn independent_error(c: &[Complex<f64>], z: Complex<f64>) -> f64 {
    let p = c
        .iter()
        .rev()
        .fold(Complex::new(0.0, 0.0), |p, c| p * z + *c);
    let b = c.iter().rev().fold(0.0, |b, c| b * z.norm() + c.norm());
    if b == 0.0 { p.norm() } else { p.norm() / b }
}
#[test]
fn decimal_analytic_oracles_match_and_reconstruct_without_duplicate_roots() {
    let mut cases =
        std::collections::BTreeMap::<String, (Vec<Complex<f64>>, Vec<Complex<f64>>)>::new();
    // Shared 100-digit Decimal fixtures use analytic identities, not another solver.
    for row in include_str!("../../scioxide-poly/tests/data/root_diagnostics.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split(';').collect();
        let c = fields[1]
            .split('|')
            .map(|c| {
                let (re, im) = c.split_once(',').unwrap();
                Complex::new(re.parse().unwrap(), im.parse().unwrap())
            })
            .collect();
        cases
            .entry(fields[0].to_owned())
            .or_insert_with(|| (c, Vec::new()))
            .1
            .push(Complex::new(
                fields[2].parse().unwrap(),
                fields[3].parse().unwrap(),
            ));
    }
    for (name, (c, expected)) in cases {
        for method in [
            Method::Auto,
            Method::Companion,
            Method::Aberth,
            Method::DurandKerner,
            Method::Laguerre,
        ] {
            let result = scioxide_rootfind::polynomial_roots_with_options(
                &c,
                &PolynomialRootOptions::default().with_method(method),
            )
            .unwrap();
            assert!(
                result.report().termination().is_success(),
                "{name} {method:?}: {:?}",
                result.report()
            );
            let mut unmatched = expected.clone();
            for (&root, d) in result.roots().iter().zip(result.diagnostics()) {
                let index = unmatched
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| (**a - root).norm().total_cmp(&(**b - root).norm()))
                    .unwrap()
                    .0;
                assert!(
                    (unmatched.remove(index) - root).norm() < 2e-7,
                    "{name} {method:?}: {root:?}"
                );
                assert!(d.is_validated());
                assert!(independent_error(&c, root) < 2e-12);
            }
            assert!(unmatched.is_empty());
            assert!(
                reconstruction(&c, result.roots()) < 2e-9,
                "{name} {method:?}"
            );
        }
    }
}
macro_rules! width_cases {
    ($name:ident,$s:ty,$tol:expr,$small:expr) => {
        #[test]
        fn $name() {
            let zero = Complex::new(0.0 as $s, 0.0);
            // A tiny conjugate pair at the origin must not be snapped to zero:
            // that projection has backward error one, regardless of root magnitude.
            let delta: $s = $small;
            let result = scioxide_rootfind::polynomial_roots_with_options(
                &[delta * delta, 0.0, 1.0],
                &PolynomialRootOptions::default(),
            )
            .unwrap();
            assert!(result.report().termination().is_success());
            assert!(result.roots().iter().all(|r| r.im != 0.0));
            for root in result.roots() {
                assert!((root.im.abs() - delta).abs() < $tol * delta);
            }
            // Exact Gaussian-integer factors: (z-i)(z-2)(z+1+i).
            let c: [Complex<$s>; 4] = [
                Complex::new(-2.0, 2.0),
                Complex::new(-1.0, -1.0),
                Complex::new(-1.0, 0.0),
                Complex::new(1.0, 0.0),
            ];
            for method in [
                Method::Auto,
                Method::Aberth,
                Method::DurandKerner,
                Method::Laguerre,
            ] {
                let roots = scioxide_rootfind::polynomial_roots_with_options(
                    &c,
                    &PolynomialRootOptions::<$s>::default().with_method(method),
                )
                .unwrap();
                assert!(
                    roots.report().termination().is_success(),
                    "{method:?}: {:?}",
                    roots.report()
                );
                for expected in [
                    Complex::new(0.0, 1.0),
                    Complex::new(2.0, 0.0),
                    Complex::new(-1.0, -1.0),
                ] {
                    assert!(
                        roots.iter().any(|r| (*r - expected).norm() < $tol),
                        "{method:?}: {roots:?}"
                    );
                }
            }
            let mut coefficients = vec![zero; 9];
            coefficients[8] = Complex::new(1.0, 0.0);
            let basis_plan = scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::ChebyshevT,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap()
            .prepare(coefficients.len() - 1, Default::default())
            .unwrap();
            let polynomial = ApproxPolynomial::orthogonal_with_plan(
                coefficients,
                basis_plan,
                PolynomialApproximationPolicy::default(),
            )
            .unwrap();
            for method in [Method::Auto, Method::Companion, Method::Aberth] {
                let roots = scioxide_rootfind::polynomial_roots_with_options(
                    &polynomial,
                    &PolynomialRootOptions::default().with_method(method),
                )
                .unwrap();
                assert!(
                    roots.report().termination().is_success(),
                    "{method:?}: {:?}",
                    roots.report()
                );
                for k in 0..8 {
                    let expected = (((2 * k + 1) as $s) * (<$s as scioxide_core::RealScalar>::PI)
                        / 16.0)
                        .cos();
                    assert!(
                        roots
                            .iter()
                            .any(|r| (*r - Complex::new(expected, 0.0)).norm() < $tol)
                    );
                }
            }
            let legendre = ApproxPolynomial::orthogonal_with_plan(
                vec![zero, zero, zero, Complex::new(1.0, 0.0)],
                scioxide_poly::OrthogonalBasisSpec::new(
                    scioxide_poly::OrthogonalFamily::Legendre,
                    scioxide_poly::BasisNormalization::Classical,
                    0.0,
                    1.0,
                )
                .unwrap()
                .prepare(3, Default::default())
                .unwrap(),
                PolynomialApproximationPolicy::default(),
            )
            .unwrap();
            let roots = scioxide_rootfind::polynomial_roots_with_options(
                &legendre,
                &PolynomialRootOptions::default(),
            )
            .unwrap();
            for expected in [-((0.6 as $s).sqrt()), 0.0, (0.6 as $s).sqrt()] {
                assert!(
                    roots
                        .iter()
                        .any(|r| (*r - Complex::new(expected, 0.0)).norm() < $tol)
                );
            }
        }
    };
}
width_cases!(single, f32, 3e-4, 2.0_f32.powi(-24));
width_cases!(double, f64, 2e-10, 2.0_f64.powi(-60));

#[test]
fn partial_final_verification_preserves_exact_candidate_diagnostics() {
    let coefficients = [2.0_f64, -3.0, 1.0];
    let options = PolynomialRootOptions::default().with_polish(false);
    let mut workspace = PolynomialRootWorkspace::new();
    PolynomialRootWorkspace::workspace_requirements(2, &PolynomialRootOptions::<f64>::default())
        .and_then(|requirements| workspace.try_reserve(&requirements))
        .unwrap();
    let mut output = [Complex::new(99.0, 99.0); 2];
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(582),
        scioxide_core::algorithm_id!("test", "polynomial-parent", 1),
        2_u128,
        0,
    );
    for limit in [1, 2] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let report = scioxide_rootfind::polynomial_roots_into(
            &coefficients,
            &options,
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(report.counts().provider(EvaluationRole::Function), limit);
        assert_eq!(
            workspace
                .diagnostics()
                .iter()
                .filter(|d| d.is_validated())
                .count(),
            limit
        );
        assert_eq!(report.termination().is_success(), limit == 2);
        for (&root, d) in output.iter().zip(workspace.diagnostics()) {
            assert!(d.is_available());
            if let Some(error) = d.backward_error() {
                assert_eq!(
                    error,
                    scioxide_poly::checked_basis_evaluation(
                        &coefficients,
                        root,
                        scioxide_poly::PolynomialRootBasisRef::Monomial
                    )
                    .unwrap()
                    .normalized_residual()
                );
            }
        }
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    let report = scioxide_rootfind::polynomial_roots_into(
        &coefficients,
        &options,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(report.termination(), RootTermination::Cancelled);
    assert_eq!(report.counts().provider(EvaluationRole::Function), 0);
    assert!(
        workspace
            .diagnostics()
            .iter()
            .all(|d| !d.is_available() && d.backward_error().is_none())
    );
}

#[test]
fn complex_phase_scaling_and_high_degree_have_independent_residuals() {
    let coefficients = [
        Complex::new(-2.0_f64, 2.0),
        Complex::new(-1.0, -1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let options = PolynomialRootOptions::default();
    let reference = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &options.clone().with_method(Method::Aberth),
    )
    .unwrap();
    for exponent in [-500, 500] {
        let factor = 2.0_f64.powi(exponent);
        let scaled = coefficients.map(|c| c * Complex::new(factor, -factor));
        let result = scioxide_rootfind::polynomial_roots_with_options(
            &scaled,
            &options.clone().with_method(Method::Aberth),
        )
        .unwrap();
        assert!(result.report().termination().is_success());
        for (&root, &expected) in result.iter().zip(reference.iter()) {
            assert!((root - expected).norm() < 1e-10);
            assert!(independent_error(&scaled, root) < 1e-12);
        }
    }
    let mut c = vec![Complex::new(0.0, 0.0); 101];
    c[0] = Complex::new(-1.0, 0.0);
    c[100] = Complex::new(1.0, 0.0);
    let result =
        scioxide_rootfind::polynomial_roots_with_options(&c, &options.with_method(Method::Aberth))
            .unwrap();
    assert!(result.report().termination().is_success());
    for k in 0..100 {
        let angle = 2.0 * core::f64::consts::PI * f64::from(k) / 100.0;
        let expected = Complex::new(angle.cos(), angle.sin());
        assert!(result.iter().any(|r| (*r - expected).norm() < 1e-10));
    }
    assert!(result.iter().all(|&r| independent_error(&c, r) < 1e-12));
}

#[cfg(feature = "std")]
#[test]
fn native_basis_scopes_accept_unrelated_parent_identity_and_match_injected_execution() {
    use scioxide_core::ThreadExecutor;
    let polynomial = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0_f64, 0.0, 0.0, 1.0],
        scioxide_poly::OrthogonalBasisSpec::new(
            scioxide_poly::OrthogonalFamily::Legendre,
            scioxide_poly::BasisNormalization::Classical,
            0.0,
            1.0,
        )
        .unwrap()
        .prepare(3, Default::default())
        .unwrap(),
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    let options = PolynomialRootOptions::default();
    let expected = scioxide_rootfind::polynomial_roots_with_options(&polynomial, &options).unwrap();
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let mut workspace = PolynomialRootWorkspace::new();
    PolynomialRootWorkspace::workspace_requirements(3, &PolynomialRootOptions::<f64>::default())
        .and_then(|requirements| workspace.try_reserve(&requirements))
        .unwrap();
    let mut output = [Complex::new(42.0, 7.0); 3];
    for _ in 0..2 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(582),
            scioxide_core::algorithm_id!("test", "polynomial-parent", 1),
            3_u128,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::new(&pool), identity, &mut budgets)
            .unwrap();
        let result = scioxide_rootfind::polynomial_roots_into(
            &polynomial,
            &options,
            &mut output,
            &mut workspace,
            &mut context,
        );
        assert!(result.unwrap().termination().is_success());
        assert_eq!(&output, expected.roots());
        assert_eq!(workspace.diagnostics(), expected.diagnostics());
        assert_eq!(context.counts().own().provider(EvaluationRole::Function), 0);
        assert!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function)
                > 0
        );
    }
}

#[test]
fn deadline_inside_eigenprovider_returns_available_partial_roots() {
    use core::sync::atomic::{AtomicU64, Ordering};
    use scioxide_core::{CapabilityError, Clock, Deadline, MonotonicInstant};
    struct TickClock(AtomicU64);
    impl Clock for TickClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(self.0.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }
    let mut coefficients = vec![0.0_f64; 21];
    coefficients[0] = -1.0;
    coefficients[20] = 1.0;
    let options = PolynomialRootOptions::default();
    for limit in [4, 8] {
        let clock = TickClock(AtomicU64::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let deadline = Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_nanos(limit)),
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(582),
            scioxide_core::algorithm_id!("test", "polynomial-parent", 1),
            20_u128,
            0,
        );
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let mut workspace = PolynomialRootWorkspace::new();
        PolynomialRootWorkspace::workspace_requirements(
            20,
            &PolynomialRootOptions::<f64>::default().with_method(Method::Companion),
        )
        .and_then(|requirements| workspace.try_reserve(&requirements))
        .unwrap();
        let mut output = [Complex::new(0.0, 0.0); 20];
        let result = scioxide_rootfind::polynomial_roots_into(
            &coefficients,
            &options.clone().with_method(Method::Companion),
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(result.termination(), RootTermination::TimeLimit);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Factorization),
            1
        );
        assert_eq!(clock.0.load(Ordering::Relaxed), limit + 1);
        assert!(workspace.diagnostics().iter().all(|d| d.is_available()));
    }
}
