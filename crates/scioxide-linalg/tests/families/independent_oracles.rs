//! Independent 100-digit and exact-rational references, never produced by SciOxide.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AlgebraicScalar, Complex, ExecutionContext, NumericScalar, RealScalar};
use scioxide_linalg::*;

fn cast<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn numbers(field: &str) -> Vec<f64> {
    if field == "-" {
        Vec::new()
    } else {
        field.split(',').map(|s| s.parse().unwrap()).collect()
    }
}
fn phase<S: NumericScalar>(i: usize) -> S {
    if S::IS_REAL {
        S::ONE
    } else {
        match i % 4 {
            0 => S::ONE,
            1 => S::from_components(S::Real::ZERO, S::Real::ONE),
            2 => -S::ONE,
            _ => S::from_components(S::Real::ZERO, -S::Real::ONE),
        }
    }
}
fn matrix<S: NumericScalar>(n: usize, entries: &[f64]) -> Matrix<S> {
    rectangular(n, n, entries)
}
fn rectangular<S: NumericScalar>(rows: usize, cols: usize, entries: &[f64]) -> Matrix<S> {
    Matrix::from_vec(
        rows,
        cols,
        entries
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                phase::<S>(i / cols) * S::from_real(cast(x)) * phase::<S>(i % cols).conj()
            })
            .collect(),
    )
    .unwrap()
}
use crate::support::comparisons::within_labeled as close;
fn references<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
    S::Real: NumericScalar<Real = S::Real>,
{
    let mut cases = 0;
    for line in include_str!("../data/dense.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let width: usize = fields[2].parse().unwrap();
        if width != core::mem::size_of::<S::Real>() * 8 {
            continue;
        }
        cases += 1;
        assert_eq!(fields.len(), 7);
        let n: usize = fields[3].parse().unwrap();
        let input = numbers(fields[4]);
        let expected = numbers(fields[5]);
        let extra = numbers(fields[6]);
        let a = matrix::<S>(n, &input);
        let eps = <S::Real as RealScalar>::EPSILON;
        let bound = cast::<S::Real>(128. * n as f64) * eps;
        let options = LinalgExecutionOptions::new();
        match fields[0] {
            "svd" => {
                let values = SvdValues::compute(&a).unwrap();
                let thin = ThinSvdDecomposition::compute(&a).unwrap();
                let full = SvdDecomposition::compute(&a).unwrap();
                for values in [
                    values.singular_values(),
                    thin.singular_values(),
                    full.singular_values(),
                ] {
                    assert_eq!(values.len(), n);
                    for (&actual, &reference) in values.iter().zip(&expected) {
                        // Graded triangular inputs preserve relative accuracy;
                        // Hilbert forward error includes the independent condition.
                        let conditioning = if extra[1] == 1. { 1. } else { extra[0] };
                        close::<S::Real>(
                            actual,
                            cast(reference),
                            bound * cast::<S::Real>(reference * conditioning),
                            fields[1],
                        );
                        assert!(actual > S::Real::ZERO);
                    }
                }
            }
            "hermitian" => {
                let values = scioxide_linalg::HermitianEigenvaluesResult::<S::Real>::compute(
                    &a,
                    scioxide_linalg::HermitianInput::Exact,
                )
                .unwrap();
                let vectors = HermitianEigenDecomposition::decompose(
                    &a,
                    scioxide_linalg::HermitianInput::Exact,
                )
                .unwrap();
                for values in [values.eigenvalues(), vectors.eigenvalues()] {
                    for (&actual, &reference) in values.iter().zip(&expected) {
                        let scale = a
                            .as_slice()
                            .iter()
                            .fold(S::Real::ZERO, |scale, x| scale.max(x.abs()));
                        close::<S::Real>(actual, cast(reference), bound * scale, fields[1]);
                    }
                }
            }
            "general" => {
                let options =
                    GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
                let values =
                    scioxide_linalg::GeneralEigenvaluesResult::<S::Real>::compute_with_options(
                        &a, &options,
                    )
                    .unwrap();
                let vectors =
                    GeneralEigenDecomposition::<S::Real>::decompose_with_options(&a, &options)
                        .unwrap();
                for values in [values.eigenvalues(), vectors.eigenvalues()] {
                    let mut remaining: Vec<_> = expected
                        .chunks_exact(2)
                        .map(|parts| {
                            Complex::new(cast::<S::Real>(parts[0]), cast::<S::Real>(parts[1]))
                        })
                        .collect();
                    let scale = a
                        .as_slice()
                        .iter()
                        .fold(S::Real::ZERO, |scale, x| scale.max(x.abs()));
                    for actual in values {
                        let (index, _) = remaining
                            .iter()
                            .enumerate()
                            .map(|(index, reference)| (index, (*actual - *reference).norm()))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        let reference = remaining.swap_remove(index);
                        close(
                            *actual,
                            reference,
                            cast::<S::Real>(8.0 * n as f64) * eps * scale,
                            fields[1],
                        );
                    }
                    assert!(remaining.is_empty());
                }
                for (actual, &reference) in vectors
                    .eigenvalue_condition_estimates()
                    .unwrap()
                    .iter()
                    .zip(&extra)
                {
                    close::<S::Real>(
                        actual.reciprocal().value(),
                        cast(reference),
                        bound,
                        "eigenvalue condition",
                    );
                }
            }
            "ldl" => {
                let factor =
                    LdlDecomposition::factor_with_options(&a, HermitianInput::Exact, &options)
                        .unwrap();
                let inertia = factor.inertia(scioxide_linalg::RankPolicy::Exact).unwrap();
                assert_eq!(
                    [inertia.positive(), inertia.negative(), inertia.zero()],
                    [
                        expected[0] as usize,
                        expected[1] as usize,
                        expected[2] as usize
                    ]
                );
                assert_eq!(
                    factor.pivot_blocks().map(|b| b.size()).collect::<Vec<_>>(),
                    [2, 1, 1]
                );
                assert_eq!(factor.permutation(), [0, 1, 2, 3]);
                for (i, &d) in factor.diagonal().iter().enumerate() {
                    close(
                        d,
                        S::from_real(cast(extra[i])),
                        bound * cast(extra[i].abs()),
                        "D diagonal",
                    );
                }
                for (i, &d) in factor.off_diagonal().iter().enumerate() {
                    let reference = if i == 0 {
                        phase::<S>(1) * S::from_real(cast(extra[n + i]))
                    } else {
                        S::ZERO
                    };
                    close(d, reference, bound, "D off diagonal");
                }
            }
            "lu" | "cholesky" | "toeplitz" => {
                let rhs_start = usize::from(fields[0] != "toeplitz");
                let b = Vector::from_vec(
                    extra[rhs_start..]
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| phase::<S>(i) * S::from_real(cast(v)))
                        .collect(),
                );
                let solution = match fields[0] {
                    "lu" => LuDecomposition::factor_with_options(&a, &options)
                        .unwrap()
                        .solve(&b)
                        .unwrap(),
                    "cholesky" => CholeskyDecomposition::factor_with_options(
                        &a,
                        HermitianInput::Exact,
                        CholeskyPivoting::None,
                        &options,
                    )
                    .unwrap()
                    .solve(&b)
                    .unwrap(),
                    _ => {
                        let column: Vec<_> = (0..n).map(|i| a.as_slice()[i * n]).collect();
                        let row = &a.as_slice()[..n];
                        let plan = StructuredSolvePlan::new(
                            StructuredMatrixView::toeplitz(&column, row).unwrap(),
                            &StructuredSolveOptions::new(),
                        )
                        .unwrap();
                        let mut x = Vector::zeros(n).unwrap();
                        plan.solve_into(
                            StructuredOperation::Normal,
                            &b,
                            x.view_mut(),
                            &mut plan
                                .workspace(
                                    &scioxide_linalg::StructuredWorkspaceRequest::solve(
                                        plan.order(),
                                        &scioxide_linalg::LinalgExecutionOptions::new(),
                                    )
                                    .with_dense_fallback(),
                                )
                                .unwrap(),
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                        x
                    }
                };
                let condition = if fields[0] == "toeplitz" {
                    4.
                } else {
                    extra[0]
                };
                assert!(
                    bound * cast::<S::Real>(condition) < cast(0.125),
                    "the forward-error allowance must reject a zero solution"
                );
                for (i, (&actual, &reference)) in
                    solution.as_slice().iter().zip(&expected).enumerate()
                {
                    close(
                        actual,
                        phase::<S>(i) * S::from_real(cast(reference)),
                        bound * cast::<S::Real>(condition * reference.abs().max(1.)),
                        fields[1],
                    );
                }
            }
            "schur" => {
                let factor = SchurDecomposition::decompose(&a).unwrap();
                let selection: Vec<_> = factor
                    .eigenvalues()
                    .iter()
                    .map(|v| v.re > cast(2.5))
                    .collect();
                let (reordered, report) = factor.reorder(&selection).unwrap();
                assert_eq!(report.selected_dimension(), 2);
                let basis = reordered.invariant_subspace(2).unwrap();
                let reference = matrix::<S>(n, &expected);
                for i in 0..n {
                    for j in 0..n {
                        let projector = (0..2).fold(S::ZERO, |sum, k| {
                            sum + basis.get(i, k).unwrap() * basis.get(j, k).unwrap().conj()
                        });
                        close(
                            projector,
                            reference.as_slice()[i * n + j],
                            bound,
                            "selected projector",
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    assert_eq!(cases, 13, "every dense reference family must be exercised");
    controls::<S>();
}
fn controls<S: NumericScalar>() {
    let mut cases = 0;
    for line in include_str!("../data/riccati.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 15);
        let width = core::mem::size_of::<S::Real>() * 8;
        if fields[2].parse::<usize>().unwrap() != width {
            continue;
        }
        cases += 1;
        let n = fields[3].parse().unwrap();
        let m = fields[4].parse().unwrap();
        let descriptor = fields[5] == "1";
        let a = matrix::<S>(n, &numbers(fields[6]));
        let b = rectangular::<S>(n, m, &numbers(fields[7]));
        let q = matrix::<S>(n, &numbers(fields[8]));
        let r = matrix::<S>(m, &numbers(fields[9]));
        let e = matrix::<S>(n, &numbers(fields[10]));
        let s = rectangular::<S>(n, m, &numbers(fields[11]));
        let x = matrix::<S>(n, &numbers(fields[12]));
        let gain = rectangular::<S>(m, n, &numbers(fields[13]));
        let spectrum = numbers(fields[14]);
        let kind = if fields[0] == "care" {
            RiccatiKind::Continuous
        } else {
            RiccatiKind::Discrete
        };
        let reference_margin = spectrum
            .chunks_exact(2)
            .map(|root| {
                let magnitude = root[0].hypot(root[1]);
                if kind == RiccatiKind::Continuous {
                    -root[0] / magnitude.max(1.)
                } else {
                    (1. - magnitude) / magnitude.max(1.)
                }
            })
            .fold(f64::INFINITY, f64::min);
        assert!(reference_margin > 0.);
        let problem = RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: descriptor.then_some(e.view()),
            s: descriptor.then_some(s.view()),
        };
        let scalar = fields[1] == "scalar";
        let label = format!(
            "{} {} n={n} width={width} descriptor={descriptor} a={:?} b={:?} q={:?} r={:?}",
            fields[0],
            fields[1],
            a.as_slice(),
            b.as_slice(),
            q.as_slice(),
            r.as_slice()
        );
        for verify in [Estimation::Skip, Estimation::Estimate] {
            if scalar && verify == Estimation::Estimate {
                continue;
            }
            let plan = RiccatiPlan::new(
                problem,
                kind,
                RiccatiOptions::new().with_closed_loop_verification(verify),
            )
            .map_err(|error| (&label, error))
            .unwrap();
            let result = plan.solve().map_err(|error| (&label, error)).unwrap();
            assert_eq!(
                result.report().termination(),
                DirectSolveTermination::Accepted,
                "{label}: {:?}",
                result.report()
            );
            let eps = <S::Real as RealScalar>::EPSILON;
            if scalar {
                // u=epsilon/2. No absolute-error floor may hide small solutions.
                let bound = cast::<S::Real>(500.) * eps;
                close(
                    result.solution().unwrap().as_slice()[0],
                    x.as_slice()[0],
                    bound * x.as_slice()[0].abs(),
                    &label,
                );
                close(
                    result.gain().unwrap().as_slice()[0],
                    gain.as_slice()[0],
                    bound * gain.as_slice()[0].abs(),
                    &label,
                );
            } else {
                // Coordinate-normalized errors expose small graded entries. The
                // near-boundary references allow sensitivity proportional to 1/gap.
                let sensitivity = if fields[1] == "near_boundary" {
                    1. / reference_margin
                } else {
                    1.
                };
                let bound = cast::<S::Real>(128. * n as f64 * sensitivity) * eps;
                for row in 0..n {
                    for col in 0..n {
                        let scale = (x.as_slice()[row * n + row].abs()
                            * x.as_slice()[col * n + col].abs())
                        .sqrt();
                        close(
                            result.solution().unwrap().as_slice()[row * n + col],
                            x.as_slice()[row * n + col],
                            bound * scale,
                            &label,
                        );
                    }
                }
                for row in 0..m {
                    for col in 0..n {
                        let scale = (x.as_slice()[col * n + col].abs()
                            / r.as_slice()[row * m + row].abs())
                        .sqrt();
                        close(
                            result.gain().unwrap().as_slice()[row * n + col],
                            gain.as_slice()[row * n + col],
                            bound * scale.max(gain.as_slice()[row * n + col].abs()),
                            &label,
                        );
                    }
                }
                assert!(
                    result.report().backward_error() <= cast::<S::Real>(4096. * n as f64) * eps,
                    "{label}: {:?}",
                    result.report()
                );
                if let Some(margin) = result.report().stability_margin() {
                    assert!(margin > S::Real::ZERO);
                    // The spectrum responds to a matrix-norm perturbation;
                    // converting the entrywise forward bound costs a factor n.
                    let spectral_bound = cast::<S::Real>(n as f64) * bound;
                    assert!(
                        (margin - cast::<S::Real>(reference_margin)).abs() <= spectral_bound,
                        "{label}: margin {margin:?}"
                    );
                }
            }
        }
    }
    let expected = if core::mem::size_of::<S::Real>() == 8 {
        2528
    } else {
        28
    };
    assert_eq!(
        cases, expected,
        "all scalar, original, graded and near-boundary families must be exercised"
    );
}
#[test]
fn real32() {
    references::<f32>();
}
#[test]
fn real64() {
    references::<f64>();
}
#[test]
fn complex32() {
    references::<Complex<f32>>();
}
#[test]
fn complex64() {
    references::<Complex<f64>>();
}
