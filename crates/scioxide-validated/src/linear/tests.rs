use super::*;
use alloc::vec;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ExecutionContext, LinearOperator, SciOxideErrorKind,
    SolveBudgets,
};

const fn options() -> ValidatedOptions {
    ValidatedOptions::new()
}
const fn seq() -> ExecutionContext<'static> {
    ExecutionContext::sequential()
}
use scioxide_linalg::{LuDecomposition, Matrix, MatrixView};
use scioxide_sparse::CsrMatrix;

fn value<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(re).unwrap(),
        S::Real::try_from_f64_lossy(im).unwrap(),
    )
}

fn scalar_family<S: NumericScalar>() {
    let lo = [value::<S>(1.75, -0.125)];
    let hi = [value::<S>(2.25, 0.125)];
    let inverse = [value::<S>(0.5, 0.)];
    let op = DenseIntervalOperator::new(
        MatrixView::from_slice(1, 1, &lo).unwrap(),
        MatrixView::from_slice(1, 1, &hi).unwrap(),
    )
    .unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &inverse).unwrap()).unwrap();
    let plan = LinearVerifier::new(&op, &r, options()).unwrap();
    let mut w = plan.workspace(1).unwrap();
    let mut cert = LinearCertificate::new(1).unwrap();
    let rhs = [interval_enclosure(value::<S>(0.875, -0.0625), value(1.125, 0.0625)).unwrap()];
    assert!(
        plan.solve_into(&rhs, cert.view_mut(), &mut w, seq())
            .unwrap()
            .is_verified()
    );
    let enclosure = cert.enclosure().unwrap()[0];
    // Dyadic coefficient grid; ordinary division is only a sampled witness here.
    for a in [1.75, 2., 2.25] {
        for b in [0.875, 1., 1.125] {
            let x = value::<S>(b, 0.) / value(a, 0.);
            assert!(enclosure.re.contains(x.real_component()));
        }
    }
    assert!(
        plan.verify_scoped(&rhs, cert.view(), &mut w, seq())
            .unwrap()
            .is_verified()
    );
    cert.radii.fill(Complex::new(S::Real::ZERO, S::Real::ZERO));
    assert_eq!(
        plan.verify_scoped(&rhs, cert.view(), &mut w, seq())
            .unwrap()
            .status(),
        LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow)
    );
}
#[test]
fn interval_families() {
    scalar_family::<f32>();
    scalar_family::<f64>();
    scalar_family::<Complex<f32>>();
    scalar_family::<Complex<f64>>();
}

#[test]
fn exact_residual_survives_underflow_and_cancellation() {
    for (a, x, b, lo, hi) in [
        (vec![1e308, 1e308], vec![2., -2.], 0., 0., 0.),
        (
            vec![f64::from_bits(1)],
            vec![0.5],
            0.,
            -f64::from_bits(1),
            0.,
        ),
        (vec![1., 1., -1.], vec![1e16, 1., 1e16], 1., 0., 0.),
        (
            vec![f64::MAX],
            vec![-1.],
            2.0_f64.powi(969),
            f64::MAX,
            f64::INFINITY,
        ),
        (
            vec![-f64::MAX],
            vec![-1.],
            -2.0_f64.powi(969),
            -f64::INFINITY,
            -f64::MAX,
        ),
    ] {
        let op =
            DenseIntervalOperator::point(MatrixView::from_slice(1, a.len(), &a).unwrap()).unwrap();
        let input: Vec<_> = x.iter().map(|&v| point_enclosure(-v).unwrap()).collect();
        let rhs = [point_enclosure(b).unwrap()];
        let mut out = [zero().unwrap()];
        let mut scratch = [zero().unwrap()];
        op.apply_enclosure_into(&input, Some(&rhs), &mut out, &mut scratch)
            .unwrap();
        assert_eq!(
            out[0].re.bounds(),
            Some((lo.is_finite().then_some(lo), hi.is_finite().then_some(hi)))
        );
    }
}

fn point_residual_matches_endpoint_reduction<S: NumericScalar>() {
    let n = 32;
    let mut state = 2125_u64;
    let mut component = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / 9007199254740992. - 0.5
    };
    let entries: Vec<_> = (0..n * n)
        .map(|_| value::<S>(component(), component()))
        .collect();
    let matrix = Matrix::from_vec(n, n, entries.clone()).unwrap();
    let copy = Matrix::from_vec(n, n, entries).unwrap();
    let fast = DenseIntervalOperator::point(matrix.view()).unwrap();
    // Separate endpoint storage exercises the general exact interval reduction.
    let reference = DenseIntervalOperator::new(matrix.view(), copy.view()).unwrap();
    let input: Vec<_> = (0..n)
        .map(|_| point_enclosure(value::<S>(component(), component())).unwrap())
        .collect();
    let rhs: Vec<_> = (0..n)
        .map(|_| point_enclosure(value::<S>(component(), component())).unwrap())
        .collect();
    let mut actual = vec![zero().unwrap(); n];
    let mut expected = actual.clone();
    let mut scratch = actual.clone();
    fast.apply_enclosure_into(&input, Some(&rhs), &mut actual, &mut scratch)
        .unwrap();
    reference
        .apply_enclosure_into(&input, Some(&rhs), &mut expected, &mut scratch)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn compact_point_residuals_retain_exact_outward_endpoints() {
    point_residual_matches_endpoint_reduction::<f32>();
    point_residual_matches_endpoint_reduction::<f64>();
    point_residual_matches_endpoint_reduction::<Complex<f32>>();
    point_residual_matches_endpoint_reduction::<Complex<f64>>();
}
#[test]
fn singular_nearly_singular_and_uncertain_regularity() {
    let r = [1., 0., 0., 1.];
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &r).unwrap()).unwrap();
    for a in [[0., 0., 0., 0.], [1., 1., 1., 1. + f64::EPSILON]] {
        let op = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &a).unwrap()).unwrap();
        let plan = LinearVerifier::new(&op, &inverse, options()).unwrap();
        let mut w = plan.workspace(1).unwrap();
        let zeroes = [0.; 2];
        let radii = [Complex::new(1., 0.); 2];
        let rhs = [point_enclosure(0.).unwrap(); 2];
        assert_eq!(
            plan.verify_scoped(
                &rhs,
                LinearCertificateView {
                    center: &zeroes,
                    radii: &radii
                },
                &mut w,
                seq(),
            )
            .unwrap()
            .status(),
            LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
                LinearProofFailure::RegularityNotProved
            ))
        );
    }
    let lower = [-1.];
    let upper = [1.];
    let unit = [1.];
    let op = DenseIntervalOperator::new(
        MatrixView::from_slice(1, 1, &lower).unwrap(),
        MatrixView::from_slice(1, 1, &upper).unwrap(),
    )
    .unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &unit).unwrap()).unwrap();
    let plan = LinearVerifier::new(&op, &r, options()).unwrap();
    let mut w = plan.workspace(1).unwrap();
    let mut cert = LinearCertificate::new(1).unwrap();
    assert!(
        !plan
            .solve_into(
                &[point_enclosure(0.).unwrap()],
                cert.view_mut(),
                &mut w,
                seq()
            )
            .unwrap()
            .is_verified()
    );
}
#[test]
fn proof_limits_and_output_rollback() {
    let a = [3.];
    let rv = [1. / 3.];
    let op = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &rv).unwrap()).unwrap();
    let plan = LinearVerifier::new(&op, &r, options()).unwrap();
    let bytes = plan.workspace_requirements(1).unwrap().required_bytes();
    let limited = LinearVerifier::new(&op, &r, options().with_byte_limit(bytes - 1)).unwrap();
    assert_eq!(
        limited.workspace(1).err().unwrap().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut w = plan.workspace(1).unwrap();
    let mut cert = LinearCertificate::new(1).unwrap();
    cert.center[0] = 77.;
    cert.radii[0].re = 77.;
    let old = cert.clone();
    let rhs = [point_enclosure(1.).unwrap()];
    for (kind, limit) in [
        (BudgetKind::Evaluation, 1),
        (BudgetKind::Evaluation, 3),
        (BudgetKind::Evaluation, 4),
    ] {
        let budgets = SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap());
        let plan = LinearVerifier::new(&op, &r, options().with_budgets(budgets)).unwrap();
        let report = plan
            .solve_into(&rhs, cert.view_mut(), &mut w, seq())
            .unwrap();
        assert!(!report.is_verified());
        assert_eq!(cert, old);
        assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Stopped);
    }
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let report = plan
        .solve_into(
            &rhs,
            cert.view_mut(),
            &mut w,
            seq().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        report.status(),
        LinearVerificationStatus::Inconclusive(LinearInconclusive::Stopped(_))
    ));
    assert_eq!(cert, old);
    // Preflight rejections name the shape and leave output and counts untouched.
    let error = plan
        .solve_into(&[], cert.view_mut(), &mut w, seq())
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(w.last_attempt().unwrap().epoch().is_none());
    assert_eq!(cert, old);
    let mut short = LinearCertificate::new(2).unwrap();
    assert_eq!(
        plan.solve_into(&rhs, short.view_mut(), &mut w, seq())
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(interval_enclosure(1., 0.).is_err());
    assert!(point_enclosure(f64::NAN).is_err());
    cert.radii[0].re = -1.;
    assert!(
        plan.verify_scoped(&rhs, cert.view(), &mut w, seq())
            .is_err()
    );
    // An undersized workspace grows within the limit, or is rejected beyond it.
    let mut empty = LinearWorkspace::new();
    let mut spare = old;
    let error = limited
        .solve_into(&rhs, spare.view_mut(), &mut empty, seq())
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert!(
        plan.solve_into(&rhs, cert.view_mut(), &mut empty, seq())
            .unwrap()
            .is_verified()
    );
}
#[test]
fn empty_identity() {
    let data: [f64; 0] = [];
    let op = DenseIntervalOperator::point(MatrixView::from_slice(0, 0, &data).unwrap()).unwrap();
    let plan = LinearVerifier::new(&op, &op, options()).unwrap();
    let mut w = plan.workspace(1).unwrap();
    let mut cert = LinearCertificate::new(0).unwrap();
    assert!(
        plan.solve_into(&[], cert.view_mut(), &mut w, seq())
            .unwrap()
            .is_verified()
    );
    let counts = w.last_attempt().unwrap().counts().total().unwrap();
    assert_eq!(counts.total_providers().unwrap(), 0);
    assert_eq!(counts.iterations(), 0);
}

struct Extension<'a> {
    inner: DenseIntervalOperator<'a, f64>,
    mode: u8,
}

#[test]
fn provider_metadata_changes_during_application_preserve_output() {
    use core::sync::atomic::{AtomicBool, Ordering};
    struct Provider<'a> {
        inner: DenseIntervalOperator<'a, f64>,
        changed: AtomicBool,
        mode: u8,
    }
    impl Provider<'_> {
        fn extra(&self, mode: u8) -> usize {
            usize::from(self.mode == mode && self.changed.load(Ordering::Relaxed))
        }
    }
    impl LinearOperator<f64> for Provider<'_> {
        fn rows(&self) -> usize {
            1 + self.extra(0)
        }
        fn cols(&self) -> usize {
            1 + self.extra(1)
        }
        fn apply_workspace_len(&self) -> Result<usize> {
            Ok(1 + self.extra(2))
        }
        fn apply_into(&self, input: &[f64], output: &mut [f64], scratch: &mut [f64]) -> Result<()> {
            self.inner.apply_into(input, output, scratch)
        }
    }
    impl IntervalLinearOperator<f64> for Provider<'_> {
        fn enclosure_workspace_len(&self) -> Result<usize> {
            Ok(1 + self.extra(3))
        }
        fn apply_enclosure_into(
            &self,
            input: &[IntervalBox<f64>],
            addend: Option<&[IntervalBox<f64>]>,
            output: &mut [IntervalBox<f64>],
            scratch: &mut [IntervalBox<f64>],
        ) -> Result<()> {
            self.inner
                .apply_enclosure_into(input, addend, output, scratch)?;
            self.changed.store(true, Ordering::Relaxed);
            Ok(())
        }
    }
    let entries = [1.0];
    let rhs = [point_enclosure(2.0).unwrap()];
    for mode in 0..4 {
        let provider = Provider {
            inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &entries).unwrap())
                .unwrap(),
            changed: AtomicBool::new(false),
            mode,
        };
        let plan = LinearVerifier::new(&provider, &provider, options()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut output = LinearCertificate::new(1).unwrap();
        output.center[0] = 77.0;
        let stored = output.clone();
        let error = plan
            .solve_into(&rhs, output.view_mut(), &mut workspace, seq())
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        assert_eq!(output, stored);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .own()
                .total_providers()
                .unwrap(),
            1
        );
    }
}
impl LinearOperator<f64> for Extension<'_> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<()> {
        match self.mode {
            6 | 7 => {
                y.fill(if self.mode == 6 {
                    f64::NAN
                } else {
                    f64::INFINITY
                });
                Ok(())
            }
            8 => Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "seed overflow",
            )),
            9 => Err(SciOxideError::singular(None)),
            10 => Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "nonfinite seed",
            )),
            11 => Err(invalid(
                scioxide_core::argument!("self"),
                "invalid seed provider contract",
            )),
            12 if x[0] > 1. => Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "seed overflow",
            )),
            _ => self.inner.apply_into(x, y, w),
        }
    }
}
impl IntervalLinearOperator<f64> for Extension<'_> {
    fn enclosure_workspace_len(&self) -> Result<usize> {
        // Mode 4 requests representable but unallocatable scratch.
        if self.mode == 4 {
            return Ok(1 << 56);
        }
        self.inner.enclosure_workspace_len()
    }
    fn apply_enclosure_into(
        &self,
        x: &[IntervalBox<f64>],
        b: Option<&[IntervalBox<f64>]>,
        y: &mut [IntervalBox<f64>],
        w: &mut [IntervalBox<f64>],
    ) -> Result<()> {
        match self.mode {
            1 => Err(invalid(scioxide_core::argument!("x"), "injected failure")),
            2 => Ok(()),
            3 => {
                y.fill(Complex::new(
                    scioxide_core::ExtendedInterval::entire(),
                    zero::<f64>()?.im,
                ));
                Ok(())
            }
            5 if b.is_some_and(|rhs| rhs[0].re.contains(-1.)) => Err(invalid(
                scioxide_core::argument!("rhs"),
                "injected RHS failure",
            )),
            _ => self.inner.apply_enclosure_into(x, b, y, w),
        }
    }
}
#[test]
fn operator_extensions_and_invalid_providers() {
    let a = [2.];
    let b = [point_enclosure(1.).unwrap()];
    let r = [0.5];
    let inv = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
    for mode in 0..4 {
        let extension = Extension {
            inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap(),
            mode,
        };
        let plan = LinearVerifier::new(&extension, &inv, options()).unwrap();
        let mut w = plan.workspace(1).unwrap();
        let mut cert = LinearCertificate::new(1).unwrap();
        cert.center[0] = 77.;
        let result = plan.solve_into(&b, cert.view_mut(), &mut w, seq());
        match mode {
            0 => assert!(result.unwrap().is_verified()),
            3 => assert_eq!(
                result.unwrap().status(),
                LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded)
            ),
            _ => assert!(result.is_err()),
        }
        if mode != 0 {
            assert_eq!(cert.center[0], 77.);
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn executor_equivalence_and_batch_rollback() {
    use core::num::NonZeroUsize;
    use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionPolicy, ThreadExecutor};
    let a = [3.];
    let rv = [0.25];
    let op = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &rv).unwrap()).unwrap();
    let rhs = [
        point_enclosure(1.).unwrap(),
        point_enclosure(2.).unwrap(),
        point_enclosure(4.).unwrap(),
    ];
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let blank = LinearVerificationReport::not_run();
    for limit in [None, Some(1), Some(2), Some(5), Some(10), Some(11)] {
        let mut reference = None;
        for parallel in [false, true] {
            let budgets = match limit {
                Some(limit) => SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap()),
                None => SolveBudgets::unbounded(),
            };
            let policy = if parallel {
                ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())
            } else {
                ExecutionPolicy::Sequential
            };
            let plan = LinearVerifier::new(
                &op,
                &r,
                options()
                    .with_budgets(budgets)
                    .with_execution_policy(policy),
            )
            .unwrap();
            let requirements = plan.workspace_requirements(3).unwrap();
            let mut w = plan.workspace(3).unwrap();
            assert_eq!(
                w.check(&requirements).unwrap(),
                scioxide_core::Capacity::Sufficient
            );
            let mut centers = [77.; 3];
            let mut radii = [Complex::new(77., 0.); 3];
            let mut reports = [blank; 3];
            let execution = if parallel {
                ExecutionContext::new(&pool)
            } else {
                seq()
            };
            plan.solve_block_into(
                BlockVectorView::new(&rhs, 1, 3).unwrap(),
                &mut BlockVectorViewMut::new(&mut centers, 1, 3).unwrap(),
                &mut BlockVectorViewMut::new(&mut radii, 1, 3).unwrap(),
                &mut reports,
                &mut w,
                execution,
            )
            .unwrap();
            if limit.is_some_and(|limit| limit < 11) {
                assert!(reports.iter().any(|r| !r.is_verified()));
                assert_eq!(centers, [77.; 3]);
                assert_eq!(radii, [Complex::new(77., 0.); 3]);
            } else {
                assert!(reports.iter().all(LinearVerificationReport::is_verified));
            }
            let counts = w.last_attempt().unwrap().counts();
            if let Some((c, r, p, work)) = reference {
                assert_eq!(centers, c);
                assert_eq!(radii, r);
                assert_eq!(reports, p);
                assert_eq!(counts, work);
            } else {
                reference = Some((centers, radii, reports, counts));
            }
        }
    }
    // Shape rejections precede numerical work and leave every output unchanged.
    let plan = LinearVerifier::new(&op, &r, options()).unwrap();
    let mut w = plan.workspace(3).unwrap();
    let mut centers = [77.; 3];
    let mut radii = [Complex::new(77., 0.); 3];
    let mut reports = [blank; 2];
    let error = plan
        .solve_block_into(
            BlockVectorView::new(&rhs, 1, 3).unwrap(),
            &mut BlockVectorViewMut::new(&mut centers, 1, 3).unwrap(),
            &mut BlockVectorViewMut::new(&mut radii, 1, 3).unwrap(),
            &mut reports,
            &mut w,
            seq(),
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(centers, [77.; 3]);
    assert_eq!(reports, [blank; 2]);
}

fn rational_fixtures<S: NumericScalar>(width: &str, complex: &str) {
    let mut cases = 0;
    for line in include_str!("../../tests/data/verified-linear.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width || f[1] != complex {
            continue;
        }
        let n: usize = f[3].parse().unwrap();
        let mut cursor = 4;
        let decode = |word: &str| {
            let b = u64::from_str_radix(word, 16).unwrap();
            S::Real::try_from_f64_lossy(if width == "32" {
                f64::from(f32::from_bits(b as u32))
            } else {
                f64::from_bits(b)
            })
            .unwrap()
        };
        let mut read_scalar = || {
            let s = S::from_components(decode(f[cursor]), decode(f[cursor + 1]));
            cursor += 2;
            s
        };
        let entries: Vec<_> = (0..n * n).map(|_| read_scalar()).collect();
        let rhs: Vec<_> = (0..n)
            .map(|_| point_enclosure(read_scalar()).unwrap())
            .collect();
        let matrix = Matrix::from_vec(n, n, entries.clone()).unwrap();
        let lu = LuDecomposition::factor(&matrix).unwrap();
        let inverse = LuIntervalInverse::from_dense(&lu, options()).unwrap();
        let dense = DenseIntervalOperator::point(matrix.view()).unwrap();
        let sparse = CsrMatrix::from_raw(
            n,
            n,
            (0..=n).map(|i| i * n).collect(),
            (0..n).flat_map(|_| 0..n).collect(),
            entries,
        )
        .unwrap();
        let sparse = SparseIntervalOperator::point(&sparse).unwrap();
        for system in [&dense as &(dyn IntervalLinearOperator<S> + Sync), &sparse] {
            let plan = LinearVerifier::new(system, &inverse, options()).unwrap();
            let mut work = plan.workspace(1).unwrap();
            let mut cert = LinearCertificate::new(n).unwrap();
            let report = plan
                .solve_into(&rhs, cert.view_mut(), &mut work, seq())
                .unwrap();
            assert!(
                report.is_verified(),
                "fixture {} {width} {complex}: {report:?}",
                f[2]
            );
            let boxes = cert.enclosure().unwrap();
            for (i, b) in boxes.iter().enumerate() {
                let k = cursor + 4 * i;
                assert!(
                    b.re.contains(decode(f[k]))
                        && b.re.contains(decode(f[k + 1]))
                        && b.im.contains(decode(f[k + 2]))
                        && b.im.contains(decode(f[k + 3])),
                    "solution fixture {} row {i}",
                    f[2]
                );
            }
            let x0: Vec<_> = (0..n)
                .map(|i| {
                    point_enclosure(-value::<S>((i + 1) as f64 / 4., -(i as f64) / 8.)).unwrap()
                })
                .collect();
            let mut output = vec![zero().unwrap(); n];
            let mut scratch = vec![zero().unwrap(); system.enclosure_workspace_len().unwrap()];
            system
                .apply_enclosure_into(&x0, Some(&rhs), &mut output, &mut scratch)
                .unwrap();
            for (i, b) in output.iter().enumerate() {
                let k = cursor + 4 * n + 4 * i;
                assert_eq!(
                    b.re.bounds(),
                    Some((Some(decode(f[k])), Some(decode(f[k + 1]))))
                );
                assert_eq!(
                    b.im.bounds(),
                    Some((Some(decode(f[k + 2])), Some(decode(f[k + 3]))))
                );
            }
        }
        cases += 1;
    }
    assert_eq!(cases, 24);
}
#[test]
fn independent_rational_real32() {
    rational_fixtures::<f32>("32", "0");
}
#[test]
fn independent_rational_real64() {
    rational_fixtures::<f64>("64", "0");
}
#[test]
fn independent_rational_complex32() {
    rational_fixtures::<Complex<f32>>("32", "1");
}
#[test]
fn independent_rational_complex64() {
    rational_fixtures::<Complex<f64>>("64", "1");
}

#[test]
fn extreme_diagonal_scaling_and_complex_pivots() {
    for exponent in [-1000, -100, 100, 1000] {
        let scale = 2.0_f64.powi(exponent);
        let a = Matrix::from_vec(1, 1, vec![Complex::new(scale, scale)]).unwrap();
        let b = [point_enclosure(Complex::new(scale, scale)).unwrap()];
        let lu = LuDecomposition::factor(&a).unwrap();
        let r = LuIntervalInverse::from_dense(&lu, options()).unwrap();
        let op = DenseIntervalOperator::point(a.view()).unwrap();
        let plan = LinearVerifier::new(&op, &r, options()).unwrap();
        let mut w = plan.workspace(1).unwrap();
        let mut cert = LinearCertificate::new(1).unwrap();
        assert!(
            plan.solve_into(&b, cert.view_mut(), &mut w, seq())
                .unwrap()
                .is_verified()
        );
        let enclosure = cert.enclosure().unwrap()[0];
        assert!(enclosure.re.contains(1.) && enclosure.im.contains(0.));
    }
}

struct FaultyExecutor(bool);
impl scioxide_core::Executor for FaultyExecutor {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<()> {
        if self.0 {
            scioxide_core::SequentialExecutor.execute(context, plan, operation)?;
            scioxide_core::SequentialExecutor.execute(context, plan, operation)
        } else {
            scioxide_core::SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
        }
    }
}
#[test]
fn dishonest_executor_cannot_publish_certificates() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let a = [2.];
    let rv = [0.5];
    let op = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &rv).unwrap()).unwrap();
    let plan = LinearVerifier::new(&op, &r, options()).unwrap();
    let rhs = [point_enclosure(1.).unwrap(); 2];
    let blank = LinearVerificationReport::not_run();
    for duplicate in [false, true] {
        let executor = FaultyExecutor(duplicate);
        let mut work = plan.workspace(2).unwrap();
        let mut centers = [77.; 2];
        let mut radii = [Complex::new(77., 0.); 2];
        let mut reports = [blank; 2];
        assert!(
            plan.solve_block_into(
                BlockVectorView::new(&rhs, 1, 2).unwrap(),
                &mut BlockVectorViewMut::new(&mut centers, 1, 2).unwrap(),
                &mut BlockVectorViewMut::new(&mut radii, 1, 2).unwrap(),
                &mut reports,
                &mut work,
                ExecutionContext::new(&executor),
            )
            .is_err()
        );
        assert_eq!(centers, [77.; 2]);
        assert_eq!(radii, [Complex::new(77., 0.); 2]);
        assert_eq!(reports, [blank; 2]);
        assert_eq!(
            work.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
}

fn shared_sweep_accounting<S: NumericScalar>() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut, EvaluationRole};
    let n = 3;
    let a = Matrix::from_vec(
        n,
        n,
        vec![
            value::<S>(2., 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            value(3., 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            value(4., 0.),
        ],
    )
    .unwrap();
    let r = Matrix::from_vec(
        n,
        n,
        vec![
            value::<S>(0.4, 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            value(0.3, 0.),
            S::ZERO,
            S::ZERO,
            S::ZERO,
            value(0.2, 0.),
        ],
    )
    .unwrap();
    let inverse = DenseIntervalOperator::point(r.view()).unwrap();
    let system = DenseIntervalOperator::point(a.view()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let rhs: Vec<_> = (0..n * 8)
        .map(|i| point_enclosure(value::<S>(1. + i as f64, 0.25)).unwrap())
        .collect();
    let mut workspace = plan.workspace(8).unwrap();
    let mut certificate = LinearCertificate::new(n).unwrap();
    let report = plan
        .solve_into(&rhs[..n], certificate.view_mut(), &mut workspace, seq())
        .unwrap();
    assert!(report.is_verified());
    let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
    assert_eq!(counts.total_providers().unwrap(), 2 * n + 3);
    assert_eq!(counts.provider(EvaluationRole::LinearOperator), n);
    assert_eq!(counts.provider(EvaluationRole::Preconditioner), n + 1);
    assert_eq!(counts.provider(EvaluationRole::Residual), 1);
    assert_eq!(counts.provider(EvaluationRole::LinearSolve), 1);
    assert_eq!(counts.iterations(), n);
    let bounds = report.bounds().unwrap();
    assert!(
        bounds.preconditioner().is_none()
            && bounds.inverse().is_none()
            && bounds.condition().is_none()
    );
    let fresh = plan.verify(&rhs[..n], certificate.view()).unwrap();
    assert!(fresh.is_verified());
    assert!(fresh.residual().unwrap() <= report.residual().unwrap());

    let center = certificate.center.clone();
    for proposed in [false, true] {
        let report = if proposed {
            plan.propose_into(
                &rhs[..n],
                &center,
                certificate.view_mut(),
                &mut workspace,
                seq(),
            )
            .unwrap()
        } else {
            plan.verify_scoped(&rhs[..n], certificate.view(), &mut workspace, seq())
                .unwrap()
        };
        assert!(report.is_verified(), "{report:?}");
        let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
        assert_eq!(counts.total_providers().unwrap(), 2 * n + 2);
        assert_eq!(counts.iterations(), n);
    }
    let bounds = plan
        .bounds_scoped(&mut workspace, seq())
        .unwrap()
        .bounds()
        .unwrap();
    let blank = LinearVerificationReport::not_run();
    assert!(
        bounds.preconditioner().is_some()
            && bounds.inverse().is_some()
            && bounds.condition().is_some()
    );
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .total_providers()
            .unwrap(),
        3 * n
    );

    for count in [1, 8] {
        let mut centers = vec![S::ZERO; n * count];
        let mut radii = vec![Complex::new(S::Real::ZERO, S::Real::ZERO); n * count];
        let mut reports = vec![blank; count];
        {
            plan.solve_block_into(
                BlockVectorView::new(&rhs[..n * count], n, count).unwrap(),
                &mut BlockVectorViewMut::new(&mut centers, n, count).unwrap(),
                &mut BlockVectorViewMut::new(&mut radii, n, count).unwrap(),
                &mut reports,
                &mut workspace,
                seq(),
            )
            .unwrap();
        };
        assert!(
            reports.iter().all(LinearVerificationReport::is_verified),
            "{reports:?}"
        );
        let counts = workspace.last_attempt().unwrap().counts();
        assert_eq!(counts.own().total_providers().unwrap(), 2 * n);
        assert_eq!(counts.nested().total_providers().unwrap(), 3 * count);
        assert_eq!(
            counts.total().unwrap().total_providers().unwrap(),
            2 * n + 3 * count
        );
        assert_eq!(counts.total().unwrap().iterations(), n);
        for (index, report) in reports.iter().enumerate() {
            assert_eq!(report.bounds(), reports[0].bounds());
            assert!(report.bounds().unwrap().preconditioner().is_none());
            let view = LinearCertificateView {
                center: &centers[index * n..(index + 1) * n],
                radii: &radii[index * n..(index + 1) * n],
            };
            assert!(
                plan.verify(&rhs[index * n..(index + 1) * n], view)
                    .unwrap()
                    .is_verified()
            );
        }
    }
}

#[test]
fn one_sweep_for_every_scalar_and_rhs_count() {
    shared_sweep_accounting::<f32>();
    shared_sweep_accounting::<f64>();
    shared_sweep_accounting::<Complex<f32>>();
    shared_sweep_accounting::<Complex<f64>>();
}

#[test]
fn external_nonuniform_radii_use_the_streamed_componentwise_image() {
    let a = [
        Complex::new(1., 0.),
        Complex::new(0., -0.5),
        Complex::new(0., 0.),
        Complex::new(1., 0.),
    ];
    let r = [
        Complex::new(1., 0.),
        Complex::new(0., 0.),
        Complex::new(0., 0.),
        Complex::new(1., 0.),
    ];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &a).unwrap()).unwrap();
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &r).unwrap()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let rhs = [
        point_enclosure(Complex::new(1., 0.)).unwrap(),
        point_enclosure(Complex::new(0., 1. / 128.)).unwrap(),
    ];
    let center = [Complex::new(0., 0.); 2];
    // The scalar max-radius check would reject this valid narrow box. The
    // imaginary off-diagonal defect contributes only 1/256 to the first real row.
    let mut radii = [
        Complex::new(1. + 1. / 128., 0.),
        Complex::new(0., 1. / 128.),
    ];
    assert!(
        plan.verify(
            &rhs,
            LinearCertificateView {
                center: &center,
                radii: &radii
            }
        )
        .unwrap()
        .is_verified()
    );
    radii[0].re = 1.;
    assert_eq!(
        plan.verify(
            &rhs,
            LinearCertificateView {
                center: &center,
                radii: &radii
            }
        )
        .unwrap()
        .status(),
        LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow)
    );
}

#[test]
fn shared_sweep_budget_boundaries_and_batch_rollback() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let a = [2., 0., 0., 2.];
    let r = [0.4, 0., 0., 0.4];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &a).unwrap()).unwrap();
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &r).unwrap()).unwrap();
    let blank = LinearVerificationReport::not_run();
    let rhs = [point_enclosure(1.).unwrap(); 6];
    for (kind, limit) in [
        (BudgetKind::Evaluation, 1),
        (BudgetKind::Evaluation, 3),
        (BudgetKind::Evaluation, 4),
        (BudgetKind::Evaluation, 12),
        (BudgetKind::Evaluation, 13),
        (BudgetKind::Iteration, 1),
        (BudgetKind::Iteration, 2),
    ] {
        let budgets = SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap());
        let plan = LinearVerifier::new(&system, &inverse, options().with_budgets(budgets)).unwrap();
        let mut workspace = plan.workspace(3).unwrap();
        let mut centers = [77.; 6];
        let mut radii = [Complex::new(77., 0.); 6];
        let mut reports = [blank; 3];
        plan.solve_block_into(
            BlockVectorView::new(&rhs, 2, 3).unwrap(),
            &mut BlockVectorViewMut::new(&mut centers, 2, 3).unwrap(),
            &mut BlockVectorViewMut::new(&mut radii, 2, 3).unwrap(),
            &mut reports,
            &mut workspace,
            seq(),
        )
        .unwrap();
        let success = limit == 13 || (kind == BudgetKind::Iteration && limit == 2);
        assert_eq!(
            reports.iter().all(LinearVerificationReport::is_verified),
            success
        );
        if !success {
            assert_eq!(centers, [77.; 6]);
            assert_eq!(radii, [Complex::new(77., 0.); 6]);
        }
        if limit == 12 {
            assert!(reports.iter().any(LinearVerificationReport::is_verified));
            assert!(reports.iter().any(|r| r.status()
                == LinearVerificationStatus::Inconclusive(LinearInconclusive::EvaluationLimit)));
        }
        if kind == BudgetKind::Iteration && limit == 1 {
            assert!(reports.iter().all(|r| r.status()
                == LinearVerificationStatus::Inconclusive(LinearInconclusive::IterationLimit)));
        }
        let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
        if kind == BudgetKind::Evaluation {
            assert!(counts.total_providers().unwrap() <= limit);
        } else {
            assert!(counts.iterations() <= limit);
        }
    }
}

#[test]
fn empty_block_skips_all_shared_work() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let a = [2.];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let plan = LinearVerifier::new(&system, &system, options()).unwrap();
    let requirements = plan.workspace_requirements(0).unwrap();
    assert_eq!(requirements.required_bytes(), 0);
    let mut workspace = plan.workspace(0).unwrap();
    plan.solve_block_into(
        BlockVectorView::new(&[], 1, 0).unwrap(),
        &mut BlockVectorViewMut::new(&mut [], 1, 0).unwrap(),
        &mut BlockVectorViewMut::new(&mut [], 1, 0).unwrap(),
        &mut [],
        &mut workspace,
        seq(),
    )
    .unwrap();
    let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
    assert_eq!(counts.total_providers().unwrap(), 0);
    assert_eq!(counts.iterations(), 0);
}

#[test]
fn shared_rejection_and_late_rhs_error_preserve_block_outputs() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let r = [0.5];
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
    for (coefficient, mode) in [(0., 0), (2., 1), (2., 5)] {
        let a = [coefficient];
        let system = Extension {
            inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap(),
            mode,
        };
        let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
        let rhs = [point_enclosure(1.).unwrap(), point_enclosure(-1.).unwrap()];
        let blank = LinearVerificationReport::not_run();
        let mut workspace = plan.workspace(2).unwrap();
        let mut centers = [77.; 2];
        let mut radii = [Complex::new(77., 0.); 2];
        let mut reports = [blank; 2];
        let result = plan.solve_block_into(
            BlockVectorView::new(&rhs, 1, 2).unwrap(),
            &mut BlockVectorViewMut::new(&mut centers, 1, 2).unwrap(),
            &mut BlockVectorViewMut::new(&mut radii, 1, 2).unwrap(),
            &mut reports,
            &mut workspace,
            seq(),
        );
        assert_eq!(centers, [77.; 2]);
        assert_eq!(radii, [Complex::new(77., 0.); 2]);
        if mode == 0 {
            result.unwrap();
            assert!(reports.iter().all(|report| report.status()
                == LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
                    LinearProofFailure::RegularityNotProved
                ))));
            assert_eq!(
                workspace
                    .last_attempt()
                    .unwrap()
                    .counts()
                    .total()
                    .unwrap()
                    .total_providers()
                    .unwrap(),
                2
            );
        } else {
            assert!(result.is_err());
            assert_eq!(reports, [blank; 2]);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
        }
    }
}

/// Test provider whose matrix may change between calls, never during one proof.
struct ChangingMatrix {
    coefficient: core::sync::atomic::AtomicU64,
    sweeps: core::sync::atomic::AtomicUsize,
}
impl LinearOperator<f64> for ChangingMatrix {
    fn rows(&self) -> usize {
        1
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        Ok(1)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], scratch: &mut [f64]) -> Result<()> {
        let a = [f64::from_bits(
            self.coefficient.load(core::sync::atomic::Ordering::Relaxed),
        )];
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a)?)?.apply_into(x, y, scratch)
    }
}
impl IntervalLinearOperator<f64> for ChangingMatrix {
    fn enclosure_workspace_len(&self) -> Result<usize> {
        Ok(1)
    }
    fn apply_enclosure_into(
        &self,
        x: &[IntervalBox<f64>],
        b: Option<&[IntervalBox<f64>]>,
        y: &mut [IntervalBox<f64>],
        scratch: &mut [IntervalBox<f64>],
    ) -> Result<()> {
        if b.is_none() {
            self.sweeps
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
        let a = [f64::from_bits(
            self.coefficient.load(core::sync::atomic::Ordering::Relaxed),
        )];
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a)?)?
            .apply_enclosure_into(x, b, y, scratch)
    }
}

#[test]
fn external_verification_recomputes_every_call_in_a_reused_workspace() {
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    let system = ChangingMatrix {
        coefficient: AtomicU64::new(2.0_f64.to_bits()),
        sweeps: AtomicUsize::new(0),
    };
    let r = [0.5];
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let mut certificate = LinearCertificate::new(1).unwrap();
    let rhs = [point_enclosure(1.).unwrap()];
    assert!(
        plan.solve_into(&rhs, certificate.view_mut(), &mut workspace, seq())
            .unwrap()
            .is_verified()
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 1);
    assert!(
        plan.verify_scoped(&rhs, certificate.view(), &mut workspace, seq())
            .unwrap()
            .is_verified()
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 2);
    system
        .coefficient
        .store(4.0_f64.to_bits(), Ordering::Relaxed);
    assert_eq!(
        plan.verify_scoped(&rhs, certificate.view(), &mut workspace, seq())
            .unwrap()
            .status(),
        LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
            LinearProofFailure::RegularityNotProved
        ))
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 3);
    system
        .coefficient
        .store(2.0_f64.to_bits(), Ordering::Relaxed);
    assert_eq!(
        plan.verify_scoped(
            &[point_enclosure(4.).unwrap()],
            certificate.view(),
            &mut workspace,
            seq()
        )
        .unwrap()
        .status(),
        LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow)
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 4);
    assert!(
        plan.propose_into(&rhs, &[0.5], certificate.view_mut(), &mut workspace, seq())
            .unwrap()
            .is_verified()
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 5);
    assert!(
        plan.bounds_scoped(&mut workspace, seq())
            .unwrap()
            .is_verified()
    );
    assert_eq!(system.sweeps.load(Ordering::Relaxed), 6);
}

#[test]
fn generated_radii_cover_rounding_boundaries_and_reject_overflow() {
    for coefficient in [1.0_f64, 0.5, f64::EPSILON / 2.] {
        let a = [coefficient];
        let r = [1.];
        let system =
            DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
        let inverse =
            DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
        let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut certificate = LinearCertificate::new(1).unwrap();
        for b in [0., f64::from_bits(1), 1.] {
            let rhs = [point_enclosure(b).unwrap()];
            let report = plan
                .propose_into(&rhs, &[0.], certificate.view_mut(), &mut workspace, seq())
                .unwrap();
            // A defect one destination step below one still bounds the error.
            assert!(
                report.is_verified(),
                "coefficient={coefficient}, b={b}: {report:?}"
            );
            // Powers of two make b / coefficient the exact solution.
            assert!(
                certificate.enclosure().unwrap()[0]
                    .re
                    .contains(b / coefficient)
            );
            assert!(plan.verify(&rhs, certificate.view()).unwrap().is_verified());
        }
        let old = certificate.clone();
        let report = plan
            .propose_into(
                &[point_enclosure(f64::MAX).unwrap()],
                &[0.],
                certificate.view_mut(),
                &mut workspace,
                seq(),
            )
            .unwrap();
        assert_eq!(
            report.status(),
            LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded)
        );
        assert_eq!(certificate, old);
    }
}

#[test]
fn generated_radii_never_reject_defects_near_one() {
    // A = [1] and R = [r] give the defect q = 1 - r and the exact solution b.
    let system =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &[1.]).unwrap()).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for inverse in [[0.1_f64], [0.03], [0.01]] {
        let r =
            DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &inverse).unwrap()).unwrap();
        let plan = LinearVerifier::new(&system, &r, options()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut proposal = LinearCertificate::new(1).unwrap();
        for step in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let b = if step == 0 {
                f64::from_bits(0x3ffb_c03b_4870_4e20)
            } else {
                f64::from_bits(0x3ff0_0000_0000_0000 | (state >> 12))
                    .scalbn((state % 41) as i32 - 20)
            };
            let rhs = [point_enclosure(b).unwrap()];
            let (report, certificate) = plan.solve(&rhs).unwrap();
            assert!(report.is_verified(), "r={inverse:?}, b={b}: {report:?}");
            let defect = report.bounds().unwrap().defect();
            assert!((defect - (1. - inverse[0])).abs() <= 2. * f64::EPSILON);
            let certificate = certificate.unwrap();
            let (center, radius) = (certificate.view().center[0], certificate.view().radii[0].re);
            assert!(certificate.enclosure().unwrap()[0].re.contains(b));
            // The contraction bound is tight here: the radius is the exact error.
            assert!(
                radius <= (b - center).abs() * (1. + 1e-12),
                "{radius} {center} {b}"
            );
            assert!(plan.verify(&rhs, certificate.view()).unwrap().is_verified());
            let report = plan
                .propose_into(&rhs, &[0.5 * b], proposal.view_mut(), &mut workspace, seq())
                .unwrap();
            assert!(report.is_verified(), "r={inverse:?}, b={b}: {report:?}");
            assert!(proposal.enclosure().unwrap()[0].re.contains(b));
            assert!(proposal.view().radii[0].re <= 0.5 * b * (1. + 1e-12));
            assert!(plan.verify(&rhs, proposal.view()).unwrap().is_verified());
        }
        // An external box that misses the solution is still rejected.
        let narrow = [Complex::new(0.25, 0.)];
        let view = LinearCertificateView {
            center: &[0.5],
            radii: &narrow,
        };
        assert_eq!(
            plan.verify(&[point_enclosure(1.).unwrap()], view)
                .unwrap()
                .status(),
            LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow)
        );
    }
}

#[test]
fn nonfinite_linear_inputs_are_nonfinite_errors() {
    use SciOxideErrorKind::{InvalidInput, NonFinite};
    let (a, r) = ([2.0_f64], [0.5]);
    let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let rhs = [point_enclosure(1.).unwrap()];
    for (center, radius, expected) in [
        (f64::NAN, Complex::new(0.25, 0.), NonFinite),
        (f64::INFINITY, Complex::new(0.25, 0.), NonFinite),
        (0.5, Complex::new(f64::INFINITY, 0.), NonFinite),
        (0.5, Complex::new(f64::NAN, 0.), NonFinite),
        (0.5, Complex::new(-0.25, 0.), InvalidInput),
        (0.5, Complex::new(0.25, 0.25), InvalidInput),
    ] {
        let (center, radii) = ([center], [radius]);
        let view = LinearCertificateView {
            center: &center,
            radii: &radii,
        };
        assert_eq!(plan.verify(&rhs, view).unwrap_err().kind(), expected);
        let mut boxes = [point_enclosure(0.).unwrap()];
        assert_eq!(
            view.enclosure_into(&mut boxes).unwrap_err().kind(),
            expected
        );
    }
    let mut workspace = plan.workspace(1).unwrap();
    let mut out = LinearCertificate::new(1).unwrap();
    for center in [f64::NAN, f64::NEG_INFINITY] {
        let error = plan
            .propose_into(&rhs, &[center], out.view_mut(), &mut workspace, seq())
            .unwrap_err();
        assert_eq!(error.kind(), NonFinite);
        assert_eq!(out, LinearCertificate::new(1).unwrap());
        let mut output = [0.];
        let mut scratch = [0.];
        let error = system
            .apply_into(&[center], &mut output, &mut scratch)
            .unwrap_err();
        assert_eq!(error.kind(), NonFinite);
        assert_eq!(point_enclosure(center).unwrap_err().kind(), NonFinite);
    }
    let unbounded = [Complex::new(
        scioxide_core::ExtendedInterval::new(Some(0.), None).unwrap(),
        zero::<f64>().unwrap().im,
    )];
    assert_eq!(plan.solve(&unbounded).unwrap_err().kind(), InvalidInput);
}

#[test]
fn unallocatable_requirements_fail_cleanly() {
    let a = [2.];
    let rv = [0.5];
    let inner = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let r = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &rv).unwrap()).unwrap();
    let huge = Extension { inner, mode: 4 };
    let plan = LinearVerifier::new(&huge, &r, options()).unwrap();
    assert_eq!(
        plan.workspace(1).err().unwrap().kind(),
        SciOxideErrorKind::AllocationFailure
    );
    let mut cert = LinearCertificate::new(1).unwrap();
    let error = plan
        .solve_into(
            &[point_enclosure(1.).unwrap()],
            cert.view_mut(),
            &mut LinearWorkspace::new(),
            seq(),
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::AllocationFailure);
    assert_eq!(cert.view().center, [0.]);
}

#[test]
fn ordinary_seed_overflow_is_inconclusive_and_preserves_output() {
    let a = [0.5];
    let r = [2.];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let rhs = [point_enclosure(f64::MAX).unwrap()];
    let mut workspace = plan.workspace(1).unwrap();
    let mut output = LinearCertificate::new(1).unwrap();
    output.center[0] = 77.;
    output.radii[0].re = 33.;
    let stored = output.clone();
    let report = plan
        .solve_into(&rhs, output.view_mut(), &mut workspace, seq())
        .unwrap();
    let failed = LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
        LinearProofFailure::SeedFailure,
    ));
    assert_eq!(report.status(), failed);
    assert!(report.bounds().is_some());
    assert!(report.residual().is_none());
    assert_eq!(output, stored);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let (report, certificate) = plan.solve(&rhs).unwrap();
    assert_eq!(report.status(), failed);
    assert!(certificate.is_none());
    let smaller = [point_enclosure(f64::MAX / 4.).unwrap()];
    let report = plan
        .solve_into(&smaller, output.view_mut(), &mut workspace, seq())
        .unwrap();
    assert!(report.is_verified());
    assert!(output.enclosure().unwrap()[0].re.contains(f64::MAX / 2.));
}

#[test]
fn seed_numerical_failures_and_provider_contract_errors_stay_distinct() {
    let a = [0.5];
    let r = [2.];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let rhs = [point_enclosure(1.).unwrap()];
    let mut output = LinearCertificate::new(1).unwrap();
    output.center[0] = 77.;
    let stored = output.clone();
    for mode in 6..=11 {
        let inverse = Extension {
            inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap(),
            mode,
        };
        let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let result = plan.solve_into(&rhs, output.view_mut(), &mut workspace, seq());
        assert_eq!(output, stored);
        if mode == 11 {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
        } else {
            assert_eq!(
                result.unwrap().status(),
                LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
                    LinearProofFailure::SeedFailure
                ))
            );
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Stopped
            );
        }
    }
}

#[test]
fn late_seed_failure_preserves_every_block_output() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let a = [0.5];
    let r = [2.];
    let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap();
    let inverse = Extension {
        inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r).unwrap()).unwrap(),
        mode: 12,
    };
    let plan = LinearVerifier::new(&system, &inverse, options()).unwrap();
    let rhs = [point_enclosure(1.).unwrap(), point_enclosure(2.).unwrap()];
    let mut centers = [77.; 2];
    let mut radii = [Complex::new(33., 0.); 2];
    let mut reports = [LinearVerificationReport::not_run(); 2];
    plan.solve_block_into(
        BlockVectorView::new(&rhs, 1, 2).unwrap(),
        &mut BlockVectorViewMut::new(&mut centers, 1, 2).unwrap(),
        &mut BlockVectorViewMut::new(&mut radii, 1, 2).unwrap(),
        &mut reports,
        &mut plan.workspace(2).unwrap(),
        seq(),
    )
    .unwrap();
    assert!(reports[0].is_verified());
    assert_eq!(
        reports[1].status(),
        LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
            LinearProofFailure::SeedFailure
        ))
    );
    assert_eq!(centers, [77.; 2]);
    assert_eq!(radii, [Complex::new(33., 0.); 2]);
}

#[test]
fn reused_workspace_preserves_surplus_linear_lanes() {
    let small_a = [0.5];
    let small_r = [2.];
    let small_system =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &small_a).unwrap()).unwrap();
    let small_inverse =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &small_r).unwrap()).unwrap();
    let small = LinearVerifier::new(&small_system, &small_inverse, options()).unwrap();
    let large_a = Matrix::<f64>::identity(8).unwrap();
    let large_r = Matrix::<f64>::identity(8).unwrap();
    let large_system = DenseIntervalOperator::point(large_a.view()).unwrap();
    let large_inverse = DenseIntervalOperator::point(large_r.view()).unwrap();
    let large = LinearVerifier::new(&large_system, &large_inverse, options()).unwrap();
    let limit = small.workspace_requirements(64).unwrap().required_bytes()
        + large.workspace_requirements(1).unwrap().required_bytes();
    let limited = LinearVerifier::new(
        &large_system,
        &large_inverse,
        options().with_byte_limit(limit),
    )
    .unwrap();
    let mut workspace = small.workspace(64).unwrap();
    let rhs = [point_enclosure(1.).unwrap(); 8];
    let mut output = LinearCertificate::new(8).unwrap();
    assert!(
        limited
            .solve_into(&rhs, output.view_mut(), &mut workspace, seq())
            .unwrap()
            .is_verified()
    );
    {
        assert!(
            limited
                .solve_into(&rhs, output.view_mut(), &mut workspace, seq())
                .unwrap()
                .is_verified()
        );
    };
}
