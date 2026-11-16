use super::*;
use crate::{DenseIntervalOperator, LuIntervalInverse, SparseIntervalOperator, interval_enclosure};
use crate::{IntervalJacobianOperator, IntervalLinearOperator, IntervalResidualOperator};
use alloc::vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, JacobianOperator, Quantity, ResidualOperator,
    SciOxideErrorKind, SolveBudgets, SolveState, StepTolerance, StructuralPatternView,
};
use scioxide_linalg::{LuDecomposition, Matrix, MatrixView};

/// Fixture provider workspace length.
const WORKSPACE: Quantity = scioxide_core::quantity!("workspace_len");

fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn s<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(r(re), r(im))
}
const fn opts() -> ValidatedOptions {
    ValidatedOptions::new()
}
const fn seq() -> ExecutionContext<'static> {
    ExecutionContext::sequential()
}
fn budgets(kind: BudgetKind, limit: usize) -> ValidatedOptions {
    opts()
        .with_budgets(SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap()))
}

fn scalar<R: RealScalar>() {
    let value = |x: D<R>| x.square()?.sub(&D::point(r(2.)).unwrap());
    let derivative = |x: D<R>| x.mul(&D::point(r(2.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    for (lo, hi, center) in [(1., 2., 1.5), (-2., -1., -1.5)] {
        let c = ScalarRootCertificate {
            region: D::<R>::new(r::<R>(lo), r::<R>(hi)).unwrap().set(),
            center: r(center),
            claim: RootClaim::Unique,
        };
        let result = plan.verify(c).unwrap();
        assert!(result.is_proved(), "{result:?}");
        assert_eq!(
            plan.verify(ScalarRootCertificate {
                claim: RootClaim::Empty,
                ..c
            })
            .unwrap()
            .status(),
            RootVerificationStatus::Refuted
        );
    }
    let excluded = ScalarRootCertificate {
        region: D::<R>::new(r::<R>(3.), r::<R>(4.)).unwrap().set(),
        center: r(3.5),
        claim: RootClaim::Exists,
    };
    assert_eq!(
        plan.verify(excluded).unwrap().status(),
        RootVerificationStatus::Refuted
    );
    assert!(
        plan.verify(ScalarRootCertificate {
            center: r(9.),
            ..excluded
        })
        .is_err()
    );
    let repeated = |x: D<R>| x.square();
    let repeated = ScalarRootVerifier::new(&repeated, &derivative, opts());
    let c = ScalarRootCertificate {
        region: D::<R>::new(r::<R>(-1.), r::<R>(1.)).unwrap().set(),
        center: r(0.),
        claim: RootClaim::Unique,
    };
    let report = repeated.verify(c).unwrap();
    assert!(matches!(
        report.status(),
        RootVerificationStatus::Inconclusive(_)
    ));
    assert_eq!(report.evidence(), Some(RootClaim::Exists));
    assert!(
        repeated
            .verify(ScalarRootCertificate {
                claim: RootClaim::Exists,
                ..c
            })
            .unwrap()
            .is_proved()
    );
    let identity = |x: D<R>| Ok(x);
    let one = |_: D<R>| Ok(D::point(r(1.)).unwrap());
    let boundary = ScalarRootVerifier::new(&identity, &one, opts());
    assert!(
        boundary
            .verify(ScalarRootCertificate {
                region: D::<R>::new(r::<R>(0.), r::<R>(1.)).unwrap().set(),
                center: r(0.5),
                claim: RootClaim::Unique
            })
            .unwrap()
            .is_proved()
    );
}
#[test]
fn scalar32() {
    scalar::<f32>();
}
#[test]
fn scalar64() {
    scalar::<f64>();
}

fn simple_polynomial<S: NumericScalar>() {
    let center = Complex::new(r(1.), r(if S::IS_REAL { 0. } else { 1. }));
    for exponent in [-100, 0, 100] {
        let factor = S::Real::ONE.scalbn(exponent);
        let coefficients = if S::IS_REAL {
            [s::<S>(-1., 0.), S::ZERO, S::ONE]
        } else {
            [s::<S>(0., -2.), S::ZERO, S::ONE]
        }
        .map(|c| c * S::from_real(factor));
        // One iteration permits only the linear inclusion pass, so success
        // cannot come from falling back to the Taylor shift.
        let plan =
            PolynomialRootPlan::new(&coefficients, budgets(BudgetKind::Iteration, 1)).unwrap();
        let certificate = PolynomialRootCertificate {
            disk: RootDisk {
                center,
                radius: r(0.001),
            },
            count: 1,
        };
        let mut workspace = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let report = plan
            .verify_scoped(certificate, &mut workspace, seq())
            .unwrap();
        assert!(report.is_proved(), "{report:?}");
        assert_eq!(
            plan.verify(PolynomialRootCertificate {
                count: 2,
                ..certificate
            })
            .unwrap()
            .status(),
            RootVerificationStatus::Refuted
        );
    }
}
#[test]
fn simple_polynomial_inclusion_all_scalars() {
    simple_polynomial::<f32>();
    simple_polynomial::<f64>();
    simple_polynomial::<Complex<f32>>();
    simple_polynomial::<Complex<f64>>();
}

#[test]
fn polynomial_iteration_admission_and_fallback_boundaries() {
    let disk = RootDisk {
        center: Complex::new(1.0, 0.0),
        radius: 0.125,
    };
    for (coefficients, count, required) in
        [([-1.0_f64, 0.0, 1.0], 1, 1), ([1.0_f64, -2.0, 1.0], 2, 4)]
    {
        for allowance in 1..=required {
            let plan =
                PolynomialRootPlan::new(&coefficients, budgets(BudgetKind::Iteration, allowance))
                    .unwrap();
            let report = plan
                .verify(PolynomialRootCertificate { disk, count })
                .unwrap();
            if allowance == required {
                assert!(report.is_proved(), "{report:?}");
                assert_eq!(report.count(), Some(count));
            } else {
                assert_eq!(
                    report.status(),
                    RootVerificationStatus::Inconclusive(RootInconclusive::IterationLimit)
                );
            }
        }
    }
    let plan =
        PolynomialRootPlan::new(&[-1.0, 0.0, 1.0], budgets(BudgetKind::Iteration, 4)).unwrap();
    let mut workspace = plan.workspace(ProofRequest::refinement(2)).unwrap();
    let mut cover = PolynomialRootCover::new(2).unwrap();
    // One isolation round, two projections and one disk pass publish exactly
    // the first root; the second pass is denied before doing arithmetic.
    let report = plan
        .isolate_into(
            &[
                disk,
                RootDisk {
                    center: Complex::new(-1.0, 0.0),
                    ..disk
                },
            ],
            &mut cover,
            &mut workspace,
            seq(),
        )
        .unwrap();
    assert_eq!(report.certificates(), 1);
    assert_eq!(report.roots(), 1);
    assert_eq!(cover.view().len(), 1);
    assert_eq!(
        report.termination(),
        RootIsolationTermination::Inconclusive(RootInconclusive::IterationLimit)
    );
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let report = plan
        .verify_scoped(
            PolynomialRootCertificate { disk, count: 1 },
            &mut workspace,
            seq().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        report.status(),
        RootVerificationStatus::Inconclusive(RootInconclusive::Stopped(_))
    ));
}

#[test]
fn polynomial_inclusion_boundary_and_overflow() {
    let plan = PolynomialRootPlan::new(&[-1., 1.], opts()).unwrap();
    for radius in [0.5, 1.0_f64.next_down(), 1., 1.0_f64.next_up(), 2.] {
        let report = plan
            .verify(PolynomialRootCertificate {
                disk: RootDisk {
                    center: Complex::new(0., 0.),
                    radius,
                },
                count: 1,
            })
            .unwrap();
        if radius == 0.5 {
            assert_eq!(report.status(), RootVerificationStatus::Refuted);
        } else if radius == 2. {
            assert!(report.is_proved());
        } else if radius == 1. {
            assert_eq!(
                report.status(),
                RootVerificationStatus::Inconclusive(RootInconclusive::Reason(
                    RootProofFailure::Inequality
                ))
            );
        } else if radius < 1. {
            assert!(!report.is_proved());
        } else {
            assert_ne!(report.status(), RootVerificationStatus::Refuted);
        }
    }
    let plan = PolynomialRootPlan::new(&[1., 0., 1.], opts()).unwrap();
    let report = plan
        .verify(PolynomialRootCertificate {
            disk: RootDisk {
                center: Complex::new(1e308, 1e308),
                radius: 1e308,
            },
            count: 1,
        })
        .unwrap();
    assert!(matches!(
        report.status(),
        RootVerificationStatus::Inconclusive(_)
    ));
}

#[test]
fn polynomial_cover_output_and_owned_isolation() {
    let coefficients = [-2.0_f64, 0., 1.];
    let plan = PolynomialRootPlan::new(&coefficients, opts()).unwrap();
    let seeds = [1.4, -1.4].map(|re| RootDisk {
        center: Complex::new(re, 0.),
        radius: 0.125,
    });
    let (report, cover) = plan.isolate(&seeds).unwrap();
    assert!(report.is_complete() && report.roots() == 2, "{report:?}");
    assert_eq!((cover.capacity(), cover.view().len()), (2, 2));
    assert!(plan.verify_cover(cover.view()).unwrap().is_proved());
    let mut prepared = PolynomialRootCover::new(2).unwrap();
    let mut workspace = RootWorkspace::new();
    assert_eq!(
        plan.isolate_into(&seeds, &mut prepared, &mut workspace, seq())
            .unwrap(),
        report
    );
    assert_eq!(prepared, cover);
    // A short cover is rejected before any proof and keeps its published certificates.
    let mut short = PolynomialRootCover::new(1).unwrap();
    let partial = plan
        .isolate_into(&seeds[..1], &mut short, &mut workspace, seq())
        .unwrap();
    assert_eq!((partial.certificates(), partial.roots()), (1, 1));
    let before = short.clone();
    let error = plan
        .isolate_into(&seeds, &mut short, &mut workspace, seq())
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_preflight(&workspace);
    assert_eq!(short, before);
    // Without seeds, isolation publishes nothing and accounts for no root.
    let (report, empty) = plan.isolate(&[]).unwrap();
    assert!(empty.view().is_empty() && empty.capacity() == 0);
    assert_eq!((report.certificates(), report.roots()), (0, 0));
    assert_eq!(
        report.termination(),
        RootIsolationTermination::Inconclusive(RootInconclusive::Reason(
            RootProofFailure::Inequality
        ))
    );
}

struct Squares<'a, S: NumericScalar> {
    mixing: &'a (dyn IntervalLinearOperator<S> + Sync),
    targets: &'a [S],
    mode: usize,
}
impl<S: NumericScalar> ResidualOperator<S> for Squares<'_, S> {
    fn input_dimension(&self) -> usize {
        self.targets.len()
    }
    fn output_dimension(&self) -> usize {
        self.targets.len()
    }
    fn residual_workspace_len(&self) -> Result<usize> {
        checked_add(
            self.targets.len(),
            self.mixing.apply_workspace_len()?,
            WORKSPACE,
        )
    }
    fn residual_into(&self, x: &[S], out: &mut [S], work: &mut [S]) -> Result<()> {
        let n = self.targets.len();
        if x.len() != n || out.len() != n || work.len() < self.residual_workspace_len()? {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "test residual shape",
            ));
        }
        let (input, scratch) = work.split_at_mut(n);
        for ((v, &x), &a) in input.iter_mut().zip(x).zip(self.targets) {
            *v = x * x - a;
        }
        self.mixing.apply_into(input, out, scratch)
    }
}
impl<S: NumericScalar> JacobianOperator<S> for Squares<'_, S> {
    fn jvp_workspace_len(&self) -> Result<usize> {
        self.residual_workspace_len()
    }
    fn jvp_into(&self, x: &[S], v: &[S], out: &mut [S], work: &mut [S]) -> Result<()> {
        let n = self.targets.len();
        if x.len() != n
            || v.len() != n
            || out.len() != n
            || work.len() < self.jvp_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "test Jacobian shape",
            ));
        }
        let (input, scratch) = work.split_at_mut(n);
        for ((slot, &x), &v) in input.iter_mut().zip(x).zip(v) {
            *slot = (S::ONE + S::ONE) * x * v;
        }
        self.mixing.apply_into(input, out, scratch)
    }
}
impl<S: NumericScalar> IntervalResidualOperator<S> for Squares<'_, S> {
    fn residual_enclosure_workspace_len(&self) -> Result<usize> {
        checked_add(
            self.targets.len(),
            self.mixing.enclosure_workspace_len()?,
            WORKSPACE,
        )
    }
    fn residual_enclosure_into(
        &self,
        x: &[IntervalBox<S::Real>],
        out: &mut [IntervalBox<S::Real>],
        work: &mut [IntervalBox<S::Real>],
    ) -> Result<()> {
        crate::shell::run_local(
            seq().into(),
            identity(),
            SolveBudgets::unbounded(),
            |context| self.residual_enclosure_into_scoped(x, out, work, context),
        )
    }
    fn residual_enclosure_into_scoped(
        &self,
        x: &[IntervalBox<S::Real>],
        out: &mut [IntervalBox<S::Real>],
        work: &mut [IntervalBox<S::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<()> {
        if self.mode == 1 {
            return Ok(());
        }
        if self.mode == 2 {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "injected provider failure",
            ));
        }
        if self.mode == 3 {
            out.fill(Complex::new(
                ExtendedInterval::entire(),
                zero::<S::Real>()?.im,
            ));
            return Ok(());
        }
        let n = self.targets.len();
        if x.len() != n || out.len() != n || work.len() < self.residual_enclosure_workspace_len()? {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "test interval residual shape",
            ));
        }
        let (input, scratch) = work.split_at_mut(n);
        for ((slot, &x), &a) in input.iter_mut().zip(x).zip(self.targets) {
            *slot = pack(unpack(x).square()?.sub(Box2::point(a)?)?);
        }
        self.mixing
            .apply_enclosure_into_scoped(
                input,
                None,
                out,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )?
            .require()
    }
}
impl<S: NumericScalar> IntervalJacobianOperator<S> for Squares<'_, S> {
    fn jvp_enclosure_workspace_len(&self) -> Result<usize> {
        self.residual_enclosure_workspace_len()
    }
    fn jvp_enclosure_into(
        &self,
        x: &[IntervalBox<S::Real>],
        v: &[IntervalBox<S::Real>],
        out: &mut [IntervalBox<S::Real>],
        work: &mut [IntervalBox<S::Real>],
    ) -> Result<()> {
        crate::shell::run_local(
            seq().into(),
            identity(),
            SolveBudgets::unbounded(),
            |context| self.jvp_enclosure_into_scoped(x, v, out, work, context),
        )
    }
    fn jvp_enclosure_into_scoped(
        &self,
        x: &[IntervalBox<S::Real>],
        v: &[IntervalBox<S::Real>],
        out: &mut [IntervalBox<S::Real>],
        work: &mut [IntervalBox<S::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<()> {
        let n = self.targets.len();
        if x.len() != n
            || v.len() != n
            || out.len() != n
            || work.len() < self.jvp_enclosure_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "test interval Jacobian shape",
            ));
        }
        let (input, scratch) = work.split_at_mut(n);
        for ((slot, &x), &v) in input.iter_mut().zip(x).zip(v) {
            *slot = pack(unpack(x).scale(2.)?.mul(unpack(v))?);
        }
        self.mixing
            .apply_enclosure_into_scoped(
                input,
                None,
                out,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )?
            .require()
    }
}
fn system<S: NumericScalar>() {
    let center = [s::<S>(1., 1.), s(2., -1.)];
    let targets = center.map(|v| v * v);
    let entries = [S::ONE, s(0.25, 0.), S::ZERO, S::ONE];
    let matrix = Matrix::from_vec(2, 2, entries.to_vec()).unwrap();
    let dense = DenseIntervalOperator::point(matrix.view()).unwrap();
    let rows = [0, 2, 3];
    let columns = [0, 1, 1];
    let sparse_values = [S::ONE, s(0.25, 0.), S::ONE];
    let sparse = SparseIntervalOperator::new(
        StructuralPatternView::new(2, 2, &rows, &columns).unwrap(),
        &sparse_values,
        &sparse_values,
    )
    .unwrap();
    let jacobian = Matrix::from_vec(
        2,
        2,
        vec![
            (S::ONE + S::ONE) * center[0],
            s::<S>(0.5, 0.) * center[1],
            S::ZERO,
            (S::ONE + S::ONE) * center[1],
        ],
    )
    .unwrap();
    let lu = LuDecomposition::factor(&jacobian).unwrap();
    let inverse = LuIntervalInverse::from_dense(&lu, crate::ValidatedOptions::new()).unwrap();
    let radius = if !S::IS_REAL {
        r(0.0625)
    } else {
        S::Real::ZERO
    };
    let region: Vec<_> = center
        .iter()
        .map(|&c| {
            pack(
                Box2::point(c)
                    .unwrap()
                    .add(Box2::symmetric(r(0.0625), radius).unwrap())
                    .unwrap(),
            )
        })
        .collect();
    for mixing in [&dense as &(dyn IntervalLinearOperator<S> + Sync), &sparse] {
        let function = Squares {
            mixing,
            targets: &targets,
            mode: 0,
        };
        let plan = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let c = SystemRootCertificateView {
            region: &region,
            center: &center,
            claim: RootClaim::Unique,
        };
        let report = plan.verify_scoped(c, &mut work, seq()).unwrap();
        assert!(report.is_proved(), "{report:?}");
        assert!(report.defect().unwrap() < S::Real::ONE);
        let mut owned = SystemRootCertificate::new(2).unwrap();
        let proposal = owned.view_mut();
        proposal.region.copy_from_slice(c.region);
        proposal.center.copy_from_slice(c.center);
        assert_eq!(*proposal.claim, RootClaim::Unique);
        assert!(
            plan.verify_scoped(owned.view(), &mut work, seq())
                .unwrap()
                .is_proved()
        );
        assert_eq!(
            plan.verify_scoped(
                SystemRootCertificateView {
                    claim: RootClaim::Empty,
                    ..c
                },
                &mut work,
                seq()
            )
            .unwrap()
            .status(),
            RootVerificationStatus::Refuted
        );
        let corrupted = [s(9., 0.), center[1]];
        assert!(
            plan.verify_scoped(
                SystemRootCertificateView {
                    center: &corrupted,
                    ..c
                },
                &mut work,
                seq()
            )
            .is_err()
        );
        // The owned form equals the prepared form over fresh storage.
        let (report, tree) = plan.isolate(&region, StepTolerance::default(), 15).unwrap();
        assert!(report.is_complete() && report.isolated() == 1, "{report:?}");
        let mut prepared = RootIsolation::new(2, 15).unwrap();
        assert_eq!(
            plan.isolate_into(
                &region,
                StepTolerance::default(),
                &mut prepared,
                &mut RootWorkspace::new(),
                seq()
            )
            .unwrap(),
            report
        );
        assert_eq!(prepared, tree);
        assert!(
            plan.verify_isolation(&region, tree.view())
                .unwrap()
                .is_complete()
        );
        // Replay refutes a forged root-free claim on the unique-root region.
        let forged = [RootIsolationNode::Leaf(Some(RootClaim::Empty))];
        let replay = plan
            .verify_isolation(
                &region,
                RootIsolationView {
                    nodes: &forged,
                    regions: &region,
                },
            )
            .unwrap();
        assert_eq!(replay.termination(), RootIsolationTermination::Refuted);
        assert_eq!((replay.nodes(), replay.unresolved()), (0, 1));
    }
}
#[test]
fn system32() {
    system::<f32>();
}
#[test]
fn system64() {
    system::<f64>();
}
#[test]
fn system_complex32() {
    system::<Complex<f32>>();
}
#[test]
fn system_complex64() {
    system::<Complex<f64>>();
}

#[test]
fn coverage_replay_corruption_and_partial_results() {
    let value = |x: D<f64>| x.square()?.sub(&D::point(r(2.)).unwrap());
    let derivative = |x: D<f64>| x.mul(&D::point(r(2.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let domain = D::new(r(-2.), r(2.)).unwrap().set();
    let mut out = RootIsolation::new(1, 63).unwrap();
    let mut w = RootWorkspace::new();
    let report = plan
        .isolate_into(domain, StepTolerance::default(), &mut out, &mut w, seq())
        .unwrap();
    assert!(report.is_complete(), "{report:?}");
    assert_eq!(report.isolated(), 2);
    assert_eq!(
        plan.isolate(domain, StepTolerance::default(), 63).unwrap(),
        (report, out.clone())
    );
    assert_eq!(
        plan.isolate(domain, StepTolerance::default(), 0)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let again = plan
        .isolate_into(domain, StepTolerance::default(), &mut out, &mut w, seq())
        .unwrap();
    assert_eq!(again, report);
    assert!(
        plan.verify_isolation(domain, out.view())
            .unwrap()
            .is_complete()
    );
    let mut regions = out.view().regions.to_vec();
    regions[1].re = D::new(r(-1.5), r(0.)).unwrap().set();
    let corrupt = RootIsolationView {
        nodes: out.view().nodes,
        regions: &regions,
    };
    assert!(
        plan.verify_isolation_scoped(domain, corrupt, &mut w, seq())
            .is_err()
    );
    let mut nodes = out.view().nodes.to_vec();
    nodes[0] = RootIsolationNode::Split {
        axis: 0,
        point: 0.,
        left: 0,
    };
    assert!(
        plan.verify_isolation_scoped(
            domain,
            RootIsolationView {
                nodes: &nodes,
                regions: out.view().regions
            },
            &mut w,
            seq()
        )
        .is_err()
    );
    let mut small = RootIsolation::new(1, 1).unwrap();
    let mut small_work = RootWorkspace::new();
    let report = plan
        .isolate_into(
            domain,
            StepTolerance::default(),
            &mut small,
            &mut small_work,
            seq(),
        )
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(report.unresolved(), 1);
    assert_eq!(
        report.termination(),
        RootIsolationTermination::Limit(crate::RefinementLimit::Capacity)
    );
    assert_eq!(small.view().regions[0].re, domain);
}

#[test]
fn replay_reports_refuted_claims_and_counts_leaves() {
    let value = |x: D<f64>| x.square()?.sub(&D::point(r(2.)).unwrap());
    let derivative = |x: D<f64>| x.mul(&D::point(r(2.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let domain = D::new(r(-2.), r(2.)).unwrap().set();
    let (report, tree) = plan.isolate(domain, StepTolerance::default(), 63).unwrap();
    let nodes = tree.view().nodes;
    let leaves = |nodes: &[RootIsolationNode<f64>]| {
        nodes
            .iter()
            .filter(|n| matches!(n, RootIsolationNode::Leaf(_)))
            .count()
    };
    assert_eq!((nodes.len(), leaves(nodes)), (7, 4));
    assert_eq!(
        (report.isolated(), report.excluded(), report.unresolved()),
        (2, 2, 0)
    );
    // Claim a root in the first root-free leaf; a direct proof refutes it.
    let empty = nodes
        .iter()
        .position(|n| *n == RootIsolationNode::Leaf(Some(RootClaim::Empty)))
        .unwrap();
    let region = tree.view().regions[empty].re;
    let (lo, hi) = D::from(region).bounds().unwrap();
    let leaf = plan
        .verify(ScalarRootCertificate {
            region,
            center: 0.5 * (lo + hi),
            claim: RootClaim::Unique,
        })
        .unwrap();
    assert_eq!(leaf.status(), RootVerificationStatus::Refuted);
    let mut forged = nodes.to_vec();
    forged[empty] = RootIsolationNode::Leaf(Some(RootClaim::Unique));
    let forged = RootIsolationView {
        nodes: &forged,
        regions: tree.view().regions,
    };
    let replay = plan.verify_isolation(domain, forged).unwrap();
    assert_eq!(replay.termination(), RootIsolationTermination::Refuted);
    assert!(!replay.is_complete());
    assert_eq!(replay.nodes(), empty);
    let proved = forged.nodes[..empty]
        .iter()
        .filter(|n| matches!(n, RootIsolationNode::Leaf(Some(_))))
        .count();
    assert_eq!(replay.isolated() + replay.excluded(), proved);
    assert_eq!(replay.unresolved(), 4 - proved);
    // An undecided claim stays inconclusive: a unique claim on the whole domain.
    let whole = [RootIsolationNode::Leaf(Some(RootClaim::Unique))];
    let replay = plan
        .verify_isolation(
            domain,
            RootIsolationView {
                nodes: &whole,
                regions: &tree.view().regions[..1],
            },
        )
        .unwrap();
    assert!(matches!(
        replay.termination(),
        RootIsolationTermination::Inconclusive(_)
    ));
    assert_eq!((replay.nodes(), replay.unresolved()), (0, 1));
    // An iteration stop after the root split leaves all four leaves unresolved.
    let limited = ScalarRootVerifier::new(&value, &derivative, budgets(BudgetKind::Iteration, 1));
    let replay = limited.verify_isolation(domain, tree.view()).unwrap();
    assert_eq!(
        replay.termination(),
        RootIsolationTermination::Inconclusive(RootInconclusive::IterationLimit)
    );
    assert_eq!(
        (
            replay.nodes(),
            replay.isolated(),
            replay.excluded(),
            replay.unresolved()
        ),
        (1, 0, 0, 4)
    );
    // A complete replay counts the same leaves as generation.
    let replay = plan.verify_isolation(domain, tree.view()).unwrap();
    assert_eq!(replay, report);
}

#[test]
fn empty_scalar_enclosures_are_domain_failures() {
    // f(x) = sqrt(x) - 1 has no values on negative inputs and its root at one.
    let one = D::point(r(1.)).unwrap();
    let value = |x: D<f64>| x.sqrt()?.sub(&one);
    let derivative = |x: D<f64>| D::point(r(0.5)).unwrap().div(&x.sqrt()?);
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    assert!(
        D::new(-2.0_f64, -1.)
            .unwrap()
            .sqrt()
            .unwrap()
            .set()
            .is_empty()
    );
    let report = plan
        .verify(ScalarRootCertificate {
            region: D::new(r(-2.), r(-1.)).unwrap().set(),
            center: -1.5,
            claim: RootClaim::Empty,
        })
        .unwrap();
    assert_eq!(
        report.status(),
        RootVerificationStatus::Inconclusive(RootInconclusive::Reason(RootProofFailure::Domain))
    );
    let domain = D::new(r(-2.), r(3.)).unwrap().set();
    let (report, tree) = plan.isolate(domain, StepTolerance::default(), 63).unwrap();
    assert!(!report.is_complete());
    assert_eq!(report.isolated(), 1, "{report:?}");
    let view = tree.view();
    let unique = view
        .nodes
        .iter()
        .position(|n| *n == RootIsolationNode::Leaf(Some(RootClaim::Unique)))
        .unwrap();
    assert!(view.regions[unique].re.contains(1.));
    assert!(D::from(view.regions[unique].re).bounds().unwrap().0 > 0.);
    let replay = plan.verify_isolation(domain, view).unwrap();
    assert_eq!(replay.isolated(), 1);
}

#[test]
fn owned_outputs_compare_and_print_published_data() {
    let value = |x: D<f64>| x.square()?.sub(&D::point(r(2.)).unwrap());
    let derivative = |x: D<f64>| x.mul(&D::point(r(2.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let (wide, narrow) = (
        D::new(r(-2.), r(2.)).unwrap().set(),
        D::new(r(1.), r(2.)).unwrap().set(),
    );
    let mut reused = RootIsolation::new(1, 63).unwrap();
    let mut work = RootWorkspace::new();
    let tolerance = StepTolerance::default();
    let isolate = |domain, out: &mut RootIsolation<f64>, work: &mut RootWorkspace<f64>| {
        let report = plan.isolate_into(domain, tolerance, out, work, seq());
        assert!(report.unwrap().is_complete());
    };
    isolate(wide, &mut reused, &mut work);
    assert_eq!(reused.view().nodes.len(), 7);
    isolate(narrow, &mut reused, &mut work);
    let (_, fresh) = plan.isolate(narrow, tolerance, 63).unwrap();
    assert_eq!(reused.view().nodes, fresh.view().nodes);
    assert_eq!(reused, fresh);
    assert_eq!(alloc::format!("{reused:?}"), alloc::format!("{fresh:?}"));
    assert_ne!(reused, RootIsolation::new(1, 62).unwrap());
    assert_ne!(reused, RootIsolation::new(1, 63).unwrap());

    let plan = PolynomialRootPlan::new(&[-2.0_f64, 0., 1.], opts()).unwrap();
    let seed = |re: f64| RootDisk {
        center: Complex::new(re, 0.),
        radius: 0.125,
    };
    let mut reused = PolynomialRootCover::new(2).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(2)).unwrap();
    let mut isolate = |seeds: &[RootDisk<f64>], out: &mut PolynomialRootCover<f64>| {
        let report = plan.isolate_into(seeds, out, &mut work, seq()).unwrap();
        assert_eq!(report.certificates(), seeds.len());
    };
    isolate(&[seed(1.4), seed(-1.4)], &mut reused);
    isolate(&[seed(1.4)], &mut reused);
    let mut fresh = PolynomialRootCover::new(2).unwrap();
    isolate(&[seed(1.4)], &mut fresh);
    assert_eq!(reused.view(), fresh.view());
    assert_eq!(reused, fresh);
    assert_eq!(alloc::format!("{reused:?}"), alloc::format!("{fresh:?}"));
    assert_ne!(reused, plan.isolate(&[seed(1.4)]).unwrap().1);
}

#[test]
fn unrepresentable_polynomial_radii_reach_the_resolution_limit() {
    let resolution = RootIsolationTermination::Limit(crate::RefinementLimit::Resolution);
    // Doubling the seed radius overflows before it reaches the root at 1.7e308.
    let plan = PolynomialRootPlan::new(&[-1.7e308_f64, 1.], opts()).unwrap();
    let seed = RootDisk {
        center: Complex::new(0., 0.),
        radius: 1.,
    };
    let (report, cover) = plan.isolate(&[seed]).unwrap();
    assert_eq!(report.termination(), resolution);
    assert_eq!((report.certificates(), cover.view().len()), (0, 0));
    // Merging two overlapping disks would need a radius beyond the finite range.
    let plan = PolynomialRootPlan::new(&[-2.0_f64, 0., 1.], opts()).unwrap();
    let seeds = [-1e308, 1e308].map(|re| RootDisk {
        center: Complex::new(re, 0.),
        radius: 1e308,
    });
    let (report, cover) = plan.isolate(&seeds).unwrap();
    assert_eq!(report.termination(), resolution);
    assert_eq!((report.certificates(), cover.view().len()), (0, 0));
}

#[test]
fn nonfinite_root_inputs_are_nonfinite_errors() {
    let kind = |result: Result<PolynomialVerificationReport>| result.unwrap_err().kind();
    for coefficients in [[f64::NAN, 1.], [1., f64::INFINITY]] {
        assert_eq!(
            PolynomialRootPlan::new(&coefficients, opts())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert_eq!(
        PolynomialRootPlan::new(&[0.0_f64, 0.], opts())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let plan = PolynomialRootPlan::new(&[-2.0_f64, 0., 1.], opts()).unwrap();
    let disk = |re: f64, radius: f64| PolynomialRootCertificate {
        disk: RootDisk {
            center: Complex::new(re, 0.),
            radius,
        },
        count: 1,
    };
    for (c, expected) in [
        (disk(f64::NAN, 0.125), SciOxideErrorKind::NonFinite),
        (disk(f64::INFINITY, 0.125), SciOxideErrorKind::NonFinite),
        (disk(1.4, f64::INFINITY), SciOxideErrorKind::NonFinite),
        (disk(1.4, f64::NAN), SciOxideErrorKind::NonFinite),
        (disk(1.4, 0.), SciOxideErrorKind::InvalidInput),
        (disk(1.4, -0.125), SciOxideErrorKind::InvalidInput),
    ] {
        assert_eq!(kind(plan.verify(c)), expected);
        assert_eq!(kind(plan.verify_cover(&[c])), expected);
        assert_eq!(plan.isolate(&[c.disk]).unwrap_err().kind(), expected);
    }
    let value = |x: D<f64>| x.square()?.sub(&D::point(r(2.)).unwrap());
    let derivative = |x: D<f64>| x.mul(&D::point(r(2.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let region = D::new(r(1.), r(2.)).unwrap().set();
    for (center, expected) in [
        (f64::NAN, SciOxideErrorKind::NonFinite),
        (f64::INFINITY, SciOxideErrorKind::NonFinite),
        (3., SciOxideErrorKind::InvalidInput),
    ] {
        let c = ScalarRootCertificate {
            region,
            center,
            claim: RootClaim::Unique,
        };
        assert_eq!(plan.verify(c).unwrap_err().kind(), expected);
    }
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(641),
        scioxide_core::algorithm_id!("roots-test", "parent", 1),
        0,
        0,
    )
}
#[test]
fn providers_budgets_singular_and_transactional_output() {
    let entries = [1.];
    let inverse_entries = [0.5];
    let targets = [2.];
    let mixing =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &entries).unwrap()).unwrap();
    let inverse =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &inverse_entries).unwrap())
            .unwrap();
    let region = [pack(Box2 {
        re: D::new(r(1.), r(2.)).unwrap(),
        im: D::point(r(0.)).unwrap(),
    })];
    let center = [1.5];
    let c = SystemRootCertificateView {
        region: &region,
        center: &center,
        claim: RootClaim::Unique,
    };
    for mode in 1..=3 {
        let function = Squares {
            mixing: &mixing,
            targets: &targets,
            mode,
        };
        let plan = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let result = plan.verify_scoped(c, &mut work, seq());
        if mode == 3 {
            assert_eq!(
                result.unwrap().status(),
                RootVerificationStatus::Inconclusive(RootInconclusive::Unbounded)
            );
        } else {
            assert!(result.is_err());
        }
        let outcome = if mode == 3 {
            AttemptOutcome::Stopped
        } else {
            AttemptOutcome::Failed
        };
        assert_eq!(work.last_attempt().unwrap().outcome(), outcome);
    }
    let function = Squares {
        mixing: &mixing,
        targets: &targets,
        mode: 0,
    };
    let plan = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
    for limit in 1..5 {
        let limited =
            SystemRootVerifier::new(&function, &inverse, budgets(BudgetKind::Evaluation, limit))
                .unwrap();
        let report = limited.verify_scoped(c, &mut work, seq()).unwrap();
        assert_eq!(
            report.status(),
            RootVerificationStatus::Inconclusive(RootInconclusive::EvaluationLimit)
        );
        let attempt = work.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
        assert_eq!(attempt.counts().own().total_providers().unwrap(), limit);
    }
    // Shape rejections precede every callback.
    let wide = [center[0]; 2];
    let error = plan
        .verify_scoped(
            SystemRootCertificateView { center: &wide, ..c },
            &mut work,
            seq(),
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let small = SystemRootVerifier::new(&function, &inverse, opts().with_byte_limit(8)).unwrap();
    let error = small
        .verify_scoped(c, &mut RootWorkspace::new(), seq())
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    let zero_inverse = [0.];
    let zero_inverse =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &zero_inverse).unwrap()).unwrap();
    let plan = SystemRootVerifier::new(&function, &zero_inverse, opts()).unwrap();
    assert!(!plan.verify_scoped(c, &mut work, seq()).unwrap().is_proved());
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            seq().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let report = plan.verify_scoped(c, &mut work, &mut context).unwrap();
    assert!(matches!(
        report.status(),
        RootVerificationStatus::Inconclusive(RootInconclusive::Stopped(_))
    ));
    assert_eq!(
        context.counts().total().unwrap().total_providers().unwrap(),
        0
    );
}

#[test]
fn clustered_complex_roots_scaling_and_bad_scalar_extensions() {
    // (z-(1+i))^2: exact Gaussian-integer coefficients, repeated nonreal root.
    let coefficients = [
        Complex::new(0., 2.),
        Complex::new(-2., -2.),
        Complex::new(1., 0.),
    ];
    for exponent in [-1000, 0, 1000] {
        let scaled = coefficients
            .map(|c| Complex::new(c.re * 2.0_f64.powi(exponent), c.im * 2.0_f64.powi(exponent)));
        let plan = PolynomialRootPlan::new(&scaled, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let disk = RootDisk {
            center: Complex::new(1., 1.),
            radius: 0.03125,
        };
        assert!(
            plan.verify_scoped(
                PolynomialRootCertificate { disk, count: 2 },
                &mut work,
                seq()
            )
            .unwrap()
            .is_proved()
        );
    }
    // Distinct roots 1 +/- 2^-10, certified as one cluster.
    let coefficients = [1. - 2.0_f64.powi(-20), -2., 1.];
    let plan = PolynomialRootPlan::new(&coefficients, opts()).unwrap();
    let seeds = [
        RootDisk {
            center: Complex::new(1. - 2.0_f64.powi(-10), 0.),
            radius: 0.125,
        },
        RootDisk {
            center: Complex::new(1. + 2.0_f64.powi(-10), 0.),
            radius: 0.125,
        },
    ];
    let mut work = plan.workspace(ProofRequest::refinement(2)).unwrap();
    let mut cover = PolynomialRootCover::new(2).unwrap();
    let report = plan
        .isolate_into(&seeds, &mut cover, &mut work, seq())
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(report.certificates(), 1);
    assert_eq!(cover.view().len(), 1);
    assert_eq!(cover.view()[0].count, 2);
    let empty = |_: D<f64>| Ok(D::empty());
    let weak = |x: D<f64>| D::with_decoration(x.set(), Decoration::Def);
    let one = |_: D<f64>| Ok(D::point(r(1.)).unwrap());
    let c = ScalarRootCertificate {
        region: D::new(r(-1.), r(1.)).unwrap().set(),
        center: 0.,
        claim: RootClaim::Unique,
    };
    for value in [
        &empty as &(dyn crate::IntervalScalarFunction<f64> + Sync),
        &weak,
    ] {
        assert_eq!(
            ScalarRootVerifier::new(value, &one, opts())
                .verify(c)
                .unwrap()
                .status(),
            RootVerificationStatus::Inconclusive(RootInconclusive::Reason(
                RootProofFailure::Domain
            ))
        );
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
fn malformed_executor_dispatch_is_rejected() {
    let value = |x: D<f64>| Ok(x);
    let derivative = |_: D<f64>| Ok(D::point(r(1.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let proposals = [ScalarRootCertificate {
        region: D::new(r(-1.), r(1.)).unwrap().set(),
        center: 0.,
        claim: RootClaim::Unique,
    }; 2];
    let blank = RootVerificationReport::not_run();
    for duplicate in [false, true] {
        let executor = FaultyExecutor(duplicate);
        let mut w = plan.workspace(ProofRequest::proofs(2)).unwrap();
        let mut reports = [blank; 2];
        let mut marked = reports;
        marked[0] =
            RootVerificationReport::unknown(RootInconclusive::Reason(RootProofFailure::Inequality));
        reports[0] = marked[0];
        assert!(
            plan.verify_batch_into(
                &proposals,
                &mut reports,
                &mut w,
                ExecutionContext::new(&executor)
            )
            .is_err()
        );
        assert_eq!(reports, marked);
        assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    }
    let mut short = [blank; 1];
    let error = plan
        .verify_batch_into(&proposals, &mut short, &mut RootWorkspace::new(), seq())
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
}
#[cfg(feature = "std")]
#[test]
fn executor_equivalence_and_stable_budget_partitioning() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let pool = ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
    let coefficients = [-1., 0., 1.];
    let proposals = [-1., 1., 4.].map(|x| PolynomialRootCertificate {
        disk: RootDisk {
            center: Complex::new(x, 0.),
            radius: 0.125,
        },
        count: 1,
    });
    for limit in [1, 2, 3, 100] {
        let mut reference = None;
        for workers in [1, 2, 3] {
            let execution = if workers == 1 {
                ExecutionContext::sequential()
            } else {
                ExecutionContext::new(&pool)
            };
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let mut context = state
                .begin_solve(execution, identity(), &mut budgets)
                .unwrap();
            let policy = if workers == 1 {
                ExecutionPolicy::Sequential
            } else {
                ExecutionPolicy::Parallel(NonZeroUsize::new(workers).unwrap())
            };
            let plan = PolynomialRootPlan::new(&coefficients, opts().with_execution_policy(policy))
                .unwrap();
            let mut w = plan.workspace(ProofRequest::proofs(3)).unwrap();
            let mut reports = [PolynomialVerificationReport::not_run(); 3];
            plan.verify_batch_into(&proposals, &mut reports, &mut w, &mut context)
                .unwrap();
            let counts = context.counts();
            if let Some((old_reports, old_counts)) = &reference {
                assert_eq!(&reports, old_reports);
                assert_eq!(&counts, old_counts);
            } else {
                reference = Some((reports, counts));
            }
        }
    }
}

fn independent_fixtures<S: NumericScalar>(width: &str) {
    let mut cases = 0;
    for line in include_str!("../../tests/data/roots.txt")
        .lines()
        .filter(|l| l.starts_with("poly "))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[1] != width || (fields[2] == "1" && S::IS_REAL) {
            continue;
        }
        let decode = |text: &str| {
            let bits = u64::from_str_radix(text, 16).unwrap();
            r::<S::Real>(if width == "32" {
                f64::from(f32::from_bits(bits as u32))
            } else {
                f64::from_bits(bits)
            })
        };
        let degree: usize = fields[4].parse().unwrap();
        let expected: isize = fields[5].parse().unwrap();
        let coefficients: Vec<_> = (0..=degree)
            .map(|i| S::from_components(decode(fields[9 + 2 * i]), decode(fields[10 + 2 * i])))
            .collect();
        let plan = PolynomialRootPlan::new(&coefficients, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let disk = RootDisk {
            center: Complex::new(decode(fields[6]), decode(fields[7])),
            radius: decode(fields[8]),
        };
        let claim = PolynomialRootCertificate {
            disk,
            count: expected.max(0) as usize,
        };
        let report = plan.verify_scoped(claim, &mut work, seq()).unwrap();
        if expected < 0 {
            assert!(
                matches!(report.status(), RootVerificationStatus::Inconclusive(_)),
                "{line}: {report:?}"
            );
        } else {
            assert!(report.is_proved(), "{line}: {report:?}");
            assert_eq!(report.count(), Some(expected as usize));
            let false_claim = PolynomialRootCertificate {
                disk,
                count: expected as usize + 1,
            };
            assert_eq!(
                plan.verify_scoped(false_claim, &mut work, seq())
                    .unwrap()
                    .status(),
                RootVerificationStatus::Refuted
            );
        }
        cases += 1;
    }
    assert_eq!(cases, if !S::IS_REAL { 10 } else { 8 });
}
#[test]
fn exact_reference_real32() {
    independent_fixtures::<f32>("32");
}
#[test]
fn exact_reference_real64() {
    independent_fixtures::<f64>("64");
}
#[test]
fn exact_reference_complex32() {
    independent_fixtures::<Complex<f32>>("32");
}
#[test]
fn exact_reference_complex64() {
    independent_fixtures::<Complex<f64>>("64");
}

#[test]
fn singular_jacobian_and_shared_boundary_do_not_invent_unique_counts() {
    let entries = [1.];
    let targets = [0.];
    let mixing =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &entries).unwrap()).unwrap();
    let function = Squares {
        mixing: &mixing,
        targets: &targets,
        mode: 0,
    };
    let plan = SystemRootVerifier::new(&function, &mixing, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
    let center = [0.];
    let region = [pack(Box2 {
        re: D::new(r(-0.125), r(0.125)).unwrap(),
        im: D::point(r(0.)).unwrap(),
    })];
    let report = plan
        .verify_scoped(
            SystemRootCertificateView {
                center: &center,
                region: &region,
                claim: RootClaim::Unique,
            },
            &mut work,
            seq(),
        )
        .unwrap();
    assert!(matches!(
        report.status(),
        RootVerificationStatus::Inconclusive(_)
    ));
    assert_eq!(report.evidence(), Some(RootClaim::Exists));
    let value = |x: D<f64>| Ok(x);
    let derivative = |_: D<f64>| Ok(D::point(r(1.)).unwrap());
    let plan = ScalarRootVerifier::new(&value, &derivative, opts());
    let nodes = [
        RootIsolationNode::Split {
            axis: 0,
            point: 0.,
            left: 1,
        },
        RootIsolationNode::Leaf(Some(RootClaim::Unique)),
        RootIsolationNode::Leaf(Some(RootClaim::Unique)),
    ];
    let regions = [(-1., 1.), (-1., 0.), (0., 1.)].map(|(lo, hi)| {
        pack(Box2 {
            re: D::new(r(lo), r(hi)).unwrap(),
            im: D::point(r(0.)).unwrap(),
        })
    });
    let report = plan
        .verify_isolation(
            D::new(r(-1.), r(1.)).unwrap().set(),
            RootIsolationView {
                nodes: &nodes,
                regions: &regions,
            },
        )
        .unwrap();
    assert_eq!(report.isolated(), 2);
    assert!(report.is_complete());
}

#[test]
fn dense_system_order_64() {
    let n = 64;
    let entries: Vec<_> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if i == j {
                    2.0_f64
                } else {
                    ((i * 17 + j * 13) % 23) as f64 / 2048.0 - 11.0 / 2048.0
                }
            })
        })
        .collect();
    let mixing = Matrix::from_vec(n, n, entries.clone()).unwrap();
    let mixing = DenseIntervalOperator::point(mixing.view()).unwrap();
    let jacobian = Matrix::from_vec(n, n, entries.iter().map(|x| 2.0 * x).collect()).unwrap();
    let factor = LuDecomposition::factor(&jacobian).unwrap();
    let inverse = LuIntervalInverse::from_dense(
        &factor,
        crate::ValidatedOptions::new()
            .with_byte_limit(LuIntervalInverse::<f64>::dense_storage_bytes(n).unwrap()),
    )
    .unwrap();
    let targets = vec![1.0; n];
    let function = Squares {
        mixing: &mixing,
        targets: &targets,
        mode: 0,
    };
    let plan = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::proofs(1)).unwrap();
    let center = vec![1.0; n];
    let region =
        vec![
            pack(Box2::real(D::new(r(1.0 - 1.0 / 128.0), r(1.0 + 1.0 / 128.0)).unwrap()).unwrap());
            n
        ];
    let report = plan
        .verify_scoped(
            SystemRootCertificateView {
                region: &region,
                center: &center,
                claim: RootClaim::Unique,
            },
            &mut work,
            seq(),
        )
        .unwrap();
    assert!(report.is_proved(), "{report:?}");
}

#[test]
fn unallocatable_refinement_fails_cleanly() {
    let value = |x: D<f64>| Ok(x);
    let plan = ScalarRootVerifier::new(&value, &value, opts());
    let request = ProofRequest::refinement(1 << 55);
    assert_eq!(
        plan.workspace(request).err().unwrap().kind(),
        SciOxideErrorKind::AllocationFailure
    );
    let mut workspace = RootWorkspace::<f64>::new();
    let requirements = plan.workspace_requirements(request).unwrap();
    assert_eq!(
        workspace.try_reserve(&requirements).err().unwrap().kind(),
        SciOxideErrorKind::AllocationFailure
    );
}

#[test]
fn reused_polynomial_workspace_grows_only_requested_lanes() {
    let small = [-2., 0., 1.];
    let mut large = vec![0.; 201];
    large[0] = -1.;
    large[200] = 1.;
    let small_plan = PolynomialRootPlan::new(&small, opts()).unwrap();
    let large_plan = PolynomialRootPlan::new(&large, opts()).unwrap();
    let small_request = small_plan
        .workspace_requirements(ProofRequest::proofs(64))
        .unwrap();
    let large_request = large_plan
        .workspace_requirements(ProofRequest::proofs(1))
        .unwrap();
    let limit = small_request.required_bytes() + large_request.required_bytes();
    let limited = PolynomialRootPlan::new(&large, opts().with_byte_limit(limit)).unwrap();
    let mut workspace = small_plan.workspace(ProofRequest::proofs(64)).unwrap();
    let certificate = PolynomialRootCertificate {
        disk: RootDisk {
            center: Complex::new(1., 0.),
            radius: 0.0001,
        },
        count: 1,
    };
    assert!(
        limited
            .verify_scoped(certificate, &mut workspace, seq())
            .unwrap()
            .is_proved()
    );
    {
        assert!(
            limited
                .verify_scoped(certificate, &mut workspace, seq())
                .unwrap()
                .is_proved()
        );
    };
    assert_eq!(
        workspace.check(&large_request).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
}

struct CountedInverse<'a> {
    inner: DenseIntervalOperator<'a, f64>,
    calls: &'a core::sync::atomic::AtomicUsize,
    scratch: Option<usize>,
    change_at: Option<usize>,
}

#[test]
fn changed_requirements_after_residual_jacobian_and_inverse_are_rejected() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let coefficients = [1.0];
    let preconditioner = [1.0 / 3.0];
    let targets = [2.0];
    let region = [pack(Box2 {
        re: D::new(r(1.0), r(2.0)).unwrap(),
        im: D::point(r(0.0)).unwrap(),
    })];
    let center = [1.5];
    let certificate = SystemRootCertificateView {
        region: &region,
        center: &center,
        claim: RootClaim::Unique,
    };
    for (mixing_change, inverse_change) in [
        (Some(1), None),
        (Some(3), None),
        (None, Some(1)),
        (None, Some(2)),
    ] {
        let mixing_calls = AtomicUsize::new(0);
        let inverse_calls = AtomicUsize::new(0);
        let mixing = CountedInverse {
            inner: DenseIntervalOperator::point(
                MatrixView::from_slice(1, 1, &coefficients).unwrap(),
            )
            .unwrap(),
            calls: &mixing_calls,
            scratch: None,
            change_at: mixing_change,
        };
        let function = Squares {
            mixing: &mixing,
            targets: &targets,
            mode: 0,
        };
        let inverse = CountedInverse {
            inner: DenseIntervalOperator::point(
                MatrixView::from_slice(1, 1, &preconditioner).unwrap(),
            )
            .unwrap(),
            calls: &inverse_calls,
            scratch: None,
            change_at: inverse_change,
        };
        let plan = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
        let mut workspace = plan.workspace(ProofRequest::proofs(1)).unwrap();
        let mut reports = [RootVerificationReport::not_run()];
        let stored = reports;
        let error = plan
            .verify_batch_into(&[certificate], &mut reports, &mut workspace, seq())
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        assert_eq!(reports, stored);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
        if let Some(count) = mixing_change {
            assert_eq!(mixing_calls.load(Ordering::Relaxed), count);
        }
        if let Some(count) = inverse_change {
            assert_eq!(inverse_calls.load(Ordering::Relaxed), count);
        }
    }
}
impl scioxide_core::LinearOperator<f64> for CountedInverse<'_> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], work: &mut [f64]) -> Result<()> {
        self.calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        self.inner.apply_into(x, y, work)
    }
}
impl IntervalLinearOperator<f64> for CountedInverse<'_> {
    fn enclosure_workspace_len(&self) -> Result<usize> {
        if self
            .change_at
            .is_some_and(|count| self.calls.load(core::sync::atomic::Ordering::Relaxed) >= count)
        {
            return Ok(self.inner.enclosure_workspace_len()? + 1);
        }
        self.scratch
            .map_or_else(|| self.inner.enclosure_workspace_len(), Ok)
    }
    fn apply_enclosure_into(
        &self,
        x: &[IntervalBox<f64>],
        b: Option<&[IntervalBox<f64>]>,
        y: &mut [IntervalBox<f64>],
        work: &mut [IntervalBox<f64>],
    ) -> Result<()> {
        self.calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        self.inner.apply_enclosure_into(x, b, y, work)
    }
}

#[test]
fn malformed_later_scalar_and_system_certificates_are_preflight_rejections() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let value = |x: D<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        x.square()?.sub(&D::point(r(2.)).unwrap())
    };
    let derivative = |x: D<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        x.mul(&D::point(r(2.)).unwrap())
    };
    let scalar = ScalarRootVerifier::new(&value, &derivative, opts());
    let good = ScalarRootCertificate {
        region: D::new(r(1.), r(2.)).unwrap().set(),
        center: 1.5,
        claim: RootClaim::Unique,
    };
    let mut workspace = RootWorkspace::new();
    let mut reports = [RootVerificationReport::not_run(); 3];
    scalar
        .verify_batch_into(&[good; 3], &mut reports, &mut workspace, seq())
        .unwrap();
    assert!(reports.iter().all(RootVerificationReport::is_proved));
    let stored = reports;
    for bad in [
        ScalarRootCertificate {
            center: f64::NAN,
            ..good
        },
        ScalarRootCertificate { center: 3., ..good },
        ScalarRootCertificate {
            region: ExtendedInterval::empty(),
            ..good
        },
        ScalarRootCertificate {
            region: ExtendedInterval::entire(),
            ..good
        },
    ] {
        calls.store(0, Ordering::Relaxed);
        assert!(
            scalar
                .verify_batch_into(&[good, good, bad], &mut reports, &mut workspace, seq())
                .is_err()
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(reports, stored);
        assert_preflight(&workspace);
    }
    let coefficients = [1.];
    let inverse_coefficients = [1. / 3.];
    let mixing = CountedInverse {
        inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &coefficients).unwrap())
            .unwrap(),
        calls: &calls,
        scratch: None,
        change_at: None,
    };
    let inverse = CountedInverse {
        inner: DenseIntervalOperator::point(
            MatrixView::from_slice(1, 1, &inverse_coefficients).unwrap(),
        )
        .unwrap(),
        calls: &calls,
        scratch: None,
        change_at: None,
    };
    let function = Squares {
        mixing: &mixing,
        targets: &[2.],
        mode: 0,
    };
    let system = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
    let center = [1.5];
    let region = [interval_enclosure(1.25, 1.75).unwrap()];
    let good = SystemRootCertificateView {
        center: &center,
        region: &region,
        claim: RootClaim::Unique,
    };
    system
        .verify_batch_into(&[good; 3], &mut reports, &mut workspace, seq())
        .unwrap();
    assert!(reports.iter().all(RootVerificationReport::is_proved));
    let stored = reports;
    let nan = [f64::NAN];
    let outside = [2.];
    let empty = [sentinel()];
    let unbounded = [Complex::new(
        ExtendedInterval::entire(),
        D::point(r(0.)).unwrap().set(),
    )];
    let complex = [Complex::new(region[0].re, D::point(r(1.)).unwrap().set())];
    for bad in [
        SystemRootCertificateView {
            center: &nan,
            ..good
        },
        SystemRootCertificateView {
            center: &outside,
            ..good
        },
        SystemRootCertificateView {
            center: &[],
            ..good
        },
        SystemRootCertificateView {
            region: &empty,
            ..good
        },
        SystemRootCertificateView {
            region: &unbounded,
            ..good
        },
        SystemRootCertificateView {
            region: &complex,
            ..good
        },
    ] {
        calls.store(0, Ordering::Relaxed);
        assert!(
            system
                .verify_batch_into(&[good, good, bad], &mut reports, &mut workspace, seq())
                .is_err()
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(reports, stored);
        assert_preflight(&workspace);
    }
}

fn assert_preflight(workspace: &RootWorkspace<f64>) {
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
    assert_eq!(
        attempt.counts().total().unwrap().total_providers().unwrap(),
        0
    );
    assert_eq!(attempt.counts().total().unwrap().iterations(), 0);
}

#[test]
fn system_requirement_overflow_is_recorded_by_every_workspace_form() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let a = [1.];
    let op = CountedInverse {
        inner: DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a).unwrap()).unwrap(),
        calls: &calls,
        scratch: None,
        change_at: None,
    };
    let huge = CountedInverse {
        scratch: Some(usize::MAX / 8),
        ..op
    };
    let function = Squares {
        mixing: &op,
        targets: &[0.],
        mode: 0,
    };
    let plan = SystemRootVerifier::new(&function, &huge, opts()).unwrap();
    let scalar = |x: D<f64>| Ok(x);
    let warm = ScalarRootVerifier::new(&scalar, &scalar, opts());
    let region = [interval_enclosure(-1., 1.).unwrap()];
    let proposal = SystemRootCertificateView {
        center: &[0.],
        region: &region,
        claim: RootClaim::Unique,
    };
    let mut workspace = RootWorkspace::new();
    let mut reports = [RootVerificationReport::not_run(); 1];
    let mut isolation = RootIsolation::new(1, 1).unwrap();
    let nodes = [RootIsolationNode::Leaf(Some(RootClaim::Unique))];
    for form in 0..4 {
        warm.verify_batch_into(
            &[ScalarRootCertificate {
                region: D::new(r(-1.), r(1.)).unwrap().set(),
                center: 0.,
                claim: RootClaim::Exists,
            }],
            &mut reports,
            &mut workspace,
            seq(),
        )
        .unwrap();
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
        calls.store(0, Ordering::Relaxed);
        let error = match form {
            0 => plan
                .verify_scoped(proposal, &mut workspace, seq())
                .map(|_| ()),
            1 => plan.verify_batch_into(&[proposal], &mut reports, &mut workspace, seq()),
            2 => plan
                .isolate_into(
                    &region,
                    StepTolerance::default(),
                    &mut isolation,
                    &mut workspace,
                    seq(),
                )
                .map(|_| ()),
            _ => plan
                .verify_isolation_scoped(
                    &region,
                    RootIsolationView {
                        nodes: &nodes,
                        regions: &region,
                    },
                    &mut workspace,
                    seq(),
                )
                .map(|_| ()),
        }
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_preflight(&workspace);
    }
}

#[test]
fn malformed_later_isolation_leaf_is_rejected_before_any_proof() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let callback = |x: D<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x)
    };
    let plan = ScalarRootVerifier::new(&callback, &callback, opts());
    let nodes = [
        RootIsolationNode::Split {
            axis: 0,
            point: 0.,
            left: 1,
        },
        RootIsolationNode::Leaf(Some(RootClaim::Empty)),
        RootIsolationNode::Leaf(Some(RootClaim::Exists)),
    ];
    let regions = [
        interval_enclosure(-1., 1.).unwrap(),
        interval_enclosure(-1., 0.).unwrap(),
        interval_enclosure(0., 1.).unwrap(),
    ];
    let mut workspace = RootWorkspace::new();
    assert!(
        plan.verify_isolation_scoped(
            regions[0].re,
            RootIsolationView {
                nodes: &nodes,
                regions: &regions
            },
            &mut workspace,
            seq()
        )
        .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_preflight(&workspace);
}

#[test]
fn scalar_shared_endpoints_complete_and_replay() {
    let one = D::point(1.0_f64).unwrap();
    let two = D::point(2.0).unwrap();
    let f = |x: D<f64>| x.square()?.sub(&one);
    let df = |x: D<f64>| x.mul(&two);
    let verifier = ScalarRootVerifier::new(&f, &df, opts());
    let domain = D::new(-2.0, 2.0).unwrap().set();
    let (report, tree) = verifier
        .isolate(domain, StepTolerance::default(), 63)
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(
        (report.nodes(), report.isolated(), report.excluded()),
        (11, 4, 2)
    );
    assert_eq!(tree.isolated_regions().count(), 4);
    assert!(
        verifier
            .verify_isolation(domain, tree.view())
            .unwrap()
            .is_complete()
    );
    assert_eq!(
        RootIsolation::<f64>::new(1, 1)
            .unwrap()
            .isolated_regions()
            .count(),
        0
    );
    // Disjoint leaves still charge all node tests and all unique-leaf pairs.
    let f = |x: D<f64>| x.square()?.sub(&two);
    let verifier = ScalarRootVerifier::new(&f, &df, opts());
    let mut tree = RootIsolation::new(1, 63).unwrap();
    let mut workspace = RootWorkspace::new();
    let report = verifier
        .isolate_into(
            domain,
            StepTolerance::default(),
            &mut tree,
            &mut workspace,
            seq(),
        )
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(report.isolated(), 2);
    let counts = workspace.last_attempt().unwrap().counts();
    let mut replay = RootWorkspace::new();
    assert!(
        verifier
            .verify_isolation_scoped(domain, tree.view(), &mut replay, seq())
            .unwrap()
            .is_complete()
    );
    assert_eq!(counts.own().iterations(), 2 * report.nodes() + 1);
    assert_eq!(
        replay.last_attempt().unwrap().counts().own().iterations(),
        report.nodes() + report.isolated() + report.excluded() + 1
    );
}

struct IdentityMap(usize);
impl ResidualOperator<f64> for IdentityMap {
    fn input_dimension(&self) -> usize {
        self.0
    }
    fn output_dimension(&self) -> usize {
        self.0
    }
    fn residual_workspace_len(&self) -> Result<usize> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<()> {
        y.copy_from_slice(x);
        Ok(())
    }
}
impl JacobianOperator<f64> for IdentityMap {
    fn jvp_workspace_len(&self) -> Result<usize> {
        Ok(0)
    }
    fn jvp_into(&self, _: &[f64], v: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<()> {
        y.copy_from_slice(v);
        Ok(())
    }
}
impl IntervalResidualOperator<f64> for IdentityMap {
    fn residual_enclosure_workspace_len(&self) -> Result<usize> {
        Ok(0)
    }
    fn residual_enclosure_into(
        &self,
        x: &[IntervalBox<f64>],
        y: &mut [IntervalBox<f64>],
        _: &mut [IntervalBox<f64>],
    ) -> Result<()> {
        y.copy_from_slice(x);
        Ok(())
    }
}
impl IntervalJacobianOperator<f64> for IdentityMap {
    fn jvp_enclosure_workspace_len(&self) -> Result<usize> {
        Ok(0)
    }
    fn jvp_enclosure_into(
        &self,
        _: &[IntervalBox<f64>],
        v: &[IntervalBox<f64>],
        y: &mut [IntervalBox<f64>],
        _: &mut [IntervalBox<f64>],
    ) -> Result<()> {
        y.copy_from_slice(v);
        Ok(())
    }
}
#[test]
fn system_shared_faces_stay_inconclusive_and_regions_keep_order() {
    let entries = [1.0];
    let inverse =
        DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &entries).unwrap()).unwrap();
    let function = IdentityMap(1);
    let verifier = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
    let nodes = [
        RootIsolationNode::Split {
            axis: 0,
            point: 0.0,
            left: 1,
        },
        RootIsolationNode::Leaf(Some(RootClaim::Unique)),
        RootIsolationNode::Leaf(Some(RootClaim::Unique)),
    ];
    let regions = [
        interval_enclosure(-1., 1.).unwrap(),
        interval_enclosure(-1., 0.).unwrap(),
        interval_enclosure(0., 1.).unwrap(),
    ];
    let report = verifier
        .verify_isolation(
            &regions[..1],
            RootIsolationView {
                nodes: &nodes,
                regions: &regions,
            },
        )
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(report.isolated(), 2);
    let entries = [1.0, 0.0, 0.0, 1.0];
    let inverse =
        DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &entries).unwrap()).unwrap();
    let function = IdentityMap(2);
    let verifier = SystemRootVerifier::new(&function, &inverse, opts()).unwrap();
    let domain = [
        interval_enclosure(-1., 1.).unwrap(),
        interval_enclosure(-2., 2.).unwrap(),
    ];
    let (report, tree) = verifier
        .isolate(&domain, StepTolerance::default(), 7)
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(
        tree.isolated_regions().collect::<Vec<_>>(),
        vec![domain.as_slice()]
    );
    assert_eq!(
        RootIsolation::<f64>::new(2, 1)
            .unwrap()
            .isolated_regions()
            .count(),
        0
    );
}

#[test]
fn scalar_verifier_calls_the_scoped_override() {
    struct Scoped;
    impl crate::IntervalScalarFunction<f64> for Scoped {
        fn enclose(&self, _: D<f64>) -> Result<D<f64>> {
            Err(crate::shell::breakdown("unsuffixed callback must not run"))
        }
        fn enclose_scoped(&self, x: D<f64>, _: &mut SolveContext<'_, '_>) -> Result<D<f64>> {
            Ok(x)
        }
    }
    let derivative = |_: D<f64>| D::point(1.0);
    let verifier = ScalarRootVerifier::new(&Scoped, &derivative, opts());
    let report = verifier
        .verify(ScalarRootCertificate {
            region: D::new(-1., 1.).unwrap().set(),
            center: 0.,
            claim: RootClaim::Unique,
        })
        .unwrap();
    assert!(report.is_proved());
}
