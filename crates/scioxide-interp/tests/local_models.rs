//! Independent polynomial identities, closed-form covariance and deletion refits.
#![allow(clippy::unwrap_used)]
mod support;

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex, ConvergenceReason, ExecutionContext,
    ExecutionPolicy, NumericScalar, RankThreshold, RealScalar, SciOxideErrorKind,
    SciOxideErrorView, SolveBudgets,
};
use scioxide_interp::{
    BatchEvaluator, BatchRequest, Covariance, CovarianceKernel, ExtrapolationPolicy,
    GaussianProcessModel, Interpolant, KrigingMean, LocalModelOptions, LocalModelPlan,
    LocalPolynomialModel, LocalPolynomialOptions, LocalRbfModel, LocalRbfOptions, LocalSamples,
    RbfKernel, RbfRegularization,
};
use support::{assert_close, from_f64};

fn numerical<S: RealScalar>() -> LocalModelOptions<S> {
    LocalModelOptions::new()
        .with_rank_threshold(
            RankThreshold::new(from_f64(0.), from_f64::<S>(16.) * S::EPSILON).unwrap(),
        )
        .with_minimum_rcond(from_f64::<S>(32.) * S::EPSILON)
        .with_max_system_order(64)
}
fn poly_options<S: RealScalar>(
    degree: usize,
    minimum: usize,
    maximum: usize,
) -> LocalPolynomialOptions<S> {
    LocalPolynomialOptions::new(degree, minimum, maximum).with_numerical(numerical())
}
fn rbf_options<S: RealScalar>(neighbors: usize, kernel: RbfKernel<S>) -> LocalRbfOptions<S> {
    LocalRbfOptions::new(neighbors, kernel)
        .unwrap()
        .with_numerical(numerical())
}

fn polynomial<S: RealScalar + NumericScalar<Real = S>>() {
    let tolerance = 400. * <S as RealScalar>::EPSILON.into_f64();
    let points = [
        [-1., -1.],
        [-1., 0.],
        [-1., 1.],
        [0., -1.],
        [0., 0.],
        [0., 1.],
        [1., -1.],
        [1., 0.],
        [1., 1.],
    ]
    .map(|p| p.map(from_f64::<S>));
    let values = points
        .map(|[x, y]| Complex::new(from_f64::<S>(2.) + x * x + from_f64::<S>(3.) * y, x - y * y));
    let data = LocalSamples::new(&points, &values).unwrap();
    let points = data.points().unwrap();
    let plan = LocalModelPlan::<S, 2>::new(&points, data.values().unwrap()).unwrap();
    let model = LocalPolynomialModel::new(&plan, [from_f64(0.); 2], poly_options(2, 3, 9)).unwrap();
    assert_eq!(model.fit_report().degree(), Some(2));
    assert_eq!(model.fit_report().neighbors(), 6);
    assert_eq!(model.fit_report().attempts(), 2);
    let mut workspace = model.workspace().unwrap();
    let mut out = [from_f64(0.); 2];
    for [x, y] in [[0.2, -0.3], [-0.8, 0.4], [1., -1.]].map(|p| p.map(from_f64::<S>)) {
        model
            .evaluate_into(
                [x, y],
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut workspace,
            )
            .unwrap();
        assert_close::<S, S>(
            out[0],
            from_f64::<S>(2.) + x * x + from_f64::<S>(3.) * y,
            tolerance,
            tolerance,
        );
        assert_close::<S, S>(out[1], x - y * y, tolerance, tolerance);
        // Owned evaluation stages the same components.
        let owned = model.evaluate([x, y], ExtrapolationPolicy::Reject).unwrap();
        assert_eq!(owned, out.to_vec());
    }
}
#[test]
fn polynomial_both_precisions_complex() {
    polynomial::<f64>();
    polynomial::<f32>();
}
#[test]
fn rank_deficiency_reduces_degree_and_preserves_deterministic_ties() {
    let data = LocalSamples::new(&[[-1., 0.], [0., 0.], [1., 0.]], &[2., 2., 2.]).unwrap();
    let points = data.points().unwrap();
    let plan = LocalModelPlan::<f64, 2>::new(&points, data.values().unwrap()).unwrap();
    let model = LocalPolynomialModel::new(&plan, [0., 0.], poly_options(1, 2, 3)).unwrap();
    assert_eq!(model.fit_report().degree(), Some(0));
    assert_eq!(model.sample_indices(), &[1, 0]);
    assert_eq!(model.fit_report().attempts(), 3);
}
fn rbf<S: RealScalar + NumericScalar<Real = S>>() {
    let tolerance = 400. * <S as RealScalar>::EPSILON.into_f64();
    let points =
        [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [0., 0.]].map(|p| p.map(from_f64::<S>));
    let values = points.map(|[x, y]| from_f64::<S>(2.) + x - from_f64::<S>(3.) * y);
    let data = LocalSamples::new(&points, &values).unwrap();
    let points = data.points().unwrap();
    let plan = LocalModelPlan::<S, 2>::new(&points, data.values().unwrap()).unwrap();
    for kernel in [
        RbfKernel::ThinPlateSpline,
        RbfKernel::Gaussian(from_f64(1.)),
        RbfKernel::InverseMultiquadric(from_f64(1.)),
        RbfKernel::Multiquadric(from_f64(1.)),
    ] {
        let model = LocalRbfModel::new(&plan, [from_f64(0.); 2], rbf_options(5, kernel)).unwrap();
        let mut w = model.workspace().unwrap();
        let mut out = [from_f64(0.)];
        for [x, y] in [[-1., -1.], [1., 1.], [0., 0.]].map(|p| p.map(from_f64::<S>)) {
            model
                .evaluate_into([x, y], ExtrapolationPolicy::Reject, &mut out, &mut w)
                .unwrap();
            assert_close::<S, S>(
                out[0],
                from_f64::<S>(2.) + x - from_f64::<S>(3.) * y,
                tolerance,
                tolerance,
            );
        }
        if matches!(kernel, RbfKernel::ThinPlateSpline) {
            model
                .evaluate_into(
                    [from_f64(0.2), from_f64(-0.3)],
                    ExtrapolationPolicy::Reject,
                    &mut out,
                    &mut w,
                )
                .unwrap();
            assert_close::<S, S>(out[0], from_f64(3.1), tolerance, tolerance);
        }
    }
}
#[test]
fn all_rbf_kernels_both_precisions() {
    rbf::<f32>();
    rbf::<f64>();
}
fn gp_closed<S: RealScalar + NumericScalar<Real = S>>() {
    let tolerance = 400. * <S as RealScalar>::EPSILON.into_f64();
    let data = LocalSamples::new(
        &[[from_f64::<S>(0.)], [from_f64(1.)]],
        &[
            Complex::new(from_f64(1.), from_f64(2.)),
            Complex::new(from_f64(3.), from_f64(-1.)),
        ],
    )
    .unwrap();
    let points = data.points().unwrap();
    let covariance =
        Covariance::new(CovarianceKernel::Matern12, from_f64(2.), [from_f64(1.)]).unwrap();
    for ordinary in [false, true] {
        let means = [from_f64(0.); 2];
        let model = GaussianProcessModel::<S, 1>::with_options(
            &points,
            data.values().unwrap(),
            covariance,
            &[from_f64(0.2); 2],
            if ordinary {
                KrigingMean::Ordinary
            } else {
                KrigingMean::Constant(&means)
            },
            numerical(),
        )
        .unwrap();
        let prediction = model.prediction(from_f64(0.3)).unwrap();
        let mut w = prediction.workspace().unwrap();
        // Mean components, latent variance, observation variance.
        let mut out = [from_f64(0.); 4];
        prediction
            .evaluate_into(
                [from_f64(0.5)],
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut w,
            )
            .unwrap();
        let a = from_f64::<S>(2.2);
        let b = from_f64::<S>(2.) * (-from_f64::<S>(1.)).exp();
        let k = from_f64::<S>(2.) * (-from_f64::<S>(0.5)).exp();
        let sum = a + b;
        if ordinary {
            assert_close::<S, S>(out[0], from_f64(2.), tolerance, tolerance);
            assert_close::<S, S>(out[1], from_f64(0.5), tolerance, tolerance);
        } else {
            assert_close::<S, S>(out[0], k * from_f64::<S>(4.) / sum, tolerance, tolerance);
            assert_close::<S, S>(out[1], k / sum, tolerance, tolerance);
        }
        let variance = from_f64::<S>(2.) - from_f64::<S>(2.) * k * k / sum
            + if ordinary {
                (from_f64::<S>(1.) - from_f64::<S>(2.) * k / sum)
                    * (from_f64::<S>(1.) - from_f64::<S>(2.) * k / sum)
                    * sum
                    / from_f64::<S>(2.)
            } else {
                from_f64(0.)
            };
        assert_close::<S, S>(out[2], variance, tolerance, tolerance);
        assert_close::<S, S>(out[3], variance + from_f64::<S>(0.3), tolerance, tolerance);
    }
}
#[test]
fn closed_form_gp_complex_both_precisions() {
    gp_closed::<f64>();
    gp_closed::<f32>();
}
#[test]
fn loo_matches_explicit_refits() {
    let tolerance = 400. * <f64 as RealScalar>::EPSILON.into_f64();
    let xs = [[-1.], [-0.3], [0.2], [0.8], [1.]];
    let ys = [1., -0.2, 0.7, 1.3, -0.5];
    let data = LocalSamples::new(&xs, &ys).unwrap();
    let points = data.points().unwrap();
    for kernel in [
        CovarianceKernel::SquaredExponential,
        CovarianceKernel::Matern12,
        CovarianceKernel::Matern32,
        CovarianceKernel::Matern52,
    ] {
        let cov = Covariance::new(kernel, 1.2, [0.7]).unwrap();
        for mean in [KrigingMean::Constant(&[0.3]), KrigingMean::Ordinary] {
            let model = GaussianProcessModel::<f64, 1>::with_options(
                &points,
                data.values().unwrap(),
                cov,
                &[0.1; 5],
                mean,
                numerical(),
            )
            .unwrap();
            let mut w = model.workspace().unwrap();
            for i in 0..5 {
                let xx: Vec<_> = xs
                    .iter()
                    .enumerate()
                    .filter_map(|(j, p)| (i != j).then_some(*p))
                    .collect();
                let yy: Vec<_> = ys
                    .iter()
                    .enumerate()
                    .filter_map(|(j, v)| (i != j).then_some(*v))
                    .collect();
                let subset = LocalSamples::new(&xx, &yy).unwrap();
                let pp = subset.points().unwrap();
                let refit = GaussianProcessModel::<f64, 1>::with_options(
                    &pp,
                    subset.values().unwrap(),
                    cov,
                    &[0.1; 4],
                    mean,
                    numerical(),
                )
                .unwrap();
                let expected = refit
                    .prediction(0.1)
                    .unwrap()
                    .evaluate(xs[i], ExtrapolationPolicy::MethodDefined)
                    .unwrap();
                let mut actual = [0.];
                let variance = model.leave_one_out_into(i, &mut actual, &mut w).unwrap();
                assert_close::<f64, f64>(actual[0], expected[0], tolerance, tolerance);
                assert_close::<f64, f64>(variance, expected[2], tolerance, tolerance);
            }
        }
    }
}
#[test]
fn exact_gp_variances_and_failures_are_transactional() {
    let tolerance = 400. * <f64 as RealScalar>::EPSILON.into_f64();
    let data = LocalSamples::new(&[[0.], [1.]], &[1., 2.]).unwrap();
    let p = data.points().unwrap();
    let cov = Covariance::new(CovarianceKernel::SquaredExponential, 1., [1.]).unwrap();
    let model = GaussianProcessModel::<f64, 1>::with_options(
        &p,
        data.values().unwrap(),
        cov,
        &[0.; 2],
        KrigingMean::Ordinary,
        numerical(),
    )
    .unwrap();
    let prediction = model.prediction(0.).unwrap();
    let mut w = prediction.workspace().unwrap();
    let mut out = [99.; 3];
    for (x, y) in [(0., 1.), (1., 2.)] {
        prediction
            .evaluate_into([x], ExtrapolationPolicy::Reject, &mut out, &mut w)
            .unwrap();
        assert_close::<f64, f64>(out[0], y, tolerance, tolerance);
        assert_close::<f64, f64>(out[1], 0., tolerance, tolerance);
    }
    out = [99.; 3];
    assert!(model.prediction(-1.).is_err());
    for point in [[f64::NAN], [2.]] {
        assert!(
            prediction
                .evaluate_into(point, ExtrapolationPolicy::Reject, &mut out, &mut w)
                .is_err()
        );
        assert_eq!(out, [99.; 3]);
    }
    let mut short = [99.; 2];
    assert!(matches!(
        prediction.evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut short, &mut w),
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert_eq!(short, [99.; 2]);
    let limited = numerical().with_maximum_bytes(0);
    let result = GaussianProcessModel::<f64, 1>::with_options(
        &p,
        data.values().unwrap(),
        cov,
        &[0.; 2],
        KrigingMean::Ordinary,
        limited,
    );
    assert!(
        matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
    );
}
#[test]
fn repeated_points_require_noise_or_smoothing() {
    let data = LocalSamples::new(&[[0.], [0.], [1.]], &[1., 1., 2.]).unwrap();
    let p = data.points().unwrap();
    let plan = LocalModelPlan::<f64, 1>::new(&p, data.values().unwrap()).unwrap();
    for smoothing in [0., 0.1] {
        let regularization = if smoothing == 0. {
            RbfRegularization::None
        } else {
            RbfRegularization::Diagonal(smoothing)
        };
        let result = LocalRbfModel::new(
            &plan,
            [0.],
            rbf_options(3, RbfKernel::Gaussian(1.)).with_regularization(regularization),
        );
        assert_eq!(result.is_ok(), smoothing > 0.);
        let cov = Covariance::new(CovarianceKernel::Matern12, 1., [1.]).unwrap();
        let gp = GaussianProcessModel::<f64, 1>::with_options(
            &p,
            data.values().unwrap(),
            cov,
            &[smoothing; 3],
            KrigingMean::Ordinary,
            numerical(),
        );
        assert_eq!(gp.is_ok(), smoothing > 0.);
    }
}

#[test]
fn strided_borrowed_data_matches_owned_and_atomic_batches() {
    let tolerance = 400. * <f64 as RealScalar>::EPSILON.into_f64();
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_spatial::PointSet;
    let coords = [0., 99., 1., 99., 2.];
    let vals = [1., 3., 5., 99., 2., 4., 6.];
    let ps =
        PointSet::new(NdView::new(&coords, StridedLayout::new(&[3, 1], &[2, 1]).unwrap()).unwrap())
            .unwrap();
    let v = NdView::new(&vals, StridedLayout::new(&[3, 2], &[1, 4]).unwrap()).unwrap();
    let plan = LocalModelPlan::<f64, 1>::new(&ps, v).unwrap();
    let poly = LocalPolynomialModel::new(&plan, [1.], poly_options(1, 3, 3).with_minimum_degree(1))
        .unwrap();
    let batch = BatchEvaluator::new(&poly, ExecutionPolicy::Sequential);
    let mut w = batch
        .workspace(BatchRequest::values(2).with_reports())
        .unwrap();
    let mut output = [99.; 5];
    let report = ExtrapolationPolicy::MethodDefined
        .resolve([scioxide_core::Interval::new(0.0, 1.0).unwrap()], [99.])
        .unwrap();
    let mut reports = [report; 2];
    for (queries, valid) in [([0.25, 1.5], true), ([0.25, 3.], false)] {
        let old = output;
        let old_reports = reports;
        let q = NdView::new(&queries, StridedLayout::new(&[2, 1], &[1, 1]).unwrap()).unwrap();
        let mut out =
            NdViewMut::new(&mut output, StridedLayout::new(&[2, 2], &[3, 1]).unwrap()).unwrap();
        let r = batch.evaluate_into(
            &q,
            ExtrapolationPolicy::Reject,
            &mut out,
            Some(&mut reports),
            &mut w,
            ExecutionContext::sequential_native(),
        );
        if valid {
            r.unwrap();
            assert_close::<f64, f64>(output[0], 1.5, tolerance, tolerance);
            assert_close::<f64, f64>(output[1], 2.5, tolerance, tolerance);
            assert_eq!(output[2], 99.);
            assert_close::<f64, f64>(output[3], 4., tolerance, tolerance);
            assert_close::<f64, f64>(output[4], 5., tolerance, tolerance);
        } else {
            assert!(r.is_err());
            assert_eq!(output, old);
            assert_eq!(reports, old_reports);
        }
    }
}
#[cfg(feature = "std")]
#[test]
fn uncertainty_executor_batches_equal_sequential() {
    let tolerance = 400. * <f64 as RealScalar>::EPSILON.into_f64();
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    let data = LocalSamples::new(&[[0.], [1.], [2.]], &[1., 0., 2.]).unwrap();
    let points = data.points().unwrap();
    let cov = Covariance::new(CovarianceKernel::Matern32, 1., [1.]).unwrap();
    let gp = GaussianProcessModel::<f64, 1>::with_options(
        &points,
        data.values().unwrap(),
        cov,
        &[0.1; 3],
        KrigingMean::Ordinary,
        numerical(),
    )
    .unwrap();
    let workers = core::num::NonZeroUsize::new(2).unwrap();
    let batch = BatchEvaluator::new(
        gp.prediction(0.2).unwrap(),
        scioxide_core::ExecutionPolicy::Parallel(workers),
    );
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let mut results = [[0.; 9]; 2];
    for (result, ctx) in results
        .iter_mut()
        .zip([ExecutionContext::sequential(), ExecutionContext::new(&pool)])
    {
        let mut workspace = batch.workspace(BatchRequest::values(3)).unwrap();
        let q = NdView::new(
            &[0.25, 0.5, 1.5],
            StridedLayout::new(&[3, 1], &[1, 1]).unwrap(),
        )
        .unwrap();
        let mut out =
            NdViewMut::new(result, StridedLayout::new(&[3, 3], &[3, 1]).unwrap()).unwrap();
        batch
            .evaluate_into(
                &q,
                ExtrapolationPolicy::Reject,
                &mut out,
                None,
                &mut workspace,
                ctx,
            )
            .unwrap();
    }
    assert_eq!(results[0], results[1]);
    for row in results[0].chunks_exact(3) {
        assert_close::<f64, f64>(row[2] - row[1], 0.2, tolerance, tolerance);
        assert!(row[1] >= 0.);
    }
}
#[test]
fn proposal_budgets_stop_adaptation_with_the_effective_report() {
    let data = LocalSamples::new(&[[-1., 0.], [0., 0.], [1., 0.]], &[1., 2., 3.]).unwrap();
    let p = data.points().unwrap();
    let plan = LocalModelPlan::<f64, 2>::new(&p, data.values().unwrap()).unwrap();
    let budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::ONE);
    let options =
        LocalPolynomialOptions::new(1, 2, 3).with_numerical(numerical().with_budgets(budgets));
    let error = LocalPolynomialModel::new(&plan, [0., 0.], options).unwrap_err();
    let SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        ..
    } = error.view()
    else {
        unreachable!("an exhausted allowance is a convergence failure: {error}")
    };
    assert_eq!(
        reason,
        ConvergenceReason::ResourceLimit(BudgetKind::Proposal)
    );
    let budget = diagnostics.budget().unwrap();
    assert_eq!(budget.kind(), BudgetKind::Proposal);
    assert_eq!(diagnostics.counts().own().proposals(), 1);
    // A sufficient allowance admits the same adaptation.
    let enough =
        SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::try_from(3).unwrap());
    let model = LocalPolynomialModel::new(
        &plan,
        [0., 0.],
        options.with_numerical(numerical().with_budgets(enough)),
    )
    .unwrap();
    assert_eq!(model.fit_report().attempts(), 3);
}
#[test]
fn owned_and_scoped_preparation_agree_and_cancellation_publishes_nothing() {
    let data = LocalSamples::new(&[[-1.], [0.], [1.], [2.]], &[1., 2., 4., 3.]).unwrap();
    let points = data.points().unwrap();
    let plan = LocalModelPlan::<f64, 1>::new(&points, data.values().unwrap()).unwrap();
    let cloned = plan.clone();
    let poly = poly_options(1, 3, 4);
    let rbf = rbf_options(4, RbfKernel::Gaussian(1.));
    let covariance = Covariance::new(CovarianceKernel::Matern12, 1., [1.]).unwrap();
    let gp = |scope: ExecutionContext<'_>| {
        GaussianProcessModel::<f64, 1>::build(
            &points,
            data.values().unwrap(),
            covariance,
            &[0.1; 4],
            KrigingMean::Ordinary,
            numerical(),
            scope,
        )
    };
    let native = ExecutionContext::sequential_native();
    let owned_poly = LocalPolynomialModel::new(&plan, [0.], poly).unwrap();
    let scoped_poly = LocalPolynomialModel::build(&cloned, [0.], poly, native).unwrap();
    let owned_rbf = LocalRbfModel::new(&plan, [0.], rbf).unwrap();
    let scoped_rbf = LocalRbfModel::build(&cloned, [0.], rbf, native).unwrap();
    let owned_gp = GaussianProcessModel::<f64, 1>::with_options(
        &points,
        data.values().unwrap(),
        covariance,
        &[0.1; 4],
        KrigingMean::Ordinary,
        numerical(),
    )
    .unwrap();
    let scoped_gp = gp(native).unwrap();
    assert_eq!(owned_poly.fit_report(), scoped_poly.fit_report());
    assert_eq!(owned_rbf.fit_report(), scoped_rbf.fit_report());
    assert_eq!(owned_gp.fit_report(), scoped_gp.fit_report());
    for q in [[-0.5], [0.25], [1.5]] {
        let policy = ExtrapolationPolicy::MethodDefined;
        assert_eq!(
            owned_poly.evaluate(q, policy).unwrap(),
            scoped_poly.evaluate(q, policy).unwrap()
        );
        assert_eq!(
            owned_rbf.evaluate(q, policy).unwrap(),
            scoped_rbf.evaluate(q, policy).unwrap()
        );
        assert_eq!(
            owned_gp
                .prediction(0.)
                .unwrap()
                .evaluate(q, policy)
                .unwrap(),
            scoped_gp
                .prediction(0.)
                .unwrap()
                .evaluate(q, policy)
                .unwrap()
        );
    }
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    let stopped = |kind: SciOxideErrorKind| kind == SciOxideErrorKind::ExecutionStopped;
    assert!(stopped(
        LocalPolynomialModel::build(&plan, [0.], poly, cancelled)
            .unwrap_err()
            .kind()
    ));
    assert!(stopped(
        LocalRbfModel::build(&plan, [0.], rbf, cancelled)
            .unwrap_err()
            .kind()
    ));
    assert!(stopped(gp(cancelled).unwrap_err().kind()));
    // Invalid centers are rejected before any epoch begins.
    assert!(matches!(
        LocalPolynomialModel::build(&plan, [f64::NAN], poly, cancelled),
        Err(error) if matches!(error.view(), SciOxideErrorView::NonFinite { .. })
    ));
}

#[test]
fn near_singular_kernel_fallback_matches_svd_and_preserves_reserved_queries() {
    use scioxide_interp::{FitConditionNorm, FitFactorization, RbfInterpolator, RbfOptions};
    use scioxide_linalg::{
        CholeskyDecomposition, CholeskyPivoting, HermitianInput, Matrix, RankPolicy,
        ThinSvdDecomposition, Vector,
    };
    let kernel = RbfKernel::Gaussian(0.02_f64);
    let coordinates = [[-1.0], [0.0], [1.0]];
    let values = [2.0, 1.0, 0.0];
    let mut entries = Vec::new();
    for p in coordinates {
        for q in coordinates {
            entries.push(kernel.evaluate((p[0] - q[0]).abs()).unwrap());
        }
    }
    let matrix = Matrix::from_vec(3, 3, entries).unwrap();
    let oracle = ThinSvdDecomposition::compute(&matrix).unwrap();
    let singular = oracle.singular_values();
    let spectral_rcond = singular[2] / singular[0];
    let direct =
        CholeskyDecomposition::factor(&matrix, HermitianInput::Exact, CholeskyPivoting::None)
            .unwrap();
    let one_rcond = direct.condition_estimate().unwrap().reciprocal().value();
    assert!(one_rcond < spectral_rcond);
    let minimum = (one_rcond + spectral_rcond) * 0.5;
    let threshold = RankThreshold::new(0.0, 1e-12).unwrap();
    let numerical = LocalModelOptions::new()
        .with_rank_threshold(threshold)
        .with_minimum_rcond(minimum);
    let samples = LocalSamples::new(&coordinates, &values).unwrap();
    let points = samples.points().unwrap();
    let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values().unwrap()).unwrap();
    let local = LocalRbfModel::new(
        &plan,
        [0.0],
        LocalRbfOptions::new(3, kernel)
            .unwrap()
            .with_numerical(numerical),
    )
    .unwrap();
    let global_options = RbfOptions::new()
        .with_rank_threshold(threshold)
        .with_minimum_rcond(minimum);
    let global = RbfInterpolator::with_options(
        coordinates.map(|[x]| [x, 0.0]).to_vec(),
        values.to_vec(),
        kernel,
        global_options,
    )
    .unwrap();
    let covariance = Covariance::new(
        CovarianceKernel::SquaredExponential,
        1.0,
        [1.0 / (2.0_f64.sqrt() * 0.02)],
    )
    .unwrap();
    let gp = GaussianProcessModel::with_options(
        &points,
        samples.values().unwrap(),
        covariance,
        &[0.0; 3],
        KrigingMean::Constant(&[0.0]),
        numerical,
    )
    .unwrap();
    for report in [local.fit_report(), gp.fit_report()] {
        assert_eq!(report.factorization(), FitFactorization::Svd);
        assert!(report.used_fallback());
        assert_eq!(report.condition_norm(), FitConditionNorm::Two);
        assert_eq!(report.rank(), 3);
        assert_eq!(report.attempts(), 1);
        assert_close::<f64, f64>(report.reciprocal_condition(), spectral_rcond, 1e-14, 1e-7);
        assert!(report.residual() < 1e-8);
    }
    assert!(global.conditioning().used_fallback());
    assert_eq!(global.conditioning().factorization(), FitFactorization::Svd);
    assert_eq!(
        global.conditioning().condition_norm(),
        FitConditionNorm::Two
    );
    assert_eq!(global.conditioning().rank_threshold(), threshold);
    assert_close::<f64, f64>(
        global.conditioning().reciprocal_condition_number(),
        spectral_rcond,
        1e-14,
        1e-7,
    );
    let weights = oracle
        .solve(
            &Vector::from_vec(values.to_vec()),
            RankPolicy::Threshold(threshold),
            0.0,
        )
        .unwrap();
    let expected: f64 = coordinates
        .iter()
        .zip(weights.as_slice())
        .map(|(p, w)| kernel.evaluate((p[0] - 0.25).abs()).unwrap() * w)
        .sum();
    let prediction = gp.prediction(0.0).unwrap();
    let mut lw = local.workspace().unwrap();
    let mut gw = global.workspace().unwrap();
    let mut pw = prediction.workspace().unwrap();
    let mut local_out = [0.0];
    let mut global_out = 0.0;
    let mut gp_out = [0.0; 3];
    {
        local
            .evaluate_into([0.25], ExtrapolationPolicy::Reject, &mut local_out, &mut lw)
            .unwrap();
        global
            .evaluate_into(
                [0.25, 0.0],
                ExtrapolationPolicy::Reject,
                &mut global_out,
                &mut gw,
            )
            .unwrap();
        prediction
            .evaluate_into([0.25], ExtrapolationPolicy::Reject, &mut gp_out, &mut pw)
            .unwrap();
    };
    for actual in [local_out[0], global_out, gp_out[0]] {
        assert_close::<f64, f64>(actual, expected, 1e-8, 1e-8);
    }
    for rcond in [f64::NAN, -0.5, 1.5] {
        assert!(
            RbfInterpolator::with_options(
                coordinates.map(|[x]| [x, 0.0]).to_vec(),
                values.to_vec(),
                kernel,
                global_options.with_minimum_rcond(rcond)
            )
            .is_err()
        );
    }
    assert!(
        RbfInterpolator::with_options(
            coordinates.map(|[x]| [x, 0.0]).to_vec(),
            values.to_vec(),
            kernel,
            global_options.with_minimum_rcond(1.0)
        )
        .is_err()
    );
    for mean in [KrigingMean::Constant(&[0.0]), KrigingMean::Ordinary] {
        let model = GaussianProcessModel::with_options(
            &points,
            samples.values().unwrap(),
            covariance,
            &[0.0; 3],
            mean,
            numerical,
        )
        .unwrap();
        assert!(model.fit_report().used_fallback());
        let mut workspace = model.workspace().unwrap();
        for index in 0..3 {
            let mut loo = [0.0];
            let variance = model
                .leave_one_out_into(index, &mut loo, &mut workspace)
                .unwrap();
            let retained_points: Vec<_> = coordinates
                .iter()
                .enumerate()
                .filter_map(|(i, p)| (i != index).then_some(*p))
                .collect();
            let retained_values: Vec<_> = values
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (i != index).then_some(*v))
                .collect();
            let retained = LocalSamples::new(&retained_points, &retained_values).unwrap();
            let refit = GaussianProcessModel::with_options(
                &retained.points().unwrap(),
                retained.values().unwrap(),
                covariance,
                &[0.0; 2],
                mean,
                numerical,
            )
            .unwrap();
            let expected = refit
                .prediction(0.0)
                .unwrap()
                .evaluate(coordinates[index], ExtrapolationPolicy::MethodDefined)
                .unwrap();
            assert_close::<f64, f64>(loo[0], expected[0], 1e-7, 1e-7);
            assert_close::<f64, f64>(variance, expected[2], 1e-12, 1e-6);
        }
    }
}
