//! Allocation, byte-budget, and transactional sampling tests.
#![allow(clippy::unwrap_used)]

use scioxide_random::*;

#[test]
fn finite_ks_tail_storage_has_a_binary64_byte_boundary() {
    fn check<S: scioxide_core::RealScalar>() {
        let n = 512;
        let x = S::try_from_f64_lossy(0.03125).unwrap();
        let m = 33;
        let bytes = (2 * m * m + 2 * m) * core::mem::size_of::<f64>();
        let options = LawOptions::<S>::default()
            .with_max_bytes(bytes)
            .with_work_budget(scioxide_core::Budget::try_from(1 << 24).unwrap());
        let law = KolmogorovSmirnovTwoSided::with_options(n, options).unwrap();
        let mut scratch = law.workspace(x).unwrap();
        {
            law.evaluate_tail(x, true, &mut scratch).unwrap();
        };
        let short =
            KolmogorovSmirnovTwoSided::with_options(n, options.with_max_bytes(bytes - 1)).unwrap();
        assert_eq!(
            short.workspace(x).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            short
                .evaluate_tail(x, true, &mut scratch)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        // Density storage includes derivative lanes and has its own admission.
        assert_eq!(
            law.log_pdf(x).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn finite_ks_work_boundary_and_large_reserved_reuse() {
    let options =
        LawOptions::default().with_work_budget(scioxide_core::Budget::try_from(1 << 30).unwrap());
    for n in [31, 512, 513, 4096] {
        let law = KolmogorovSmirnovTwoSided::<f64>::with_options(n, options).unwrap();
        let x = 0.6 / (n as f64).sqrt();
        let mut scratch = law.workspace(x).unwrap();
        let report = law.evaluate_tail(x, true, &mut scratch).unwrap();
        let exact_options =
            options.with_work_budget(scioxide_core::Budget::try_from(report.evaluations).unwrap());
        let exact = KolmogorovSmirnovTwoSided::with_options(n, exact_options).unwrap();
        {
            let repeated = exact.evaluate_tail(x, true, &mut scratch).unwrap();
            assert_eq!(repeated.evaluations, report.evaluations);
            assert_eq!(repeated.log_probability, report.log_probability);
            let lower = exact.evaluate_tail(x, false, &mut scratch).unwrap();
            assert!(
                (lower.log_probability.exp() + report.log_probability.exp() - 1.0).abs() < 2e-14
            );
        };
        let short = KolmogorovSmirnovTwoSided::with_options(
            n,
            exact_options
                .with_work_budget(scioxide_core::Budget::try_from(report.evaluations - 1).unwrap()),
        )
        .unwrap();
        assert_eq!(
            short
                .evaluate_tail(x, true, &mut scratch)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        let lower_limit =
            KolmogorovSmirnovTwoSided::with_options(n, options.with_max_bytes(1)).unwrap();
        assert_eq!(
            lower_limit
                .evaluate_tail(x, true, &mut scratch)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            law.evaluate_tail(x, true, &mut KolmogorovWorkspace::new())
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        // A smaller subsequent matrix and a matrix-free tail reuse retained storage.
        {
            law.evaluate_tail(0.8 * x, false, &mut scratch).unwrap();
            law.evaluate_tail(0.6, true, &mut scratch).unwrap();
        };
    }
}
#[test]
fn reserved_and_fixed_storage() {
    let ks = KolmogorovSmirnovTwoSided::<f64>::with_options(
        20,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let requirements = ks.workspace_requirements(0.15).unwrap();
    assert_eq!(requirements.dimension(), 7);
    let mut scratch = KolmogorovWorkspace::new();
    assert_eq!(
        scratch.check(&requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    scratch.try_reserve(&requirements).unwrap();
    assert_eq!(
        scratch.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    let lower = ks.log_cdf(0.15).unwrap();
    let upper = ks.log_sf(0.15).unwrap();
    {
        let report = ks.evaluate_tail(0.15, false, &mut scratch).unwrap();
        assert_eq!(report.log_probability, lower);
        let report = ks.evaluate_tail(0.15, true, &mut scratch).unwrap();
        assert_eq!(report.log_probability, upper);
    };
    // A boundary-region query needs no matrices.
    assert_eq!(ks.workspace_requirements(0.6).unwrap().dimension(), 0);
    let laws: [Box<dyn ContinuousLaw<f64>>; 8] = [
        Box::new(
            FisherF::with_options(
                5.0,
                11.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralF::with_options(
                5.0,
                11.0,
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralChiSquared::with_options(
                5.0,
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralStudentT::with_options(
                5.0,
                2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            Rice::with_options(
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            KolmogorovSmirnovOneSided::with_options(
                5,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            Kolmogorov::with_options(
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            StudentizedRange::with_options(
                3.0,
                10.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
    ];
    for law in laws {
        {
            law.log_pdf(0.7).unwrap();
            law.log_sf(0.7).unwrap();
        };
    }
}
#[test]
fn sampling_failure_rolls_back_and_success_reuses_buffers() {
    let options = LawOptions::default()
        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        .with_sample_word_budget(scioxide_core::Budget::ONE);
    let law = NoncentralStudentT::with_options(5.0_f64, 2.0, options).unwrap();
    let mut rng = SplitMix64::from_seed(540);
    let before = rng.export_state().unwrap();
    let mut out = [77.0; 4];
    let batch = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let mut work = batch
        .workspace(scioxide_random::SamplingRequest::scalars(4))
        .unwrap();
    assert!(
        law.sample_n_into(&mut rng, batch, &mut out, &mut work)
            .is_err()
    );
    assert_eq!(out, [77.0; 4]);
    assert_eq!(rng.export_state().unwrap(), before);
    let law = Rice::with_options(
        3.0_f64,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    law.sample_n_into(
        &mut rng,
        SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
        &mut out,
        &mut work,
    )
    .unwrap();
}

#[test]
fn tail_report_counts_define_the_exact_work_boundary() {
    let law = NoncentralStudentT::with_options(
        5.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let report = law.evaluate_tail(0.7, false).unwrap();
    assert!(report.evaluations > 1);
    let options = LawOptions::default()
        .with_work_budget(scioxide_core::Budget::try_from(report.evaluations).unwrap());
    let exact = NoncentralStudentT::with_options(5.0, 2.0, options).unwrap();
    let repeated = exact.evaluate_tail(0.7, false).unwrap();
    assert_eq!(repeated.evaluations, report.evaluations);
    assert!((repeated.log_probability - report.log_probability).abs() <= f64::EPSILON);
    let short = NoncentralStudentT::with_options(
        5.0,
        2.0,
        (options)
            .with_work_budget(scioxide_core::Budget::try_from(report.evaluations - 1).unwrap()),
    )
    .unwrap();
    assert_eq!(
        short.evaluate_tail(0.7, false).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    {
        assert_eq!(
            exact.quantile(0.3).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
    };
}

#[test]
fn inverse_keeps_accuracy_estimates_local_to_each_tail() {
    let law = KolmogorovSmirnovTwoSided::<f64>::with_options(
        5,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for p in [0.625, 0.75, 0.875] {
        let value = law.quantile(p).unwrap();
        assert!((law.cdf(value).unwrap() - p).abs() <= 1e-12);
    }
}
