#![allow(clippy::unwrap_used)]
use scioxide_core::NumericScalar;
use scioxide_core::RealScalar;
use scioxide_linalg::Matrix;
use scioxide_random::*;
fn gaussian<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let law = MultivariateNormal::new(
        &[c(2.), c(-1.)],
        &Matrix::from_vec(2, 2, vec![c(4.), c(2.), c(2.), c(1.)]).unwrap(),
    )
    .unwrap();
    assert_eq!(law.rank(), 1);
    let transport = QmcMultivariateNormal::new(&law);
    let plan = QmcPlan::sobol(1)
        .unwrap()
        .with_randomization(
            QmcRandomization::LinearMatrixShift,
            &mut SplitMix64::from_seed(42),
        )
        .unwrap();
    let mut seq = plan.sequence::<S>().unwrap();
    let mut owned_seq = seq.clone();
    let first = transport.sample(&mut owned_seq).unwrap();
    let mut out = vec![c(0.); 4096];
    let mut work = transport.workspace().unwrap();
    let requirements = SamplingPolicy::rollback(OutputPolicy::Atomic)
        .workspace_requirements(SamplingRequest::rows(out.len() / 2, 2))
        .unwrap();
    assert_eq!(
        work.check(&requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    work.try_reserve(&requirements).unwrap();
    assert_eq!(
        work.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    transport
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut work,
        )
        .unwrap();
    assert_eq!(first, out[..2]);
    let mean = out.chunks_exact(2).map(|x| x[0].into_f64()).sum::<f64>() / 2048.;
    let variance = out
        .chunks_exact(2)
        .map(|x| (x[0].into_f64() - 2.).powi(2))
        .sum::<f64>()
        / 2048.;
    assert!((mean - 2.).abs() < 0.01);
    assert!((variance - 4.).abs() < 0.04);
    for row in out.chunks_exact(2) {
        assert!(((row[0].into_f64() - 2.) / 2. - (row[1].into_f64() + 1.)).abs() < 3e-6);
    }
    let bare = QmcPlan::sobol(1).unwrap();
    let mut seq = bare.sequence::<S>().unwrap();
    let saved = seq.position();
    let mut out = [c(123.); 2];
    assert!(
        transport
            .sample_n_into(
                &mut seq,
                SamplingPolicy::rollback(OutputPolicy::Atomic),
                &mut out,
                &mut work
            )
            .is_err()
    );
    assert_eq!(out, [c(123.); 2]);
    assert_eq!(seq.position(), saved);
    let zero =
        MultivariateNormal::new(&[c(3.)], &Matrix::from_vec(1, 1, vec![c(0.)]).unwrap()).unwrap();
    let deterministic = QmcMultivariateNormal::new(&zero);
    let mut work = deterministic.workspace().unwrap();
    let mut out = [c(0.); 4];
    deterministic
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut work,
        )
        .unwrap();
    assert_eq!(out, [c(3.); 4]);
    assert_eq!(seq.position(), saved);
    assert_eq!(deterministic.sample(&mut seq).unwrap(), [c(3.)]);
    assert_eq!(seq.position(), saved);
}
fn multinomial<S: RealScalar>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let law = Multinomial::with_options(
        4,
        &[c(0.25), c(0.), c(0.75)],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let transport = QmcMultinomial::new(&law).unwrap();
    let plan = QmcPlan::sobol(4).unwrap();
    let mut seq = plan.sequence::<S>().unwrap();
    let mut owned_seq = seq.clone();
    let first = transport.sample(&mut owned_seq).unwrap();
    let mut out = vec![0; 3 * 1024];
    let mut work = transport.workspace().unwrap();
    transport
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut work,
        )
        .unwrap();
    assert_eq!(first, out[..3]);
    for row in out.chunks_exact(3) {
        assert_eq!(row.iter().sum::<usize>(), 4);
        assert_eq!(row[1], 0);
    }
    assert_eq!(out.chunks_exact(3).map(|r| r[0]).sum::<usize>(), 1024);
    let saved = seq.position();
    let mut invalid_out = [99; 3];
    // A workspace prepared for three trials has the wrong coordinate dimension.
    let three = Multinomial::with_options(
        3,
        &[c(0.25), c(0.), c(0.75)],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut wrong = QmcMultinomial::new(&three).unwrap().workspace().unwrap();
    assert!(
        transport
            .sample_n_into(
                &mut seq,
                SamplingPolicy::rollback(OutputPolicy::Atomic),
                &mut invalid_out,
                &mut wrong
            )
            .is_err()
    );
    assert_eq!(invalid_out, [99; 3]);
    assert_eq!(seq.position(), saved);
    transport
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut [],
            &mut work,
        )
        .unwrap();
    assert_eq!(seq.position(), saved);
    let zero = Multinomial::with_options(
        0,
        &[c(0.), c(1.)],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let saved = seq.position();
    let mut out = [99; 4];
    let zero_transport = QmcMultinomial::new(&zero).unwrap();
    let mut work = zero_transport.workspace().unwrap();
    zero_transport
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut work,
        )
        .unwrap();
    assert_eq!(out, [0; 4]);
    assert_eq!(seq.position(), saved);
    assert_eq!(
        QmcMultinomial::new(&zero)
            .unwrap()
            .sample(&mut seq)
            .unwrap(),
        [0; 2]
    );
    assert_eq!(seq.position(), saved);
    assert!(
        Multinomial::with_options(
            1,
            &[c(0.2), c(0.2)],
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        Multinomial::<S>::with_options(
            1,
            &[],
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
}
#[test]
fn gaussian_both_widths() {
    gaussian::<f32>();
    gaussian::<f64>();
}
#[test]
fn multinomial_both_widths() {
    multinomial::<f32>();
    multinomial::<f64>();
}

#[test]
fn count_transport_exhaustion_obeys_both_transaction_policies() {
    let law = Multinomial::new(1, &[0.5_f64, 0.5]).unwrap();
    let adapter = QmcMultinomial::new(&law).unwrap();
    let plan = QmcPlan::sobol(1).unwrap();
    for publication in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        let mut sequence = plan.sequence::<f64>().unwrap();
        sequence.skip_to(u64::MAX).unwrap();
        let saved = sequence.position();
        let first = adapter.sample(&mut sequence.clone()).unwrap();
        let mut workspace = adapter.workspace().unwrap();
        let mut output = [99; 4];
        assert!(
            adapter
                .sample_n_into(
                    &mut sequence,
                    SamplingPolicy::rollback(publication),
                    &mut output,
                    &mut workspace
                )
                .is_err()
        );
        assert_eq!(sequence.position(), saved);
        if publication == OutputPolicy::Atomic {
            assert_eq!(output, [99; 4]);
        } else {
            assert_eq!(output[..2], first);
            assert_eq!(output[2..], [99; 2]);
        }
        output.fill(99);
        assert!(
            adapter
                .sample_n_into(
                    &mut sequence,
                    SamplingPolicy::advance_on_error(publication),
                    &mut output,
                    &mut workspace
                )
                .is_err()
        );
        assert_eq!(sequence.position().next_index(), None);
        if publication == OutputPolicy::Atomic {
            assert_eq!(output, [99; 4]);
        } else {
            assert_eq!(output[..2], first);
            assert_eq!(output[2..], [99; 2]);
        }
    }
}

#[test]
fn transport_owned_batches_equal_caller_buffer_batches() {
    let law = MultivariateNormal::new(
        &[1.0_f64, -1.0, 0.5],
        &Matrix::from_vec(3, 3, vec![2.0, 0.5, 0.0, 0.5, 1.0, 0.25, 0.0, 0.25, 3.0]).unwrap(),
    )
    .unwrap();
    let gaussian = QmcMultivariateNormal::new(&law);
    let plan = QmcPlan::sobol(gaussian.input_dimension())
        .unwrap()
        .with_randomization(
            QmcRandomization::LinearMatrixShift,
            &mut SplitMix64::from_seed(42),
        )
        .unwrap();
    let mut owned = plan.sequence::<f64>().unwrap();
    let samples = gaussian.sample_n(&mut owned, 33).unwrap();
    let mut sequence = plan.sequence::<f64>().unwrap();
    let mut out = vec![0.0; 33 * 3];
    let mut work = gaussian.workspace().unwrap();
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    gaussian
        .sample_n_into(&mut sequence, policy, &mut out, &mut work)
        .unwrap();
    assert_eq!(samples, out);
    assert_eq!(owned.position(), sequence.position());
    assert!(gaussian.sample_n(&mut owned, 0).unwrap().is_empty());
    assert_eq!(owned.position(), sequence.position());

    let law = Multinomial::new(3, &[0.2_f64, 0.3, 0.5]).unwrap();
    let counts = QmcMultinomial::new(&law).unwrap();
    let plan = QmcPlan::halton(3).unwrap();
    let mut owned = plan.sequence::<f64>().unwrap();
    let samples = counts.sample_n(&mut owned, 17).unwrap();
    let mut sequence = plan.sequence::<f64>().unwrap();
    let mut out = vec![0; 17 * 3];
    let mut work = counts.workspace().unwrap();
    let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
    counts
        .sample_n_into(&mut sequence, policy, &mut out, &mut work)
        .unwrap();
    assert_eq!(samples, out);
    assert_eq!(owned.position(), sequence.position());
    // Past the last point the owned form fails and leaves the sequence advanced.
    owned.skip_to(u64::MAX).unwrap();
    assert!(counts.sample_n(&mut owned, 2).is_err());
    assert_eq!(owned.position().next_index(), None);
}
