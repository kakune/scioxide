//! Resource, provider failure, transaction and execution tests.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_random::*;
#[path = "../support/universal_density.rs"]
mod providers;
use providers::{Gaussian, Unit};
#[derive(Clone)]
struct Script {
    values: Vec<f64>,
    calls: usize,
}
impl Script {
    fn constant(x: f64) -> Self {
        Self {
            values: vec![x],
            calls: 0,
        }
    }
}
impl BitGenerator for Script {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(0)
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Ok(0)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        out.fill(0);
        Ok(())
    }
}
impl<S: RealScalar> Rng<S> for Script {
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        let x = self.values[self.calls % self.values.len()];
        self.calls += 1;
        Ok(if x.is_nan() {
            S::NAN
        } else if x == f64::INFINITY {
            S::INFINITY
        } else {
            S::try_from_f64_lossy(x)?
        })
    }
}
impl SnapshotGenerator for Script {
    type Snapshot = usize;
    fn snapshot(&self) -> Result<usize, SciOxideError> {
        Ok(self.calls)
    }
    fn restore_snapshot(&mut self, n: usize) {
        self.calls = n;
    }
}
fn bounded_rng<S: RealScalar>() {
    let p = Gaussian { offset: 0.0 };
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let options = UniversalSamplingOptions::default().with_rng_calls(Budget::try_from(3).unwrap());
    let ratio = RatioUniformsSampler::<S, _>::with_options(p, options).unwrap();
    let ars = AdaptiveRejectionSampler::with_options(p, &[s(-1.0), s(1.0)], options).unwrap();
    let tdr = TransformedDensitySampler::with_options(p, &[s(-1.0), s(1.0)], options).unwrap();
    let inv = InversionSampler::with_options(Unit, options).unwrap();
    for value in [0.0, -0.1, 1.0, f64::NAN, f64::INFINITY] {
        for sampler in [0, 1, 2, 3] {
            let mut rng = Script::constant(value);
            let result = match sampler {
                0 => ratio.sample(&mut rng),
                1 => ars.sample(&mut rng),
                2 => tdr.sample(&mut rng),
                _ => inv.sample(&mut rng),
            };
            assert!(result.is_err());
            assert_eq!(rng.calls, if value == 0.0 { 3 } else { 1 });
        }
    }
    let mut rng = Script::constant(0.0);
    let err = ratio.sample(&mut rng).unwrap_err();
    if let scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } = err.view() {
        assert_eq!(diagnostics.counts().own().proposals(), 1);
        assert_eq!(
            diagnostics
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Sample),
            3
        );
    } else {
        assert_eq!(
            err.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
    }
    let mut rng = Script {
        values: vec![0.0, 0.5],
        calls: 0,
    };
    let ratio = RatioUniformsSampler::<S, _>::with_options(
        p,
        options.with_rng_calls(Budget::try_from(4).unwrap()),
    )
    .unwrap();
    let result = ratio.sample_report(&mut rng).unwrap();
    assert_eq!(result.diagnostics.proposals, 1);
    assert_eq!(result.diagnostics.rng_calls, 4);
    assert_eq!(result.diagnostics.evaluations, 1);
}
#[test]
fn checked_rng_both_widths() {
    bounded_rng::<f32>();
    bounded_rng::<f64>();
}

#[test]
fn universal_open_uniform_rejection_has_the_shared_attempt_limit() {
    let sampler = InversionSampler::<f64, _>::new(Unit).unwrap();
    let mut rng = Script::constant(0.0);
    assert_eq!(
        sampler.sample(&mut rng).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(rng.calls, 1024);
}
#[test]
fn transactions_and_empty_batches() {
    let inv = InversionSampler::new(Unit).unwrap();
    for publication in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        for rollback in [false, true] {
            let mut rng = Script {
                values: vec![0.25, 0.75, f64::NAN],
                calls: 0,
            };
            let mut out = [-1.0; 4];
            let mut w = SamplingWorkspace::new();
            let result = if rollback {
                inv.sample_n_into(
                    &mut rng,
                    SamplingPolicy::rollback(publication),
                    &mut out,
                    &mut w,
                )
            } else {
                inv.sample_n_into(
                    &mut rng,
                    SamplingPolicy::advance_on_error(publication),
                    &mut out,
                    &mut w,
                )
            };
            assert!(result.is_err());
            assert_eq!(rng.calls, if rollback { 0 } else { 3 });
            if publication == OutputPolicy::Atomic {
                assert_eq!(out, [-1.0; 4]);
            } else {
                assert!((out[0] - 0.25_f64).abs() < 1e-6);
                assert!((out[1] - 0.75_f64).abs() < 1e-6);
                assert_eq!(&out[2..], &[-1.0; 2]);
            }
        }
    }
    let mut rng = Script::constant(f64::NAN);
    inv.sample_n_into(
        &mut rng,
        SamplingPolicy::rollback(OutputPolicy::Atomic),
        &mut [],
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert_eq!(rng.calls, 0);
}
#[test]
fn adaptive_workspace_provenance_and_publication() {
    let p = Gaussian { offset: 0.0 };
    let a = AdaptiveRejectionSampler::new(p, &[-1.0, 1.0]).unwrap();
    let b = AdaptiveRejectionSampler::new(p, &[-1.0, 1.0]).unwrap();
    let mut scratch = a.workspace().unwrap();
    let mut rng = Script::constant(0.5);
    assert!(b.sample_adaptive(&mut rng, &mut scratch).is_err());
    assert_eq!(rng.calls, 0);
    let mut rng = Script {
        values: vec![0.4, 0.9, f64::NAN],
        calls: 0,
    };
    let mut out = [99.0; 2];
    let mut staging = SamplingWorkspace::new();
    assert!(
        a.sample_adaptive_into(
            &mut rng,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut scratch,
            &mut staging
        )
        .is_err()
    );
    assert_eq!(out, [99.0; 2]);
    assert_eq!(rng.calls, 0);
    assert!(scratch.nodes() >= 2);
}

struct Reverse;
impl scioxide_core::Executor for Reverse {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: scioxide_core::ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        for index in (0..plan.worker_count()).rev() {
            scioxide_core::Executor::execute(
                &scioxide_core::SequentialExecutor,
                context,
                plan,
                &|p, (), s| {
                    if p.worker_index() == index {
                        operation(p, (), s)
                    } else {
                        Ok(())
                    }
                },
            )?;
        }
        scioxide_core::Executor::execute(
            &scioxide_core::SequentialExecutor,
            context,
            plan,
            &|_, (), _| Ok(()),
        )
    }
}
fn indexed_replay<S: RealScalar>() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy};
    let workers = core::num::NonZeroUsize::new(4).unwrap();
    let root = Pcg64::from_seed(543);
    let before = root.snapshot().unwrap();
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let o = UniversalSamplingOptions::default();
    assert_eq!(o.execution_policy(), ExecutionPolicy::Sequential);
    let parallel = o.with_execution_policy(ExecutionPolicy::Parallel(workers));
    let p = Gaussian { offset: 0.0 };
    let knots = [s(-1.0), s(1.0)];
    let mut w = SamplingWorkspace::new();
    let mut x = [S::ZERO; 48];
    let mut y = x;
    for sampler in 0..4 {
        for (out, context, options) in [
            (&mut x[..], ExecutionContext::sequential(), o),
            (&mut y[..], ExecutionContext::new(&Reverse), parallel),
        ] {
            match sampler {
                0 => AdaptiveRejectionSampler::with_options(p, &knots, options)
                    .unwrap()
                    .sample_indexed_into(&root, out, &mut w, context),
                1 => TransformedDensitySampler::with_options(p, &knots, options)
                    .unwrap()
                    .sample_indexed_into(&root, out, &mut w, context),
                2 => RatioUniformsSampler::<S, _>::with_options(p, options)
                    .unwrap()
                    .sample_indexed_into(&root, out, &mut w, context),
                _ => InversionSampler::with_options(Unit, options)
                    .unwrap()
                    .sample_indexed_into(&root, out, &mut w, context),
            }
            .unwrap();
        }
        assert_eq!(x, y);
    }
    let mut original = before;
    let mut root_copy = root;
    assert_eq!(original.next_u64().unwrap(), root_copy.next_u64().unwrap());
}
#[test]
fn executor_replays_both_widths() {
    indexed_replay::<f32>();
    indexed_replay::<f64>();
}

struct BadBounds;
impl<S: RealScalar> Support<S> for BadBounds {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        Gaussian { offset: 0.0 }.support()
    }
}
impl<S: RealScalar> LogDensity<S> for BadBounds {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        Gaussian { offset: 0.0 }.log_pdf(x)
    }
}
impl<S: RealScalar> UserDensity<S> for BadBounds {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> BoundedDensity<S> for BadBounds {
    fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<S>, SciOxideError> {
        Ok(RatioUniformsBounds {
            shift: S::ZERO,
            u_max: S::try_from_f64_lossy(0.1)?,
            v_min: -S::ONE,
            v_max: S::ONE,
        })
    }
}
struct Convex;
impl<S: RealScalar> Support<S> for Convex {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(-S::ONE), Some(S::ONE))
    }
}
impl<S: RealScalar> LogDensity<S> for Convex {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(x * x)
    }
}
impl<S: RealScalar> UserDensity<S> for Convex {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> ConcaveDensity<S> for Convex {
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError> {
        Ok((S::TWO) * x)
    }
    // Deliberately false guarantee: preparation must still detect the contradiction.
    fn validate_concavity(&self, _: DensityTransform) -> Result<(), SciOxideError> {
        Ok(())
    }
}
#[test]
fn failed_assumptions_and_exact_evaluation_limits() {
    let o = UniversalSamplingOptions::default();
    let ratio = RatioUniformsSampler::<f64, _>::with_options(BadBounds, o).unwrap();
    assert!(ratio.sample(&mut Script::constant(0.5)).is_err());
    assert!(AdaptiveRejectionSampler::with_options(Convex, &[-0.5, 0.5], o).is_err());
    let p = Gaussian { offset: 0.0 };
    let a = AdaptiveRejectionSampler::with_options(
        p,
        &[-1.0, 1.0],
        o.with_sample_evaluations(Budget::ONE),
    )
    .unwrap();
    let mut aw = a.workspace().unwrap();
    let mut rng = Script {
        values: vec![0.4, 0.99],
        calls: 0,
    };
    let before = aw.nodes();
    assert!(a.sample_adaptive(&mut rng, &mut aw).is_err());
    assert_eq!(aw.nodes(), before);
    assert_eq!(rng.calls, 2);
    // Every proposal rejects: the shared guard stops after 1,024 proposals.
    let r = RatioUniformsSampler::<f64, _>::with_options(p, o).unwrap();
    let mut rng = Script {
        values: vec![0.9, 0.99],
        calls: 0,
    };
    let err = r.sample(&mut rng).unwrap_err();
    assert_eq!(rng.calls, 2 * 1_024);
    let counts = match err.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } => {
            diagnostics.counts().own()
        }
        _ => scioxide_core::EvaluationCounts::new(),
    };
    assert_eq!((counts.iterations(), counts.proposals()), (1_024, 1_024));
    assert_eq!(counts.rejected(), 1_024);
    assert_eq!(
        counts.provider(scioxide_core::EvaluationRole::Function),
        1_024
    );
    assert_eq!(
        counts.provider(scioxide_core::EvaluationRole::Sample),
        2 * 1_024
    );
    let inv = InversionSampler::with_options(Unit, o.with_sample_evaluations(Budget::ONE)).unwrap();
    assert!(inv.transport(0.5_f64).is_err());
}
#[test]
fn storage_preflight_and_owned_output() {
    let o = UniversalSamplingOptions::default().with_max_bytes(256);
    let r = RatioUniformsSampler::<f64, _>::with_options(Gaussian { offset: 0.0 }, o).unwrap();
    let mut rng = Script::constant(0.5);
    let mut out = [99.0; 128];
    let mut staging = SamplingWorkspace::new();
    assert_eq!(
        r.sample_n_into(
            &mut rng,
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            &mut out,
            &mut staging
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(rng.calls, 0);
    assert_eq!(out, [99.0; 128]);
    assert_eq!(
        r.sample_n(&mut rng, 128).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(rng.calls, 0);
    let owned = r.sample_n(&mut rng, 4).unwrap();
    assert_eq!(owned, vec![0.0; 4]);
    let root = Pcg64::from_seed(543);
    assert_eq!(
        r.sample_indexed_into(
            &root,
            &mut out,
            &mut staging,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(out, [99.0; 128]);
}
