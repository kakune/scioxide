//! One transaction contract for every batch form: owned batches that replay
//! the scalar stream and leave the generator advanced on error; caller-buffer
//! batches with atomic and incremental publication, each with advancing and
//! rolled-back generators, chunked replay of the scalar stream, and
//! allocation-free reserved batches.
#![allow(clippy::unwrap_used, dead_code)]
use core::fmt::Debug;

use scioxide_core::Complex;
use scioxide_core::{SciOxideError, SciOxideErrorKind};
use scioxide_random::*;

pub const DRAWS: usize = 12;

/// SplitMix64 uniform words that fail at one request index. Snapshots restore
/// the request counter together with the engine.
#[derive(Clone)]
pub struct Failing {
    pub inner: SplitMix64,
    pub requests: u64,
    fail: Option<u64>,
}
impl Failing {
    pub fn new(fail: Option<u64>) -> Self {
        Self {
            inner: SplitMix64::from_seed(1340),
            requests: 0,
            fail,
        }
    }
    fn request(&mut self) -> Result<(), SciOxideError> {
        let index = self.requests;
        self.requests += 1;
        if Some(index) == self.fail {
            Err(SciOxideError::not_implemented("injected provider failure"))
        } else {
            Ok(())
        }
    }
}
impl BitGenerator for Failing {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.request()?;
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.request()?;
        self.inner.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.request()?;
        self.inner.next_uniform_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.request()?;
        self.inner.next_uniform_u64()
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        self.request()?;
        self.inner.fill_bytes(output)
    }
}
impl SnapshotGenerator for Failing {
    type Snapshot = Self;
    fn snapshot(&self) -> Result<Self, SciOxideError> {
        Ok(self.clone())
    }
    fn restore_snapshot(&mut self, snapshot: Self) {
        *self = snapshot;
    }
}

/// The batch forms of one law: scalar count laws have rows of width one.
pub trait Batch {
    type Item: Copy + PartialEq + Debug;
    fn width(&self) -> usize;
    fn row(&self, rng: &mut Failing, row: &mut [Self::Item]) -> Result<(), SciOxideError>;
    fn owned(&self, rng: &mut Failing, n: usize) -> Result<Vec<Self::Item>, SciOxideError>;
    fn workspace<P>(&self, policy: SamplingPolicy<P>) -> SamplingWorkspace<Self::Item>;
    fn fill<P: StatePolicy<Failing>>(
        &self,
        rng: &mut Failing,
        policy: SamplingPolicy<P>,
        output: &mut [Self::Item],
        workspace: &mut SamplingWorkspace<Self::Item>,
    ) -> Result<(), SciOxideError>;
}
pub struct Scalar<'a, D>(pub &'a D);
impl<D: CountDistribution> Batch for Scalar<'_, D>
where
    D::Count: PartialEq + Debug,
{
    type Item = D::Count;
    fn width(&self) -> usize {
        1
    }
    fn row(&self, rng: &mut Failing, row: &mut [D::Count]) -> Result<(), SciOxideError> {
        row[0] = self.0.sample(rng)?;
        Ok(())
    }
    fn owned(&self, rng: &mut Failing, n: usize) -> Result<Vec<D::Count>, SciOxideError> {
        self.0.sample_n(rng, n)
    }
    fn workspace<P>(&self, policy: SamplingPolicy<P>) -> SamplingWorkspace<D::Count> {
        policy.workspace(SamplingRequest::scalars(DRAWS)).unwrap()
    }
    fn fill<P: StatePolicy<Failing>>(
        &self,
        rng: &mut Failing,
        policy: SamplingPolicy<P>,
        output: &mut [D::Count],
        workspace: &mut SamplingWorkspace<D::Count>,
    ) -> Result<(), SciOxideError> {
        self.0.sample_n_into(rng, policy, output, workspace)
    }
}
// Every vector- and matrix-valued law exposes the same row forms.
macro_rules! vector_batches {
    ($($law:ty => $item:ty),+ $(,)?) => {$(
        impl Batch for $law {
            type Item = $item;
            fn width(&self) -> usize {
                self.sample_width()
            }
            fn row(&self, rng: &mut Failing, row: &mut [$item]) -> Result<(), SciOxideError> {
                row.copy_from_slice(&self.sample(rng)?);
                Ok(())
            }
            fn owned(&self, rng: &mut Failing, n: usize) -> Result<Vec<$item>, SciOxideError> {
                self.sample_n(rng, n)
            }
            fn workspace<P>(&self, policy: SamplingPolicy<P>) -> SamplingWorkspace<$item> {
                self.workspace(policy, DRAWS).unwrap()
            }
            fn fill<P: StatePolicy<Failing>>(
                &self,
                rng: &mut Failing,
                policy: SamplingPolicy<P>,
                output: &mut [$item],
                workspace: &mut SamplingWorkspace<$item>,
            ) -> Result<(), SciOxideError> {
                self.sample_n_into(rng, policy, output, workspace)
            }
        }
    )+};
}
vector_batches!(
    Multinomial<f64> => usize,
    DirichletMultinomial<f64> => usize,
    MultivariateHypergeometric<f64> => usize,
    RandomTable<f64> => usize,
    Dirichlet<f64> => f64,
    MultivariateNormal<f64> => f64,
    MultivariateStudentT<f64> => f64,
    NormalInverseGamma<f64> => f64,
    Wishart<f64> => f64,
    InverseWishart<f64> => f64,
    MatrixNormal<f64> => f64,
    MatrixT<f64> => f64,
    OrthogonalGroup<f64> => f64,
    SpecialOrthogonalGroup<f64> => f64,
    UnitaryGroup<f64> => Complex<f64>,
    RandomCorrelation<f64> => f64,
    UniformDirection<f64> => f64,
    VonMisesFisher<f64> => f64,
);

/// Row-major scalar reference and the request count after each draw.
pub fn reference<B: Batch>(batch: &B, sentinel: B::Item) -> (Vec<B::Item>, Vec<u64>) {
    let mut rng = Failing::new(None);
    let mut values = vec![sentinel; DRAWS * batch.width()];
    let ends = values
        .chunks_exact_mut(batch.width())
        .map(|row| {
            batch.row(&mut rng, row).unwrap();
            rng.requests
        })
        .collect();
    (values, ends)
}

/// Owned batches equal `n` single draws, draw nothing for `n = 0`, and leave
/// the generator advanced by every request served before an error.
fn owned<B: Batch>(name: &str, batch: &B, sentinel: B::Item) {
    let (expected, ends) = reference(batch, sentinel);
    let mut rng = Failing::new(None);
    assert_eq!(batch.owned(&mut rng, DRAWS).unwrap(), expected, "{name}");
    assert_eq!(rng.requests, ends[DRAWS - 1], "{name}");
    let mut rng = Failing::new(None);
    assert!(batch.owned(&mut rng, 0).unwrap().is_empty(), "{name}");
    assert_eq!(rng.requests, 0, "{name}");
    let start = ends[DRAWS / 2 - 1];
    let mut rng = Failing::new(Some(start));
    let error = batch.owned(&mut rng, DRAWS).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NotImplemented, "{name}");
    assert_eq!(rng.requests, start + 1, "{name}");
}

fn failures<B: Batch, P: StatePolicy<Failing> + Copy>(
    name: &str,
    batch: &B,
    sentinel: B::Item,
    policy: SamplingPolicy<P>,
    rollback: bool,
) {
    let width = batch.width();
    let (expected, ends) = reference(batch, sentinel);
    for failed in [0, DRAWS / 2, DRAWS - 1] {
        let start = if failed == 0 { 0 } else { ends[failed - 1] };
        assert!(
            ends[failed] > start,
            "{name}: draw {failed} consumes no word"
        );
        let mut rng = Failing::new(Some(start));
        let mut output = vec![sentinel; DRAWS * width];
        let mut workspace = SamplingWorkspace::new();
        let error = batch
            .fill(&mut rng, policy, &mut output, &mut workspace)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NotImplemented, "{name}");
        let published = match policy.output_policy() {
            OutputPolicy::Atomic => 0,
            OutputPolicy::Incremental => failed * width,
        };
        assert_eq!(output[..published], expected[..published], "{name}");
        assert!(
            output[published..].iter().all(|&value| value == sentinel),
            "{name}"
        );
        if rollback {
            assert_eq!((rng.requests, rng.inner.position()), (0, Some(0)), "{name}");
        } else {
            assert_eq!(rng.requests, start + 1, "{name}");
        }
    }
    // Successful batches, whole or chunked, reproduce the scalar stream.
    let mut rng = Failing::new(None);
    let mut output = vec![sentinel; DRAWS * width];
    let mut workspace = SamplingWorkspace::new();
    let (head, tail) = output.split_at_mut(5 * width);
    batch.fill(&mut rng, policy, head, &mut workspace).unwrap();
    batch.fill(&mut rng, policy, tail, &mut workspace).unwrap();
    assert_eq!(output, expected, "{name}");
    assert_eq!(rng.requests, ends[DRAWS - 1], "{name}");
}

fn allocations<B: Batch>(name: &str, batch: &B, sentinel: B::Item) {
    for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        let policy = SamplingPolicy::rollback(output_policy);
        let mut workspace = batch.workspace(policy);
        let mut rng = Failing::new(None);
        let mut output = vec![sentinel; DRAWS * batch.width()];
        {
            batch
                .fill(&mut rng, policy, &mut output, &mut workspace)
                .unwrap();
        };
        assert!(output.iter().all(|&value| value != sentinel), "{name}");
    }
}

/// Transactions and allocation-free reserved batches.
pub fn contract<B: Batch>(name: &str, batch: &B, sentinel: B::Item) {
    transactions(name, batch, sentinel);
    allocations(name, batch, sentinel);
}

/// Owned batches, then atomic and incremental publication, each with
/// advancing and rolled-back generators, and chunked replay of the scalar
/// stream.
pub fn transactions<B: Batch>(name: &str, batch: &B, sentinel: B::Item) {
    owned(name, batch, sentinel);
    for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        failures(
            name,
            batch,
            sentinel,
            SamplingPolicy::advance_on_error(output_policy),
            false,
        );
        failures(
            name,
            batch,
            sentinel,
            SamplingPolicy::rollback(output_policy),
            true,
        );
    }
}
