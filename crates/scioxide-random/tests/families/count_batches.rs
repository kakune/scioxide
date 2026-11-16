//! Count-law batches: publication, rollback, allocation and budget contracts.
#![allow(clippy::unwrap_used)]
use crate::vector_laws::{DRAWS, Failing, Scalar, contract, transactions};
use core::fmt::Debug;
use scioxide_core::{Budget, SciOxideError, SciOxideErrorKind};
use scioxide_random::*;

fn counts<D: CountDistribution>(name: &str, law: &D, sentinel: D::Count)
where
    D::Count: PartialEq + Debug,
{
    contract(name, &Scalar(law), sentinel);
}

#[test]
fn unsigned_count_law_batches() {
    let o = LawOptions::<f64>::default;
    counts(
        "Geometric",
        &Geometric::with_options(0.25, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "NegativeBinomial",
        &NegativeBinomial::with_options(2.5, 0.25, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "BetaBinomial Pólya",
        &BetaBinomial::with_options(20, 0.3, 0.3, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "BetaBinomial mixture",
        &BetaBinomial::with_options(200, 2.0, 3.0, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "BetaNegativeBinomial",
        &BetaNegativeBinomial::with_options(2.0, 3.0, 2.0, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "LogSeries",
        &LogSeries::with_options(0.5, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "YuleSimon",
        &YuleSimon::with_options(4.0, o()).unwrap(),
        usize::MAX,
    );
    counts("Zipf", &Zipf::with_options(2.0, o()).unwrap(), usize::MAX);
    counts(
        "Zipfian",
        &Zipfian::with_options(0.5, 100, o()).unwrap(),
        usize::MAX,
    );
}

#[test]
fn finite_count_law_batches() {
    let o = LawOptions::<f64>::default;
    counts(
        "Boltzmann",
        &Boltzmann::with_options(0.5, 16, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "PoissonBinomial",
        &PoissonBinomial::with_options(&[0.25, 0.5, 0.75], o()).unwrap(),
        usize::MAX,
    );
    counts(
        "Hypergeometric",
        &Hypergeometric::with_options(50, 20, 12, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "FisherNoncentralHypergeometric",
        &FisherNoncentralHypergeometric::with_options(50, 20, 12, 2.0, o()).unwrap(),
        usize::MAX,
    );
    counts(
        "WalleniusNoncentralHypergeometric",
        &WalleniusNoncentralHypergeometric::with_options(50, 20, 12, 2.0, o()).unwrap(),
        usize::MAX,
    );
}

#[test]
fn prepared_count_sampler_batches() {
    for (case, n, p) in [
        ("Binomial direct", 16, 0.5),
        ("Binomial BINV", 1000, 0.001),
        ("Binomial BTPE", 1000, 0.4),
    ] {
        let law = Binomial::new(n, p).unwrap();
        counts(case, &law.prepare_sampler().unwrap(), usize::MAX);
        counts(case, &law, usize::MAX);
    }
    for (case, rate) in [("Poisson inversion", 5.0), ("Poisson PTRS", 30.0)] {
        let law = Poisson::new(rate).unwrap();
        counts(case, &law.prepare_sampler().unwrap(), usize::MAX);
        counts(case, &law, usize::MAX);
    }
    let skellam = Skellam::new(30.0, 20.0).unwrap();
    counts("Skellam", &skellam.prepare_sampler().unwrap(), i64::MIN);
    counts("Skellam", &skellam, i64::MIN);
}

#[test]
fn signed_count_law_batches() {
    let o = LawOptions::<f64>::default;
    counts(
        "DiscreteLaplace",
        &DiscreteLaplace::with_options(0.5, o()).unwrap(),
        i64::MIN,
    );
    counts(
        "DiscreteUniform",
        &DiscreteUniform::with_options(-10, 11, o()).unwrap(),
        i64::MIN,
    );
}

#[test]
fn every_vector_and_matrix_law_batch() {
    let o = || {
        LawOptions::<f64>::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
    };
    let real = -7777.0;
    let identity = scioxide_linalg::Matrix::identity(2).unwrap();
    let spread = scioxide_linalg::Matrix::from_vec(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap();
    let mean =
        scioxide_linalg::Matrix::from_vec(2, 3, vec![0.0, 1.0, -1.0, 2.0, 0.5, 0.0]).unwrap();
    let columns = scioxide_linalg::Matrix::identity(3).unwrap();
    let gaussian = MultivariateNormalOptions::default;
    contract(
        "Multinomial",
        &Multinomial::with_options(20, &[0.2, 0.3, 0.5], o()).unwrap(),
        usize::MAX,
    );
    contract(
        "DirichletMultinomial",
        &DirichletMultinomial::with_options(20, &[0.5, 2.0, 1.0], o()).unwrap(),
        usize::MAX,
    );
    contract(
        "MultivariateHypergeometric",
        &MultivariateHypergeometric::with_options(&[10, 20, 30], 15, o()).unwrap(),
        usize::MAX,
    );
    contract(
        "RandomTable",
        &RandomTable::with_options(&[5, 7, 8], &[6, 4, 10], o()).unwrap(),
        usize::MAX,
    );
    contract(
        "Dirichlet",
        &Dirichlet::with_options(&[0.3, 2.0, 1.5], o()).unwrap(),
        real,
    );
    contract(
        "MultivariateNormal",
        &MultivariateNormal::with_options(&[1.0, -2.0], &spread, gaussian(), o()).unwrap(),
        real,
    );
    contract(
        "MultivariateStudentT",
        &MultivariateStudentT::with_options(&[1.0, -2.0], &spread, 3.5, gaussian(), o()).unwrap(),
        real,
    );
    contract(
        "NormalInverseGamma",
        &NormalInverseGamma::with_options(0.5, 2.0, 3.0, 1.5, o()).unwrap(),
        real,
    );
    contract(
        "Wishart",
        &Wishart::with_options(4.5, &spread, o()).unwrap(),
        real,
    );
    contract(
        "InverseWishart",
        &InverseWishart::with_options(5.5, &spread, o()).unwrap(),
        real,
    );
    contract(
        "MatrixNormal",
        &MatrixNormal::with_options(&mean, &identity, &columns, o()).unwrap(),
        real,
    );
    contract(
        "MatrixT",
        &MatrixT::with_options(&mean, &identity, &columns, 4.0, o()).unwrap(),
        real,
    );
    transactions(
        "OrthogonalGroup",
        &OrthogonalGroup::with_options(3, o()).unwrap(),
        real,
    );
    transactions(
        "SpecialOrthogonalGroup",
        &SpecialOrthogonalGroup::with_options(3, o()).unwrap(),
        real,
    );
    transactions(
        "UnitaryGroup",
        &UnitaryGroup::with_options(2, o()).unwrap(),
        scioxide_core::Complex { re: real, im: real },
    );
    transactions(
        "RandomCorrelation",
        &RandomCorrelation::with_options(&[0.5, 1.0, 1.5], o()).unwrap(),
        real,
    );
    contract(
        "UniformDirection",
        &UniformDirection::with_options(3, o()).unwrap(),
        real,
    );
    contract(
        "VonMisesFisher",
        &VonMisesFisher::with_options(&[0.0, 0.6, 0.8], 4.0, o()).unwrap(),
        real,
    );
}

#[test]
fn configured_byte_limits_precede_drawing() {
    let law =
        Geometric::with_options(0.25, LawOptions::<f64>::default().with_max_bytes(0)).unwrap();
    let mut rng = Failing::new(None);
    let mut output = [0; DRAWS];
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let error = law
        .sample_n_into(&mut rng, policy, &mut output, &mut SamplingWorkspace::new())
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    let error = law.sample_n(&mut rng, DRAWS).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(rng.requests, 0);
    // Incremental scalar batches stage nothing, so they fit a zero limit.
    let policy = SamplingPolicy::rollback(OutputPolicy::Incremental);
    law.sample_n_into(&mut rng, policy, &mut output, &mut SamplingWorkspace::new())
        .unwrap();
    assert_eq!(rng.requests, DRAWS as u64);
}

#[test]
fn one_draw_allowance_bounds_every_count_sampler() {
    let words =
        |n| LawOptions::<f64>::default().with_sample_word_budget(Budget::try_from(n).unwrap());
    // The Pólya path draws one Bernoulli word or more per trial.
    let law = BetaBinomial::with_options(100, 0.125, 0.25, words(99)).unwrap();
    let mut rng = Failing::new(None);
    let error = law.sample(&mut rng).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(rng.requests, 99);
    // Query work no longer limits uniform-word requests.
    let options = LawOptions::<f64>::default().with_work_budget(Budget::ONE);
    let law = BetaBinomial::with_options(100, 0.125, 0.25, options).unwrap();
    assert!(law.sample(&mut Failing::new(None)).unwrap() <= 100);
    // Prepared samplers keep their law's allowance: 16 direct trials need 16 words.
    let sampler = Binomial::with_options(16, 0.5, words(15))
        .unwrap()
        .prepare_sampler()
        .unwrap();
    let error = sampler.sample(&mut Failing::new(None)).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    let sampler = Binomial::with_options(16, 0.5, words(16))
        .unwrap()
        .prepare_sampler()
        .unwrap();
    assert!(sampler.sample(&mut Failing::new(None)).unwrap() <= 16);
    let sampler = Skellam::with_options(30.0, 20.0, words(1))
        .unwrap()
        .prepare_sampler()
        .unwrap();
    let mut rng = Failing::new(None);
    assert!(sampler.sample(&mut rng).is_err());
    assert_eq!(rng.requests, 1);
    // Unsupported trial counts are refused before any word is drawn.
    if let Ok(n) = usize::try_from(1_u64 << 52) {
        let law = BetaBinomial::new(n, 2.0, 3.0).unwrap();
        let mut rng = Failing::new(None);
        let error = law.sample(&mut rng).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NotImplemented);
        assert_eq!(rng.requests, 0);
    }
}

#[test]
fn rejection_guard_stops_zipf_after_1024_proposals() {
    // Alternating uniforms 0.4 and 0.9 propose x = 2 and always reject it.
    struct Cycle {
        calls: u64,
    }
    impl BitGenerator for Cycle {
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            Ok((self.next_u64()? >> 32) as u32)
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            let p = if self.calls.is_multiple_of(2) {
                0.4
            } else {
                0.9
            };
            self.calls += 1;
            Ok(((p * 4_503_599_627_370_496.0_f64) as u64) << 12)
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
            output.fill(0);
            Ok(())
        }
    }
    let law = Zipf::new(2.0).unwrap();
    let mut rng = Cycle { calls: 0 };
    let error = law.sample(&mut rng).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(rng.calls, 2 * 1024);
}

#[test]
fn zipfian_direct_limits_and_provider_failures() {
    // Normalization is prepared once within the constructor work allowance.
    assert!(
        Zipfian::with_options(
            1.1,
            1_000_000,
            LawOptions::<f64>::default().with_work_budget(Budget::ONE)
        )
        .is_err()
    );
    // Direct draws need no tail evaluations after preparation, even when the
    // query work allowance cannot finish inversion over a large support.
    let law = Zipfian::with_options(
        1.1,
        1_000_000,
        LawOptions::<f64>::default().with_work_budget(Budget::try_from(64).unwrap()),
    )
    .unwrap();
    assert_eq!(
        law.quantile(0.5).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!((1..=1_000_000).contains(&law.sample(&mut Failing::new(None)).unwrap()));
    let mut failing = Failing::new(Some(0));
    assert_eq!(
        law.sample(&mut failing).unwrap_err().kind(),
        SciOxideErrorKind::NotImplemented
    );
    assert_eq!(failing.requests, 1);
    let singleton = Zipfian::new(200.0, 1).unwrap();
    let mut failing = Failing::new(Some(0));
    assert_eq!(singleton.sample(&mut failing).unwrap(), 1);
    assert_eq!(failing.requests, 0);
    for exponent in [200.0, f64::MAX] {
        let law = Zipfian::new(exponent, 1000).unwrap();
        let mut rng = Failing::new(None);
        for _ in 0..64 {
            assert_eq!(law.sample(&mut rng).unwrap(), 1);
        }
    }
}

#[test]
fn zipfian_rejection_respects_candidate_and_word_limits() {
    struct Rejected {
        calls: usize,
        word: u64,
    }
    impl BitGenerator for Rejected {
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.calls += 1;
            Ok(self.word)
        }
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            Ok((self.next_u64()? >> 32) as u32)
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
            output.fill(0);
            Ok(())
        }
    }
    // At exponent one H=ln. A proposal just above 1.5 rounds to 2
    // and lies in the rejected portion of that bin's integral envelope.
    let high = 1000.5_f64.ln();
    let uniform = (high - 1.5001_f64.ln()) / (high - 1.5_f64.ln() + 1.0);
    let word = ((uniform * 4_503_599_627_370_496.0) as u64) << 12;
    for (budget, expected) in [(1, 1), (4096, 1024)] {
        let options =
            LawOptions::<f64>::default().with_sample_word_budget(Budget::try_from(budget).unwrap());
        let law = Zipfian::with_options(1.0, 1000, options).unwrap();
        let mut rng = Rejected { calls: 0, word };
        assert_eq!(
            law.sample(&mut rng).unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(rng.calls, expected);
    }
}
