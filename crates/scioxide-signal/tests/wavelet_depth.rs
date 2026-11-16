//! Independent non-periodic 1-D and 2-D coefficient and useful-depth contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, SciOxideErrorView};
use scioxide_signal::{
    SignalBoundary, Wavelet2dLayout, Wavelet2dPlan, WaveletFamily, WaveletOptions, WaveletPhase,
    WaveletPlan, wavelet_max_level,
};

fn reference<S: RealScalar>() {
    for line in include_str!("fixtures/nonperiodic_wavelets.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let mut fields = line.split_whitespace();
        let id: usize = fields.next().unwrap().parse().unwrap();
        let family = match id {
            2 => WaveletFamily::Haar,
            4..=20 => WaveletFamily::Daubechies(id),
            _ => WaveletFamily::BiorthogonalSpline(id),
        };
        let n: usize = fields.next().unwrap().parse().unwrap();
        let levels: usize = fields.next().unwrap().parse().unwrap();
        let phase = if fields.next().unwrap() == "0" {
            WaveletPhase::Uncentered
        } else {
            WaveletPhase::Centered
        };
        let boundary = if fields.next().unwrap() == "zero" {
            SignalBoundary::Zero
        } else {
            SignalBoundary::Reflect
        };
        let maximum: usize = fields.next().unwrap().parse().unwrap();
        assert_eq!(wavelet_max_level(n, family).unwrap(), maximum);
        let expected: Vec<f64> = fields.map(|s| s.parse().unwrap()).collect();
        let options = WaveletOptions::default()
            .with_boundary(boundary)
            .with_phase(phase);
        let plan = WaveletPlan::<S>::new(family, n, levels, options).unwrap();
        let input: Vec<S> = (0..n)
            .map(|i| S::try_from_f64_lossy(((i * 7 + 3) % 17) as f64 / 8. - 1.).unwrap())
            .collect();
        let actual = plan.analyze(&input).unwrap();
        assert_eq!(actual.len(), expected.len(), "{family:?} {n} {phase:?}");
        let tolerance = (64. * S::EPSILON.into_f64()).min(1e-5);
        for (a, &b) in actual.iter().zip(&expected) {
            assert!(
                (a.into_f64() - b).abs() <= tolerance * b.abs().max(1.),
                "{family:?} {n} {phase:?} {boundary:?}: {} != {b}",
                a.into_f64()
            );
        }
        let independent: Vec<S> = expected
            .into_iter()
            .map(|x| S::try_from_f64_lossy(x).unwrap())
            .collect();
        for (a, b) in plan.synthesize(&independent).unwrap().iter().zip(&input) {
            assert!(
                (a.into_f64() - b.into_f64()).abs() <= 2. * tolerance,
                "inverse {family:?} {n} {phase:?} {boundary:?}"
            );
        }
        let mut count = n;
        for level in plan.levels() {
            count += 2 * level.band_len() - level.input_len();
        }
        assert_eq!(count, plan.coefficient_len());
        let k = plan.filters().analysis_low().len();
        let per_level = if phase == WaveletPhase::Centered && (k / 2) % 2 == 1 {
            k
        } else {
            k - 1
        };
        assert!(count <= n + levels * per_level);
        if (phase == WaveletPhase::Uncentered || (k / 2).is_multiple_of(2))
            && plan
                .levels()
                .iter()
                .all(|level| level.input_len().is_multiple_of(2))
        {
            assert!(count <= n + levels * (k - 2));
        }
        assert!(
            matches!(WaveletPlan::<S>::new(family, n, maximum + 1, options), Err(e) if matches!(e.view(), SciOxideErrorView::InvalidInput { .. }))
        );
    }
}

#[test]
fn pywavelets_f64() {
    reference::<f64>();
}
#[test]
fn pywavelets_f32() {
    reference::<f32>();
}

/// Separable PyWavelets references (`wavedec2` and `fswavedecn`, mapped to
/// SciOxide's subband packing by the generator); same tolerances as 1-D.
fn reference_2d<S: RealScalar>() {
    let tolerance = (64. * S::EPSILON.into_f64()).min(1e-5);
    let lines: Vec<_> = include_str!("fixtures/nonperiodic_wavelets_2d.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(3) {
        let header: Vec<_> = case[0].split_whitespace().collect();
        let field = |i: usize| -> usize { header[i].parse().unwrap() };
        let family = match field(0) {
            2 => WaveletFamily::Haar,
            id if id <= 20 => WaveletFamily::Daubechies(id),
            id => WaveletFamily::BiorthogonalSpline(id),
        };
        let shape = [field(1), field(2)];
        let layout = if header[3] == "pyramid" {
            Wavelet2dLayout::Pyramid
        } else {
            Wavelet2dLayout::Tensor
        };
        let options = WaveletOptions::default()
            .with_boundary(if header[7] == "zero" {
                SignalBoundary::Zero
            } else {
                SignalBoundary::Reflect
            })
            .with_phase(if header[6] == "0" {
                WaveletPhase::Uncentered
            } else {
                WaveletPhase::Centered
            });
        let plan =
            Wavelet2dPlan::<S>::new(family, shape, [field(4), field(5)], layout, options).unwrap();
        let subbands: Vec<usize> = case[1]
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(plan.subbands().len() * 4, subbands.len(), "{}", case[0]);
        let mut offset = 0;
        for (band, expected) in plan.subbands().iter().zip(subbands.chunks_exact(4)) {
            assert_eq!(band.offset(), offset, "{}", case[0]);
            assert_eq!(band.shape(), [expected[0], expected[1]], "{}", case[0]);
            assert_eq!(
                band.detail_levels(),
                [expected[2], expected[3]],
                "{}",
                case[0]
            );
            offset += expected[0] * expected[1];
        }
        let expected: Vec<f64> = case[2]
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(plan.coefficient_len(), expected.len(), "{}", case[0]);
        let input: Vec<S> = (0..shape[0] * shape[1])
            .map(|k| {
                let (r, c) = (k / shape[1], k % shape[1]);
                S::try_from_f64_lossy(((r * 7 + c * 5 + 3) % 17) as f64 / 8. - 1.).unwrap()
            })
            .collect();
        for (a, &b) in plan.analyze(&input).unwrap().iter().zip(&expected) {
            assert!(
                (a.into_f64() - b).abs() <= tolerance * b.abs().max(1.),
                "{}: {} != {b}",
                case[0],
                a.into_f64()
            );
        }
        let independent: Vec<S> = expected
            .into_iter()
            .map(|x| S::try_from_f64_lossy(x).unwrap())
            .collect();
        for (a, b) in plan.synthesize(&independent).unwrap().iter().zip(&input) {
            assert!(
                (a.into_f64() - b.into_f64()).abs() <= 2. * tolerance,
                "inverse {}",
                case[0]
            );
        }
    }
}

#[test]
fn pywavelets_2d_f64() {
    reference_2d::<f64>();
}
#[test]
fn pywavelets_2d_f32() {
    reference_2d::<f32>();
}

#[test]
fn short_long_filter_is_identity_and_periodic_remains_explicit() {
    let family = WaveletFamily::Daubechies(20);
    for boundary in [SignalBoundary::Zero, SignalBoundary::Reflect] {
        let options = WaveletOptions::default().with_boundary(boundary);
        let plan =
            WaveletPlan::<f64>::new(family, 8, wavelet_max_level(8, family).unwrap(), options)
                .unwrap();
        assert_eq!(plan.coefficient_len(), 8);
        assert_eq!(plan.analyze(&[1.; 8]).unwrap(), [1.; 8]);
        assert!(
            matches!(Wavelet2dPlan::<f64>::new(family, [8, 8], [1, 1], Wavelet2dLayout::Pyramid, options), Err(e) if matches!(e.view(), SciOxideErrorView::InvalidInput { .. }))
        );
    }
    assert_eq!(
        WaveletPlan::<f64>::new(family, 8, 3, WaveletOptions::default())
            .unwrap()
            .coefficient_len(),
        8
    );
    assert!(
        matches!(wavelet_max_level(8, WaveletFamily::Daubechies(3)), Err(e) if matches!(e.view(), SciOxideErrorView::InvalidInput { .. }))
    );
}

#[test]
fn odd_padding_requires_the_parity_correction() {
    let plan = WaveletPlan::<f64>::new(
        WaveletFamily::Haar,
        3,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Zero),
    )
    .unwrap();
    assert_eq!(plan.coefficient_len(), 4);
    assert_eq!(
        wavelet_max_level(usize::MAX, WaveletFamily::Haar).unwrap(),
        usize::BITS as usize - 1
    );
}
