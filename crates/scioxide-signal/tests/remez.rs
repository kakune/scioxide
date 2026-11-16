//! Independent SciPy references and barycentric-exchange boundary cases.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, SciOxideErrorKind};
use scioxide_signal::{
    FirBand, FirSymmetry, FirTermination, RemezOptions, design_fir_remez_with_options,
};

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}
fn amplitude<S: RealScalar>(h: &[S], f: f64, odd: bool) -> f64 {
    let midpoint = (h.len() - 1) as f64 / 2.0;
    h.iter()
        .enumerate()
        .map(|(i, h)| {
            let phase = std::f64::consts::PI * f * (midpoint - i as f64);
            h.into_f64() * if odd { phase.sin() } else { phase.cos() }
        })
        .sum()
}
fn references<S: RealScalar>(large: bool) {
    let lines: Vec<_> = include_str!("fixtures/remez_scipy.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(5) {
        let header: Vec<_> = case[0].split_whitespace().collect();
        let name = header[0];
        let n: usize = header[1].parse().unwrap();
        if n > 64 && !large {
            continue;
        }
        let odd = header[2] == "1";
        let density: usize = header[3].parse().unwrap();
        let bands: Vec<_> = numbers(case[1])
            .chunks_exact(5)
            .map(|b| {
                FirBand::new(
                    S::try_from_f64_lossy(b[0]).unwrap(),
                    S::try_from_f64_lossy(b[1]).unwrap(),
                    S::try_from_f64_lossy(b[2]).unwrap(),
                    S::try_from_f64_lossy(b[3]).unwrap(),
                )
                .with_weight(S::try_from_f64_lossy(b[4]).unwrap())
            })
            .collect();
        let expected = numbers(case[2]);
        let diagnostics = numbers(case[3]);
        let extrema = numbers(case[4]);
        let symmetry = if odd {
            FirSymmetry::Odd
        } else {
            FirSymmetry::Even
        };
        let result = design_fir_remez_with_options(
            n,
            symmetry,
            &bands,
            RemezOptions::default()
                .with_grid_density(density)
                .with_tolerance(
                    scioxide_core::ResidualTolerance::new(
                        S::EPSILON * S::try_from_f64_lossy(16.0).unwrap(),
                        S::EPSILON * S::try_from_f64_lossy(16.0).unwrap(),
                    )
                    .unwrap(),
                ),
        )
        .unwrap();
        assert_eq!(
            result.termination(),
            FirTermination::Converged,
            "{name} {} {:?}",
            result.iterations(),
            result.weighted_ripple()
        );
        assert!(result.condition_estimate().is_none());
        let m = if odd { n / 2 } else { n.div_ceil(2) };
        let step = 1.0 / ((m + 1) * density) as f64;
        // At a smooth extremum, missing the maximum on a grid costs at most
        // M2*step²/8. Endpoints are included exactly. Both implementations use
        // finite grids, so their ripple may differ by the sum of these bounds.
        let grid_bound = diagnostics[1] * step * step / 4.0;
        let rounding = 64.0 * n as f64 * S::EPSILON.into_f64();
        let tolerance = grid_bound + rounding;
        let ripple = result.weighted_ripple().unwrap().into_f64();
        assert!(
            (ripple - diagnostics[0]).abs() <= tolerance,
            "{name}: {ripple} vs {} bound {tolerance}",
            diagnostics[0]
        );
        assert_eq!(result.extrema().len(), m + 1);
        for &f in result.extrema() {
            let f = f.into_f64();
            assert!(
                extrema
                    .iter()
                    .any(|&e| (e - f).abs() <= 2.0 * step + rounding),
                "{name}: extremum {f}"
            );
        }
        for b in &bands {
            for k in 0..=128 {
                let f = b.lower().into_f64()
                    + (b.upper().into_f64() - b.lower().into_f64()) * k as f64 / 128.0;
                let actual = amplitude(result.coefficients(), f, odd);
                let reference = amplitude(&expected, f, odd);
                // Minimax coefficients depend on the exchange grid. Bound the
                // response difference at the discretization scale, not 1 ulp.
                assert!(
                    (actual - reference).abs() <= 8.0 * tolerance / b.weight().into_f64(),
                    "{name}: f={f} {actual} vs {reference}, bound {}",
                    8.0 * tolerance / b.weight().into_f64()
                );
                let target = b.desired_lower().into_f64();
                assert!(
                    (actual - target).abs() * b.weight().into_f64() <= ripple + tolerance,
                    "{name}: taps disagree with diagnostics at {f}"
                );
            }
        }
    }
}
#[test]
fn scipy_f64_including_large_narrow_transition() {
    references::<f64>(true);
}
#[test]
fn scipy_f32() {
    references::<f32>(false);
}

fn band(lower: f64, upper: f64, desired_lower: f64, desired_upper: f64) -> FirBand<f64> {
    FirBand::new(lower, upper, desired_lower, desired_upper)
}
fn tight<S: RealScalar>() -> RemezOptions<S> {
    let sixteen = S::try_from_f64_lossy(16.0).unwrap();
    RemezOptions::default().with_tolerance(
        scioxide_core::ResidualTolerance::new(sixteen * S::EPSILON, sixteen * S::EPSILON).unwrap(),
    )
}
// Lowpass specs whose binary32 exchange used to underflow its barycentric
// weights at these lengths, although the binary64 ripple is far above
// binary32 resolution.
#[test]
fn binary32_matches_binary64_at_ordinary_orders() {
    for (pass, stop, n) in [(0.2_f32, 0.3_f32, 81), (0.3, 0.35, 121), (0.4, 0.42, 241)] {
        let narrow = [
            FirBand::new(0.0_f32, pass, 1.0, 1.0),
            FirBand::new(stop, 1.0, 0.0, 0.0),
        ];
        let wide = narrow.map(|b| {
            FirBand::new(
                f64::from(b.lower()),
                f64::from(b.upper()),
                f64::from(b.desired_lower()),
                f64::from(b.desired_upper()),
            )
        });
        let single = design_fir_remez_with_options(n, FirSymmetry::Even, &narrow, tight()).unwrap();
        let double = design_fir_remez_with_options(n, FirSymmetry::Even, &wide, tight()).unwrap();
        assert_eq!(single.termination(), FirTermination::Converged, "{n}");
        assert_eq!(double.termination(), FirTermination::Converged, "{n}");
        // Both exchanges meet a 16-epsilon spread on the same minimax problem;
        // binary32 reconstruction rounds at the scale of its coefficient mass.
        let epsilon = f64::from(f32::EPSILON);
        let mass: f64 = double.coefficients().iter().map(|h| h.abs()).sum();
        for (&a, &b) in single.coefficients().iter().zip(double.coefficients()) {
            assert!(
                (f64::from(a) - b).abs() <= 16.0 * epsilon * mass,
                "{n}: {a} vs {b}"
            );
        }
        let ripple = double.weighted_ripple().unwrap();
        assert!(ripple > 1e-4, "{n}: well posed in binary32");
        let difference = (f64::from(single.weighted_ripple().unwrap()) - ripple).abs();
        assert!(difference <= 64.0 * n as f64 * epsilon * ripple, "{n}");
    }
}
// The first reference follows the equilibrium measure of the bands, so the
// exchange resolves ripples down to near binary64 resolution where a uniform
// reference stalled (201, bandpass 8) or underflowed (801). The weighted
// error alternates at the last reference with the levelled magnitude, which
// bounds the grid minimax error from below (de la Vallée Poussin); the final
// exchange step moves that magnitude by at most 2.7e-4 of the ripple
// (observed with std and no_std elementary functions).
#[test]
fn binary64_long_designs_certify_alternation() {
    let lowpass = |pass, stop| vec![band(0.0, pass, 1.0, 1.0), band(stop, 1.0, 0.0, 0.0)];
    let bandpass = vec![
        band(0.0, 0.1, 0.0, 0.0),
        band(0.2, 0.4, 1.0, 1.0),
        band(0.5, 1.0, 0.0, 0.0),
    ];
    for (n, bands) in [
        (201, lowpass(0.2, 0.3)),
        (801, lowpass(0.4, 0.42)),
        (8, bandpass),
    ] {
        let result = design_fir_remez_with_options(
            n,
            FirSymmetry::Even,
            &bands,
            tight::<f64>().with_grid_density(16),
        )
        .unwrap();
        assert_eq!(result.termination(), FirTermination::Converged, "{n}");
        let ripple = result.weighted_ripple().unwrap();
        let mut sign = 0.0;
        for &f in result.extrema() {
            let b = bands
                .iter()
                .find(|b| (b.lower()..=b.upper()).contains(&f))
                .unwrap();
            let error = amplitude(result.coefficients(), f, false) - b.desired_lower();
            assert!(error * sign <= 0.0, "{n}: alternation at {f}");
            assert!(error.abs() >= (1.0 - 1e-3) * ripple, "{n}: {error} at {f}");
            sign = error;
        }
    }
}
#[test]
fn minimum_sizes_and_exact_zero_response() {
    for (n, symmetry) in [
        (1, FirSymmetry::Even),
        (2, FirSymmetry::Even),
        (2, FirSymmetry::Odd),
        (3, FirSymmetry::Odd),
    ] {
        let result = design_fir_remez_with_options(
            n,
            symmetry,
            &[band(0.0, 1.0, 0.0, 0.0)],
            RemezOptions::default(),
        )
        .unwrap();
        assert_eq!(result.termination(), FirTermination::Converged);
        assert!(result.coefficients().iter().all(|&h| h == 0.0));
        assert_eq!(result.weighted_ripple(), Some(0.0));
        assert_eq!(result.iterations(), 1);
    }
    let constant = design_fir_remez_with_options(
        1,
        FirSymmetry::Even,
        &[band(0.0, 1.0, 2.0, 2.0)],
        RemezOptions::default(),
    )
    .unwrap();
    assert!((constant.coefficients()[0] - 2.0).abs() < 16.0 * f64::EPSILON);
}
#[test]
fn linear_targets_and_narrow_bands() {
    for bands in [
        vec![band(0.0, 0.8, 0.0, 1.0)],
        vec![band(0.2, 0.20001, 1.0, 1.0), band(0.4, 1.0, 0.0, 0.0)],
    ] {
        let result =
            design_fir_remez_with_options(15, FirSymmetry::Even, &bands, RemezOptions::default())
                .unwrap();
        assert_eq!(result.termination(), FirTermination::Converged);
        for b in bands {
            for k in 0..=128 {
                let f = b.lower() + (b.upper() - b.lower()) * k as f64 / 128.0;
                let desired = b.desired_lower()
                    + (b.desired_upper() - b.desired_lower()) * (f - b.lower())
                        / (b.upper() - b.lower());
                assert!(
                    (amplitude(result.coefficients(), f, false) - desired).abs()
                        <= result.weighted_ripple().unwrap() + 2e-4
                );
            }
        }
    }
}
#[test]
fn unrepresentable_nodes_and_invalid_options_are_typed() {
    let bands = [band(0.0, 0.3, 1.0, 1.0), band(0.4, 1.0, 0.0, 0.0)];
    for options in [
        RemezOptions::default().with_max_iterations(0),
        RemezOptions::default().with_grid_density(15),
    ] {
        assert_eq!(
            design_fir_remez_with_options(31, FirSymmetry::Even, &bands, options)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    let near_zero = [band(0.0, 1e-12, 1.0, 1.0)];
    assert_eq!(
        design_fir_remez_with_options(3, FirSymmetry::Even, &near_zero, RemezOptions::default())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn default_tolerance_reaches_the_minimax_ripple() {
    // A sqrt(epsilon) absolute floor used to stop binary32 exchanges once the
    // ripple fell below 3.5e-4, far above the optimum, as "converged".
    fn ripples<S: RealScalar>(pass: f64, stop: f64, n: usize) -> (f64, f64) {
        let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
        let bands = [
            FirBand::new(s(0.0), s(pass), S::ONE, S::ONE),
            FirBand::new(s(stop), S::ONE, S::ZERO, S::ZERO),
        ];
        let default =
            design_fir_remez_with_options(n, FirSymmetry::Even, &bands, RemezOptions::new())
                .unwrap();
        let reference =
            design_fir_remez_with_options(n, FirSymmetry::Even, &bands, tight()).unwrap();
        assert!(default.is_success() && reference.is_success(), "{n}");
        (
            default.weighted_ripple().unwrap().into_f64(),
            reference.weighted_ripple().unwrap().into_f64(),
        )
    }
    for (pass, stop, n) in [(0.2, 0.3, 81), (0.2, 0.3, 121), (0.1, 0.2, 61)] {
        let (default, optimum) = ripples::<f32>(pass, stop, n);
        assert!(default <= optimum * 1.01, "f32 {n}: {default} vs {optimum}");
    }
    for (pass, stop, n) in [(0.2, 0.3, 201), (0.2, 0.21, 2001)] {
        let (default, optimum) = ripples::<f64>(pass, stop, n);
        assert!(default <= optimum * 1.01, "f64 {n}: {default} vs {optimum}");
    }
}
