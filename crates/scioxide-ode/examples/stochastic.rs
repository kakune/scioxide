//! Seeded normal draws, a reproducible Brownian bridge, and two interpretations.
use scioxide_core::SciOxideError;
use scioxide_linalg::MatrixView;
use scioxide_ode::{SdeFn, SdeMethod, SdeOptions, SdeProblem};
use scioxide_random::{BrownianBridgePlan, Distribution, Normal, SeedableRng, SplitMix64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let times: Vec<_> = (0..=256).map(|i| i as f64 / 256.).collect();
    let normals = Normal::<f64>::standard().sample_n(&mut SplitMix64::from_seed(663), 256)?;
    let path = BrownianBridgePlan::new(&times)?.apply(&normals)?;
    let increments: Vec<_> = core::iter::once(path[0])
        .chain(path.windows(2).map(|p| p[1] - p[0]))
        .collect();
    for method in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun] {
        let options = SdeOptions::new().with_max_workspace_bytes(4096);
        let system = SdeFn::new(
            1,
            1,
            |_: f64, y: &[f64], out: &mut [f64]| {
                out[0] = 0.4 * y[0];
                Ok(())
            },
            |_: f64, y: &[f64], dw: &[f64], out: &mut [f64]| {
                out[0] = 0.7 * y[0] * dw[0];
                Ok(())
            },
        );
        let problem = SdeProblem::new(
            &system,
            &times,
            MatrixView::from_slice(256, 1, &increments)?,
        );
        let solution = method
            .solve(&problem, &options, &[1.])
            .map_err(SciOxideError::from)?;
        let (state, report) = solution.into_parts();
        let y = state.values()[0];
        let correction = if method == SdeMethod::EulerMaruyama {
            -0.5 * 0.7 * 0.7
        } else {
            0.
        };
        let exact = (0.4 + correction + 0.7 * path[255]).exp();
        println!(
            "{method:?}: {:?}, y={y:.8}, exact={exact:.8}, error={:.3e}",
            report.termination(),
            (y - exact).abs()
        );
    }
    // The solvers only borrowed increments. Sampling/generator state did not change.
    Ok(())
}
