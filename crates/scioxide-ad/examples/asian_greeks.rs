//! Price and Greeks of an arithmetic-average Asian call with one batch function.
//!
//! The same `BatchFunction` runs on plain SIMD lanes for the price and on forward
//! mode for the Greeks; a counter-based normal source supplies the draws.
use scioxide_ad::{
    BatchDataFn, BatchFunction, BatchOptions, BatchOutputs, BatchPlan, BatchReduction, Real,
};
use scioxide_core::{ExecutionContext, SciOxideError};

/// Arithmetic-average Asian call under geometric Brownian motion.
struct AsianCall {
    strike: f64,
    maturity: f64,
    steps: usize,
}

/// Quantities that are the same on every path.
struct Market<R> {
    spot: R,
    drift: R,
    vol: R,
    discount: R,
}

impl BatchFunction for AsianCall {
    type Shared<R: Real> = Market<R>;

    fn prepare<R: Real>(&self, parameters: &[R]) -> Result<Market<R>, SciOxideError> {
        let &[spot, sigma, rate] = parameters else {
            return Err(SciOxideError::dimension_mismatch(3, parameters.len()));
        };
        let dt = self.maturity / self.steps as f64;
        Ok(Market {
            spot,
            drift: (rate - sigma * sigma * 0.5) * dt,
            vol: sigma * dt.sqrt(),
            discount: (-rate * self.maturity).exp(),
        })
    }

    fn item<R: Real>(
        &self,
        market: &Market<R>,
        _: &[R],
        normals: &[R::Base],
    ) -> Result<R, SciOxideError> {
        let mut spot = market.spot;
        let mut sum = R::zero();
        for &z in normals {
            spot *= (market.drift + market.vol * R::passive(z)).exp();
            sum += spot;
        }
        let average = sum / normals.len() as f64;
        Ok(market.discount * (average - self.strike).max(R::zero()))
    }
}

fn uniform(counter: u64) -> f64 {
    // SplitMix64 finalizer of the counter.
    let mut z = counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64) + 0.5 / (1_u64 << 53) as f64
}

/// Writes standard normal draws of `step` for the items `first..`: the counter
/// `(item, step)` gives two uniforms, and Box–Muller a normal.
fn normals(
    first: usize,
    width: usize,
    step: usize,
    lanes: &mut [f64],
) -> Result<(), SciOxideError> {
    for (item, z) in (first..).zip(lanes) {
        let counter = 2 * (item as u64 * width as u64 + step as u64);
        let (u, v) = (uniform(counter), uniform(counter + 1));
        *z = (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos();
    }
    Ok(())
}

fn main() -> Result<(), SciOxideError> {
    let option = AsianCall {
        strike: 100.0,
        maturity: 1.0,
        steps: 252,
    };
    let theta = [100.0, 0.2, 0.03]; // spot, volatility, rate
    let paths = 4096;
    let plan = BatchPlan::<8>::new(
        paths,
        3,
        0,
        option.steps,
        BatchReduction::Mean,
        BatchOptions::new(),
    )?;
    let mut workspace = plan.workspace(1)?;
    let cx = ExecutionContext::sequential_native();
    // Nothing is stored: the source draws each value when a group asks for it.
    let draws = BatchDataFn::new(normals);
    // Price only: `item` runs on plain SIMD lanes.
    let price = plan.evaluate_into(
        &option,
        &theta,
        &[],
        &draws,
        BatchOutputs::new(),
        &mut workspace,
        cx,
    )?;
    // Price and Greeks: the same function, plan and workspace.
    let mut greeks = [0.0; 3];
    let mut errors = [0.0; 3];
    let outputs = BatchOutputs::new()
        .with_parameter_gradient(&mut greeks)
        .with_parameter_gradient_error(&mut errors);
    let estimate = plan.evaluate_into(&option, &theta, &[], &draws, outputs, &mut workspace, cx)?;
    assert_eq!(price.value().to_bits(), estimate.value().to_bits());
    let se = estimate.standard_error().unwrap_or(f64::NAN);
    println!(
        "price {:.4} ± {se:.4}, {:?}",
        estimate.value(),
        estimate.mode()
    );
    for (name, (g, e)) in ["delta", "vega", "rho"]
        .iter()
        .zip(greeks.iter().zip(&errors))
    {
        println!("{name:>5} {g:.4} ± {e:.4}");
    }
    assert!((estimate.value() - 5.24).abs() < 4.0 * se);
    assert!(greeks[0] > 0.0 && greeks[1] > 0.0 && greeks[2] > 0.0);
    // One path on plain numbers, for a debugger or a unit test.
    let market = option.prepare(&theta)?;
    let mut row = [0.0; 252];
    for (step, z) in row.iter_mut().enumerate() {
        normals(0, option.steps, step, core::slice::from_mut(z))?;
    }
    let payoff: f64 = option.item(&market, &[], &row)?;
    assert!(payoff >= 0.0);
    Ok(())
}
