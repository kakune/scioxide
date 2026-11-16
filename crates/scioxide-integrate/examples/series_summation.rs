//! Run with `cargo run -p scioxide-integrate --example series_summation`.
use scioxide_core::{Budget, ErrorTolerance, NormSelection, SciOxideError};
use scioxide_integrate::{
    EulerMaclaurinProvider, SeriesMethod, SeriesOptions, SeriesProblem, SeriesSummation,
    SeriesTerms,
};

struct InverseSquare;

impl SeriesTerms<f64> for InverseSquare {
    fn term(&self, n: usize, output: &mut [f64]) -> Result<(), SciOxideError> {
        let x = n as f64;
        output[0] = 1.0 / (x * x);
        Ok(())
    }
}

impl EulerMaclaurinProvider<f64> for InverseSquare {
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        value[0] = 1.0 / at as f64;
        error[0] = 64.0 * f64::EPSILON * value[0].abs();
        Ok(())
    }

    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order + 1).fold(1.0, |acc, k| acc * k as f64);
        value[0] = -factorial / (at as f64).powi((order + 2) as i32);
        error[0] = 64.0 * f64::EPSILON * value[0].abs();
        Ok(())
    }

    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        output: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order).fold(1.0, |acc, k| acc * k as f64);
        output[0] = factorial / (at as f64).powi((order + 1) as i32);
        output[0] *= 1.0 + 64.0 * f64::EPSILON;
        Ok(())
    }
}

fn main() -> Result<(), SciOxideError> {
    let tolerance = ErrorTolerance::scalar(1e-8_f64, 0.0, NormSelection::Infinity)?;
    let levin = SeriesSummation::<f64>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(24)?)
            .with_order(Budget::try_from(10)?)
            .with_tolerance(tolerance.clone()),
    )?;
    let accelerated = levin.sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
        let value: Result<_, SciOxideError> = {
            let x = (n + 1) as f64;
            Ok(1.0 / (x * x))
        };
        row[0] = value?;
        Ok(())
    }))?;
    println!("Levin-u estimate: {accelerated:?}");

    let smooth = SeriesSummation::<f64>::new(
        SeriesMethod::EulerMaclaurin,
        1,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(32)?)
            .with_order(Budget::try_from(2)?)
            .with_tolerance(tolerance),
    )?;
    let bounded = smooth.sum(SeriesProblem::EulerMaclaurin(&InverseSquare))?;
    println!("Euler–Maclaurin with conditional tail bound: {bounded:?}");
    Ok(())
}
