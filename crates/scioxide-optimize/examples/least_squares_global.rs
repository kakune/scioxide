#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args,
    clippy::similar_names,
    clippy::cast_precision_loss
)]
//! Least-squares and global optimization demonstration for scioxide-optimize.
//!
//! Demonstrates curve fitting and global search solvers:
//!
//! 1. **Gauss–Newton** — Exponential curve fitting with `LeastSquares`
//! 2. **Levenberg–Marquardt** — Same problem, comparing globalization
//! 3. **Multistart** — Multi-start global search on Rastrigin
//! 4. **DifferentialEvolution** — Evolutionary optimization on Rastrigin
//!
//! Run: `cargo run --example least_squares_global -p scioxide-optimize`

use scioxide_core::{Bounds, Budget, SciOxideError, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    BoundedObjective, DeStrategy, DifferentialEvolution, DifferentialEvolutionOptions,
    GlobalSearchOptions, LeastSquares, LeastSquaresGlobalization, LeastSquaresOptions,
    LeastSquaresProblem, MultiStart, ObjectiveFunction, PowellLocalFactory, UniformStartDesign,
};

/// Tolerance for floating-point comparisons.
const TOL: f64 = 0.1;

// ---------------------------------------------------------------------------
// Helper: Exponential fit y = a * exp(b * x) + c
// ---------------------------------------------------------------------------

struct ExpFit {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl ExpFit {
    fn new() -> Self {
        let x_data: Vec<f64> = (0..10).map(|i| f64::from(i) * 0.5).collect();
        // Generate data from y = 2.0 * exp(-0.5 * x) + 0.5
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 2.0 * (-0.5 * x).exp() + 0.5)
            .collect();
        Self { x_data, y_data }
    }
}

impl LeastSquaresProblem<f64> for ExpFit {
    fn param_dim(&self) -> usize {
        3 // a, b, c
    }

    fn residual_dim(&self) -> usize {
        self.x_data.len()
    }

    fn residual_into(&self, params: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        let a = params[0];
        let b = params[1];
        let c = params[2];
        output
            .iter_mut()
            .zip(self.x_data.iter().zip(&self.y_data))
            .for_each(|(residual, (&x, &y))| *residual = a * (b * x).exp() + c - y);
        Ok(())
    }

    fn jacobian_into(
        &self,
        params: &[f64],
        output: &mut Matrix<f64>,
    ) -> Result<bool, SciOxideError> {
        let a = params[0];
        let b = params[1];
        output
            .as_mut_slice()
            .chunks_exact_mut(3)
            .zip(&self.x_data)
            .for_each(|(row, &x)| {
                let eb = (b * x).exp();
                row.copy_from_slice(&[eb, a * x * eb, 1.0]);
            });
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Helper: Rastrigin function (common global optimization test)
// ---------------------------------------------------------------------------

struct Rastrigin {
    n: usize,
}

impl ObjectiveFunction<f64> for Rastrigin {
    fn dim(&self) -> usize {
        self.n
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        let n = self.n as f64;
        let sum: f64 = x
            .iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        Ok(10.0 * n + sum)
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.iter_mut().zip(x).for_each(|(gradient, &xi)| {
            *gradient =
                2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin();
        });
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Part 1: GaussNewton
// ---------------------------------------------------------------------------

fn part1_gauss_newton() -> Result<(), SciOxideError> {
    println!("=== Part 1: GaussNewton curve fitting ===");
    println!();

    let problem = ExpFit::new();
    let x0 = [1.5, -0.3, 0.3];
    let options = LeastSquaresOptions::default()
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_max_iterations(Budget::try_from(200)?)
        .with_globalization(LeastSquaresGlobalization::gauss_newton());
    let res = LeastSquares::<f64>::new(options)?.solve(&problem, &x0)?;

    println!("  Fitting y = a * exp(b*x) + c to synthetic data:");
    println!(
        "  a = {:.6}, b = {:.6}, c = {:.6}  cost = {:.2e}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.x()[2],
        res.cost()?,
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "GaussNewton should converge");
    assert!((res.x()[0] - 2.0).abs() < TOL, "a should be near 2.0");
    assert!((res.x()[1] - (-0.5)).abs() < TOL, "b should be near -0.5");
    assert!((res.x()[2] - 0.5).abs() < TOL, "c should be near 0.5");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: LevenbergMarquardt
// ---------------------------------------------------------------------------

fn part2_levenberg_marquardt() -> Result<(), SciOxideError> {
    println!("=== Part 2: LevenbergMarquardt curve fitting ===");
    println!();

    let problem = ExpFit::new();
    let x0 = [1.0, -1.0, 1.0];
    let options = LeastSquaresOptions::default()
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_max_iterations(Budget::try_from(200)?)
        .with_globalization(LeastSquaresGlobalization::levenberg_marquardt());
    let res = LeastSquares::<f64>::new(options)?.solve(&problem, &x0)?;

    println!("  Fitting y = a * exp(b*x) + c (Levenberg-Marquardt):");
    println!(
        "  a = {:.6}, b = {:.6}, c = {:.6}  cost = {:.2e}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.x()[2],
        res.cost()?,
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "LevenbergMarquardt should converge");
    assert!((res.x()[0] - 2.0).abs() < TOL, "a should be near 2.0");
    assert!((res.x()[1] - (-0.5)).abs() < TOL, "b should be near -0.5");
    assert!((res.x()[2] - 0.5).abs() < TOL, "c should be near 0.5");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Multistart global search
// ---------------------------------------------------------------------------

fn part3_multistart() -> Result<(), SciOxideError> {
    println!("=== Part 3: Multistart on Rastrigin ===");
    println!();

    let f = Rastrigin { n: 2 };
    let bounds = Bounds::new(vec![(Some(-5.12), Some(5.12)); 2])?;
    let options = GlobalSearchOptions::default()
        .with_starts(Budget::try_from(20)?)
        .with_seed(42);
    let solver = MultiStart::new(options, UniformStartDesign, PowellLocalFactory::default())?;
    let search = solver.minimize(&BoundedObjective::new(&f, &bounds)?)?;
    let res = search
        .best()
        .ok_or_else(|| SciOxideError::external("no finite candidate"))?;
    println!(
        "  Global termination: {:?}; local successes: {}",
        search.termination(),
        search.successes()
    );

    println!("  Multistart (20 starts) on 2D Rastrigin:");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Global minimum at (0, 0) with value 0
    assert!(res.fun() < 1.0, "Should find near-global minimum");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: DifferentialEvolution
// ---------------------------------------------------------------------------

fn part4_differential_evolution() -> Result<(), SciOxideError> {
    println!("=== Part 4: DifferentialEvolution on Rastrigin ===");
    println!();

    let f = Rastrigin { n: 2 };
    let x0 = [0.0, 0.0];
    let bounds = Bounds::new(vec![(Some(-5.12), Some(5.12)); 2])?;
    let problem = BoundedObjective::new(&f, &bounds)?;
    let options = DifferentialEvolutionOptions::default()
        .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(1e-8, 1e-8)?,
        ))
        .with_max_generations(Budget::try_from(500)?)
        .with_seed(42);

    // Rand1 strategy
    let solver_rand1 =
        DifferentialEvolution::new(options.clone().with_strategy(DeStrategy::Rand1))?;
    let res_rand1 = solver_rand1.minimize(&problem, &x0)?;

    println!("  DE Rand1 on 2D Rastrigin:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.6}  iters = {}",
        res_rand1.x()[0],
        res_rand1.x()[1],
        res_rand1.fun(),
        res_rand1.iterations()
    );

    // Best1 strategy
    let solver_best1 = DifferentialEvolution::new(options.with_strategy(DeStrategy::Best1))?;
    let res_best1 = solver_best1.minimize(&problem, &x0)?;

    println!("  DE Best1 on 2D Rastrigin:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.6}  iters = {}",
        res_best1.x()[0],
        res_best1.x()[1],
        res_best1.fun(),
        res_best1.iterations()
    );

    assert!(res_rand1.fun() < 5.0, "Rand1 should find a good minimum");
    assert!(res_best1.fun() < 5.0, "Best1 should find a good minimum");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-optimize Least Squares & Global Optimization");
    println!("=====================================================");
    println!();

    part1_gauss_newton()?;
    part2_levenberg_marquardt()?;
    part3_multistart()?;
    part4_differential_evolution()?;

    println!("All examples completed successfully.");
    Ok(())
}
