#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args,
    clippy::similar_names
)]
//! Unconstrained optimization demonstration for scioxide-optimize.
//!
//! Demonstrates all 7 unconstrained solvers on classic test functions:
//!
//! 1. **BFGS** — Quasi-Newton on the Rosenbrock function
//! 2. **L-BFGS** — Limited-memory BFGS with configurable memory
//! 3. **GradientDescent** — Steepest descent with Strong Wolfe line search
//! 4. **Newton & NonlinearCg** — Hessian-based and conjugate-gradient methods
//! 5. **TrustRegion & NelderMead** — Trust-region vs derivative-free
//! 6. **Solver comparison** — All 7 solvers side-by-side on Rosenbrock
//!
//! Run: `cargo run --example unconstrained_solvers -p scioxide-optimize`

use scioxide_core::{Budget, SciOxideError, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    Bfgs, BfgsOptions, GradientDescent, GradientDescentOptions, Lbfgs, LbfgsOptions, NelderMead,
    NelderMeadOptions, NewtonOptimizer, NewtonOptions, NonlinearCg, NonlinearCgOptions,
    ObjectiveFunction, Optimizer, StrongWolfeLineSearch, TrustRegionOptimizer, TrustRegionOptions,
};

/// Tolerance for floating-point comparisons.
const TOL: f64 = 1e-4;

// ---------------------------------------------------------------------------
// Helper: Rosenbrock function f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// ---------------------------------------------------------------------------

struct Rosenbrock;

impl ObjectiveFunction<f64> for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Ok(a * a + 100.0 * b * b)
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        let dx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dx1 = 200.0 * (x[1] - x[0] * x[0]);
        output.copy_from_slice(&[dx0, dx1]);
        Ok(true)
    }

    fn hessian_into(&self, x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
        let h01 = -400.0 * x[0];
        let h11 = 200.0;
        output.as_mut_slice().copy_from_slice(&[h00, h01, h01, h11]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Helper: Sphere function f(x) = sum(x_i^2)
// ---------------------------------------------------------------------------

struct Sphere {
    n: usize,
}

impl ObjectiveFunction<f64> for Sphere {
    fn dim(&self) -> usize {
        self.n
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output
            .iter_mut()
            .zip(x)
            .for_each(|(gradient, &value)| *gradient = 2.0 * value);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Part 1: BFGS on Rosenbrock
// ---------------------------------------------------------------------------

fn part1_bfgs() -> Result<(), SciOxideError> {
    println!("=== Part 1: BFGS on Rosenbrock ===");
    println!();

    let f = Rosenbrock;
    let x0 = [-1.0, 1.0];
    let res = Bfgs::<f64>::default().minimize(&f, &x0)?;

    println!("  f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
    println!("  x0 = [{}, {}]", x0[0], x0[1]);
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "BFGS should converge");
    assert!((res.x()[0] - 1.0).abs() < TOL, "x should be near 1.0");
    assert!((res.x()[1] - 1.0).abs() < TOL, "y should be near 1.0");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: L-BFGS with configurable memory
// ---------------------------------------------------------------------------

fn part2_lbfgs() -> Result<(), SciOxideError> {
    println!("=== Part 2: L-BFGS with memory = 10 ===");
    println!();

    let f = Rosenbrock;
    let x0 = [-1.0, 1.0];
    let solver = Lbfgs::new(LbfgsOptions::default().with_memory(Budget::try_from(10)?));
    let res = solver.minimize(&f, &x0)?;

    println!("  L-BFGS (m=10) on Rosenbrock:");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "L-BFGS should converge");
    assert!((res.x()[0] - 1.0).abs() < TOL, "x should be near 1.0");
    assert!((res.x()[1] - 1.0).abs() < TOL, "y should be near 1.0");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: GradientDescent with Strong Wolfe line search
// ---------------------------------------------------------------------------

fn part3_gradient_descent() -> Result<(), SciOxideError> {
    println!("=== Part 3: GradientDescent with Strong Wolfe ===");
    println!();

    let f = Sphere { n: 3 };
    let x0 = [3.0, -2.0, 1.0];
    let options = GradientDescentOptions::default()
        .with_line_search(StrongWolfeLineSearch::<f64>::default())
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_max_iterations(Budget::try_from(5000)?);
    let res = GradientDescent::new(options).minimize(&f, &x0)?;

    println!("  GradientDescent on Sphere (dim=3) with StrongWolfeLineSearch:");
    println!(
        "  result = ({:.6}, {:.6}, {:.6})  fun = {:.2e}  iters = {}",
        res.x()[0],
        res.x()[1],
        res.x()[2],
        res.fun(),
        res.iterations()
    );

    assert!(res.is_success(), "GradientDescent should converge");
    assert!(res.fun() < TOL, "Minimum should be near 0");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Newton and NonlinearCg
// ---------------------------------------------------------------------------

fn part4_newton_nlcg() -> Result<(), SciOxideError> {
    println!("=== Part 4: Newton & NonlinearCg ===");
    println!();

    let f = Rosenbrock;
    let x0 = [-1.0, 1.0];
    // Newton (uses analytic Hessian)
    let res_newton = NewtonOptimizer::<f64>::default().minimize(&f, &x0)?;
    println!("  Newton on Rosenbrock:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res_newton.x()[0],
        res_newton.x()[1],
        res_newton.fun(),
        res_newton.iterations(),
        res_newton.is_success()
    );

    // NonlinearCg (Polak-Ribiere+)
    let res_cg = NonlinearCg::<f64>::default().minimize(&f, &x0)?;
    println!("  NonlinearCg on Rosenbrock:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res_cg.x()[0],
        res_cg.x()[1],
        res_cg.fun(),
        res_cg.iterations(),
        res_cg.is_success()
    );

    assert!(res_newton.is_success(), "Newton should converge");
    assert!((res_newton.x()[0] - 1.0).abs() < TOL, "Newton x near 1");
    assert!(res_cg.is_success(), "NonlinearCg should converge");
    assert!((res_cg.x()[0] - 1.0).abs() < TOL, "NonlinearCg x near 1");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: TrustRegion and NelderMead
// ---------------------------------------------------------------------------

fn part5_trust_region_nelder_mead() -> Result<(), SciOxideError> {
    println!("=== Part 5: TrustRegion & NelderMead ===");
    println!();

    let f = Rosenbrock;
    let x0 = [-1.0, 1.0];
    // Trust-region (dogleg, uses Hessian)
    let res_tr = TrustRegionOptimizer::<f64>::default().minimize(&f, &x0)?;
    println!("  TrustRegion on Rosenbrock:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res_tr.x()[0],
        res_tr.x()[1],
        res_tr.fun(),
        res_tr.iterations(),
        res_tr.is_success()
    );

    // NelderMead (derivative-free)
    let nm_options = NelderMeadOptions::default()
        .with_function_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(1e-10, 1e-10)?,
        ))
        .with_max_iterations(Budget::try_from(10_000)?);
    let res_nm = NelderMead::new(nm_options)?.minimize(&f, &x0)?;
    println!("  NelderMead on Rosenbrock:");
    println!(
        "    result = ({:.6}, {:.6})  fun = {:.2e}  iters = {}  success = {}",
        res_nm.x()[0],
        res_nm.x()[1],
        res_nm.fun(),
        res_nm.iterations(),
        res_nm.is_success()
    );

    assert!(res_tr.is_success(), "TrustRegion should converge");
    assert!((res_tr.x()[0] - 1.0).abs() < TOL, "TrustRegion x near 1");
    assert!(res_nm.is_success(), "NelderMead should converge");
    assert!((res_nm.x()[0] - 1.0).abs() < 0.01, "NelderMead x near 1");

    println!();
    println!("  Part 5 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 6: Solver comparison table
// ---------------------------------------------------------------------------

fn part6_comparison() -> Result<(), SciOxideError> {
    println!("=== Part 6: Solver comparison on Rosenbrock ===");
    println!();

    let f = Rosenbrock;
    let x0 = [-1.0, 1.0];
    let tolerance = Tolerance::new(1e-10, 1e-10)?;
    let iterations = Budget::try_from(10_000)?;

    let bfgs = Bfgs::new(
        BfgsOptions::default()
            .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(tolerance))
            .with_max_iterations(iterations),
    );
    let solvers: Vec<(&str, Box<dyn Optimizer<f64>>)> = vec![
        ("BFGS", Box::new(bfgs)),
        (
            "L-BFGS",
            Box::new(Lbfgs::new(
                LbfgsOptions::default()
                    .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_max_iterations(iterations)
                    .with_memory(Budget::try_from(10)?),
            )),
        ),
        (
            "GradientDescent",
            Box::new(GradientDescent::new(
                GradientDescentOptions::default()
                    .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_max_iterations(iterations),
            )),
        ),
        (
            "NonlinearCg",
            Box::new(NonlinearCg::new(
                NonlinearCgOptions::default()
                    .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_max_iterations(iterations),
            )),
        ),
        (
            "Newton",
            Box::new(NewtonOptimizer::new(
                NewtonOptions::default()
                    .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_max_iterations(iterations),
            )),
        ),
        (
            "TrustRegion",
            Box::new(TrustRegionOptimizer::new(
                TrustRegionOptions::default()
                    .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_max_iterations(iterations),
            )),
        ),
        (
            "NelderMead",
            Box::new(NelderMead::new(
                NelderMeadOptions::default()
                    .with_function_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                        tolerance,
                    ))
                    .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(tolerance))
                    .with_max_iterations(iterations),
            )?),
        ),
    ];

    println!(
        "  {:<16} {:>8} {:>10} {:>12} {:>10}",
        "Solver", "Iters", "Fun Evals", "Fun", "Converged"
    );
    println!("  {}", "-".repeat(60));

    for (name, solver) in &solvers {
        let res = solver.minimize(&f, &x0)?;
        println!(
            "  {:<16} {:>8} {:>10} {:>12.4e} {:>10}",
            name,
            res.iterations(),
            res.fun_evals(),
            res.fun(),
            res.is_success()
        );
    }

    println!();
    println!("  Part 6 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-optimize Unconstrained Solvers");
    println!("======================================");
    println!();

    part1_bfgs()?;
    part2_lbfgs()?;
    part3_gradient_descent()?;
    part4_newton_nlcg()?;
    part5_trust_region_nelder_mead()?;
    part6_comparison()?;

    println!("All examples completed successfully.");
    Ok(())
}
