#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args,
    clippy::similar_names
)]
//! Constrained optimization demonstration for scioxide-optimize.
//!
//! Demonstrates 4 constrained solvers with equality, inequality, and bounds:
//!
//! 1. **PenaltyMethod** — Inequality constraint on a quadratic
//! 2. **AugmentedLagrangian** — Equality constraint on a quadratic
//! 3. **SQP** — Rosenbrock with nonlinear inequality
//! 4. **InteriorPoint** — Primal-dual nonlinear interior point with inequality constraints
//! 5. **Bounds** — Box constraints with BFGS
//!
//! Run: `cargo run --example constrained_optimization -p scioxide-optimize`

use scioxide_core::Bounds;
use scioxide_core::SciOxideError;
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    AugmentedLagrangian, AugmentedLagrangianOptions, ConstrainedProblem, ConstraintFunction,
    Constraints, InteriorPoint, InteriorPointOptions, ObjectiveFunction, PenaltyMethod,
    PenaltyOptions, Sqp, SqpOptions,
};

/// Tolerance for floating-point comparisons.
const TOL: f64 = 0.05;

// ---------------------------------------------------------------------------
// Helper: Quadratic f(x) = x0^2 + x1^2
// ---------------------------------------------------------------------------

struct SumQuadratic;

impl ObjectiveFunction<f64> for SumQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] * x[1])
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        Ok(true)
    }

    fn hessian_into(&self, _x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 2.0]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Helper: Rosenbrock
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
// Helper: Constraint x0 + x1 - 1  (for <= 0 or == 0)
// ---------------------------------------------------------------------------

struct SumConstraint;

impl ConstraintFunction<f64> for SumConstraint {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 1.0)
    }

    fn jacobian_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.copy_from_slice(&[1.0, 1.0]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Helper: Nonlinear constraint x0^2 + x1^2 - 1.5 <= 0
// ---------------------------------------------------------------------------

struct CircleConstraint;

impl ConstraintFunction<f64> for CircleConstraint {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] * x[1] - 1.5)
    }

    fn jacobian_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Part 1: PenaltyMethod with inequality constraint
// ---------------------------------------------------------------------------

fn part1_penalty() -> Result<(), SciOxideError> {
    println!("=== Part 1: PenaltyMethod ===");
    println!();

    let f = SumQuadratic;
    let x0 = [2.0, 2.0];
    let constraints = Constraints::new(2)?.inequality(SumConstraint)?;
    let problem = ConstrainedProblem::new(&f, &constraints)?;
    let res = PenaltyMethod::new(PenaltyOptions::default()).minimize(&problem, &x0)?;

    println!("  min x0^2 + x1^2  s.t. x0 + x1 <= 1");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Unconstrained min is (0,0), constraint x0+x1<=1 is satisfied at origin
    assert!(res.is_success(), "PenaltyMethod should converge");
    assert!(res.fun() < TOL, "Minimum should be near 0");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: AugmentedLagrangian with equality constraint
// ---------------------------------------------------------------------------

fn part2_augmented_lagrangian() -> Result<(), SciOxideError> {
    println!("=== Part 2: AugmentedLagrangian ===");
    println!();

    let f = SumQuadratic;
    let x0 = [2.0, 0.0];
    let constraints = Constraints::new(2)?.equality(SumConstraint)?;
    let problem = ConstrainedProblem::new(&f, &constraints)?;
    let res =
        AugmentedLagrangian::new(AugmentedLagrangianOptions::default()).minimize(&problem, &x0)?;

    println!("  min x0^2 + x1^2  s.t. x0 + x1 = 1");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Lagrangian optimum at (0.5, 0.5)
    assert!(res.is_success(), "AugmentedLagrangian should converge");
    assert!(
        (res.x()[0] - 0.5).abs() < TOL,
        "x0 should be near 0.5, got {}",
        res.x()[0]
    );
    assert!(
        (res.x()[1] - 0.5).abs() < TOL,
        "x1 should be near 0.5, got {}",
        res.x()[1]
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: SQP with nonlinear inequality on Rosenbrock
// ---------------------------------------------------------------------------

fn part3_sqp() -> Result<(), SciOxideError> {
    println!("=== Part 3: SQP on Rosenbrock ===");
    println!();

    let f = Rosenbrock;
    let x0 = [0.0, 0.0];
    let constraints = Constraints::new(2)?.inequality(CircleConstraint)?;
    let problem = ConstrainedProblem::new(&f, &constraints)?;
    let res = Sqp::new(SqpOptions::default()).minimize(&problem, &x0)?;

    println!("  min Rosenbrock  s.t. x0^2 + x1^2 <= 1.5");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Verify constraint satisfaction
    let constraint_val = res.x()[0] * res.x()[0] + res.x()[1] * res.x()[1];
    println!(
        "  constraint value: x0^2+x1^2 = {:.6} (limit 1.5)",
        constraint_val
    );
    assert!(constraint_val < 1.5 + TOL, "Constraint should be satisfied");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: InteriorPoint with inequality constraints
// ---------------------------------------------------------------------------

fn part4_interior_point() -> Result<(), SciOxideError> {
    println!("=== Part 4: InteriorPoint ===");
    println!();

    let f = SumQuadratic;
    // Start bound-interior (x0+x1 < 1)
    let x0 = [0.1, 0.1];
    let constraints = Constraints::new(2)?.inequality(SumConstraint)?;
    let problem = ConstrainedProblem::new(&f, &constraints)?;
    let res = InteriorPoint::new(InteriorPointOptions::default()).minimize(&problem, &x0)?;

    println!("  min x0^2 + x1^2  s.t. x0 + x1 <= 1 (InteriorPoint)");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "InteriorPoint should converge");
    assert!(res.fun() < TOL, "Minimum should be near 0");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Bounds constraints with PenaltyMethod
// ---------------------------------------------------------------------------

fn part5_bounds() -> Result<(), SciOxideError> {
    println!("=== Part 5: Bounds constraints ===");
    println!();

    let f = SumQuadratic;
    let x0 = [2.0, 2.0];
    let bounds = Bounds::new(vec![(Some(0.5), Some(3.0)); 2])?;
    let constraints = Constraints::new(2)?.bounds(bounds)?;
    let problem = ConstrainedProblem::new(&f, &constraints)?;
    let res = PenaltyMethod::new(PenaltyOptions::default()).minimize(&problem, &x0)?;

    println!("  min x0^2 + x1^2  s.t. 0.5 <= x0,x1 <= 3.0");
    println!(
        "  result = ({:.6}, {:.6})  fun = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Constrained minimum at (0.5, 0.5)
    assert!(
        res.is_success(),
        "PenaltyMethod with bounds should converge"
    );
    assert!(
        (res.x()[0] - 0.5).abs() < TOL,
        "x0 should be near 0.5, got {}",
        res.x()[0]
    );

    println!();
    println!("  Part 5 assertions passed.");
    println!();
    Ok(())
}

// Restoration is explicit and counted, even with a short local iteration allowance:
// each restoration rejects the outer proposal before its feasibility subproblem.
fn part6_restoration() -> Result<(), SciOxideError> {
    struct Square;
    impl ObjectiveFunction<f64> for Square {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            g[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    let constraints = Constraints::new(1)?.equality(
        scioxide_optimize::ConstraintFn::new(1, |x: &[f64]| Ok(x[0] * x[0] - 1.0)).with_jacobian(
            |x: &[f64], output| {
                output.copy_from_slice(&[2.0 * x[0]]);
                Ok(())
            },
        ),
    )?;
    let problem = ConstrainedProblem::new(&Square, &constraints)?;
    let solver = AugmentedLagrangian::new(AugmentedLagrangianOptions::default())
        .with_inner_optimizer(scioxide_optimize::Bfgs::new(
            scioxide_optimize::BfgsOptions::default()
                .with_max_iterations(scioxide_core::Budget::try_from(2)?),
        ));
    let mut workspace = solver.workspace(&problem)?;
    let mut result = scioxide_optimize::OptimizeResult::empty();
    solver.minimize_into(
        &problem,
        &[0.1],
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    let restorations = workspace
        .last_attempt()
        .map_or(0, |attempt| attempt.counts().own().rejected());
    println!(
        "Restoration: x={:?}, termination={}, feasibility={:?}, restoration solves={}",
        result.x(),
        result.termination(),
        result.primal_feasibility(),
        restorations
    );
    assert!(result.is_success());
    assert!(restorations > 0);
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-optimize Constrained Optimization");
    println!("==========================================");
    println!();

    part1_penalty()?;
    part2_augmented_lagrangian()?;
    part3_sqp()?;
    part4_interior_point()?;
    part5_bounds()?;
    part6_restoration()?;

    println!("All examples completed successfully.");
    Ok(())
}
