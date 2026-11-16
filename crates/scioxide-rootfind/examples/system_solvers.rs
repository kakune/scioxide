//! System root-finding demonstration for scioxide-rootfind.
//!
//! Demonstrates solving F(x) = 0 where F: ℝⁿ → ℝⁿ:
//!
//! 1. **NewtonSystem (2D)** — Intersection of two parabolas
//! 2. **Broyden (2D)** — Same system, comparing iterations to Newton
//! 3. **Trust-region (2D)** — Global convergence from a distant start
//! 4. **System with analytic Jacobian** — Trigonometric system with J provided
//! 5. **3D system with line search** — Sphere + hyperboloid + plane intersection
//!
//! Run: `cargo run --example system_solvers -p scioxide-rootfind`

use scioxide_core::{ResidualOperator, SciOxideError};
use scioxide_rootfind::{
    BacktrackingLineSearch, Broyden, DenseJacobian, NewtonSystem, SystemRootSolver,
    TrustRegionSystem,
};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Helper: F(x,y) = (x²+y−1, x+y²−1), roots at (1,0) and (0,1)
// ---------------------------------------------------------------------------

struct TwoParabolas;

impl ResidualOperator<f64> for TwoParabolas {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        2
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let u = x[0];
        let v = x[1];
        out[0] = u * u + v - 1.0;
        out[1] = u + v * v - 1.0;
        Ok(())
    }
}
impl DenseJacobian<f64> for TwoParabolas {}

// ---------------------------------------------------------------------------
// Helper: F(x,y) = (sin(x)+y−1, x+cos(y)−1) with analytic Jacobian
// ---------------------------------------------------------------------------

struct TrigSystem;

impl ResidualOperator<f64> for TrigSystem {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        2
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let u = x[0];
        let v = x[1];
        out[0] = u.sin() + v - 1.0;
        out[1] = u + v.cos() - 1.0;
        Ok(())
    }
}
impl DenseJacobian<f64> for TrigSystem {
    fn jacobian_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<bool, SciOxideError> {
        let u = x[0];
        let v = x[1];
        // J = [[cos(u), 1], [1, -sin(v)]]
        out.copy_from_slice(&[u.cos(), 1.0, 1.0, -v.sin()]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Helper: 3D system: x²+y²+z²−3=0, xy−1=0, x+y+z−3=0; root at (1,1,1)
// ---------------------------------------------------------------------------

struct System3D;

impl ResidualOperator<f64> for System3D {
    fn input_dimension(&self) -> usize {
        3
    }
    fn output_dimension(&self) -> usize {
        3
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let u = x[0];
        let v = x[1];
        let w = x[2];
        out[0] = u * u + v * v + w * w - 3.0;
        out[1] = u * v - 1.0;
        out[2] = u + v + w - 3.0;
        Ok(())
    }
}
impl DenseJacobian<f64> for System3D {}

// ---------------------------------------------------------------------------
// Part 1: NewtonSystem (2D)
// ---------------------------------------------------------------------------

fn part1_newton_system() -> Result<(), SciOxideError> {
    println!("=== Part 1: NewtonSystem (2D) ===");
    println!();

    let f = TwoParabolas;

    let res = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::default().with_residual_scale(1.0),
    )?
    .solve(&f, &[0.5, 0.5])?;

    println!("  F(x,y) = (x²+y−1, x+y²−1), x₀ = (0.5, 0.5)");
    println!("  root = ({:.10}, {:.10})", res.point()[0], res.point()[1]);
    println!(
        "  |F(root)| = {:?}  iters = {}  converged = {}",
        res.residual_norm(),
        res.iterations(),
        res.is_success()
    );

    // System has roots at (1,0), (0,1), and (φ−1,φ−1) ≈ (0.618,0.618)
    assert!(res.is_success(), "NewtonSystem should converge");
    assert!(
        res.residual_norm().is_some_and(|norm| norm < TOL),
        "NewtonSystem residual should be small"
    );

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Broyden
// ---------------------------------------------------------------------------

fn part2_broyden() -> Result<(), SciOxideError> {
    println!("=== Part 2: Broyden (2D) ===");
    println!();

    let f = TwoParabolas;

    let res = Broyden::new(
        scioxide_rootfind::BroydenUpdate::First,
        scioxide_rootfind::BroydenOptions::default().with_residual_scale(1.0),
    )?
    .solve(&f, &[0.5, 0.5])?;

    println!("  F(x,y) = (x²+y−1, x+y²−1), x₀ = (0.5, 0.5)");
    println!("  root = ({:.10}, {:.10})", res.point()[0], res.point()[1]);
    println!(
        "  |F(root)| = {:?}  iters = {}  converged = {}",
        res.residual_norm(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "Broyden should converge");
    assert!(
        res.residual_norm().is_some_and(|norm| norm < TOL),
        "Broyden residual should be small"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Trust-region from distant start
// ---------------------------------------------------------------------------

fn part3_trust_region() -> Result<(), SciOxideError> {
    println!("=== Part 3: Trust-region (global convergence) ===");
    println!();

    let f = TwoParabolas;

    let res = TrustRegionSystem::new(
        scioxide_rootfind::TrustRegionSystemOptions::default().with_residual_scale(1.0),
    )?
    .solve(&f, &[5.0, 5.0])?;

    println!("  F(x,y) = (x²+y−1, x+y²−1), x₀ = (5.0, 5.0) [distant start]");
    println!("  root = ({:.10}, {:.10})", res.point()[0], res.point()[1]);
    println!(
        "  |F(root)| = {:?}  iters = {}  converged = {}",
        res.residual_norm(),
        res.iterations(),
        res.is_success()
    );

    assert!(
        res.is_success(),
        "TrustRegion should converge even from distant start"
    );
    assert!(
        res.residual_norm().is_some_and(|norm| norm < TOL),
        "TrustRegion residual should be small"
    );

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: System with analytic Jacobian
// ---------------------------------------------------------------------------

fn part4_analytic_jacobian() -> Result<(), SciOxideError> {
    println!("=== Part 4: System with analytic Jacobian ===");
    println!();

    let f = TrigSystem;

    let res = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::default().with_residual_scale(1.0),
    )?
    .solve(&f, &[0.5, 0.5])?;

    println!("  F(x,y) = (sin(x)+y−1, x+cos(y)−1) with analytic Jacobian");
    println!("  root = ({:.10}, {:.10})", res.point()[0], res.point()[1]);
    println!(
        "  |F(root)| = {:?}  iters = {}  converged = {}",
        res.residual_norm(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "Newton with Jacobian should converge");
    assert!(
        res.residual_norm().is_some_and(|norm| norm < TOL),
        "Residual should be small"
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: 3D system with line search
// ---------------------------------------------------------------------------

fn part5_3d_with_line_search() -> Result<(), SciOxideError> {
    println!("=== Part 5: 3D system with line search ===");
    println!();

    let f = System3D;

    let res = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::default()
            .with_line_search(BacktrackingLineSearch::<f64>::default()),
    )?
    .solve(&f, &[1.5, 0.8, 0.8])?;

    println!("  F(x,y,z) = (x²+y²+z²−3, xy−1, x+y+z−3)");
    println!("  x₀ = (1.5, 0.8, 0.8) with BacktrackingLineSearch");
    println!(
        "  root = ({:.10}, {:.10}, {:.10})",
        res.point()[0],
        res.point()[1],
        res.point()[2]
    );
    println!(
        "  |F(root)| = {:?}  iters = {}  converged = {}",
        res.residual_norm(),
        res.iterations(),
        res.is_success()
    );

    // Root should be near (1, 1, 1)
    assert!(res.is_success(), "Newton+line search should converge");
    assert!(
        res.residual_norm().is_some_and(|norm| norm < 1e-6),
        "Residual should be small"
    );

    println!();
    println!("  Part 5 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-rootfind System Solvers");
    println!("===============================");
    println!();

    part1_newton_system()?;
    part2_broyden()?;
    part3_trust_region()?;
    part4_analytic_jacobian()?;
    part5_3d_with_line_search()?;

    println!("All examples completed successfully.");
    Ok(())
}
