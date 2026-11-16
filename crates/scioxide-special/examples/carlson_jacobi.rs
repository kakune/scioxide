//! Parameter conventions, scaled results and scoped evaluation under a parent solve.
use scioxide_core::{
    ExecutionContext, ProblemToken, SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_special::{
    EllipticOptions, carlson_rd_with_options, carlson_rf, elliptic_k, jacobi_amplitude,
    jacobi_cn_scoped, jacobi_dn_scoped, jacobi_sn, jacobi_sn_scoped,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let m = 0.5_f64;
    println!(
        "K(m)={} = RF(0,1-m,1)={}",
        elliptic_k(m)?,
        carlson_rf(0.0, 1.0 - m, 1.0)?
    );
    println!(
        "sn(0.7|m)={}, amplitude={}",
        jacobi_sn(0.7, m)?,
        jacobi_amplitude(0.7, m)?
    );
    let scaled = carlson_rd_with_options(1e-300_f64, 2e-300, 3e-300, EllipticOptions::default())?;
    println!(
        "RD scalar status={:?}, scaled={:?}",
        scaled.scalar_representation(),
        scaled.scaled()
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(497),
        scioxide_core::algorithm_id!("elliptic", "example", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let options = EllipticOptions::default();

    // Each scoped call runs in a child epoch; the parent sees its work as nested counts.
    let sn = jacobi_sn_scoped(0.7, m, options, &mut context)?.try_value()?;
    let cn = jacobi_cn_scoped(0.7, m, options, &mut context)?.try_value()?;
    let dn = jacobi_dn_scoped(0.7, m, options, &mut context)?.try_value()?;
    println!(
        "sn={sn}, cn={cn}, dn={dn}, sn²+cn²={}, nested iterations={}",
        sn * sn + cn * cn,
        context.counts().nested().iterations()
    );
    Ok(())
}
