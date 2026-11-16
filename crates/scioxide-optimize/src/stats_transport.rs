//! Discrete optimal transport as a sparse linear program.
use crate::{
    LinearConstraintSense, LinearConstraints, LpProblem, OptimizeTermination,
    ProgrammingCertificate, ProgrammingResult, SimplexSolver,
};
use core::mem::size_of;
use scioxide_core::{Bounds, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_stats::{StatsTermination, TransportCandidate, TransportProblem, TransportSolver};

use crate::validation::invalid;
/// Solves discrete optimal-transport problems with [`SimplexSolver`], as a
/// [`TransportSolver`](scioxide_stats::TransportSolver) for `scioxide-stats`.
///
/// For source weights `a ∈ ℝᵐ`, target weights `b ∈ ℝⁿ`, both summing to 1,
/// and costs `cᵢⱼ ≥ 0`, it solves the linear program
///
/// ```text
/// min Σᵢⱼ cᵢⱼ xᵢⱼ   subject to   Σⱼ xᵢⱼ = aᵢ,  Σᵢ xᵢⱼ = bⱼ,  xᵢⱼ ≥ 0,
/// ```
///
/// with the revised simplex method on a sparse constraint matrix, and returns
/// the coupling `x` and the dual variables of the marginal constraints. One
/// target constraint is dropped because it follows from the others. Each
/// solve allocates its linear program and runs on the calling thread.
///
/// # Errors
///
/// - `InvalidInput` if a support is empty, the output shapes differ from the
///   problem, a weight or cost is negative or not finite, the cost scale is not
///   positive and finite, or the weights do not each sum to 1 within the
///   feasibility tolerance of the solver.
/// - `ResourceLimit` if the number of pairs `m n` exceeds `max_pairs`, or the
///   size of the linear program exceeds `max_model_bytes`.
#[derive(Debug, Clone)]
pub struct SimplexTransportSolver<S: RealScalar> {
    solver: SimplexSolver<S>,
    max_pairs: usize,
    max_model_bytes: usize,
}
impl<S: RealScalar> SimplexTransportSolver<S> {
    /// Creates the adapter. `max_pairs` limits the number of source–target
    /// pairs and `max_model_bytes` the size of the assembled linear program, as
    /// computed by [`Self::required_model_bytes`]. Limits on the simplex
    /// factors are set on `solver`.
    pub const fn new(solver: SimplexSolver<S>, max_pairs: usize, max_model_bytes: usize) -> Self {
        Self {
            solver,
            max_pairs,
            max_model_bytes,
        }
    }
    /// Returns the bytes of the linear program assembled for `rows` sources and
    /// `cols` targets, excluding the storage of the simplex solver and of the
    /// allocator.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `cols` is zero.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_model_bytes(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
        let (_, m, nnz) = shape(rows, cols)?;
        let pairs = rows.checked_mul(cols).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("transport_pairs"))
        })?;
        [
            pairs
                .checked_mul(size_of::<S>() + size_of::<(Option<S>, Option<S>)>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("transport_model"))
                })?,
            m.checked_mul(size_of::<S>() + size_of::<LinearConstraintSense>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("transport_rows"))
                })?,
            nnz.checked_mul(size_of::<S>() + size_of::<usize>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("transport_csr"))
                })?,
            m.checked_add(1)
                .and_then(|v| v.checked_mul(size_of::<usize>()))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("transport_csr_pointers"))
                })?,
        ]
        .into_iter()
        .try_fold(0_usize, |sum, bytes| {
            sum.checked_add(bytes).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("transport_model_bytes"))
            })
        })
    }
}
fn shape(rows: usize, cols: usize) -> Result<(usize, usize, usize), SciOxideError> {
    if rows == 0 || cols == 0 {
        return Err(invalid(
            scioxide_core::argument!("p"),
            "transport supports must be nonempty",
        ));
    }
    let pairs = rows
        .checked_mul(cols)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("transport_pairs")))?;
    let m = rows
        .checked_add(cols - 1)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("transport_rows")))?;
    let nnz = pairs
        .checked_add(rows.checked_mul(cols - 1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("transport_nonzeros"))
        })?)
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("transport_nonzeros"))
        })?;
    Ok((pairs, m, nnz))
}
impl<S: RealScalar + NumericScalar<Real = S>> TransportSolver<S> for SimplexTransportSolver<S> {
    fn solve(
        &mut self,
        p: TransportProblem<'_, S>,
        candidate: TransportCandidate<'_, S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        let (n, m, nnz) = shape(p.source.len(), p.target.len())?;
        if p.costs.len() != n
            || candidate.coupling.len() != n
            || candidate.source_dual.len() != p.source.len()
            || candidate.target_dual.len() != p.target.len()
        {
            return Err(invalid(
                scioxide_core::argument!("candidate"),
                "transport problem/output shape",
            ));
        }
        if n > self.max_pairs {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("transport_pairs"),
                n,
                self.max_pairs,
            ));
        }
        let bytes = Self::required_model_bytes(p.source.len(), p.target.len())?;
        if bytes > self.max_model_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("transport_model_bytes"),
                bytes,
                self.max_model_bytes,
            ));
        }
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        if !p.cost_scale.is_finite()
            || p.cost_scale <= zero
            || p.source
                .iter()
                .chain(p.target)
                .chain(p.costs)
                .any(|v| !v.is_finite() || *v < zero)
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                "transport probabilities/costs/scale must be finite and nonnegative",
            ));
        }
        let total = |values: &[S]| -> Result<S, SciOxideError> {
            let mut sum = scioxide_core::ProductSum::new();
            for &v in values {
                sum.add_product(v, one)?;
            }
            sum.finish()
        };
        let a = total(p.source)?;
        let b = total(p.target)?;
        let tolerance = self.solver.tolerances().feasibility().threshold(one)?;
        if NumericScalar::abs(a - one) > tolerance
            || NumericScalar::abs(b - one) > tolerance
            || NumericScalar::abs(a - b) > tolerance
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                "transport marginals must have equal unit mass",
            ));
        }
        candidate.coupling.fill(zero);
        candidate.source_dual.fill(zero);
        candidate.target_dual.fill(zero);
        if let Some(stop) = context.stop_reason()? {
            return Ok(StatsTermination::Stopped(stop));
        }
        use scioxide_core::backend::{try_copy_slice, try_filled, try_with_capacity};
        let cols = p.target.len();
        let rows = p.source.len();
        let mut ptr = try_with_capacity(m + 1, scioxide_core::quantity!("transport_csr_pointers"))?;
        let mut ind = try_with_capacity(nnz, scioxide_core::quantity!("transport_csr_indices"))?;
        let mut values = try_with_capacity(nnz, scioxide_core::quantity!("transport_csr_values"))?;
        ptr.push(0);
        for i in 0..rows {
            ind.extend((0..cols).map(|j| i * cols + j));
            values.resize(ind.len(), one);
            ptr.push(ind.len());
        }
        for j in 0..cols - 1 {
            ind.extend((0..rows).map(|i| i * cols + j));
            values.resize(ind.len(), one);
            ptr.push(ind.len());
        }
        let matrix = scioxide_sparse::CsrMatrix::from_raw(m, n, ptr, ind, values)?;
        let mut rhs = try_with_capacity(m, scioxide_core::quantity!("transport_marginals"))?;
        rhs.extend_from_slice(p.source);
        rhs.extend_from_slice(&p.target[..cols - 1]);
        let constraints = LinearConstraints::new(
            matrix,
            rhs,
            try_filled(
                m,
                LinearConstraintSense::Equal,
                scioxide_core::quantity!("transport_senses"),
            )?,
        )?;
        let lp = LpProblem::new(try_copy_slice(
            p.costs,
            scioxide_core::quantity!("transport_costs"),
        )?)?
        .with_constraints(constraints)?
        .with_bounds(Bounds::new(try_filled(
            n,
            (Some(zero), None),
            scioxide_core::quantity!("transport_bounds"),
        )?)?)?;
        let Some(iterations) =
            crate::stats_adapters::iteration_limit(context, self.solver.max_iterations(&lp))?
        else {
            return Ok(StatsTermination::BudgetExhausted);
        };
        let solver = self.solver.clone().with_max_iterations(iterations);
        let mut workspace = solver.workspace(&lp)?;
        let mut result = ProgrammingResult::empty();
        solver.solve_into(&lp, &mut result, &mut workspace, &mut *context)?;
        let stop = match result.termination() {
            OptimizeTermination::Converged => StatsTermination::Converged,
            OptimizeTermination::Cancelled => {
                StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
            }
            OptimizeTermination::TimeLimit => {
                StatsTermination::Stopped(scioxide_core::ExecutionStopReason::DeadlineReached)
            }
            OptimizeTermination::IterationLimit
            | OptimizeTermination::EvaluationLimit
            | OptimizeTermination::MemoryLimit => StatsTermination::BudgetExhausted,
            _ => {
                return Err(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::NestedSolver,
                    scioxide_core::ConvergenceDiagnostics::new().with_counts(result.counts()),
                    "transport LP did not produce an admissible result",
                ));
            }
        };
        // A stopped solver can have no evaluated point. Overwrite every output
        // so stale candidates never escape through the narrow provider contract.
        if result.x().len() == n {
            candidate.coupling.copy_from_slice(result.x());
        }
        if let Some(ProgrammingCertificate::Optimality { dual, .. }) = result.certificate() {
            if dual.rows.len() != m {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "transport LP dual shape",
                ));
            }
            // LP uses c + A^T y - lower = 0; transport uses u+v <= c.
            candidate
                .source_dual
                .iter_mut()
                .zip(&dual.rows[..rows])
                .for_each(|(u, &y)| *u = -y);
            candidate.target_dual[..cols - 1]
                .iter_mut()
                .zip(&dual.rows[rows..])
                .for_each(|(v, &y)| *v = -y);
        } else if stop == StatsTermination::Converged {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "converged transport LP lacks original-coordinate dual evidence",
            ));
        }
        Ok(stop)
    }
}
