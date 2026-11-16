//! Continuation problem: the residual map and its optional solvers.
use super::*;

/// Describes the equation `F(x, λ) = 0` that continuation follows.
///
/// `F: ℝⁿ⁺¹ → ℝⁿ` maps the state `x ∈ ℝⁿ` and the scalar parameter `λ` to `n`
/// residuals. The parameter is the last of the `n + 1` inputs, which this crate
/// calls the augmented point `y = (x, λ)`. The map is given as a
/// [`JacobianOperator`], so it needs residuals and Jacobian–vector products but
/// not an assembled Jacobian; dense, sparse and matrix-free providers all work.
/// [`crate::ResidualFn::with_parameter`] with [`crate::ResidualFn::with_jvp`]
/// wraps two closures as such a map.
///
/// By default each correction and tangent solve uses GMRES on the bordered
/// `(n + 1) × (n + 1)` system. [`Self::with_linear_solver`] replaces it, and
/// [`Self::with_diagnostic`] adds a branch-point diagnostic.
#[derive(Clone, Copy)]
pub struct ContinuationProblem<'a, R: RealScalar> {
    pub(crate) jacobian: &'a dyn JacobianOperator<R>,
    pub(crate) linear: Option<&'a dyn NonlinearLinearSolver<R>>,
    pub(crate) diagnostic: Option<&'a dyn ContinuationDiagnosticProvider<R>>,
}
impl<'a, R: RealScalar> ContinuationProblem<'a, R> {
    /// Creates a continuation problem from a residual map with directional derivatives.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `jacobian` has no outputs or does not have exactly one
    ///   more input than outputs.
    /// - `SizeOverflow` if `n + 1` overflows `usize`.
    /// - Any error returned by the provider's workspace-length queries.
    pub fn new(jacobian: &'a dyn JacobianOperator<R>) -> Result<Self, SciOxideError> {
        if jacobian.output_dimension() == 0
            || jacobian.input_dimension()
                != checked_size(
                    jacobian.output_dimension(),
                    1,
                    scioxide_core::quantity!("augmented_coordinates"),
                )?
        {
            return Err(invalid(
                scioxide_core::argument!("jacobian"),
                InvalidInputReason::Inconsistent,
                "require n residuals and n+1 augmented coordinates",
            ));
        }
        jacobian.residual_workspace_len()?;
        jacobian.jvp_workspace_len()?;
        Ok(Self {
            jacobian,
            linear: None,
            diagnostic: None,
        })
    }
    /// Uses `solver` for the bordered linear systems of order `n + 1` in the
    /// corrector and the tangent computation, instead of the built-in GMRES.
    pub fn with_linear_solver(mut self, solver: &'a dyn NonlinearLinearSolver<R>) -> Self {
        self.linear = Some(solver);
        self
    }
    /// Evaluates `provider` at each accepted point to detect branch points.
    ///
    /// This is the alternative to the dense diagnostic of
    /// [`ContinuationOptions::with_dense_diagnostics`]; [`Continuation`] rejects
    /// a problem that requests both.
    pub fn with_diagnostic(mut self, provider: &'a dyn ContinuationDiagnosticProvider<R>) -> Self {
        self.diagnostic = Some(provider);
        self
    }
    /// Returns the number of state coordinates `n`, excluding the parameter.
    pub fn dimension(&self) -> usize {
        self.jacobian.output_dimension()
    }
    pub(crate) fn corrector<'b>(&'b self, a: &'b Augmented<'_, R>) -> NewtonKrylovProblem<'b, R> {
        let p = NewtonKrylovProblem::with_jacobian(a);
        self.linear.map_or(p, |s| p.with_linear_solver(s))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProviderShape {
    pub(crate) dimension: usize,
    pub(crate) residual: usize,
    pub(crate) jvp: usize,
    pub(crate) linear: bool,
    pub(crate) diagnostic: bool,
}
pub(crate) fn provider_shape<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
) -> Result<ProviderShape, SciOxideError> {
    Ok(ProviderShape {
        dimension: p.dimension(),
        residual: p.jacobian.residual_workspace_len()?,
        jvp: p.jacobian.jvp_workspace_len()?,
        linear: p.linear.is_some(),
        diagnostic: p.diagnostic.is_some(),
    })
}
