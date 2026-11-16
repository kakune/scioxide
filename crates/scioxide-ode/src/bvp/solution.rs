//! Piecewise cubic solutions of boundary value problems.
use super::*;

/// Stores the solution of a [`BvpPlan`] solve: a C¹ piecewise cubic, the parameters and
/// the interval residuals.
///
/// On each mesh interval the cubic is the Hermite interpolant of the node values and of
/// the derivatives `f(xⱼ, yⱼ, p)`. [`Self::new`] creates an empty solution;
/// [`BvpPlan::solve_into`] fills it and grows its storage only when the final mesh is
/// larger than its capacity.
#[derive(Debug, Clone)]
pub struct BvpSolution<T: NumericScalar> {
    pub(super) mesh: Vec<T::Real>,
    pub(super) values: Vec<T>,
    pub(super) parameters: Vec<T>,
    pub(super) derivatives: Vec<T>,
    pub(super) residuals: Vec<T::Real>,
    pub(super) n: usize,
    pub(super) report: Option<BvpReport<T::Real>>,
}
impl<T: NumericScalar> Default for BvpSolution<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> BvpSolution<T> {
    /// Creates an empty solution without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mesh: Vec::new(),
            values: Vec::new(),
            parameters: Vec::new(),
            derivatives: Vec::new(),
            residuals: Vec::new(),
            n: 0,
            report: None,
        }
    }
    /// Returns the final mesh, which is strictly increasing; empty before a solve.
    pub fn mesh(&self) -> &[T::Real] {
        &self.mesh
    }
    /// Returns the state values at the mesh nodes, node by node (`n` values per node).
    pub fn values(&self) -> &[T] {
        &self.values
    }
    /// Returns the computed parameters `p`.
    pub fn parameters(&self) -> &[T] {
        &self.parameters
    }
    /// Returns the derivatives `f(xⱼ, yⱼ, p)` of the cubic at the mesh nodes, node by
    /// node.
    pub fn derivatives(&self) -> &[T] {
        &self.derivatives
    }
    /// Returns the scaled residual of each mesh interval, as defined in
    /// [`BvpOptions::with_tolerances`].
    pub fn residuals(&self) -> &[T::Real] {
        &self.residuals
    }
    /// Returns the report of the solve that wrote this solution; `None` before a solve.
    pub const fn report(&self) -> Option<BvpReport<T::Real>> {
        self.report
    }
    pub(super) fn max_residual(&self) -> T::Real {
        self.residuals
            .iter()
            .copied()
            .fold(T::Real::ZERO, RealScalar::max)
    }
    /// Evaluates the cubic at `x` into `output`.
    ///
    /// `x` may be any point of the closed mesh span; there is no extrapolation. At a
    /// mesh node the stored value is returned exactly. The cost is `O(log m + n)` for
    /// `m` nodes and `n` states, without allocation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if no solve has written this solution, if `x` is not finite or
    ///   lies outside the mesh span, or if `output` does not have length `n`. The
    ///   output is unchanged.
    /// - `NonFinite` (origin `Intermediate`) if the interpolated value overflows.
    #[inline]
    pub fn evaluate_into(&self, x: T::Real, output: &mut [T]) -> Result<(), SciOxideError> {
        self.evaluate(x, output, None)
    }

    /// Evaluates the cubic at `x` into `output` and its derivative into `derivative`.
    ///
    /// At a mesh node the stored value and derivative `f(xⱼ, yⱼ, p)` are returned
    /// exactly. The cost is that of [`Self::evaluate_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::evaluate_into`], and `InvalidInput` if `derivative` does
    /// not have length `n`. The outputs are unchanged.
    pub fn evaluate_with_derivative_into(
        &self,
        x: T::Real,
        output: &mut [T],
        derivative: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, output, Some(derivative))
    }

    fn evaluate(
        &self,
        x: T::Real,
        output: &mut [T],
        mut derivative: Option<&mut [T]>,
    ) -> Result<(), SciOxideError> {
        if self.mesh.len() < 2 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "solution has not been published",
            ));
        }
        if !x.is_finite()
            || x < self.mesh[0]
            || x > self.mesh[self.mesh.len() - 1]
            || output.len() != self.n
            || derivative.as_ref().is_some_and(|d| d.len() != self.n)
        {
            return Err(invalid(
                if output.len() != self.n {
                    scioxide_core::argument!("output")
                } else if derivative.as_ref().is_some_and(|d| d.len() != self.n) {
                    scioxide_core::argument!("derivative")
                } else {
                    scioxide_core::argument!("x")
                },
                "dense evaluation coordinate or output shape invalid",
            ));
        }
        let node = self.mesh.partition_point(|a| *a <= x).saturating_sub(1);
        if self.mesh[node] == x {
            output.copy_from_slice(&self.values[node * self.n..(node + 1) * self.n]);
            if let Some(d) = derivative {
                d.copy_from_slice(&self.derivatives[node * self.n..(node + 1) * self.n]);
            }
            return Ok(());
        }
        let i = node.min(self.mesh.len() - 2);
        hermite(
            self.mesh[i + 1] - self.mesh[i],
            (x - self.mesh[i]) / (self.mesh[i + 1] - self.mesh[i]),
            &self.values[i * self.n..(i + 1) * self.n],
            &self.values[(i + 1) * self.n..(i + 2) * self.n],
            &self.derivatives[i * self.n..(i + 1) * self.n],
            &self.derivatives[(i + 1) * self.n..(i + 2) * self.n],
            output,
            derivative.as_deref_mut(),
        );
        finite(output, NonFiniteOrigin::Intermediate)?;
        if let Some(derivative) = derivative {
            finite(derivative, NonFiniteOrigin::Intermediate)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn nodal_queries_preserve_stored_values_without_unused_overflow() {
        let mut solution = BvpSolution::<f64>::new();
        solution.n = 1;
        solution.mesh = alloc::vec![0., 1.];
        solution.values = alloc::vec![f64::MAX, -f64::MAX];
        solution.derivatives = alloc::vec![-0., 1.];
        for node in 0..2 {
            let (mut value, mut derivative) = ([0.], [0.]);
            solution
                .evaluate_with_derivative_into(solution.mesh[node], &mut value, &mut derivative)
                .unwrap();
            assert_eq!(value[0].to_bits(), solution.values[node].to_bits());
            assert_eq!(
                derivative[0].to_bits(),
                solution.derivatives[node].to_bits()
            );
        }
    }
}
