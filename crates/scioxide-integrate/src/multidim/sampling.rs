//! Batch evaluation of multidimensional integrands in point order.
use scioxide_core::{ExecutionPolicy, RealScalar, SciOxideError, SolveContext};
pub(crate) trait BatchSampler<S: RealScalar> {
    fn sample(
        &mut self,
        points: &[S],
        values: &mut [S],
        dimension: usize,
        components: usize,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Evaluates a whole-value integrand at a batch of points under an execution policy.
pub(crate) struct ValueSampler<'a, F: ?Sized> {
    pub(crate) function: &'a F,
    pub(crate) policy: ExecutionPolicy,
}
impl<S, V, F> BatchSampler<S> for ValueSampler<'_, F>
where
    S: RealScalar,
    V: scioxide_core::AffineValue<Scale = S>,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
{
    fn sample(
        &mut self,
        points: &[S],
        values: &mut [S],
        dimension: usize,
        components: usize,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let count = points.len().checked_div(dimension).unwrap_or(1);
        crate::evaluation::sample_nd(
            self.function,
            points,
            dimension,
            count,
            components,
            values,
            self.policy,
            cx,
        )
    }
}
