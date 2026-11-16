//! Sealed supertraits of the number contract.
//!
//! The traits are `pub` inside a private module: other crates can neither name nor
//! implement them, and their hooks take a [`Token`] only this crate can build.

use scioxide_core::RealScalar;

/// Proof that a hook is called from inside the crate.
#[derive(Debug, Clone, Copy)]
pub struct Token(pub(crate) ());

/// Names the IEEE scalar of a number type without a type parameter.
pub trait Value {
    /// The IEEE scalar.
    type Elem: RealScalar;
}

/// Per-type hooks of [`Real::map_scalar`](crate::Real::map_scalar) and
/// [`Real::gather`](crate::Real::gather).
pub trait Map<S: RealScalar>: Sized {
    /// Returns `[f(x), f′(x), f″(x)]` evaluated in `Self` arithmetic from the scalar
    /// rule `[f, f′, f″]`; components above total order two are NaN.
    fn map_all<F: FnMut(S) -> [S; 3]>(token: Token, x: Self, rule: &mut F) -> [Self; 3];

    /// Builds the value whose lane `k` is lane `k` of `project(&table[index(k)])`.
    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        token: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self;
}

/// Lane access of a mask.
pub trait MaskLanes {
    /// Returns the number of lanes that hold.
    fn count(self) -> u64;
    /// Returns the number of lanes among the first `lanes` that hold.
    fn count_first(self, lanes: usize) -> u64;
}

/// Per-type rules of [`dot`](crate::dot) and [`solve_into`](crate::solve_into).
pub trait Linear<S: RealScalar>: Sized {
    /// Returns `Σₖ aₖ·bₖ` over `pairs`, in order; one statement on a tape.
    fn dot<I: Iterator<Item = (Self, Self)> + Clone>(token: Token, pairs: I) -> Self
    where
        Self: crate::Real<S>,
    {
        let _ = token;
        // From −0 so that a sum of −0 products keeps its sign; an empty sum is +0.
        if pairs.clone().next().is_none() {
            return Self::zero();
        }
        pairs.fold(Self::constant(S::NEG_ZERO), |sum, (a, b)| sum + a * b)
    }

    /// Solves `A·X = B` (or `Aᵀ·X = B`) in place in `rhs` with the factorization
    /// of the base values of `matrix`, applying it to every derivative component.
    fn solve_factored(
        token: Token,
        factor: &crate::structured::Factor<S>,
        matrix: &[Self],
        rhs: &mut [Self],
        transpose: bool,
    ) -> Result<(), scioxide_core::SciOxideError>;

    /// Solves `A·X = B` for an `order × order` matrix and `columns` right-hand
    /// sides into `out`.
    fn solve(
        token: Token,
        matrix: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        order: usize,
    ) -> Result<(), scioxide_core::SciOxideError>
    where
        Self: crate::Real<S>,
    {
        let factor = crate::structured::Factor::new(matrix, order)?;
        // Solve in scratch so that `out` is unchanged when a later step fails.
        let mut x =
            scioxide_core::backend::try_copy_slice(rhs, scioxide_core::quantity!("solve_scratch"))?;
        Self::solve_factored(token, &factor, matrix, &mut x, false)?;
        out.copy_from_slice(&x);
        Ok(())
    }
}

/// Per-type rule of [`checkpointed_loop`](crate::checkpointed_loop): every type but
/// tape variables runs the plain loop.
pub trait Loop<S: RealScalar>: Sized {
    /// Runs `steps` steps of `body` on `state`.
    fn run_loop<B: crate::LoopBody<S> + Send + Sync + 'static>(
        token: Token,
        body: &alloc::sync::Arc<B>,
        steps: usize,
        state: &mut [Self],
        parameters: &[Self],
        schedule: &crate::CheckpointSchedule,
    ) -> Result<(), scioxide_core::SciOxideError>
    where
        Self: crate::Real<S>,
    {
        let _ = (token, schedule);
        if steps == 0 {
            return Ok(());
        }
        let mut staged = scioxide_core::backend::try_copy_slice(
            state,
            scioxide_core::quantity!("loop_scratch"),
        )?;
        (0..steps).try_for_each(|index| body.step(index, &mut staged, parameters))?;
        state.copy_from_slice(&staged);
        Ok(())
    }
}
