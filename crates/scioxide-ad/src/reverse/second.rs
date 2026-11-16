//! Second-order products on tapes of duals: forward over reverse.

use alloc::vec::Vec;

use scioxide_core::{
    ColoringSchedule, ExecutionContext, RealScalar, SciOxideError, StructuralPattern,
};

use super::Var;
use super::linearization::Linearization;
use super::operator::HessianOperator;
use super::tape::{Taken, Tape};
use crate::Dual;

impl<S: RealScalar> Tape<Dual<S>> {
    /// Returns the structural pattern of the Hessian of `function` at `point`
    /// (inputs × inputs), by nonlinear interaction domains (Walther 2008).
    ///
    /// The function is the closure of [`Tape::hvp_into`] and is recorded once with
    /// zero directions, so the pattern, its coloring and
    /// [`Tape::sparse_hessian_into`] use one tape and one closure. The pattern
    /// equals [`Linearization::hessian_pattern`](crate::Linearization::hessian_pattern)
    /// of the same function recorded on a `Tape<S>`, and it is valid for the points
    /// that take the same branches. The tape is left as after
    /// [`Tape::linearize`].
    ///
    /// # Errors
    ///
    /// - The recording errors of [`Tape::gradient_into`].
    /// - `AllocationFailure` if the seeds or the pattern cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_ad::{Dual, Real, Tape};
    /// use scioxide_core::ExecutionContext;
    /// // f = x₀² + x₁·x₂: the Hessian couples x₁ and x₂ only, besides the diagonal of x₀.
    /// let mut tape = Tape::<Dual<f64>>::default();
    /// let cx = ExecutionContext::sequential();
    /// let pattern = tape.hessian_pattern(|x| Ok(x[0] * x[0] + x[1] * x[2]), &[1.0, 2.0, 3.0], cx)?;
    /// assert_eq!(pattern.view().row_ptr(), [0, 1, 2, 3]);
    /// assert_eq!(pattern.view().col_indices(), [0, 2, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn hessian_pattern<F>(
        &mut self,
        function: F,
        point: &[S],
        context: ExecutionContext<'_>,
    ) -> Result<StructuralPattern, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
    {
        context.check()?;
        let mut seeds = self.take_buffer(Taken::Seeds);
        let outcome = (|| {
            self.grow_taken(
                &mut seeds,
                Taken::Seeds,
                point.len(),
                scioxide_core::quantity!("tape_seeds"),
            )?;
            seeds.clear();
            seeds.extend(point.iter().map(|&x| Dual::new(x, S::ZERO)));
            self.record(
                &seeds,
                &|inputs: &[Var<'_, Dual<S>>], outputs: &mut Vec<Var<'_, Dual<S>>>| {
                    outputs.push(function(inputs)?);
                    Ok(())
                },
            )
        })();
        self.restore(seeds, Taken::Seeds);
        outcome?;
        Linearization::new(self).hessian_pattern()
    }

    /// Records `function` with seeds `Dual(xᵢ, direction(i))`, sweeps it in reverse
    /// and hands the value and the input adjoints `(∇f, H·v)` to `publish`.
    pub(crate) fn hvp_seeded<F>(
        &mut self,
        function: &F,
        point: &[S],
        direction: impl Fn(usize) -> S,
        publish: impl FnOnce(Dual<S>, &[Dual<S>]),
        context: ExecutionContext<'_>,
    ) -> Result<Dual<S>, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
    {
        context.check()?;
        let mut seeds = self.take_buffer(Taken::Seeds);
        let outcome = (|| {
            self.grow_taken(
                &mut seeds,
                Taken::Seeds,
                point.len(),
                scioxide_core::quantity!("tape_seeds"),
            )?;
            seeds.clear();
            seeds.extend(
                point
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| Dual::new(x, direction(k))),
            );
            self.record(
                &seeds,
                &|inputs: &[Var<'_, Dual<S>>], outputs: &mut Vec<Var<'_, Dual<S>>>| {
                    outputs.push(function(inputs)?);
                    Ok(())
                },
            )
        })();
        self.restore(seeds, Taken::Seeds);
        outcome?;
        let value = self
            .output_values
            .first()
            .copied()
            .unwrap_or_else(|| Dual::new(S::ZERO, S::ZERO));
        self.clean_work();
        if let Some(&slot) = self.output_slots.first()
            && let Some(target) = self.work.get_mut(slot as usize)
        {
            *target = Dual::new(S::ONE, S::ZERO);
        }
        self.sweep_reverse(context)?;
        let n = point.len();
        let checked = self.check_results(&self.work[1..=n]);
        if checked.is_ok() {
            publish(value, &self.work[1..=n]);
        }
        self.clear_inputs(n);
        checked.map(|()| value)
    }

    /// Computes the Hessian-vector product `H·v` of a scalar function at `point`
    /// and returns the value.
    ///
    /// The function is recorded on duals seeded `Dual(xᵢ, vᵢ)`, so one reverse
    /// sweep returns the gradient in the values and `H·v` in the derivatives;
    /// the cost is a small constant times a gradient.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `direction` or `product` has not `point.len()`
    ///   entries.
    /// - The recording and product errors of [`Tape::gradient_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_ad::{Dual, Real, Tape, TapeOptions};
    /// use scioxide_core::ExecutionContext;
    ///
    /// let mut tape = Tape::<Dual<f64>>::new(TapeOptions::new());
    /// let mut product = [0.0; 2];
    /// let cx = ExecutionContext::sequential();
    /// let value = tape.hvp_into(|x| Ok(x[0] * x[0] * x[1]), &[3.0, 2.0], &[1.0, 0.0], &mut product, cx)?;
    /// assert_eq!(value, 18.0);
    /// assert_eq!(product, [4.0, 6.0]); // H = [[2y, 2x], [2x, 0]]
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn hvp_into<F>(
        &mut self,
        function: F,
        point: &[S],
        direction: &[S],
        product: &mut [S],
        context: ExecutionContext<'_>,
    ) -> Result<S, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
    {
        crate::support::check_len(direction.len(), point.len())?;
        crate::support::check_len(product.len(), point.len())?;
        let value = self.hvp_seeded(
            &function,
            point,
            |k| direction[k],
            |_, adjoints| {
                for (target, adjoint) in product.iter_mut().zip(adjoints) {
                    *target = adjoint.derivative();
                }
            },
            context,
        )?;
        Ok(value.value())
    }

    /// Writes the entries of a sparse Hessian in the CSR order of `coloring`'s
    /// pattern.
    ///
    /// The function is recorded once per color with the summed unit directions of
    /// that color as seeds, and each entry is read directly from `H·d_c` (direct
    /// recovery; valid for a column coloring of a pattern that contains the
    /// Hessian's, such as `scioxide_sparse::derivative_coloring` of
    /// [`Linearization::hessian_pattern`](crate::Linearization::hessian_pattern)).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `coloring`, `UnsupportedConfiguration`) unless
    ///   the coloring is a derivative-column coloring
    ///   (`ColoringKind::DerivativeColumns`): a distance-1 coloring does not make
    ///   the columns of a color structurally orthogonal.
    /// - `DimensionMismatch` if the pattern is not `n × n` or `values` has not one
    ///   entry per pattern entry.
    /// - The errors of [`Tape::hvp_into`].
    pub fn sparse_hessian_into<F>(
        &mut self,
        function: F,
        point: &[S],
        coloring: &ColoringSchedule,
        values: &mut [S],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
    {
        context.check()?;
        crate::support::check_coloring(coloring)?;
        let pattern = coloring.pattern();
        let n = point.len();
        if pattern.rows() != n || pattern.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                [n, n],
                [pattern.rows(), pattern.cols()],
            ));
        }
        let entries = pattern.col_indices().len();
        crate::support::check_len(values.len(), entries)?;
        let mut stage = self.take_buffer(Taken::Stage);
        let outcome = (|| {
            self.grow_taken(
                &mut stage,
                Taken::Stage,
                entries,
                scioxide_core::quantity!("sparse_hessian"),
            )?;
            let colors = coloring.colors();
            for color in 0..coloring.color_count() {
                self.hvp_seeded(
                    &function,
                    point,
                    |k| if colors[k] == color { S::ONE } else { S::ZERO },
                    |_, adjoints| {
                        for (row, adjoint) in adjoints.iter().enumerate() {
                            for entry in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                                if colors[pattern.col_indices()[entry]] == color {
                                    stage[entry] = Dual::new(adjoint.derivative(), S::ZERO);
                                }
                            }
                        }
                    },
                    context,
                )?;
            }
            Ok(())
        })();
        if outcome.is_ok() {
            for (target, value) in values.iter_mut().zip(&stage) {
                *target = value.value();
            }
        }
        self.restore(stage, Taken::Stage);
        outcome
    }

    /// Returns `H` of `function` at `point` as a matrix-free operator.
    ///
    /// # Errors
    ///
    /// The recording errors of [`Tape::hvp_into`] at the zero direction.
    pub fn hessian_operator<'a, F>(
        &'a mut self,
        function: F,
        point: &'a [S],
        context: ExecutionContext<'a>,
    ) -> Result<HessianOperator<'a, S, F>, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
    {
        HessianOperator::new(self, function, point, context)
    }
}
