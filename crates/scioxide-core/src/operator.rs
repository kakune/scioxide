//! Matrix-free linear operators `y = A x`, their adjoints, and preconditioners `y = M⁻¹ x`.
//!
//! Iterative solvers in other SciOxide crates (Krylov methods, eigensolvers)
//! accept these traits, so `A` can be a dense matrix, a sparse matrix, a
//! factorization or a closure. The caller first asks for the workspace length,
//! then passes the input, an output slice and the workspace to each call.
//! Implementations check all lengths before they write to the output.
//!
//! The `*_scoped` methods also receive an [`OperatorContext`]. The calling
//! solver has already counted the application (see
//! [solve accounting](crate#solve-accounting)); the implementation charges only
//! extra work of its own. Their default implementations call the unscoped
//! method.

mod context;
pub use context::{DenseOperatorInput, DenseOperatorKernel, OperatorContext};

use core::marker::PhantomData;

use crate::{Dimensions, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};

fn validate_finite_bound<R: RealScalar>(maximum: R) -> Result<(), SciOxideError> {
    if !maximum.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "non-finite input bound",
        ));
    }
    if maximum < R::ZERO {
        return Err(SciOxideError::invalid_input(
            crate::argument!("maximum"),
            crate::InvalidInputReason::OutOfRange,
            "negative input bound",
        ));
    }
    Ok(())
}

fn validate_vector(values: &[impl Sized], expected: usize) -> Result<(), SciOxideError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(values.len()),
        ))
    }
}

fn checked_block_len(rows: usize, right_hand_sides: usize) -> Result<usize, SciOxideError> {
    rows.checked_mul(right_hand_sides)
        .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("block_vector_storage")))
}

/// A block of vectors stored column by column, as for several right-hand sides.
///
/// Column `j` is `values[j * rows..(j + 1) * rows]`.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlockVectorView<'a, S> {
    values: &'a [S],
    rows: usize,
    right_hand_sides: usize,
}

impl<'a, S> BlockVectorView<'a, S> {
    /// Creates a view of `right_hand_sides` columns of length `rows`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * right_hand_sides` overflows.
    /// - `DimensionMismatch` if `values.len() != rows * right_hand_sides`.
    pub fn new(
        values: &'a [S],
        rows: usize,
        right_hand_sides: usize,
    ) -> Result<Self, SciOxideError> {
        let expected = checked_block_len(rows, right_hand_sides)?;
        validate_vector(values, expected)?;
        Ok(Self {
            values,
            rows,
            right_hand_sides,
        })
    }

    /// Returns the length of each column.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Returns the column-major storage.
    pub const fn as_slice(&self) -> &[S] {
        self.values
    }

    /// Returns column `column`, counting from zero.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `column >= self.right_hand_sides()`.
    pub fn column(&self, column: usize) -> Result<&[S], SciOxideError> {
        if column >= self.right_hand_sides {
            return Err(SciOxideError::out_of_bounds(
                crate::quantity!("block_vector_columns"),
                column,
                self.right_hand_sides,
            ));
        }
        let start = column * self.rows;
        Ok(&self.values[start..start + self.rows])
    }
}

/// A mutable block of vectors stored column by column; see [`BlockVectorView`].
#[derive(Debug)]
#[must_use]
pub struct BlockVectorViewMut<'a, S> {
    values: &'a mut [S],
    rows: usize,
    right_hand_sides: usize,
}

impl<'a, S> BlockVectorViewMut<'a, S> {
    /// Creates a mutable view of `right_hand_sides` columns of length `rows`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * right_hand_sides` overflows.
    /// - `DimensionMismatch` if `values.len() != rows * right_hand_sides`.
    pub fn new(
        values: &'a mut [S],
        rows: usize,
        right_hand_sides: usize,
    ) -> Result<Self, SciOxideError> {
        let expected = checked_block_len(rows, right_hand_sides)?;
        validate_vector(values, expected)?;
        Ok(Self {
            values,
            rows,
            right_hand_sides,
        })
    }

    /// Returns the length of each column.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Returns the column-major storage.
    pub fn as_slice(&self) -> &[S] {
        self.values
    }

    /// Returns the column-major storage, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        self.values
    }

    /// Returns column `column`, counting from zero, mutably.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `column >= self.right_hand_sides()`.
    pub fn column_mut(&mut self, column: usize) -> Result<&mut [S], SciOxideError> {
        if column >= self.right_hand_sides {
            return Err(SciOxideError::out_of_bounds(
                crate::quantity!("block_vector_columns"),
                column,
                self.right_hand_sides,
            ));
        }
        let start = column * self.rows;
        Ok(&mut self.values[start..start + self.rows])
    }
}

/// A linear operator `A` of size `rows × cols`, applied as `output = A · input`.
///
/// The trait is object safe, so solvers can take `&dyn LinearOperator<S>`.
pub trait LinearOperator<S: NumericScalar> {
    /// Returns the number of rows, which is the output length.
    fn rows(&self) -> usize;

    /// Returns the number of columns, which is the input length.
    fn cols(&self) -> usize;

    /// Returns the workspace length that [`Self::apply_into`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError>;

    /// Writes `A · input` to `output`.
    ///
    /// Implementations check all lengths before writing to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input`, `output` or `workspace` has the wrong length.
    /// - `NonFinite` if the product contains NaN or infinity.
    /// - Numerical errors of the implementation.
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError>;

    /// Writes `A · input` to `output`, charging extra work to `context`.
    ///
    /// The calling solver has already counted this application under
    /// [`crate::EvaluationRole::LinearOperator`]. The result is
    /// [`Invocation::Evaluated`](crate::Invocation::Evaluated) when `output`
    /// holds the product, [`Invocation::BudgetExhausted`](crate::Invocation::BudgetExhausted)
    /// when a budget denied extra work, and
    /// [`Invocation::ExecutionStopped`](crate::Invocation::ExecutionStopped) after
    /// cancellation or a deadline. The default calls [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::apply_into`], and errors from nested work.
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        _context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<crate::Invocation<()>, SciOxideError> {
        self.apply_into(input, output, workspace)
            .map(crate::Invocation::Evaluated)
    }

    /// Writes `A · input` to `output` for an input that the solver already checked.
    ///
    /// The caller guarantees that every component of `input` is finite and
    /// that `maximum` bounds the magnitude of their real and imaginary parts.
    /// Implementations may skip the scan for non-finite input; they still check
    /// lengths and `maximum`. `output` is solver scratch: on a numerical error it
    /// may be partly written. On success it is finite. Accounting is as for
    /// [`Self::apply_into_scoped`], which the default calls after checking
    /// `maximum`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `maximum` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `maximum` is negative.
    /// - The errors of [`Self::apply_into_scoped`].
    ///
    /// # Examples
    ///
    /// ```
    /// fn apply(
    ///     a: &dyn scioxide_core::LinearOperator<f64>,
    ///     y: &mut [f64],
    ///     w: &mut [f64],
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    /// ) -> Result<(), scioxide_core::SciOxideError> {
    ///     // Count the application first, then apply A to x = [1.0].
    ///     if c.record_provider(scioxide_core::EvaluationRole::LinearOperator)?.is_granted() {
    ///         let mut context = scioxide_core::OperatorContext::new(c);
    ///         a.apply_finite_into_scoped(&[1.0], 1.0, y, w, &mut context)?.require()?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    fn apply_finite_into_scoped(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        workspace: &mut [S],
        context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<crate::Invocation<()>, SciOxideError> {
        validate_finite_bound(maximum)?;
        self.apply_into_scoped(input, output, workspace, context)
    }
}

/// A linear operator that can also apply its conjugate transpose `Aᴴ`.
///
/// For real scalars `Aᴴ` is the transpose `Aᵀ`.
pub trait AdjointOperator<S: NumericScalar>: LinearOperator<S> {
    /// Returns the workspace length that [`Self::apply_adjoint_into`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError>;

    /// Writes `Aᴴ · input` to `output`.
    ///
    /// `input` has length `rows()` and `output` has length `cols()`.
    /// Implementations check all lengths before writing to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice has the wrong length.
    /// - `NonFinite` if the product contains NaN or infinity.
    /// - Numerical errors of the implementation.
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError>;

    /// Writes `Aᴴ · input` to `output`, charging extra work to `context`.
    ///
    /// The calling solver has already counted this application under
    /// [`crate::EvaluationRole::LinearOperator`]. The result has the same
    /// meaning as for [`LinearOperator::apply_into_scoped`]. The default calls
    /// [`Self::apply_adjoint_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::apply_adjoint_into`], and errors from nested work.
    ///
    /// # Examples
    ///
    /// ```
    /// fn adjoint(
    ///     a: &dyn scioxide_core::AdjointOperator<f64>,
    ///     x: &[f64],
    ///     y: &mut [f64],
    ///     w: &mut [f64],
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    /// ) -> Result<(), scioxide_core::SciOxideError> {
    ///     if c.record_provider(scioxide_core::EvaluationRole::LinearOperator)?.is_granted() {
    ///         let mut context = scioxide_core::OperatorContext::new(c);
    ///         a.apply_adjoint_into_scoped(x, y, w, &mut context)?.require()?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        _context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<crate::Invocation<()>, SciOxideError> {
        self.apply_adjoint_into(input, output, workspace)
            .map(crate::Invocation::Evaluated)
    }

    /// Writes `Aᴴ · input` to `output` for an input that the solver already checked.
    ///
    /// The preconditions on `input` and `maximum`, the use of `output` as
    /// scratch and the accounting are those of
    /// [`LinearOperator::apply_finite_into_scoped`]. The default checks
    /// `maximum` and calls [`Self::apply_adjoint_into_scoped`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `maximum` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `maximum` is negative.
    /// - The errors of [`Self::apply_adjoint_into_scoped`].
    fn apply_adjoint_finite_into_scoped(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        workspace: &mut [S],
        context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<crate::Invocation<()>, SciOxideError> {
        validate_finite_bound(maximum)?;
        self.apply_adjoint_into_scoped(input, output, workspace, context)
    }
}

/// A linear operator applied to several vectors at once, `Y = A X`.
///
/// The default methods apply [`LinearOperator::apply_into`] column by column.
pub trait BlockOperator<S: NumericScalar>: LinearOperator<S> {
    /// Returns the workspace length that [`Self::apply_block_into`] needs for
    /// `right_hand_sides` columns.
    ///
    /// The default is `rows() * right_hand_sides + apply_workspace_len()`, or
    /// zero for zero columns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    fn apply_block_workspace_len(&self, right_hand_sides: usize) -> Result<usize, SciOxideError> {
        if right_hand_sides == 0 {
            return Ok(0);
        }
        checked_block_len(self.rows(), right_hand_sides)?
            .checked_add(self.apply_workspace_len()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(crate::quantity!("block_operator_workspace"))
            })
    }

    /// Writes `A · X` to `output`, one column at a time.
    ///
    /// The output is written only after every column has succeeded, so on
    /// error it is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the block shapes do not match `A`, the column
    ///   counts differ, or `workspace` has the wrong length.
    /// - The errors of [`LinearOperator::apply_into`] for any column.
    fn apply_block_into(
        &self,
        input: BlockVectorView<'_, S>,
        output: &mut BlockVectorViewMut<'_, S>,
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        if input.rows() != self.cols()
            || output.rows() != self.rows()
            || input.right_hand_sides() != output.right_hand_sides()
        {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(self.cols(), input.right_hand_sides()),
                Dimensions::two(input.rows(), output.right_hand_sides()),
            ));
        }
        let required = self.apply_block_workspace_len(input.right_hand_sides())?;
        validate_vector(workspace, required)?;
        // Workspace layout: a column-major staging block for all outputs,
        // followed by the single-column workspace, which every column reuses. The
        // caller's block is written only after every column has succeeded, so a
        // failure in a later column cannot leave it partly updated.
        let staged_len = checked_block_len(self.rows(), input.right_hand_sides())?;
        let (staged, operation_workspace) = workspace.split_at_mut(staged_len);
        (0..input.right_hand_sides()).try_for_each(|column| {
            let start = column * self.rows();
            self.apply_into(
                input.column(column)?,
                &mut staged[start..start + self.rows()],
                operation_workspace,
            )
        })?;
        output.as_mut_slice().copy_from_slice(staged);
        Ok(())
    }
}

/// A preconditioner `M⁻¹` of size `n × n`, applied as `output = M⁻¹ · input`.
///
/// The trait is object safe, so solvers can take `&dyn Preconditioner<S>`.
pub trait Preconditioner<S: NumericScalar> {
    /// Returns `n`, the length of the input and of the output.
    fn dimension(&self) -> usize;

    /// Returns the workspace length that [`Self::precondition_into`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError>;

    /// Writes `M⁻¹ · input` to `output`.
    ///
    /// Implementations check all lengths before writing to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice has the wrong length.
    /// - `NonFinite` if the result contains NaN or infinity.
    /// - Numerical errors of the implementation.
    fn precondition_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError>;

    /// Writes `M⁻¹ · input` to `output`, charging extra work to `context`.
    ///
    /// `output` and `workspace` are as for [`Self::precondition_into`]. The
    /// calling solver has already counted this application under
    /// [`crate::EvaluationRole::Preconditioner`]. The default calls
    /// [`Self::precondition_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::precondition_into`], and errors from nested work.
    ///
    /// # Examples
    ///
    /// ```
    /// fn apply(
    ///     p: &dyn scioxide_core::Preconditioner<f64>,
    ///     x: &[f64],
    ///     y: &mut [f64],
    ///     w: &mut [f64],
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    /// ) -> Result<(), scioxide_core::SciOxideError> {
    ///     if c.record_provider(scioxide_core::EvaluationRole::Preconditioner)?.is_granted() {
    ///         let mut context = scioxide_core::OperatorContext::new(c);
    ///         p.precondition_into_scoped(x, y, w, &mut context)?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    fn precondition_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        _context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<(), SciOxideError> {
        self.precondition_into(input, output, workspace)
    }

    /// Returns the workspace length that [`Self::precondition_block_into`] needs
    /// for `right_hand_sides` columns.
    ///
    /// The default is `dimension() * right_hand_sides + precondition_workspace_len()`,
    /// or zero for zero columns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    fn precondition_block_workspace_len(
        &self,
        right_hand_sides: usize,
    ) -> Result<usize, SciOxideError> {
        if right_hand_sides == 0 {
            return Ok(0);
        }
        checked_block_len(self.dimension(), right_hand_sides)?
            .checked_add(self.precondition_workspace_len()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(crate::quantity!("block_preconditioner_workspace"))
            })
    }

    /// Writes `M⁻¹ · X` to `output`, one column at a time.
    ///
    /// The output is written only after every column has succeeded, so on
    /// error it is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the block shapes do not match `n`, the column
    ///   counts differ, or `workspace` has the wrong length.
    /// - The errors of [`Self::precondition_into`] for any column.
    fn precondition_block_into(
        &self,
        input: BlockVectorView<'_, S>,
        output: &mut BlockVectorViewMut<'_, S>,
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        if input.rows() != self.dimension()
            || output.rows() != self.dimension()
            || input.right_hand_sides() != output.right_hand_sides()
        {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(self.dimension(), input.right_hand_sides()),
                Dimensions::two(input.rows(), output.right_hand_sides()),
            ));
        }
        let required = self.precondition_block_workspace_len(input.right_hand_sides())?;
        validate_vector(workspace, required)?;
        // Same staging layout and all-or-nothing commit as `apply_block_into`.
        let staged_len = checked_block_len(self.dimension(), input.right_hand_sides())?;
        let (staged, operation_workspace) = workspace.split_at_mut(staged_len);
        (0..input.right_hand_sides()).try_for_each(|column| {
            let start = column * self.dimension();
            self.precondition_into(
                input.column(column)?,
                &mut staged[start..start + self.dimension()],
                operation_workspace,
            )
        })?;
        output.as_mut_slice().copy_from_slice(staged);
        Ok(())
    }
}

// Runs `apply(input, workspace)` and copies `workspace` to `output` when every
// component is finite. The lengths are checked first, so `output` is unchanged
// on every error.
fn apply_staged<S: NumericScalar>(
    apply: &impl Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    (input, input_len): (&[S], usize),
    (output, output_len): (&mut [S], usize),
    workspace: &mut [S],
    detail: &'static str,
) -> Result<(), SciOxideError> {
    validate_vector(input, input_len)?;
    validate_vector(output, output_len)?;
    validate_vector(workspace, output_len)?;
    apply(input, workspace)?;
    if workspace
        .iter()
        .copied()
        .all(NumericScalar::has_finite_components)
    {
        output.copy_from_slice(workspace);
        Ok(())
    } else {
        Err(SciOxideError::non_finite(NonFiniteOrigin::Callback, detail))
    }
}

/// A [`LinearOperator`] defined by a closure `apply(input, output)`, with an optional adjoint.
///
/// `apply` receives `cols` input values and writes `A · input` to `rows`
/// output values. [`Self::with_adjoint`] adds a closure that receives `rows`
/// values and writes `Aᴴ · input` to `cols` values, which makes the operator an
/// [`AdjointOperator`]. A closure writes into the workspace, which holds `rows`
/// values for [`LinearOperator::apply_into`] and `cols` values for
/// [`AdjointOperator::apply_adjoint_into`]; the output is written only after the
/// closure succeeds and every component it wrote is finite. A non-finite
/// component is a `NonFinite` error with origin
/// [`Callback`](crate::NonFiniteOrigin::Callback), and an error of the closure
/// is returned unchanged. A closure that needs scratch owns it, for example in a
/// captured `RefCell`.
///
/// The example applies `A = [[1, 2], [0, 1]]` and its transpose to `(1, 1)`.
///
/// Run `cargo run -p scioxide-core --example workspace_operations` for a complete caller-owned storage example.
pub struct ClosureOperator<S, F, G = ()> {
    rows: usize,
    cols: usize,
    apply: F,
    adjoint: G,
    scalar: PhantomData<fn(S) -> S>,
}

impl<S, F> ClosureOperator<S, F>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    /// Creates a `rows × cols` operator; `apply(input, output)` writes `A · input` to `output`.
    pub const fn new(rows: usize, cols: usize, apply: F) -> Self {
        Self {
            rows,
            cols,
            apply,
            adjoint: (),
            scalar: PhantomData,
        }
    }

    /// Adds the adjoint; `adjoint(input, output)` writes `Aᴴ · input` to `output`.
    pub fn with_adjoint<G>(self, adjoint: G) -> ClosureOperator<S, F, G>
    where
        G: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ClosureOperator {
            rows: self.rows,
            cols: self.cols,
            apply: self.apply,
            adjoint,
            scalar: PhantomData,
        }
    }
}

impl<S, F, G> LinearOperator<S> for ClosureOperator<S, F, G>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows)
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        apply_staged(
            &self.apply,
            (input, self.cols),
            (output, self.rows),
            workspace,
            "operator callback returned a non-finite component",
        )
    }
}

impl<S, F, G> BlockOperator<S> for ClosureOperator<S, F, G>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
}

impl<S, F, G> AdjointOperator<S> for ClosureOperator<S, F, G>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    G: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols)
    }

    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        apply_staged(
            &self.adjoint,
            (input, self.rows),
            (output, self.cols),
            workspace,
            "operator callback returned a non-finite component",
        )
    }
}

/// A [`Preconditioner`] defined by a closure `apply(input, output)`.
///
/// `apply` receives `dimension` input values and writes `M⁻¹ · input` to
/// `dimension` output values. It writes into the workspace, which holds
/// `dimension` values; the output is written only after the closure succeeds
/// and every component it wrote is finite. A non-finite component is a
/// `NonFinite` error with origin [`Callback`](crate::NonFiniteOrigin::Callback),
/// and an error of the closure is returned unchanged.
///
/// The example is the Jacobi preconditioner of `diag(4, 2)`.
///
/// ```
/// use scioxide_core::{ClosurePreconditioner, Preconditioner};
///
/// let m = ClosurePreconditioner::<f64, _>::new(2, |r, z| {
///     z[0] = r[0] / 4.0;
///     z[1] = r[1] / 2.0;
///     Ok(())
/// });
/// let (mut z, mut workspace) = ([0.0; 2], [0.0; 2]);
/// m.precondition_into(&[2.0, 3.0], &mut z, &mut workspace)?;
/// assert!((z[0] - 0.5).abs() < 1.0e-15 && (z[1] - 1.5).abs() < 1.0e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct ClosurePreconditioner<S, F> {
    dimension: usize,
    apply: F,
    scalar: PhantomData<fn(S) -> S>,
}

impl<S, F> ClosurePreconditioner<S, F>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    /// Creates a preconditioner of size `dimension × dimension`; `apply(input, output)` writes `M⁻¹ · input` to `output`.
    pub const fn new(dimension: usize, apply: F) -> Self {
        Self {
            dimension,
            apply,
            scalar: PhantomData,
        }
    }
}

impl<S, F> Preconditioner<S> for ClosurePreconditioner<S, F>
where
    S: NumericScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.dimension)
    }

    fn precondition_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        apply_staged(
            &self.apply,
            (input, self.dimension),
            (output, self.dimension),
            workspace,
            "preconditioner callback returned a non-finite component",
        )
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::{
        BlockOperator, BlockVectorView, BlockVectorViewMut, ClosureOperator, LinearOperator,
    };
    use crate::SciOxideError;

    #[test]
    fn scoped_preconditioner_default_preserves_provider_and_admission() {
        use crate::Preconditioner;
        struct Provider(Cell<usize>);
        impl Preconditioner<f64> for Provider {
            fn dimension(&self) -> usize {
                1
            }
            fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn precondition_into(
                &self,
                x: &[f64],
                y: &mut [f64],
                _: &mut [f64],
            ) -> Result<(), SciOxideError> {
                self.0.set(self.0.get() + 1);
                super::validate_vector(x, 1)?;
                super::validate_vector(y, 1)?;
                y[0] = x[0] / 2.0;
                Ok(())
            }
        }
        let provider = Provider(Cell::new(0));
        let p: &dyn Preconditioner<f64> = &provider;
        let mut state = crate::SolveState::new();
        let mut budgets = crate::SolveBudgets::unbounded();
        let mut solve = state
            .begin_solve(
                crate::ExecutionContext::sequential(),
                crate::SolveIdentity::new(
                    crate::ProblemToken::new(0),
                    crate::algorithm_id!("test", "preconditioner", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        assert!(
            solve
                .record_provider(crate::EvaluationRole::Preconditioner)
                .unwrap()
                .is_granted()
        );
        let mut output = [17.];
        p.precondition_into_scoped(
            &[6.],
            &mut output,
            &mut [],
            &mut crate::OperatorContext::new(&mut solve),
        )
        .unwrap();
        assert_eq!(output, [3.]);
        assert_eq!(provider.0.get(), 1);
        assert_eq!(
            solve
                .counts()
                .own()
                .provider(crate::EvaluationRole::Preconditioner),
            1
        );
        assert!(
            p.precondition_into_scoped(
                &[],
                &mut output,
                &mut [],
                &mut crate::OperatorContext::new(&mut solve)
            )
            .is_err()
        );
        assert_eq!(output, [3.]);
    }

    #[test]
    fn finite_hook_defaults_preserve_checked_provider_and_reject_invalid_bounds() {
        let calls = Cell::new(0);
        let operator = ClosureOperator::new(1, 1, |x: &[f64], y: &mut [f64]| {
            calls.set(calls.get() + 1);
            y[0] = 2.0 * x[0];
            Ok(())
        });
        let mut state = crate::SolveState::new();
        let mut budgets = crate::SolveBudgets::unbounded();
        let identity = crate::SolveIdentity::new(
            crate::ProblemToken::new(0),
            crate::algorithm_id!("test", "finite_operator", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(
                crate::ExecutionContext::sequential(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let mut y = [7.];
        let mut scratch = [0.];
        for maximum in [f64::NAN, f64::INFINITY, -1.] {
            assert!(
                operator
                    .apply_finite_into_scoped(
                        &[1.],
                        maximum,
                        &mut y,
                        &mut scratch,
                        &mut crate::OperatorContext::new(&mut context)
                    )
                    .is_err()
            );
            assert_eq!(y, [7.]);
            assert_eq!(calls.get(), 0);
        }
        operator
            .apply_finite_into_scoped(
                &[3.],
                3.,
                &mut y,
                &mut scratch,
                &mut crate::OperatorContext::new(&mut context),
            )
            .and_then(crate::Invocation::require)
            .unwrap();
        assert_eq!(y, [6.]);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn block_views_cover_zero_extents_and_columns() {
        let empty = BlockVectorView::<f64>::new(&[], 0, 3).unwrap();
        assert_eq!(empty.column(2).unwrap(), &[] as &[f64]);

        let values = [1.0, 2.0, 3.0, 4.0];
        let view = BlockVectorView::new(&values, 2, 2).unwrap();
        assert_eq!(view.column(1).unwrap(), &[3.0, 4.0]);

        let calls = Cell::new(0);
        let operator = ClosureOperator::<f64, _>::new(2, 2, |_, _| {
            calls.set(calls.get() + 1);
            Ok(())
        });
        let input = BlockVectorView::new(&[], 2, 0).unwrap();
        let mut output_values = [];
        let mut output = BlockVectorViewMut::new(&mut output_values, 2, 0).unwrap();
        operator
            .apply_block_into(input, &mut output, &mut [])
            .unwrap();
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn closure_operator_stages_output_and_supports_blocks() {
        let operator = ClosureOperator::new(2, 2, |input: &[f64], output: &mut [f64]| {
            output[0] = 2.0 * input[0];
            output[1] = 3.0 * input[1];
            Ok(())
        });
        let mut output = [9.0, 9.0];
        let mut workspace = [0.0; 6];
        operator
            .apply_into(&[2.0, 4.0], &mut output, &mut workspace[..2])
            .unwrap();
        assert_eq!(output, [4.0, 12.0]);

        let input_values = [1.0, 2.0, 3.0, 4.0];
        let mut output_values = [0.0; 4];
        let input = BlockVectorView::new(&input_values, 2, 2).unwrap();
        let mut outputs = BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap();
        operator
            .apply_block_into(input, &mut outputs, &mut workspace)
            .unwrap();
        assert_eq!(output_values, [2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn closure_operator_does_not_commit_failed_or_non_finite_output() {
        let operator = ClosureOperator::new(1, 1, |_: &[f64], output: &mut [f64]| {
            output[0] = f64::NAN;
            Ok(())
        });
        let mut output = [7.0];
        let mut workspace = [0.0];
        assert!(
            operator
                .apply_into(&[1.0], &mut output, &mut workspace)
                .is_err()
        );
        assert_eq!(output, [7.0]);
    }
    #[test]
    fn scoped_adjoint_default_delegates_through_a_trait_object() {
        struct Scale;
        impl LinearOperator<f64> for Scale {
            fn rows(&self) -> usize {
                1
            }
            fn cols(&self) -> usize {
                1
            }
            fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn apply_into(
                &self,
                x: &[f64],
                y: &mut [f64],
                _: &mut [f64],
            ) -> Result<(), SciOxideError> {
                y[0] = 2.0 * x[0];
                Ok(())
            }
        }
        impl super::AdjointOperator<f64> for Scale {
            fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn apply_adjoint_into(
                &self,
                x: &[f64],
                y: &mut [f64],
                w: &mut [f64],
            ) -> Result<(), SciOxideError> {
                self.apply_into(x, y, w)
            }
        }
        let op: &dyn super::AdjointOperator<f64> = &Scale;
        let mut state = crate::SolveState::new();
        let mut budgets = crate::SolveBudgets::unbounded();
        let id = crate::SolveIdentity::new(
            crate::ProblemToken::new(1),
            crate::algorithm_id!("test", "adjoint", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(crate::ExecutionContext::sequential(), id, &mut budgets)
            .unwrap();
        assert!(
            context
                .record_provider(crate::EvaluationRole::LinearOperator)
                .unwrap()
                .is_granted()
        );
        let mut output = [0.0];
        op.apply_adjoint_into_scoped(
            &[3.0],
            &mut output,
            &mut [],
            &mut crate::OperatorContext::new(&mut context),
        )
        .and_then(crate::Invocation::require)
        .unwrap();
        assert!((output[0] - 6.0).abs() < f64::EPSILON);
        assert_eq!(
            context
                .counts()
                .own()
                .provider(crate::EvaluationRole::LinearOperator),
            1
        );
    }
}
