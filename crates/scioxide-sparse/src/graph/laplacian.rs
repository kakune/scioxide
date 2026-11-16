//! Graph Laplacians as sparse matrices.
use super::{GraphDirection, GraphPlan, finite_candidate};
use crate::SparseView;
use crate::{CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    AdjointOperator, BlockOperator, LinearOperator, NumericScalar, RealScalar, SciOxideError,
};
/// Selects the vertex degree used in a graph Laplacian.
///
/// `A[u, v]` is the weight of the edge from `u` to `v`. For an undirected graph
/// `A` is symmetric and both choices give the same degrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaplacianDegree {
    /// Out-degree `d_u = Σ_v A[u, v]`, the row sums. The unnormalized Laplacian
    /// then maps the constant vector to zero: `L 1 = 0`.
    Out,
    /// In-degree `d_v = Σ_u A[u, v]`, the column sums. The unnormalized Laplacian
    /// then satisfies `1ᵀ L = 0`.
    In,
}
/// Selects the scaling of a graph Laplacian.
///
/// `D` is the diagonal matrix of degrees and `A` the weighted adjacency matrix,
/// both without self-loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaplacianNormalization {
    /// `L = D − A`.
    Unnormalized,
    /// `L = D^(−1/2) (D − A) D^(−1/2)`. A vertex of degree zero uses scale 1 in
    /// place of `d^(−1/2)`, so its diagonal entry is zero.
    Symmetric,
}
/// Holds the Laplacian of a weighted graph as a sparse matrix.
///
/// The Laplacian is built from the degrees selected by [`LaplacianDegree`] and the
/// scaling selected by [`LaplacianNormalization`]. Self-loops are left out of both
/// the degrees and the adjacency matrix, and weights must be nonnegative. For an
/// undirected graph that stores both `A[u, v]` and `A[v, u]`, the edge weight is
/// the smaller of the two, as in every [`GraphPlan`]. The weights are real; the
/// scalar type `T` of the matrix may be real or complex of the same precision, so
/// that the Laplacian can be applied to complex vectors. The plan is a
/// [`LinearOperator`](scioxide_core::LinearOperator); for block or transposed
/// products, call [`sparse_dense_into`](crate::sparse_dense_into) on
/// [`matrix`](Self::matrix).
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, GraphDirection, GraphPlan, LaplacianDegree, LaplacianNormalization,
/// };
/// // Undirected edge 0 - 1 of weight 2.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![1], vec![2.0_f64])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Undirected)?;
/// let (out, plain) = (LaplacianDegree::Out, LaplacianNormalization::Unnormalized);
/// let plan = graph.laplacian::<f64>(out, plain)?;
/// assert_eq!(plan.degrees(), &[2.0, 2.0]);
/// let l = plan.matrix();
/// assert_eq!(
///     [l.get(0, 0)?, l.get(0, 1)?, l.get(1, 0)?, l.get(1, 1)?],
///     [2.0, -2.0, -2.0, 2.0]
/// );
/// // Negative weights lie outside the Laplacian domain.
/// let b = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![1], vec![-1.0_f64])?;
/// let graph = GraphPlan::new(b.view(), GraphDirection::Directed)?;
/// let error = graph.laplacian::<f64>(out, plain).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. R. K. Chung, *Spectral Graph Theory*, CBMS Regional Conference Series in
/// Mathematics 92, AMS, 1997.
#[derive(Debug, Clone)]
pub struct LaplacianPlan<T: NumericScalar> {
    matrix: CsrMatrix<T>,
    degrees: Vec<T::Real>,
}
impl<S: RealScalar> GraphPlan<S> {
    /// Computes the graph Laplacian with the given degree and scaling.
    ///
    /// `T` is the scalar type of the resulting matrix: the weight type `S`, or the
    /// complex type with real part `S`. The matrix stores the diagonal and one
    /// entry for every edge between distinct vertices, including edges of weight
    /// zero. See [`LaplacianPlan`] for the definition.
    ///
    /// # Errors
    ///
    /// - `DomainError` if a weight is negative, including the weight of a
    ///   self-loop.
    /// - `NumericalInstability`: `Overflow` if a degree overflows, `Underflow` if
    ///   a positive weight scales to zero under the symmetric normalization.
    /// - `SizeOverflow` if the number of entries overflows `usize`.
    /// - `AllocationFailure` if the matrix cannot be allocated.
    pub fn laplacian<T: NumericScalar<Real = S>>(
        &self,
        degree: LaplacianDegree,
        normalization: LaplacianNormalization,
    ) -> Result<LaplacianPlan<T>, SciOxideError> {
        let n = self.vertex_count();
        let mut degrees = try_filled(n, S::ZERO, scioxide_core::quantity!("degrees"))?;
        for &(u, v, weight) in &self.edges {
            if weight < S::ZERO {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("self"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "Laplacian weights must be nonnegative",
                ));
            }
            if u != v {
                let index = if degree == LaplacianDegree::Out { u } else { v };
                degrees[index] = finite_candidate(degrees[index] + weight)?;
            }
        }
        let mut builder = SparseBuilder::with_capacity(
            n,
            n,
            checked_add(
                n,
                self.edges.len(),
                scioxide_core::quantity!("laplacian_entries"),
            )?,
        )?;
        for (u, &d) in degrees.iter().enumerate() {
            let diagonal = if normalization == LaplacianNormalization::Symmetric && d > S::ZERO {
                S::ONE
            } else {
                d
            };
            builder.push(u, u, T::from_real(diagonal))?;
        }
        for &(u, v, weight) in &self.edges {
            if u == v {
                continue;
            }
            let value = if normalization == LaplacianNormalization::Unnormalized {
                weight
            } else {
                let a = if degrees[u] > S::ZERO {
                    degrees[u].sqrt()
                } else {
                    S::ONE
                };
                let b = if degrees[v] > S::ZERO {
                    degrees[v].sqrt()
                } else {
                    S::ONE
                };
                scioxide_core::Scaled::from_value(weight)?
                    .div(scioxide_core::Scaled::from_value(a)?)?
                    .div(scioxide_core::Scaled::from_value(b)?)?
                    .round()?
            };
            if weight > S::ZERO && value == S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "normalized Laplacian edge underflowed",
                ));
            }
            builder.push(u, v, T::from_real(-value))?;
        }
        Ok(LaplacianPlan {
            matrix: builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)?,
            degrees,
        })
    }
}
impl<T: NumericScalar> LaplacianPlan<T> {
    /// Returns the degree `d_u` of each vertex, without self-loops.
    pub fn degrees(&self) -> &[T::Real] {
        &self.degrees
    }
    /// Returns the Laplacian as a CSR matrix.
    pub fn matrix(&self) -> &CsrMatrix<T> {
        &self.matrix
    }
    /// Returns the Laplacian as a CSR matrix, consuming the plan without a copy.
    pub fn into_matrix(self) -> CsrMatrix<T> {
        self.matrix
    }
}
impl<T: NumericScalar> LinearOperator<T> for LaplacianPlan<T> {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }
    fn cols(&self) -> usize {
        self.matrix.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.matrix.apply_into(input, output, workspace)
    }
}
impl<T: NumericScalar> AdjointOperator<T> for LaplacianPlan<T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.matrix.apply_adjoint_into(input, output, workspace)
    }
}
impl<T: NumericScalar> BlockOperator<T> for LaplacianPlan<T> {}

/// Returns the graph Laplacian `L = D − A` of the directed graph of a real
/// square sparse matrix, with `D = diag(A 1)` the out-degrees.
///
/// Every stored entry `A[u, v]` is the weight of an edge from `u` to `v`;
/// self-loops are left out of `D` and `A`, and weights must be nonnegative.
/// Then `L 1 = 0`. This is [`GraphPlan::new`] with [`GraphDirection::Directed`]
/// followed by [`GraphPlan::laplacian`] with [`LaplacianDegree::Out`] and
/// [`LaplacianNormalization::Unnormalized`], in O(V + E log E). For a
/// symmetric `A` the result is the Laplacian of the undirected graph.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
/// // Edges 0 → 1 (2) and 1 → 0 (2).
/// let a = CsrMatrix::from_triplets(2, 2, &[0, 1], &[1, 0], &[2.0_f64, 2.0])?;
/// let l = scioxide_sparse::laplacian(a.view())?;
/// assert_eq!(
///     [l.get(0, 0)?, l.get(0, 1)?, l.get(1, 0)?, l.get(1, 1)?],
///     [2.0, -2.0, -2.0, 2.0]
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if the matrix is not square.
/// - `DomainError` if a stored weight is negative.
/// - `NumericalInstability` (`Overflow`) if a degree overflows.
/// - `SizeOverflow` or `AllocationFailure` if the plan or the matrix cannot be
///   allocated.
pub fn laplacian<S: RealScalar>(matrix: SparseView<'_, S>) -> Result<CsrMatrix<S>, SciOxideError> {
    Ok(GraphPlan::new(matrix, GraphDirection::Directed)?
        .laplacian::<S>(LaplacianDegree::Out, LaplacianNormalization::Unnormalized)?
        .into_matrix())
}
