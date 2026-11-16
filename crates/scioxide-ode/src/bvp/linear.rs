//! Direct Newton steps by dense or sparse LU of the assembled collocation Jacobian.
use super::*;
use crate::stiff::linear::{LinearStorage, OdeStageBackend};
use assembly::AssemblyWorkspace;
use scioxide_core::StructuralPatternView;
use scioxide_linalg::{IterativeSolveReport, LinearSolveTermination, Vector};
use scioxide_rootfind::{NonlinearLinearSolver, NonlinearLinearization};

struct Storage<T: NumericScalar> {
    linear: LinearStorage<T>,
    assembly: AssemblyWorkspace<T>,
    rhs: Vec<T>,
    // Newton freezes the point through correction and independent verification;
    // the injected solve assembles this matrix immediately before that verification.
    valid: bool,
}
pub(super) struct DirectCorrection<T: NumericScalar> {
    storage: RefCell<Storage<T>>,
    layout: Layout,
    nodes: usize,
    limits: SparseLuLimits,
    rows: Vec<usize>,
    columns: Vec<usize>,
    pattern_scratch: Vec<usize>,
}

/// Boundary rows couple both endpoints; every other row couples adjacent nodes.
fn pattern(
    n: usize,
    k: usize,
    nodes: usize,
    rows: &mut Vec<usize>,
    columns: &mut Vec<usize>,
) -> Result<(), SciOxideError> {
    let dim = nodes * n + k;
    let width = 2 * n + k;
    super::resize(rows, dim + 1, 0)?;
    super::resize(columns, dim * width, 0)?;
    rows[0] = 0;
    for row in 0..dim {
        let start = if row < (nodes - 1) * n {
            (row / n) * n
        } else {
            0
        };
        let end = if row < (nodes - 1) * n {
            start + n
        } else {
            (nodes - 1) * n
        };
        for (slot, column) in columns[row * width..(row + 1) * width]
            .iter_mut()
            .zip((start..start + n).chain(end..end + n).chain(nodes * n..dim))
        {
            *slot = column;
        }
        rows[row + 1] = (row + 1) * width;
    }
    Ok(())
}
impl<T: NumericScalar> DirectCorrection<T> {
    pub(super) fn new(
        layout: Layout,
        nodes: usize,
        capacity: usize,
        bytes: usize,
    ) -> Result<Self, SciOxideError> {
        let dim = layout.size(nodes)?;
        let (mut rows, mut columns) = (Vec::new(), Vec::new());
        pattern(layout.n, layout.k, nodes, &mut rows, &mut columns)?;
        let limits = layout.sparse_limits(capacity, bytes)?;
        let linear = match layout.backend {
            BvpBackend::Dense => LinearStorage::new(dim, &OdeStageBackend::Dense)?,
            BvpBackend::Sparse { .. } => {
                let p = StructuralPatternView::new(dim, dim, &rows, &columns)?;
                LinearStorage::bordered_sparse(p, layout.n, layout.k, nodes, limits)?
            }
            BvpBackend::Krylov { .. } => {
                return Err(invalid(
                    scioxide_core::argument!("backend"),
                    "direct correction requires assembled storage",
                ));
            }
        };
        Ok(Self {
            storage: RefCell::new(Storage {
                linear,
                assembly: AssemblyWorkspace::new(layout.n, layout.k)?,
                rhs: try_filled(dim, T::ZERO, scioxide_core::quantity!("bvp_linear_rhs"))?,
                valid: false,
            }),
            layout,
            nodes,
            limits,
            rows,
            columns,
            pattern_scratch: Vec::new(),
        })
    }
    pub(super) fn resize(
        &mut self,
        nodes: usize,
        capacity: usize,
        bytes: usize,
    ) -> Result<(), SciOxideError> {
        let dim = self.layout.size(nodes)?;
        let limits = self.layout.sparse_limits(capacity, bytes)?;
        let storage = self.storage.get_mut();
        storage.valid = false;
        super::resize(&mut storage.rhs, dim, T::ZERO)?;
        if self.nodes == nodes && self.limits == limits {
            return Ok(());
        }
        pattern(
            self.layout.n,
            self.layout.k,
            nodes,
            &mut self.rows,
            &mut self.columns,
        )?;
        let p = StructuralPatternView::new(dim, dim, &self.rows, &self.columns)?;
        super::resize(
            &mut self.pattern_scratch,
            dim + 1 + 2 * self.columns.len(),
            0,
        )?;
        storage
            .linear
            .resize_direct(dim, p, limits, &mut self.pattern_scratch)?;
        self.nodes = nodes;
        self.limits = limits;
        Ok(())
    }
    pub(super) fn bind<'a, 'b, 'p, P: BvpSystem<T> + ?Sized>(
        &'a self,
        collocation: &'a Collocation<'b, 'p, T, P>,
    ) -> BoundCorrection<'a, 'b, 'p, T, P> {
        BoundCorrection {
            direct: self,
            collocation,
        }
    }
    pub(super) fn apply(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        let storage = self.storage.borrow();
        if !storage.valid {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "assembled Jacobian is not ready for verification",
            ));
        }
        storage.linear.apply_assembled(input, output, scratch)
    }
}
pub(super) struct BoundCorrection<'a, 'b, 'p, T: NumericScalar, P: BvpSystem<T> + ?Sized> {
    direct: &'a DirectCorrection<T>,
    collocation: &'a Collocation<'b, 'p, T, P>,
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> NonlinearLinearSolver<T>
    for BoundCorrection<'_, '_, '_, T, P>
{
    fn solve_into(
        &self,
        linearization: &NonlinearLinearization<'_, T>,
        rhs: &Vector<T>,
        output: &mut Vector<T>,
        _: ResidualTolerance<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<T::Real>, SciOxideError> {
        let mut storage = self.direct.storage.borrow_mut();
        let Storage {
            linear,
            assembly,
            rhs: physical_rhs,
            valid,
        } = &mut *storage;
        *valid = false;
        let (n, k) = self.collocation.dimensions()?;
        let m = self.direct.nodes;
        match linear {
            LinearStorage::Dense { matrix, .. } => {
                let dim = rhs.len();
                matrix.as_mut_slice().fill(T::ZERO);
                self.collocation.assemble(
                    linearization.point(),
                    linearization.variable_divisors(),
                    assembly,
                    context,
                    |row, values| {
                        let a = if row < (m - 1) * n { (row / n) * n } else { 0 };
                        let b = if row < (m - 1) * n {
                            a + n
                        } else {
                            (m - 1) * n
                        };
                        let target = &mut matrix.as_mut_slice()[row * dim..(row + 1) * dim];
                        target[a..a + n].copy_from_slice(&values[..n]);
                        target[b..b + n].copy_from_slice(&values[n..2 * n]);
                        target[m * n..].copy_from_slice(&values[2 * n..]);
                        Ok(())
                    },
                )?;
            }
            _ => linear.assemble_sparse(|_, values| {
                let width = 2 * n + k;
                self.collocation.assemble(
                    linearization.point(),
                    linearization.variable_divisors(),
                    assembly,
                    context,
                    |row, local| {
                        values[row * width..(row + 1) * width].copy_from_slice(local);
                        Ok(())
                    },
                )
            })?,
        }
        *valid = true;
        admit(context, EvaluationRole::Factorization)?;
        linear.factor(context.execution())?;
        // Factor in physical units, then map the scaled RHS/correction at the boundary.
        // This avoids forming ill-scaled matrix entries for very large state units.
        for ((out, &value), &scale) in physical_rhs
            .iter_mut()
            .zip(rhs.as_slice())
            .zip(linearization.row_divisors())
        {
            *out = value * T::from_real(scale);
        }
        finite(physical_rhs, NonFiniteOrigin::Intermediate)?;
        admit(context, EvaluationRole::LinearSolve)?;
        linear.solve_direct(physical_rhs, output.as_mut_slice(), context.execution())?;
        for (value, &scale) in output
            .as_mut_slice()
            .iter_mut()
            .zip(linearization.variable_divisors())
        {
            *value = *value / T::from_real(scale);
        }
        finite(output.as_slice(), NonFiniteOrigin::Intermediate)?;
        Ok(IterativeSolveReport::new(
            LinearSolveTermination::Converged,
            0,
            None,
            PartialResultTrust::DiagnosticOnly,
        ))
    }
}
