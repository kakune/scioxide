//! Input checks, budget accounting and reordering helpers shared by the eigensolvers.
use crate::MatrixView;
use crate::workspace::{RealRegion, ScalarRegion};
use scioxide_core::NumericScalar;
use scioxide_core::{Dimensions, SciOxideError};

/// Validate a square finite view once at a public boundary.
pub(super) fn square_finite<'a, S: NumericScalar>(
    view: MatrixView<'a, S>,
    dispatch: scioxide_core::backend::KernelDispatch,
) -> Result<crate::FiniteMatrixView<'a, S>, SciOxideError> {
    let n = view.rows();
    if n != view.cols() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(n, n),
            Dimensions::two(n, view.cols()),
        ));
    }
    view.try_finite_dispatched(dispatch)
}

/// Copy a logical square view into row-major scratch.
pub(super) fn copy_view<S: NumericScalar>(view: MatrixView<'_, S>, target: &mut [S]) {
    let n = view.cols();
    target
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = view.get_element(i / n, i % n));
}

/// Fill row-major square scratch with the identity.
pub(super) fn set_identity<S: NumericScalar>(target: &mut [S], n: usize) {
    target.fill(S::ZERO);
    (0..n).for_each(|i| target[i * n + i] = S::ONE);
}

pub(super) fn reorder_step(
    used: &mut usize,
    limit: usize,
    solve: &mut scioxide_core::SolveContext<'_, '_>,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    solve.execution().check()?;
    if *used >= limit {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new().with_counts(solve.counts()),
            detail,
        ));
    }
    crate::solve_support::admit_iteration(
        None,
        Some(&mut *solve),
        "solve iteration budget exhausted",
    )?;
    *used += 1;
    Ok(())
}

/// Checked `n * n`.
pub(super) fn square(n: usize) -> Result<usize, SciOxideError> {
    times(n, n)
}

/// Checked `factor * value`.
pub(super) fn times(factor: usize, value: usize) -> Result<usize, SciOxideError> {
    factor.checked_mul(value).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("eigen_scratch_elements"))
    })
}

/// Checked sum of arena region lengths.
pub(super) fn total(parts: &[usize]) -> Result<usize, SciOxideError> {
    parts.iter().try_fold(0usize, |sum, &part| {
        sum.checked_add(part).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("eigen_scratch_elements"))
        })
    })
}

/// Requirements with only the direct regions populated.
pub(super) fn direct_requirements<S: NumericScalar>(
    lengths: DirectLengths,
    byte_limit: Option<usize>,
) -> Result<crate::LinalgRequirements, SciOxideError> {
    let mut requirements = crate::LinalgRequirements::new(byte_limit);
    *requirements.scalar_mut(ScalarRegion::Direct) = lengths.scalars;
    *requirements.real_mut(RealRegion::Direct) = lengths.reals;
    requirements.indices = lengths.indices;
    requirements.extended = lengths.extended;
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

/// Direct-arena lengths of one eigen job.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DirectLengths {
    pub(super) scalars: usize,
    pub(super) reals: usize,
    pub(super) indices: usize,
    pub(super) extended: usize,
}

/// Solve identity of an eigen stage of order `n`.
pub(super) fn identity(
    algorithm: scioxide_core::AlgorithmId,
    n: usize,
) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(scioxide_core::ProblemToken::new(0), algorithm, n as u128, 0)
}

/// Check a leading Schur or deflating subspace dimension.
pub(super) fn validate_subspace(
    n: usize,
    dimension: usize,
    blocks: &[super::SchurBlock],
) -> Result<(), SciOxideError> {
    if dimension > n {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("dimension"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "selected dimension exceeds matrix order",
        ));
    }
    if blocks
        .iter()
        .any(|b| b.start() < dimension && dimension < b.start() + b.size())
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("dimension"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "selected dimension splits an indivisible real 2x2 block",
        ));
    }
    Ok(())
}

/// Stable selected-first traversal, independent of ordinary/generalized swaps.
pub(super) fn reorder_blocks(
    blocks: &mut [[usize; 2]],
    selected: &mut [usize],
    selection: &[bool],
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
    mut swap: impl FnMut(usize, usize, usize) -> Result<(), SciOxideError>,
) -> Result<(usize, usize), SciOxideError> {
    let mut dimension = 0;
    for (flag, &[start, size]) in selected.iter_mut().zip(blocks.iter()) {
        let yes = selection[start..start + size].iter().any(|&v| v);
        *flag = usize::from(yes);
        if yes {
            dimension += size;
        }
    }
    let (mut target, mut swaps) = (0, 0);
    for index in 0..blocks.len() {
        if selected[index] == 0 {
            continue;
        }
        let mut current = index;
        while current > target {
            admit()?;
            let [start, left] = blocks[current - 1];
            let [_, right] = blocks[current];
            swap(start, left, right)?;
            blocks[current - 1] = [start, right];
            blocks[current] = [start + right, left];
            selected.swap(current - 1, current);
            current -= 1;
            swaps += 1;
        }
        target += 1;
    }
    Ok((dimension, swaps))
}

pub(super) fn projector_rhs<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    m: usize,
    rhs: &mut [crate::workspace::Extended<S>],
) -> Result<(), SciOxideError> {
    let cols = matrix.cols() - m;
    for (index, value) in rhs.iter_mut().enumerate() {
        *value =
            crate::kernel::frames::extended(matrix.get_element(index / cols, m + index % cols))?;
    }
    Ok(())
}

type ProjectorEstimates<R, const N: usize, const D: usize> =
    ([R; N], [Option<scioxide_core::Scaled<R>>; D]);

/// Shared projector norm and inverse-operator traversal. The callback owns
/// ordinary/coupled Sylvester semantics and counts each operator application.
pub(super) fn projector_estimates<R: scioxide_core::RealScalar, const N: usize, const D: usize>(
    rhs: &mut [scioxide_core::ScaledValue<scioxide_core::Complex<R>>],
    mut apply: impl FnMut(
        &mut [scioxide_core::ScaledValue<scioxide_core::Complex<R>>],
        bool,
        bool,
    ) -> Result<bool, SciOxideError>,
) -> Result<ProjectorEstimates<R, N, D>, SciOxideError> {
    use scioxide_core::Scaled;
    let mut conditions = [R::ZERO; N];
    let mut separations = [None; D];
    if !apply(rhs, false, false)? {
        return Ok((conditions, separations));
    }
    let one = Scaled::from_value(R::ONE)?;
    for (result, values) in conditions.iter_mut().zip(rhs.chunks_exact(rhs.len() / N)) {
        let mut norm = one;
        for &value in values {
            let magnitude = value.magnitude()?;
            norm = norm.add(magnitude.mul(magnitude)?)?;
        }
        *result = one.div(norm.sqrt()?)?.round()?;
        if *result == R::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "projector reciprocal condition is not representable",
            ));
        }
    }
    for (direction, separation) in separations.iter_mut().enumerate() {
        *separation = crate::estimate::separation_estimate(rhs, |rhs, adjoint| {
            apply(rhs, adjoint, direction != 0)
        })?;
    }
    Ok((conditions, separations))
}
