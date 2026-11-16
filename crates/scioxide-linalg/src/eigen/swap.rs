//! Local trial transforms and backward-error acceptance shared by Schur and QZ.
use crate::kernel::frames::scalar;
use crate::workspace::storage::MatrixStorage;
use scioxide_core::{
    AlgebraicScalar, NumericScalar, NumericalIssue, RealScalar, Scaled, ScaledValue, SciOxideError,
};

/// Standard Schur similarity: 64 epsilon times the local maximum entry magnitude.
pub(super) const SCHUR_ROUNDOFF: usize = 64;
/// Generalized two-basis exchange: keep the conservative 128 epsilon allowance
/// for each pencil factor, including any explicitly restored infinite-eigenvalue zeros.
pub(super) const QZ_ROUNDOFF: usize = 128;

pub(super) fn rejected() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::IllConditioned,
        "adjacent blocks cannot be separated with a backward-stable swap",
    )
}

/// Return a staged `U^H A V` after weak zero-entry and strong reconstruction
/// tests. Both use `roundoff * epsilon * max_ij |A_ij|`. All required zeros
/// participate in the strong test; callers write results only after every factor passes.
pub(super) fn accept<S: NumericScalar>(
    a: crate::MatrixView<'_, S>,
    u: &[S; 16],
    v: &[S; 16],
    zero: impl Fn(usize, usize) -> bool,
    roundoff: usize,
) -> Result<[S; 16], SciOxideError> {
    let d = a.rows();
    let mut trial = [S::ZERO; 16];
    let norm = (0..d)
        .flat_map(|i| (0..d).map(move |j| a.get_element(i, j).abs()))
        .fold(S::Real::ZERO, S::Real::max);
    let threshold = crate::checked::rounded_count::<S::Real>(roundoff)? * S::Real::EPSILON * norm;
    for i in 0..d {
        for j in 0..d {
            let value = crate::checked::finite_candidate((0..d).fold(S::ZERO, |sum, r| {
                sum + u[r * 4 + i].conj()
                    * (0..d).fold(S::ZERO, |sum, c| sum + a.get_element(r, c) * v[c * 4 + j])
            }))?;
            trial[i * 4 + j] = if zero(i, j) {
                if value.abs() > threshold {
                    return Err(rejected());
                }
                S::ZERO
            } else {
                value
            };
        }
    }
    for i in 0..d {
        for j in 0..d {
            let reconstructed =
                crate::checked::finite_candidate((0..d).fold(S::ZERO, |sum, r| {
                    sum + u[i * 4 + r]
                        * (0..d).fold(S::ZERO, |sum, c| {
                            sum + trial[r * 4 + c] * v[j * 4 + c].conj()
                        })
                }))?;
            if (reconstructed - a.get_element(i, j)).abs() > threshold {
                return Err(rejected());
            }
        }
    }
    Ok(trial)
}

/// Orthonormal completion of the scaled graph [X; I], shared by both Schur families.
///
pub(super) fn graph_basis<S: NumericScalar>(
    rhs: &[ScaledValue<scioxide_core::Complex<S::Real>>],
    p: usize,
    q: usize,
) -> Result<[S; 16], SciOxideError> {
    let d = p + q;
    let mut graph = [S::ZERO; 16];
    let mut u = [S::ZERO; 16];
    for i in 0..d {
        u[i * 4 + i] = S::ONE;
    }
    // Column j of `[X; I]` is divided by `max(1, max_i |x_ij|)` before
    // rounding back to the working scalar, so a huge X (nearly coincident
    // eigenvalues) cannot overflow; this scales columns only and keeps the
    // spanned subspace unchanged.
    for j in 0..q {
        let mut maximum = Scaled::from_value(S::Real::ONE)?;
        for i in 0..p {
            let value = rhs[i * q + j].magnitude()?;
            if !value.abs_le(maximum) {
                maximum = value;
            }
        }
        let denominator = ScaledValue::from_components(maximum, Scaled::ZERO);
        for i in 0..p {
            graph[i * 4 + j] = scalar(rhs[i * q + j].div(denominator)?)?;
        }
        graph[(p + j) * 4 + j] =
            S::from_real(Scaled::from_value(S::Real::ONE)?.div(maximum)?.round()?);
    }
    // Householder QR of [X; I]. Its leading q columns span the second block's
    // invariant graph. Complete the orthogonal/unitary basis via the same reflectors.
    for k in 0..q {
        let mut v = [S::ZERO; 4];
        for i in k..d {
            v[i] = graph[i * 4 + k];
        }
        let h = crate::reflector::generate(crate::VectorViewMut::from_slice(&mut v[k..d]))?;
        if h.beta == S::Real::ZERO {
            return Err(rejected());
        }
        let v = crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(&v[k..d]));
        crate::reflector::apply(
            v,
            h.tau.conj(),
            crate::MatrixViewMut::from_slice_with_strides(
                d - k,
                q - k,
                &mut graph[k * 4 + k..],
                4,
                1,
            )?,
        )?;
        crate::reflector::apply(
            v.conjugate(),
            h.tau,
            crate::MatrixViewMut::from_slice_with_strides(d, d - k, &mut u[k..], 4, 1)?.transpose(),
        )?;
    }
    Ok(u)
}

pub(super) fn apply_basis_right<S: NumericScalar>(
    a: &mut impl MatrixStorage<S>,
    start: usize,
    d: usize,
    u: &[S; 16],
) -> Result<(), SciOxideError> {
    let n = a.rows();
    for row in 0..n {
        let mut old = [S::ZERO; 4];
        for (i, v) in old[..d].iter_mut().enumerate() {
            *v = a.get_element(row, start + i);
        }
        for j in 0..d {
            a.as_mut_slice()[row * n + start + j] = crate::checked::finite_candidate(
                (0..d).fold(S::ZERO, |sum, k| sum + old[k] * u[k * 4 + j]),
            )?;
        }
    }
    Ok(())
}
pub(super) fn apply_basis_left<S: NumericScalar>(
    a: &mut impl MatrixStorage<S>,
    start: usize,
    d: usize,
    u: &[S; 16],
) -> Result<(), SciOxideError> {
    let n = a.rows();
    for col in 0..n {
        let mut old = [S::ZERO; 4];
        for (i, v) in old[..d].iter_mut().enumerate() {
            *v = a.get_element(start + i, col);
        }
        for i in 0..d {
            a.as_mut_slice()[(start + i) * n + col] = crate::checked::finite_candidate(
                (0..d).fold(S::ZERO, |sum, k| sum + u[k * 4 + i].conj() * old[k]),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn weak_thresholds_include_forced_diagonal_zeros_and_strong_reconstruction() {
        let mut basis = [0.; 16];
        basis[0] = 1.;
        basis[5] = 1.;
        for multiplier in [SCHUR_ROUNDOFF, QZ_ROUNDOFF] {
            for target in [(1, 0), (0, 0)] {
                for factor in [0.5, 1., 2.] {
                    let mut a = [1., 0., 0., 1.];
                    a[target.0 * 2 + target.1] = factor * multiplier as f64 * f64::EPSILON;
                    let trial = accept(
                        crate::MatrixView::from_slice(2, 2, &a).unwrap(),
                        &basis,
                        &basis,
                        |i, j| (i, j) == target,
                        multiplier,
                    );
                    assert_eq!(trial.is_ok(), factor <= 1.);
                    if let Ok(trial) = trial {
                        assert_eq!(trial[target.0 * 4 + target.1], 0.);
                    }
                }
            }
        }
        let a = crate::MatrixView::from_slice(2, 2, &[1., 0., 0., 1.]).unwrap();
        basis[0] = 2.;
        assert!(accept(a, &basis, &basis, |_, _| false, SCHUR_ROUNDOFF).is_err());
    }
}
