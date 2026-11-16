use super::*;
use scioxide_linalg::{Matrix, VectorView};

/// Explicit Euclidean-orthonormal support-compatible rigid modes.
/// Pass the borrowed basis to BorderedSystem to verify operator annihilation,
/// load compatibility and a caller-selected gauge. Completeness is not certified.
pub struct ElasticityRigidModes<V: NumericScalar> {
    values: Vec<V>,
    rows: usize,
    columns: usize,
}
impl<V: NumericScalar> ElasticityRigidModes<V> {
    /// Number of surviving translations/rotations.
    pub const fn nullity(&self) -> usize {
        self.columns
    }
    /// Borrow the column-major basis for the shared nullspace contracts.
    pub fn basis(&self) -> Result<BlockVectorView<'_, V>, SciOxideError> {
        BlockVectorView::new(&self.values, self.rows, self.columns)
    }
}
pub(super) fn build<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &Space<R, V, D>,
    tolerance: Tolerance<R>,
    max_modes: usize,
) -> Result<ElasticityRigidModes<V>, SciOxideError> {
    let nodes = space.space.nodes;
    let graph = space.space.components()?;
    let labels = graph.labels();
    let count = D * (D + 1) / 2;
    let bound = size(
        graph.count(),
        count,
        scioxide_core::quantity!("rigid_modes"),
    )?;
    if bound > max_modes {
        return Err(invalid(
            scioxide_core::argument!("max_modes"),
            "rigid-mode column ceiling exceeded",
        ));
    }
    let mut values = try_with_capacity(
        size(space.n(), bound, scioxide_core::quantity!("values"))?,
        scioxide_core::quantity!("values"),
    )?;
    let mut modes = try_filled(
        size(space.n(), count, scioxide_core::quantity!("modes"))?,
        V::ZERO,
        scioxide_core::quantity!("modes"),
    )?;
    let mut candidate = try_filled(space.n(), V::ZERO, scioxide_core::quantity!("candidate"))?;
    let mut columns = 0;
    for component in 0..graph.count() {
        modes.fill(V::ZERO);
        let members = || (0..nodes).filter(move |&i| labels[i] == component);
        // Labels are ordered by smallest node, so the first member is the origin.
        let origin = members()
            .next()
            .map(|node| space.space.dof_point(node))
            .transpose()?
            .unwrap_or([R::ZERO; D]);
        let mut scale = R::ZERO;
        for node in members() {
            let x = space.space.dof_point(node)?;
            for j in 0..D {
                scale = scale.max(
                    finite(
                        x[j] - origin[j],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?
                    .abs(),
                );
            }
        }
        if scale == R::ZERO {
            scale = R::ONE;
        }
        for node in members() {
            let x = space.space.dof_point(node)?;
            for a in 0..D {
                modes[a * space.n() + a * nodes + node] = V::ONE;
            }
            let mut mode = D;
            for a in 0..D {
                for b in a + 1..D {
                    modes[mode * space.n() + a * nodes + node] =
                        V::from_real(-(x[b] - origin[b]) / scale);
                    modes[mode * space.n() + b * nodes + node] =
                        V::from_real((x[a] - origin[a]) / scale);
                    mode += 1;
                }
            }
        }
        // Only a 3x3 or 6x6 support Gram matrix is factored; no global dense matrix.
        let mut gram = Matrix::<R>::zeros(count, count)?;
        for i in 0..count {
            for j in i..count {
                let value = (0..space.n())
                    .filter(|&row| space.prescribed(row).is_some())
                    .fold(R::ZERO, |s, row| {
                        s + modes[i * space.n() + row].real_component()
                            * modes[j * space.n() + row].real_component()
                    });
                gram.as_mut_slice()[i * count + j] =
                    finite(value, scioxide_core::NonFiniteOrigin::Intermediate)?;
                gram.as_mut_slice()[j * count + i] = value;
            }
        }
        let factor = scioxide_linalg::SvdDecomposition::compute(&gram)?;
        let largest = factor.singular_values().first().copied().unwrap_or(R::ZERO);
        let cutoff = tolerance.absolute() + tolerance.relative() * largest;
        for (k, &s) in factor.singular_values().iter().enumerate() {
            if s > cutoff {
                continue;
            }
            candidate.fill(V::ZERO);
            for (row, value) in candidate.iter_mut().enumerate() {
                for j in 0..count {
                    *value = finite(
                        *value
                            + modes[j * space.n() + row]
                                * V::from_real(factor.vt_matrix().as_slice()[k * count + j]),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
            // Twice reorthogonalize the at-most-six local rigid directions using
            // shared conjugate dots/norms, including combinations fixed by supports.
            for _ in 0..2 {
                for previous in values.chunks_exact(space.n()) {
                    let dot = VectorView::from_slice(previous)
                        .dotc_left(VectorView::from_slice(&candidate))?;
                    for (v, &q) in candidate.iter_mut().zip(previous) {
                        *v = finite(*v - q * dot, scioxide_core::NonFiniteOrigin::Intermediate)?;
                    }
                }
            }
            let norm = VectorView::from_slice(&candidate).norm2()?;
            if norm <= tolerance.absolute() {
                continue;
            }
            for v in &mut candidate {
                *v = finite(
                    *v / V::from_real(norm),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            // Gram classification is only a candidate test; enforce displacement
            // supports directly before publishing a putative null direction.
            if (0..space.n())
                .filter(|&i| space.prescribed(i).is_some())
                .any(|i| candidate[i].abs() > tolerance.absolute() + tolerance.relative())
            {
                return Err(invalid(
                    scioxide_core::argument!("tolerance"),
                    "ambiguous rigid-mode support rank; adjust tolerance",
                ));
            }
            values.extend_from_slice(&candidate);
            columns += 1;
        }
    }
    Ok(ElasticityRigidModes {
        values,
        rows: space.n(),
        columns,
    })
}
