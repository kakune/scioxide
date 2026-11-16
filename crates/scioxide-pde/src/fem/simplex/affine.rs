//! Reference moments for spatially constant affine forms.
use super::*;

pub(super) struct AffineForms<R: RealScalar, const D: usize> {
    mass: Vec<R>,
    stiffness: Vec<[[R; D]; D]>,
}
impl<R: RealScalar, const D: usize> AffineForms<R, D> {
    pub(super) fn new(space: &SimplexSpace<R, D>) -> Result<Self, SciOxideError> {
        let n = space.order.nodes(D);
        let mut mass = try_filled(n * n, R::ZERO, scioxide_core::quantity!("reference_mass"))?;
        let mut stiffness = try_filled(
            n * n,
            [[R::ZERO; D]; D],
            scioxide_core::quantity!("reference_stiffness"),
        )?;
        for reference in space.volume_nodes() {
            let sample = space.reference_node(reference);
            for i in 0..n {
                for j in 0..n {
                    mass[i * n + j] +=
                        sample.weight * sample.solution.values[i] * sample.solution.values[j];
                    for (a, row) in stiffness[i * n + j].iter_mut().enumerate() {
                        for (b, entry) in row.iter_mut().enumerate() {
                            *entry += sample.weight
                                * sample.solution.gradients[i][a]
                                * sample.solution.gradients[j][b];
                        }
                    }
                }
            }
        }
        Ok(Self { mass, stiffness })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "explicit affine operands and caller-owned output"
    )]
    pub(super) fn cell<V: NumericScalar<Real = R>>(
        &self,
        map: &SimplexFemMappedPoint<R, D>,
        components: usize,
        diffusion: &[V],
        reaction: &[V],
        transformed: &mut [V],
        matrix: &mut [V],
        mass: &mut [R],
    ) -> Result<bool, SciOxideError> {
        let kd = components * D;
        for a in 0..components {
            for b in 0..components {
                for alpha in 0..D {
                    for beta in 0..D {
                        let mut value = V::ZERO;
                        for i in 0..D {
                            for j in 0..D {
                                let coefficient = diffusion[(a * D + i) * kd + b * D + j];
                                if coefficient == V::ZERO {
                                    continue;
                                }
                                let x = map.inverse_jacobian[alpha][i];
                                let y = map.inverse_jacobian[beta][j];
                                let geometry = x * y;
                                if x != R::ZERO && y != R::ZERO && !geometry.is_normal() {
                                    return Ok(false);
                                }
                                let term = coefficient * V::from_real(geometry);
                                if geometry != R::ZERO && !term.abs().is_normal() {
                                    return Ok(false);
                                }
                                value = value + term;
                            }
                        }
                        if !value.has_finite_components() {
                            return Ok(false);
                        }
                        transformed[(a * D + alpha) * kd + b * D + beta] = value;
                    }
                }
            }
        }
        let n = mass.len().isqrt();
        let m = n * components;
        for i in 0..n {
            for j in 0..n {
                let weight = reference::product(map.determinant, self.mass[i * n + j])?;
                mass[i * n + j] = weight;
                for a in 0..components {
                    for b in 0..components {
                        let mut value = reaction[a * components + b] * V::from_real(weight);
                        for alpha in 0..D {
                            for beta in 0..D {
                                value = value
                                    + transformed[(a * D + alpha) * kd + b * D + beta]
                                        * V::from_real(
                                            map.determinant
                                                * self.stiffness[i * n + j][alpha][beta],
                                        );
                            }
                        }
                        matrix[(a * n + i) * m + b * n + j] =
                            finite(value, NonFiniteOrigin::Intermediate)?;
                    }
                }
            }
        }
        Ok(true)
    }
}
