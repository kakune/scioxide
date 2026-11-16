use super::*;

#[derive(Clone)]
pub(crate) struct ReferenceBasis<R: RealScalar, const D: usize> {
    order: SimplexFemOrder,
    scalar: core::marker::PhantomData<R>,
}
#[derive(Clone, Copy)]
pub(crate) struct BasisSample<R: RealScalar, const D: usize> {
    pub(crate) values: [R; MAX_NODES],
    pub(crate) gradients: [[R; D]; MAX_NODES],
}
impl<R: RealScalar, const D: usize> ReferenceBasis<R, D> {
    pub(crate) fn new(order: SimplexFemOrder) -> Result<Self, SciOxideError> {
        if !(1..=3).contains(&D) {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "simplex dimension must be one through three",
            ));
        }
        Ok(Self {
            order,
            scalar: core::marker::PhantomData,
        })
    }
    /// Exact constant reference Hessians of P1/P2 Lagrange polynomials.
    pub(crate) fn hessians(&self) -> [[[R; D]; D]; MAX_NODES] {
        let mut out = [[[R::ZERO; D]; D]; MAX_NODES];
        if self.order == SimplexFemOrder::P1 {
            return out;
        }
        let derivative = |node: usize, axis: usize| {
            if node == 0 {
                -R::ONE
            } else if node == axis + 1 {
                R::ONE
            } else {
                R::ZERO
            }
        };
        let four = R::ONE.scalbn(2);
        for (i, hessian) in out.iter_mut().enumerate().take(D + 1) {
            for (a, row) in hessian.iter_mut().enumerate() {
                for (b, value) in row.iter_mut().enumerate() {
                    *value = four * derivative(i, a) * derivative(i, b);
                }
            }
        }
        let mut node = D + 1;
        for i in 0..=D {
            for j in i + 1..=D {
                for (a, row) in out[node].iter_mut().enumerate() {
                    for (b, value) in row.iter_mut().enumerate() {
                        *value = four
                            * (derivative(i, a) * derivative(j, b)
                                + derivative(j, a) * derivative(i, b));
                    }
                }
                node += 1;
            }
        }
        out
    }
    pub(crate) fn sample(&self, point: [R; D]) -> Result<BasisSample<R, D>, SciOxideError> {
        if point.iter().any(|x| !x.is_finite() || *x < R::ZERO)
            || point.iter().copied().fold(R::ZERO, |s, x| s + x)
                > R::ONE + R::EPSILON * R::ONE.scalbn(3)
        {
            return Err(invalid(
                scioxide_core::argument!("reference"),
                "point lies outside reference simplex",
            ));
        }
        let mut result = BasisSample {
            values: [R::ZERO; MAX_NODES],
            gradients: [[R::ZERO; D]; MAX_NODES],
        };
        let mut lambda = [R::ZERO; 4];
        lambda[0] = R::ONE - point.iter().copied().fold(R::ZERO, |a, b| a + b);
        lambda[1..=D].copy_from_slice(&point);
        let derivative = |node: usize, axis: usize| {
            if node == 0 {
                -R::ONE
            } else if node == axis + 1 {
                R::ONE
            } else {
                R::ZERO
            }
        };
        for (i, &lambda_i) in lambda.iter().enumerate().take(D + 1) {
            result.values[i] = if self.order == SimplexFemOrder::P1 {
                lambda_i
            } else {
                lambda_i * (R::TWO * lambda_i - R::ONE)
            };
            for axis in 0..D {
                result.gradients[i][axis] = derivative(i, axis)
                    * if self.order == SimplexFemOrder::P1 {
                        R::ONE
                    } else {
                        R::ONE.scalbn(2) * lambda_i - R::ONE
                    };
            }
        }
        if self.order == SimplexFemOrder::P2 {
            let mut node = D + 1;
            for i in 0..=D {
                for j in i + 1..=D {
                    result.values[node] = R::ONE.scalbn(2) * lambda[i] * lambda[j];
                    for axis in 0..D {
                        result.gradients[node][axis] = R::ONE.scalbn(2)
                            * (derivative(i, axis) * lambda[j] + lambda[i] * derivative(j, axis));
                    }
                    node += 1;
                }
            }
        }
        Ok(result)
    }
}

pub(super) fn reference_node<R: RealScalar, const D: usize>(node: usize) -> [R; D] {
    let mut point = [R::ZERO; D];
    if node <= D {
        if node > 0 {
            point[node - 1] = R::ONE;
        }
        return point;
    }
    let mut next = D + 1;
    for i in 0..=D {
        for j in i + 1..=D {
            if next == node {
                if i > 0 {
                    point[i - 1] = R::HALF;
                }
                point[j - 1] = R::HALF;
                return point;
            }
            next += 1;
        }
    }
    point
}

/// Mapped coordinates of finite mesh points overflow rather than fail input checks.
fn geometry<R: RealScalar>(value: R) -> Result<R, SciOxideError> {
    crate::discretization::representable(value, "element geometry is unrepresentable")
}

/// Multiply ordinary geometry scalars directly, retaining the range-safe path.
pub(crate) fn product<R: RealScalar>(a: R, b: R) -> Result<R, SciOxideError> {
    let value = a * b;
    if value.is_normal() || a == R::ZERO || b == R::ZERO {
        Ok(value)
    } else {
        scioxide_core::checked_float_product(a, b)
    }
}

/// Shared isoparametric contraction and scaled inverse for prepared element kernels.
pub(crate) fn map_coordinates<R: RealScalar, const D: usize>(
    coordinates: &[[R; D]],
    cutoff: R,
    basis: &BasisSample<R, D>,
    allow_reversed: bool,
) -> Result<SimplexFemMappedPoint<R, D>, SciOxideError> {
    let origin = coordinates[0];
    let mut point = origin;
    let mut jacobian = [[R::ZERO; D]; D];
    for (node, coordinate) in coordinates.iter().enumerate().skip(1) {
        for axis in 0..D {
            let delta = geometry(coordinate[axis] - origin[axis])?;
            point[axis] = geometry(point[axis] + basis.values[node] * delta)?;
            for (r, derivative) in jacobian[axis].iter_mut().enumerate() {
                *derivative = geometry(*derivative + basis.gradients[node][r] * delta)?;
            }
        }
    }
    let scale = jacobian
        .iter()
        .flatten()
        .copied()
        .fold(R::ZERO, |s, x| s.max(x.abs()));
    if scale == R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "singular geometric Jacobian",
        ));
    }
    let mut a = [[R::ZERO; 3]; 3];
    for i in 0..D {
        for j in 0..D {
            a[i][j] = jacobian[i][j] / scale;
        }
    }
    let mut adj = [[R::ZERO; 3]; 3];
    if D == 1 {
        adj[0][0] = R::ONE;
    } else if D == 2 {
        adj[0][0] = a[1][1];
        adj[0][1] = -a[0][1];
        adj[1][0] = -a[1][0];
        adj[1][1] = a[0][0];
    } else {
        for i in 0..3 {
            for j in 0..3 {
                adj[j][i] = a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3]
                    - a[(i + 1) % 3][(j + 2) % 3] * a[(i + 2) % 3][(j + 1) % 3];
            }
        }
    }
    let det = (0..D).fold(R::ZERO, |s, j| s + a[0][j] * adj[j][0]);
    if !det.is_finite() || det == R::ZERO || (!allow_reversed && det < R::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "inverted or singular geometric Jacobian",
        ));
    }
    let norm = (0..D)
        .map(|i| (0..D).fold(R::ZERO, |s, j| s + a[i][j].abs()))
        .fold(R::ZERO, |a, b| a.max(b));
    let adj_norm = (0..D)
        .map(|i| (0..D).fold(R::ZERO, |s, j| s + adj[i][j].abs()))
        .fold(R::ZERO, |a, b| a.max(b));
    let reciprocal_condition = finite(
        det.abs() / (norm * adj_norm),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    if reciprocal_condition <= cutoff {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "geometric Jacobian is numerically unusable",
        ));
    }
    let mut inverse_jacobian = [[R::ZERO; D]; D];
    for i in 0..D {
        for j in 0..D {
            let quotient = adj[i][j] / det;
            let native = quotient / scale;
            inverse_jacobian[i][j] =
                if adj[i][j] == R::ZERO || (quotient.is_normal() && native.is_normal()) {
                    native
                } else {
                    scioxide_core::Scaled::from_value(adj[i][j])?
                        .div(scioxide_core::Scaled::from_value(det)?)?
                        .div(scioxide_core::Scaled::from_value(scale)?)?
                        .round()?
                };
        }
    }
    let mut native = det.abs();
    let mut ordinary = native.is_normal();
    for _ in 0..D {
        native *= scale;
        ordinary &= native.is_normal();
    }
    let determinant = if ordinary {
        native
    } else {
        let mut extended = scioxide_core::Scaled::from_value(det.abs())?;
        for _ in 0..D {
            extended = extended.mul(scioxide_core::Scaled::from_value(scale)?)?;
        }
        extended.round()?
    };
    if determinant == R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "geometric volume underflows",
        ));
    }
    Ok(SimplexFemMappedPoint {
        point,
        jacobian,
        inverse_jacobian,
        determinant,
        reciprocal_condition,
    })
}

/// Physical basis gradients `J⁻ᵀ∇̂φᵢ` of the first `local` reference gradients.
pub(crate) fn physical_gradients<R: RealScalar, const D: usize>(
    map: &SimplexFemMappedPoint<R, D>,
    reference: &BasisSample<R, D>,
    local: usize,
) -> Result<[[R; D]; MAX_NODES], SciOxideError> {
    let mut gradients = [[R::ZERO; D]; MAX_NODES];
    for (gradient, reference) in gradients.iter_mut().zip(&reference.gradients).take(local) {
        for (axis, derivative) in gradient.iter_mut().enumerate() {
            *derivative = finite(
                (0..D).fold(R::ZERO, |s, j| {
                    s + map.inverse_jacobian[j][axis] * reference[j]
                }),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
    }
    Ok(gradients)
}

/// Reference points and weights of `rule`, whose dimension is at most `D`:
/// barycentric coordinates `λ₁…λₖ` become the leading reference coordinates.
pub(crate) fn rule_nodes<R: RealScalar, const D: usize>(
    rule: &SimplexRule<R>,
) -> impl Iterator<Item = ([R; D], R)> + '_ {
    let dimension = rule.dimension();
    rule.barycentric_nodes()
        .chunks_exact(dimension + 1)
        .zip(rule.weights())
        .map(move |(bary, &weight)| {
            let mut point = [R::ZERO; D];
            point[..dimension].copy_from_slice(&bary[1..]);
            (point, weight)
        })
}

/// Reference coordinates of a point given by barycentric coordinates on the
/// facet opposite local vertex `opposite`, listed in increasing vertex order.
pub(crate) fn facet_point<R: RealScalar, const D: usize>(opposite: usize, facet: &[R]) -> [R; D] {
    let mut point = [R::ZERO; D];
    for (&lambda, vertex) in facet.iter().zip((0..=D).filter(|&v| v != opposite)) {
        if vertex > 0 {
            point[vertex - 1] = lambda;
        }
    }
    point
}

/// Unnormalized outward reference normal of the facet opposite local vertex `opposite`.
pub(crate) fn facet_normal<R: RealScalar, const D: usize>(opposite: usize) -> [R; D] {
    let mut normal = [R::ZERO; D];
    if opposite == 0 {
        normal.fill(R::ONE);
    } else {
        normal[opposite - 1] = -R::ONE;
    }
    normal
}

/// Unit physical normal `J⁻ᵀn̂ / |J⁻ᵀn̂|` of a reference normal `n̂`, and the
/// surface scale `det J · |J⁻ᵀn̂|` that maps reference to physical facet measure.
pub(crate) fn mapped_normal<R: RealScalar, const D: usize>(
    map: &SimplexFemMappedPoint<R, D>,
    reference: [R; D],
) -> Result<([R; D], R), SciOxideError> {
    let mut n = [R::ZERO; D];
    for (axis, component) in n.iter_mut().enumerate() {
        *component = (0..D).fold(R::ZERO, |s, j| {
            s + map.inverse_jacobian[j][axis] * reference[j]
        });
    }
    let scale = n.iter().copied().fold(R::ZERO, |s, x| s.max(x.abs()));
    if scale == R::ZERO || !scale.is_finite() {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "unusable facet normal",
        ));
    }
    let norm = n
        .iter()
        .fold(R::ZERO, |s, x| s + (*x / scale) * (*x / scale))
        .sqrt();
    for x in &mut n {
        *x = (*x / scale) / norm;
    }
    let first = map.determinant * scale;
    let measure = if first.is_normal() && (first * norm).is_normal() {
        first * norm
    } else {
        scioxide_core::Scaled::from_value(map.determinant)?
            .mul(scioxide_core::Scaled::from_value(scale)?)?
            .mul(scioxide_core::Scaled::from_value(norm)?)?
            .round()?
    };
    Ok((n, measure))
}

/// Unit outward normal and surface scale of the facet opposite local vertex `opposite`.
pub(crate) fn normal<R: RealScalar, const D: usize>(
    map: &SimplexFemMappedPoint<R, D>,
    opposite: usize,
) -> Result<([R; D], R), SciOxideError> {
    mapped_normal(map, facet_normal(opposite))
}

/// Pure local weak-form contraction shared by conforming and adaptive assembly.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps operands, storage and execution controls explicit"
)]
pub(crate) fn accumulate_forms<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    local: usize,
    c: usize,
    weight: R,
    values: &[R; MAX_NODES],
    gradients: &[[R; D]; MAX_NODES],
    diffusion: &[V],
    reaction: &[V],
    matrix: &mut [V],
    mass_out: &mut [R],
) -> Result<(), SciOxideError> {
    let n = local * c;
    let kd = c * D;
    for i in 0..local {
        for j in 0..local {
            mass_out[i * local + j] = finite(
                mass_out[i * local + j] + weight * values[i] * values[j],
                NonFiniteOrigin::Intermediate,
            )?;
        }
    }
    for a in 0..c {
        for b in 0..c {
            for j in 0..local {
                let transformed = weighted_gradient(diffusion, c, a, b, weight, &gradients[j]);
                for i in 0..local {
                    let mass = weight * values[i] * values[j];
                    let original = || {
                        let mut value = reaction[a * c + b] * V::from_real(mass);
                        for axis in 0..D {
                            for direction in 0..D {
                                value = value
                                    + diffusion[(a * D + axis) * kd + b * D + direction]
                                        * V::from_real(
                                            weight * gradients[i][axis] * gradients[j][direction],
                                        );
                            }
                        }
                        value
                    };
                    let value = transformed
                        .map(|g| {
                            g.iter()
                                .zip(&gradients[i])
                                .fold(reaction[a * c + b] * V::from_real(mass), |s, (&x, &y)| {
                                    s + x * V::from_real(y)
                                })
                        })
                        .filter(|v| v.has_finite_components())
                        .unwrap_or_else(original);
                    let slot = (a * local + i) * n + b * local + j;
                    matrix[slot] = crate::discretization::entry(matrix[slot] + value)?;
                }
            }
        }
    }
    Ok(())
}

/// Precontract a weighted physical gradient only when its intermediate range is safe.
fn weighted_gradient<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    diffusion: &[V],
    components: usize,
    a: usize,
    b: usize,
    weight: R,
    gradient: &[R; D],
) -> Option<[V; D]> {
    let mut transformed = [V::ZERO; D];
    for (axis, value) in transformed.iter_mut().enumerate() {
        for (direction, &g) in gradient.iter().enumerate() {
            let coefficient = diffusion[(a * D + axis) * (components * D) + b * D + direction];
            if coefficient == V::ZERO || g == R::ZERO || weight == R::ZERO {
                continue;
            }
            let weighted = weight * g;
            if !weighted.is_normal() {
                return None;
            }
            let term = coefficient * V::from_real(weighted);
            if !term.abs().is_normal() {
                return None;
            }
            *value = *value + term;
        }
        if !value.has_finite_components() {
            return None;
        }
    }
    Some(transformed)
}
