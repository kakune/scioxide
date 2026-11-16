//! Local Jacobians and the chain rule for the block collocation matrix.
use super::*;
use crate::stiff::jacobian::{Differences, group};
use scioxide_differentiate::DifferenceScheme;

pub(super) struct AssemblyWorkspace<T: NumericScalar> {
    left: Vec<T>,
    right: Vec<T>,
    middle: Vec<T>,
    boundary: Vec<T>,
    row: Vec<T>,
    input: Vec<T>,
    floors: Vec<T::Real>,
    rhs_difference: Differences<T>,
    boundary_difference: Differences<T>,
    scratch: Vec<T>,
}
impl<T: NumericScalar> AssemblyWorkspace<T> {
    pub(super) fn new(n: usize, k: usize) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("bvp_local_jacobian");
        let block = checked_mul(n, checked_add(n, k, q)?, q)?;
        let width = checked_add(checked_mul(2, n, q)?, k, q)?;
        Ok(Self {
            left: try_filled(block, T::ZERO, q)?,
            right: try_filled(block, T::ZERO, q)?,
            middle: try_filled(block, T::ZERO, q)?,
            boundary: try_filled(checked_mul(n + k, width, q)?, T::ZERO, q)?,
            row: try_filled(width, T::ZERO, q)?,
            input: try_filled(width, T::ZERO, q)?,
            floors: try_filled(width, T::Real::ZERO, q)?,
            rhs_difference: Differences::rectangular(n + k, n)?,
            boundary_difference: Differences::rectangular(width, n + k)?,
            scratch: try_filled(n * 2, T::ZERO, q)?,
        })
    }
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Collocation<'_, '_, T, P> {
    /// Writes the block rows of the collocation Jacobian, in row order, to dense or
    /// sparse storage. The residual sweep supplies all RHS values and midpoint states,
    /// so assembly does not evaluate them again.
    pub(super) fn assemble(
        &self,
        point: &[T],
        units: &[T::Real],
        w: &mut AssemblyWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
        mut write: impl FnMut(usize, &[T]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let (n, k) = self.dimensions()?;
        let m = self.mesh.len();
        let cache = self.cached()?;
        let p = &point[m * n..m * n + k];
        let width = n + k;
        let sixth = T::ONE / T::from_real(r(6.)?);
        let two = T::from_real(r(2.)?);
        let four = T::from_real(r(4.)?);
        self.local_jacobian(
            self.mesh[0],
            &point[..n],
            p,
            &cache.rhs[..n],
            &units[..n],
            &units[m * n..],
            &mut w.left,
            &mut w.input,
            &mut w.floors,
            &mut w.rhs_difference,
            &mut w.scratch,
            context,
        )?;
        for (i, interval) in self.mesh.windows(2).enumerate() {
            let h = T::from_real(interval[1] - interval[0]);
            self.local_jacobian(
                interval[1],
                &point[(i + 1) * n..(i + 2) * n],
                p,
                &cache.rhs[(i + 1) * n..(i + 2) * n],
                &units[(i + 1) * n..(i + 2) * n],
                &units[m * n..],
                &mut w.right,
                &mut w.input,
                &mut w.floors,
                &mut w.rhs_difference,
                &mut w.scratch,
                context,
            )?;
            self.local_jacobian(
                interval[0] + (interval[1] - interval[0]) * T::Real::HALF,
                &cache.middle[i * n..(i + 1) * n],
                p,
                &cache.middle_rhs[i * n..(i + 1) * n],
                &units[i * n..(i + 1) * n],
                &units[m * n..],
                &mut w.middle,
                &mut w.input,
                &mut w.floors,
                &mut w.rhs_difference,
                &mut w.scratch,
                context,
            )?;
            let twelfth_h = h / T::from_real(r(12.)?);
            for row in 0..n {
                for column in 0..n {
                    let identity = if row == column { T::ONE / h } else { T::ZERO };
                    let (mut left_product, mut right_product) = (T::ZERO, T::ZERO);
                    for inner in 0..n {
                        let jm = w.middle[row * width + inner];
                        left_product = left_product + jm * w.left[inner * width + column];
                        right_product = right_product + jm * w.right[inner * width + column];
                    }
                    let midpoint = two * w.middle[row * width + column];
                    w.row[column] = -identity
                        - sixth * (w.left[row * width + column] + midpoint)
                        - twelfth_h * left_product;
                    w.row[n + column] = identity
                        - sixth * (w.right[row * width + column] + midpoint)
                        + twelfth_h * right_product;
                }
                for parameter in 0..k {
                    let mut product = T::ZERO;
                    for inner in 0..n {
                        product = product
                            + w.middle[row * width + inner]
                                * (w.left[inner * width + n + parameter]
                                    - w.right[inner * width + n + parameter]);
                    }
                    w.row[2 * n + parameter] = -sixth
                        * (w.left[row * width + n + parameter]
                            + w.right[row * width + n + parameter]
                            + four * w.middle[row * width + n + parameter])
                        - twelfth_h * product;
                }
                finite(&w.row, NonFiniteOrigin::Intermediate)?;
                write(i * n + row, &w.row)?;
            }
            core::mem::swap(&mut w.left, &mut w.right);
        }
        let b = n + k;
        let columns = 2 * n + k;
        w.boundary.fill(T::from_real(T::Real::NAN));
        if let Some(jacobian) = self.problem.jacobian {
            let (a, rest) = w.boundary.split_at_mut(b * n);
            let (z, p_out) = rest.split_at_mut(b * n);
            admit(context, EvaluationRole::Jacobian)?;
            (jacobian.boundary)(
                self.problem.system,
                &point[..n],
                &point[(m - 1) * n..m * n],
                p,
                a,
                z,
                p_out,
                context,
            )?;
            finite(&w.boundary, NonFiniteOrigin::Callback)?;
            for row in 0..b {
                w.row[..n].copy_from_slice(&w.boundary[row * n..(row + 1) * n]);
                w.row[n..2 * n]
                    .copy_from_slice(&w.boundary[b * n + row * n..b * n + (row + 1) * n]);
                w.row[2 * n..]
                    .copy_from_slice(&w.boundary[2 * b * n + row * k..2 * b * n + (row + 1) * k]);
                write((m - 1) * n + row, &w.row)?;
            }
        } else {
            w.input[..n].copy_from_slice(&point[..n]);
            w.input[n..2 * n].copy_from_slice(&point[(m - 1) * n..m * n]);
            w.input[2 * n..].copy_from_slice(p);
            w.floors[..n].copy_from_slice(&units[..n]);
            w.floors[n..2 * n].copy_from_slice(&units[(m - 1) * n..m * n]);
            w.floors[2 * n..].copy_from_slice(&units[m * n..]);
            for column in 0..columns {
                group(
                    &w.input,
                    &cache.residual[(m - 1) * n..],
                    &w.floors,
                    DifferenceScheme::Forward,
                    core::slice::from_ref(&column),
                    if column < 2 * n {
                        0..2 * n
                    } else {
                        2 * n..columns
                    },
                    &mut w.boundary_difference,
                    context,
                    |input, output, _, cx| {
                        admit(cx, EvaluationRole::Constraint)?;
                        output.as_mut_slice().fill(T::from_real(T::Real::NAN));
                        self.problem.system.boundary(
                            &input[..n],
                            &input[n..2 * n],
                            &input[2 * n..],
                            output.as_mut_slice(),
                            cx,
                        )?;
                        finite(output.as_slice(), NonFiniteOrigin::Callback)
                    },
                    |column, numerator, offset, retry| {
                        for (row, &value) in numerator.iter().enumerate() {
                            let entry = &mut w.boundary[row * columns + column];
                            if !retry || *entry == T::ZERO {
                                *entry = value / T::from_real(offset);
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            finite(&w.boundary, NonFiniteOrigin::Intermediate)?;
            for (row, values) in w.boundary.chunks_exact(columns).enumerate() {
                write((m - 1) * n + row, values)?;
            }
        }
        Ok(())
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn local_jacobian(
        &self,
        x: T::Real,
        y: &[T],
        p: &[T],
        base: &[T],
        y_units: &[T::Real],
        p_units: &[T::Real],
        out: &mut [T],
        input: &mut [T],
        floors: &mut [T::Real],
        differences: &mut Differences<T>,
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = y.len();
        let k = p.len();
        let width = n + k;
        if let Some(jacobian) = self.problem.jacobian {
            let (jy, jp) = out.split_at_mut(n * n);
            jy.fill(T::from_real(T::Real::NAN));
            jp.fill(T::from_real(T::Real::NAN));
            admit(context, EvaluationRole::Jacobian)?;
            (jacobian.rhs)(self.problem.system, x, y, p, jy, jp, context)?;
            finite(out, NonFiniteOrigin::Callback)?;
            // Convert the provider's separate state/parameter blocks to row-major rows.
            if k != 0 {
                // Rotate each parameter row behind its state row, retaining row-major order.
                for row in 0..n {
                    let start = row * width + n;
                    let source = n * n + row * k;
                    out[start..source + k].rotate_right(k);
                }
            }
            if self.plan.singular.is_none() {
                return Ok(());
            }
            if let Some(s) = &self.plan.singular {
                if x == self.mesh[0] {
                    for column in 0..width {
                        for (row, value) in scratch[..n].iter_mut().enumerate() {
                            *value = out[row * width + column];
                        }
                        let (src, dst) = scratch.split_at_mut(n);
                        mat_apply(&s.derivative, src, dst);
                        for (row, &value) in dst.iter().enumerate() {
                            out[row * width + column] = value;
                        }
                    }
                } else {
                    let distance = T::from_real(x - self.mesh[0]);
                    for row in 0..n {
                        for column in 0..n {
                            out[row * width + column] = out[row * width + column]
                                + s.matrix.as_slice()[row * n + column] / distance;
                        }
                    }
                }
            }
        } else {
            input[..n].copy_from_slice(y);
            input[n..width].copy_from_slice(p);
            floors[..n].copy_from_slice(y_units);
            floors[n..width].copy_from_slice(p_units);
            for column in 0..width {
                group(
                    &input[..width],
                    base,
                    &floors[..width],
                    DifferenceScheme::Forward,
                    core::slice::from_ref(&column),
                    if column < n { 0..n } else { n..width },
                    differences,
                    context,
                    |input, output, _, cx| {
                        self.rhs(
                            x,
                            &input[..n],
                            &input[n..],
                            output.as_mut_slice(),
                            &mut scratch[..n],
                            cx,
                        )
                    },
                    |column, numerator, offset, retry| {
                        for (row, &value) in numerator.iter().enumerate() {
                            let entry = &mut out[row * width + column];
                            if !retry || *entry == T::ZERO {
                                *entry = value / T::from_real(offset);
                            }
                        }
                        Ok(())
                    },
                )?;
            }
        }
        finite(out, NonFiniteOrigin::Intermediate)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::needless_range_loop)]
mod tests {
    use super::*;
    use scioxide_core::{JacobianOperator, ResidualOperator};
    struct Polynomial;
    fn v<T: NumericScalar>(x: f64) -> T {
        T::from_real(T::Real::try_from_f64_lossy(x).unwrap())
    }
    impl<T: NumericScalar> BvpSystem<T> for Polynomial {
        fn dimension(&self) -> usize {
            2
        }
        fn parameter_dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: T::Real,
            y: &[T],
            p: &[T],
            f: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            f.copy_from_slice(&[y[0] * y[0] + y[1] + p[0], y[0] * y[1] + p[0] * p[0]]);
            Ok(())
        }
        fn boundary(
            &self,
            a: &[T],
            b: &[T],
            p: &[T],
            out: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[a[0] + p[0] * p[0], a[1] + b[1], b[0] * p[0]]);
            Ok(())
        }
    }
    impl<T: NumericScalar> BvpJacobian<T> for Polynomial {
        fn rhs_jacobian(
            &self,
            _: T::Real,
            y: &[T],
            p: &[T],
            j: &mut [T],
            jp: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            j.copy_from_slice(&[v::<T>(2.) * y[0], T::ONE, y[1], y[0]]);
            jp.copy_from_slice(&[T::ONE, v::<T>(2.) * p[0]]);
            Ok(())
        }
        fn boundary_jacobian(
            &self,
            _: &[T],
            b: &[T],
            p: &[T],
            a: &mut [T],
            z: &mut [T],
            jp: &mut [T],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            a.copy_from_slice(&[T::ONE, T::ZERO, T::ZERO, T::ONE, T::ZERO, T::ZERO]);
            z.copy_from_slice(&[T::ZERO, T::ZERO, T::ZERO, T::ONE, p[0], T::ZERO]);
            jp.copy_from_slice(&[v::<T>(2.) * p[0], T::ZERO, b[0]]);
            Ok(())
        }
    }
    fn qualify<T: NumericScalar>() {
        let mesh: Vec<_> = [0., 0.2, 0.7, 1.]
            .map(|x| v::<T>(x).real_component())
            .to_vec();
        let point: Vec<T> = [0.1, 0., 0.3, 0.2, 0.4, -0.1, 0.2, 0.5, 0.7]
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                T::from_components(
                    v::<T>(x).real_component(),
                    if i == 1 {
                        T::Real::ZERO
                    } else {
                        v::<T>(0.1).real_component()
                    },
                )
            })
            .collect();
        let direction: Vec<T> = (0..9).map(|i| v::<T>(0.1 * f64::from(i + 1))).collect();
        let units = vec![T::Real::ONE; 9];
        let mut assembled = Vec::new();
        for singular in [false, true] {
            for analytic in [true, false] {
                let mut plan = BvpPlan::new(
                    2,
                    1,
                    &mesh,
                    BvpOptions::new()
                        .with_max_nodes(16)
                        .with_max_workspace_bytes(1 << 24),
                )
                .unwrap();
                if singular {
                    plan = plan
                        .with_singular(
                            Matrix::from_vec(2, 2, vec![T::ZERO, T::ZERO, T::ZERO, v::<T>(-1.)])
                                .unwrap()
                                .view(),
                        )
                        .unwrap();
                }
                let problem = if analytic {
                    BvpProblem::analytic(&Polynomial)
                } else {
                    BvpProblem::numerical(&Polynomial)
                };
                let cache = RefCell::new(collocation::EvaluationCache::new());
                cache.borrow_mut().resize(2, 1, 4, analytic).unwrap();
                let c = Collocation {
                    plan: &plan,
                    problem: &problem,
                    mesh: &mesh,
                    cache: Some(&cache),
                    direct: None,
                };
                let mut scratch = vec![T::ZERO; c.residual_workspace_len().unwrap()];
                let mut residual = vec![T::ZERO; 9];
                let mut matrix = vec![T::ZERO; 81];
                let mut w = AssemblyWorkspace::new(2, 1).unwrap();
                SolveScope::from(ExecutionContext::sequential())
                    .run(
                        &mut scioxide_core::SolveState::new(),
                        SolveIdentity::new(
                            ProblemToken::new(0),
                            scioxide_core::algorithm_id!("test", "bvp-assembly", 1),
                            9,
                            0,
                        ),
                        SolveBudgets::unbounded(),
                        |cx| {
                            c.residual_into_scoped(&point, &mut residual, &mut scratch, cx)?;
                            c.assemble(&point, &units, &mut w, cx, |row, values| {
                                let a = if row < 6 { row / 2 * 2 } else { 0 };
                                let b = if row < 6 { a + 2 } else { 6 };
                                matrix[row * 9 + a..row * 9 + a + 2].copy_from_slice(&values[..2]);
                                matrix[row * 9 + b..row * 9 + b + 2].copy_from_slice(&values[2..4]);
                                matrix[row * 9 + 8] = values[4];
                                Ok(())
                            })
                        },
                    )
                    .unwrap();
                let h = T::Real::CBRT_EPSILON;
                let plus: Vec<_> = point
                    .iter()
                    .zip(&direction)
                    .map(|(&x, &d)| x + T::from_real(h) * d)
                    .collect();
                let minus: Vec<_> = point
                    .iter()
                    .zip(&direction)
                    .map(|(&x, &d)| x - T::from_real(h) * d)
                    .collect();
                let mut fp = vec![T::ZERO; 9];
                let mut fm = vec![T::ZERO; 9];
                c.residual_into(&plus, &mut fp, &mut scratch).unwrap();
                c.residual_into(&minus, &mut fm, &mut scratch).unwrap();
                for row in 0..9 {
                    let action = matrix[row * 9..(row + 1) * 9]
                        .iter()
                        .zip(&direction)
                        .fold(T::ZERO, |s, (&a, &d)| s + a * d);
                    let numerical = (fp[row] - fm[row]) / T::from_real(h + h);
                    assert!(
                        (action - numerical).abs()
                            <= v::<T>(16.).abs()
                                * T::Real::SQRT_EPSILON
                                * (T::Real::ONE + numerical.abs()),
                        "analytic={analytic}, singular={singular}, row={row}"
                    );
                }
                if analytic {
                    // The independent Krylov product uses the local chain rule directly.
                    let mut product = vec![T::ZERO; 9];
                    let mut work = vec![T::ZERO; 12 * 2 + 3 * 5];
                    // Call the matrix-free kernel explicitly, independent of dense/CSR assembly.
                    collocation_product(&c, &point, &direction, &mut product, &mut work);
                    for row in 0..9 {
                        let action = matrix[row * 9..(row + 1) * 9]
                            .iter()
                            .zip(&direction)
                            .fold(T::ZERO, |s, (&a, &d)| s + a * d);
                        assert!(
                            (action - product[row]).abs()
                                <= v::<T>(64.).abs()
                                    * T::Real::EPSILON
                                    * (T::Real::ONE + product[row].abs())
                        );
                    }
                    assembled = matrix;
                } else {
                    for (&a, &b) in matrix.iter().zip(&assembled) {
                        assert!(
                            (a - b).abs()
                                <= v::<T>(32.).abs()
                                    * T::Real::SQRT_EPSILON
                                    * (T::Real::ONE + b.abs())
                        );
                    }
                }
            }
        }
    }
    fn collocation_product<T: NumericScalar>(
        c: &Collocation<'_, '_, T, Polynomial>,
        point: &[T],
        direction: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) {
        c.jvp_into(point, direction, out, scratch).unwrap();
    }
    #[test]
    fn blocks_match_independent_products_and_residual_differences() {
        qualify::<f32>();
        qualify::<f64>();
        qualify::<scioxide_core::Complex32>();
        qualify::<scioxide_core::Complex64>();
    }
    struct ParameterExponential;
    impl BvpSystem<f64> for ParameterExponential {
        fn dimension(&self) -> usize {
            2
        }
        fn parameter_dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            p: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[p[0].exp(), 0.]);
            Ok(())
        }
        fn boundary(
            &self,
            a: &[f64],
            b: &[f64],
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[a[0], b[0], a[1]]);
            Ok(())
        }
    }
    #[test]
    fn unresolved_zero_rows_do_not_retry_parameters_in_state_units() {
        let mesh = [0., 0.3, 1.];
        let plan = BvpPlan::new(
            ParameterExponential.dimension(),
            ParameterExponential.parameter_dimension(),
            &mesh,
            BvpOptions::new()
                .with_max_nodes(16)
                .with_max_workspace_bytes(1 << 24),
        )
        .unwrap();
        let problem = BvpProblem::numerical(&ParameterExponential);
        let cache = RefCell::new(collocation::EvaluationCache::new());
        cache.borrow_mut().resize(2, 1, 3, false).unwrap();
        let c = Collocation {
            plan: &plan,
            problem: &problem,
            mesh: &mesh,
            cache: Some(&cache),
            direct: None,
        };
        let point = [1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 0.];
        let units = [1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e-3];
        SolveScope::from(ExecutionContext::sequential())
            .run(
                &mut scioxide_core::SolveState::new(),
                SolveIdentity::new(
                    ProblemToken::new(0),
                    scioxide_core::algorithm_id!("test", "parameter-units", 1),
                    7,
                    0,
                ),
                SolveBudgets::unbounded(),
                |cx| {
                    c.ensure_evaluated(&point, &mut [0.; 10], cx)?;
                    c.assemble(
                        &point,
                        &units,
                        &mut AssemblyWorkspace::new(2, 1)?,
                        cx,
                        |row, values| {
                            if row < 4 {
                                assert!(
                                    (values[4] - if row % 2 == 0 { -1. } else { 0. }).abs() < 1e-4
                                );
                            }
                            Ok(())
                        },
                    )
                },
            )
            .unwrap();
    }
}
