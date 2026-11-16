//! Collocation residuals, Jacobian products and scaling over one active mesh.
use super::*;
use scioxide_core::{JacobianOperator, ResidualOperator, SolveState};
use scioxide_linalg::{IterativeSolveReport, Vector};
use scioxide_rootfind::{NewtonKrylovProblem, NonlinearLinearSolver, NonlinearScaling};

/// Collocation equations of `plan` and `problem` on `mesh`.
pub(super) struct Collocation<'a, 'b, T: NumericScalar, P: BvpSystem<T> + ?Sized> {
    pub(super) plan: &'a BvpPlan<T>,
    pub(super) problem: &'a BvpProblem<'b, T, P>,
    pub(super) cache: Option<&'a RefCell<EvaluationCache<T>>>,
    pub(super) direct: Option<&'a DirectCorrection<T>>,
    pub(super) mesh: &'a [T::Real],
}

/// One successful residual evaluation, tagged by its full point. Trials invalidate
/// the cache before any callback; only a wholly finite evaluation becomes reusable.
///
/// Matrix-free analytic products also cache the local Jacobians of that point once:
/// row-major `[J_y | J_p]` blocks of every node, then of every midpoint, then the
/// boundary blocks `[A | B | P]`.
pub(super) struct EvaluationCache<T: NumericScalar> {
    pub(super) valid: bool,
    pub(super) point: Vec<T>,
    pub(super) rhs: Vec<T>,
    pub(super) middle: Vec<T>,
    pub(super) middle_rhs: Vec<T>,
    pub(super) quadrature_rhs: Vec<T>,
    pub(super) residual: Vec<T>,
    pub(super) jacobians_valid: bool,
    pub(super) jacobians: Vec<T>,
}
impl<T: NumericScalar> EvaluationCache<T> {
    pub(super) fn new() -> Self {
        Self {
            valid: false,
            point: Vec::new(),
            rhs: Vec::new(),
            middle: Vec::new(),
            middle_rhs: Vec::new(),
            quadrature_rhs: Vec::new(),
            residual: Vec::new(),
            jacobians_valid: false,
            jacobians: Vec::new(),
        }
    }
    /// Size the cache for `nodes`, with local Jacobian storage only when `products`.
    pub(super) fn resize(
        &mut self,
        n: usize,
        k: usize,
        nodes: usize,
        products: bool,
    ) -> Result<(), SciOxideError> {
        self.valid = false;
        self.jacobians_valid = false;
        super::resize(&mut self.point, nodes * n + k, T::ZERO)?;
        super::resize(&mut self.rhs, nodes * n, T::ZERO)?;
        super::resize(&mut self.middle, (nodes - 1) * n, T::ZERO)?;
        super::resize(&mut self.middle_rhs, (nodes - 1) * n, T::ZERO)?;
        super::resize(&mut self.quadrature_rhs, 2 * (nodes - 1) * n, T::ZERO)?;
        super::resize(&mut self.residual, nodes * n + k, T::ZERO)?;
        if products {
            super::resize(&mut self.jacobians, jacobian_len(n, k, nodes)?, T::ZERO)
        } else {
            self.jacobians = Vec::new();
            Ok(())
        }
    }
}

/// Scalars of the cached local Jacobians on `nodes >= 1` nodes: `2*nodes - 1` RHS
/// blocks of shape `(n, n+k)` and the boundary blocks of shape `(n+k, 2n+k)`.
pub(super) fn jacobian_len(n: usize, k: usize, nodes: usize) -> Result<usize, SciOxideError> {
    let quantity = scioxide_core::quantity!("jacobian_blocks");
    let width = checked_add(n, k, quantity)?;
    let points = checked_mul(nodes, 2, quantity)?.saturating_sub(1);
    checked_add(
        checked_mul(checked_mul(points, n, quantity)?, width, quantity)?,
        checked_mul(
            width,
            checked_add(checked_mul(n, 2, quantity)?, k, quantity)?,
            quantity,
        )?,
        quantity,
    )
}

/// Placeholder for the injected direct correction when only storage is sized. Nonlinear
/// requirements depend on the presence of an injected solver, never on its behaviour.
pub(super) struct Unprepared;
impl<T: NumericScalar> NonlinearLinearSolver<T> for Unprepared {
    fn solve_into(
        &self,
        _: &scioxide_rootfind::NonlinearLinearization<'_, T>,
        _: &Vector<T>,
        _: &mut Vector<T>,
        _: ResidualTolerance<T::Real>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<T::Real>, SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("self"),
            "requirement sizing cannot solve corrections",
        ))
    }
}

/// Runs `evaluate` outside a collocation solve, in its own sequential solve scope.
fn unscoped<R>(
    evaluate: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError> {
    let mut state = SolveState::new();
    SolveScope::from(ExecutionContext::sequential()).run(
        &mut state,
        SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "bvp-unscoped-evaluation", 1),
            0,
            0,
        ),
        SolveBudgets::unbounded(),
        evaluate,
    )
}

impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Collocation<'_, '_, T, P> {
    #[inline(always)]
    pub(super) fn dimensions(&self) -> Result<(usize, usize), SciOxideError> {
        let n = self.problem.system.dimension();
        let k = self.problem.system.parameter_dimension();
        if n != self.plan.layout.n || k != self.plan.layout.k {
            Err(invalid(
                scioxide_core::argument!("problem"),
                "provider dimensions changed after preparation",
            ))
        } else {
            Ok((n, k))
        }
    }
    /// The nonlinear problem of these equations, with an optional injected correction.
    pub(super) fn nonlinear_problem<'s>(
        &'s self,
        correction: Option<&'s dyn NonlinearLinearSolver<T>>,
    ) -> NewtonKrylovProblem<'s, T> {
        let mut problem = if self.problem.jacobian.is_some()
            || !matches!(self.plan.layout.backend, BvpBackend::Krylov { .. })
        {
            NewtonKrylovProblem::with_jacobian(self)
        } else {
            NewtonKrylovProblem::numerical(self)
        }
        .with_scaling(self);
        if let Some(correction) = correction {
            problem = problem.with_linear_solver(correction);
        }
        problem
    }
}

impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> NonlinearScaling<T> for Collocation<'_, '_, T, P> {
    fn scaling_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.residual_workspace_len()
    }
    fn scales_into(
        &self,
        point: &[T],
        residual: &mut [T::Real],
        variables: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let (n, _) = self.dimensions()?;
        let m = self.mesh.len();
        let cache = self.cached()?;
        let _ = (scratch, context);
        let absolute = self.plan.residual.absolute() * r(0.001)?;
        let relative = (self.plan.residual.relative() * r(0.001)?).max(T::Real::EPSILON * r(32.)?);
        let mut rhs_scale = T::Real::ZERO;
        for i in 0..m - 1 {
            for j in 0..n {
                let magnitude = cache.rhs[i * n + j]
                    .abs()
                    .max(cache.rhs[(i + 1) * n + j].abs());
                residual[i * n + j] = absolute + relative * magnitude;
                rhs_scale = rhs_scale.max(magnitude);
            }
        }
        residual[(m - 1) * n..].fill(self.plan.boundary.absolute() * r(0.001)?);
        // RHS over the span supplies initial units even for a zero state guess.
        // This also keeps numerical products above cancellation in large RHS terms.
        let initial_scale = (self.mesh[m - 1] - self.mesh[0]) * rhs_scale;
        for (i, (scale, &value)) in variables.iter_mut().zip(point).enumerate() {
            let floor = if i < m * n {
                initial_scale
            } else {
                T::Real::ZERO
            };
            *scale = absolute / relative + value.abs().max(floor);
        }
        Ok(())
    }
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Collocation<'_, '_, T, P> {
    /// Evaluates and counts the RHS including the singular term, and checks input and
    /// output for finite values.
    #[inline(always)]
    pub(super) fn rhs(
        &self,
        x: T::Real,
        y: &[T],
        p: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        finite(y, NonFiniteOrigin::Intermediate)?;
        finite(p, NonFiniteOrigin::Intermediate)?;
        admit(context, EvaluationRole::RightHandSide)?;
        out.fill(T::from_real(T::Real::NAN));
        self.problem.system.rhs(x, y, p, out, context)?;
        finite(out, NonFiniteOrigin::Callback)?;
        let n = y.len();
        if let Some(s) = &self.plan.singular {
            if x == self.mesh[0] {
                scratch.copy_from_slice(out);
                mat_apply(&s.derivative, scratch, out);
                for (i, &f) in scratch.iter().enumerate() {
                    let sy = s.matrix.as_slice()[i * n..(i + 1) * n]
                        .iter()
                        .zip(out.iter())
                        .fold(T::ZERO, |sum, (&a, &b)| sum + a * b);
                    let difference = (out[i] - sy - f).abs();
                    let threshold = self.plan.boundary.absolute() * (T::Real::ONE + f.abs());
                    if !difference.is_finite() || !threshold.is_finite() {
                        return Err(SciOxideError::non_finite(
                            NonFiniteOrigin::Intermediate,
                            "singular compatibility normalization overflow",
                        ));
                    }
                    if difference > threshold {
                        return Err(invalid(
                            scioxide_core::argument!("problem"),
                            "incompatible resonant singular-endpoint derivative",
                        ));
                    }
                }
            } else {
                mat_apply(&s.matrix, y, scratch);
                let distance = T::from_real(x - self.mesh[0]);
                out.iter_mut()
                    .zip(scratch)
                    .for_each(|(a, b)| *a = *a + *b / distance);
            }
        }
        finite(out, NonFiniteOrigin::Callback)
    }
    /// Borrow the successful current evaluation. Newton requests scaling/assembly
    /// only after evaluating that accepted point; assessment calls ensure_evaluated.
    pub(super) fn cached(&self) -> Result<core::cell::Ref<'_, EvaluationCache<T>>, SciOxideError> {
        let cache = self
            .cache
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "collocation cache is not prepared",
                )
            })?
            .borrow();
        if !cache.valid {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "collocation cache does not match the evaluated point",
            ));
        }
        Ok(cache)
    }
    pub(super) fn ensure_evaluated(
        &self,
        point: &[T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut cache = self
            .cache
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "collocation cache is not prepared",
                )
            })?
            .borrow_mut();
        if cache.valid && cache.point == point {
            return Ok(());
        }
        cache.valid = false;
        cache.jacobians_valid = false;
        let (n, k) = self.dimensions()?;
        let m = self.mesh.len();
        let p = &point[m * n..m * n + k];
        let (ym, work) = scratch.split_at_mut(n);
        let half = T::from_real(T::Real::HALF);
        let eight = T::from_real(r(8.)?);
        let six = T::from_real(r(6.)?);
        let four = T::from_real(r(4.)?);
        for (i, &x) in self.mesh.iter().enumerate() {
            self.rhs(
                x,
                &point[i * n..(i + 1) * n],
                p,
                &mut cache.rhs[i * n..(i + 1) * n],
                &mut work[..n],
                context,
            )?;
        }
        for (i, interval) in self.mesh.windows(2).enumerate() {
            let a = &point[i * n..(i + 1) * n];
            let b = &point[(i + 1) * n..(i + 2) * n];
            let h = T::from_real(interval[1] - interval[0]);
            for (j, y) in ym.iter_mut().enumerate() {
                *y = (a[j] + b[j]) * half
                    - h / eight * (cache.rhs[(i + 1) * n + j] - cache.rhs[i * n + j]);
            }
            cache.middle[i * n..(i + 1) * n].copy_from_slice(ym);
            self.rhs(
                interval[0] + (interval[1] - interval[0]) * T::Real::HALF,
                ym,
                p,
                &mut cache.middle_rhs[i * n..(i + 1) * n],
                &mut work[..n],
                context,
            )?;
            for j in 0..n {
                cache.residual[i * n + j] = (b[j] - a[j]) / h
                    - (cache.rhs[i * n + j]
                        + four * cache.middle_rhs[i * n + j]
                        + cache.rhs[(i + 1) * n + j])
                        / six;
            }
            let offset = r::<T::Real>(3. / 7.)?.sqrt() * T::Real::HALF;
            for (q, fraction) in [T::Real::HALF - offset, T::Real::HALF + offset]
                .into_iter()
                .enumerate()
            {
                hermite(
                    interval[1] - interval[0],
                    fraction,
                    a,
                    b,
                    &cache.rhs[i * n..(i + 1) * n],
                    &cache.rhs[(i + 1) * n..(i + 2) * n],
                    ym,
                    None,
                );
                self.rhs(
                    interval[0] + (interval[1] - interval[0]) * fraction,
                    ym,
                    p,
                    &mut cache.quadrature_rhs[(2 * i + q) * n..(2 * i + q + 1) * n],
                    &mut work[..n],
                    context,
                )?;
            }
        }
        let bc = &mut cache.residual[(m - 1) * n..];
        bc.fill(T::from_real(T::Real::NAN));
        admit(context, EvaluationRole::Constraint)?;
        self.problem
            .system
            .boundary(&point[..n], &point[(m - 1) * n..m * n], p, bc, context)?;
        finite(&cache.residual, NonFiniteOrigin::Callback)?;
        cache.point.copy_from_slice(point);
        cache.valid = true;
        Ok(())
    }
    fn evaluate(
        &self,
        point: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.ensure_evaluated(point, scratch, context)?;
        out.copy_from_slice(&self.cached()?.residual);
        Ok(())
    }
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> ResidualOperator<T> for Collocation<'_, '_, T, P> {
    fn input_dimension(&self) -> usize {
        self.mesh.len() * self.plan.layout.n + self.plan.layout.k
    }
    fn output_dimension(&self) -> usize {
        self.input_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_mul(
            self.plan.layout.n,
            5,
            scioxide_core::quantity!("residual_scratch"),
        )
    }
    fn residual_into(
        &self,
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        unscoped(|context| self.evaluate(x, out, scratch, context))
    }
    fn residual_into_scoped(
        &self,
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, out, scratch, context)
    }
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> JacobianOperator<T> for Collocation<'_, '_, T, P> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        if !matches!(self.plan.layout.backend, BvpBackend::Krylov { .. }) {
            return self.plan.layout.size(self.mesh.len());
        }
        let (n, _) = self.dimensions()?;
        checked_mul(n, 5, scioxide_core::quantity!("jvp_scratch"))
    }
    fn jvp_into(
        &self,
        x: &[T],
        v: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        unscoped(|context| self.jvp_into_scoped(x, v, out, scratch, context))
    }
    fn jvp_into_scoped(
        &self,
        x: &[T],
        v: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if let Some(direct) = self.direct {
            direct.apply(v, out, scratch)
        } else {
            self.product(x, v, out, scratch, context)
        }
    }
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Collocation<'_, '_, T, P> {
    fn jacobian(&self) -> Result<&JacobianCallbacks<T, P>, SciOxideError> {
        self.problem.jacobian.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("problem"),
                "analytic product requires Jacobian provider",
            )
        })
    }
    /// Evaluate the local analytic Jacobians at the cached point once per linearization:
    /// `2m-1` RHS Jacobians at the nodes and midpoints, and one boundary Jacobian.
    fn ensure_jacobians(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        let mut guard = self
            .cache
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "collocation cache is not prepared",
                )
            })?
            .borrow_mut();
        let cache = &mut *guard;
        if !cache.valid {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "collocation cache does not match the evaluated point",
            ));
        }
        if cache.jacobians_valid {
            return Ok(());
        }
        let (n, k) = self.dimensions()?;
        let m = self.mesh.len();
        let width = n * (n + k);
        let jacobian = self.jacobian()?;
        let p = &cache.point[m * n..m * n + k];
        let (blocks, boundary) = cache.jacobians.split_at_mut((2 * m - 1) * width);
        for (index, block) in blocks.chunks_exact_mut(width).enumerate() {
            let (x, y) = if index < m {
                (self.mesh[index], &cache.point[index * n..(index + 1) * n])
            } else {
                let i = index - m;
                (
                    self.mesh[i] + (self.mesh[i + 1] - self.mesh[i]) * T::Real::HALF,
                    &cache.middle[i * n..(i + 1) * n],
                )
            };
            let (jy, jp) = block.split_at_mut(n * n);
            admit(context, EvaluationRole::Jacobian)?;
            jy.fill(T::from_real(T::Real::NAN));
            jp.fill(T::from_real(T::Real::NAN));
            (jacobian.rhs)(self.problem.system, x, y, p, jy, jp, context)?;
            finite(jy, NonFiniteOrigin::Callback)?;
            finite(jp, NonFiniteOrigin::Callback)?;
        }
        let b = n + k;
        let (ja, rest) = boundary.split_at_mut(b * n);
        let (jb, jp) = rest.split_at_mut(b * n);
        ja.fill(T::from_real(T::Real::NAN));
        jb.fill(T::from_real(T::Real::NAN));
        jp.fill(T::from_real(T::Real::NAN));
        admit(context, EvaluationRole::Jacobian)?;
        (jacobian.boundary)(
            self.problem.system,
            &cache.point[..n],
            &cache.point[(m - 1) * n..m * n],
            p,
            ja,
            jb,
            jp,
            context,
        )?;
        finite(ja, NonFiniteOrigin::Callback)?;
        finite(jb, NonFiniteOrigin::Callback)?;
        finite(jp, NonFiniteOrigin::Callback)?;
        cache.jacobians_valid = true;
        Ok(())
    }
    /// Apply one cached `[J_y | J_p]` block at `x` to `(v, vp)`, including the singular term.
    fn apply_local(
        &self,
        x: T::Real,
        block: &[T],
        v: &[T],
        vp: &[T],
        out: &mut [T],
        work: &mut [T],
    ) -> Result<(), SciOxideError> {
        let n = out.len();
        let k = vp.len();
        let (jy, jp) = block.split_at(n * n);
        for (i, a) in out.iter_mut().enumerate() {
            *a = jy[i * n..(i + 1) * n]
                .iter()
                .zip(v)
                .chain(jp[i * k..(i + 1) * k].iter().zip(vp))
                .fold(T::ZERO, |sum, (&j, &v)| sum + j * v);
        }
        if let Some(s) = &self.plan.singular {
            if x == self.mesh[0] {
                work[..n].copy_from_slice(out);
                mat_apply(&s.derivative, &work[..n], out);
            } else {
                mat_apply(&s.matrix, v, &mut work[..n]);
                let distance = T::from_real(x - self.mesh[0]);
                out.iter_mut()
                    .zip(&work[..n])
                    .for_each(|(a, b)| *a = *a + *b / distance);
            }
        }
        finite(out, NonFiniteOrigin::Intermediate)
    }
    /// Matrix-free collocation Jacobian product from the cached linearization: no RHS or
    /// Jacobian callback runs once the point and its local Jacobians are cached.
    fn product(
        &self,
        x: &[T],
        v: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.ensure_evaluated(x, scratch, context)?;
        self.ensure_jacobians(context)?;
        let cache = self.cached()?;
        let (n, k) = self.dimensions()?;
        let m = self.mesh.len();
        let width = n * (n + k);
        let blocks = &cache.jacobians;
        let vp = &v[m * n..m * n + k];
        let (da, s) = scratch.split_at_mut(n);
        let (db, s) = s.split_at_mut(n);
        let (vm, s) = s.split_at_mut(n);
        let (dm, work) = s.split_at_mut(n);
        let half = T::from_real(T::Real::HALF);
        let eight = T::from_real(r(8.)?);
        let six = T::from_real(r(6.)?);
        let four = T::from_real(r(4.)?);
        self.apply_local(self.mesh[0], &blocks[..width], &v[..n], vp, da, work)?;
        for (i, interval) in self.mesh.windows(2).enumerate() {
            let va = &v[i * n..(i + 1) * n];
            let vb = &v[(i + 1) * n..(i + 2) * n];
            let h = T::from_real(interval[1] - interval[0]);
            self.apply_local(
                interval[1],
                &blocks[(i + 1) * width..(i + 2) * width],
                vb,
                vp,
                db,
                work,
            )?;
            for j in 0..n {
                vm[j] = (va[j] + vb[j]) * half - h / eight * (db[j] - da[j]);
            }
            self.apply_local(
                interval[0] + (interval[1] - interval[0]) * T::Real::HALF,
                &blocks[(m + i) * width..(m + i + 1) * width],
                vm,
                vp,
                dm,
                work,
            )?;
            for (j, a) in out[i * n..(i + 1) * n].iter_mut().enumerate() {
                *a = (vb[j] - va[j]) / h - (da[j] + four * dm[j] + db[j]) / six;
            }
            da.copy_from_slice(db);
        }
        let b = n + k;
        let boundary = &blocks[(2 * m - 1) * width..];
        let (ja, rest) = boundary.split_at(b * n);
        let (jb, jp) = rest.split_at(b * n);
        for (i, a) in out[(m - 1) * n..].iter_mut().enumerate() {
            *a = ja[i * n..(i + 1) * n]
                .iter()
                .zip(&v[..n])
                .chain(jb[i * n..(i + 1) * n].iter().zip(&v[(m - 1) * n..m * n]))
                .chain(jp[i * k..(i + 1) * k].iter().zip(vp))
                .fold(T::ZERO, |sum, (&j, &v)| sum + j * v);
        }
        finite(out, NonFiniteOrigin::Intermediate)
    }
}
