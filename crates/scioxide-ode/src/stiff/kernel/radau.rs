//! Radau IIA's real eigenmode and complex conjugate pair of stage modes.
//!
//! Hairer–Wanner, *Solving ODEs II*, IV.8: `W = T⁻¹Y` block-diagonalizes
//! `A⁻¹`. The normalization here is `M − hγJ`, with `γ = 1/μ`; consequently
//! an original stage residual is transformed by `T⁻¹`, without division by h.
//! The eigenvector matrix T is scaled so that its last row is `[1, 1, 0]`.
//! Complex states require both conjugate eigenmodes: their Jacobian need not
//! be real, so the two shifted factors cannot be recovered by conjugation.
use super::*;
use core::cell::RefCell;
use core::marker::PhantomData;
use scioxide_core::{AlgebraicScalar, Complex};

struct Transform<S> {
    forward: [S; 9],
    inverse: [S; 9],
    real: S,
    complex: Complex<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Transform<S> {
    fn new() -> Result<Self, SciOxideError> {
        let constants = [
            0.09443876248897524,
            -0.141_255_295_020_954_2,
            0.03002919410514742,
            0.250_213_122_965_333_3,
            0.20412935229379994,
            -0.382_942_112_757_261_9,
            1.0,
            1.0,
            0.0,
        ];
        let mut forward = [S::ZERO; 9];
        for (value, constant) in forward.iter_mut().zip(constants) {
            *value = S::try_from_f64_lossy(constant)?;
        }
        let [a, b, c, d, e, f, g, h, i] = forward;
        let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        let inverse = [
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        ]
        .map(|v| v / determinant);
        let two = S::ONE + S::ONE;
        let three = two + S::ONE;
        let root = three.cbrt();
        let square = root * root;
        let mu = Complex::new(
            three + (root - square) / two,
            -three.sqrt() * (root + square) / two,
        );
        Ok(Self {
            forward: [a, b, c, d, e, f, g, h, i],
            inverse,
            real: S::ONE / (three + square - root),
            complex: Complex::new(S::ONE, S::ZERO) / mu,
        })
    }

    fn apply<T: NumericScalar<Real = S>>(
        matrix: &[S; 9],
        input: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        let n = input.len() / 3;
        for (row, coefficients) in matrix.chunks_exact(3).enumerate() {
            for (i, value) in output[row * n..(row + 1) * n].iter_mut().enumerate() {
                *value = crate::safety::stage_affine(
                    T::ZERO,
                    S::ONE,
                    coefficients
                        .iter()
                        .enumerate()
                        .map(|(j, &coefficient)| (coefficient, input[j * n + i])),
                )?;
            }
        }
        Ok(())
    }
}

/// One cache bundle, sharing the parent kernel's single raw Jacobian.
pub(super) struct RadauStorage<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>
{
    transform: Transform<S>,
    real: LinearStorage<T>,
    plus: Option<LinearStorage<Complex<S>>>,
    minus: Option<LinearStorage<Complex<S>>>,
    modes: Vec<T>,
    rhs: Vec<Complex<S>>,
    delta: Vec<Complex<S>>,
    native: Vec<T>,
    filter_product: Vec<T>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> RadauStorage<S, T> {
    pub(super) fn new(
        n: usize,
        backend: &OdeStageBackend,
        provided: bool,
    ) -> Result<Self, SciOxideError> {
        let mut effective = backend.clone();
        let native_len = match &mut effective {
            OdeStageBackend::Krylov {
                operator_scratch,
                preconditioner_scratch,
                ..
            } => {
                let len = scioxide_core::backend::checked_add(
                    checked_mul(n, 5, scioxide_core::quantity!("radau_lift_scratch"))?,
                    (*operator_scratch).max(*preconditioner_scratch),
                    scioxide_core::quantity!("radau_lift_scratch"),
                )?;
                // Lifted operators borrow the reserved native-scalar scratch.
                *operator_scratch = 0;
                *preconditioner_scratch = 0;
                len
            }
            _ => 0,
        };
        Ok(Self {
            transform: Transform::new()?,
            real: LinearStorage::new(n, &effective)?,
            plus: if provided {
                None
            } else {
                Some(LinearStorage::new(n, &effective)?)
            },
            minus: if T::IS_REAL || provided {
                None
            } else {
                Some(LinearStorage::new(n, &effective)?)
            },
            modes: try_filled(
                checked_mul(
                    n,
                    if provided { 0 } else { 3 },
                    scioxide_core::quantity!("radau_modes"),
                )?,
                T::ZERO,
                scioxide_core::quantity!("radau_modes"),
            )?,
            rhs: try_filled(
                if provided { 0 } else { n },
                Complex::ZERO,
                scioxide_core::quantity!("radau_rhs"),
            )?,
            delta: try_filled(
                if provided { 0 } else { n },
                Complex::ZERO,
                scioxide_core::quantity!("radau_delta"),
            )?,
            native: try_filled(
                native_len,
                T::ZERO,
                scioxide_core::quantity!("radau_lift_scratch"),
            )?,
            filter_product: try_filled(
                n,
                T::ZERO,
                scioxide_core::quantity!("radau_filter_product"),
            )?,
        })
    }

    pub(super) fn factor(
        &mut self,
        lin: &Linearization<S, T>,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        factor_mode(
            lin,
            &mut self.real,
            T::from_real(lin.step * self.transform.real),
            request,
            cx,
        )?;
        let shift = self.transform.complex * lin.step;
        if let Some(plus) = &mut self.plus {
            factor_mode(lin, plus, shift, request, cx)?;
        }
        if let Some(minus) = &mut self.minus {
            factor_mode(lin, minus, shift.conj(), request, cx)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        lin: &Linearization<S, T>,
        system: &P,
        backend: &OdeStageBackend,
        request: StepRequest<'_, S, T>,
        relative: S,
        residual: &[T],
        correction: &mut [T],
        krylov: &mut SolveState,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = lin.n;
        let plus = self.plus.as_mut().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "native Radau stage storage required",
            )
        })?;
        Transform::apply(&self.transform.inverse, residual, &mut self.modes)?;
        solve_mode(
            &mut self.real,
            lin,
            system,
            backend,
            T::from_real(lin.step * self.transform.real),
            request,
            relative,
            &self.modes[..n],
            &mut correction[..n],
            &mut self.native,
            krylov,
            0,
            cx,
        )?;
        let imaginary = Complex::new(S::ZERO, S::ONE);
        for (i, rhs) in self.rhs.iter_mut().enumerate() {
            *rhs = promote(self.modes[n + i]) + imaginary * promote(self.modes[2 * n + i]);
        }
        let shift = self.transform.complex * lin.step;
        solve_mode(
            plus,
            lin,
            system,
            backend,
            shift,
            request,
            relative,
            &self.rhs,
            &mut self.delta,
            &mut self.native,
            krylov,
            1,
            cx,
        )?;
        if let Some(minus) = &mut self.minus {
            for (i, rhs) in self.rhs.iter_mut().enumerate() {
                correction[n + i] = convert(self.delta[i]);
                *rhs = promote(self.modes[n + i]) - imaginary * promote(self.modes[2 * n + i]);
            }
            solve_mode(
                minus,
                lin,
                system,
                backend,
                shift.conj(),
                request,
                relative,
                &self.rhs,
                &mut self.delta,
                &mut self.native,
                krylov,
                2,
                cx,
            )?;
            let half = S::ONE / (S::ONE + S::ONE);
            for (i, &minus) in self.delta.iter().enumerate() {
                let plus = promote(correction[n + i]);
                correction[n + i] = convert((plus + minus) * half);
                correction[2 * n + i] = convert((plus - minus) * Complex::new(S::ZERO, -half));
            }
        } else {
            for (i, delta) in self.delta.iter().enumerate() {
                correction[n + i] = T::from_real(delta.re);
                correction[2 * n + i] = T::from_real(delta.im);
            }
        }
        Transform::apply(&self.transform.forward, correction, &mut self.modes)?;
        correction.copy_from_slice(&self.modes);
        finite(correction, scioxide_core::NonFiniteOrigin::Callback)
    }

    /// Filter a state-unit embedded defect through the already prepared real mode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn filter<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        lin: &Linearization<S, T>,
        system: &P,
        backend: &OdeStageBackend,
        request: StepRequest<'_, S, T>,
        relative: S,
        rhs: &[T],
        error: &mut [T],
        krylov: &mut SolveState,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        solve_mode(
            &mut self.real,
            lin,
            system,
            backend,
            T::from_real(lin.step * self.transform.real),
            request,
            relative,
            rhs,
            error,
            &mut self.native,
            krylov,
            0,
            cx,
        )?;
        if self.real.is_iterative() {
            let native = RefCell::new(self.native.as_mut_slice());
            let operator = Shifted {
                lin,
                system,
                request,
                shift: T::from_real(lin.step * self.transform.real),
                native: &native,
            };
            admit_provider(cx, EvaluationRole::LinearOperator, request.trace)?;
            let outcome = SolveScope::from(&mut *cx).run(
                krylov,
                product_identity(lin.n),
                SolveBudgets::unbounded(),
                |context| operator.apply(error, &mut self.filter_product, Some(context)),
            );

            outcome?;
            verify_defect(&self.filter_product, rhs, relative)?;
        }
        Ok(())
    }

    pub(super) fn real_shift(&self) -> S {
        self.transform.real
    }
}

fn promote<T: NumericScalar>(value: T) -> Complex<T::Real> {
    Complex::new(value.real_component(), value.imaginary_component())
}
fn convert<T: NumericScalar, U: NumericScalar<Real = T::Real>>(value: U) -> T {
    T::from_components(value.real_component(), value.imaginary_component())
}

/// Assemble only an n×n shifted matrix, keeping the original sparse pattern.
fn factor_mode<S, T, U>(
    lin: &Linearization<S, T>,
    linear: &mut LinearStorage<U>,
    shift: U,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    U: NumericScalar<Real = S>,
{
    let entry = |mass: T, jacobian: T| {
        let jacobian = convert::<U, _>(jacobian);
        // Compensated native sums promote to scaled arithmetic only when range
        // requires it, including cancellation in M − shift*J.
        crate::safety::stage_affine(
            convert::<U, _>(mass),
            S::ONE,
            [
                (-shift.real_component(), jacobian),
                (
                    -shift.imaginary_component(),
                    U::from_components(-jacobian.imaginary_component(), jacobian.real_component()),
                ),
            ],
        )
    };
    match &lin.jacobians {
        Jacobians::Dense(blocks) => {
            let matrix = linear.dense_mut()?.as_mut_slice();
            for (k, (value, &jacobian)) in matrix.iter_mut().zip(blocks[0].as_slice()).enumerate() {
                let mass = lin.mass.as_ref().map_or_else(
                    || {
                        if k / lin.n == k % lin.n {
                            T::ONE
                        } else {
                            T::ZERO
                        }
                    },
                    |mass| mass.as_slice()[k],
                );
                *value = entry(mass, jacobian)?;
            }
        }
        Jacobians::Sparse { columns, values } => linear.assemble_sparse(|pattern, output| {
            for (row, bounds) in pattern.row_ptr().windows(2).enumerate() {
                for k in bounds[0]..bounds[1] {
                    output[k] = entry(
                        if pattern.col_indices()[k] == row {
                            T::ONE
                        } else {
                            T::ZERO
                        },
                        values[k],
                    )?;
                }
            }
            debug_assert_eq!(columns.len(), output.len());
            Ok(())
        })?,
        Jacobians::Operator { .. } => return Ok(()),
    }
    admit_provider(cx, EvaluationRole::Factorization, request.trace)?;
    linear.factor(cx.execution())?;
    Ok(())
}

/// Lift a native action to complex scalars. Real systems apply J separately to
/// the real and imaginary directions; complex systems make one native product.
fn lift<T, U>(
    input: &[U],
    output: &mut [U],
    native: &mut [T],
    role: EvaluationRole,
    mut context: Option<&mut SolveContext<'_, '_>>,
    mut apply: impl FnMut(
        &[T],
        &mut [T],
        &mut [T],
        Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError>
where
    T: NumericScalar,
    U: NumericScalar<Real = T::Real>,
{
    let n = input.len();
    let (direction, rest) = native.split_at_mut(n);
    let (product, scratch) = rest.split_at_mut(n);
    for (x, &value) in direction.iter_mut().zip(input) {
        *x = convert(value);
    }
    product.fill(T::from_real(T::Real::NAN));
    apply(direction, product, scratch, context.as_deref_mut())?;
    finite(product, scioxide_core::NonFiniteOrigin::Callback)?;
    for (out, &value) in output.iter_mut().zip(&*product) {
        *out = convert(value);
    }
    if T::IS_REAL && !U::IS_REAL {
        if let Some(context) = context.as_deref_mut() {
            crate::termination::record(context, role)?;
        }
        for (x, &value) in direction.iter_mut().zip(input) {
            *x = T::from_real(value.imaginary_component());
        }
        product.fill(T::from_real(T::Real::NAN));
        apply(direction, product, scratch, context)?;
        finite(product, scioxide_core::NonFiniteOrigin::Callback)?;
        for (out, &value) in output.iter_mut().zip(&*product) {
            *out = U::from_components(out.real_component(), value.real_component());
        }
    }
    finite(output, scioxide_core::NonFiniteOrigin::Callback)
}

struct Shifted<'a, S: RealScalar, T: NumericScalar<Real = S>, U, P: ?Sized> {
    lin: &'a Linearization<S, T>,
    system: &'a P,
    request: StepRequest<'a, S, T>,
    shift: U,
    native: &'a RefCell<&'a mut [T]>,
}
impl<S, T, U, P> Shifted<'_, S, T, U, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    U: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn apply(
        &self,
        input: &[U],
        output: &mut [U],
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let lin = self.lin;
        if input.len() != lin.n || output.len() != lin.n {
            return Err(invalid(
                scioxide_core::argument!("input"),
                "Radau shifted operator shape mismatch",
            ));
        }
        let Jacobians::Operator { sources, scratch } = &lin.jacobians else {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "Radau matrix-free storage required",
            ));
        };
        lift(
            input,
            output,
            &mut self.native.borrow_mut(),
            EvaluationRole::LinearOperator,
            context,
            |direction, product, work, context| {
                // Products evaluate system callbacks, which need the invoking scope.
                let Some(context) = context else {
                    return Err(invalid(
                        scioxide_core::argument!("operator"),
                        "matrix-free stage operators require scoped application",
                    ));
                };
                super::super::jacobian::product(
                    self.system,
                    lin.strategy,
                    &sources[0],
                    ProductPoint {
                        time: lin.times[0],
                        state: &lin.points[..lin.n],
                        base: &lin.rhs[..lin.n],
                        floors: &lin.floors,
                    },
                    direction,
                    product,
                    &mut work[..*scratch + 3 * lin.n],
                    self.request,
                    context,
                )?;
                Ok(())
            },
        )?;
        for (out, &value) in output.iter_mut().zip(input) {
            *out = value - self.shift * *out;
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}
impl<S, T, U, P> LinearOperator<U> for Shifted<'_, S, T, U, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    U: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn rows(&self) -> usize {
        self.lin.n
    }
    fn cols(&self) -> usize {
        self.lin.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[U], y: &mut [U], _: &mut [U]) -> Result<(), SciOxideError> {
        self.apply(x, y, None)
    }
    fn apply_into_scoped(
        &self,
        x: &[U],
        y: &mut [U],
        _: &mut [U],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, U>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(x, y, Some(&mut **context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}

struct LiftedPreconditioner<'a, T: NumericScalar, U> {
    provider: &'a dyn Preconditioner<T>,
    native: &'a RefCell<&'a mut [T]>,
    scalar: PhantomData<U>,
}
impl<T: NumericScalar, U: NumericScalar<Real = T::Real>> LiftedPreconditioner<'_, T, U> {
    fn apply(
        &self,
        x: &[U],
        y: &mut [U],
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let n = self.provider.dimension();
        if x.len() != n || y.len() != n {
            return Err(invalid(
                scioxide_core::argument!("input"),
                "Radau preconditioner shape mismatch",
            ));
        }
        let len = self.provider.precondition_workspace_len()?;
        lift(
            x,
            y,
            &mut self.native.borrow_mut(),
            EvaluationRole::Preconditioner,
            context,
            |x, y, work, context| match context {
                Some(context) => self.provider.precondition_into_scoped(
                    x,
                    y,
                    &mut work[..len],
                    &mut scioxide_core::OperatorContext::new(context),
                ),
                None => self.provider.precondition_into(x, y, &mut work[..len]),
            },
        )
    }
}
impl<T: NumericScalar, U: NumericScalar<Real = T::Real>> Preconditioner<U>
    for LiftedPreconditioner<'_, T, U>
{
    fn dimension(&self) -> usize {
        self.provider.dimension()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(&self, x: &[U], y: &mut [U], _: &mut [U]) -> Result<(), SciOxideError> {
        self.apply(x, y, None)
    }
    fn precondition_into_scoped(
        &self,
        x: &[U],
        y: &mut [U],
        _: &mut [U],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, U>,
    ) -> Result<(), SciOxideError> {
        self.apply(x, y, Some(&mut **context))
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_mode<S, T, U, P>(
    linear: &mut LinearStorage<U>,
    lin: &Linearization<S, T>,
    system: &P,
    backend: &OdeStageBackend,
    shift: U,
    request: StepRequest<'_, S, T>,
    relative: S,
    rhs: &[U],
    output: &mut [U],
    native: &mut [T],
    krylov: &mut SolveState,
    mode: u128,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    U: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    admit_provider(cx, EvaluationRole::LinearSolve, request.trace)?;

    if !linear.is_iterative() {
        return linear
            .solve_direct(rhs, output, cx.execution())
            .and_then(|()| finite(output, scioxide_core::NonFiniteOrigin::Callback));
    }
    let provider = request.trace.provider(system.stage_preconditioner(
        lin.times[0],
        &lin.points[..lin.n],
        shift.real_component(),
        cx,
    ))?;
    if let Some(provider) = provider {
        let capacity = match backend {
            OdeStageBackend::Krylov {
                preconditioner_scratch,
                ..
            } => *preconditioner_scratch,
            _ => 0,
        };
        if provider.dimension() != lin.n || provider.precondition_workspace_len()? > capacity {
            return Err(invalid(
                scioxide_core::argument!("system"),
                "Radau preconditioner exceeds prepared capacity",
            ));
        }
    }
    let native = RefCell::new(native);

    let operator = Shifted {
        lin,
        system,
        request,
        shift,
        native: &native,
    };
    let preconditioner = provider.map(|provider| LiftedPreconditioner::<T, U> {
        provider,
        native: &native,
        scalar: PhantomData,
    });
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("ode", "radau-stage-gmres", 1),
        lin.n as u128,
        mode,
    );
    // Leave room for the stage transformation; the caller checks the original defect.
    let relative = relative / S::try_from_f64_lossy(32.0)?;
    let result =
        SolveScope::from(&mut *cx).run(krylov, identity, SolveBudgets::unbounded(), |context| {
            linear
                .solve_iterative(
                    &operator,
                    preconditioner.as_ref().map(|p| p as &dyn Preconditioner<U>),
                    rhs,
                    output,
                    relative,
                    request.inner_limit.get(),
                    context,
                )
                .map(drop)
        });

    result.and_then(|()| finite(output, scioxide_core::NonFiniteOrigin::Callback))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn dimension<T: NumericScalar>(linear: &LinearStorage<T>, n: usize) {
        let LinearStorage::Dense { matrix, .. } = linear else {
            unreachable!()
        };
        assert_eq!((matrix.rows(), matrix.cols()), (n, n));
    }

    fn factors<S, T>()
    where
        S: RealScalar + NumericScalar<Real = S>,
        T: NumericScalar<Real = S>,
    {
        let kernel = StageKernel::<S, T>::new(
            7,
            KernelShape {
                backend: &OdeStageBackend::Dense,
                layout: StageLayout::Radau,
                stages: 3,
                blocks: 1,
                provided: false,
            },
        )
        .unwrap();
        let Jacobians::Dense(jacobians) = &kernel.lin.jacobians else {
            unreachable!()
        };
        assert_eq!(jacobians.len(), 1);
        let EffectiveStorage::Radau(storage) = &kernel.linear else {
            unreachable!()
        };
        dimension(&storage.real, 7);
        dimension(storage.plus.as_ref().unwrap(), 7);
        assert_eq!(storage.minus.is_none(), T::IS_REAL);
        if let Some(minus) = &storage.minus {
            dimension(minus, 7);
        }
    }
    #[test]
    fn only_n_by_n_factors_and_one_jacobian_in_every_domain() {
        factors::<f32, f32>();
        factors::<f64, f64>();
        factors::<f32, Complex<f32>>();
        factors::<f64, Complex<f64>>();
    }

    fn cancellation<S, T>()
    where
        S: RealScalar + NumericScalar<Real = S>,
        T: NumericScalar<Real = S>,
    {
        let large = S::MAX * S::try_from_f64_lossy(0.75).unwrap();
        let value = T::from_components(large, if T::IS_REAL { S::ZERO } else { large });
        let two = S::ONE + S::ONE;
        let matrix = [
            two,
            S::ZERO,
            -two,
            S::ZERO,
            S::ONE,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ONE,
        ];
        let mut output = [T::ZERO; 3];
        Transform::apply(&matrix, &[value, T::ONE, value], &mut output).unwrap();
        assert_eq!(output, [T::ZERO, T::ONE, value]);
    }

    #[test]
    fn native_transform_keeps_extended_range_cancellation() {
        cancellation::<f32, f32>();
        cancellation::<f64, f64>();
        cancellation::<f32, Complex<f32>>();
        cancellation::<f64, Complex<f64>>();
    }

    fn pole<T: NumericScalar<Real = f64>>(complex: bool) {
        let mut kernel = StageKernel::<f64, T>::new(
            1,
            KernelShape {
                backend: &OdeStageBackend::Dense,
                layout: StageLayout::Radau,
                stages: 3,
                blocks: 1,
                provided: false,
            },
        )
        .unwrap();
        let EffectiveStorage::Radau(storage) = &mut kernel.linear else {
            unreachable!()
        };
        let mass = if complex {
            convert(storage.transform.complex)
        } else {
            T::from_real(storage.transform.real)
        };
        kernel.lin.mass.as_mut().unwrap().as_mut_slice()[0] = mass;
        let Jacobians::Dense(blocks) = &mut kernel.lin.jacobians else {
            unreachable!()
        };
        blocks[0].as_mut_slice()[0] = T::ONE;
        kernel.lin.step = 1.0;
        let trace = crate::stepper::StepTrace::default();
        let error = crate::stepper::testing::with_context(|cx| {
            storage.factor(&kernel.lin, StepRequest::new(&trace), cx)
        })
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
    }
    #[test]
    fn real_and_complex_shift_poles_return_singular_errors() {
        pole::<f64>(false);
        pole::<Complex<f64>>(false);
        pole::<Complex<f64>>(true);
    }
}
