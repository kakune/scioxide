//! Original-pencil verification for composed shift-invert partial eigenproblems.
pub(super) struct ShiftedProblem<'a, S: NumericScalar> {
    pub(super) a: &'a dyn LinearOperator<S>,
    pub(super) inverse: &'a dyn LinearOperator<S>,
    pub(super) mass: Option<&'a dyn LinearOperator<S>>,
    pub(super) shift: S,
}
pub(super) struct OperatorScratch<'a, S> {
    pub(super) input: &'a mut [S],
    pub(super) image: &'a mut [S],
    pub(super) scratch: &'a mut [S],
}
use super::*;
use crate::actions::ShiftInvertOperator;
use scioxide_core::AlgebraicScalar;

/// Scratch for products with the original `A` and `M`, kept across shift-invert cycles.
#[derive(Debug)]
pub(super) struct ShiftInvertBuffers<S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    input: Vec<S>,
    image: Vec<S>,
    av: Vec<Complex<S::Real>>,
    mv: Vec<Complex<S::Real>>,
    a_scratch: Vec<S>,
    m_scratch: Vec<S>,
}
impl<S: NumericScalar> ShiftInvertBuffers<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    pub(super) fn empty() -> Self {
        Self {
            input: Vec::new(),
            image: Vec::new(),
            av: Vec::new(),
            mv: Vec::new(),
            a_scratch: Vec::new(),
            m_scratch: Vec::new(),
        }
    }
    pub(super) fn storage_plan(
        &self,
        plan: &mut StoragePlan,
        n: usize,
        layout: ShiftLayout,
    ) -> Result<(), SciOxideError> {
        plan.buffer(&self.input, n)?;
        plan.buffer(&self.image, n)?;
        plan.buffer(&self.av, n)?;
        plan.buffer(&self.mv, n)?;
        plan.buffer(&self.a_scratch, layout.a)?;
        plan.buffer(&self.m_scratch, layout.mass)
    }
    pub(super) fn configure(&mut self, n: usize, layout: ShiftLayout) -> Result<(), SciOxideError> {
        resize(&mut self.input, n, S::ZERO)?;
        resize(&mut self.image, n, S::ZERO)?;
        resize(&mut self.av, n, Complex::ZERO)?;
        resize(&mut self.mv, n, Complex::ZERO)?;
        resize(&mut self.a_scratch, layout.a, S::ZERO)?;
        resize(&mut self.m_scratch, layout.mass, S::ZERO)
    }
}
impl<R: RealScalar> SpectrumCall<R> {
    /// Solve the transformed problem, map eigenvalues and independently verify `Av-lambda Mv`.
    /// Selection options apply to transformed values: largest magnitude targets nearest shift.
    pub(super) fn shift_run<S: NumericScalar<Real = R>>(
        &self,
        problem: ShiftedProblem<'_, S>,
        initial: VectorView<'_, S>,
        result: &mut PartialSpectrumBuffer<R>,
        w: &mut PartialSpectrumWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<PartialSpectrumReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let ShiftedProblem {
            a,
            inverse,
            mass,
            shift,
        } = problem;

        let shifted = ShiftInvertOperator::new(inverse, mass)?;
        let mut b = w.shift_invert.take().ok_or_else(|| {
            crate::checked::inconsistent(
                scioxide_core::argument!("w"),
                "workspace was not reserved for a shift-invert operator",
            )
        })?;
        let problem = ShiftInvertProblem { a, mass, shift };
        let outcome = self.run(&shifted, initial, result, w, work, Some((&problem, &mut b)));
        w.shift_invert = Some(b);
        outcome
    }
}
pub(super) struct ShiftInvertProblem<'a, S: NumericScalar> {
    a: &'a dyn LinearOperator<S>,
    mass: Option<&'a dyn LinearOperator<S>>,
    shift: S,
}
pub(super) struct VerifiedPair<R: RealScalar> {
    pub(super) value: Complex<R>,
    pub(super) residual: R,
    pub(super) scale: R,
}
impl<S: NumericScalar> ShiftInvertProblem<'_, S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Return original eigenvalue, residual and acceptance scale for a complete pair.
    pub(super) fn verify(
        &self,
        mu: Complex<S::Real>,
        vectors: &Matrix<Complex<S::Real>>,
        column: usize,
        b: &mut ShiftInvertBuffers<S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Option<VerifiedPair<S::Real>>, SciOxideError> {
        if mu == Complex::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "zero transformed Ritz value cannot be mapped to a finite eigenvalue",
            ));
        }
        let lambda = crate::checked::finite_candidate(
            crate::eigen::into_complex(self.shift) + Complex::ONE / mu,
        )?;
        if !complex_apply(
            self.a,
            vectors,
            column,
            &mut b.av,
            OperatorScratch {
                input: &mut b.input,
                image: &mut b.image,
                scratch: &mut b.a_scratch,
            },
            work,
        )? {
            return Ok(None);
        }
        if let Some(mass) = self.mass {
            if !complex_apply(
                mass,
                vectors,
                column,
                &mut b.mv,
                OperatorScratch {
                    input: &mut b.input,
                    image: &mut b.image,
                    scratch: &mut b.m_scratch,
                },
                work,
            )? {
                return Ok(None);
            }
        } else {
            for (i, value) in b.mv.iter_mut().enumerate() {
                *value = vectors.get_element(i, column);
            }
        }
        let scale =
            crate::reduction::l2_norm_with_context(VectorView::from_slice(&b.av), work.kernels)?
                .max(
                    lambda.norm()
                        * crate::reduction::l2_norm_with_context(
                            VectorView::from_slice(&b.mv),
                            work.kernels,
                        )?,
                );
        for (a, &m) in b.av.iter_mut().zip(&b.mv) {
            *a -= lambda * m;
        }
        Ok(Some(VerifiedPair {
            value: lambda,
            residual: crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&b.av),
                work.kernels,
            )?,
            scale,
        }))
    }
}

pub(super) fn complex_apply<S: NumericScalar>(
    a: &dyn LinearOperator<S>,
    vectors: &Matrix<Complex<S::Real>>,
    column: usize,
    out: &mut [Complex<S::Real>],
    storage: OperatorScratch<'_, S>,
    work: &mut Work<'_, '_, '_>,
) -> Result<bool, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let OperatorScratch {
        input,
        image,
        scratch,
    } = storage;

    // A real operator cannot take a complex vector directly; by linearity
    // `A (x + i y) = A x + i A y`, so it is applied to the real and imaginary
    // parts separately when the imaginary part is nonzero (a real S keeps only the
    // real part of the input).
    for (i, v) in input.iter_mut().enumerate() {
        let z = vectors.get_element(i, column);
        *v = S::from_components(z.re, z.im);
    }
    if !work.apply(a, input, image, scratch)? {
        return Ok(false);
    }
    for (z, &v) in out.iter_mut().zip(image.iter()) {
        *z = crate::eigen::into_complex(v);
    }
    if S::IS_REAL
        && (0..vectors.rows()).any(|row| vectors.get_element(row, column).im != S::Real::ZERO)
    {
        for (i, v) in input.iter_mut().enumerate() {
            *v = S::from_real(vectors.get_element(i, column).im);
        }
        if !work.apply(a, input, image, scratch)? {
            return Ok(false);
        }
        for (z, &v) in out.iter_mut().zip(image.iter()) {
            z.im = v.real_component();
        }
    }
    Ok(true)
}
impl<S: NumericScalar> StorageBytes for ShiftInvertBuffers<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.input.storage_bytes()?,
            self.image.storage_bytes()?,
            self.av.storage_bytes()?,
            self.mv.storage_bytes()?,
            self.a_scratch.storage_bytes()?,
            self.m_scratch.storage_bytes()?,
        ])
    }
}
