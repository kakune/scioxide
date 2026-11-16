//! Isotropic coefficient adapters shared by finite differences, volumes and elements.
use crate::{FvmCoefficients, SimplexFemCoefficients, StructuredFdmCoefficients};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// Supply constant isotropic coefficients of `−∇·(k∇u) + c u` for every component.
/// Plans validate the diffusion `k` and reaction `c`; evaluation allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantDiffusion<R> {
    diffusion: R,
    reaction: R,
}
impl<R: RealScalar> ConstantDiffusion<R> {
    /// Set diffusion `k`, with zero reaction, in constant time without rounding.
    /// Plans validate the coefficient when they sample it.
    /// ```
    /// use scioxide_pde::ConstantDiffusion;
    /// let coefficients = ConstantDiffusion::new(2.0_f64).with_reaction(0.5);
    /// assert!((coefficients.diffusion() - 2.0).abs() < 1e-15);
    /// ```
    pub const fn new(diffusion: R) -> Self {
        Self {
            diffusion,
            reaction: R::ZERO,
        }
    }
    /// Set the diagonal reaction coefficient `c`.
    pub const fn with_reaction(mut self, reaction: R) -> Self {
        self.reaction = reaction;
        self
    }
    /// Return the isotropic diffusion coefficient.
    pub const fn diffusion(&self) -> R {
        self.diffusion
    }
    /// Return the diagonal reaction coefficient.
    pub const fn reaction(&self) -> R {
        self.reaction
    }
}
/// Supply isotropic diffusion `k(x)` of `−∇·(k∇u)` for every component.
/// Each evaluation calls the closure once and allocates nothing.
#[derive(Clone, Copy)]
pub struct IsotropicDiffusion<F> {
    diffusion: F,
}
impl<F> IsotropicDiffusion<F> {
    /// Bind the scalar coefficient as a function of position without evaluating it.
    /// The adapter introduces no approximation beyond the selected discretization.
    /// ```
    /// use scioxide_pde::{IsotropicDiffusion, StructuredFdmCoefficients};
    /// let coefficient = IsotropicDiffusion::new(|x: [f64; 1]| Ok(1.0 + x[0]));
    /// let k = <_ as StructuredFdmCoefficients<f64, f64, 1>>::diffusion(&coefficient, 0, 0, [0.5])?;
    /// assert!((k - 1.5).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new<R: RealScalar, const D: usize>(diffusion: F) -> Self
    where
        F: Fn([R; D]) -> Result<R, SciOxideError>,
    {
        Self { diffusion }
    }
}
impl<F> core::fmt::Debug for IsotropicDiffusion<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IsotropicDiffusion").finish_non_exhaustive()
    }
}
fn diagonal<V: NumericScalar>(out: &mut [V], value: V) {
    out.fill(V::ZERO);
    let n = out.len().isqrt();
    for i in 0..n {
        out[i * n + i] = value;
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StructuredFdmCoefficients<R, V, D>
    for ConstantDiffusion<R>
{
    fn diffusion(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
        Ok(self.diffusion)
    }
    fn reaction(&self, row: usize, column: usize, _: [R; D]) -> Result<V, SciOxideError> {
        Ok(if row == column {
            V::from_real(self.reaction)
        } else {
            V::ZERO
        })
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> FvmCoefficients<R, V, D>
    for ConstantDiffusion<R>
{
    fn diffusion(&self, _: [R; D]) -> Result<[[R; D]; D], SciOxideError> {
        Ok(core::array::from_fn(|i| {
            core::array::from_fn(|j| if i == j { self.diffusion } else { R::ZERO })
        }))
    }
    fn reaction(&self, _: [R; D]) -> Result<V, SciOxideError> {
        Ok(V::from_real(self.reaction))
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemCoefficients<R, V, D>
    for ConstantDiffusion<R>
{
    fn is_spatially_constant(&self) -> bool {
        true
    }
    fn evaluate(
        &self,
        _: [R; D],
        diffusion: &mut [V],
        reaction: &mut [V],
    ) -> Result<(), SciOxideError> {
        diagonal(diffusion, V::from_real(self.diffusion));
        diagonal(reaction, V::from_real(self.reaction));
        Ok(())
    }
}
impl<
    R: RealScalar,
    V: NumericScalar<Real = R>,
    F: Fn([R; D]) -> Result<R, SciOxideError>,
    const D: usize,
> StructuredFdmCoefficients<R, V, D> for IsotropicDiffusion<F>
{
    fn diffusion(&self, _: usize, _: usize, point: [R; D]) -> Result<R, SciOxideError> {
        (self.diffusion)(point)
    }
}
impl<
    R: RealScalar,
    V: NumericScalar<Real = R>,
    F: Fn([R; D]) -> Result<R, SciOxideError>,
    const D: usize,
> FvmCoefficients<R, V, D> for IsotropicDiffusion<F>
{
    fn diffusion(&self, point: [R; D]) -> Result<[[R; D]; D], SciOxideError> {
        let k = (self.diffusion)(point)?;
        Ok(core::array::from_fn(|i| {
            core::array::from_fn(|j| if i == j { k } else { R::ZERO })
        }))
    }
}
impl<
    R: RealScalar,
    V: NumericScalar<Real = R>,
    F: Fn([R; D]) -> Result<R, SciOxideError> + Sync,
    const D: usize,
> SimplexFemCoefficients<R, V, D> for IsotropicDiffusion<F>
{
    fn evaluate(
        &self,
        point: [R; D],
        diffusion: &mut [V],
        reaction: &mut [V],
    ) -> Result<(), SciOxideError> {
        let k = (self.diffusion)(point)?;
        diagonal(diffusion, V::from_real(k));
        reaction.fill(V::ZERO);
        Ok(())
    }
}
