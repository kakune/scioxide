//! Indexed start designs over a finite box.
use scioxide_core::backend::checked_mul;
use scioxide_core::{RealScalar, SciOxideError};

use crate::validation::invalid;
use crate::{FiniteBounds, SearchRandom};

/// Generates the start points of a multistart search inside a finite box.
///
/// A design is indexed: `point` depends only on the start index, the box and
/// the random source it receives, never on an earlier call, so a search with
/// the same seed produces the same starts. Implementations must write every
/// output coordinate.
pub trait StartDesign<S: RealScalar> {
    /// Checks the dimension and the requested number of starts before the
    /// first objective evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error if the design does not support `dimension` or has
    /// fewer than `starts` points.
    fn validate(&self, dimension: usize, starts: usize) -> Result<(), SciOxideError>;
    /// Writes start point `index` to `output`, in the coordinates of `bounds`.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounds or output dimension differs from the
    /// design dimension. Returns design errors, and the errors of `random` when a draw is
    /// refused or the generator fails.
    fn point(
        &self,
        index: usize,
        bounds: &FiniteBounds<S>,
        random: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError>;
}

/// Draws each start coordinate independently and uniformly from its interval
/// of the box, one sample per coordinate.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformStartDesign;
impl<S: RealScalar> StartDesign<S> for UniformStartDesign {
    fn validate(&self, _: usize, _: usize) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn point(
        &self,
        _: usize,
        bounds: &FiniteBounds<S>,
        random: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::validation::shape(bounds.len(), output.len())?;
        output
            .iter_mut()
            .zip(bounds.as_slice())
            .try_for_each(|(x, &(lo, hi))| {
                *x = scioxide_core::checked_affine(lo, hi, random.uniform()?)?;
                Ok(())
            })
    }
}

/// Uses start points that you supply, stored row by row (one point per row).
///
/// Row `i` is start `i`, including the first start.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedStartDesign<'a, S> {
    points: &'a [S],
    dimension: usize,
}
impl<'a, S: RealScalar> BorrowedStartDesign<'a, S> {
    /// Borrows `points` as rows of length `dimension`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `dimension` is zero, `points` is empty, its length is
    /// not a multiple of `dimension`, or a coordinate is NaN or infinite.
    pub fn new(points: &'a [S], dimension: usize) -> Result<Self, SciOxideError> {
        if dimension == 0
            || points.is_empty()
            || !points.len().is_multiple_of(dimension)
            || points.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("points"),
                "invalid start matrix",
            ));
        }
        Ok(Self { points, dimension })
    }
}
impl<S: RealScalar> StartDesign<S> for BorrowedStartDesign<'_, S> {
    fn validate(&self, dimension: usize, starts: usize) -> Result<(), SciOxideError> {
        if dimension != self.dimension || starts > self.points.len() / dimension {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "start matrix shape or row count differs",
            ));
        }
        Ok(())
    }
    fn point(
        &self,
        index: usize,
        bounds: &FiniteBounds<S>,
        _: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::validation::shape(self.dimension, bounds.len())?;
        crate::validation::shape(self.dimension, output.len())?;
        let first = checked_mul(index, self.dimension, scioxide_core::quantity!("start_row"))?;
        let row = self
            .points
            .get(first..)
            .and_then(|x| x.get(..self.dimension))
            .ok_or_else(|| invalid(scioxide_core::argument!("index"), "start row out of range"))?;
        output.copy_from_slice(row);
        Ok(())
    }
}

/// Uses point `i` of a quasi-Monte Carlo sequence, mapped affinely to the box,
/// as start `i`.
///
/// Points come from [`scioxide_random::QmcPlan::point_into`] and are identical
/// to the sequence at the same index. No sample is charged.
impl<S: RealScalar> StartDesign<S> for scioxide_random::QmcPlan {
    fn validate(&self, dimension: usize, _: usize) -> Result<(), SciOxideError> {
        if self.dim() != dimension {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "QMC dimension differs",
            ));
        }
        Ok(())
    }
    fn point(
        &self,
        index: usize,
        bounds: &FiniteBounds<S>,
        _: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::validation::shape(self.dim(), bounds.len())?;
        crate::validation::shape(self.dim(), output.len())?;
        self.point_into(
            u64::try_from(index)
                .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("qmc_index")))?,
            output,
        )?;
        output
            .iter_mut()
            .zip(bounds.as_slice())
            .try_for_each(|(x, &(lo, hi))| {
                *x = scioxide_core::checked_affine(lo, hi, *x)?;
                Ok(())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_random::{Pcg64, SeedableRng};

    struct NoDraws;
    impl crate::random::DrawAdmission for NoDraws {
        fn admit(&mut self) -> Result<Option<crate::OptimizeTermination>, SciOxideError> {
            Ok(Some(crate::OptimizeTermination::SampleLimit))
        }
        fn refusal(&self, _: crate::OptimizeTermination) -> SciOxideError {
            invalid(scioxide_core::argument!("random"), "unexpected random draw")
        }
    }

    #[test]
    fn start_design_shapes_are_checked_before_output_or_random_draws() -> Result<(), SciOxideError>
    {
        let bounds = FiniteBounds::new(alloc::vec![(0.0_f64, 1.0); 2])?;
        let mut generator = Pcg64::from_seed(42);
        let mut admission = NoDraws;
        let mut random = SearchRandom::new(&mut generator, &mut admission);
        for length in [0, 1, 3] {
            let mut output = alloc::vec![9.0; length];
            let error = UniformStartDesign
                .point(0, &bounds, &mut random, &mut output)
                .unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output, alloc::vec![9.0; length]);
            assert!(random.stopped().is_none());
        }
        let mut output = [9.0];
        let qmc = scioxide_random::QmcPlan::halton(1)?;
        assert!(StartDesign::point(&qmc, 0, &bounds, &mut random, &mut output).is_err());
        assert_eq!(output, [9.0]);
        let borrowed = BorrowedStartDesign::new(&[0.5], 1)?;
        assert!(
            borrowed
                .point(0, &bounds, &mut random, &mut output)
                .is_err()
        );
        assert_eq!(output, [9.0]);
        let borrowed = BorrowedStartDesign::new(&[0.5, 0.5], 2)?;
        assert_eq!(
            borrowed
                .point(0, &bounds, &mut random, &mut output)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, [9.0]);
        Ok(())
    }
}
