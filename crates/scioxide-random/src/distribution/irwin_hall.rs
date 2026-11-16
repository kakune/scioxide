//! Irwin–Hall distribution.
use crate::distribution::formula::formula_law;
use crate::distribution::formula::{Formula, Inverse};
use crate::law::constant as number;
use crate::law::{finite, invalid};
use crate::workspace::{admit_bytes, storage_bytes};
use crate::{LawOptions, Mean, Rng, Variance};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{RealScalar, SciOxideError};

fn filled<S: RealScalar>(n: usize) -> Result<Vec<S>, SciOxideError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("irwin_hall_preparation"), n)
    })?;
    out.resize(n, S::ZERO);
    Ok(out)
}

/// Irwin–Hall distribution: the sum of `n` independent standard uniform variables.
///
/// The support is `[0, n]`. On each segment `[j, j + 1]` the density is a
/// polynomial of degree `n − 1`, the cardinal B-spline of order `n`. Each
/// segment is evaluated in the Bernstein basis, whose coefficients are
/// nonnegative, so the density and CDF have no cancellation.
/// Quantiles are computed by numerical inversion of the CDF. A draw sums `n`
/// uniforms.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, IrwinHall, LawOptions, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = IrwinHall::<f64>::new(2)?;
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-8);
/// assert!((law.cdf(law.quantile(0.9)?)? - 0.9).abs() < 1e-8);
/// assert!((law.mean()? - 1.0).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(65);
/// assert!((0.0..=2.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct IrwinHall<S: RealScalar> {
    options: LawOptions<S>,
    n: usize,
    end: S,
    pdf: Vec<Vec<S>>,
    cdf: Vec<Vec<S>>,
    log_fact: Vec<S>,
    normalizer: S,
}
impl<S: RealScalar> IrwinHall<S> {
    /// Creates an Irwin–Hall distribution with `n` terms.
    ///
    /// Construction costs O(n³) operations and stores O(n²) coefficients. Each
    /// density or CDF evaluation then costs O(n). Construction counts about
    /// `n³/3` units against [`LawOptions::work_budget`], so the default limit
    /// of 65,536 admits `n ≤ 56`; raise it for larger `n`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n = 0`.
    /// - `ResourceLimit` if the coefficient tables exceed [`LawOptions::max_bytes`].
    /// - `SizeOverflow` or `AllocationFailure` if the tables cannot be sized or allocated.
    /// - `ConvergenceFailure` if construction exhausts the work limit.
    pub fn new(n: usize) -> Result<Self, SciOxideError> {
        Self::with_options(n, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(n: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        if n == 0 {
            return Err(invalid(
                scioxide_core::argument!("n"),
                "Irwin-Hall order must be positive",
            ));
        }
        let end = S::try_from_usize_exact(n)?;
        let quantity = scioxide_core::quantity!("workspace_bytes");
        let coefficients =
            storage_bytes::<S>(checked_mul(checked_mul(n, n, quantity)?, 4, quantity)?)?;
        let headers = storage_bytes::<Vec<S>>(checked_mul(n, 4, quantity)?)?;
        let factorials = storage_bytes::<S>(checked_add(n, 1, quantity)?)?;
        admit_bytes(
            checked_add(
                checked_add(coefficients, headers, quantity)?,
                factorials,
                quantity,
            )?,
            options.max_bytes(),
        )?;
        // The density of the sum of m uniforms is a degree-(m - 1) polynomial
        // on each unit segment [j, j + 1]. Each segment is stored in the
        // Bernstein basis, whose coefficients are nonnegative, so evaluation
        // is a convex combination with no cancellation (unlike the
        // alternating-sign closed form sum (-1)^k C(n, k) (x - k)^(n-1)).
        // Order m is built from order m - 1 by convolving with U(0, 1): the
        // coefficients of segment j combine those of segments j - 1 and j
        // with degree-elevation weights.
        let left = &mut options.work_budget().get();
        let mut previous = Vec::new();
        previous.try_reserve_exact(1).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("irwin_hall_preparation"), 1)
        })?;
        let mut first = filled(1)?;
        first[0] = S::ONE;
        previous.push(first);
        for m in 2..=n {
            let mut next = Vec::new();
            next.try_reserve_exact(m).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("irwin_hall_preparation"),
                    m,
                )
            })?;
            let denom = number::<S>((m - 1) as f64)?;
            for j in 0..m {
                let mut row = filled(m)?;
                for (i, value) in row.iter_mut().enumerate() {
                    crate::law::charge(left)?;
                    let r = number::<S>(i as f64)? / denom;
                    if j < m - 1 {
                        let p = &previous[j];
                        if i < m - 1 {
                            *value += number::<S>(j as f64)? * (S::ONE - r) * p[i] / denom;
                        }
                        if i > 0 {
                            *value += number::<S>((j + 1) as f64)? * r * p[i - 1] / denom;
                        }
                    }
                    if j > 0 {
                        let p = &previous[j - 1];
                        if i < m - 1 {
                            *value += number::<S>((m - j) as f64)? * (S::ONE - r) * p[i] / denom;
                        }
                        if i > 0 {
                            *value += number::<S>((m - j - 1) as f64)? * r * p[i - 1] / denom;
                        }
                    }
                }
                next.push(row);
            }
            previous = next;
        }
        let mut cdf = Vec::new();
        cdf.try_reserve_exact(n).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("irwin_hall_preparation"), n)
        })?;
        // Integrating a degree-(n - 1) Bernstein polynomial yields degree n
        // with coefficients equal to running sums of the old ones divided by
        // n. Offsetting each segment by the accumulated mass `prefix` makes
        // the CDF continuous across segments, and the final total becomes the
        // normalizer, so the CDF reaches exactly one at x = n despite rounding.
        let mut prefix = S::ZERO;
        let scale = number::<S>(n as f64)?;
        for row in &previous {
            let mut integral = filled(n + 1)?;
            integral[0] = prefix;
            for (i, &v) in row.iter().enumerate() {
                crate::law::charge(left)?;
                integral[i + 1] = integral[i] + v / scale;
            }
            prefix = integral[n];
            cdf.push(integral);
        }
        crate::law::positive_result(prefix)?;
        let mut log_fact = filled(n + 1)?;
        for i in 1..=n {
            crate::law::charge(left)?;
            log_fact[i] = log_fact[i - 1] + number::<S>(i as f64)?.ln();
        }
        Ok(Self {
            options,
            n,
            end,
            pdf: previous,
            cdf,
            log_fact,
            normalizer: prefix,
        })
    }
    fn bernstein(&self, coeff: &[S], t: S, left: &mut usize) -> Result<S, SciOxideError> {
        let degree = coeff.len() - 1;
        if t == S::ZERO {
            return Ok(coeff[0]);
        }
        if t == S::ONE {
            return Ok(coeff[degree]);
        }
        // Bernstein weights C(d, i) t^i (1 - t)^(d - i) underflow at the ends
        // for large degree. Start at the largest weight (near i = d t),
        // computed in log space, and walk outward with exact term ratios so
        // every weight is formed relative to a representable one.
        let mode = ((degree + 1) as f64 * t.into_f64()).floor() as usize;
        let mode = mode.min(degree);
        let m = number::<S>(mode as f64)?;
        let d = number::<S>(degree as f64)?;
        let logw = self.log_fact[degree] - self.log_fact[mode] - self.log_fact[degree - mode]
            + m * t.ln()
            + (d - m) * (-t).ln_1p();
        let mut left_w = logw.exp();
        let mut total = coeff[mode] * left_w;
        for i in (1..=mode).rev() {
            crate::law::charge(left)?;
            left_w = left_w * number::<S>(i as f64)? / number::<S>((degree - i + 1) as f64)?
                * (S::ONE - t)
                / t;
            total += coeff[i - 1] * left_w;
        }
        let mut right_w = logw.exp();
        for i in mode..degree {
            crate::law::charge(left)?;
            right_w = right_w * number::<S>((degree - i) as f64)? / number::<S>((i + 1) as f64)?
                * t
                / (S::ONE - t);
            total += coeff[i + 1] * right_w;
        }
        finite(total)
    }
    fn segment(&self, x: S) -> Result<(usize, S), SciOxideError> {
        let j = x.floor().into_f64() as usize;
        Ok((j.min(self.n - 1), x - number::<S>(j as f64)?))
    }
}
impl<S: RealScalar> Formula<S> for IrwinHall<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), Some(self.end))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if self.n == 1 && x == S::ZERO {
            return Ok(S::ZERO);
        }
        if x <= S::ZERO || x >= self.end {
            return Ok(S::NEG_INFINITY);
        }
        let (j, t) = self.segment(x)?;
        Ok(
            (self.bernstein(&self.pdf[j], t, &mut self.options.work_budget().get())?
                / self.normalizer)
                .ln(),
        )
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            // The law is symmetric about n / 2, so the upper tail at x equals the
            // lower tail at n - x and reuses the accurate small-probability path.
            self.tail_with_work(self.end - x, false, left)
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            if x >= self.end {
                return Ok(S::ZERO);
            }
            let (j, t) = self.segment(x)?;
            Ok((self.bernstein(&self.cdf[j], t, left)? / self.normalizer).ln())
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;

    fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        if self.n == 1 && x == S::ZERO {
            return Ok(S::ZERO);
        }
        if x <= S::ZERO || x >= self.end {
            return Ok(S::NEG_INFINITY);
        }
        let (j, t) = self.segment(x)?;
        Ok((self.bernstein(&self.pdf[j], t, left)? / self.normalizer).ln())
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // Sampling follows the definition directly; it is exact in law and
        // cheaper than inverting the piecewise CDF.
        let mut rng = crate::distribution::words::Words::forwarding(
            rng,
            self.options.sample_word_budget().get(),
        );
        let mut sum = S::ZERO;
        for _ in 0..self.n {
            crate::law::charge(left)?;
            sum += crate::util::validated_unit_float(&mut rng, scioxide_core::argument!("rng"))?;
        }
        finite(sum)
    }
}
formula_law!(IrwinHall);
impl<S: RealScalar> Mean<S> for IrwinHall<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.end * S::HALF)
    }
}
impl<S: RealScalar> Variance<S> for IrwinHall<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        Ok(self.end / number::<S>(12.)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Cdf, Density, Survival};
    #[test]
    fn default_work_limit_admits_fifty_six_terms() {
        assert!(IrwinHall::<f64>::new(56).is_ok());
        assert_eq!(
            IrwinHall::<f64>::new(57).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        let options = LawOptions::default()
            .with_work_budget(scioxide_core::Budget::try_from(1 << 20).unwrap());
        assert!(IrwinHall::<f64>::with_options(57, options).is_ok());
    }
    #[test]
    fn spline_identities() {
        let one = IrwinHall::<f64>::new(1).unwrap();
        assert!((one.cdf(0.3).unwrap() - 0.3).abs() < 1e-13);
        let two = IrwinHall::<f64>::new(2).unwrap();
        assert!((two.pdf(0.5).unwrap() - 0.5).abs() < 1e-13);
        assert!((two.cdf(0.5).unwrap() - 0.125).abs() < 1e-13);
        let three = IrwinHall::<f64>::new(3).unwrap();
        assert!((three.cdf(1.5).unwrap() - 0.5).abs() < 1e-12);
        assert!((three.sf(1.5).unwrap() - 0.5).abs() < 1e-12);
        let larger = IrwinHall::<f64>::new(20).unwrap();
        assert!((larger.cdf(10.).unwrap() - 0.5).abs() < 1e-10);
    }
    #[test]
    fn preparation_storage_is_bounded_by_options() {
        // Order 2: 16 coefficients, 8 table headers and 3 factorials.
        let exact = 16 * 8 + 8 * core::mem::size_of::<Vec<f64>>() + 3 * 8;
        assert!(
            IrwinHall::<f64>::with_options(2, LawOptions::default().with_max_bytes(exact)).is_ok()
        );
        let error =
            IrwinHall::<f64>::with_options(2, LawOptions::default().with_max_bytes(exact - 1))
                .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            // A power of two is exactly representable, so sizing fails first.
            IrwinHall::<f64>::new(usize::MAX / 2 + 1)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::SizeOverflow
        );
    }
}
