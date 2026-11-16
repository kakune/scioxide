use super::simplex::SimplexParameters;
use super::*;
use crate::BetaBinomial;
use crate::distribution::LawOptions;
use crate::distribution::beta::BetaKernel;
use crate::distribution::discrete_math::log_choose;
use crate::distribution::finite_urn::central_draw;

/// Multivariate hypergeometric distribution: the category counts in `n` draws
/// without replacement from a population with `Kᵢ` items in category `i`.
///
/// With `N = Σ Kᵢ`, the mass of `x` with `Σ xᵢ = n` and `0 ≤ xᵢ ≤ Kᵢ` is
/// `Πᵢ C(Kᵢ, xᵢ) / C(N, n)`. The mean is `n Kᵢ / N`. A draw takes one univariate
/// hypergeometric draw per category, conditional on the earlier categories.
///
/// ```
/// use scioxide_random::{Mass, Mean, MultivariateHypergeometric, SeedableRng, SplitMix64};
/// let law = MultivariateHypergeometric::<f64>::new(&[2, 3, 4], 4)?;
/// // C(2, 1) C(3, 1) C(4, 2) / C(9, 4) = 36 / 126
/// assert!((law.pmf(&[1, 1, 2])? - 2.0 / 7.0).abs() < 1e-12);
/// assert!((law.mean()?[2] - 16.0 / 9.0).abs() < 1e-12);
/// let counts = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&counts)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateHypergeometric<S: RealScalar> {
    population: Vec<usize>,
    draws: usize,
    total: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> MultivariateHypergeometric<S> {
    /// Creates the law with category sizes `population` and `draws = n`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `population` is empty or `draws > N`.
    /// - `InvalidInput` if `N` overflows `usize`.
    /// - `NotImplemented` if `N > 2⁵³`.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new(population: &[usize], draws: usize) -> Result<Self, SciOxideError> {
        Self::with_options(population, draws, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        population: &[usize],
        draws: usize,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if population.is_empty() {
            return Err(crate::law::domain(
                scioxide_core::argument!("population"),
                "population must have categories",
            ));
        }
        let total = total(population, scioxide_core::argument!("population"))?;
        real_count(total)?;
        if draws > total {
            return Err(crate::law::domain(
                scioxide_core::argument!("draws"),
                "draws exceed population",
            ));
        }
        options.admit::<usize>(population.len())?;
        options.steps(population.len())?;
        let mut counts = scioxide_core::backend::try_with_capacity(
            population.len(),
            scioxide_core::quantity!("population"),
        )?;
        counts.extend_from_slice(population);
        Ok(Self {
            population: counts,
            draws,
            total,
            options,
        })
    }
    /// Returns the number of categories.
    pub fn dimension(&self) -> usize {
        self.population.len()
    }
    /// Returns the number of draws `n`.
    pub fn draws(&self) -> usize {
        self.draws
    }
    /// Returns the category sizes `Kᵢ`.
    pub fn population(&self) -> &[usize] {
        &self.population
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether `Σ xᵢ = n` and `xᵢ ≤ Kᵢ` for every category.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` does not have one entry per category.
    /// - `InvalidInput` if `Σ xᵢ` overflows `usize`.
    pub fn contains_support(&self, x: &[usize]) -> Result<bool, SciOxideError> {
        size(x.len(), self.dimension())?;
        Ok(total(x, scioxide_core::argument!("x"))? == self.draws
            && x.iter().zip(&self.population).all(|(x, m)| x <= m))
    }
    /// Returns the covariance matrix `n (N − n)/(N − 1) (diag(p) − p pᵀ)`, `pᵢ = Kᵢ / N`.
    ///
    /// It is zero when `N ≤ 1`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the result cannot be stored.
    /// - `NumericalInstability` if an entry is outside the range of `S`.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        self.options
            .admit::<S>(
                self.dimension()
                    .checked_mul(self.dimension())
                    .ok_or_else(|| {
                        invalid(scioxide_core::argument!("self"), "covariance size overflow")
                    })?,
            )?;
        let p = converted::<f64>(self.population.iter().map(|&m| {
            if self.total == 0 {
                0.0
            } else {
                m as f64 / self.total as f64
            }
        }))?;
        let scale = if self.total <= 1 {
            0.0
        } else {
            self.draws as f64 * (self.total - self.draws) as f64 / (self.total - 1) as f64
        };
        covariance(&p, scale)
    }
}
impl<S: RealScalar> LogMass<S, &[usize]> for MultivariateHypergeometric<S> {
    fn log_pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        number(hypergeometric_log(
            self.population.iter().copied(),
            self.total,
            self.draws,
            x,
        )?)
    }
}
/// Log mass of `x` drawn without replacement from categories of the given
/// sizes, as a product of conditional two-category masses. The last category
/// is determined by the others.
fn hypergeometric_log(
    population: impl Iterator<Item = usize>,
    total: usize,
    draws: usize,
    x: &[usize],
) -> Result<f64, SciOxideError> {
    let (mut remaining, mut draws, mut log) = (total, draws, 0.0);
    for (m, &k) in population.zip(x).take(x.len().saturating_sub(1)) {
        log += log_choose(m, k)? + log_choose(remaining - m, draws - k)?
            - log_choose(remaining, draws)?;
        remaining -= m;
        draws -= k;
    }
    Ok(log)
}
impl<S: RealScalar> Mass<S, &[usize]> for MultivariateHypergeometric<S> {
    fn pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        number(self.log_pmf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for MultivariateHypergeometric<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.population.iter().map(|&m| {
            if self.total == 0 {
                0.0
            } else {
                self.draws as f64 * (m as f64 / self.total as f64)
            }
        }))
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for MultivariateHypergeometric<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.population.iter().map(|&m| {
            if self.total <= 1 {
                0.0
            } else {
                let p = m as f64 / self.total as f64;
                self.draws as f64 * p * (1.0 - p) * (self.total - self.draws) as f64
                    / (self.total - 1) as f64
            }
        }))
    }
}
impl<S: RealScalar> RowLaw<S, usize> for MultivariateHypergeometric<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [usize],
        _: Scratch<'_, usize>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        // Sequential univariate hypergeometric draws: category i versus all
        // categories not yet decided, with the draw count reduced as we go.
        // The last category takes whatever draws remain, so the total is exact.
        let mut remaining = self.total;
        let mut draws = self.draws;
        let mut work = self.options.work.get();
        for (i, &m) in self.population.iter().enumerate().take(out.len() - 1) {
            out[i] = central_draw(remaining, m, draws, &mut words, &mut work)?;
            remaining -= m;
            draws -= out[i];
        }
        out[out.len() - 1] = draws;
        Ok(())
    }
}
vector_sampling!(MultivariateHypergeometric, usize, 0);

/// Random `r × c` contingency tables with fixed row sums `aᵢ` and column sums `bⱼ`,
/// under independence of rows and columns.
///
/// With total `N = Σ aᵢ = Σ bⱼ`, the mass of a table `x` with these margins is
///
/// ```text
/// p(x) = Πᵢ aᵢ! Πⱼ bⱼ! / (N! Πᵢⱼ xᵢⱼ!)
/// ```
///
/// A draw fills each row with a multivariate hypergeometric draw of its row sum
/// from the column sums left by earlier rows. Tables are stored row by row.
///
/// ```
/// use scioxide_random::{Mass, Mean, RandomTable, SeedableRng, SplitMix64};
/// let law = RandomTable::<f64>::new(&[2, 2], &[2, 2])?;
/// assert_eq!(law.shape(), (2, 2));
/// // Table [[1, 1], [1, 1]]: 2! 2! 2! 2! / (4! 1! 1! 1! 1!) = 2/3.
/// assert!((law.pmf(&[1, 1, 1, 1])? - 2.0 / 3.0).abs() < 1e-12);
/// assert!(law.mean()?.iter().all(|m| (m - 1.0).abs() < 1e-12));
/// let table = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&table)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RandomTable<S: RealScalar> {
    rows: Vec<usize>,
    cols: Vec<usize>,
    total: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> RandomTable<S> {
    /// Creates the law with row sums `rows` and column sums `cols`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `cols` is empty, if their totals differ or
    ///   overflow.
    /// - `NotImplemented` if `N > 2⁵³`.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new(rows: &[usize], cols: &[usize]) -> Result<Self, SciOxideError> {
        Self::with_options(rows, cols, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        rows: &[usize],
        cols: &[usize],
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if rows.is_empty() || cols.is_empty() {
            return Err(invalid(
                if rows.is_empty() {
                    scioxide_core::argument!("rows")
                } else {
                    scioxide_core::argument!("cols")
                },
                "table margins must be nonempty",
            ));
        }
        let sum = total(rows, scioxide_core::argument!("rows"))?;
        if sum != total(cols, scioxide_core::argument!("cols"))? {
            return Err(invalid(
                scioxide_core::argument!("cols"),
                "table margin totals differ",
            ));
        }
        real_count(sum)?;
        let width = rows
            .len()
            .checked_mul(cols.len())
            .ok_or_else(|| invalid(scioxide_core::argument!("cols"), "table shape overflow"))?;
        options.admit::<usize>(width)?;
        options.steps(width)?;
        let mut r = scioxide_core::backend::try_with_capacity(
            rows.len(),
            scioxide_core::quantity!("row_margins"),
        )?;
        r.extend_from_slice(rows);
        let mut c = scioxide_core::backend::try_with_capacity(
            cols.len(),
            scioxide_core::quantity!("column_margins"),
        )?;
        c.extend_from_slice(cols);
        Ok(Self {
            rows: r,
            cols: c,
            total: sum,
            options,
        })
    }
    /// Returns the shape `(r, c)` of a sample.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.cols.len())
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether a table, stored row by row, has the law's margins.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ r c`.
    /// - `InvalidInput` if a row or column sum overflows `usize`.
    pub fn contains_support(&self, x: &[usize]) -> Result<bool, SciOxideError> {
        size(x.len(), self.width())?;
        let n = self.cols.len();
        for (row, &margin) in x.chunks_exact(n).zip(&self.rows) {
            if total(row, scioxide_core::argument!("x"))? != margin {
                return Ok(false);
            }
        }
        for (j, &margin) in self.cols.iter().enumerate() {
            let mut sum = 0usize;
            for row in x.chunks_exact(n) {
                sum = sum.checked_add(row[j]).ok_or_else(|| {
                    invalid(scioxide_core::argument!("x"), "table total overflow")
                })?;
            }
            if sum != margin {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
impl<S: RealScalar> LogMass<S, &[usize]> for RandomTable<S> {
    fn log_pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        // Row i is a multivariate hypergeometric draw of its margin from the
        // column totals left by the earlier rows; the last row is determined.
        let width = self.cols.len();
        let mut total = self.total;
        let mut log = 0.0;
        for (i, (&draws, row)) in self.rows.iter().zip(x.chunks_exact(width)).enumerate() {
            if i + 1 == self.rows.len() {
                break;
            }
            let population = (0..width).map(|j| {
                self.cols[j] - x[..i * width].iter().skip(j).step_by(width).sum::<usize>()
            });
            log += hypergeometric_log(population, total, draws, row)?;
            total -= draws;
        }
        number(log)
    }
}
impl<S: RealScalar> Mass<S, &[usize]> for RandomTable<S> {
    fn pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        number(self.log_pmf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for RandomTable<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.rows.iter().flat_map(|&r| {
            self.cols.iter().map(move |&c| {
                if self.total == 0 {
                    0.0
                } else {
                    r as f64 * (c as f64 / self.total as f64)
                }
            })
        }))
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for RandomTable<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.rows.iter().flat_map(|&r| {
            self.cols.iter().map(move |&c| {
                if self.total <= 1 {
                    0.0
                } else {
                    let p = c as f64 / self.total as f64;
                    r as f64 * p * (1.0 - p) * (self.total - r) as f64 / (self.total - 1) as f64
                }
            })
        }))
    }
}
impl<S: RealScalar> RowLaw<S, usize> for RandomTable<S> {
    fn width(&self) -> usize {
        self.rows.len() * self.cols.len()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [usize],
        _: Scratch<'_, usize>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        // Fixed-margin tables under independence (Patefield-style): each row is a
        // multivariate hypergeometric draw of its row total from the remaining
        // column margins, which are then depleted. The final row is forced to
        // the leftover column margins, so both margin sets hold exactly; it
        // holds those margins while they deplete, so no scratch is needed.
        let n = self.cols.len();
        let (body, cols) = out.split_at_mut((self.rows.len() - 1) * n);
        cols.copy_from_slice(&self.cols);
        let mut population = self.total;
        let mut work = self.options.work.get();
        for (i, &r) in self.rows.iter().enumerate().take(self.rows.len() - 1) {
            let mut draws = r;
            let mut remaining = population;
            for j in 0..n - 1 {
                let value = central_draw(remaining, cols[j], draws, &mut words, &mut work)?;
                body[i * n + j] = value;
                remaining -= cols[j];
                cols[j] -= value;
                draws -= value;
            }
            body[i * n + n - 1] = draws;
            cols[n - 1] -= draws;
            population -= r;
        }
        Ok(())
    }
}
vector_sampling!(RandomTable, usize, 0);

impl<S: RealScalar> RandomTable<S> {
    /// Computes the log mass of a table given as a borrowed, possibly strided view.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the view is not `r × c`, or if a margin of `x` overflows `usize`.
    /// - `ResourceLimit` or `AllocationFailure` if the table copy cannot be stored.
    pub fn log_pmf_view(
        &self,
        x: scioxide_core::NdView<'_, '_, usize>,
    ) -> Result<S, SciOxideError> {
        let (rows, cols) = self.shape();
        if x.layout().shape().extents() != [rows, cols] {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "table view shape mismatch",
            ));
        }
        self.options.admit::<usize>(self.width())?;
        let mut flat = scioxide_core::backend::try_with_capacity(
            self.width(),
            scioxide_core::quantity!("table_query_view"),
        )?;
        flat.extend(x.iter().copied());
        self.log_pmf(flat.as_slice())
    }
}
impl<S: RealScalar> RandomTable<S> {
    /// Computes the mass of a table given as a borrowed, possibly strided view.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::log_pmf_view`].
    pub fn pmf_view(&self, x: scioxide_core::NdView<'_, '_, usize>) -> Result<S, SciOxideError> {
        number(self.log_pmf_view(x)?.into_f64().exp())
    }
}

/// Dirichlet-multinomial distribution with `n` trials and concentrations `α₁, …, α_k > 0`.
///
/// The counts are multinomial with probabilities drawn from a Dirichlet(α) law.
/// With `α₀ = Σ αᵢ`, the mass of `x` with `Σ xᵢ = n` is
///
/// ```text
/// p(x) = n! Γ(α₀) / Γ(n + α₀) · Πᵢ Γ(xᵢ + αᵢ) / (xᵢ! Γ(αᵢ))
/// ```
///
/// and the mean is `n αᵢ / α₀`.
///
/// ```
/// use scioxide_random::{DirichletMultinomial, Mass, Mean, SeedableRng, SplitMix64};
/// let law = DirichletMultinomial::new(4, &[1.0_f64, 1.0, 1.0])?;
/// // Unit concentrations make all 15 count vectors with total 4 equally likely.
/// assert!((law.pmf(&[1, 1, 2])? - 1.0 / 15.0).abs() < 1e-12);
/// assert!((law.mean()?[0] - 4.0 / 3.0).abs() < 1e-12);
/// let counts = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&counts)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct DirichletMultinomial<S: RealScalar> {
    base: SimplexParameters<S>,
    trials: usize,
    suffix: Vec<f64>,
    kernels: Vec<BetaKernel<f64>>,
}
impl<S: RealScalar> DirichletMultinomial<S> {
    /// Creates the law with `n = trials` and concentrations `alpha`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `alpha` is empty.
    /// - `NonFinite` if a concentration is NaN or infinite.
    /// - `DomainError` if a concentration is not positive.
    /// - `NotImplemented` if `trials > 2⁵³`.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new(trials: usize, alpha: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(trials, alpha, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        trials: usize,
        alpha: &[S],
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        real_count(trials)?;
        let base = SimplexParameters::new(alpha, options)?;
        let q = scioxide_core::quantity!("dirichlet_multinomial_parameters");
        options.admit::<u8>(scioxide_core::backend::checked_add(
            crate::workspace::storage_bytes::<[f64; 2]>(alpha.len())?,
            crate::workspace::storage_bytes::<BetaKernel<f64>>(alpha.len() - 1)?,
            q,
        )?)?;
        let mut suffix = scioxide_core::backend::try_filled(
            alpha.len(),
            0.0,
            scioxide_core::quantity!("dirichlet_multinomial_suffix"),
        )?;
        let mut rest = 0.0;
        for (slot, &a) in suffix.iter_mut().zip(&base.alpha).rev() {
            *slot = rest;
            rest += a;
        }
        let mut kernels = scioxide_core::backend::try_with_capacity(
            alpha.len() - 1,
            scioxide_core::quantity!("dirichlet_multinomial_kernels"),
        )?;
        for (&a, &b) in base.alpha.iter().zip(&suffix).take(alpha.len() - 1) {
            kernels.push(BetaKernel::new(a, b)?);
        }
        let law = Self {
            base,
            trials,
            suffix,
            kernels,
        };
        crate::workspace::admit_bytes(law.parameter_bytes()?, options.max_bytes())?;
        Ok(law)
    }
    fn parameter_bytes(&self) -> Result<usize, SciOxideError> {
        use crate::workspace::storage_bytes;
        let q = scioxide_core::quantity!("dirichlet_multinomial_parameters");
        scioxide_core::backend::checked_add(
            storage_bytes::<f64>(self.base.alpha.capacity())?,
            scioxide_core::backend::checked_add(
                storage_bytes::<f64>(self.suffix.capacity())?,
                storage_bytes::<BetaKernel<f64>>(self.kernels.capacity())?,
                q,
            )?,
            q,
        )
    }
    /// Returns the number of trials `n`.
    pub fn trials(&self) -> usize {
        self.trials
    }
    /// Returns the number of categories `k`.
    pub fn dimension(&self) -> usize {
        self.base.dimension()
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.base.options
    }
    /// Tests whether `Σ xᵢ = n`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ k`.
    /// - `InvalidInput` if `Σ xᵢ` overflows `usize`.
    pub fn contains_support(&self, x: &[usize]) -> Result<bool, SciOxideError> {
        size(x.len(), self.dimension())?;
        Ok(total(x, scioxide_core::argument!("x"))? == self.trials)
    }
    /// Returns the covariance matrix `n (n + α₀)/(1 + α₀) (diag(p) − p pᵀ)`, `pᵢ = αᵢ / α₀`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the result cannot be stored.
    /// - `NumericalInstability` if an entry is outside the range of `S`.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        self.options()
            .admit::<S>(
                self.dimension()
                    .checked_mul(self.dimension())
                    .ok_or_else(|| {
                        invalid(scioxide_core::argument!("self"), "covariance size overflow")
                    })?,
            )?;
        let n = self.trials as f64;
        let p = converted::<f64>(self.base.alpha.iter().map(|a| a / self.base.sum))?;
        covariance(&p, n * ((n + self.base.sum) / (1.0 + self.base.sum)))
    }
}
impl<S: RealScalar> LogMass<S, &[usize]> for DirichletMultinomial<S> {
    fn log_pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        // Conditional beta-binomial probabilities avoid differences of huge gamma sums.
        let mut n = self.trials;
        let mut log = 0.0;
        for (i, &count) in x[..x.len() - 1].iter().enumerate() {
            let b = self.suffix[i];
            log += BetaBinomial::new(n, self.base.alpha[i], b)?.log_pmf(count)?;
            n -= count;
        }
        number(log)
    }
}
impl<S: RealScalar> Mass<S, &[usize]> for DirichletMultinomial<S> {
    fn pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        number(self.log_pmf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for DirichletMultinomial<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        converted(
            self.base
                .alpha
                .iter()
                .map(|a| self.trials as f64 * (a / self.base.sum)),
        )
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for DirichletMultinomial<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        let n = self.trials as f64;
        converted(self.base.alpha.iter().map(|a| {
            let p = a / self.base.sum;
            n * p * (1.0 - p) * ((n + self.base.sum) / (1.0 + self.base.sum))
        }))
    }
}
impl<S: RealScalar> RowLaw<S, usize> for DirichletMultinomial<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.base.options
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.parameter_bytes()
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [usize],
        _: Scratch<'_, usize>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options().sample_words.get());
        // Sequential conditioning: marginally X_i given the remaining trials is
        // beta-binomial(left, alpha_i, sum of later alphas). This stays exact
        // without drawing the continuous Dirichlet vector, which could round
        // small components to zero.
        let mut left = self.trials;
        let mut work = self.options().work_budget().get();
        out.fill(0);
        let last = out.len() - 1;
        for (value, beta) in out[..last].iter_mut().zip(&self.kernels) {
            if left == 0 {
                break;
            }
            let odds = beta.sample_log_odds(&mut words)?;
            let reflected = odds > 0.0;
            let small = (-odds.abs()).exp();
            let draw = super::super::binomial::conditional_binomial(
                left,
                small / (1.0 + small),
                &mut words,
                &mut work,
            )?;
            *value = if reflected { left - draw } else { draw };
            left -= *value;
        }
        out[out.len() - 1] = left;
        Ok(())
    }
}
vector_sampling!(DirichletMultinomial, usize, 0);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Hypergeometric, LogMass};

    // At two-million margins the old difference of log-factorials lost about
    // 1e-9 absolute accuracy; the guarded kernel agrees with the multivariate
    // hypergeometric row law exactly and with Loader's univariate mass.
    #[test]
    fn random_table_log_mass_uses_the_guarded_kernel_at_large_margins() {
        let rows = [700_000, 1_300_000];
        let cols = [900_000, 1_100_000];
        let table = RandomTable::<f64>::new(&rows, &cols).unwrap();
        let x = [315_123, 384_877, 584_877, 715_123];
        let log = table.log_pmf(&x[..]).unwrap();
        let row = MultivariateHypergeometric::<f64>::new(&cols, rows[0])
            .unwrap()
            .log_pmf(&x[..2])
            .unwrap();
        assert_eq!(log, row);
        let univariate = Hypergeometric::<f64>::new(2_000_000, cols[0], rows[0])
            .unwrap()
            .log_pmf(x[0])
            .unwrap();
        assert!(
            (log - univariate).abs() <= 1e-11 * univariate.abs(),
            "{log} {univariate}"
        );
        // Three rows chain the column totals left by earlier rows.
        let table = RandomTable::<f64>::new(&[2, 3, 4], &[4, 5]).unwrap();
        let mass = table.log_pmf(&[1, 1, 2, 1, 1, 3][..]).unwrap().exp();
        // C(4,1)C(5,1)/C(9,2) * C(3,2)C(4,1)/C(7,3) = 20/36 * 12/35.
        assert!((mass - 20.0 / 36.0 * 12.0 / 35.0).abs() < 1e-14);
    }
}
