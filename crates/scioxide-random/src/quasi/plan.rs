//! Quasi-random sequence plans and cursors.

use super::{DirectionNumberSet, direction_numbers, halton, niederreiter, scramble};
use crate::{QuasiRng, SnapshotGenerator};
use alloc::vec::Vec;
use scioxide_core::{InvalidInputReason, RealScalar, SciOxideError};

/// Low-discrepancy sequence family of a [`QmcPlan`].
///
/// The discriminants are the family tags stored in checkpoints. See [`QmcPlan`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QmcFamily {
    /// Sobol sequence in base 2, with Joe–Kuo or user-supplied direction numbers, enumerated in
    /// Gray-code order.
    Sobol = 32,
    /// Halton sequence: coordinate `j` of point `n` is the radical inverse of `n` in the prime
    /// base `bⱼ`.
    Halton = 33,
    /// Reverse Halton sequence: Halton with each nonzero digit `a` replaced by `b − a`
    /// (Vandewoestyne and Cools 2006).
    ReverseHalton = 34,
    /// Niederreiter sequence in base 2 (Bratley, Fox and Niederreiter 1992), enumerated in
    /// Gray-code order.
    NiederreiterBase2 = 35,
}

/// Randomization of a [`QmcPlan`] determined by a 64-bit seed.
///
/// Apply it with [`QmcPlan::with_scramble`]. The same seed always gives the same points. The
/// families each variant accepts are listed in the table on [`QmcPlan`]. A randomization drawn
/// from a generator is the separate type [`QmcRandomization`]:
///
/// ```compile_fail
/// use scioxide_random::{QmcPlan, QmcRandomization};
/// let plan = QmcPlan::sobol(2)?.with_scramble(QmcRandomization::LinearMatrixShift)?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QmcScramble {
    /// Random digital shift: XOR of each 64-bit coordinate with a word derived from the seed
    /// (Sobol, Niederreiter base 2). It keeps the net structure. For a uniformly random seed,
    /// each coordinate is uniform on its decoded finite grid. A single seed
    /// does not give independent shifts across coordinates.
    DigitalShift(u64),
    /// Random digit permutation (Halton): for each base `b`, one random permutation of the
    /// digits `1, …, b − 1` drawn from the seed, applied at every digit position. Digit 0 stays
    /// 0, so the radical inverse remains a finite sum.
    DigitPermutation(u64),
}

/// Randomization of a [`QmcPlan`] drawn from a generator.
///
/// Apply it with [`QmcPlan::with_randomization`]. The families each variant accepts are listed
/// in the table on [`QmcPlan`]. A randomization determined by a seed is the separate type
/// [`QmcScramble`]:
///
/// ```compile_fail
/// use scioxide_random::{QmcPlan, QmcScramble, SeedableRng, SplitMix64};
/// let plan = QmcPlan::sobol(2)?
///     .with_randomization(QmcScramble::DigitalShift(7), &mut SplitMix64::from_seed(1))?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QmcRandomization {
    /// Linear matrix scramble with digital shift (Matoušek 1998), for Sobol and Niederreiter
    /// base 2.
    ///
    /// Each coordinate is multiplied over GF(2) by a random nonsingular lower-triangular 64 × 64
    /// binary matrix and XORed with a random shift. It draws 65 words per coordinate. The
    /// scrambled points keep the net structure and are uniform on `[0, 1)ᵈ`.
    LinearMatrixShift,
    /// Random shift modulo 1 (Cranley and Patterson 1976), for Halton.
    ///
    /// Each coordinate `x` becomes `(x + uⱼ) mod 1` with one uniform `uⱼ` per coordinate. It
    /// draws one word per coordinate.
    RandomShift,
}

/// The scramble a plan has applied; the discriminants are the stable
/// checkpoint tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Applied {
    None,
    DigitalShift(u64),
    LinearMatrixShift,
    RandomShift,
    DigitPermutation(u64),
    CustomPermutation,
}
impl Applied {
    /// Checkpoint tag and seed; unseeded choices store seed zero.
    pub(super) const fn tag(self) -> (u8, u64) {
        match self {
            Self::None => (0, 0),
            Self::DigitalShift(seed) => (1, seed),
            Self::LinearMatrixShift => (2, 0),
            Self::RandomShift => (3, 0),
            Self::DigitPermutation(seed) => (4, seed),
            Self::CustomPermutation => (5, 0),
        }
    }
    /// The choice of a checkpoint tag and seed.
    pub(super) const fn from_tag(tag: u8, seed: u64) -> Option<Self> {
        Some(match tag {
            0 => Self::None,
            1 => Self::DigitalShift(seed),
            2 => Self::LinearMatrixShift,
            3 => Self::RandomShift,
            4 => Self::DigitPermutation(seed),
            5 => Self::CustomPermutation,
            _ => return None,
        })
    }
}
impl From<QmcScramble> for Applied {
    fn from(method: QmcScramble) -> Self {
        match method {
            QmcScramble::DigitalShift(seed) => Self::DigitalShift(seed),
            QmcScramble::DigitPermutation(seed) => Self::DigitPermutation(seed),
        }
    }
}
impl From<QmcRandomization> for Applied {
    fn from(method: QmcRandomization) -> Self {
        match method {
            QmcRandomization::LinearMatrixShift => Self::LinearMatrixShift,
            QmcRandomization::RandomShift => Self::RandomShift,
        }
    }
}

/// The one family compatibility rule, tabulated on [`QmcPlan`]: digital
/// families take digital shifts and linear matrix scrambles, Halton takes
/// digit permutations and random shifts, and reverse Halton keeps its
/// defining permutation.
pub(super) const fn compatible(family: QmcFamily, applied: Applied) -> bool {
    matches!(
        (family, applied),
        (_, Applied::None)
            | (
                QmcFamily::Sobol | QmcFamily::NiederreiterBase2,
                Applied::DigitalShift(_) | Applied::LinearMatrixShift
            )
            | (
                QmcFamily::Halton,
                Applied::DigitPermutation(_) | Applied::RandomShift | Applied::CustomPermutation
            )
    )
}

fn incompatible(argument: scioxide_core::Argument) -> SciOxideError {
    invalid(
        argument,
        "scramble method is incompatible with the QMC family",
    )
}

#[derive(Debug, Clone)]
pub(super) enum QmcData {
    Sobol {
        directions: Vec<[u64; 64]>,
        source: DirectionNumberSet,
    },
    Niederreiter {
        directions: Vec<[u64; 64]>,
    },
    Radical {
        bases: Vec<u32>,
        dividers: Vec<halton::Base>,
        permutations: Vec<Vec<u32>>,
    },
}

/// Low-discrepancy (quasi-random) sequence in `[0, 1)ᵈ`, optionally randomized.
///
/// A plan holds the generator data of one sequence: 64 direction numbers per coordinate for Sobol
/// and Niederreiter base 2, prime bases for Halton and reverse Halton, and the randomization. A
/// linear matrix scramble is applied to the direction numbers once, when the plan is randomized,
/// at a cost of O(64² d) word operations. Point `n`, for `n = 0, 1, …, 2⁶⁴ − 1`, is a
/// fixed function of `n` and the plan. Draw points through a [`QmcSequence`] cursor from
/// [`Self::sequence`], or one at a time with [`Self::point_into`]. The plan is immutable and does
/// not depend on the output type, so one plan serves `f32` and `f64` cursors on any number of
/// threads.
///
/// - Sobol ([`Self::sobol`]) and Niederreiter base 2 ([`Self::niederreiter_base2`]) are digital
///   sequences in base 2 with 64-bit generator matrices. They are enumerated in Gray-code order
///   (Antonov and Saleev 1979). The first `2ᵐ` points are the same set as in natural order, and
///   without randomization point 0 is the origin. A point costs O(d) from the previous one, and
///   seeking to index `n` costs O(d · popcount(g)) for its Gray code `g = n ^ (n >> 1)`.
/// - Halton ([`Self::halton`]) takes coordinate `j` of point `n` as the radical inverse of `n`
///   in the prime base `bⱼ`. Reverse Halton ([`Self::reverse_halton`]) replaces each nonzero
///   digit `a` by `b − a`. A point costs O(d log n).
///
/// Independent coordinate shifts make each point uniform under the ideal continuous-uniform
/// model while keeping the structure of the point set. This gives unbiased randomized
/// estimates with an empirical error estimate ([`QmcReplicatePlan`](crate::QmcReplicatePlan));
/// finite output grids add discretization error. A seeded digital shift has uniform coordinate
/// marginals, but a single seed does not establish their independence.
/// Digit permutations, seeded or your own, change the points but do not make them uniform.
/// [`Self::with_scramble`] applies a randomization fixed by a seed, [`Self::with_randomization`]
/// one drawn from a generator, and [`Self::with_permutations`] your own digit permutations. Each
/// call replaces any earlier choice. The accepted combinations are:
///
/// | Family | [`QmcScramble`] | [`QmcRandomization`] | permutations |
/// |---|---|---|---|
/// | Sobol, Niederreiter base 2 | `DigitalShift` | `LinearMatrixShift` | no |
/// | Halton | `DigitPermutation` | `RandomShift` | yes |
/// | Reverse Halton | none | none | no |
///
/// Other combinations are `InvalidInput`. See [`QmcSequence`] for batch generation and
/// [`QmcCheckpoint`](crate::QmcCheckpoint) for a checkpoint round trip.
///
/// # Accuracy
///
/// For an integrand of bounded variation `V` in the sense of Hardy and Krause, the error of the
/// equal-weight average over the first `N` points is at most `V · D*_N` (Koksma–Hlawka
/// inequality), where the star discrepancy `D*_N` of these sequences is `O((log N)ᵈ / N)`.
/// The bound gives no usable error estimate for a given integrand. Use randomized replicates for
/// that. The constants grow quickly with `d`, and for Halton the finite prefixes of large bases
/// are strongly correlated.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     QmcFamily, QmcPlan, QmcRandomization, QmcScramble, QuasiRng, SeedableRng, SplitMix64,
/// };
/// let sobol = QmcPlan::sobol(3)?.with_scramble(QmcScramble::DigitalShift(7))?;
/// let halton = QmcPlan::halton(3)?.with_scramble(QmcScramble::DigitPermutation(7))?;
/// let reverse = QmcPlan::reverse_halton(3)?;
/// let niederreiter = QmcPlan::niederreiter_base2(3)?
///     .with_randomization(QmcRandomization::LinearMatrixShift, &mut SplitMix64::from_seed(42))?;
/// assert_eq!(reverse.bases(), Some(&[2, 3, 5][..]));
/// assert_eq!(niederreiter.family(), QmcFamily::NiederreiterBase2);
/// assert!(QmcPlan::halton(3)?.with_scramble(QmcScramble::DigitalShift(7)).is_err());
/// assert_eq!(niederreiter.randomization_words().len(), 3 * 65);
/// for plan in [&sobol, &halton, &reverse, &niederreiter] {
///     let point = plan.sequence::<f64>()?.next_point()?;
///     assert!(point.iter().all(|x| (0.0..1.0).contains(x)));
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - H. Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM, 1992.
/// - I. A. Antonov and V. M. Saleev, "An economic method of computing LPτ-sequences", *USSR
///   Comput. Math. Math. Phys.* 19(1), 1979, 252–256.
/// - J. Dick and F. Pillichshammer, *Digital Nets and Sequences*, Cambridge University Press,
///   2010.
#[derive(Debug, Clone)]
pub struct QmcPlan {
    pub(super) family: QmcFamily,
    pub(super) data: QmcData,
    pub(super) scramble: Applied,
    pub(super) randomization: Vec<u64>,
    digital: DigitalPreparation,
}

#[derive(Debug, Clone, Default)]
struct DigitalPreparation {
    columns: Vec<[u64; 64]>,
    shifts: Vec<u64>,
}

impl QmcPlan {
    /// Creates a Sobol sequence in `dimensions` coordinates with Joe and Kuo's D(6) direction
    /// numbers.
    ///
    /// The D(6) table covers up to 21,201 coordinates and is chosen for good two-dimensional
    /// projections (Joe and Kuo 2008). The leading coordinates have the best projections, so put
    /// the most important variables first.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimensions` is zero or exceeds 21,201.
    /// - `AllocationFailure` if the direction table cannot be allocated.
    ///
    /// # References
    ///
    /// - I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of
    ///   integrals", *USSR Comput. Math. Math. Phys.* 7(4), 1967, 86–112.
    /// - S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
    ///   projections", *SIAM J. Sci. Comput.* 30(5), 2008, 2635–2654.
    pub fn sobol(dimensions: usize) -> Result<Self, SciOxideError> {
        Self::sobol_with_direction_numbers(dimensions, DirectionNumberSet::JoeKuoD6)
    }

    /// Creates a Sobol sequence in `dimensions` coordinates with the given direction numbers.
    ///
    /// With [`DirectionNumberSet::Custom`], coordinate 1 is the van der Corput sequence and
    /// coordinates `2, …, d` use the first `d − 1` definitions. The quality of the projections
    /// depends on your direction numbers. Custom definitions are stored in the plan and its
    /// checkpoints.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimensions` is zero or exceeds the coordinates the source defines.
    /// - `AllocationFailure` if the direction table cannot be allocated.
    pub fn sobol_with_direction_numbers(
        dimensions: usize,
        mut source: DirectionNumberSet,
    ) -> Result<Self, SciOxideError> {
        nonzero(dimensions)?;
        let directions = direction_numbers::generate_direction_numbers(dimensions, &source)?;
        if let DirectionNumberSet::Custom(definitions) = &mut source {
            definitions.truncate(dimensions - 1);
        }
        Ok(Self {
            family: QmcFamily::Sobol,
            data: QmcData::Sobol { directions, source },
            scramble: Applied::None,
            randomization: Vec::new(),
            digital: DigitalPreparation::default(),
        })
    }

    /// Creates a Halton sequence on the first `dimensions` primes `2, 3, 5, …`.
    ///
    /// Coordinate `j` of point `n` is the radical inverse `φ_b(n) = Σₖ aₖ b^(−k−1)`, where
    /// `n = Σₖ aₖ bᵏ` in the `j`-th prime base `b` (Halton 1960). Prefixes of the sequence in large
    /// bases are strongly correlated between coordinates. A digit permutation
    /// ([`QmcScramble::DigitPermutation`]) reduces this but does not make estimates unbiased.
    /// Use [`QmcRandomization::RandomShift`] for unbiased replicates.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimensions` is outside `1..=100`.
    /// - `AllocationFailure` if the bases cannot be allocated.
    ///
    /// # References
    ///
    /// J. H. Halton, "On the efficiency of certain quasi-random sequences of points in evaluating
    /// multi-dimensional integrals", *Numer. Math.* 2, 1960, 84–90.
    pub fn halton(dimensions: usize) -> Result<Self, SciOxideError> {
        Self::halton_with_bases(halton::default_bases(dimensions)?)
    }

    /// Creates a Halton sequence with your own prime bases, one per coordinate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `bases` is empty, repeats a base, or contains a number that is not
    ///   prime.
    /// - `AllocationFailure` if the bases cannot be stored.
    pub fn halton_with_bases(bases: Vec<u32>) -> Result<Self, SciOxideError> {
        halton::validate_bases(&bases)?;
        let mut dividers = scioxide_core::backend::try_with_capacity(
            bases.len(),
            scioxide_core::quantity!("halton_dividers"),
        )?;
        dividers.extend(bases.iter().copied().map(halton::Base::new));
        Ok(Self {
            family: QmcFamily::Halton,
            data: QmcData::Radical {
                bases,
                dividers,
                permutations: Vec::new(),
            },
            scramble: Applied::None,
            randomization: Vec::new(),
            digital: DigitalPreparation::default(),
        })
    }

    /// Creates a reverse Halton sequence on the first `dimensions` primes `2, 3, 5, …`.
    ///
    /// Each nonzero digit `a` of `n` in base `b` is replaced by `b − a` before the radical
    /// inverse is taken. This fixed permutation reduces the correlation between coordinates in
    /// large bases (Vandewoestyne and Cools 2006). The plan accepts no further randomization.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimensions` is outside `1..=100`.
    /// - `AllocationFailure` if the bases cannot be allocated.
    ///
    /// # References
    ///
    /// B. Vandewoestyne and R. Cools, "Good permutations for deterministic scrambled Halton
    /// sequences in terms of L2-discrepancy", *J. Comput. Appl. Math.* 189, 2006, 341–361.
    pub fn reverse_halton(dimensions: usize) -> Result<Self, SciOxideError> {
        Self::reverse_halton_with_bases(halton::default_bases(dimensions)?)
    }

    /// Creates a reverse Halton sequence with your own prime bases, one per coordinate.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::halton_with_bases`].
    pub fn reverse_halton_with_bases(bases: Vec<u32>) -> Result<Self, SciOxideError> {
        let mut plan = Self::halton_with_bases(bases)?;
        plan.family = QmcFamily::ReverseHalton;
        Ok(plan)
    }

    /// Creates a Niederreiter sequence in base 2 in 1 to 12 coordinates.
    ///
    /// The generator matrices follow Niederreiter's construction as implemented by Bratley, Fox
    /// and Niederreiter (1992), with 64 output digits. Coordinate `j` uses the `j`-th monic
    /// irreducible polynomial over GF(2), ordered by degree and then by the integer whose bit
    /// `k` is the coefficient of `xᵏ`: `x`, `x + 1`, `x² + x + 1`, `x³ + x + 1`, `x³ + x² + 1`, ….
    /// The Laurent-series coefficients that the construction leaves free are set to 1.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimensions` is outside `1..=12`.
    /// - `AllocationFailure` if the generator matrices cannot be allocated.
    ///
    /// # References
    ///
    /// P. Bratley, B. L. Fox and H. Niederreiter, "Implementation and tests of low-discrepancy
    /// sequences", *ACM Trans. Model. Comput. Simul.* 2(3), 1992, 195–213.
    pub fn niederreiter_base2(dimensions: usize) -> Result<Self, SciOxideError> {
        let directions = niederreiter::directions(dimensions)?;
        Ok(Self {
            family: QmcFamily::NiederreiterBase2,
            data: QmcData::Niederreiter { directions },
            scramble: Applied::None,
            randomization: Vec::new(),
            digital: DigitalPreparation::default(),
        })
    }

    /// Returns the plan with the seeded randomization `method`, replacing any earlier one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the family does not accept `method` (see the table on [`QmcPlan`]).
    /// - `AllocationFailure` if the randomization data cannot be allocated.
    pub fn with_scramble(mut self, method: QmcScramble) -> Result<Self, SciOxideError> {
        let applied = Applied::from(method);
        if !compatible(self.family, applied) {
            return Err(incompatible(scioxide_core::argument!("method")));
        }
        if let QmcData::Radical {
            bases,
            permutations,
            ..
        } = &mut self.data
        {
            *permutations = match method {
                QmcScramble::DigitPermutation(seed) => halton::seeded_permutations(bases, seed)?,
                QmcScramble::DigitalShift(_) => Vec::new(),
            };
        }
        self.randomization.clear();
        self.scramble = applied;
        self.prepare_digital()?;
        Ok(self)
    }

    /// Returns the plan with the randomization `method` drawn from `rng`, replacing any earlier
    /// one.
    ///
    /// [`QmcRandomization::LinearMatrixShift`] draws 65 uniform 64-bit words per coordinate (64
    /// matrix rows, then the shift). [`QmcRandomization::RandomShift`] draws one word per
    /// coordinate. A uniform word may take several native outputs of `rng`. Independent draws
    /// give independent replicates of the randomized sequence.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the family does not accept `method` (see the table on [`QmcPlan`]).
    ///   This is checked before any draw.
    /// - Errors of `rng`, and `AllocationFailure` if the randomization data cannot be allocated.
    ///
    /// On error the plan is consumed and `rng` stays advanced by the words it consumed.
    ///
    /// # References
    ///
    /// - J. Matoušek, "On the L2-discrepancy for anchored boxes", *J. Complexity* 14(4), 1998,
    ///   527–556.
    /// - R. Cranley and T. N. L. Patterson, "Randomization of number theoretic methods for
    ///   multiple integration", *SIAM J. Numer. Anal.* 13(6), 1976, 904–914.
    pub fn with_randomization<R: crate::BitGenerator>(
        mut self,
        method: QmcRandomization,
        rng: &mut R,
    ) -> Result<Self, SciOxideError> {
        let applied = Applied::from(method);
        if !compatible(self.family, applied) {
            return Err(incompatible(scioxide_core::argument!("method")));
        }
        let width = scramble::randomization_width(applied);
        let len = scioxide_core::backend::checked_mul(
            self.dim(),
            width,
            scioxide_core::quantity!("randomization"),
        )?;
        let mut data = scioxide_core::backend::try_filled(
            len,
            0_u64,
            scioxide_core::quantity!("randomization"),
        )?;
        self.reserve_digital(applied)?;
        for coordinate in data.chunks_exact_mut(width) {
            for (row, word) in coordinate.iter_mut().enumerate() {
                let draw = rng.next_uniform_u64()?;
                *word = scramble::canonical_randomization_word(applied, row, draw);
            }
        }
        if let QmcData::Radical { permutations, .. } = &mut self.data {
            permutations.clear();
        }
        self.randomization = data;
        self.scramble = applied;
        self.fill_digital();
        Ok(self)
    }
    fn reserve_digital(&mut self, method: Applied) -> Result<(), SciOxideError> {
        self.digital = DigitalPreparation {
            columns: scioxide_core::backend::try_filled(
                if method == Applied::LinearMatrixShift {
                    self.dim()
                } else {
                    0
                },
                [0; 64],
                scioxide_core::quantity!("qmc_scrambled_columns"),
            )?,
            shifts: scioxide_core::backend::try_filled(
                if matches!(
                    method,
                    Applied::DigitalShift(_) | Applied::LinearMatrixShift
                ) {
                    self.dim()
                } else {
                    0
                },
                0,
                scioxide_core::quantity!("qmc_digital_shifts"),
            )?,
        };
        Ok(())
    }

    fn fill_digital(&mut self) {
        for (dimension, shift) in self.digital.shifts.iter_mut().enumerate() {
            *shift = scramble::transform(0, dimension, self.scramble, &self.randomization);
        }
        let directions = match &self.data {
            QmcData::Sobol { directions, .. } | QmcData::Niederreiter { directions } => directions,
            QmcData::Radical { .. } => return,
        };
        for (dimension, (target, source)) in
            self.digital.columns.iter_mut().zip(directions).enumerate()
        {
            for (target, &word) in target.iter_mut().zip(source) {
                // Remove the affine shift: only the linear map acts on recurrence columns.
                *target = scramble::transform(word, dimension, self.scramble, &self.randomization)
                    ^ self.digital.shifts[dimension];
            }
        }
    }

    pub(super) fn prepare_digital(&mut self) -> Result<(), SciOxideError> {
        self.reserve_digital(self.scramble)?;
        self.fill_digital();
        Ok(())
    }

    fn execution_directions(&self) -> Option<&[[u64; 64]]> {
        if self.digital.columns.is_empty() {
            self.direction_words()
        } else {
            Some(&self.digital.columns)
        }
    }

    /// Returns the randomization words in coordinate order: 65 per coordinate for a linear
    /// matrix scramble (64 matrix rows, then the shift), one per coordinate for a random shift,
    /// and none otherwise.
    pub fn randomization_words(&self) -> &[u64] {
        &self.randomization
    }

    /// Returns the Halton plan with your own digit permutations, replacing any earlier
    /// randomization.
    ///
    /// `permutations[j]` is a permutation of `0, …, bⱼ − 1` with `permutations[j][0] = 0`, applied
    /// to every digit in base `bⱼ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the family is not Halton (see the table on [`QmcPlan`]), if there is
    ///   not one permutation per base, or if a permutation has the wrong length, is not a
    ///   bijection, or does not map 0 to 0.
    /// - `AllocationFailure` if validation storage cannot be allocated.
    pub fn with_permutations(mut self, permutations: Vec<Vec<u32>>) -> Result<Self, SciOxideError> {
        if !compatible(self.family, Applied::CustomPermutation) {
            return Err(incompatible(scioxide_core::argument!("permutations")));
        }
        if let QmcData::Radical {
            bases,
            permutations: target,
            ..
        } = &mut self.data
        {
            halton::validate_permutations(bases, &permutations)?;
            *target = permutations;
        }
        self.randomization.clear();
        self.scramble = Applied::CustomPermutation;
        Ok(self)
    }

    /// Returns the sequence family.
    pub const fn family(&self) -> QmcFamily {
        self.family
    }

    /// Returns the number of coordinates `d` of each point.
    pub fn dim(&self) -> usize {
        match &self.data {
            QmcData::Sobol { directions, .. } | QmcData::Niederreiter { directions } => {
                directions.len()
            }
            QmcData::Radical { bases, .. } => bases.len(),
        }
    }

    /// Returns the 64 generator-matrix columns of each coordinate, as 64-bit words, for Sobol
    /// and Niederreiter base 2, or `None` for the Halton families.
    ///
    /// These are the unrandomized columns.
    pub fn direction_words(&self) -> Option<&[[u64; 64]]> {
        match &self.data {
            QmcData::Sobol { directions, .. } | QmcData::Niederreiter { directions } => {
                Some(directions)
            }
            _ => None,
        }
    }

    /// Returns the prime base of each coordinate for the Halton families, or `None` otherwise.
    pub fn bases(&self) -> Option<&[u32]> {
        match &self.data {
            QmcData::Radical { bases, .. } => Some(bases),
            _ => None,
        }
    }

    /// Returns the digit permutations of a Halton family, or `None` for digital families.
    ///
    /// The slice is empty if no permutation is applied.
    pub fn permutations(&self) -> Option<&[Vec<u32>]> {
        match &self.data {
            QmcData::Radical { permutations, .. } => Some(permutations),
            _ => None,
        }
    }

    /// Creates a cursor at index 0 with output type `S`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the cursor's O(d) working storage cannot be allocated.
    pub fn sequence<S: RealScalar>(&self) -> Result<QmcSequence<'_, S>, SciOxideError> {
        let mut cursor = QmcSequence {
            plan: self,
            position: QmcPosition {
                index: 0,
                exhausted: false,
            },
            words: scioxide_core::backend::try_filled(
                if self.direction_words().is_some() {
                    self.dim()
                } else {
                    0
                },
                0,
                scioxide_core::quantity!("qmc_cursor_words"),
            )?,
            staging: scioxide_core::backend::try_filled(
                if self.direction_words().is_none() {
                    self.dim()
                } else {
                    0
                },
                S::ZERO,
                scioxide_core::quantity!("qmc_point_staging"),
            )?,
        };
        if !self.digital.shifts.is_empty() {
            cursor.words.copy_from_slice(&self.digital.shifts);
        }
        Ok(cursor)
    }

    /// Writes point `index` of the sequence into `out`, without a cursor.
    ///
    /// The point is bit-identical to the one a [`QmcSequence`] of this plan returns at `index`,
    /// so index ranges can be generated in any order and on any thread. Sobol and Niederreiter
    /// points cost O(d · popcount(n)) for the Gray code `n` of `index`. Halton points cost
    /// O(d log index).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out.len()` differs from [`Self::dim`]. `out` is unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{QmcPlan, QuasiRng};
    /// let plan = QmcPlan::sobol(2)?;
    /// let mut sequence = plan.sequence::<f64>()?;
    /// sequence.skip_to(5)?;
    /// let mut point = [0.0; 2];
    /// plan.point_into(5, &mut point)?;
    /// assert_eq!(point.to_vec(), sequence.next_point()?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn point_into<S: RealScalar>(
        &self,
        index: u64,
        out: &mut [S],
    ) -> Result<(), SciOxideError> {
        if out.len() != self.dim() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dim()),
                scioxide_core::Dimensions::one(out.len()),
            ));
        }
        if let Some(directions) = self.execution_directions() {
            let gray = index ^ (index >> 1);
            for (dimension, (value, columns)) in out.iter_mut().zip(directions).enumerate() {
                let mut word = self.digital.shifts.get(dimension).copied().unwrap_or(0);
                let mut remaining = gray;
                while remaining != 0 {
                    word ^= columns[remaining.trailing_zeros() as usize];
                    remaining &= remaining - 1;
                }
                *value = S::from_u64_half_open01(word);
            }
        } else if let QmcData::Radical {
            dividers,
            permutations,
            ..
        } = &self.data
        {
            for (dimension, (value, base)) in out.iter_mut().zip(dividers).enumerate() {
                let point = halton::radical_inverse(
                    index,
                    base,
                    permutations.get(dimension).map(Vec::as_slice),
                    self.family == QmcFamily::ReverseHalton,
                )?;
                *value = if self.scramble == Applied::RandomShift {
                    halton::rotate(
                        point,
                        Some(S::from_u64_half_open01(self.randomization[dimension])),
                    )?
                } else {
                    point
                };
            }
        }
        Ok(())
    }
}

/// Position of a [`QmcSequence`] cursor: the index of the next point, or the end.
///
/// It does not depend on the output type. See [`QmcCheckpoint`](crate::QmcCheckpoint) for an
/// example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmcPosition {
    pub(super) index: u64,
    pub(super) exhausted: bool,
}
impl QmcPosition {
    /// Returns the index of the next point, or `None` after the last point (index `2⁶⁴ − 1`).
    pub const fn next_index(self) -> Option<u64> {
        if self.exhausted {
            None
        } else {
            Some(self.index)
        }
    }
}

/// Cursor over the points of a [`QmcPlan`], with output type `S`.
///
/// The cursor implements [`QuasiRng`](crate::QuasiRng): [`next_point_into`] writes one point of
/// `d` coordinates, [`points_into`] writes consecutive points row by row, and [`skip_to`]
/// moves to any index. Sobol and Niederreiter points cost O(d) each and seeking costs
/// O(d · popcount(g)) for the Gray code `g` of the index. Halton points cost O(d log n) each and seeking costs O(1). Each cursor
/// keeps its own position, so several cursors over one plan can generate disjoint index ranges
/// on different threads with results independent of scheduling.
///
/// The sequence has `2⁶⁴` points. Drawing past the last point is `InvalidInput`
/// (`OutOfRange`), and a point of the wrong length is `DimensionMismatch`. Batches follow the
/// [`SamplingPolicy`](crate::SamplingPolicy) (see [batch sampling](crate#batch-sampling)).
///
/// [`next_point_into`]: crate::QuasiRng::next_point_into
/// [`points_into`]: crate::QuasiRng::points_into
/// [`skip_to`]: crate::QuasiRng::skip_to
///
/// # Examples
///
/// Reserve one workspace for the batch shape, then fill rows of an output slice:
///
/// ```
/// use scioxide_random::{OutputPolicy, QmcPlan, QuasiRng, SamplingPolicy, SamplingRequest};
/// let plan = QmcPlan::sobol(2)?;
/// let mut sequence = plan.sequence::<f64>()?;
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let mut workspace = policy.workspace(SamplingRequest::rows(4, sequence.dim()))?;
/// let mut points = [0.0; 8];
/// sequence.points_into(policy, &mut points, &mut workspace)?;
/// // Unscrambled Sobol points are exact dyadic rationals.
/// assert_eq!(points, [0.0, 0.0, 0.5, 0.5, 0.75, 0.25, 0.25, 0.75]);
/// sequence.skip_to(1)?;
/// assert_eq!(sequence.points(1)?, [0.5, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct QmcSequence<'plan, S> {
    pub(super) plan: &'plan QmcPlan,
    pub(super) position: QmcPosition,
    words: Vec<u64>,
    staging: Vec<S>,
}
impl<S: RealScalar> QmcSequence<'_, S> {
    /// Returns the plan this cursor reads.
    pub const fn plan(&self) -> &QmcPlan {
        self.plan
    }
    /// Returns the cursor position: the index of the next point, or the end.
    pub const fn position(&self) -> QmcPosition {
        self.position
    }

    fn restore_position(&mut self, position: QmcPosition) {
        // Digital sequences are enumerated in Gray-code order, so the state for
        // index i is the XOR of the direction columns selected by the set bits
        // of gray(i) = i ^ (i >> 1). Seeking therefore costs O(popcount) per
        // dimension, with no need to replay the sequence.
        if let Some(directions) = self.plan.execution_directions() {
            let gray = position.index ^ (position.index >> 1);
            self.words.iter_mut().zip(directions).enumerate().for_each(
                |(dimension, (word, columns))| {
                    let mut remaining = gray;
                    *word = self
                        .plan
                        .digital
                        .shifts
                        .get(dimension)
                        .copied()
                        .unwrap_or(0);
                    while remaining != 0 {
                        *word ^= columns[remaining.trailing_zeros() as usize];
                        remaining &= remaining - 1;
                    }
                },
            );
        }
        self.position = position;
    }
}
impl<S: RealScalar> QuasiRng<S> for QmcSequence<'_, S> {
    fn dim(&self) -> usize {
        self.plan.dim()
    }
    fn next_point(&mut self) -> Result<Vec<S>, SciOxideError> {
        let mut output = scioxide_core::backend::try_filled(
            self.dim(),
            S::ZERO,
            scioxide_core::quantity!("qmc_output"),
        )?;
        self.next_point_into(&mut output)?;
        Ok(output)
    }
    fn next_point_into(&mut self, output: &mut [S]) -> Result<(), SciOxideError> {
        if output.len() != self.dim() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dim()),
                scioxide_core::Dimensions::one(output.len()),
            ));
        }
        if self.position.exhausted {
            return Err(exhausted(scioxide_core::argument!("self")));
        }
        let next_index = self.position.index.checked_add(1);
        match &self.plan.data {
            QmcData::Sobol { .. } | QmcData::Niederreiter { .. } => {
                if let Some(directions) = self.plan.execution_directions() {
                    digital_point(output, &mut self.words, directions, next_index);
                }
            }
            QmcData::Radical {
                dividers,
                permutations,
                ..
            } => {
                self.staging
                    .iter_mut()
                    .zip(dividers)
                    .enumerate()
                    .try_for_each(|(dimension, (value, base))| {
                        *value = halton::radical_inverse(
                            self.position.index,
                            base,
                            permutations.get(dimension).map(Vec::as_slice),
                            self.plan.family == QmcFamily::ReverseHalton,
                        )?;
                        if self.plan.scramble == Applied::RandomShift {
                            *value = halton::rotate(
                                *value,
                                Some(S::from_u64_half_open01(self.plan.randomization[dimension])),
                            )?;
                        }
                        Ok::<_, SciOxideError>(())
                    })?;
                output.copy_from_slice(&self.staging);
            }
        }
        if let Some(index) = next_index {
            self.position.index = index;
        } else {
            self.position.exhausted = true;
        }
        Ok(())
    }
    fn points_into(
        &mut self,
        policy: crate::SamplingPolicy<impl crate::StatePolicy<Self>>,
        output: &mut [S],
        workspace: &mut crate::SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let dim = self.dim();
        if !output.len().is_multiple_of(dim) {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(dim),
                scioxide_core::Dimensions::one(output.len() % dim),
            ));
        }
        if output.is_empty() {
            return Ok(());
        }
        let rows = output.len() / dim;
        let requirements =
            policy.workspace_requirements(crate::SamplingRequest::rows(rows, dim))?;
        workspace.stage(&requirements, S::ZERO)?;
        policy.run(self, |cursor| {
            let atomic = policy.output_policy() == crate::OutputPolicy::Atomic;
            let target = if atomic {
                workspace.as_mut_slice()
            } else {
                &mut *output
            };
            // After shape/range admission, built-in generation cannot fail:
            // digital decoding is sealed, and radical fractions and rotations
            // are finite in [0,1]. This permits direct incremental publication.
            // Requests crossing exhaustion keep the checked pointwise path so
            // completed-prefix output and state rollback remain independent.
            let last = u64::try_from(rows - 1)
                .ok()
                .and_then(|offset| cursor.position.index.checked_add(offset));
            if let Some(last) = last
                && !cursor.position.exhausted
            {
                if let Some(directions) = cursor.plan.execution_directions() {
                    let mut index = cursor.position.index;
                    for row in target.chunks_exact_mut(dim) {
                        let next = index.checked_add(1);
                        digital_point(row, &mut cursor.words, directions, next);
                        index = next.unwrap_or(u64::MAX);
                    }
                } else if let QmcData::Radical {
                    dividers,
                    permutations,
                    ..
                } = &cursor.plan.data
                {
                    for (dimension, base) in dividers.iter().enumerate() {
                        let shift = (cursor.plan.scramble == Applied::RandomShift)
                            .then(|| S::from_u64_half_open01(cursor.plan.randomization[dimension]));
                        halton::fill_coordinate(
                            &mut target[dimension..],
                            dim,
                            cursor.position.index..=last,
                            base,
                            permutations.get(dimension).map(Vec::as_slice),
                            cursor.plan.family == QmcFamily::ReverseHalton,
                            shift,
                        )?;
                    }
                }
                cursor.position = QmcPosition {
                    index: last.saturating_add(1),
                    exhausted: last == u64::MAX,
                };
            } else {
                for row in target.chunks_exact_mut(dim) {
                    cursor.next_point_into(row)?;
                }
            }
            if atomic {
                output.copy_from_slice(workspace.as_slice());
            }
            Ok(())
        })
    }
    fn skip(&mut self, count: u64) -> Result<(), SciOxideError> {
        if count == 0 {
            return Ok(());
        }
        if self.position.exhausted {
            return Err(exhausted(scioxide_core::argument!("self")));
        }
        // Skipping exactly to one past u64::MAX is representable as the
        // exhausted end state; anything further is an error.
        if let Some(index) = self.position.index.checked_add(count) {
            self.skip_to(index)
        } else if count - 1 == u64::MAX - self.position.index {
            self.position = QmcPosition {
                index: u64::MAX,
                exhausted: true,
            };
            Ok(())
        } else {
            Err(exhausted(scioxide_core::argument!("count")))
        }
    }
    fn skip_to(&mut self, index: u64) -> Result<(), SciOxideError> {
        self.restore_position(QmcPosition {
            index,
            exhausted: false,
        });
        Ok(())
    }
}
// Shape and position are admitted by the caller. Emit before advancing so index
// zero is the origin (or its shift); the final point has no recurrence update.
#[inline]
fn digital_point<S: RealScalar>(
    output: &mut [S],
    words: &mut [u64],
    directions: &[[u64; 64]],
    next: Option<u64>,
) {
    if let Some(next) = next {
        let bit = next.trailing_zeros() as usize;
        for ((value, word), columns) in output.iter_mut().zip(words).zip(directions) {
            *value = S::from_u64_half_open01(*word);
            *word ^= columns[bit];
        }
    } else {
        for (value, &word) in output.iter_mut().zip(words.iter()) {
            *value = S::from_u64_half_open01(word);
        }
    }
}

impl<S: RealScalar> SnapshotGenerator for QmcSequence<'_, S> {
    type Snapshot = QmcPosition;
    fn snapshot(&self) -> Result<Self::Snapshot, SciOxideError> {
        Ok(self.position)
    }
    fn restore_snapshot(&mut self, snapshot: Self::Snapshot) {
        self.restore_position(snapshot);
    }
}
pub(super) fn invalid(field: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(field, InvalidInputReason::Inconsistent, detail)
}
fn nonzero(dimensions: usize) -> Result<(), SciOxideError> {
    if dimensions == 0 {
        Err(invalid(
            scioxide_core::argument!("dimensions"),
            "QMC requires at least one coordinate",
        ))
    } else {
        Ok(())
    }
}
/// Exhausted-sequence error against `argument`.
fn exhausted(argument: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        InvalidInputReason::OutOfRange,
        "QMC sequence index exhausted",
    )
}
