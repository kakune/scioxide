//! Pivot blocks and inertia of `LDLᴴ` factorizations.

use core::{cmp::Ordering, ops::Range};
use scioxide_core::SciOxideError;

/// Identifies a 1×1 or 2×2 diagonal block of `D` in an `LDLᴴ` factorization.
///
/// The block starts at factor row `start` and covers one or two consecutive
/// rows. Check a block with [`checked_range`](Self::checked_range) before using
/// it to index factor storage.
///
/// ```
/// use scioxide_sparse::LdlPivotBlock;
/// let block = LdlPivotBlock::Two { start: 1 };
/// assert_eq!(block.checked_range(3)?, 1..3);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdlPivotBlock {
    /// One diagonal entry.
    One {
        /// Factor row.
        start: usize,
    },
    /// A Hermitian 2×2 block.
    Two {
        /// First of two consecutive factor rows.
        start: usize,
    },
}

impl LdlPivotBlock {
    /// Returns the first factor row of the block.
    pub const fn start(self) -> usize {
        match self {
            Self::One { start } | Self::Two { start } => start,
        }
    }
    /// Returns the number of rows of the block, one or two.
    pub const fn size(self) -> usize {
        match self {
            Self::One { .. } => 1,
            Self::Two { .. } => 2,
        }
    }
    /// Returns the rows `start..start + size` of the block, after checking them
    /// against the order `order` of the factor.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a row of the block is not less than `order`.
    /// - `SizeOverflow` if `start + size` overflows `usize`.
    pub fn checked_range(self, order: usize) -> Result<Range<usize>, SciOxideError> {
        let end = self.start().checked_add(self.size()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("ldl_pivot_end"))
        })?;
        if end > order {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("ldl_factor_rows"),
                end - 1,
                order,
            ));
        }
        Ok(self.start()..end)
    }
}

/// Holds the inertia of a Hermitian matrix: the numbers of positive, negative
/// and zero eigenvalues.
///
/// By Sylvester's law of inertia, congruent matrices have the same inertia, so
/// the inertia of `D` in `A = L D Lᴴ` is that of `A`. The total count always
/// fits in `usize`. The default has all counts zero.
///
/// ```
/// use scioxide_sparse::LdlInertia;
/// let inertia = LdlInertia::new(2, 1, 0)?;
/// assert_eq!(inertia.total(), 3);
/// assert_eq!(inertia.checked_add(LdlInertia::ZERO)?, inertia);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LdlInertia {
    positive: usize,
    negative: usize,
    zero: usize,
}

impl LdlInertia {
    /// The inertia with all counts zero, the identity of
    /// [`checked_add`](Self::checked_add).
    pub const ZERO: Self = Self {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    /// Creates an inertia from its three counts.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the total exceeds `usize::MAX`.
    pub fn new(positive: usize, negative: usize, zero: usize) -> Result<Self, SciOxideError> {
        positive
            .checked_add(negative)
            .and_then(|count| count.checked_add(zero))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ldl_inertia_total"))
            })?;
        Ok(Self {
            positive,
            negative,
            zero,
        })
    }

    /// Returns the inertia of one eigenvalue with the given sign: `Less` for
    /// negative, `Equal` for zero and `Greater` for positive.
    pub const fn from_sign(sign: Ordering) -> Self {
        match sign {
            Ordering::Less => Self {
                positive: 0,
                negative: 1,
                zero: 0,
            },
            Ordering::Equal => Self {
                positive: 0,
                negative: 0,
                zero: 1,
            },
            Ordering::Greater => Self {
                positive: 1,
                negative: 0,
                zero: 0,
            },
        }
    }
    /// Returns the number of positive eigenvalues.
    pub const fn positive(self) -> usize {
        self.positive
    }
    /// Returns the number of negative eigenvalues.
    pub const fn negative(self) -> usize {
        self.negative
    }
    /// Returns the number of zero eigenvalues.
    pub const fn zero(self) -> usize {
        self.zero
    }
    /// Returns the total count, the order of the matrix.
    pub const fn total(self) -> usize {
        self.positive + self.negative + self.zero
    }
    /// Returns the sum of the counts, which is the inertia of a block diagonal
    /// matrix with blocks of inertia `self` and `other`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count or the total overflows `usize`.
    pub fn checked_add(self, other: Self) -> Result<Self, SciOxideError> {
        let add = |left: usize, right: usize| {
            left.checked_add(right).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ldl_inertia_count"))
            })
        };
        Self::new(
            add(self.positive, other.positive)?,
            add(self.negative, other.negative)?,
            add(self.zero, other.zero)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivot_ranges_check_each_row_and_overflow() -> Result<(), SciOxideError> {
        assert_eq!(LdlPivotBlock::One { start: 0 }.checked_range(1)?, 0..1);
        assert_eq!(LdlPivotBlock::Two { start: 1 }.checked_range(3)?, 1..3);
        for block in [
            LdlPivotBlock::One { start: 0 },
            LdlPivotBlock::Two { start: 0 },
        ] {
            assert!(block.checked_range(0).is_err());
        }
        Ok(())
    }
    #[test]
    fn malformed_locations_are_structured_errors() {
        for (block, order) in [
            (LdlPivotBlock::One { start: 1 }, 1),
            (LdlPivotBlock::Two { start: 0 }, 1),
        ] {
            assert!(matches!(
                block
                    .checked_range(order)
                    .as_ref()
                    .map_err(SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::OutOfBounds { .. })
            ));
        }
        assert!(matches!(
            LdlPivotBlock::Two { start: usize::MAX }
                .checked_range(usize::MAX)
                .as_ref()
                .map_err(SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::SizeOverflow { .. })
        ));
    }
    #[test]
    fn inertia_identity_and_overflow_preserve_operands() -> Result<(), SciOxideError> {
        assert_eq!(LdlInertia::default(), LdlInertia::ZERO);
        assert!(LdlInertia::new(usize::MAX, 1, 0).is_err());
        let maximum = LdlInertia::new(usize::MAX, 0, 0)?;
        assert_eq!(maximum.checked_add(LdlInertia::ZERO)?, maximum);
        for sign in [Ordering::Less, Ordering::Equal, Ordering::Greater] {
            let direction = LdlInertia::from_sign(sign);
            let error = maximum.checked_add(direction);
            assert!(matches!(
                error.as_ref().map_err(SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::SizeOverflow { .. })
            ));
            assert_eq!(maximum.total(), usize::MAX);
            assert_eq!(direction.total(), 1);
        }
        let mixed = LdlInertia::from_sign(Ordering::Less)
            .checked_add(LdlInertia::from_sign(Ordering::Greater))?;
        assert_eq!(
            (
                mixed.positive(),
                mixed.negative(),
                mixed.zero(),
                mixed.total()
            ),
            (1, 1, 0, 2)
        );
        Ok(())
    }
}
