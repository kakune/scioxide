//! Window shape and boundary handling for rolling statistics.
use scioxide_core::SciOxideError;
/// Handling of window positions that fall before the first or after the last
/// observation of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RollingBoundary {
    /// Drops positions outside the batch, so windows near the ends are shorter.
    #[default]
    Truncate,
    /// Repeats the first or last observation, with its mask entry and weight.
    Repeat,
    /// Uses observations equal to zero. Only unweighted views accept it.
    ZeroPad,
}
/// A rolling window of `left + 1 + right` positions around each observation.
///
/// The window of observation `i` covers positions `i − left` through
/// `i + right`. [`RollingBoundary`] sets how positions outside the batch are
/// treated. Missing and non-finite observations inside the window are handled
/// by the [`StatisticsOptions`](crate::StatisticsOptions) policies after the
/// window is formed, so a skipped observation shortens that window.
#[derive(Clone, Copy, Debug)]
pub struct RollingWindow {
    /// Positions before the current observation.
    pub(crate) left: usize,
    /// Positions after the current observation.
    pub(crate) right: usize,
    /// Handling of positions outside the batch.
    pub(crate) boundary: RollingBoundary,
}
impl RollingWindow {
    /// Creates a window with `left` positions before and `right` positions after
    /// the current observation, and [`RollingBoundary::Truncate`] boundaries.
    pub const fn new(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            boundary: RollingBoundary::Truncate,
        }
    }
    /// Sets the number of positions before the current observation.
    pub const fn with_left(mut self, left: usize) -> Self {
        self.left = left;
        self
    }
    /// Sets the number of positions after the current observation.
    pub const fn with_right(mut self, right: usize) -> Self {
        self.right = right;
        self
    }
    /// Sets the boundary handling.
    pub const fn with_boundary(mut self, boundary: RollingBoundary) -> Self {
        self.boundary = boundary;
        self
    }
    /// Returns the number of positions before the current observation.
    pub const fn left(self) -> usize {
        self.left
    }
    /// Returns the number of positions after the current observation.
    pub const fn right(self) -> usize {
        self.right
    }
    /// Returns the boundary handling.
    pub const fn boundary(self) -> RollingBoundary {
        self.boundary
    }
    /// Returns the window width `left + right + 1`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the width overflows `usize`.
    pub fn width(self) -> Result<usize, SciOxideError> {
        self.left
            .checked_add(self.right)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("rolling_width")))
    }
    pub(crate) fn index(self, center: usize, offset: usize, n: usize) -> Option<usize> {
        let index = if offset < self.left {
            center.checked_sub(self.left - offset)
        } else {
            center.checked_add(offset - self.left)
        };
        if let Some(i) = index
            && i < n
        {
            return Some(i);
        }
        if self.boundary == RollingBoundary::Repeat && n > 0 {
            Some(if offset < self.left && center < self.left - offset {
                0
            } else {
                n - 1
            })
        } else {
            None
        }
    }
}
