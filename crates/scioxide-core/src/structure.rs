//! Triangular-matrix flags, sparsity patterns in CSR form, and checked graph colorings.
use crate::size::{checked_add, try_filled, try_with_capacity};
use crate::{InvalidInputReason, SciOxideError};
use alloc::vec::Vec;

/// Which triangle of a matrix is stored, before any transpose or adjoint is applied.
///
/// ```
/// use scioxide_core::{DiagonalKind, Triangle};
///
/// let stored = (Triangle::Lower, DiagonalKind::Unit);
/// assert_eq!(stored.0, Triangle::Lower);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum Triangle {
    /// Entries on or below the diagonal.
    Lower,
    /// Entries on or above the diagonal.
    Upper,
}

/// Whether a triangular matrix has a stored diagonal or a unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum DiagonalKind {
    /// The diagonal entries are stored; a solve requires them to be nonzero.
    Explicit,
    /// Every diagonal entry is one and is not read. Sparse triangular storage
    /// must omit the diagonal; dense and packed storage keep the diagonal slots
    /// but ignore their contents.
    Unit,
}

fn invalid(argument: crate::Argument, message: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Malformed, message)
}
fn copy(values: &[usize]) -> Result<Vec<usize>, SciOxideError> {
    let mut result = try_with_capacity(values.len(), crate::quantity!("structural_indices"))?;
    result.extend_from_slice(values);
    Ok(result)
}
/// Borrowed sparsity pattern of a matrix in compressed sparse row (CSR) form.
///
/// Row `i` holds the column indices `col_indices[row_ptr[i]..row_ptr[i + 1]]`,
/// sorted in strictly increasing order. Every stored position belongs to the
/// pattern, including positions whose value is an explicit zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralPatternView<'a> {
    rows: usize,
    cols: usize,
    pointers: &'a [usize],
    indices: &'a [usize],
}
impl<'a> StructuralPatternView<'a> {
    /// Creates a pattern view of a `rows × cols` matrix from CSR row pointers and column indices.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Malformed`) if `pointers` does not have `rows + 1`
    ///   entries, does not start at zero, decreases, or does not end at
    ///   `indices.len()`; if a column index is not less than `cols`; or if the
    ///   indices of a row are not strictly increasing.
    /// - `SizeOverflow` if `rows + 1` overflows.
    pub fn new(
        rows: usize,
        cols: usize,
        pointers: &'a [usize],
        indices: &'a [usize],
    ) -> Result<Self, SciOxideError> {
        if pointers.len() != checked_add(rows, 1, crate::quantity!("structural_pointers"))?
            || pointers.first() != Some(&0)
            || pointers.last() != Some(&indices.len())
            || pointers.windows(2).any(|p| p[0] > p[1])
            || pointers.iter().any(|&p| p > indices.len())
            || indices.iter().any(|&i| i >= cols)
        {
            return Err(invalid(
                crate::argument!("pointers"),
                "invalid CSR structure",
            ));
        }
        if pointers
            .windows(2)
            .any(|p| indices[p[0]..p[1]].windows(2).any(|w| w[0] >= w[1]))
        {
            return Err(invalid(
                crate::argument!("indices"),
                "row indices must be strictly increasing",
            ));
        }
        Ok(Self {
            rows,
            cols,
            pointers,
            indices,
        })
    }
    /// Returns the number of rows.
    pub fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub fn cols(self) -> usize {
        self.cols
    }
    /// Returns the `rows + 1` row pointers.
    pub fn row_ptr(self) -> &'a [usize] {
        self.pointers
    }
    /// Returns the column indices, row by row.
    pub fn col_indices(self) -> &'a [usize] {
        self.indices
    }
    /// Returns an owned copy of the pattern.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the index storage cannot be reserved.
    pub fn to_owned(self) -> Result<StructuralPattern, SciOxideError> {
        Ok(StructuralPattern {
            rows: self.rows,
            cols: self.cols,
            pointers: copy(self.pointers)?,
            indices: copy(self.indices)?,
        })
    }
}
/// Owned sparsity pattern in CSR form; see [`StructuralPatternView`] for the layout.
///
/// Two patterns are equal when their shapes and all indices are equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralPattern {
    rows: usize,
    cols: usize,
    pointers: Vec<usize>,
    indices: Vec<usize>,
}
impl StructuralPattern {
    /// Creates a pattern from CSR row pointers and column indices, taking ownership of them.
    ///
    /// # Errors
    ///
    /// The same as [`StructuralPatternView::new`].
    pub fn new(
        rows: usize,
        cols: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
    ) -> Result<Self, SciOxideError> {
        StructuralPatternView::new(rows, cols, &pointers, &indices)?;
        Ok(Self {
            rows,
            cols,
            pointers,
            indices,
        })
    }
    /// Reserves storage for a pattern with `rows` rows and `entries` stored entries.
    ///
    /// The pattern itself does not change, also on error.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows + 1` overflows.
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::StructuralPattern;
    ///
    /// let mut p = StructuralPattern::new(1, 1, vec![0, 1], vec![0])?;
    /// p.try_reserve(4, 8)?;
    /// assert_eq!(p.view().rows(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(&mut self, rows: usize, entries: usize) -> Result<(), SciOxideError> {
        let q = crate::quantity!("structural_pattern");
        let pointers = checked_add(rows, 1, q)?.saturating_sub(self.pointers.len());
        let indices = entries.saturating_sub(self.indices.len());
        crate::backend::try_reserve(&mut self.pointers, pointers, q)?;
        crate::backend::try_reserve(&mut self.indices, indices, q)
    }
    /// Replaces this pattern with a copy of `pattern`, reusing the existing storage.
    ///
    /// Storage is never shrunk. On error the pattern is unchanged, although its
    /// capacity may have grown.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot grow.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::StructuralPattern;
    ///
    /// let mut p = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1])?;
    /// let q = StructuralPattern::new(1, 1, vec![0, 1], vec![0])?;
    /// p.try_assign(q.view())?;
    /// assert_eq!(p.view(), q.view());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_assign(&mut self, pattern: StructuralPatternView<'_>) -> Result<(), SciOxideError> {
        self.try_reserve(pattern.rows(), pattern.col_indices().len())?;
        self.pointers.clear();
        self.pointers.extend_from_slice(pattern.row_ptr());
        self.indices.clear();
        self.indices.extend_from_slice(pattern.col_indices());
        self.rows = pattern.rows();
        self.cols = pattern.cols();
        Ok(())
    }
    /// Returns a borrowed view of the pattern.
    pub fn view(&self) -> StructuralPatternView<'_> {
        StructuralPatternView {
            rows: self.rows,
            cols: self.cols,
            pointers: &self.pointers,
            indices: &self.indices,
        }
    }
}
/// Which pairs of columns must receive different colors in a [`ColoringSchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColoringKind {
    /// Distance-1 coloring of the graph of a square pattern: if `(i, j)` or
    /// `(j, i)` is stored with `i ≠ j`, vertices `i` and `j` differ in color.
    DistanceOne,
    /// Column coloring for compressed Jacobian estimation (Curtis–Powell–Reid):
    /// two columns with a stored entry in the same row differ in color.
    DerivativeColumns,
}
/// A checked coloring of the columns of a sparsity pattern, grouped by color.
///
/// Columns of one color can be evaluated together, for example in one
/// finite-difference or forward-mode derivative sweep. The schedule keeps a copy
/// of its pattern so that it can check that it is used with the same pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColoringSchedule {
    pattern: StructuralPattern,
    kind: ColoringKind,
    colors: Vec<usize>,
    offsets: Vec<usize>,
    members: Vec<usize>,
}
impl ColoringSchedule {
    /// Creates a schedule from one color label per column.
    ///
    /// Labels must be `0, 1, …, k − 1` with every label used. A pattern with no
    /// columns has zero colors. Each group lists its columns in increasing order.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Malformed`) if `colors.len()` differs from the number
    ///   of columns, a label is not less than the number of columns, a label in
    ///   `0..k` is unused, two conflicting columns share a color, or `kind` is
    ///   [`ColoringKind::DistanceOne`] and the pattern is not square.
    /// - `AllocationFailure` if the group storage cannot be reserved.
    pub fn new(
        pattern: StructuralPattern,
        kind: ColoringKind,
        colors: Vec<usize>,
    ) -> Result<Self, SciOxideError> {
        let p = pattern.view();
        if colors.len() != p.cols
            || (kind == ColoringKind::DistanceOne && p.rows != p.cols)
            || colors.iter().any(|&c| c >= p.cols)
        {
            return Err(invalid(
                crate::argument!("colors"),
                "invalid coloring shape or label",
            ));
        }
        let count = colors.iter().copied().max().map_or(0, |c| c + 1);
        // `marks[c]` records the last row in which color c appeared. Rows are
        // scanned in order, so a repeat within one row means two columns of the
        // same color share that row, which would make their finite-difference
        // contributions indistinguishable. This check is O(nnz) with O(colors)
        // scratch.
        let mut marks = try_filled(count, usize::MAX, crate::quantity!("color_validation"))?;
        for (row, bounds) in p.pointers.windows(2).enumerate() {
            for &col in &p.indices[bounds[0]..bounds[1]] {
                match kind {
                    ColoringKind::DistanceOne if row != col && colors[row] == colors[col] => {
                        return Err(invalid(
                            crate::argument!("colors"),
                            "adjacent vertices share a color",
                        ));
                    }
                    ColoringKind::DerivativeColumns => {
                        if marks[colors[col]] == row {
                            return Err(invalid(
                                crate::argument!("colors"),
                                "columns sharing a row share a color",
                            ));
                        }
                        marks[colors[col]] = row;
                    }
                    _ => {}
                }
            }
        }
        let mut offsets = try_filled(
            checked_add(count, 1, crate::quantity!("color_offsets"))?,
            0usize,
            crate::quantity!("color_offsets"),
        )?;
        // Counting sort by color: offsets become CSR-style group starts, and
        // members lists vertices in increasing order within each group, so groups
        // are deterministic.
        for &color in &colors {
            offsets[color + 1] += 1;
        }
        if offsets[1..].contains(&0) {
            return Err(invalid(
                crate::argument!("colors"),
                "color labels must be contiguous",
            ));
        }
        for i in 0..count {
            offsets[i + 1] += offsets[i];
        }
        let mut cursor = copy(&offsets[..count])?;
        let mut members = try_filled(p.cols, 0, crate::quantity!("color_members"))?;
        for (vertex, &color) in colors.iter().enumerate() {
            members[cursor[color]] = vertex;
            cursor[color] += 1;
        }
        Ok(Self {
            pattern,
            kind,
            colors,
            offsets,
            members,
        })
    }
    /// Returns the pattern the coloring was checked against.
    pub fn pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }
    /// Returns the kind of coloring.
    pub fn kind(&self) -> ColoringKind {
        self.kind
    }
    /// Returns the color of each column.
    pub fn colors(&self) -> &[usize] {
        &self.colors
    }
    /// Returns the number of colors.
    pub fn color_count(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Returns the columns of each color, in color order; each group is sorted.
    pub fn groups(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.offsets.windows(2).map(|p| &self.members[p[0]..p[1]])
    }
    /// Checks that `pattern` equals the pattern of this schedule.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Malformed`) if the shape or any index differs.
    pub fn validate_pattern(
        &self,
        pattern: StructuralPatternView<'_>,
    ) -> Result<(), SciOxideError> {
        if self.pattern() == pattern {
            Ok(())
        } else {
            Err(invalid(
                crate::argument!("pattern"),
                "coloring pattern identity mismatch",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn distance_conflicts_and_structural_boundaries() {
        for (rows, cols, pointers, indices) in [
            (1, 1, &[1, 1][..], &[][..]),
            (2, 1, &[0, 1, 0][..], &[][..]),
            (1, 1, &[0, 1][..], &[1][..]),
            (1, 2, &[0, 2][..], &[1, 0][..]),
        ] {
            assert!(matches!(
                (StructuralPatternView::new(rows, cols, pointers, indices))
                    .as_ref()
                    .unwrap_err()
                    .view(),
                crate::SciOxideErrorView::InvalidInput { .. }
            ));
        }
        assert!(matches!(
            (StructuralPatternView::new(usize::MAX, 0, &[], &[]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        let edge = StructuralPattern::new(2, 2, vec![0, 1, 1], vec![1]).unwrap();
        assert!(matches!(
            (ColoringSchedule::new(edge, ColoringKind::DistanceOne, vec![0, 0]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        let diagonal = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        let schedule =
            ColoringSchedule::new(diagonal, ColoringKind::DistanceOne, vec![0, 0]).unwrap();
        assert_eq!(schedule.groups().next(), Some(&[0, 1][..]));
        let hole = StructuralPattern::new(0, 3, vec![0], vec![]).unwrap();
        assert!(
            ColoringSchedule::new(hole, ColoringKind::DerivativeColumns, vec![0, 2, 2]).is_err()
        );
    }
    #[test]
    fn malformed_and_conflicting_schedules() {
        assert!(StructuralPatternView::new(1, 2, &[0, 2], &[1, 1]).is_err());
        assert!(StructuralPatternView::new(1, 2, &[0, 3], &[0]).is_err());
        let pattern = || StructuralPattern::new(1, 2, vec![0, 2], vec![0, 1]).unwrap();
        assert!(
            ColoringSchedule::new(pattern(), ColoringKind::DerivativeColumns, vec![0, 0]).is_err()
        );
        assert!(
            ColoringSchedule::new(pattern(), ColoringKind::DerivativeColumns, vec![1, 1]).is_err()
        );
        let schedule =
            ColoringSchedule::new(pattern(), ColoringKind::DerivativeColumns, vec![0, 1]).unwrap();
        assert_eq!(
            schedule.groups().collect::<Vec<_>>(),
            vec![&[0][..], &[1][..]]
        );
        assert!(
            schedule
                .validate_pattern(StructuralPatternView::new(1, 2, &[0, 1], &[0]).unwrap())
                .is_err()
        );
        let empty = StructuralPattern::new(0, 0, vec![0], vec![]).unwrap();
        assert_eq!(
            ColoringSchedule::new(empty, ColoringKind::DistanceOne, vec![])
                .unwrap()
                .color_count(),
            0
        );
    }
}
