//! Strided multidimensional arrays over slices you provide.
//!
//! A [`StridedLayout`] maps a coordinate `(i₀, …, i_{r−1})` to the storage
//! offset `Σ iₖ·strideₖ`. Extents borrow a slice you provide. Strides borrow a
//! second slice, or [`StridedLayout::row_major`] derives them from the extents;
//! they count elements, not bytes. Strides are unsigned, so an array cannot
//! run backward; to start at an offset, pass a subslice. A row-major `m × n`
//! matrix has strides `[n, 1]` and a column-major one has `[1, m]`. Logical
//! iteration is always row-major (last axis fastest), whatever the strides.
//!
//! A rank-zero layout holds one scalar, and a zero extent makes the layout
//! empty. [`NdView`] and [`NdViewMut`] bind a layout to a slice.

use core::borrow::Borrow;

use crate::size::{checked_add, checked_mul, checked_product};
use crate::{Dimensions, InvalidInputReason, SciOxideError};

fn invalid(argument: crate::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}

/// The extents of an array, with the element count checked for overflow.
#[derive(Debug, Clone, Copy)]
pub struct Shape<'a> {
    extents: &'a [usize],
    len: usize,
}

impl<'a> Shape<'a> {
    /// Creates a shape from its extents.
    ///
    /// A shape with a zero extent has zero elements, even if the product of the
    /// other extents would overflow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the element count of a nonempty shape exceeds `usize::MAX`.
    pub fn new(extents: &'a [usize]) -> Result<Self, SciOxideError> {
        Ok(Self {
            extents,
            len: checked_product(extents, crate::quantity!("layout_cardinality"))?,
        })
    }

    /// Returns the extents, including zero extents.
    pub const fn extents(self) -> &'a [usize] {
        self.extents
    }
    /// Returns the extent of axis `axis`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= self.rank()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let shape = scioxide_core::Shape::new(&[2, 0, 3])?;
    /// assert_eq!(shape.extent(1)?, 0);
    /// assert_eq!(
    ///     shape.extent(3).unwrap_err().kind(),
    ///     scioxide_core::SciOxideErrorKind::OutOfBounds
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn extent(self, axis: usize) -> Result<usize, SciOxideError> {
        self.extents.get(axis).copied().ok_or_else(|| {
            SciOxideError::out_of_bounds(crate::quantity!("layout_axis"), axis, self.rank())
        })
    }

    /// Returns the number of axes.
    pub const fn rank(self) -> usize {
        self.extents.len()
    }
    /// Returns the number of elements, the product of the extents.
    pub const fn len(self) -> usize {
        self.len
    }
    /// Returns whether any extent is zero.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Extents and element strides of an array, with its storage span checked for overflow.
///
/// Element `(i₀, …, i_{r−1})` is stored at offset `Σ iₖ·strideₖ`. [`Self::new`]
/// requires that distinct elements have distinct offsets.
/// [`Self::new_broadcast`] also accepts layouts where elements share an
/// offset, for example a zero stride; such a layout can be read but bound to
/// [`NdViewMut`] only if distinct offsets were proved anyway.
///
/// For rank up to two the distinct-offset test is exact. For higher ranks it
/// is a sufficient test: sorted by stride, each stride must be at least the
/// span of the axes with smaller strides. Some layouts with distinct offsets
/// fail it.
///
/// [`Self::row_major`] creates the contiguous row-major layout of a shape
/// without a stride slice.
#[derive(Clone, Copy)]
pub struct StridedLayout<'a> {
    shape: Shape<'a>,
    strides: Strides<'a>,
    span: usize,
    injective: bool,
}

#[derive(Clone, Copy)]
enum Strides<'a> {
    /// Borrowed from the caller: `new`, `new_broadcast`, `permuted`.
    Explicit(&'a [usize]),
    /// Derived by `row_major`; the entries from the rank on are zero.
    RowMajor([usize; StridedLayout::ROW_MAJOR_MAX_RANK]),
}

impl core::fmt::Debug for StridedLayout<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("StridedLayout")
            .field("shape", &self.shape)
            .field("strides", &self.strides())
            .field("span", &self.span)
            .field("injective", &self.injective)
            .finish()
    }
}

impl<'a> StridedLayout<'a> {
    /// Largest rank that [`Self::row_major`] accepts.
    pub const ROW_MAJOR_MAX_RANK: usize = 8;

    /// Creates the contiguous row-major layout of `shape`: `strideₖ = Πⱼ₍ⱼ>ₖ₎ shapeⱼ`.
    ///
    /// `shapeⱼ` is the extent of axis `j`, so the last axis has stride 1 and
    /// the elements fill `0..len` in row-major order (last axis fastest). The
    /// layout is injective by construction: an offset is the mixed-radix
    /// numeral of the row-major linear index. Construction costs O(rank) and
    /// runs no injectivity search; the strides are stored in the layout,
    /// which therefore supports at most
    /// [`Self::ROW_MAJOR_MAX_RANK`] axes. Use [`Self::new`] for padded,
    /// transposed or higher-rank arrays.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `shape.len() > Self::ROW_MAJOR_MAX_RANK`.
    /// - `SizeOverflow` if the element count of a nonempty shape exceeds `usize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::StridedLayout;
    ///
    /// let layout = StridedLayout::row_major(&[2, 3])?;
    /// assert_eq!(layout.strides(), [3, 1]);
    /// assert_eq!(layout.offset([1, 2])?, 5);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn row_major(shape: &'a [usize]) -> Result<Self, SciOxideError> {
        if shape.len() > Self::ROW_MAJOR_MAX_RANK {
            return Err(SciOxideError::invalid_input(
                crate::argument!("shape"),
                InvalidInputReason::OutOfRange,
                "row-major layouts support at most 8 axes",
            ));
        }
        let checked = Shape::new(shape)?;
        // For a nonempty shape every partial product divides the checked
        // element count, so only a shape with a zero extent can saturate; it
        // addresses no element.
        let mut strides = [0; Self::ROW_MAJOR_MAX_RANK];
        strides[..shape.len()].iter_mut().zip(shape).rev().fold(
            1_usize,
            |stride, (slot, &extent)| {
                *slot = stride;
                stride.saturating_mul(extent)
            },
        );
        Ok(Self {
            shape: checked,
            strides: Strides::RowMajor(strides),
            span: checked.len,
            injective: true,
        })
    }

    /// Creates a layout whose elements have distinct offsets.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `extents` and `strides` differ in length.
    /// - `SizeOverflow` if the element count or the storage span overflows.
    /// - `InvalidInput` (`Inconsistent`) if two elements may share an offset.
    pub fn new(extents: &'a [usize], strides: &'a [usize]) -> Result<Self, SciOxideError> {
        let layout = Self::new_broadcast(extents, strides)?;
        layout.require_injective(crate::argument!("strides"))?;
        Ok(layout)
    }

    /// Creates a layout in which elements may share offsets, as in broadcasting with zero strides.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `extents` and `strides` differ in length.
    /// - `SizeOverflow` if the element count or the storage span overflows.
    pub fn new_broadcast(
        extents: &'a [usize],
        strides: &'a [usize],
    ) -> Result<Self, SciOxideError> {
        let shape = Shape::new(extents)?;
        let span = shape_strided_span(shape, strides)?;
        Ok(Self {
            shape,
            strides: Strides::Explicit(strides),
            span,
            injective: layout_is_injective(extents, strides),
        })
    }

    /// Returns the shape.
    pub const fn shape(self) -> Shape<'a> {
        self.shape
    }
    /// Returns the strides, in elements.
    pub const fn strides(&self) -> &[usize] {
        match &self.strides {
            Strides::Explicit(strides) => strides,
            Strides::RowMajor(strides) => strides.split_at(self.shape.rank()).0,
        }
    }
    /// Returns the minimum storage length, `1 + Σ (extentₖ − 1)·strideₖ`, or zero for an empty layout.
    pub const fn required_span(self) -> usize {
        self.span
    }
    /// Returns whether distinct elements were proved to have distinct offsets.
    pub const fn is_proven_injective(self) -> bool {
        self.injective
    }

    pub(crate) fn is_contiguous(self) -> bool {
        let mut expected = 1_usize;
        self.shape
            .extents
            .iter()
            .zip(self.strides())
            .rev()
            .all(|(&extent, &stride)| {
                if extent > 1 && stride != expected {
                    return false;
                }
                if let Some(next) = expected.checked_mul(extent) {
                    expected = next;
                    true
                } else {
                    false
                }
            })
    }

    fn require_injective(self, argument: crate::Argument) -> Result<(), SciOxideError> {
        if self.injective {
            Ok(())
        } else {
            Err(invalid(
                argument,
                "mutable layout non-overlap could not be proved",
            ))
        }
    }

    /// Checks that storage of length `len` is long enough; extra trailing storage is allowed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `len < self.required_span()`.
    pub fn validate_backing_len(self, len: usize) -> Result<(), SciOxideError> {
        validate_layout_backing_len(len, self.span)
    }

    /// Returns the storage offset `Σ iₖ·strideₖ` of a coordinate, in O(rank) time.
    ///
    /// The coordinate can be any exact-size sequence of indices, such as a
    /// slice, an array or a mapped iterator.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the coordinate length differs from the rank.
    /// - `OutOfBounds` if a component is not less than its extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::StridedLayout;
    ///
    /// // Row-major 2 × 3: element (1, 2) is at 1·3 + 2·1 = 5.
    /// let layout = StridedLayout::new(&[2, 3], &[3, 1])?;
    /// assert_eq!(layout.offset(&[1, 2])?, 5);
    /// assert_eq!(layout.offset([2, 1].iter().rev())?, 5);
    /// assert!(layout.offset(&[2, 0]).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn offset<I>(self, coordinates: I) -> Result<usize, SciOxideError>
    where
        I: IntoIterator,
        I::Item: Borrow<usize>,
        I::IntoIter: ExactSizeIterator,
    {
        let coordinates = coordinates.into_iter();
        if coordinates.len() != self.shape.rank() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.shape.rank()),
                Dimensions::one(coordinates.len()),
            ));
        }
        // The sum is returned only when every component is in range; then it
        // is below `span`, which was computed with checked arithmetic at
        // construction. Earlier terms of a rejected coordinate (for example in
        // an empty layout with huge strides) may wrap, but are discarded.
        coordinates
            .zip(self.shape.extents)
            .zip(self.strides())
            .try_fold(0_usize, |offset, ((index, &extent), &stride)| {
                let index = *index.borrow();
                if index >= extent {
                    return Err(SciOxideError::out_of_bounds(
                        crate::quantity!("layout_coordinate"),
                        index,
                        extent,
                    ));
                }
                Ok(offset.wrapping_add(index.wrapping_mul(stride)))
            })
    }

    /// Returns the storage offsets of all elements in row-major order (last axis fastest).
    pub fn offsets(self) -> impl ExactSizeIterator<Item = usize> + 'a {
        (0..self.shape.len).map(move |index| self.linear_offset(index, None))
    }

    pub(crate) fn linear_offset(self, mut index: usize, skip: Option<usize>) -> usize {
        // Decode a row-major linear index, last axis fastest. With `skip`, that
        // axis is left out of the decoding, which yields the base offset of one
        // lane along it.
        let mut offset = 0;
        for (axis, (&extent, &stride)) in self
            .shape
            .extents
            .iter()
            .zip(self.strides())
            .enumerate()
            .rev()
        {
            if Some(axis) != skip {
                offset += (index % extent) * stride;
                index /= extent;
            }
        }
        offset
    }

    /// Splits the array into one-dimensional lanes along `axis`.
    ///
    /// Each lane runs along `axis`, and the lanes are enumerated by the other
    /// coordinates in row-major order. See [`BatchLayout`].
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= rank`.
    /// - `SizeOverflow` if the number of lanes overflows.
    pub fn axis(self, axis: usize) -> Result<BatchLayout<'a>, SciOxideError> {
        self.shape.extent(axis)?;
        let empty_batches = self
            .shape
            .extents
            .iter()
            .enumerate()
            .any(|(i, &n)| i != axis && n == 0);
        let batches = if empty_batches {
            0
        } else {
            self.shape
                .extents
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != axis)
                .try_fold(1, |count, (_, &n)| {
                    checked_mul(count, n, crate::quantity!("layout_batch_count"))
                })?
        };
        Ok(BatchLayout {
            layout: self,
            axis,
            batches,
        })
    }

    /// Returns the layout with permuted axes, such as a transpose.
    ///
    /// `order[new_axis]` is the original axis. You provide the permuted
    /// `extents` and `strides`, `extents[k] = self.extents[order[k]]` and
    /// likewise for strides, so that the result can borrow them; this method
    /// checks them.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `order`, `extents` or `strides` has a length
    ///   other than the rank.
    /// - `InvalidInput` (`Inconsistent`) if `order` is not a permutation, or
    ///   `extents` or `strides` does not match it.
    pub fn permuted<'b>(
        self,
        order: &[usize],
        extents: &'b [usize],
        strides: &'b [usize],
    ) -> Result<StridedLayout<'b>, SciOxideError> {
        for actual in [order.len(), extents.len(), strides.len()] {
            if actual != self.shape.rank() {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::one(self.shape.rank()),
                    Dimensions::one(actual),
                ));
            }
        }
        for (i, &axis) in order.iter().enumerate() {
            if axis >= order.len() || order[..i].contains(&axis) {
                return Err(invalid(
                    crate::argument!("order"),
                    "axis order must be a permutation",
                ));
            }
            if extents[i] != self.shape.extents[axis] {
                return Err(invalid(
                    crate::argument!("extents"),
                    "permuted metadata must match the original axes",
                ));
            }
            if strides[i] != self.strides()[axis] {
                return Err(invalid(
                    crate::argument!("strides"),
                    "permuted metadata must match the original axes",
                ));
            }
        }
        // Permuting axes preserves both the span and the existing alias proof.
        Ok(StridedLayout {
            shape: Shape {
                extents,
                len: self.shape.len,
            },
            strides: Strides::Explicit(strides),
            span: self.span,
            injective: self.injective,
        })
    }
}

/// One lane of a [`BatchLayout`]: offsets `start + i·stride` for `i < len`.
#[derive(Debug, Clone, Copy)]
pub struct AxisLayout {
    offset: usize,
    len: usize,
    stride: usize,
}
impl AxisLayout {
    /// Returns the offset of the first element; zero for an empty lane.
    pub const fn start(self) -> usize {
        self.offset
    }
    /// Returns the number of elements in the lane.
    pub const fn len(self) -> usize {
        self.len
    }
    /// Returns whether the lane is empty.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
    /// Returns the stride along the lane, in elements.
    pub const fn stride(self) -> usize {
        self.stride
    }
    /// Returns the storage offsets of the lane, in order.
    pub fn offsets(self) -> impl ExactSizeIterator<Item = usize> + Clone {
        (0..self.len).map(move |index| self.offset + index * self.stride)
    }
}

/// The one-dimensional lanes of an array along one axis, created by [`StridedLayout::axis`].
///
/// The lanes are enumerated by the other coordinates in row-major order. If
/// the selected axis has extent zero the lanes are empty; if another axis has
/// extent zero there are no lanes.
#[derive(Debug, Clone, Copy)]
pub struct BatchLayout<'a> {
    layout: StridedLayout<'a>,
    axis: usize,
    batches: usize,
}
impl BatchLayout<'_> {
    /// Returns the number of lanes, the product of the other extents.
    pub const fn len(self) -> usize {
        self.batches
    }
    /// Returns whether there are no lanes.
    pub const fn is_empty(self) -> bool {
        self.batches == 0
    }
    /// Returns the selected axis.
    pub const fn axis(self) -> usize {
        self.axis
    }
    /// Returns lane `batch`, counting from zero in row-major order of the other coordinates.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `batch >= self.len()`.
    pub fn lane(self, batch: usize) -> Result<AxisLayout, SciOxideError> {
        if batch >= self.batches {
            return Err(SciOxideError::out_of_bounds(
                crate::quantity!("layout_batch"),
                batch,
                self.batches,
            ));
        }
        let len = self.layout.shape.extents[self.axis];
        let offset = if len == 0 {
            0
        } else {
            self.layout.linear_offset(batch, Some(self.axis))
        };
        Ok(AxisLayout {
            offset,
            len,
            stride: self.layout.strides()[self.axis],
        })
    }
}

/// A read-only strided array: a slice together with a [`StridedLayout`].
///
/// A view is `Copy`: it holds a shared slice and a `Copy` layout.
#[derive(Debug)]
pub struct NdView<'data, 'layout, T> {
    data: &'data [T],
    layout: StridedLayout<'layout>,
}
impl<T> Clone for NdView<'_, '_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for NdView<'_, '_, T> {}
impl<'data, 'layout, T> NdView<'data, 'layout, T> {
    /// Returns the whole underlying slice, in storage order.
    ///
    /// The slice includes padding and trailing elements, and an element that
    /// several coordinates share appears once. Use [`Self::layout`] or
    /// [`Self::iter`] to find the array elements. Padding is not checked.
    ///
    /// ```
    /// use scioxide_core::{NdView, StridedLayout};
    ///
    /// // Stride 2 skips the padding element 99.
    /// let data = [1, 99, 2];
    /// let shape = [2];
    /// let strides = [2];
    /// let view = NdView::new(&data, StridedLayout::new(&shape, &strides)?)?;
    /// assert_eq!(view.storage(), &data);
    /// assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn storage(&self) -> &'data [T] {
        self.data
    }

    /// Creates a view of `data` with `layout`; broadcast layouts are allowed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `data` is shorter than `layout.required_span()`.
    pub fn new(data: &'data [T], layout: StridedLayout<'layout>) -> Result<Self, SciOxideError> {
        validate_layout_backing_len(data.len(), layout.span)?;
        Ok(Self { data, layout })
    }

    /// Creates a view of `data` with the row-major layout [`StridedLayout::row_major`] of `shape`.
    ///
    /// Storage beyond the `Π shapeₖ` elements of the array is allowed and
    /// ignored.
    ///
    /// # Errors
    ///
    /// - The errors of [`StridedLayout::row_major`].
    /// - `DimensionMismatch` if `data.len() < Π shapeₖ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::NdView;
    ///
    /// let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let view = NdView::row_major(&x, &[2, 3])?;
    /// assert!((view.get([1, 2])? - 6.0).abs() < 1.0e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn row_major(data: &'data [T], shape: &'layout [usize]) -> Result<Self, SciOxideError> {
        Self::new(data, StridedLayout::row_major(shape)?)
    }
    /// Returns the layout.
    pub const fn layout(&self) -> StridedLayout<'layout> {
        self.layout
    }
    /// Returns the element at a coordinate.
    ///
    /// # Errors
    ///
    /// The errors of [`StridedLayout::offset`].
    pub fn get<I>(&self, coordinates: I) -> Result<&'data T, SciOxideError>
    where
        I: IntoIterator,
        I::Item: Borrow<usize>,
        I::IntoIter: ExactSizeIterator,
    {
        Ok(&self.data[self.layout.offset(coordinates)?])
    }
    /// Returns the elements in row-major order; shared elements appear once per coordinate.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'data T> + '_ {
        self.layout.offsets().map(|offset| &self.data[offset])
    }
}

/// A mutable strided array whose elements are proved to have distinct offsets.
#[derive(Debug)]
pub struct NdViewMut<'data, 'layout, T> {
    data: &'data mut [T],
    layout: StridedLayout<'layout>,
}
impl<'data, 'layout, T> NdViewMut<'data, 'layout, T> {
    /// Creates a mutable view of `data` with `layout`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `data` is shorter than `layout.required_span()`.
    /// - `InvalidInput` (`Inconsistent`) if distinct offsets were not proved
    ///   for `layout` (see [`StridedLayout::is_proven_injective`]).
    pub fn new(
        data: &'data mut [T],
        layout: StridedLayout<'layout>,
    ) -> Result<Self, SciOxideError> {
        validate_layout_backing_len(data.len(), layout.span)?;
        layout.require_injective(crate::argument!("layout"))?;
        Ok(Self { data, layout })
    }

    /// Creates a mutable view of `data` with the row-major layout [`StridedLayout::row_major`] of `shape`.
    ///
    /// Storage beyond the `Π shapeₖ` elements of the array is allowed and left
    /// untouched. A row-major layout is injective, so the distinct-offset
    /// check of [`Self::new`] always passes.
    ///
    /// # Errors
    ///
    /// - The errors of [`StridedLayout::row_major`].
    /// - `DimensionMismatch` if `data.len() < Π shapeₖ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::NdViewMut;
    ///
    /// let mut y = [0.0_f64; 3];
    /// let mut view = NdViewMut::row_major(&mut y, &[3])?;
    /// *view.get_mut([1])? = 2.5;
    /// assert!((y[1] - 2.5).abs() < 1.0e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn row_major(data: &'data mut [T], shape: &'layout [usize]) -> Result<Self, SciOxideError> {
        Self::new(data, StridedLayout::row_major(shape)?)
    }
    /// Returns the layout.
    pub const fn layout(&self) -> StridedLayout<'layout> {
        self.layout
    }
    /// Returns a read-only view of the same array.
    pub fn as_view(&self) -> NdView<'_, 'layout, T> {
        NdView {
            data: self.data,
            layout: self.layout,
        }
    }
    /// Calls `visit` on each element in row-major order.
    ///
    /// An empty layout makes no calls.
    pub fn for_each_mut(&mut self, mut visit: impl FnMut(&mut T)) {
        if self.layout.is_contiguous() {
            for value in &mut self.data[..self.layout.shape.len] {
                visit(value);
            }
            return;
        }
        for offset in self.layout.offsets() {
            visit(&mut self.data[offset]);
        }
    }

    /// Returns the elements as one slice in row-major order, if the layout is
    /// contiguous row-major without padding; otherwise `None`.
    ///
    /// ```
    /// use scioxide_core::NdViewMut;
    ///
    /// let mut data = [0, 0, 0];
    /// let mut view = NdViewMut::row_major(&mut data, &[3])?;
    /// view.contiguous_mut().map(|values| values.copy_from_slice(&[1, 2, 3]));
    /// assert_eq!(data, [1, 2, 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn contiguous_mut(&mut self) -> Option<&mut [T]> {
        self.layout
            .is_contiguous()
            .then(|| &mut self.data[..self.layout.shape.len])
    }

    /// Returns the element at a coordinate, mutably.
    ///
    /// # Errors
    ///
    /// The errors of [`StridedLayout::offset`].
    pub fn get_mut<I>(&mut self, coordinates: I) -> Result<&mut T, SciOxideError>
    where
        I: IntoIterator,
        I::Item: Borrow<usize>,
        I::IntoIter: ExactSizeIterator,
    {
        Ok(&mut self.data[self.layout.offset(coordinates)?])
    }
}

/// Returns the storage span `1 + Σ (extentₖ − 1)·strideₖ` of a strided layout, checked for overflow.
///
/// Any strides are accepted, including zero strides; whether offsets are
/// distinct is not checked. A rank-zero layout spans 1 and a layout with a zero
/// extent spans 0.
///
/// # Errors
///
/// - `DimensionMismatch` if `extents` and `strides` differ in length.
/// - `SizeOverflow` if the element count or the span overflows.
///
/// ```
/// use scioxide_core::backend::checked_strided_span;
///
/// assert_eq!(checked_strided_span(&[2, 3], &[3, 1])?, 6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_strided_span(extents: &[usize], strides: &[usize]) -> Result<usize, SciOxideError> {
    shape_strided_span(Shape::new(extents)?, strides)
}

fn shape_strided_span(shape: Shape<'_>, strides: &[usize]) -> Result<usize, SciOxideError> {
    let extents = shape.extents;
    if extents.len() != strides.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(extents.len()),
            Dimensions::one(strides.len()),
        ));
    }
    if shape.is_empty() {
        return Ok(0);
    }
    extents
        .iter()
        .zip(strides)
        // The largest offset is sum((extent - 1) * stride), so the required span is
        // one more than that.
        .try_fold(1, |span, (&extent, &stride)| {
            checked_add(
                span,
                checked_mul(extent - 1, stride, crate::quantity!("layout_storage_span"))?,
                crate::quantity!("layout_storage_span"),
            )
        })
}

/// Checks that storage of length `actual` covers a span of `required` elements.
///
/// Only the length is checked, not the offsets.
///
/// # Errors
///
/// - `DimensionMismatch` if `actual < required`.
///
/// ```
/// use scioxide_core::backend::validate_layout_backing_len;
///
/// validate_layout_backing_len(8, 6)?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn validate_layout_backing_len(actual: usize, required: usize) -> Result<(), SciOxideError> {
    if actual < required {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(required),
            Dimensions::one(actual),
        ))
    } else {
        Ok(())
    }
}

/// Returns whether a `rows × cols` strided matrix gives distinct offsets to distinct entries.
///
/// The test is exact. An empty matrix passes, and a zero stride on an axis of
/// length above one fails. It uses Euclid's algorithm on the strides, in
/// O(log max(row_stride, col_stride)) time. The storage span is not checked.
///
/// ```
/// use scioxide_core::backend::matrix_layout_is_injective;
///
/// // Row-major 2 × 3 has distinct offsets; with both strides 1 entries collide.
/// assert!(matrix_layout_is_injective(2, 3, 3, 1));
/// assert!(!matrix_layout_is_injective(2, 3, 1, 1));
/// ```
pub fn matrix_layout_is_injective(
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> bool {
    if rows == 0 || cols == 0 || (rows == 1 && cols == 1) {
        return true;
    }
    if (rows > 1 && row_stride == 0) || (cols > 1 && col_stride == 0) {
        return false;
    }
    if rows == 1 || cols == 1 {
        return true;
    }
    // Two elements collide when di * s = dj * t for some |di| < rows and
    // |dj| < cols, not both zero. With g = gcd(s, t), the smallest nonzero
    // solution is di = t / g, dj = s / g, so the map is injective exactly when
    // one of those minimal steps already leaves its extent.
    let divisor = crate::backend::gcd(row_stride, col_stride);
    rows <= col_stride / divisor || cols <= row_stride / divisor
}

// Called only after cardinality and reachable-span validation.
fn layout_is_injective(extents: &[usize], strides: &[usize]) -> bool {
    if extents.contains(&0) {
        return true;
    }
    let mut active = extents.iter().zip(strides).filter(|&(&n, _)| n > 1);
    let first = active.next();
    let second = active.next();
    if active.next().is_none() {
        return match (first, second) {
            (None, _) => true,
            (Some((_, &s)), None) => s != 0,
            (Some((&n, &s)), Some((&m, &t))) => matrix_layout_is_injective(n, m, s, t),
        };
    }
    // Higher ranks use a sufficient test: visit the axes in increasing stride
    // order and require every stride to be at least the span covered by all
    // smaller-stride axes (nested, non-interleaving strides). Some injective
    // layouts fail it, which is why the result is "proven" injective.
    let mut previous = None;
    let mut span = 1;
    loop {
        let next = extents
            .iter()
            .zip(strides)
            .enumerate()
            .filter(|&(axis, (&n, &stride))| {
                n > 1 && previous.is_none_or(|key| (stride, axis) > key)
            })
            .min_by_key(|&(axis, (_, &stride))| (stride, axis));
        let Some((axis, (&extent, &stride))) = next else {
            return true;
        };
        if stride < span {
            return false;
        }
        span += (extent - 1) * stride;
        previous = Some((stride, axis));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex;
    use alloc::vec::Vec;

    #[test]
    fn coordinates_may_be_slices_arrays_or_computed_iterators() {
        let layout = StridedLayout::new(&[2, 3], &[4, 1]).unwrap();
        let mut data = [0_u32; 7];
        let permutation = [1_usize, 0];
        let exponents = [2_usize, 1];
        let computed = || permutation.iter().map(|&axis| exponents[axis]);
        assert_eq!(layout.offset(&[1_usize, 2][..]).unwrap(), 6);
        assert_eq!(layout.offset([1, 2]).unwrap(), 6);
        assert_eq!(layout.offset(computed()).unwrap(), 6);
        let mut view = NdViewMut::new(&mut data, layout).unwrap();
        *view.get_mut(computed()).unwrap() = 7;
        assert_eq!(*view.as_view().get([1_usize, 2]).unwrap(), 7);
        for wrong_rank in [&[1_usize][..], &[0, 0, 0]] {
            assert!(matches!(
                layout.offset(wrong_rank).unwrap_err().view(),
                crate::SciOxideErrorView::DimensionMismatch { .. }
            ));
        }
        for out_of_bounds in [[2_usize, 0], [0, 3]] {
            assert!(matches!(
                view.get_mut(out_of_bounds).unwrap_err().view(),
                crate::SciOxideErrorView::OutOfBounds { .. }
            ));
        }
        assert_eq!(data, [0, 0, 0, 0, 0, 0, 7]);
    }

    #[test]
    fn immutable_storage_retains_padding_broadcasts_and_empty_backing() {
        let data = [1, 99, 2, 99, 3, 100];
        let shape = [2, 2];
        let strides = [0, 2];
        let view = NdView::new(
            &data,
            StridedLayout::new_broadcast(&shape, &strides).unwrap(),
        )
        .unwrap();
        assert_eq!(view.storage(), &data);
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2, 1, 2]);
        let empty_shape = [0, 2];
        let empty =
            NdView::new(&data, StridedLayout::new(&empty_shape, &strides).unwrap()).unwrap();
        assert_eq!(empty.storage(), &data);
        assert_eq!(empty.iter().count(), 0);
    }

    #[test]
    fn mutable_visits_follow_logical_order_without_touching_padding() {
        let mut data = [99; 9];
        let layout = StridedLayout::new(&[2, 2], &[2, 5]).unwrap();
        let mut view = NdViewMut::new(&mut data, layout).unwrap();
        let mut index = 0;
        view.for_each_mut(|value| {
            *value = index;
            index += 1;
        });
        assert_eq!(data, [0, 99, 2, 99, 99, 1, 99, 3, 99]);
        let mut empty = [];
        NdViewMut::new(&mut empty, StridedLayout::new(&[0], &[0]).unwrap())
            .unwrap()
            .for_each_mut(|_: &mut i32| index += 1);
        assert_eq!(index, 4);
    }

    #[test]
    fn scalar_empty_and_overflow_contracts() {
        let scalar = StridedLayout::new(&[], &[]).unwrap();
        assert_eq!(scalar.shape().len(), 1);
        assert_eq!(scalar.offset([0_usize; 0]).unwrap(), 0);
        assert_eq!(scalar.offsets().collect::<Vec<_>>(), [0]);
        assert!(NdView::new(&[] as &[u8], scalar).is_err());
        for shape in [[0, usize::MAX, 2], [usize::MAX, 0, 2], [usize::MAX, 2, 0]] {
            let empty = StridedLayout::new(&shape, &[usize::MAX; 3]).unwrap();
            assert_eq!(empty.required_span(), 0);
            assert_eq!(empty.offsets().count(), 0);
            assert!(NdViewMut::new(&mut [] as &mut [u8], empty).is_ok());
            assert!(matches!(
                (empty.offset([2, 1, 0])).as_ref().unwrap_err().view(),
                crate::SciOxideErrorView::OutOfBounds { .. }
            ));
        }
        assert!(matches!(
            (Shape::new(&[usize::MAX, 2])).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert!(matches!(
            (StridedLayout::new(&[2], &[usize::MAX]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert!(matches!(
            (StridedLayout::new(&[2, 2], &[usize::MAX - 1, 1]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert!(matches!(
            (StridedLayout::new(&[2], &[])).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::DimensionMismatch { .. }
        ));
        let singleton = StridedLayout::new(&[1, 1], &[usize::MAX, usize::MAX]).unwrap();
        assert_eq!(singleton.required_span(), 1);
    }

    #[test]
    fn exhaustive_small_layouts_match_independent_addresses() {
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for x in 0..7 {
                        for y in 0..7 {
                            for z in 0..7 {
                                let shape = [a, b, c];
                                let strides = [x, y, z];
                                let layout =
                                    StridedLayout::new_broadcast(&shape, &strides).unwrap();
                                let mut expected = Vec::new();
                                for i in 0..a {
                                    for j in 0..b {
                                        for k in 0..c {
                                            expected.push(i * x + j * y + k * z);
                                            assert_eq!(
                                                layout.offset([i, j, k]).unwrap(),
                                                i * x + j * y + k * z
                                            );
                                        }
                                    }
                                }
                                assert_eq!(layout.offsets().collect::<Vec<_>>(), expected);
                                assert_eq!(
                                    layout.required_span(),
                                    expected.iter().max().map_or(0, |n| n + 1)
                                );
                                let count = expected.len();
                                expected.sort_unstable();
                                expected.dedup();
                                let exact = count == expected.len();
                                if layout.is_proven_injective() {
                                    assert!(exact, "{shape:?} {strides:?}");
                                }
                                if shape.iter().filter(|&&n| n > 1).count() <= 2 {
                                    assert_eq!(layout.is_proven_injective(), exact);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn axis_extents_share_bounds_diagnostics() {
        for extents in [&[][..], &[0, 3][..], &[2, 0, 4][..]] {
            let shape = Shape::new(extents).unwrap();
            for (axis, &extent) in extents.iter().enumerate() {
                assert_eq!(shape.extent(axis).unwrap(), extent);
            }
            for axis in [extents.len(), usize::MAX] {
                let error = shape.extent(axis).unwrap_err();
                assert!(
                    matches!(error.view(), crate::SciOxideErrorView::OutOfBounds {
                    collection, index, size,
                } if collection == crate::quantity!("layout_axis")
                    && index == axis && size == extents.len())
                );
            }
        }
    }

    #[test]
    fn axes_batches_permutations_and_validation() {
        let layout = StridedLayout::new(&[2, 3, 4], &[20, 1, 4]).unwrap();
        let batches = layout.axis(1).unwrap();
        assert_eq!(batches.len(), 8);
        for batch in 0..8 {
            let lane = batches.lane(batch).unwrap();
            assert_eq!(
                lane.offsets().collect::<Vec<_>>(),
                [0, 1, 2].map(|j| (batch / 4) * 20 + (batch % 4) * 4 + j)
            );
        }
        assert!(batches.lane(8).is_err());
        assert!(layout.axis(3).is_err());
        assert!(layout.offset([0, 0]).is_err());
        assert!(layout.offset([2, 0, 0]).is_err());
        assert!(NdView::new(&[0; 34], layout).is_err());
        let transposed = layout
            .permuted(&[2, 0, 1], &[4, 2, 3], &[4, 20, 1])
            .unwrap();
        assert_eq!(transposed.offset([3, 1, 2]).unwrap(), 34);
        assert!(
            layout
                .permuted(&[0, 0, 2], &[2, 2, 4], &[20, 20, 4])
                .is_err()
        );
        assert!(
            layout
                .permuted(&[2, 0, 1], &[4, 2, 3], &[4, 21, 1])
                .is_err()
        );
        assert!(
            layout
                .permuted(&[0, 1, 3], &[2, 3, 4], &[20, 1, 4])
                .is_err()
        );
        assert!(layout.permuted(&[], &[], &[]).is_err());
        let empty = StridedLayout::new(&[3, 0, 2], &[usize::MAX; 3]).unwrap();
        assert_eq!(empty.axis(1).unwrap().len(), 6);
        assert_eq!(empty.axis(1).unwrap().lane(5).unwrap().start(), 0);
        assert!(empty.axis(1).unwrap().lane(5).unwrap().is_empty());
        assert!(empty.axis(0).unwrap().is_empty());
        assert!(
            StridedLayout::new(&[usize::MAX, 0, 2], &[0; 3])
                .unwrap()
                .axis(1)
                .is_err()
        );
    }

    #[test]
    fn aliasing_broadcast_and_conservative_rejection() {
        assert!(StridedLayout::new(&[3], &[0]).is_err());
        let broadcast = StridedLayout::new_broadcast(&[2, 3], &[0, 1]).unwrap();
        assert_eq!(
            NdView::new(&[1, 2, 3], broadcast)
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [1, 2, 3, 1, 2, 3]
        );
        let mut output = [7; 3];
        assert!(NdViewMut::new(&mut output, broadcast).is_err());
        assert_eq!(output, [7; 3]);
        // This injective 3D layout fails the deliberately sufficient span proof.
        let exotic = StridedLayout::new_broadcast(&[2, 2, 2], &[2, 3, 4]).unwrap();
        assert!(!exotic.is_proven_injective());
        assert_eq!(
            exotic.offsets().collect::<Vec<_>>(),
            [0, 4, 3, 7, 2, 6, 5, 9]
        );
        // Exact two-axis proof preserves interleaving that span separation rejects.
        assert!(StridedLayout::new(&[3, 2], &[2, 3]).is_ok());
        assert!(StridedLayout::new(&[2, 2, 2], &[1, 1, 4]).is_err());
    }

    fn saturating_row_major_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = alloc::vec![0; shape.len()];
        let mut stride = 1_usize;
        for (slot, &extent) in strides.iter_mut().zip(shape).rev() {
            *slot = stride;
            stride = stride.saturating_mul(extent);
        }
        strides
    }

    fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
        let len: usize = shape.iter().product();
        (0..len)
            .map(|mut index| {
                let mut coordinate = alloc::vec![0; shape.len()];
                for (component, &extent) in coordinate.iter_mut().zip(shape).rev() {
                    *component = index % extent;
                    index /= extent;
                }
                coordinate
            })
            .collect()
    }

    fn assert_same_layout(derived: StridedLayout<'_>, explicit: StridedLayout<'_>) {
        let shape = explicit.shape().extents();
        assert_eq!(derived.shape().extents(), shape);
        assert_eq!(derived.shape().len(), explicit.shape().len());
        assert_eq!(derived.strides(), explicit.strides());
        assert_eq!(derived.required_span(), explicit.required_span());
        assert_eq!(
            derived.is_proven_injective(),
            explicit.is_proven_injective()
        );
        assert_eq!(
            derived.offsets().collect::<Vec<_>>(),
            explicit.offsets().collect::<Vec<_>>()
        );
        if !explicit.shape().is_empty() {
            let short = explicit.required_span() - 1;
            assert_eq!(
                derived.validate_backing_len(short).unwrap_err().kind(),
                explicit.validate_backing_len(short).unwrap_err().kind()
            );
        }
        if explicit.shape().len() <= 24 {
            for coordinate in coordinates(shape) {
                assert_eq!(
                    derived.offset(&coordinate).unwrap(),
                    explicit.offset(&coordinate).unwrap()
                );
            }
        }
        let outside = shape.to_vec();
        if !outside.is_empty() {
            assert_eq!(
                derived.offset(&outside).unwrap_err().kind(),
                explicit.offset(&outside).unwrap_err().kind()
            );
        }
        for axis in 0..=shape.len() {
            match (derived.axis(axis), explicit.axis(axis)) {
                (Ok(derived), Ok(explicit)) => {
                    assert_eq!(derived.len(), explicit.len());
                    for batch in 0..explicit.len() {
                        let (derived, explicit) =
                            (derived.lane(batch).unwrap(), explicit.lane(batch).unwrap());
                        assert_eq!(derived.start(), explicit.start());
                        assert_eq!(derived.len(), explicit.len());
                        assert_eq!(derived.stride(), explicit.stride());
                        assert!(derived.offsets().eq(explicit.offsets()));
                    }
                }
                (derived, explicit) => assert_eq!(
                    derived.err().map(|error| error.kind()),
                    explicit.err().map(|error| error.kind()),
                    "axis {axis} of {shape:?}"
                ),
            }
        }
    }

    #[test]
    fn row_major_equals_explicit_row_major_strides() {
        let shapes: [&[usize]; 10] = [
            &[],
            &[0],
            &[5],
            &[1, 1],
            &[2, 3],
            &[3, 0, 2],
            &[0, usize::MAX, usize::MAX],
            &[1, 4, 1],
            &[2, 3, 4],
            &[2; 8],
        ];
        for shape in shapes {
            let strides = saturating_row_major_strides(shape);
            let derived = StridedLayout::row_major(shape).unwrap();
            let explicit = StridedLayout::new(shape, &strides).unwrap();
            assert_same_layout(derived, explicit);
            assert_eq!(derived.required_span(), derived.shape().len());
            assert!(derived.is_proven_injective());
            assert!(derived.offsets().eq(0..derived.shape().len()));
        }
        let overflowing_lanes = StridedLayout::row_major(&[0, usize::MAX, usize::MAX]).unwrap();
        assert_eq!(
            overflowing_lanes.axis(0).unwrap_err().kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(
            overflowing_lanes.axis(3).unwrap_err().kind(),
            crate::SciOxideErrorKind::OutOfBounds
        );
        let derived = StridedLayout::row_major(&[2, 3])
            .unwrap()
            .permuted(&[1, 0], &[3, 2], &[1, 3])
            .unwrap();
        let explicit = StridedLayout::new(&[2, 3], &[3, 1])
            .unwrap()
            .permuted(&[1, 0], &[3, 2], &[1, 3])
            .unwrap();
        assert_same_layout(derived, explicit);
    }

    #[test]
    fn row_major_rejects_high_rank_overflow_and_short_storage() {
        assert!(matches!(
            StridedLayout::row_major(&[2; 9]).unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput {
                argument,
                reason: InvalidInputReason::OutOfRange,
                ..
            } if argument == crate::argument!("shape")
        ));
        assert_eq!(
            StridedLayout::row_major(&[usize::MAX, 2])
                .unwrap_err()
                .kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        let x = [0.0_f64; 7];
        let mut y = [0.0_f64; 7];
        assert_eq!(
            NdView::row_major(&x[..5], &[2, 3]).unwrap_err().kind(),
            crate::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(
            NdViewMut::row_major(&mut y[..5], &[2, 3])
                .unwrap_err()
                .kind(),
            crate::SciOxideErrorKind::DimensionMismatch
        );
        assert!(NdView::row_major(&x, &[2, 3]).is_ok());
        let mut view = NdViewMut::row_major(&mut y, &[2, 3]).unwrap();
        assert_eq!(view.contiguous_mut().map(|values| values.len()), Some(6));
    }

    fn values<T: Copy + core::fmt::Debug + PartialEq>(values: [T; 4]) {
        let layout = StridedLayout::new(&[2, 2], &[1, 2]).unwrap();
        let view = NdView::new(&values, layout).unwrap();
        assert_eq!(
            view.iter().copied().collect::<Vec<_>>(),
            [values[0], values[2], values[1], values[3]]
        );
        let mut storage = values;
        let mut output = NdViewMut::new(&mut storage, layout).unwrap();
        *output.get_mut([1, 0]).unwrap() = values[3];
        assert_eq!(*output.as_view().get([1, 0]).unwrap(), values[3]);
        assert!(output.get_mut([2, 0]).is_err());
    }

    #[test]
    fn all_scalar_storage_domains() {
        values([0.0_f32, 1.0, 2.0, 3.0]);
        values([0.0_f64, 1.0, 2.0, 3.0]);
        values([0.0_f32, 1.0, 2.0, 3.0].map(|x| Complex::new(x, -x)));
        values([0.0_f64, 1.0, 2.0, 3.0].map(|x| Complex::new(x, -x)));
    }
}
