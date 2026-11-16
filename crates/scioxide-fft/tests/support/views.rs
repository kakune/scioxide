//! Borrow valid test arrays with independent data and layout lifetimes.
#![expect(clippy::unwrap_used)]

use scioxide_core::{NdView, StridedLayout};

pub fn view<'d, 'l, T>(
    data: &'d [T],
    shape: &'l [usize],
    strides: &'l [usize],
) -> NdView<'d, 'l, T> {
    NdView::new(data, StridedLayout::new(shape, strides).unwrap()).unwrap()
}
