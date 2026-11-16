#[allow(unused_macros)]
macro_rules! view {
    ($values:expr) => {
        scioxide_core::NdView::new(
            $values,
            scioxide_core::StridedLayout::new(&[($values).len()], &[1]).unwrap(),
        )
        .unwrap()
    };
}
#[allow(unused_macros)]
macro_rules! view_mut {
    ($values:expr) => {
        match scioxide_core::StridedLayout::new(&[($values).len()], &[1]).unwrap() {
            layout => scioxide_core::NdViewMut::new($values, layout).unwrap(),
        }
    };
}
