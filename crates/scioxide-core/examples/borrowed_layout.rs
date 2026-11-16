//! Traverse a transposed, padded layout and update caller-owned output.
use scioxide_core::{NdView, NdViewMut, SciOxideError, StridedLayout};

fn main() -> Result<(), SciOxideError> {
    let storage = [1.0_f64, 2.0, 0.0, 3.0, 4.0];
    let layout = StridedLayout::new(&[2, 2], &[3, 1])?;
    let transposed = layout.permuted(&[1, 0], &[2, 2], &[1, 3])?;
    let view = NdView::new(&storage, transposed)?;
    let mut sums = [0.0; 2];
    let mut output = NdViewMut::row_major(&mut sums, &[2])?;
    let lanes = view.layout().axis(1)?;
    for batch in 0..lanes.len() {
        *output.get_mut([batch])? = lanes.lane(batch)?.offsets().map(|i| storage[i]).sum();
    }
    assert_eq!(sums, [4.0, 6.0]);
    Ok(())
}
