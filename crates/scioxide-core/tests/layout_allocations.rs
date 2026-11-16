//! Successful borrowed layout operations require no heap storage.
#![allow(clippy::unwrap_used)]

use scioxide_core::{NdView, NdViewMut, Shape, StridedLayout};

#[test]
fn construction_binding_permutation_and_traversal_execute() {
    let mut output = [0_u32; 35];
    let input = [1_u32; 35];
    {
        assert_eq!(Shape::new(&[2, 3, 4]).unwrap().len(), 24);
        let layout = StridedLayout::new(&[2, 3, 4], &[20, 1, 4]).unwrap();
        let view = NdView::new(&input, layout).unwrap();
        assert_eq!(view.iter().copied().sum::<u32>(), 24);
        let permuted = layout
            .permuted(&[2, 0, 1], &[4, 2, 3], &[4, 20, 1])
            .unwrap();
        let mut result = NdViewMut::new(&mut output, permuted).unwrap();
        *result.get_mut([3, 1, 2]).unwrap() = 7;
        assert_eq!(*result.as_view().get([3, 1, 2]).unwrap(), 7);
        let batch = layout.axis(1).unwrap();
        for i in 0..batch.len() {
            assert_eq!(batch.lane(i).unwrap().offsets().count(), 3);
        }
        assert!(StridedLayout::new_broadcast(&[2, 3], &[0, 1]).is_ok());
    };
    assert_eq!(output[34], 7);
}

#[test]
fn row_major_construction_and_traversal_execute() {
    use scioxide_core::SciOxideErrorKind;
    let input = [1_u32; 24];
    let mut output = [0_u32; 24];
    {
        let layout = StridedLayout::row_major(&[2, 3, 4]).unwrap();
        assert_eq!(layout.strides(), [12, 4, 1]);
        assert_eq!(layout.offset([1, 2, 3]).unwrap(), 23);
        assert!(layout.offsets().eq(0..24));
        let view = NdView::row_major(&input, &[2, 3, 4]).unwrap();
        assert_eq!(view.iter().copied().sum::<u32>(), 24);
        let lanes = view.layout().axis(1).unwrap();
        for i in 0..lanes.len() {
            assert_eq!(lanes.lane(i).unwrap().offsets().count(), 3);
        }
        let mut result = NdViewMut::row_major(&mut output, &[2, 3, 4]).unwrap();
        *result.get_mut([1, 2, 3]).unwrap() = 7;
        assert_eq!(
            StridedLayout::row_major(&[2; 9]).unwrap_err().kind(),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            StridedLayout::row_major(&[usize::MAX, 2])
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    };
    assert_eq!(output[23], 7);
}

#[test]
fn permutation_ranks_are_dimensions_and_values_are_invalid_input() {
    use scioxide_core::{InvalidInputReason, SciOxideErrorKind, SciOxideErrorView};
    let layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
    {
        for (order, extents, strides, actual) in [
            (&[0][..], &[2, 3][..], &[3, 1][..], 1),
            (&[0, 1, 2][..], &[2, 3][..], &[3, 1][..], 3),
            (&[0, 1][..], &[2][..], &[3, 1][..], 1),
            (&[0, 1][..], &[2, 3, 1][..], &[3, 1][..], 3),
            (&[0, 1][..], &[2, 3][..], &[3][..], 1),
            (&[0, 1][..], &[2, 3][..], &[3, 1, 1][..], 3),
        ] {
            let error = layout.permuted(order, extents, strides).unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::DimensionMismatch {
                 expected, actual: found,
            } if expected == [2] && found == [actual])
            );
        }
        for (order, extents, strides) in [
            (&[0, 0][..], &[2, 3][..], &[3, 1][..]),
            (&[0, 2][..], &[2, 3][..], &[3, 1][..]),
            (&[1, 0][..], &[2, 3][..], &[1, 3][..]),
            (&[1, 0][..], &[3, 2][..], &[3, 1][..]),
        ] {
            let error = layout.permuted(order, extents, strides).unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::Inconsistent,
                    ..
                }
            ));
        }
        let scalar = StridedLayout::new(&[], &[]).unwrap();
        assert_eq!(scalar.permuted(&[], &[], &[]).unwrap().shape().rank(), 0);
        let empty = StridedLayout::new(&[0, 2], &[2, 1]).unwrap();
        assert_eq!(
            empty
                .permuted(&[1, 0], &[2, 0], &[1, 2])
                .unwrap()
                .shape()
                .len(),
            0
        );
    };
}
