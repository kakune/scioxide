//! Dense vectors and matrices, and strided views of borrowed slices.
//!
//! [`Vector`] and [`Matrix`] own contiguous row-major storage. [`VectorView`],
//! [`VectorViewMut`], [`MatrixView`] and [`MatrixViewMut`] borrow a slice with
//! element strides. [`FiniteMatrixView`] records that every element was
//! checked to be finite, so later calls do not check again.

mod arithmetic;
mod kernel;
mod matrix;
mod vector;
mod view;

pub use matrix::Matrix;
pub use vector::Vector;
pub use view::{
    CopyPolicy, FiniteMatrixView, MatrixView, MatrixViewMut, VectorView, VectorViewMut,
};

pub(crate) use view::FiniteVectorView;
