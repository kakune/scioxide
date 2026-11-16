//! Interface for an injected BLAS provider.
//!
//! All matrix operands of one call share one layout. Dimensions and leading
//! dimensions describe the stored matrix; the call's transform applies to it.
//! Band and packed operands are column-major. Level-3 routines copy a dense
//! matrix view without a direct BLAS encoding into provider scratch; the GEMV
//! matrix and input vector views are passed as they are, or the call follows the
//! fallback policy.
//! Conjugate-only matrix views are never passed as a BLAS transpose argument.
//! Empty products and zero-inner-dimension updates run in SciOxide, including the
//! `β` scaling. For updates of one triangle, only that triangle is checked.

pub(crate) mod dispatch;

use core::num::NonZeroUsize;

use crate::MatrixSide;
use scioxide_core::backend::ScalarKind;
use scioxide_core::{CapabilityError, Complex32, Complex64, DiagonalKind, Triangle};

/// BLAS routine that a [`BlasProvider`] can declare and execute.
///
/// Each variant corresponds to one [`BlasCall`] variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[must_use]
pub enum BlasOperation {
    /// `y := x` (COPY).
    Copy,
    /// `x := α x` (SCAL).
    Scale,
    /// `y := α x + y` (AXPY).
    Axpy,
    /// Unconjugated dot product `Σ xᵢ yᵢ` (DOT, DOTU).
    Dotu,
    /// Dot product `Σ conj(xᵢ) yᵢ` that conjugates `x` (DOTC).
    Dotc,
    /// Euclidean norm `‖x‖₂` (NRM2).
    Norm2,
    /// General matrix-vector product `y := α op(A) x + β y` (GEMV).
    Gemv,
    /// General matrix product `C := α op(A) op(B) + β C` (GEMM).
    Gemm,
    /// Symmetric rank-k update `C := α op(A) op(A)ᵀ + β C` (SYRK).
    Syrk,
    /// Hermitian rank-k update `C := α op(A) op(A)ᴴ + β C` (HERK).
    Herk,
    /// Triangular matrix product `B := α op(A) B` or `B := α B op(A)` (TRMM).
    Trmm,
    /// Triangular solve `op(A) X = α B` or `X op(A) = α B` (TRSM).
    Trsm,
    /// Exchanges `x` and `y` (SWAP).
    Swap,
    /// Sum `Σ (|Re xᵢ| + |Im xᵢ|)` (ASUM).
    Asum,
    /// Zero-based index of the first maximum of `|Re xᵢ| + |Im xᵢ|` (IAMAX).
    Iamax,
    /// Generates a real plane rotation (ROTG).
    Rotg,
    /// Generates a real modified plane rotation (ROTMG).
    Rotmg,
    /// Applies a real plane rotation (ROT).
    Rot,
    /// Applies a real modified plane rotation (ROTM).
    Rotm,
    /// `f32` dot product accumulated and returned in `f64` (DSDOT).
    WideDot,
    /// `f32` offset plus an `f32` dot product accumulated in `f64` (SDSDOT).
    OffsetDot,
    /// General band matrix-vector product (GBMV).
    Gbmv,
    /// Symmetric matrix-vector product (SYMV).
    Symv,
    /// Hermitian matrix-vector product (HEMV).
    Hemv,
    /// Symmetric band matrix-vector product (SBMV).
    Sbmv,
    /// Hermitian band matrix-vector product (HBMV).
    Hbmv,
    /// Symmetric packed matrix-vector product (SPMV).
    Spmv,
    /// Hermitian packed matrix-vector product (HPMV).
    Hpmv,
    /// Dense triangular solve `op(A) x = b` (TRSV).
    Trsv,
    /// Band triangular solve `op(A) x = b` (TBSV).
    Tbsv,
    /// Packed triangular solve `op(A) x = b` (TPSV).
    Tpsv,
    /// General rank-one update `A := α x yᵀ + A` (GER, GERU).
    Geru,
    /// General rank-one update `A := α x yᴴ + A` (GERC).
    Gerc,
    /// Symmetric rank-one update `A := α x xᵀ + A` (SYR).
    Syr,
    /// Symmetric rank-two update `A := α x yᵀ + α y xᵀ + A` (SYR2).
    Syr2,
    /// Hermitian rank-one update `A := α x xᴴ + A` with real `α` (HER).
    Her,
    /// Hermitian rank-two update `A := α x yᴴ + conj(α) y xᴴ + A` (HER2).
    Her2,
    /// Packed symmetric rank-one update (SPR).
    Spr,
    /// Packed symmetric rank-two update (SPR2).
    Spr2,
    /// Packed Hermitian rank-one update (HPR).
    Hpr,
    /// Packed Hermitian rank-two update (HPR2).
    Hpr2,
    /// Symmetric matrix product `C := α A B + β C` or `C := α B A + β C` (SYMM).
    Symm,
    /// Hermitian matrix product `C := α A B + β C` or `C := α B A + β C` (HEMM).
    Hemm,
    /// Symmetric rank-2k update `C := α op(A) op(B)ᵀ + α op(B) op(A)ᵀ + β C` (SYR2K).
    Syr2k,
    /// Hermitian rank-2k update `C := α op(A) op(B)ᴴ + conj(α) op(B) op(A)ᴴ + β C` (HER2K).
    Her2k,
}

const OPERATION_COUNT: usize = 45;

impl BlasOperation {
    /// Returns whether BLAS defines this routine for the scalar type.
    ///
    /// The symmetric Level 2 routines (`Symv`, `Sbmv`, `Spmv`, `Syr`, `Syr2`, `Spr`, `Spr2`)
    /// and the rotation routines are real only. The Hermitian routines and `Gerc` are complex
    /// only. `WideDot` and `OffsetDot` are `f32` only. All others accept every scalar type.
    pub const fn supports_scalar(self, scalar: ScalarKind) -> bool {
        let real = matches!(scalar, ScalarKind::F32 | ScalarKind::F64);
        match self {
            Self::Symv
            | Self::Sbmv
            | Self::Spmv
            | Self::Syr
            | Self::Syr2
            | Self::Spr
            | Self::Spr2
            | Self::Rotg
            | Self::Rotmg
            | Self::Rot
            | Self::Rotm => real,
            Self::Herk
            | Self::Hemv
            | Self::Hbmv
            | Self::Hpmv
            | Self::Gerc
            | Self::Her
            | Self::Her2
            | Self::Hpr
            | Self::Hpr2
            | Self::Hemm
            | Self::Her2k => !real,
            Self::WideDot | Self::OffsetDot => matches!(scalar, ScalarKind::F32),
            _ => true,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Dense storage order of a matrix operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum BlasLayout {
    /// Consecutive elements belong to one row.
    RowMajor,
    /// Consecutive elements belong to one column.
    ColumnMajor,
}

/// Transform `op(A)` applied to a stored matrix operand, or conjugation of a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum BlasTransform {
    /// `op(A) = A`.
    None,
    /// `op(A) = Aᵀ`.
    Transpose,
    /// Elementwise conjugation without transposition.
    ///
    /// Used only for vector operands. It is an extension to the BLAS interface, so a
    /// provider must declare it before SciOxide passes a conjugated complex vector.
    Conjugate,
    /// `op(A) = Aᴴ`.
    Adjoint,
}

/// Integer width of the provider's dimension, stride and leading-dimension arguments.
///
/// SciOxide does not pass an operand to the provider if one of these values exceeds the
/// signed maximum of the declared width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum BlasIntegerWidth {
    /// Signed 32-bit integers (LP64 interface).
    Lp64,
    /// Signed 64-bit integers (ILP64 interface).
    Ilp64,
}

/// How a provider uses the thread count passed with each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum BlasThreading {
    /// The provider runs on one thread. SciOxide always passes `workers = 1`.
    SingleThreaded,
    /// The provider uses at most the `workers` count passed with each call.
    Budgeted,
}

/// Accuracy that a provider declares for one routine, as a maximum distance in ulps.
///
/// The bound applies to each component of each output entry, inclusive. SciOxide does
/// not measure it. It uses the value only to decide eligibility: under
/// [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), a call goes to the provider only if
/// the routine declares `BlasUlp::new(0)`. Under `Accuracy::Standard` any declared bound
/// is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct BlasUlp {
    maximum: u32,
}

impl BlasUlp {
    /// Creates a bound of at most `maximum` ulps per component.
    pub const fn new(maximum: u32) -> Self {
        Self { maximum }
    }

    /// Returns the maximum distance in ulps.
    pub const fn maximum(self) -> u32 {
        self.maximum
    }
}

/// Operations and calling conventions that a [`BlasProvider`] supports.
///
/// Start from [`empty`](Self::empty) and add each supported layout, transform and
/// routine. SciOxide sends a call to the provider only if every operand fits this
/// declaration: a declared layout, transform and routine for the scalar type,
/// dimensions within the integer width, and slices aligned to
/// [`minimum_alignment`](Self::minimum_alignment). Other calls follow the
/// [`BlasFallbackPolicy`].
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::ScalarKind;
/// use scioxide_linalg::{
///     BlasCapabilities, BlasIntegerWidth, BlasLayout, BlasOperation, BlasTransform, BlasUlp,
/// };
///
/// let capabilities = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
///     .with_layout(BlasLayout::ColumnMajor)
///     .with_transform(BlasTransform::None)
///     .with_transform(BlasTransform::Transpose)
///     .with_operation(ScalarKind::F64, BlasOperation::Gemm, BlasUlp::new(4))?;
/// assert!(capabilities.supports_layout(BlasLayout::ColumnMajor));
/// let gemm = capabilities.operation_ulp(ScalarKind::F64, BlasOperation::Gemm);
/// assert_eq!(gemm, Some(BlasUlp::new(4)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct BlasCapabilities {
    layout_mask: u8,
    transform_mask: u8,
    integer_width: BlasIntegerWidth,
    minimum_alignment: usize,
    threading: BlasThreading,
    ulp: [[Option<BlasUlp>; OPERATION_COUNT]; 4],
}

impl BlasCapabilities {
    /// Creates a declaration with no layouts, transforms or routines.
    ///
    /// The alignment is one byte and the threading is [`BlasThreading::SingleThreaded`].
    pub const fn empty(integer_width: BlasIntegerWidth) -> Self {
        Self {
            layout_mask: 0,
            transform_mask: 0,
            integer_width,
            minimum_alignment: 1,
            threading: BlasThreading::SingleThreaded,
            ulp: [[None; OPERATION_COUNT]; 4],
        }
    }

    /// Adds a supported dense layout.
    pub const fn with_layout(mut self, layout: BlasLayout) -> Self {
        self.layout_mask |= 1 << layout_index(layout);
        self
    }

    /// Adds a supported operand transform.
    ///
    /// Vector operands also need a declared transform: a plain vector requires
    /// [`BlasTransform::None`] and a conjugated complex vector requires
    /// [`BlasTransform::Conjugate`].
    pub const fn with_transform(mut self, transform: BlasTransform) -> Self {
        self.transform_mask |= 1 << transform_index(transform);
        self
    }

    /// Declares a routine for one scalar type, with its accuracy bound.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if BLAS does not define `operation` for `scalar`
    ///   (see [`BlasOperation::supports_scalar`]).
    pub fn with_operation(
        mut self,
        scalar: ScalarKind,
        operation: BlasOperation,
        ulp: BlasUlp,
    ) -> Result<Self, scioxide_core::SciOxideError> {
        if !operation.supports_scalar(scalar) {
            return Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("operation"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "BLAS routine does not support this scalar representation",
            ));
        }
        self.ulp[scalar_index(scalar) as usize][operation.index()] = Some(ulp);
        Ok(self)
    }

    /// Sets the required address alignment of operand slices, in bytes.
    ///
    /// The value must be a nonzero power of two. Otherwise every call that consults
    /// the provider fails with `InvalidInput`, whatever the fallback policy.
    pub const fn with_minimum_alignment(mut self, alignment: usize) -> Self {
        self.minimum_alignment = alignment;
        self
    }

    /// Sets how the provider uses the `workers` count.
    pub const fn with_threading(mut self, threading: BlasThreading) -> Self {
        self.threading = threading;
        self
    }

    /// Returns whether a dense layout is declared.
    pub const fn supports_layout(&self, layout: BlasLayout) -> bool {
        self.layout_mask & (1 << layout_index(layout)) != 0
    }

    /// Returns whether an operand transform is declared.
    pub const fn supports_transform(&self, transform: BlasTransform) -> bool {
        self.transform_mask & (1 << transform_index(transform)) != 0
    }

    /// Returns the declared accuracy bound, or `None` if the routine is not declared for
    /// `scalar`.
    pub const fn operation_ulp(
        &self,
        scalar: ScalarKind,
        operation: BlasOperation,
    ) -> Option<BlasUlp> {
        self.ulp[scalar_index(scalar) as usize][operation.index()]
    }

    /// Returns the integer width.
    pub const fn integer_width(&self) -> BlasIntegerWidth {
        self.integer_width
    }

    /// Returns the required address alignment in bytes.
    ///
    /// The default is one.
    pub const fn minimum_alignment(&self) -> usize {
        self.minimum_alignment
    }

    /// Returns how the provider uses the `workers` count.
    pub const fn threading(&self) -> BlasThreading {
        self.threading
    }
}

const fn scalar_index(scalar: ScalarKind) -> u8 {
    match scalar {
        ScalarKind::F32 => 0,
        ScalarKind::F64 => 1,
        ScalarKind::Complex32 => 2,
        ScalarKind::Complex64 => 3,
    }
}

const fn layout_index(layout: BlasLayout) -> u8 {
    match layout {
        BlasLayout::RowMajor => 0,
        BlasLayout::ColumnMajor => 1,
    }
}

const fn transform_index(transform: BlasTransform) -> u8 {
    match transform {
        BlasTransform::None => 0,
        BlasTransform::Transpose => 1,
        BlasTransform::Conjugate => 2,
        BlasTransform::Adjoint => 3,
    }
}

/// What SciOxide does with a call that the provider cannot take.
///
/// A call is not eligible when its routine, layout, transform, alignment or integer width
/// is not declared in [`BlasCapabilities`], when an operand has no direct BLAS encoding
/// and is not copied, or when [`Accuracy::Exact`](scioxide_core::Accuracy::Exact) is
/// requested and the routine's [`BlasUlp`] bound is not zero. Modified Givens calls
/// requiring extended-range intermediates are also ineligible.
///
/// A dense matrix view has a direct encoding when, in a declared layout, it is a stored
/// matrix with a leading dimension or the transpose of one, and a conjugated complex view
/// is the transpose of one (passed as [`BlasTransform::Adjoint`]), because CBLAS has no
/// conjugate-only matrix argument. Level-3 routines (GEMM, SYRK, HERK, SYMM, HEMM, SYR2K,
/// HER2K, TRMM and TRSM) copy any other matrix view into the provider-packing scratch of
/// their workspace request and are not eligible only when that scratch is too small. GEMV
/// never copies its matrix, which therefore needs a direct encoding, and no routine copies
/// an input vector view: each needs a nonzero stride and, when it is a conjugated complex
/// view, a declared [`BlasTransform::Conjugate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum BlasFallbackPolicy {
    /// Runs SciOxide's own kernel instead.
    Reference,
    /// Returns a `CapabilityUnavailable` error.
    Error,
}

/// A BLAS provider together with the policy for calls it cannot take.
///
/// Attach it to a context with
/// [`LinalgExecutionContext::with_blas`](crate::LinalgExecutionContext::with_blas). Dense kernels
/// that run with that context offer each eligible call to the provider. The others follow the
/// [`BlasFallbackPolicy`].
#[derive(Clone, Copy)]
#[must_use]
pub struct BlasDispatch<'a> {
    provider: &'a dyn BlasProvider,
    fallback: BlasFallbackPolicy,
}

impl core::fmt::Debug for BlasDispatch<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlasDispatch")
            .field("provider", &self.provider.name())
            .field("fallback", &self.fallback)
            .finish()
    }
}

impl<'a> BlasDispatch<'a> {
    /// Creates a dispatch for `provider` with the given fallback policy.
    pub const fn new(provider: &'a dyn BlasProvider, fallback: BlasFallbackPolicy) -> Self {
        Self { provider, fallback }
    }

    pub(crate) fn capabilities(self) -> Result<BlasCapabilities, scioxide_core::SciOxideError> {
        let capabilities = self.provider.capabilities();
        if !capabilities.minimum_alignment.is_power_of_two() {
            return Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("minimum_alignment"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "BLAS minimum alignment must be a non-zero power of two",
            ));
        }
        Ok(capabilities)
    }

    /// Returns the provider.
    pub const fn provider(self) -> &'a dyn BlasProvider {
        self.provider
    }

    /// Returns the fallback policy.
    pub const fn fallback(self) -> BlasFallbackPolicy {
        self.fallback
    }
}

/// Read-only strided vector operand passed to a provider.
///
/// Element `i` is `values()[i * stride()]`, for `i < len()`.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlasVector<'a, S> {
    values: &'a [S],
    len: usize,
    stride: usize,
    conjugated: bool,
}

impl<'a, S> BlasVector<'a, S> {
    pub(crate) const fn new(values: &'a [S], len: usize, stride: usize, conjugated: bool) -> Self {
        Self {
            values,
            len,
            stride,
            conjugated,
        }
    }

    /// Returns the storage that holds every element.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
    /// Returns the number of elements.
    pub const fn len(self) -> usize {
        self.len
    }
    /// Returns whether the vector has no elements.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
    /// Returns the distance between consecutive elements, in scalars.
    pub const fn stride(self) -> usize {
        self.stride
    }
    /// Returns whether the provider must use `conj(xᵢ)` instead of the stored values.
    ///
    /// Only complex vectors are passed conjugated, and only if the provider declares
    /// [`BlasTransform::Conjugate`].
    pub const fn is_conjugated(self) -> bool {
        self.conjugated
    }
}

/// Mutable strided vector operand passed to a provider.
///
/// Element `i` is `values()[i * stride()]`, for `i < len()`. Output vectors point to SciOxide
/// scratch, not to the user's output.
#[derive(Debug)]
#[must_use]
pub struct BlasVectorMut<'a, S> {
    values: &'a mut [S],
    len: usize,
    stride: usize,
}

impl<'a, S> BlasVectorMut<'a, S> {
    pub(crate) const fn new(values: &'a mut [S], len: usize, stride: usize) -> Self {
        Self {
            values,
            len,
            stride,
        }
    }

    /// Returns the storage.
    pub fn values(&self) -> &[S] {
        self.values
    }

    /// Returns the storage that holds every element, for writing.
    pub fn values_mut(&mut self) -> &mut [S] {
        self.values
    }
    /// Returns the number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Returns whether the vector has no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Returns the distance between consecutive elements, in scalars.
    pub const fn stride(&self) -> usize {
        self.stride
    }
}

/// Read-only dense matrix operand passed to a provider.
///
/// The dimensions describe the stored matrix. The [`BlasCall`] carries the transform
/// applied to it. Entry `(i, j)` is at `i * leading_dimension() + j` in row-major layout and
/// at `j * leading_dimension() + i` in column-major layout.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlasMatrix<'a, S> {
    values: &'a [S],
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    layout: BlasLayout,
}

impl<'a, S> BlasMatrix<'a, S> {
    pub(crate) const fn new(
        values: &'a [S],
        rows: usize,
        cols: usize,
        leading_dimension: usize,
        layout: BlasLayout,
    ) -> Self {
        Self {
            values,
            rows,
            cols,
            leading_dimension,
            layout,
        }
    }

    /// Returns the storage that holds every entry.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
    /// Returns the number of stored rows, before the transform.
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of stored columns, before the transform.
    pub const fn cols(self) -> usize {
        self.cols
    }
    /// Returns the leading dimension, in scalars.
    pub const fn leading_dimension(self) -> usize {
        self.leading_dimension
    }
    /// Returns the storage order.
    pub const fn layout(self) -> BlasLayout {
        self.layout
    }
}

/// Mutable dense matrix operand passed to a provider.
///
/// Indexing is as for [`BlasMatrix`]. Output matrices point to SciOxide scratch, not to the
/// user's output.
#[derive(Debug)]
#[must_use]
pub struct BlasMatrixMut<'a, S> {
    values: &'a mut [S],
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    layout: BlasLayout,
}

impl<'a, S> BlasMatrixMut<'a, S> {
    pub(crate) const fn new(
        values: &'a mut [S],
        rows: usize,
        cols: usize,
        leading_dimension: usize,
        layout: BlasLayout,
    ) -> Self {
        Self {
            values,
            rows,
            cols,
            leading_dimension,
            layout,
        }
    }

    /// Returns the storage.
    pub fn values(&self) -> &[S] {
        self.values
    }

    /// Returns the storage that holds every entry, for writing.
    pub fn values_mut(&mut self) -> &mut [S] {
        self.values
    }
    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Returns the leading dimension, in scalars.
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
    /// Returns the storage order.
    pub const fn layout(&self) -> BlasLayout {
        self.layout
    }
}

/// One BLAS call, with operands already checked against the provider's capabilities.
///
/// Each variant names the routine in its summary and carries the routine's arguments.
/// `α` and `β` are the scale factors of the BLAS definition. Matrix transforms belong to the
/// call, not to the operands. A `triangle` names the stored triangle that the routine reads or
/// updates. All matrices of one call have the same layout. Norms, sums and rotation operands use
/// the real type `S::Real`. Mutable operands point to SciOxide scratch.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::BlasCall;
/// fn inspect(call: &BlasCall<'_, f64>) {
///     if let BlasCall::Gemm { a, b, c, .. } = call {
///         // Every matrix of one call has the same layout.
///         assert_eq!(a.layout(), c.layout());
///         assert_eq!(b.layout(), c.layout());
///     }
/// }
/// ```
#[derive(Debug)]
#[must_use]
pub enum BlasCall<'a, S: scioxide_core::NumericScalar> {
    /// `y := x` (COPY).
    Copy {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `x := α x` (SCAL).
    Scale {
        /// Scale factor `α`.
        alpha: S,
        /// Vector `x`, scaled in place.
        x: BlasVectorMut<'a, S>,
    },
    /// `y := α x + y` (AXPY).
    Axpy {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Vector updated in place.
        y: BlasVectorMut<'a, S>,
    },
    /// `result := Σ xᵢ yᵢ` (DOT, DOTU).
    Dotu {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Output scalar.
        result: &'a mut S,
    },
    /// `result := Σ conj(xᵢ) yᵢ` (DOTC).
    Dotc {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Output scalar.
        result: &'a mut S,
    },
    /// `result := ‖x‖₂` (NRM2).
    Norm2 {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Output scalar.
        result: &'a mut S::Real,
    },
    /// `result := Σ (|Re xᵢ| + |Im xᵢ|)` (ASUM).
    Asum {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Output scalar.
        result: &'a mut S::Real,
    },
    /// Writes to `result` the zero-based index of the first maximum of `|Re xᵢ| + |Im xᵢ|`
    /// (IAMAX).
    Iamax {
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Output scalar.
        result: &'a mut usize,
    },
    /// `y := α op(A) x + β y` (GEMV).
    Gemv {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `C := α op(A) op(B) + β C` (GEMM).
    Gemm {
        /// Transform `op` applied to `a`.
        left_transform: BlasTransform,
        /// Transform `op` applied to `b`.
        right_transform: BlasTransform,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Matrix `B`.
        b: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `C := α op(A) op(A)ᵀ + β C` on one triangle of `C` (SYRK).
    Syrk {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `C := α op(A) op(A)ᴴ + β C` on one triangle of `C`, with real `α`, `β` (HERK).
    Herk {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S::Real,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S::Real,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `B := α op(A) B` or `B := α B op(A)` with triangular `A` (TRMM).
    Trmm {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Side on which `A` multiplies.
        side: MatrixSide,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Whether the diagonal of `A` is stored or implicitly one.
        diagonal: DiagonalKind,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Right-hand matrix, overwritten by the product.
        b: BlasMatrixMut<'a, S>,
    },
    /// Solves `op(A) X = α B` or `X op(A) = α B` for triangular `A`; `X` overwrites `B` (TRSM).
    Trsm {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Side on which `A` multiplies.
        side: MatrixSide,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Whether the diagonal of `A` is stored or implicitly one.
        diagonal: DiagonalKind,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Right-hand side, overwritten by the solution.
        b: BlasMatrixMut<'a, S>,
    },
    /// Exchanges `x` and `y` (SWAP).
    Swap {
        /// Vector `x`, updated in place.
        x: BlasVectorMut<'a, S>,
        /// Vector `y`, updated in place.
        y: BlasVectorMut<'a, S>,
    },
    /// `[xᵢ; yᵢ] := [c s; −s c] [xᵢ; yᵢ]` for every `i` (ROT).
    Rot {
        /// Vector `x`, updated in place.
        x: BlasVectorMut<'a, S::Real>,
        /// Vector `y`, updated in place.
        y: BlasVectorMut<'a, S::Real>,
        /// Cosine `c`.
        cosine: S::Real,
        /// Sine `s`.
        sine: S::Real,
    },
    /// Applies a modified rotation `H` to every pair `[xᵢ; yᵢ]` (ROTM).
    Rotm {
        /// Vector `x`, updated in place.
        x: BlasVectorMut<'a, S::Real>,
        /// Vector `y`, updated in place.
        y: BlasVectorMut<'a, S::Real>,
        /// BLAS parameter array `[flag, h11, h21, h12, h22]` of the modified rotation.
        parameters: [S::Real; 5],
    },
    /// Generates `c`, `s` with `[c s; −s c] [a; b] = [r; 0]` (ROTG).
    Rotg {
        /// Coordinate `a`, overwritten by `r`.
        a: &'a mut S::Real,
        /// Coordinate `b`, overwritten by `z`.
        b: &'a mut S::Real,
        /// Cosine `c`.
        cosine: &'a mut S::Real,
        /// Sine `s`.
        sine: &'a mut S::Real,
    },
    /// Generates a modified rotation for the weighted pair `(√d1 x1, √d2 y1)` (ROTMG).
    Rotmg {
        /// Weight `d1`, updated in place.
        d1: &'a mut S::Real,
        /// Weight `d2`, updated in place.
        d2: &'a mut S::Real,
        /// Coordinate `x1`, updated in place.
        x1: &'a mut S::Real,
        /// Coordinate `y1`.
        y1: S::Real,
        /// BLAS parameter array `[flag, h11, h21, h12, h22]` of the modified rotation.
        parameters: &'a mut [S::Real; 5],
    },
    /// `result := Σ xᵢ yᵢ`, accumulated and returned in `f64` (DSDOT).
    WideDot {
        /// Input vector `x`.
        x: BlasVector<'a, f32>,
        /// Input vector `y`.
        y: BlasVector<'a, f32>,
        /// Output scalar.
        result: &'a mut f64,
    },
    /// `result := offset + Σ xᵢ yᵢ`, accumulated in `f64` and rounded to `f32` (SDSDOT).
    OffsetDot {
        /// Input vector `x`.
        x: BlasVector<'a, f32>,
        /// Input vector `y`.
        y: BlasVector<'a, f32>,
        /// Value added to the dot product.
        offset: f32,
        /// Output scalar.
        result: &'a mut f32,
    },
    /// `y := α op(A) x + β y` with band `A` (GBMV).
    Gbmv {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Scale factor `α`.
        alpha: S,
        /// Band matrix `A`.
        a: BlasBandMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with symmetric `A` (SYMV).
    Symv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with Hermitian `A` (HEMV).
    Hemv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with symmetric band `A` (SBMV).
    Sbmv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Band matrix `A`.
        a: BlasBandMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with Hermitian band `A` (HBMV).
    Hbmv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Band matrix `A`.
        a: BlasBandMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with symmetric packed `A` (SPMV).
    Spmv {
        /// Scale factor `α`.
        alpha: S,
        /// Packed matrix `A`.
        a: BlasPackedMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// `y := α A x + β y` with Hermitian packed `A` (HPMV).
    Hpmv {
        /// Scale factor `α`.
        alpha: S,
        /// Packed matrix `A`.
        a: BlasPackedMatrix<'a, S>,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output vector `y`.
        y: BlasVectorMut<'a, S>,
    },
    /// Solves `op(A) x = b` for triangular `A`; `x` overwrites `b` (TRSV).
    Trsv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Whether the diagonal of `A` is stored or implicitly one.
        diagonal: DiagonalKind,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Right-hand side, overwritten by the solution.
        x: BlasVectorMut<'a, S>,
    },
    /// Solves `op(A) x = b` for triangular band `A`; `x` overwrites `b` (TBSV).
    Tbsv {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Whether the diagonal of `A` is stored or implicitly one.
        diagonal: DiagonalKind,
        /// Band matrix `A`.
        a: BlasBandMatrix<'a, S>,
        /// Right-hand side, overwritten by the solution.
        x: BlasVectorMut<'a, S>,
    },
    /// Solves `op(A) x = b` for triangular packed `A`; `x` overwrites `b` (TPSV).
    Tpsv {
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Whether the diagonal of `A` is stored or implicitly one.
        diagonal: DiagonalKind,
        /// Packed matrix `A`.
        a: BlasPackedMatrix<'a, S>,
        /// Right-hand side, overwritten by the solution.
        x: BlasVectorMut<'a, S>,
    },
    /// `A := α x yᵀ + A` (GER, GERU).
    Geru {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x yᴴ + A` (GERC).
    Gerc {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x xᵀ + A` on one triangle of `A` (SYR).
    Syr {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x yᵀ + α y xᵀ + A` on one triangle of `A` (SYR2).
    Syr2 {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x xᴴ + A` with real `α`, on one triangle of `A` (HER).
    Her {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S::Real,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x yᴴ + conj(α) y xᴴ + A` on one triangle of `A` (HER2).
    Her2 {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Matrix `A`, updated in place.
        a: BlasMatrixMut<'a, S>,
    },
    /// `A := α x xᵀ + A` with symmetric packed `A` (SPR).
    Spr {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Packed matrix `A`, updated in place.
        a: BlasPackedMatrixMut<'a, S>,
    },
    /// `A := α x yᵀ + α y xᵀ + A` with symmetric packed `A` (SPR2).
    Spr2 {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Packed matrix `A`, updated in place.
        a: BlasPackedMatrixMut<'a, S>,
    },
    /// `A := α x xᴴ + A` with Hermitian packed `A` and real `α` (HPR).
    Hpr {
        /// Scale factor `α`.
        alpha: S::Real,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Packed matrix `A`, updated in place.
        a: BlasPackedMatrixMut<'a, S>,
    },
    /// `A := α x yᴴ + conj(α) y xᴴ + A` with Hermitian packed `A` (HPR2).
    Hpr2 {
        /// Scale factor `α`.
        alpha: S,
        /// Input vector `x`.
        x: BlasVector<'a, S>,
        /// Input vector `y`.
        y: BlasVector<'a, S>,
        /// Packed matrix `A`, updated in place.
        a: BlasPackedMatrixMut<'a, S>,
    },
    /// `C := α A B + β C` or `C := α B A + β C` with symmetric `A` (SYMM).
    Symm {
        /// Side on which `A` multiplies.
        side: MatrixSide,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Matrix `B`.
        b: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `C := α A B + β C` or `C := α B A + β C` with Hermitian `A` (HEMM).
    Hemm {
        /// Side on which `A` multiplies.
        side: MatrixSide,
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Matrix `B`.
        b: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `C := α op(A) op(B)ᵀ + α op(B) op(A)ᵀ + β C` on one triangle of `C` (SYR2K).
    Syr2k {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Matrix `B`.
        b: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
    /// `C := α op(A) op(B)ᴴ + conj(α) op(B) op(A)ᴴ + β C` on one triangle of `C` (HER2K).
    ///
    /// `β` is real.
    Her2k {
        /// Triangle of the matrix that is read or updated.
        triangle: Triangle,
        /// Transform `op` applied to `a`.
        transform: BlasTransform,
        /// Scale factor `α`.
        alpha: S,
        /// Matrix `A`.
        a: BlasMatrix<'a, S>,
        /// Matrix `B`.
        b: BlasMatrix<'a, S>,
        /// Scale factor `β` of the prior output.
        beta: S::Real,
        /// Output matrix `C`.
        c: BlasMatrixMut<'a, S>,
    },
}

impl<S: scioxide_core::NumericScalar> BlasCall<'_, S> {
    /// Returns the routine of this call.
    pub const fn operation(&self) -> BlasOperation {
        match self {
            Self::Copy { .. } => BlasOperation::Copy,
            Self::Scale { .. } => BlasOperation::Scale,
            Self::Axpy { .. } => BlasOperation::Axpy,
            Self::Dotu { .. } => BlasOperation::Dotu,
            Self::Dotc { .. } => BlasOperation::Dotc,
            Self::Norm2 { .. } => BlasOperation::Norm2,
            Self::Asum { .. } => BlasOperation::Asum,
            Self::Iamax { .. } => BlasOperation::Iamax,
            Self::Gemv { .. } => BlasOperation::Gemv,
            Self::Gemm { .. } => BlasOperation::Gemm,
            Self::Syrk { .. } => BlasOperation::Syrk,
            Self::Herk { .. } => BlasOperation::Herk,
            Self::Trmm { .. } => BlasOperation::Trmm,
            Self::Trsm { .. } => BlasOperation::Trsm,
            Self::Swap { .. } => BlasOperation::Swap,
            Self::Rot { .. } => BlasOperation::Rot,
            Self::Rotm { .. } => BlasOperation::Rotm,
            Self::Rotg { .. } => BlasOperation::Rotg,
            Self::Rotmg { .. } => BlasOperation::Rotmg,
            Self::WideDot { .. } => BlasOperation::WideDot,
            Self::OffsetDot { .. } => BlasOperation::OffsetDot,
            Self::Gbmv { .. } => BlasOperation::Gbmv,
            Self::Symv { .. } => BlasOperation::Symv,
            Self::Hemv { .. } => BlasOperation::Hemv,
            Self::Sbmv { .. } => BlasOperation::Sbmv,
            Self::Hbmv { .. } => BlasOperation::Hbmv,
            Self::Spmv { .. } => BlasOperation::Spmv,
            Self::Hpmv { .. } => BlasOperation::Hpmv,
            Self::Trsv { .. } => BlasOperation::Trsv,
            Self::Tbsv { .. } => BlasOperation::Tbsv,
            Self::Tpsv { .. } => BlasOperation::Tpsv,
            Self::Geru { .. } => BlasOperation::Geru,
            Self::Gerc { .. } => BlasOperation::Gerc,
            Self::Syr { .. } => BlasOperation::Syr,
            Self::Syr2 { .. } => BlasOperation::Syr2,
            Self::Her { .. } => BlasOperation::Her,
            Self::Her2 { .. } => BlasOperation::Her2,
            Self::Spr { .. } => BlasOperation::Spr,
            Self::Spr2 { .. } => BlasOperation::Spr2,
            Self::Hpr { .. } => BlasOperation::Hpr,
            Self::Hpr2 { .. } => BlasOperation::Hpr2,
            Self::Symm { .. } => BlasOperation::Symm,
            Self::Hemm { .. } => BlasOperation::Hemm,
            Self::Syr2k { .. } => BlasOperation::Syr2k,
            Self::Her2k { .. } => BlasOperation::Her2k,
        }
    }
}

/// Read-only band matrix operand in BLAS band storage.
///
/// Storage is column-major: column `j` starts at `j * leading_dimension()` and holds the
/// entries of rows `j − upper()` through `j + lower()`, with entry `(i, j)` at
/// `j * leading_dimension() + upper() + i − j`. For triangular and symmetric band calls,
/// the [`BlasCall`] names the stored triangle.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlasBandMatrix<'a, S> {
    values: &'a [S],
    rows: usize,
    cols: usize,
    lower: usize,
    upper: usize,
    leading_dimension: usize,
}

impl<'a, S> BlasBandMatrix<'a, S> {
    pub(crate) const fn new(
        values: &'a [S],
        rows: usize,
        cols: usize,
        lower: usize,
        upper: usize,
        leading_dimension: usize,
    ) -> Self {
        Self {
            values,
            rows,
            cols,
            lower,
            upper,
            leading_dimension,
        }
    }
    /// Returns the column-major storage, including unused positions.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
    /// Returns the number of rows.
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(self) -> usize {
        self.cols
    }
    /// Returns the number of subdiagonals `kl`.
    pub const fn lower(self) -> usize {
        self.lower
    }
    /// Returns the number of superdiagonals `ku`.
    pub const fn upper(self) -> usize {
        self.upper
    }
    /// Returns the distance between columns, in scalars.
    pub const fn leading_dimension(self) -> usize {
        self.leading_dimension
    }
}

/// Read-only packed triangular matrix operand in BLAS packed storage.
///
/// The stored triangle is packed column by column. For the upper triangle, entry `(i, j)` with
/// `i ≤ j` is at `i + j (j + 1) / 2`. For the lower triangle, entry `(i, j)` with `i ≥ j` is
/// at `i + j (2n − j − 1) / 2`, where `n` is the order.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlasPackedMatrix<'a, S> {
    values: &'a [S],
    order: usize,
    triangle: Triangle,
}

impl<'a, S> BlasPackedMatrix<'a, S> {
    pub(crate) const fn new(values: &'a [S], order: usize, triangle: Triangle) -> Self {
        Self {
            values,
            order,
            triangle,
        }
    }
    /// Returns the packed storage.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
    /// Returns the order `n`.
    pub const fn order(self) -> usize {
        self.order
    }
    /// Returns the stored triangle.
    pub const fn triangle(self) -> Triangle {
        self.triangle
    }
}

/// Mutable packed triangular matrix operand.
///
/// Storage is as for [`BlasPackedMatrix`]. It points to SciOxide scratch, not to the user's
/// output.
#[derive(Debug)]
#[must_use]
pub struct BlasPackedMatrixMut<'a, S> {
    values: &'a mut [S],
    order: usize,
    triangle: Triangle,
}

impl<'a, S> BlasPackedMatrixMut<'a, S> {
    pub(crate) const fn new(values: &'a mut [S], order: usize, triangle: Triangle) -> Self {
        Self {
            values,
            order,
            triangle,
        }
    }
    /// Returns the packed storage.
    pub fn values(&self) -> &[S] {
        self.values
    }
    /// Returns the packed storage, for writing.
    pub fn values_mut(&mut self) -> &mut [S] {
        self.values
    }
    /// Returns the order `n`.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the stored triangle.
    pub const fn triangle(&self) -> Triangle {
        self.triangle
    }
}

/// Physical matrix encoding used while preparing structured calls.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub(crate) enum BlasStructuredMatrix<'a, S> {
    /// Dense matrix with a selected triangle when the call uses one.
    Dense {
        /// Dense matrix data and layout.
        matrix: BlasMatrix<'a, S>,
        /// Selected triangle.
        triangle: Triangle,
    },
    /// General or structured band data.
    Band {
        matrix: BlasBandMatrix<'a, S>,
        triangle: Option<Triangle>,
    },
    /// Packed triangular or self-adjoint data.
    Packed(BlasPackedMatrix<'a, S>),
}

/// Mutable storage used while preparing structured rank updates.
#[derive(Debug)]
#[must_use]
pub(crate) enum BlasStructuredMatrixMut<'a, S> {
    /// Dense mutable candidate.
    Dense(BlasMatrixMut<'a, S>),
    /// Packed mutable candidate.
    Packed(BlasPackedMatrixMut<'a, S>),
}

/// Adapter that runs SciOxide's dense kernels on an external BLAS library.
///
/// Implement this trait around a CBLAS-compatible library, wrap it in a [`BlasDispatch`] and
/// attach it with
/// [`LinalgExecutionContext::with_blas`](crate::LinalgExecutionContext::with_blas). An
/// implementation:
///
/// - returns a fixed name from `name`, used in error messages;
/// - declares in `capabilities` the layouts, transforms, integer width, alignment,
///   threading and, for each scalar type, the routines it supports with their [`BlasUlp`]
///   bounds;
/// - in each `execute_*` method, matches the [`BlasCall`] variant, passes its operands to the
///   routine named in the variant, and returns `Err` if the library reports a failure.
///
/// SciOxide sends only declared routines whose operands fit the declaration. It passes
/// initialized, bounds-checked slices and no raw pointers, so the implementation can stay safe
/// even when it calls foreign code. Calls with an empty extent never reach the provider.
///
/// `workers` is the number of threads the provider may use. It is one under
/// [`ExecutionPolicy::Sequential`](scioxide_core::ExecutionPolicy::Sequential) and at most the
/// requested count under `Parallel(n)`. It is always one for
/// [`BlasThreading::SingleThreaded`].
///
/// SciOxide checks each result before it writes to the user's output:
///
/// - a NaN or infinity in an output entry is a `NonFinite` error with callback origin;
/// - a negative norm or sum, an `Iamax` index out of range, a Hermitian diagonal with nonzero
///   imaginary part, or an invalid modified-rotation flag is a `CapabilityFailure`;
/// - an `Err` returned by the provider is a `CapabilityFailure` that carries its detail.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations --features blas` for a complete caller-owned storage example.
pub trait BlasProvider: Sync {
    /// Returns the provider name used in error messages.
    fn name(&self) -> &'static str;
    /// Returns the supported routines and calling conventions.
    fn capabilities(&self) -> BlasCapabilities;
    /// Executes an `f32` call.
    fn execute_f32(
        &self,
        call: BlasCall<'_, f32>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError>;
    /// Executes an `f64` call.
    fn execute_f64(
        &self,
        call: BlasCall<'_, f64>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError>;
    /// Executes a `Complex32` call.
    fn execute_complex32(
        &self,
        call: BlasCall<'_, Complex32>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError>;
    /// Executes a `Complex64` call.
    fn execute_complex64(
        &self,
        call: BlasCall<'_, Complex64>,
        workers: NonZeroUsize,
    ) -> Result<(), CapabilityError>;
}
