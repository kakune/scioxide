//! Plans for products, composition, Taylor shifts and division of monomial
//! coefficient slices.
use crate::PolynomialExecutionOptions;
use crate::util::{bytes_of, invalid, minimum_len, try_grow, validate_slice};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::AlgebraicScalar;
use scioxide_core::ScaledValue;
use scioxide_core::backend::{checked_add, standard, try_filled, try_with_capacity};
use scioxide_core::{
    Capacity, Complex, ExecutionContext, ExecutionPolicy, InvalidInputReason, NumericScalar,
    RealScalar, SciOxideError,
};
use scioxide_fft::{
    FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec, Normalization,
};

// The direct kernel retains componentwise accuracy. These power-of-two gates
// also define the first prepared transform size; tune with scioxide-compare.
const REAL_CROSSOVER: usize = 32;
const COMPLEX_CROSSOVER: usize = 16;
const fn crossover<C: NumericScalar>() -> usize {
    if C::IS_REAL {
        REAL_CROSSOVER
    } else {
        COMPLEX_CROSSOVER
    }
}
mod layout;
pub use layout::PolynomialOperationRequest;

#[derive(Clone)]
struct ProductTransform<R: RealScalar> {
    len: usize,
    forward: FftPlan<R>,
    inverse: FftPlan<R>,
    complex_scratch: usize,
    real_scratch: usize,
}

pub(crate) struct ProductScratch<'a, R: RealScalar> {
    pub(crate) complex: &'a mut [Complex<R>],
    pub(crate) real: &'a mut [R],
}

/// Names the algorithm that [`PolynomialOperationPlan::multiply_into`] used for
/// a product.
///
/// The choice depends on the operand lengths and coefficient magnitudes; see
/// [`PolynomialOperationPlan::multiply_into`].
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::PolynomialProductAlgorithm;
/// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
///
/// // 2 · 3 = 6; short operands use the direct sum.
/// let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
///     left_len: 1,
///     right_len: 1,
/// }])?;
/// let mut ws = plan.workspace(plan.requests())?;
/// let mut out = [0.0_f64; 1];
/// let cx = ExecutionContext::sequential();
/// let algorithm = plan.multiply_into(&[2.0_f64], &[3.0_f64], &mut out, &mut ws, cx)?;
/// assert_eq!(algorithm, PolynomialProductAlgorithm::Schoolbook);
/// assert!((out[0] - 6.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialProductAlgorithm {
    /// Direct convolution `cₖ = Σ_{i+j=k} aᵢ bⱼ`, `O(m n)` operations for
    /// operands of lengths `m` and `n`. A coefficient whose terms come close to
    /// overflow or underflow is recomputed in extended-exponent arithmetic, so
    /// each coefficient has a small componentwise relative error.
    Schoolbook,
    /// Convolution by fast Fourier transform of length `N`, the smallest power
    /// of two at least `m + n − 1`, in `O(N log N)` operations. The operands
    /// are scaled by powers of two before the transform. The error is
    /// normwise, of order `ε‖a‖‖b‖` in each coefficient, where `ε` is the
    /// machine epsilon.
    Fft,
}

/// Multiplies, composes, shifts and divides polynomials given by ascending
/// monomial coefficients, for operand lengths fixed in advance.
///
/// A coefficient slice `[c₀, c₁, …, cₙ]` represents `c₀ + c₁x + … + cₙxⁿ`.
/// The plan is built from a list of [`PolynomialOperationRequest`] shapes. It
/// precomputes the FFT plans of the transform lengths those shapes can use, with
/// twiddle tables linear in each length, and the storage each shape needs; the
/// storage itself lives in a [`PolynomialWorkspace`]. A call with a
/// shape that was not prepared is an error. The plan never removes
/// coefficients or changes the basis.
///
/// | Operation | Method |
/// |---|---|
/// | [`Self::multiply_into`] | direct convolution, or FFT convolution for long operands with a narrow range of magnitudes |
/// | [`Self::compose_into`] | divide-and-conquer on the outer polynomial with precomputed powers `q^(2^k)` |
/// | [`Self::taylor_shift_into`] | composition with `x + s` |
/// | [`Self::div_rem_into`] | long division, with error diagnostics |
///
/// Real coefficients use real-input transforms and complex coefficients use
/// complex transforms. See [Plans and workspaces](scioxide_core#plans-and-workspaces)
/// for the shared workspace rules.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
///
/// // (1 + 2x)(1 − x) = 1 + x − 2x².
/// let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
///     left_len: 2,
///     right_len: 2,
/// }])?;
/// let mut ws = plan.workspace(plan.requests())?;
/// let mut product = [0.0_f64; 3];
/// let (a, b) = ([1.0_f64, 2.0], [1.0_f64, -1.0]);
/// let cx = ExecutionContext::sequential();
/// plan.multiply_into(&a, &b, &mut product, &mut ws, cx)?;
/// let want = [1.0_f64, 1.0, -2.0];
/// assert!(product.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-13));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct PolynomialOperationPlan<C: NumericScalar> {
    requests: Vec<PolynomialOperationRequest>,
    transforms: Vec<ProductTransform<C::Real>>,
    requirements: Vec<PolynomialWorkspaceRequirements<C>>,
    options: PolynomialExecutionOptions,
}

/// Reports the buffer lengths a [`PolynomialWorkspace`] needs for a set of
/// prepared shapes.
///
/// Operations run one at a time, so the requirement of several shapes is the
/// largest requirement of each buffer, not the sum. Division needs `O(n)`
/// storage and no FFT buffers; a product needs FFT buffers only when its
/// shape can use the FFT path. Composition and Taylor shifts also count the
/// stored powers of the inner polynomial and the recursion stack. There is no
/// memory limit.
///
/// ```
/// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
///
/// let request = PolynomialOperationRequest::Divide {
///     dividend_len: 8,
///     divisor_len: 8,
/// };
/// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
/// let requirements = plan.workspace_requirements(plan.requests())?;
/// assert!(requirements.coefficient_len() >= 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialWorkspaceRequirements<C: NumericScalar> {
    coefficients: usize,
    fft: usize,
    real: usize,
    scaled: usize,
    scalar: PhantomData<fn() -> C>,
}

impl<C: NumericScalar> PolynomialWorkspaceRequirements<C> {
    /// Returns the number of extended-exponent values used by division and
    /// composition. Division retains operands and error diagnostics;
    /// composition reserves two output-sized buffers for range recovery.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // Division of lengths 8 and 8 needs 8 + 8 values.
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// assert_eq!(
    ///     plan.workspace_requirements(plan.requests())?.scaled_len(),
    ///     16
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn scaled_len(self) -> usize {
        self.scaled
    }
    /// Returns the number of coefficient values, including the composition
    /// stack.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// assert!(requirements.coefficient_len() >= 8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn coefficient_len(self) -> usize {
        self.coefficients
    }
    /// Returns the number of complex values used by FFT products.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // Division uses no transforms.
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// assert_eq!(requirements.fft_len(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn fft_len(self) -> usize {
        self.fft
    }
    /// Returns the number of real values used by FFT products of real
    /// coefficients; zero for complex coefficients.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// assert_eq!(requirements.real_len(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn real_len(self) -> usize {
        self.real
    }
    /// Returns the total size in bytes of the required buffers, without the
    /// `Vec` headers.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// let coefficient_bytes = requirements.coefficient_len() * size_of::<f64>();
    /// assert!(requirements.required_bytes()? >= coefficient_bytes);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("workspace_bytes");
        checked_add(
            checked_add(
                bytes_of::<C>(self.coefficients)?,
                bytes_of::<ScaledValue<C>>(self.scaled)?,
                quantity,
            )?,
            checked_add(
                bytes_of::<Complex<C::Real>>(self.fft)?,
                bytes_of::<C::Real>(self.real)?,
                quantity,
            )?,
            quantity,
        )
    }
}

/// Holds the scratch storage of a [`PolynomialOperationPlan`].
///
/// Create one with [`PolynomialOperationPlan::workspace`], or size an empty
/// workspace with [`Self::try_reserve`]. A workspace serves any plan and shape
/// whose requirements it meets, so one workspace reserved for the largest
/// shape serves smaller ones without growing. A call with a workspace that is
/// too small returns `DimensionMismatch`. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_poly::PolynomialOperationRequest::Divide;
/// use scioxide_poly::{PolynomialOperationPlan, PolynomialWorkspace};
/// let large = Divide { dividend_len: 128, divisor_len: 128 };
/// let small = Divide { dividend_len: 8, divisor_len: 8 };
/// let (large_plan, small_plan) = (
///     PolynomialOperationPlan::<f64>::new(&[large])?,
///     PolynomialOperationPlan::<f64>::new(&[small])?,
/// );
/// let need_large = large_plan.workspace_requirements(large_plan.requests())?;
/// let need_small = small_plan.workspace_requirements(small_plan.requests())?;
/// let mut workspace = PolynomialWorkspace::new();
/// assert_eq!(workspace.check(&need_small)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&need_large)?;
/// assert_eq!(workspace.check(&need_small)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialWorkspace<C: NumericScalar> {
    pub(crate) coefficients: Vec<C>,
    pub(crate) fft: Vec<Complex<C::Real>>,
    pub(crate) real: Vec<C::Real>,
    pub(crate) scaled: Vec<ScaledValue<C>>,
}

impl<C: NumericScalar> Default for PolynomialWorkspace<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: NumericScalar> PolynomialWorkspace<C> {
    /// Returns an empty workspace without allocating.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest, PolynomialWorkspace};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// let ws = PolynomialWorkspace::<f64>::new();
    /// assert_eq!(ws.check(&requirements)?, Capacity::NeedsGrowth);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            fft: Vec::new(),
            real: Vec::new(),
            scaled: Vec::new(),
        }
    }
    /// Reports whether the workspace meets `requirements`, without allocating.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the `check` method of other SciOxide workspaces.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// let ws = plan.workspace(plan.requests())?;
    /// assert_eq!(ws.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(
        &self,
        requirements: &PolynomialWorkspaceRequirements<C>,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.fits(requirements).is_ok() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace until it meets `requirements`. Buffers that are
    /// already large enough are left as they are.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if a buffer cannot grow. Buffers grown before the
    ///   failure keep their new size.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest, PolynomialWorkspace};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 8,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// let mut ws = PolynomialWorkspace::<f64>::new();
    /// ws.try_reserve(&requirements)?;
    /// assert_eq!(ws.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &PolynomialWorkspaceRequirements<C>,
    ) -> Result<(), SciOxideError> {
        try_grow(
            &mut self.coefficients,
            requirements.coefficients,
            C::ZERO,
            scioxide_core::quantity!("coefficient_scratch"),
        )?;
        try_grow(
            &mut self.fft,
            requirements.fft,
            Complex::new(C::Real::ZERO, C::Real::ZERO),
            scioxide_core::quantity!("fft_scratch"),
        )?;
        try_grow(
            &mut self.scaled,
            requirements.scaled,
            ScaledValue::ZERO,
            scioxide_core::quantity!("scaled_polynomial_scratch"),
        )?;
        try_grow(
            &mut self.real,
            requirements.real,
            C::Real::ZERO,
            scioxide_core::quantity!("real_scratch"),
        )
    }
    fn fits(&self, requirements: &PolynomialWorkspaceRequirements<C>) -> Result<(), SciOxideError> {
        minimum_len(requirements.coefficients, self.coefficients.len())?;
        minimum_len(requirements.fft, self.fft.len())?;
        minimum_len(requirements.scaled, self.scaled.len())?;
        minimum_len(requirements.real, self.real.len())
    }
}

impl<C: NumericScalar> PolynomialOperationPlan<C> {
    /// Prepares a plan for the given shapes with sequential execution.
    ///
    /// Duplicate shapes are prepared once. The plan builds the FFT plans that
    /// the declared shapes can use, whatever the coefficient values turn out to
    /// be. Use [`Self::build`] to set execution options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `requests` is empty or a request has a zero length.
    /// - `SizeOverflow` if a buffer length or byte count does not fit in `usize`.
    /// - `AllocationFailure` if the plan cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let requests = [PolynomialOperationRequest::Multiply {
    ///     left_len: 8,
    ///     right_len: 8,
    /// }];
    /// let plan = PolynomialOperationPlan::<f64>::new(&requests)?;
    /// assert_eq!(plan.requests(), requests);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(requests: &[PolynomialOperationRequest]) -> Result<Self, SciOxideError> {
        Self::build(
            requests,
            PolynomialExecutionOptions::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Prepares a plan for the given shapes with execution options.
    ///
    /// The options set how many workers the FFTs may use; `Auto` lets each
    /// transform decide by its length, up to the worker bound. `execution` is
    /// used only while the plan is built and is not stored; each call takes
    /// its own [`ExecutionContext`].
    ///
    /// # Errors
    ///
    /// - As [`Self::new`].
    /// - `InvalidInput` if the worker bound in `options` exceeds 64.
    /// - `ExecutionStopped` on cancellation or deadline during preparation.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let requests = [PolynomialOperationRequest::TaylorShift { len: 8 }];
    /// let plan = PolynomialOperationPlan::<f64>::build(
    ///     &requests,
    ///     Default::default(),
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(plan.requests(), requests);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn build(
        requests: &[PolynomialOperationRequest],
        options: PolynomialExecutionOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        execution.check()?;
        options.validate()?;
        validate_requests(requests)?;
        let mut retained = try_with_capacity(
            requests.len(),
            scioxide_core::quantity!("operation_requests"),
        )?;
        let mut requirements = try_with_capacity(
            requests.len(),
            scioxide_core::quantity!("operation_requirements"),
        )?;
        let mut layouts = try_with_capacity(
            requests.len(),
            scioxide_core::quantity!("operation_layouts"),
        )?;
        let mut sizes = 0_usize;
        for &request in requests {
            if retained
                .iter()
                .any(|prepared: &PolynomialOperationRequest| {
                    prepared.canonical() == request.canonical()
                })
            {
                continue;
            }
            execution.check()?;
            let layout = layout::analyze::<C>(request)?;
            let required = PolynomialWorkspaceRequirements {
                coefficients: layout.coefficients,
                scaled: layout.scaled,
                fft: 0,
                real: 0,
                scalar: PhantomData,
            };
            required.required_bytes()?;
            retained.push(request);
            requirements.push(required);
            layouts.push(layout);
            sizes |= layout.transforms;
        }
        let mut transforms = try_with_capacity(
            sizes.count_ones() as usize,
            scioxide_core::quantity!("fft_plans"),
        )?;
        // Resolve Auto's worker request so FFT dispatch cannot exceed the bound.
        let policy = match options.execution_policy() {
            ExecutionPolicy::Auto => ExecutionPolicy::Parallel(options.worker_bound()),
            _ => options.resolved_policy(),
        };
        if sizes != 0 {
            let mut cache =
                scioxide_fft::FftPlanCache::new(scioxide_fft::FftCachePolicy::new(1, usize::MAX)?)?;
            for power in 0..usize::BITS {
                let n = 1_usize << power;
                if sizes & n == 0 {
                    continue;
                }
                execution.check()?;
                let spec = |direction| {
                    FftPlanSpec::new(
                        n,
                        direction,
                        Normalization::Inverse,
                        if C::IS_REAL {
                            FftLayout::Real
                        } else {
                            FftLayout::Complex
                        },
                    )
                    .with_execution_policy(policy)
                };
                let forward = cache.plan(spec(FftDirection::Forward), execution)?;
                let inverse = cache.plan(spec(FftDirection::Inverse), execution)?;
                let bins = if C::IS_REAL { n / 2 + 1 } else { n };
                let complex_scratch = checked_add(
                    scioxide_core::backend::checked_mul(
                        bins,
                        2,
                        scioxide_core::quantity!("fft_scratch"),
                    )?,
                    forward
                        .workspace_requirements(())?
                        .complex_len()
                        .max(inverse.workspace_requirements(())?.complex_len()),
                    scioxide_core::quantity!("fft_scratch"),
                )?;
                transforms.push(ProductTransform {
                    len: n,
                    forward,
                    inverse,
                    complex_scratch,
                    real_scratch: if C::IS_REAL { n } else { 0 },
                });
            }
        }
        for (required, layout) in requirements.iter_mut().zip(layouts) {
            for transform in &transforms {
                if layout.transforms & transform.len != 0 {
                    required.fft = required.fft.max(transform.complex_scratch);
                    required.real = required.real.max(transform.real_scratch);
                }
            }
            required.required_bytes()?;
        }
        execution.check()?;
        Ok(Self {
            requests: retained,
            transforms,
            requirements,
            options,
        })
    }

    /// Returns the execution options of the plan.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::TaylorShift { len: 8 };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// assert_eq!(plan.options(), Default::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn options(&self) -> PolynomialExecutionOptions {
        self.options
    }

    /// Returns the prepared shapes in the order they were first given, without
    /// duplicates.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::TaylorShift { len: 8 };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request, request])?;
    /// assert_eq!(plan.requests(), &[request]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn requests(&self) -> &[PolynomialOperationRequest] {
        &self.requests
    }

    /// Returns the workspace requirements of a nonempty subset of the prepared
    /// shapes.
    ///
    /// The requirement is the largest of the individual requirements, buffer by
    /// buffer, because operations run one at a time.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `requests` is empty or contains a shape that was not
    ///   prepared.
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 8,
    ///     divisor_len: 3,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// assert_eq!(plan.workspace_requirements(&[request])?.fft_len(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        requests: &[PolynomialOperationRequest],
    ) -> Result<PolynomialWorkspaceRequirements<C>, SciOxideError> {
        validate_requests(requests)?;
        let mut required = PolynomialWorkspaceRequirements {
            coefficients: 0,
            fft: 0,
            real: 0,
            scaled: 0,
            scalar: PhantomData,
        };
        for &request in requests {
            let next = self.requirement(request)?;
            required.coefficients = required.coefficients.max(next.coefficients);
            required.fft = required.fft.max(next.fft);
            required.real = required.real.max(next.real);
            required.scaled = required.scaled.max(next.scaled);
        }
        required.required_bytes()?;
        Ok(required)
    }

    /// Allocates a workspace for a nonempty subset of the prepared shapes.
    ///
    /// # Errors
    ///
    /// - As [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be allocated.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// let request = PolynomialOperationRequest::TaylorShift { len: 8 };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let workspace = plan.workspace(plan.requests())?;
    /// let requirements = plan.workspace_requirements(plan.requests())?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        &self,
        requests: &[PolynomialOperationRequest],
    ) -> Result<PolynomialWorkspace<C>, SciOxideError> {
        let mut workspace = PolynomialWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(requests)?)?;
        Ok(workspace)
    }

    #[inline(always)]
    fn requirement(
        &self,
        request: PolynomialOperationRequest,
    ) -> Result<&PolynomialWorkspaceRequirements<C>, SciOxideError> {
        self.requests
            .iter()
            .position(|prepared| prepared.canonical() == request.canonical())
            .map(|index| &self.requirements[index])
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("request"),
                    InvalidInputReason::Inconsistent,
                    "operation shape was not prepared",
                )
            })
    }

    #[inline(always)]
    pub(crate) fn admit(
        &self,
        workspace: &PolynomialWorkspace<C>,
        request: PolynomialOperationRequest,
    ) -> Result<&PolynomialWorkspaceRequirements<C>, SciOxideError> {
        let required = self.requirement(request)?;
        workspace.fits(required)?;
        Ok(required)
    }
    /// Computes the product `c = a·b` of two coefficient slices and writes it to
    /// `output`.
    ///
    /// `output` must have `a.len() + b.len() − 1` entries. The shape
    /// `Multiply { left_len: a.len(), right_len: b.len() }` must have been
    /// prepared; a prepared shape also serves the swapped operand order.
    ///
    /// The product uses FFT convolution when the shorter operand has at least
    /// 32 coefficients (16 for complex coefficients) and, in each operand,
    /// every nonzero real and imaginary part is at least `√ε` times the
    /// largest one. Otherwise it uses the direct sum, whose componentwise error
    /// stays small for coefficients of widely varying size. The returned
    /// [`PolynomialProductAlgorithm`] says which was used and describes the
    /// accuracy of each.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `a` or `b` is empty, or the shape was not prepared.
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace` is
    ///   too small.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a product coefficient overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // (1 + x)(1 − x) = 1 − x².
    /// let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
    ///     left_len: 2,
    ///     right_len: 2,
    /// }])?;
    /// let mut ws = plan.workspace(plan.requests())?;
    /// let mut out = [0.0_f64; 3];
    /// let cx = ExecutionContext::sequential();
    /// plan.multiply_into(&[1.0_f64, 1.0], &[1.0_f64, -1.0], &mut out, &mut ws, cx)?;
    /// let want = [1.0_f64, 0.0, -1.0];
    /// assert!(out.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-13));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn multiply_into(
        &self,
        a: &[C],
        b: &[C],
        output: &mut [C],
        workspace: &mut PolynomialWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<PolynomialProductAlgorithm, SciOxideError> {
        // The product is formed in workspace scratch and copied out only after every
        // fallible step (including the trailing cancellation check) has succeeded, so
        // `output` is never partially overwritten.
        context.check()?;
        nonempty(a, scioxide_core::argument!("a"))?;
        nonempty(b, scioxide_core::argument!("b"))?;
        let expected = self
            .admit(
                workspace,
                PolynomialOperationRequest::Multiply {
                    left_len: a.len(),
                    right_len: b.len(),
                },
            )?
            .coefficients;
        if output.len() != expected {
            return Err(SciOxideError::dimension_mismatch(expected, output.len()));
        }
        let result = &mut workspace.coefficients[..output.len()];
        let algorithm = self.product(
            a,
            b,
            result,
            &mut ProductScratch {
                complex: &mut workspace.fft,
                real: &mut workspace.real,
            },
            context,
        )?;
        // Re-poll cancellation before publishing the result.
        context.check()?;
        output.copy_from_slice(result);
        Ok(algorithm)
    }

    /// Multiplies `a` and `b` into `out` with the plan's transforms. A shape
    /// mismatch is reported against an empty operand `a` or `b`, or otherwise
    /// against `output`.
    pub(crate) fn product(
        &self,
        a: &[C],
        b: &[C],
        out: &mut [C],
        scratch: &mut ProductScratch<'_, C::Real>,
        context: ExecutionContext<'_>,
    ) -> Result<PolynomialProductAlgorithm, SciOxideError> {
        nonempty(a, scioxide_core::argument!("a"))?;
        nonempty(b, scioxide_core::argument!("b"))?;
        let len = checked_add(
            a.len(),
            b.len(),
            scioxide_core::quantity!("product_coefficients"),
        )? - 1;
        if out.len() != len {
            return Err(SciOxideError::dimension_mismatch(len, out.len()));
        }
        validate_slice(a)?;
        validate_slice(b)?;
        // The FFT error is normwise (about eps * ||a|| * ||b|| per output
        // coefficient), so it is only used when every nonzero coefficient is within
        // sqrt(eps) of its operand's largest one; wider dynamic ranges fall back to
        // the componentwise-accurate schoolbook sum.
        if a.len().min(b.len()) >= crossover::<C>() && suitable(a) && suitable(b) {
            let n = scioxide_core::backend::checked_next_power_of_two(
                out.len(),
                scioxide_core::quantity!("transform_length"),
            )?;
            let index = self
                .transforms
                .binary_search_by_key(&n, |transform| transform.len)
                .map_err(|_| {
                    invalid(
                        scioxide_core::argument!("request"),
                        InvalidInputReason::Inconsistent,
                        "transform size was not prepared",
                    )
                })?;
            let ProductTransform {
                forward, inverse, ..
            } = &self.transforms[index];
            context.check()?;
            let sa = standard::exponent(scale(a))?;
            let sb = standard::exponent(scale(b))?;
            let bins = if C::IS_REAL { n / 2 + 1 } else { n };
            let (left, rest) = scratch.complex.split_at_mut(bins);
            let (right, fft_work) = rest.split_at_mut(bins);
            if C::IS_REAL {
                let samples = &mut scratch.real[..n];
                for (operand, exponent, spectrum) in [(a, sa, &mut *left), (b, sb, &mut *right)] {
                    samples.fill(C::Real::ZERO);
                    for (sample, coefficient) in samples.iter_mut().zip(operand) {
                        *sample = coefficient.real_component().scalbn(-exponent);
                    }
                    forward.execute_into(
                        &*samples,
                        FftOutput::hermitian(n, spectrum),
                        fft_work,
                        context,
                    )?;
                }
                for (x, y) in left.iter_mut().zip(right.iter()) {
                    *x *= *y;
                }
                inverse.execute_into(
                    FftInput::hermitian(n, left),
                    &mut *samples,
                    fft_work,
                    context,
                )?;
                for (o, &x) in out.iter_mut().zip(samples.iter()) {
                    *o = standard::restore(C::from_real(x), sa + sb)?;
                }
            } else {
                for (operand, exponent, spectrum) in [(a, sa, &mut *left), (b, sb, &mut *right)] {
                    spectrum.fill(Complex::new(C::Real::ZERO, C::Real::ZERO));
                    for (sample, &coefficient) in spectrum.iter_mut().zip(operand) {
                        let value = standard::scale(coefficient, -exponent);
                        *sample = Complex::new(value.real_component(), value.imaginary_component());
                    }
                    forward.execute_in_place(spectrum, fft_work, context)?;
                }
                for (x, y) in left.iter_mut().zip(right.iter()) {
                    *x *= *y;
                }
                inverse.execute_in_place(left, fft_work, context)?;
                for (o, x) in out.iter_mut().zip(left.iter()) {
                    *o = standard::restore(C::from_components(x.re, x.im), sa + sb)?;
                }
            }
            return Ok(PolynomialProductAlgorithm::Fft);
        }
        context.check()?;
        schoolbook(a, b, out, context)?;
        Ok(PolynomialProductAlgorithm::Schoolbook)
    }
}
/// Rejects an empty coefficient operand before its shape is looked up.
pub(crate) fn nonempty<C>(
    operand: &[C],
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if operand.is_empty() {
        return Err(crate::util::empty_error(argument));
    }
    Ok(())
}

fn validate_requests(requests: &[PolynomialOperationRequest]) -> Result<(), SciOxideError> {
    if requests.is_empty() {
        return Err(invalid(
            scioxide_core::argument!("requests"),
            InvalidInputReason::Empty,
            "at least one operation shape is required",
        ));
    }
    Ok(())
}
fn scale<C: NumericScalar>(a: &[C]) -> C::Real {
    a.iter().fold(C::Real::ZERO, |s, x| {
        s.max(x.real_component().abs())
            .max(x.imaginary_component().abs())
    })
}
fn suitable<C: NumericScalar>(a: &[C]) -> bool {
    let s = scale(a);
    s > C::Real::ZERO
        && a.iter().all(|x| {
            [x.real_component(), x.imaginary_component()]
                .iter()
                // Exact zeros are harmless; only tiny nonzero coefficients would lose their
                // relative accuracy in the normwise FFT error.
                .all(|v| *v == C::Real::ZERO || v.abs() / s >= C::Real::SQRT_EPSILON)
        })
}
fn schoolbook<C: NumericScalar>(
    a: &[C],
    b: &[C],
    out: &mut [C],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    for (k, result) in out.iter_mut().enumerate() {
        context.check()?;
        *result = match native_coefficient::<C, 4, true, true>(
            k,
            a.len(),
            b.len(),
            |i| Some(a[i]),
            |i| Some(b[i]),
        ) {
            Some(value) => value,
            None => match framed_coefficient(k, a, b)? {
                Some(value) => value,
                None => convolution_coefficient(
                    k,
                    a.len(),
                    b.len(),
                    |i| ScaledValue::from_value(a[i]),
                    |i| ScaledValue::from_value(b[i]),
                )?
                .round()?,
            },
        };
    }
    Ok(())
}
/// Returns the product of two nonempty coefficient slices in a new vector.
pub(crate) fn multiply<C: NumericScalar>(a: &[C], b: &[C]) -> Result<Vec<C>, SciOxideError> {
    let n = checked_add(
        a.len(),
        b.len(),
        scioxide_core::quantity!("product_coefficients"),
    )? - 1;
    let mut result = try_filled(n, C::ZERO, scioxide_core::quantity!("product_coefficients"))?;
    // Mirrors the selection in `product`, so a one-shot plan is only built when
    // the FFT path will actually be taken.
    if a.len().min(b.len()) < crossover::<C>() || !suitable(a) || !suitable(b) {
        schoolbook(a, b, &mut result, ExecutionContext::sequential_native())?;
    } else {
        let request = PolynomialOperationRequest::Multiply {
            left_len: a.len(),
            right_len: b.len(),
        };
        let plan = PolynomialOperationPlan::new(&[request])?;
        let mut workspace = plan.workspace(&[request])?;
        plan.multiply_into(
            a,
            b,
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
    }
    Ok(result)
}

/// Computes one product coefficient in working precision, or `None` when a
/// term comes close to overflow or underflow and the whole coefficient must be
/// recomputed in extended range. Four independent partial sums shorten the
/// dependency chain without changing the componentwise error. Negligible tiny
/// terms are dropped only for working-precision output; root-tree products keep
/// their sequential sum and extended retry.
#[inline]
fn native_coefficient<
    C: NumericScalar,
    const LANES: usize,
    const CHECK_RANGE: bool,
    const FILTER_TINY: bool,
>(
    k: usize,
    left_len: usize,
    right_len: usize,
    left: impl Fn(usize) -> Option<C>,
    right: impl Fn(usize) -> Option<C>,
) -> Option<C> {
    let mut sums = [C::ZERO; LANES];
    let mut skipped = 0_usize;
    let mut skipped_exponent = i32::MIN;
    let small = C::Real::MIN_POSITIVE.sqrt();
    let low = standard::exponent(C::Real::MIN_POSITIVE).ok()?;
    for (lane, i) in (k.saturating_sub(right_len - 1)..left_len.min(k + 1)).enumerate() {
        let a = left(i)?;
        let b = right(k - i)?;
        if FILTER_TINY && C::IS_REAL {
            let (a, b) = (a.real_component(), b.real_component());
            if a == C::Real::ZERO || b == C::Real::ZERO {
                continue;
            }
            // Screen before multiplication: hardware subnormal products are
            // expensive even when they cannot affect an ordinary coefficient.
            if a.abs() < small || b.abs() < small {
                let exponent =
                    standard::exponent(a.abs()).ok()? + standard::exponent(b.abs()).ok()?;
                if exponent <= low {
                    skipped += 1;
                    skipped_exponent = skipped_exponent.max(exponent);
                    continue;
                }
            }
        }
        let product = a * b;
        if CHECK_RANGE && standard::product_needs_retry(a, b, product) {
            return None;
        }
        sums[lane % LANES] = sums[lane % LANES] + product;
    }
    let sum = sums.into_iter().fold(C::ZERO, |sum, lane| sum + lane);
    if !sum.has_finite_components() {
        return None;
    }
    if skipped > 0 {
        // Each omitted product is smaller than 2^skipped_exponent. Compare
        // their combined upper bound with epsilon*abs(sum) in exponent space;
        // the bound itself may be below the subnormal grid. Cancellation must
        // not hide a significant omitted term.
        let value = sum.real_component().abs();
        if value == C::Real::ZERO {
            return None;
        }
        let growth = usize::BITS - (skipped - 1).leading_zeros();
        if skipped_exponent + growth as i32
            > standard::exponent(value).ok()? - C::Real::MANTISSA_DIGITS as i32
        {
            return None;
        }
    }
    Some(sum)
}

/// Computes a real coefficient in a local binary frame. Each factor is
/// normalized before multiplication, only normal terms are summed, and the sum
/// is rounded once at the end. Terms below the frame's precision are omitted
/// only with an explicit bound; cancellation that makes that bound significant
/// falls back to the extended retry.
fn framed_coefficient<C: NumericScalar>(
    k: usize,
    a: &[C],
    b: &[C],
) -> Result<Option<C>, SciOxideError> {
    if !C::IS_REAL {
        return Ok(None);
    }
    let indices = k.saturating_sub(b.len() - 1)..a.len().min(k + 1);
    let mut frame = i32::MIN;
    for i in indices.clone() {
        let (a, b) = (a[i].real_component(), b[k - i].real_component());
        if a != C::Real::ZERO && b != C::Real::ZERO {
            frame = frame.max(standard::exponent(a.abs())? + standard::exponent(b.abs())?);
        }
    }
    if frame == i32::MIN {
        return Ok(Some(C::ZERO));
    }
    let growth = usize::BITS - indices.len().max(1).leading_zeros();
    let cutoff = -((2 * C::Real::MANTISSA_DIGITS + growth) as i32);
    let mut sums = [C::Real::ZERO; 4];
    let mut skipped = 0;
    for (lane, i) in indices.enumerate() {
        let (a, b) = (a[i].real_component(), b[k - i].real_component());
        if a == C::Real::ZERO || b == C::Real::ZERO {
            continue;
        }
        let ae = standard::exponent(a.abs())?;
        let be = standard::exponent(b.abs())?;
        let shift = ae + be - frame;
        if shift < cutoff {
            skipped += 1;
        } else {
            let product = a.scalbn(-ae) * b.scalbn(-be);
            sums[lane % 4] += product.scalbn(shift);
        }
    }
    let sum = sums.into_iter().fold(C::Real::ZERO, |sum, lane| sum + lane);
    let omitted = C::Real::ONE.scalbn(cutoff) * C::Real::try_from_usize_exact(skipped)?;
    if omitted > C::Real::EPSILON * sum.abs() {
        return Ok(None);
    }
    let result = C::from_real(sum.scalbn(frame));
    Ok(result.has_finite_components().then_some(result))
}

/// Computes coefficient `k` of the same convolution in extended range.
pub(crate) fn convolution_coefficient<C: NumericScalar>(
    k: usize,
    left_len: usize,
    right_len: usize,
    left: impl Fn(usize) -> Result<ScaledValue<C>, SciOxideError>,
    right: impl Fn(usize) -> Result<ScaledValue<C>, SciOxideError>,
) -> Result<ScaledValue<C>, SciOxideError> {
    if C::IS_REAL {
        let mut sum = scioxide_core::Scaled::ZERO;
        for i in k.saturating_sub(right_len - 1)..left_len.min(k + 1) {
            sum = sum.add(left(i)?.real().mul(right(k - i)?.real())?)?;
        }
        return Ok(ScaledValue::from_components(
            sum,
            scioxide_core::Scaled::ZERO,
        ));
    }
    let mut sum = ScaledValue::ZERO;
    // Only index pairs `i + j = k` with `i < left_len` and `j < right_len`
    // contribute; accumulation is in `ScaledValue` so neither partial products
    // nor sums can overflow before the final rounding.
    for i in k.saturating_sub(right_len - 1)..left_len.min(k + 1) {
        sum = sum.add(left(i)?.mul(right(k - i)?)?)?;
    }
    Ok(sum)
}

/// Multiplies root-tree factors whose coefficients may lie outside the
/// floating-point range. Exact working-precision copies are cached once per
/// operand so the ordinary coefficient kernel runs wherever possible. A rounded
/// subnormal is never used as an operand.
pub(crate) fn multiply_scaled<C: NumericScalar>(
    left: &[ScaledValue<C>],
    right: &[ScaledValue<C>],
) -> Result<Vec<ScaledValue<C>>, SciOxideError> {
    let quantity = scioxide_core::quantity!("scaled_polynomial_product");
    let length = checked_add(left.len(), right.len(), quantity)? - 1;
    if let Some((sa, sb)) = product_frame(left, right)? {
        let project = |values: &[ScaledValue<C>], exponent: i64| {
            let component = |part: scioxide_core::Scaled<C::Real>| {
                if part.is_zero() {
                    part.significand()
                } else {
                    part.significand()
                        .scalbn((part.exponent() - exponent) as i32)
                }
            };
            crate::util::try_collect(
                values.iter().map(|value| {
                    Ok(C::from_components(
                        component(value.real()),
                        component(value.imaginary()),
                    ))
                }),
                quantity,
            )
        };
        let a = project(left, sa)?;
        let b = project(right, sb)?;
        let mut product = try_filled(length, ScaledValue::ZERO, quantity)?;
        for (k, coefficient) in product.iter_mut().enumerate() {
            *coefficient = match native_coefficient::<C, 1, false, false>(
                k,
                a.len(),
                b.len(),
                |i| Some(a[i]),
                |i| Some(b[i]),
            ) {
                Some(value) => ScaledValue::from_parts(value, sa + sb)?,
                None => convolution_coefficient(
                    k,
                    left.len(),
                    right.len(),
                    |i| Ok(left[i]),
                    |i| Ok(right[i]),
                )?,
            };
        }
        return Ok(product);
    }
    let native = |values: &[ScaledValue<C>]| {
        crate::util::try_collect(
            values.iter().map(|&scaled| {
                Ok(match scaled.round() {
                    Ok(value) if ScaledValue::from_value(value)? == scaled => Some(value),
                    _ => None,
                })
            }),
            quantity,
        )
    };
    let a = native(left)?;
    let b = native(right)?;
    let mut product = try_filled(length, ScaledValue::ZERO, quantity)?;
    for (k, coefficient) in product.iter_mut().enumerate() {
        // Keep the root tree's coefficient order: its clustered cancellation
        // fixtures are sensitive to changing the reduction topology.
        *coefficient = match native_coefficient::<C, 1, true, false>(
            k,
            left.len(),
            right.len(),
            |i| a[i],
            |i| b[i],
        ) {
            Some(value) => ScaledValue::from_value(value)?,
            None => convolution_coefficient(
                k,
                left.len(),
                right.len(),
                |i| Ok(left[i]),
                |i| Ok(right[i]),
            )?,
        };
    }
    Ok(product)
}

/// Center both operands in a binary frame when every nonzero component product
/// is normal and even the absolute coefficient sums have finite headroom.
/// The extended coefficients are never rounded in the unit frame.
fn product_frame<C: NumericScalar>(
    left: &[ScaledValue<C>],
    right: &[ScaledValue<C>],
) -> Result<Option<(i64, i64)>, SciOxideError> {
    let bounds = |values: &[ScaledValue<C>]| {
        values
            .iter()
            .flat_map(|value| [value.real(), value.imaginary()])
            .filter(|value| !value.is_zero())
            .fold(None, |bounds: Option<(i64, i64)>, value| {
                Some(match bounds {
                    None => (value.exponent(), value.exponent()),
                    Some((lo, hi)) => (lo.min(value.exponent()), hi.max(value.exponent())),
                })
            })
            .unwrap_or((0, 0))
    };
    let (alo, ahi) = bounds(left);
    let (blo, bhi) = bounds(right);
    let sa = ((i128::from(alo) + i128::from(ahi)) / 2) as i64;
    let sb = ((i128::from(blo) + i128::from(bhi)) / 2) as i64;
    let low = i128::from(standard::exponent(C::Real::MIN_POSITIVE)? - 1);
    let high = i128::from(standard::exponent(C::Real::MAX)? - 1);
    let growth = (usize::BITS - left.len().min(right.len()).leading_zeros()) as i128 + 3;
    let amin = i128::from(alo) - i128::from(sa);
    let amax = i128::from(ahi) - i128::from(sa);
    let bmin = i128::from(blo) - i128::from(sb);
    let bmax = i128::from(bhi) - i128::from(sb);
    Ok((amin >= low
        && bmin >= low
        && amax < high
        && bmax < high
        && amin + bmin >= low
        && amax + bmax + growth < high
        && sa.checked_add(sb).is_some())
    .then_some((sa, sb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_frames_preserve_subnormal_sums_and_cancellation_retries() {
        let tiny = f64::from_bits(1);
        let a = [tiny, 1.0, tiny];
        let b = [1.0, tiny, 1.0];
        assert_eq!(
            native_coefficient::<f64, 4, true, true>(
                1,
                a.len(),
                b.len(),
                |i| Some(a[i]),
                |i| Some(b[i])
            ),
            Some(1.0)
        );
        assert_eq!(framed_coefficient(2, &a, &b).unwrap(), Some(3.0 * tiny));
        assert_eq!(
            framed_coefficient(1, &[tiny, tiny], &[0.5, 0.5]).unwrap(),
            Some(tiny)
        );
        assert_eq!(
            framed_coefficient(0, &[tiny], &[1.5]).unwrap(),
            Some(2.0 * tiny)
        );
        assert_eq!(framed_coefficient(0, &[tiny], &[0.5]).unwrap(), Some(0.0));
        assert_eq!(
            framed_coefficient(1, &[f64::MAX; 2], &[2.0, -2.0]).unwrap(),
            Some(0.0)
        );
        // Tiny terms cannot be discarded when the ordinary terms cancel.
        let mut out = [0.0; 5];
        schoolbook(
            &[1.0, 1.0, tiny],
            &[1.0, -1.0, 1.0],
            &mut out,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(out[2], tiny);
        assert_eq!(
            framed_coefficient(2, &[1.0, 1.0, tiny], &[1.0, -1.0, 1.0]).unwrap(),
            None
        );
        assert!(
            schoolbook(
                &[f64::MAX],
                &[2.0],
                &mut out[..1],
                ExecutionContext::sequential_native()
            )
            .is_err()
        );

        let tiny = f32::from_bits(1);
        assert_eq!(
            framed_coefficient(1, &[tiny, tiny], &[0.5, 0.5]).unwrap(),
            Some(tiny)
        );
    }

    #[test]
    fn preparation_deduplicates_only_selected_transforms_and_combines_buffer_maxima() {
        use PolynomialOperationRequest::*;
        let small = Multiply {
            left_len: 64,
            right_len: 64,
        };
        let large = Multiply {
            left_len: 257,
            right_len: 257,
        };
        let division = Divide {
            dividend_len: 1 << 20,
            divisor_len: 1 << 20,
        };
        let plan = PolynomialOperationPlan::<f64>::new(&[small, large, division, small]).unwrap();
        assert_eq!(plan.requests(), &[small, large, division]);
        assert_eq!(
            plan.transforms
                .iter()
                .map(|transform| transform.len)
                .collect::<Vec<_>>(),
            [128, 1024]
        );
        let product = plan.workspace_requirements(&[large]).unwrap();
        assert_eq!(product.coefficient_len(), 513);
        assert_eq!(product.scaled_len(), 0);
        let divided = plan.workspace_requirements(&[division]).unwrap();
        assert_eq!(divided.coefficient_len(), (1 << 20) + 1);
        assert_eq!(divided.scaled_len(), 2 << 20);
        assert_eq!(divided.fft_len(), 0);
        assert_eq!(divided.real_len(), 0);
        let combined = plan.workspace_requirements(plan.requests()).unwrap();
        assert_eq!(combined.coefficient_len(), divided.coefficient_len());
        assert_eq!(combined.scaled_len(), divided.scaled_len());
        assert_eq!(combined.fft_len(), product.fft_len());
        assert_eq!(combined.real_len(), product.real_len());
        let division_only = PolynomialOperationPlan::<f64>::new(&[division]).unwrap();
        assert!(division_only.transforms.is_empty());
        assert!(PolynomialOperationPlan::<f64>::new(&[]).is_err());
        assert!(plan.workspace_requirements(&[]).is_err());
        assert!(
            plan.workspace_requirements(&[Multiply {
                left_len: 63,
                right_len: 64
            }])
            .is_err()
        );
    }

    fn crossover_boundaries<C: NumericScalar>() {
        let gate = crossover::<C>();
        let requests = [gate - 1, gate, gate + 1].map(|n| PolynomialOperationRequest::Multiply {
            left_len: n,
            right_len: n,
        });
        let plan = PolynomialOperationPlan::<C>::new(&requests).unwrap();
        let mut work = plan.workspace(plan.requests()).unwrap();
        let expected_bytes = work.coefficients.len() * core::mem::size_of::<C>()
            + work.fft.len() * core::mem::size_of::<Complex<C::Real>>()
            + work.real.len() * core::mem::size_of::<C::Real>()
            + work.scaled.len() * core::mem::size_of::<ScaledValue<C>>();
        let requirements = plan.workspace_requirements(plan.requests()).unwrap();
        assert_eq!(requirements.required_bytes().unwrap(), expected_bytes);
        assert_eq!(
            work.check(&requirements).unwrap(),
            scioxide_core::Capacity::Sufficient
        );
        for n in [gate - 1, gate, gate + 1] {
            let a = alloc::vec![C::ONE; n];
            let mut out = alloc::vec![C::ZERO; 2 * n - 1];
            let algorithm = plan
                .multiply_into(
                    &a,
                    &a,
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(
                algorithm,
                if n < gate {
                    PolynomialProductAlgorithm::Schoolbook
                } else {
                    PolynomialProductAlgorithm::Fft
                }
            );
            for (k, value) in out.into_iter().enumerate() {
                let expected = C::from_real(
                    C::Real::try_from_usize_exact((k + 1).min(2 * n - 1 - k)).unwrap(),
                );
                assert!(
                    (value - expected).abs()
                        <= C::Real::EPSILON * C::Real::try_from_usize_exact(1024 * n).unwrap()
                );
            }
        }
    }

    #[test]
    fn prepared_crossover_shapes_and_workspace_accounting() {
        crossover_boundaries::<f32>();
        crossover_boundaries::<f64>();
        crossover_boundaries::<Complex<f32>>();
        crossover_boundaries::<Complex<f64>>();
    }

    #[test]
    fn root_products_retain_extended_inputs_in_native_frames_and_fallbacks() {
        let scaled = |value, exponent| ScaledValue::<f64>::from_parts(value, exponent).unwrap();
        let product =
            multiply_scaled(&[scaled(1., 1500), scaled(0., 0)], &[scaled(1., -1500)]).unwrap();
        assert_eq!(product, [ScaledValue::ONE, ScaledValue::ZERO]);

        let product = multiply_scaled(
            &[scaled(1., -2000), ScaledValue::ONE],
            &[scaled(1., 1000), ScaledValue::ONE],
        )
        .unwrap();
        assert_eq!(
            product,
            [scaled(1., -1000), scaled(1., 1000), ScaledValue::ONE]
        );

        // Rounding an intermediate to a nearby subnormal would lose a term
        // which a later factor amplifies back into the normal range.
        let tiny = scaled(1.25, -1074);
        let product = multiply_scaled(&[tiny], &[scaled(1., 1074)]).unwrap();
        assert_eq!(product, [scaled(1.25, 0)]);
    }
}
