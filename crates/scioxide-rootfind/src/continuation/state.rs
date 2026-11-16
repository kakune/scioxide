//! The state of one continuation path.
use super::*;

/// Holds the progress of one continuation path: the current point, its unit
/// tangent and the history of accepted points.
///
/// Create it with [`Continuation::state`] and pass it to
/// [`Continuation::follow`] or to every [`Continuation::advance`] call of the
/// same continuation. The history has room for `max_points` points, reserved
/// when the state is created. [`Self::checkpoint`] copies the state so that
/// [`Continuation::resume`] can restore it, and [`Self::reset`] starts a new
/// path in the same storage.
#[derive(Debug)]
pub struct ContinuationState<R: RealScalar> {
    pub(crate) path: ContinuationPath<R>,
    pub(crate) options: ContinuationOptions<R>,
    pub(crate) point: Vec<R>,
    pub(crate) residual: Vec<R>,
    pub(crate) report: Option<scioxide_core::ResidualReport<R>>,
    pub(crate) tangent: Vec<R>,
    pub(crate) points: Vec<R>,
    pub(crate) history: Vec<ContinuationObservation<R>>,
    pub(crate) history_token: u128,
    pub(crate) next_target: usize,
    pub(crate) distance: R,
    pub(crate) next_step: R,
    pub(crate) status: RootTermination,
    pub(crate) metadata: Option<SolveCheckpointMetadata>,
    pub(crate) provider_shape: ProviderShape,
}
impl<R: RealScalar> ContinuationState<R> {
    /// Reserves the history and copies the seed `(x₀, λ₀)` without evaluating `F`.
    ///
    /// # Errors
    ///
    /// Returns shape, non-finite, path, size-overflow or allocation errors.
    pub(crate) fn new(
        problem: &ContinuationProblem<'_, R>,
        path: ContinuationPath<R>,
        seed: &[R],
        options: ContinuationOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let m = checked_size(
            problem.dimension(),
            1,
            scioxide_core::quantity!("augmented_coordinates"),
        )?;
        if seed.len() != m {
            return Err(invalid(
                scioxide_core::argument!("seed"),
                InvalidInputReason::Inconsistent,
                "augmented seed shape mismatch",
            ));
        }
        finite(seed, NonFiniteOrigin::Input)?;
        match &path {
            ContinuationPath::Parameters(v) => {
                if v.is_empty() {
                    return Err(invalid(
                        scioxide_core::argument!("path"),
                        InvalidInputReason::Inconsistent,
                        "empty parameter path",
                    ));
                }
                finite(v, NonFiniteOrigin::Input)?;
            }
            ContinuationPath::PseudoArclength(s) => {
                if !s.is_finite() || *s <= R::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("path"),
                        InvalidInputReason::Inconsistent,
                        "arclength must be finite and positive",
                    ));
                }
            }
        }
        if options.dense_diagnostics && problem.diagnostic.is_some() {
            return Err(invalid(
                scioxide_core::argument!("options"),
                InvalidInputReason::Inconsistent,
                "select dense or injected diagnostics",
            ));
        }
        let capacity = m.checked_mul(options.max_points.get()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("history_points"))
        })?;
        let mut tangent =
            scioxide_core::backend::try_filled(m, R::ZERO, scioxide_core::quantity!("tangent"))?;
        tangent[m - 1] = R::ONE;
        Ok(Self {
            path,
            next_step: options.initial_step,
            points: scioxide_core::backend::try_with_capacity(
                capacity,
                scioxide_core::quantity!("history_points"),
            )?,
            history: scioxide_core::backend::try_with_capacity(
                options.max_points.get(),
                scioxide_core::quantity!("history"),
            )?,
            history_token: 0,
            options,
            point: try_copy_slice(seed, scioxide_core::quantity!("point"))?,
            residual: scioxide_core::backend::try_filled(
                problem.dimension(),
                R::ZERO,
                scioxide_core::quantity!("accepted_residual"),
            )?,
            report: None,
            tangent,
            next_target: 0,
            distance: R::ZERO,
            status: RootTermination::NotStarted,
            metadata: None,
            provider_shape: provider_shape(problem)?,
        })
    }
    /// Sets the direction in which the path starts.
    ///
    /// The first [`Continuation::advance`] computes the unit tangent `t` at
    /// the corrected seed with `tangent` as the border row, so `t` points to the
    /// same side as `tangent`. The default is `+λ`, the unit vector of the
    /// parameter.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the path has already started, `tangent` does not have
    ///   length `n + 1`, or `tangent` is zero.
    /// - `NonFinite` if `tangent` contains NaN or infinity.
    pub fn with_initial_tangent(mut self, tangent: &[R]) -> Result<Self, SciOxideError> {
        if self.metadata.is_some() || tangent.len() != self.point.len() {
            return Err(invalid(
                scioxide_core::argument!("tangent"),
                InvalidInputReason::Inconsistent,
                "initial tangent shape or lifecycle mismatch",
            ));
        }
        finite(tangent, NonFiniteOrigin::Input)?;
        if norm(tangent)? == R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("tangent"),
                InvalidInputReason::Inconsistent,
                "zero initial tangent",
            ));
        }
        self.tangent.copy_from_slice(tangent);
        Ok(self)
    }
    /// Identifies the path by its scalar type, path, options and provider shape.
    pub(crate) fn identity(&self) -> SolveIdentity {
        crate::context::identity(
            scioxide_core::algorithm_id!("rootfind", "continuation", 1),
            ProblemToken::new(0),
            self.point.len(),
            crate::context::options_token(&(
                core::any::type_name::<R>(),
                &self.path,
                &self.options,
                self.provider_shape,
            )),
        )
    }
    /// Returns the last accepted point `(x, λ)`, or the seed before the first `advance`.
    pub fn point(&self) -> &[R] {
        &self.point
    }
    /// Returns `F(x, λ)` at the last accepted point.
    ///
    /// Before the first accepted point, this is zero and
    /// [`Self::residual_report`] is `None`.
    pub fn residual(&self) -> &[R] {
        &self.residual
    }
    /// Returns the residual norm of the last accepted point and the reference
    /// used in its relative test, or `None` before the first accepted point.
    pub const fn residual_report(&self) -> Option<scioxide_core::ResidualReport<R>> {
        self.report
    }
    /// Returns `‖F(x, λ)‖₂` at the last accepted point, or `None` before the first.
    pub fn residual_norm(&self) -> Option<R> {
        self.report.map(|r| r.raw_norm())
    }
    /// Returns the number of trial steps on this path, including rejected ones.
    pub fn iterations(&self) -> usize {
        self.counts().iterations()
    }
    /// Copies the last accepted point `(x, λ)` into a strided one-dimensional output.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if no point has been accepted yet, or `output` is not
    ///   one-dimensional with length `n + 1`.
    pub fn copy_point_into(
        &self,
        mut output: scioxide_core::NdViewMut<'_, '_, R>,
    ) -> Result<(), SciOxideError> {
        if self.metadata.is_none()
            || output.layout().shape().rank() != 1
            || output.layout().shape().len() != self.point.len()
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                InvalidInputReason::Inconsistent,
                "accepted point output shape or lifecycle mismatch",
            ));
        }
        let mut source = self.point.iter();
        output.for_each_mut(|slot| {
            if let Some(&value) = source.next() {
                *slot = value;
            }
        });
        Ok(())
    }
    /// Returns the unit tangent at the last accepted point.
    ///
    /// Successive tangents point the same way along the path. Before the first
    /// accepted point this is the initial orientation.
    pub fn tangent(&self) -> &[R] {
        &self.tangent
    }
    /// Returns the accepted points in order, starting with the corrected seed.
    pub fn history(&self) -> &[ContinuationObservation<R>] {
        &self.history
    }
    /// Returns the coordinates `(x, λ)` of accepted point `index`, if it exists.
    pub fn history_point(&self, index: usize) -> Option<&[R]> {
        let m = self.point.len();
        index
            .checked_mul(m)
            .and_then(|i| self.points.get(i..i.checked_add(m)?))
    }
    /// Returns the termination status of the path.
    pub fn termination(&self) -> RootTermination {
        self.status
    }
    /// Returns whether the whole requested path has been followed (`Complete`).
    pub fn is_success(&self) -> bool {
        self.status == RootTermination::Complete
    }
    /// Returns the work done by continuation itself, as of the last completed call.
    pub fn counts(&self) -> EvaluationCounts {
        self.metadata
            .map_or(EvaluationCounts::new(), |m| m.counts().own())
    }
    /// Returns the work of continuation and its nested solves, as of the last completed call.
    pub fn accounting(&self) -> scioxide_core::SolveCounts {
        self.metadata
            .map_or(scioxide_core::SolveCounts::new(), |metadata| {
                metadata.counts()
            })
    }

    /// Starts a new path of the same continuation from `seed`, reusing this
    /// state's storage.
    ///
    /// The history and work counts are cleared. `orientation`, when given, sets the
    /// initial direction as in [`Self::with_initial_tangent`]; otherwise it is `+λ`.
    /// On error the state is unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `seed` or `orientation` does not have length `n + 1`,
    ///   or `orientation` is zero.
    /// - `NonFinite` if `seed` or `orientation` contains NaN or infinity.
    pub fn reset(&mut self, seed: &[R], orientation: Option<&[R]>) -> Result<(), SciOxideError> {
        if seed.len() != self.point.len() || orientation.is_some_and(|t| t.len() != seed.len()) {
            return Err(invalid(
                if seed.len() == self.point.len() {
                    scioxide_core::argument!("orientation")
                } else {
                    scioxide_core::argument!("seed")
                },
                InvalidInputReason::Inconsistent,
                "reset seed or orientation shape mismatch",
            ));
        }
        finite(seed, NonFiniteOrigin::Input)?;
        if let Some(t) = orientation {
            finite(t, NonFiniteOrigin::Input)?;
            if norm(t)? == R::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("orientation"),
                    InvalidInputReason::Inconsistent,
                    "zero orientation",
                ));
            }
        }
        self.point.copy_from_slice(seed);
        self.residual.fill(R::ZERO);
        self.report = None;
        self.tangent.fill(R::ZERO);
        let n = self.point.len() - 1;
        self.tangent[n] = R::ONE;
        if let Some(t) = orientation {
            self.tangent.copy_from_slice(t);
        }
        self.points.clear();
        self.history.clear();
        self.history_token = 0;
        self.next_target = 0;
        self.distance = R::ZERO;
        self.next_step = self.options.initial_step;
        self.status = RootTermination::NotStarted;
        self.metadata = None;
        Ok(())
    }

    /// Checks that `p`, `w` and `c` match this path.
    pub(crate) fn preflight(
        &self,
        p: &ContinuationProblem<'_, R>,
        w: &ContinuationWorkspace<R>,
        c: &SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let argument = if c.identity() != self.identity() {
            Some(scioxide_core::argument!("c"))
        } else if w.shape != self.provider_shape {
            Some(scioxide_core::argument!("w"))
        } else if provider_shape(p)? != self.provider_shape {
            Some(scioxide_core::argument!("p"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "context, workspace or provider identity mismatch",
            ));
        }
        if let Some(meta) = self.metadata {
            let now = c.checkpoint(self.state_token())?;
            if now.accepted() != meta.accepted() || now.counts() != meta.counts() {
                return Err(invalid(
                    scioxide_core::argument!("c"),
                    InvalidInputReason::Inconsistent,
                    "resume an explicit checkpoint or use the active context",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn append(
        &mut self,
        raw: R,
        iterations: usize,
        rejected: usize,
        turning: bool,
        mut diagnostic: Option<ContinuationDiagnostic<R>>,
    ) {
        if let (Some(previous), Some(current)) = (
            self.history.last().and_then(|o| o.diagnostic),
            diagnostic.as_mut(),
        ) {
            current.branch_point_candidate |= previous
                .bordered_sign
                .zip(current.bordered_sign)
                .is_some_and(|(a, b)| a != 0 && b != 0 && a != b);
        }
        self.points.extend_from_slice(&self.point);
        self.history.push(ContinuationObservation {
            parameter: self.point[self.point.len() - 1],
            arclength: self.distance,
            residual_norm: raw,
            corrector_iterations: iterations,
            rejected_steps: rejected,
            turning_point_candidate: turning,
            diagnostic,
        });
        self.history_token = crate::context::options_token(&(
            self.history_token,
            self.point.as_slice(),
            self.history.last(),
        ));
    }
    pub(crate) fn update_completion(&mut self) {
        self.status = RootTermination::Advanced;
        match &self.path {
            ContinuationPath::Parameters(values) => {
                while self.next_target < values.len()
                    && values[self.next_target] == self.point[self.point.len() - 1]
                {
                    self.next_target += 1;
                }
                if self.next_target == values.len() {
                    self.status = RootTermination::Complete;
                }
            }
            ContinuationPath::PseudoArclength(total) => {
                if self.distance >= *total {
                    self.status = RootTermination::Complete;
                }
            }
        }
    }
}
