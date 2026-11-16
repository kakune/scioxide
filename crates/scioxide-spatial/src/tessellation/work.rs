//! Work counting for exact predicates and per-operation limits.
use crate::predicates::fixed_upper;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::storage::Storage;
use crate::tessellation::{
    GeometryOptions, GeometryProvenance, GeometryWorkspace, GeometryWorkspaceRequirements, Result,
    sign_value, unstable,
};
use scioxide_core::backend::checked_add;
use scioxide_core::{Argument, BudgetKind, Capacity, RealScalar, SciOxideError, SolveContext};
// Stops are polled after at most this many admitted work units: predicate
// stages, canonical units and loop steps. Polling may read a clock.
pub(super) const POLL_UNITS: usize = 256;
fn poll(
    unpolled: &mut usize,
    context: &Option<&mut SolveContext<'_, '_>>,
    units: usize,
) -> Result<()> {
    *unpolled = unpolled.saturating_add(units);
    if *unpolled >= POLL_UNITS {
        *unpolled = 0;
        if let Some(context) = context {
            context.execution().check()?;
        }
    }
    Ok(())
}
// Predicate and canonical work admitted during one public operation.
pub(super) struct Work<'w, 'c, 's, 'p> {
    pub(super) workspace: &'w mut GeometryWorkspace,
    pub(super) options: GeometryOptions,
    pub(super) context: Option<&'c mut SolveContext<'s, 'p>>,
    pub(super) ties: bool,
    pub(super) canonical_work: usize,
    pub(super) storage: &'w Storage,
}

/// A facet and optional last row, without rebuilding an owned query vector.
#[derive(Clone, Copy)]
struct SiteRows<'a> {
    facet: &'a [usize],
    point: Option<usize>,
}
impl SiteRows<'_> {
    fn iter(self) -> impl Iterator<Item = usize> + Clone {
        self.facet.iter().copied().chain(self.point)
    }
    fn len(self) -> usize {
        self.facet.len() + usize::from(self.point.is_some())
    }
    fn at(self, index: usize) -> usize {
        match self.point {
            Some(point) if index == self.facet.len() => point,
            _ => self.facet[index],
        }
    }
}

impl<'w, 'c, 's, 'p> Work<'w, 'c, 's, 'p> {
    pub(super) fn new(
        workspace: &'w mut GeometryWorkspace,
        dimension: usize,
        options: GeometryOptions,
        context: Option<&'c mut SolveContext<'s, 'p>>,
        source: Argument,
        storage: &'w Storage,
    ) -> Result<Self> {
        let requirements = GeometryWorkspaceRequirements::checked(dimension, options, source)?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(SciOxideError::dimension_mismatch(
                [dimension],
                [workspace.dimension],
            ));
        }
        Self::prepared(workspace, options, context, storage)
    }
    // Preparation has already checked dimensions, limits and reserved scratch.
    pub(super) fn prepared(
        workspace: &'w mut GeometryWorkspace,
        options: GeometryOptions,
        context: Option<&'c mut SolveContext<'s, 'p>>,
        storage: &'w Storage,
    ) -> Result<Self> {
        if let Some(context) = &context {
            context.execution().check()?;
        }
        workspace.terms = 0;
        workspace.unpolled = 0;
        Ok(Self {
            workspace,
            options,
            context,
            ties: false,
            canonical_work: 0,
            storage,
        })
    }
    // One loop step; stops are polled after at most `POLL_UNITS` units.
    pub(super) fn check(&mut self) -> Result<()> {
        poll(&mut self.workspace.unpolled, &self.context, 1)
    }
    // Insertions are reported to the parent as nested iterations. Local
    // construction budgets are unbounded, so recording never denies.
    pub(super) fn iteration(&mut self) -> Result<()> {
        self.check()?;
        if let Some(context) = &mut self.context {
            let _ = context.record_iteration()?;
        }
        Ok(())
    }
    pub(super) fn raw(&mut self, count: usize, dimension: usize, lifted: bool) -> Result<i8> {
        self.determinant(count, dimension, lifted, false)
    }
    pub(super) fn admit(&mut self, count: usize) -> Result<()> {
        admit_terms(
            &mut self.workspace.terms,
            &mut self.workspace.unpolled,
            self.options.limits.max_predicate_terms,
            &mut self.context,
            count,
        )
    }
    pub(super) fn determinant(
        &mut self,
        count: usize,
        dimension: usize,
        lifted: bool,
        matrix: bool,
    ) -> Result<i8> {
        if !matrix && (dimension == 2 || dimension == 3) {
            self.admit(1)?;
            if let Some(sign) = fixed_rows::<false>(&self.workspace.rows, dimension, lifted) {
                return Ok(sign);
            }
            self.admit(fixed_upper(dimension, lifted))?;
            return Ok(
                fixed_rows::<true>(&self.workspace.rows, dimension, lifted).unwrap_or_default()
            );
        }
        let Self {
            workspace,
            options,
            context,
            ..
        } = self;
        let GeometryWorkspace {
            exact,
            rows,
            terms,
            unpolled,
            ..
        } = workspace;
        let mut admit = |count| {
            admit_terms(
                terms,
                unpolled,
                options.limits.max_predicate_terms,
                context,
                count,
            )
        };
        if matrix {
            exact.matrix_sign(&rows[..count * count], count, &mut admit)
        } else {
            exact.sign(&rows[..count * dimension], dimension, lifted, &mut admit)
        }
    }
    pub(super) fn sign<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        indices: &[usize],
        lifted: bool,
        perturb: bool,
    ) -> Result<i8> {
        self.sign_rows(
            cloud,
            SiteRows {
                facet: indices,
                point: None,
            },
            lifted,
            perturb,
        )
    }
    pub(super) fn sign_with_point<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        facet: &[usize],
        point: usize,
        lifted: bool,
        perturb: bool,
    ) -> Result<i8> {
        self.sign_rows(
            cloud,
            SiteRows {
                facet,
                point: Some(point),
            },
            lifted,
            perturb,
        )
    }
    #[inline]
    fn sign_rows<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        indices: SiteRows<'_>,
        lifted: bool,
        perturb: bool,
    ) -> Result<i8> {
        let d = cloud.dimension;
        // Projective clouds (dual halfspace constructions) store full
        // homogeneous rows. Their orientation is a plain square determinant,
        // with no implicit column of ones.
        if let Some(homogeneous) = &cloud.homogeneous {
            let weight = homogeneous[indices.at(0) * (d + 1) + d];
            let affine = (d == 2 || d == 3)
                && weight != 0.
                && indices
                    .iter()
                    .all(|i| homogeneous[i * (d + 1) + d] == weight);
            if affine {
                self.admit(1)?;
                let sign = if let Some(sign) =
                    fixed_homogeneous::<false>(homogeneous, indices, d, cloud.filter_envelope)
                {
                    sign
                } else {
                    self.admit(fixed_upper(d, false))?;
                    fixed_homogeneous::<true>(homogeneous, indices, d, cloud.filter_envelope)
                        .unwrap_or_default()
                };
                return Ok(if weight < 0. { -sign } else { sign });
            }
            for (row, i) in indices.iter().enumerate() {
                self.workspace.rows[row * (d + 1)..(row + 1) * (d + 1)]
                    .copy_from_slice(&homogeneous[i * (d + 1)..(i + 1) * (d + 1)]);
            }
            return self.determinant(d + 1, d + 1, false, true);
        }
        let sign = if d == 2 || d == 3 {
            self.admit(1)?;
            if let Some(sign) = fixed_cloud::<false, S>(cloud, indices, lifted) {
                sign
            } else {
                self.admit(fixed_upper(d, lifted))?;
                fixed_cloud::<true, S>(cloud, indices, lifted).unwrap_or_default()
            }
        } else {
            for (row, i) in indices.iter().enumerate() {
                for (axis, &value) in cloud.point(i).iter().enumerate() {
                    self.workspace.rows[row * d + axis] = value.into_f64();
                }
            }
            self.determinant(indices.len(), d, lifted, false)?
        };
        if sign != 0 || !lifted || !perturb {
            return Ok(sign);
        }
        self.symbolic_sign(cloud, indices)
    }
    #[cold]
    #[inline(never)]
    fn symbolic_sign<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        indices: SiteRows<'_>,
    ) -> Result<i8> {
        let d = cloud.dimension;
        // Both policies construct the same perturbed triangulation; Strict then
        // rejects exact cospherical groups among the published cells only.
        // Lexicographic infinitesimals in the norm column, in canonical site
        // order. Cofactors are unperturbed orientations. This is one coherent
        // regular triangulation, not independent arbitrary choices at each tie.
        // Perturbing site k's lifted entry by eps^(2^k) adds terms of the
        // form (-1)^(row + d) * eps^(2^k) * cofactor(row, d). The dominant
        // term is the first site in canonical order with a nonzero cofactor,
        // which is the orientation of the remaining points. Its sign decides
        // the tie.
        // Visit only rows of this determinant, in canonical site order.
        let mut previous = None;
        for _ in 0..indices.len() {
            let row = indices
                .iter()
                .enumerate()
                .filter(|(_, index)| previous.is_none_or(|p| *index > p))
                .min_by_key(|(_, index)| *index)
                .map(|(row, _)| row)
                .ok_or_else(|| unstable("repeated symbolic site"))?;
            previous = Some(indices.at(row));
            let mut target = 0;
            for (source, i) in indices.iter().enumerate() {
                if source == row {
                    continue;
                }
                for (axis, &value) in cloud.point(i).iter().enumerate() {
                    self.workspace.rows[target * d + axis] = value.into_f64();
                }
                target += 1;
            }
            let cofactor = self.raw(indices.len() - 1, d, false)?;
            if cofactor != 0 {
                self.ties = true;
                return Ok(if (row + d).is_multiple_of(2) {
                    cofactor
                } else {
                    -cofactor
                });
            }
        }
        Ok(0)
    }
}

fn admit_terms(
    terms: &mut usize,
    unpolled: &mut usize,
    maximum: usize,
    context: &mut Option<&mut SolveContext<'_, '_>>,
    count: usize,
) -> Result<()> {
    if count <= maximum - *terms {
        poll(unpolled, context, count)?;
        let granted = if let Some(context) = context.as_mut() {
            context.try_consume(BudgetKind::Node, count)?.is_granted()
        } else {
            true
        };
        if granted {
            *terms += count;
            return Ok(());
        }
    }
    // A denied bulk admission consumes nothing. Preserve the last admitted
    // unit in both the local counter and every ancestor before returning error.
    for _ in 0..count {
        if *terms >= maximum {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("geometry_predicate_terms"),
                checked_add(
                    *terms,
                    1,
                    scioxide_core::quantity!("geometry_predicate_terms"),
                )?,
                maximum,
            ));
        }
        poll(unpolled, context, 1)?;
        if let Some(context) = context.as_mut()
            && !context.try_consume(BudgetKind::Node, 1)?.is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node),
                scioxide_core::ConvergenceDiagnostics::new()
                    .with_counts(context.counts())
                    .with_budget(context.effective_budget_report(BudgetKind::Node)),
                "geometry predicate allowance exhausted",
            ));
        }
        *terms += 1;
    }
    Ok(())
}

// One fixed-dimension stage: the A filter, or with `EXACT` the out-of-line
// exact predicate, which always decides.
macro_rules! stage {
    ($exact:expr, $unchecked:ident, $filter:ident, $($argument:expr),+) => {
        if $exact {
            Some(sign_value(crate::predicates::$unchecked($($argument),+)))
        } else {
            crate::predicates::$filter($($argument),+).map(sign_value)
        }
    };
}

// Both primitive scalar representations widen exactly. Borrow their original
// rows in the common filter path; only the cold exact stage builds f64 rows.
#[inline]
fn fixed_cloud<const EXACT: bool, S: RealScalar>(
    cloud: &Cloud<S>,
    sites: SiteRows<'_>,
    lifted: bool,
) -> Option<i8> {
    let envelope = cloud.filter_envelope;
    if cloud.dimension == 2 {
        let points = cloud.points.as_chunks::<2>().0;
        let p = |i| &points[sites.at(i)];
        if lifted {
            stage!(
                EXACT,
                incircle_unchecked,
                incircle_filter,
                p(0),
                p(1),
                p(2),
                p(3),
                envelope
            )
        } else {
            stage!(
                EXACT,
                orient2d_unchecked,
                orient2d_filter,
                p(0),
                p(1),
                p(2),
                envelope
            )
        }
    } else {
        let points = cloud.points.as_chunks::<3>().0;
        let p = |i| &points[sites.at(i)];
        if lifted {
            stage!(
                EXACT,
                insphere_unchecked,
                insphere_filter,
                p(0),
                p(1),
                p(2),
                p(3),
                p(4),
                envelope
            )
        } else {
            stage!(
                EXACT,
                orient3d_unchecked,
                orient3d_filter,
                p(0),
                p(1),
                p(2),
                p(3),
                envelope
            )
        }
    }
}

// Equal homogeneous weights reduce to the affine orientation of their prefixes.
#[inline]
fn fixed_homogeneous<const EXACT: bool>(
    rows: &[f64],
    sites: SiteRows<'_>,
    dimension: usize,
    envelope: bool,
) -> Option<i8> {
    if dimension == 2 {
        let rows = rows.as_chunks::<3>().0;
        let p = |i| &rows[sites.at(i)][..2].as_chunks::<2>().0[0];
        stage!(
            EXACT,
            orient2d_unchecked,
            orient2d_filter,
            p(0),
            p(1),
            p(2),
            envelope
        )
    } else {
        let rows = rows.as_chunks::<4>().0;
        let p = |i| &rows[sites.at(i)][..3].as_chunks::<3>().0[0];
        stage!(
            EXACT,
            orient3d_unchecked,
            orient3d_filter,
            p(0),
            p(1),
            p(2),
            p(3),
            envelope
        )
    }
}

// Packed workspace rows; cloud and query rows are validated at their input
// boundaries.
#[inline]
pub(super) fn fixed_rows<const EXACT: bool>(
    rows: &[f64],
    dimension: usize,
    lifted: bool,
) -> Option<i8> {
    if dimension == 2 {
        let p = |i: usize| [rows[2 * i], rows[2 * i + 1]];
        if lifted {
            stage!(
                EXACT,
                incircle_unchecked,
                incircle_filter,
                &p(0),
                &p(1),
                &p(2),
                &p(3),
                false
            )
        } else {
            stage!(
                EXACT,
                orient2d_unchecked,
                orient2d_filter,
                &p(0),
                &p(1),
                &p(2),
                false
            )
        }
    } else {
        let p = |i: usize| [rows[3 * i], rows[3 * i + 1], rows[3 * i + 2]];
        if lifted {
            stage!(
                EXACT,
                insphere_unchecked,
                insphere_filter,
                &p(0),
                &p(1),
                &p(2),
                &p(3),
                &p(4),
                false
            )
        } else {
            stage!(
                EXACT,
                orient3d_unchecked,
                orient3d_filter,
                &p(0),
                &p(1),
                &p(2),
                &p(3),
                false
            )
        }
    }
}

pub(super) fn provenance<S: RealScalar>(
    cloud: &Cloud<S>,
    work: &Work<'_, '_, '_, '_>,
) -> GeometryProvenance {
    GeometryProvenance {
        input_points: cloud.mapping.len(),
        unique_points: cloud.len(),
        predicate_terms: work.workspace.terms,
        canonical_work: work.canonical_work,
        resolved_ties: work.ties,
    }
}
