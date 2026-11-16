//! Indexed component heaps and exact running endpoint sums.
use super::*;
use crate::shell::{add, grow, grown_bytes, mul};
use core::cmp::Ordering;
use scioxide_core::checked_midpoint;

#[derive(Debug, Clone, Copy)]
struct Entry<R> {
    panel: usize,
    radius: R,
}

pub(super) struct Refinement<R: RealScalar> {
    heaps: Vec<Entry<R>>,
    positions: Vec<usize>,
    order: Vec<usize>,
    sums: Vec<BoxSum<R>>,
}

impl<R: RealScalar> Refinement<R> {
    pub(super) const fn new() -> Self {
        Self {
            heaps: Vec::new(),
            positions: Vec::new(),
            order: Vec::new(),
            sums: Vec::new(),
        }
    }
    pub(super) fn bytes(&self, panels: usize, components: usize) -> Result<usize> {
        let entries = mul(panels, components)?;
        [
            grown_bytes(&self.heaps, entries)?,
            grown_bytes(&self.positions, entries)?,
            grown_bytes(&self.order, panels)?,
            grown_bytes(&self.sums, if panels == 0 { 0 } else { components })?,
        ]
        .into_iter()
        .try_fold(0, add)
    }
    pub(super) fn fits(&self, panels: usize, components: usize) -> Result<bool> {
        let entries = mul(panels, components)?;
        Ok(self.heaps.len() >= entries
            && self.positions.len() >= entries
            && self.order.len() >= panels
            && (panels == 0 || self.sums.len() >= components))
    }
    pub(super) fn reserve(&mut self, panels: usize, components: usize) -> Result<()> {
        let entries = mul(panels, components)?;
        grow(
            &mut self.heaps,
            entries,
            Entry {
                panel: 0,
                radius: R::ZERO,
            },
            scioxide_core::quantity!("workspace.heaps"),
        )?;
        grow(
            &mut self.positions,
            entries,
            0,
            scioxide_core::quantity!("workspace.positions"),
        )?;
        grow(
            &mut self.order,
            panels,
            0,
            scioxide_core::quantity!("workspace.order"),
        )?;
        if panels != 0 {
            grow(
                &mut self.sums,
                components,
                BoxSum::new(Box2::zero()?, scioxide_core::argument!("self"))?,
                scioxide_core::quantity!("workspace.sums"),
            )?;
        }
        Ok(())
    }
    pub(super) fn view(&mut self, panels: usize, components: usize) -> Scratch<'_, R> {
        Scratch {
            heaps: &mut self.heaps[..panels * components],
            positions: &mut self.positions[..panels * components],
            order: &mut self.order[..panels],
            sums: &mut self.sums[..if panels == 0 { 0 } else { components }],
            capacity: panels,
            components,
            len: 0,
            unresolved: false,
        }
    }
}

pub(super) struct Scratch<'a, R: RealScalar> {
    heaps: &'a mut [Entry<R>],
    positions: &'a mut [usize],
    order: &'a mut [usize],
    pub(super) sums: &'a mut [BoxSum<R>],
    capacity: usize,
    components: usize,
    len: usize,
    pub(super) unresolved: bool,
}

/// Earlier leaves win equal-radius comparisons, independently of storage order.
fn domain_order<R: RealScalar>(a: QuadraturePanel<R>, b: QuadraturePanel<R>) -> Ordering {
    a.piece.cmp(&b.piece).then_with(|| {
        a.domain
            .bounds()
            .and_then(|(lo, _)| lo)
            .partial_cmp(&b.domain.bounds().and_then(|(lo, _)| lo))
            .unwrap_or(Ordering::Equal)
    })
}
fn greater<R: RealScalar>(a: Entry<R>, b: Entry<R>, panels: &[QuadraturePanel<R>]) -> bool {
    a.radius > b.radius
        || (a.radius == b.radius && domain_order(panels[a.panel], panels[b.panel]).is_lt())
}
fn exchange<R: RealScalar>(heap: &mut [Entry<R>], positions: &mut [usize], a: usize, b: usize) {
    heap.swap(a, b);
    positions[heap[a].panel] = a;
    positions[heap[b].panel] = b;
}
fn repair<R: RealScalar>(
    heap: &mut [Entry<R>],
    positions: &mut [usize],
    mut pos: usize,
    panels: &[QuadraturePanel<R>],
) {
    while pos != 0 {
        let parent = (pos - 1) / 2;
        if !greater(heap[pos], heap[parent], panels) {
            break;
        }
        exchange(heap, positions, pos, parent);
        pos = parent;
    }
    loop {
        let left = 2 * pos + 1;
        if left >= heap.len() {
            break;
        }
        let right = left + 1;
        let child = if right < heap.len() && greater(heap[right], heap[left], panels) {
            right
        } else {
            left
        };
        if !greater(heap[child], heap[pos], panels) {
            break;
        }
        exchange(heap, positions, pos, child);
        pos = child;
    }
}
/// The two halves of a panel of a regular piece, when its midpoint is interior.
pub(super) fn bisect<R: RealScalar>(
    panel: QuadraturePanel<R>,
) -> Result<Option<[QuadraturePanel<R>; 2]>> {
    let (a, b) = finite_endpoints(panel.domain, UNBOUNDED_PANEL)?;
    let midpoint = checked_midpoint(a, b)?;
    if !(a < midpoint && midpoint < b) {
        return Ok(None);
    }
    Ok(Some([
        QuadraturePanel {
            domain: ExtendedInterval::new(Some(a), Some(midpoint))?,
            ..panel
        },
        QuadraturePanel {
            domain: ExtendedInterval::new(Some(midpoint), Some(b))?,
            ..panel
        },
    ]))
}
impl<R: RealScalar> Scratch<'_, R> {
    /// Schedule panel `index`, whose piece has a regular assumption, by the radii of
    /// its finite local `bounds`. A panel without finite bounds must be bisectable; it
    /// takes infinite radius in every component, so [`Self::select`] returns it first.
    pub(super) fn insert(
        &mut self,
        index: usize,
        panels: &[QuadraturePanel<R>],
        bounds: Option<&[IntervalBox<R>]>,
    ) -> Result<()> {
        let splittable = bisect(panels[index])?.is_some();
        let mut nonzero = false;
        for component in 0..self.components {
            let radius = match bounds {
                Some(bounds) => adaptive::radius_scale(bounds[component])?.0,
                None => R::INFINITY,
            };
            nonzero |= radius != R::ZERO;
            if splittable {
                let start = component * self.capacity;
                self.heaps[start + self.len] = Entry {
                    panel: index,
                    radius,
                };
                self.positions[start + index] = self.len;
                repair(
                    &mut self.heaps[start..start + self.len + 1],
                    &mut self.positions[start..start + self.capacity],
                    self.len,
                    panels,
                );
            }
        }
        if splittable {
            self.len += 1;
        } else {
            self.unresolved |= nonzero;
        }
        Ok(())
    }
    /// The scheduled panel to bisect next: one without finite bounds, earliest first,
    /// and otherwise the one with the largest radius relative to the component
    /// tolerance of `total`, which is then read and must be finite; `None` when every
    /// scheduled radius is zero.
    pub(super) fn select(
        &self,
        panels: &[QuadraturePanel<R>],
        total: &[IntervalBox<R>],
        tolerance: &ErrorTolerance<R>,
    ) -> Result<Option<usize>> {
        let mut selected = None;
        let mut largest = R::ZERO;
        if self.len == 0 {
            return Ok(None);
        }
        // Infinite radii top every component heap; zero-tolerance ties cannot pass them.
        if let Some(entry) = self.heaps.first().filter(|entry| !entry.radius.is_finite()) {
            return Ok(Some(entry.panel));
        }
        for (component, &value) in total.iter().enumerate() {
            let entry = self.heaps[component * self.capacity];
            let priority =
                adaptive::radius_ratio(entry.radius, value, tolerance, component, total.len())?;
            if priority > largest
                || (priority != R::ZERO
                    && priority == largest
                    && selected
                        .is_some_and(|old| domain_order(panels[entry.panel], panels[old]).is_lt()))
            {
                selected = Some(entry.panel);
                largest = priority;
            }
        }
        Ok(selected)
    }
    pub(super) fn remove(&mut self, index: usize, panels: &[QuadraturePanel<R>]) {
        self.len -= 1;
        for (heap, positions) in self
            .heaps
            .chunks_exact_mut(self.capacity)
            .zip(self.positions.chunks_exact_mut(self.capacity))
        {
            let pos = positions[index];
            exchange(heap, positions, pos, self.len);
            if pos < self.len {
                repair(&mut heap[..self.len], positions, pos, panels);
            }
        }
    }
    pub(super) fn sort(
        &mut self,
        panels: &mut [QuadraturePanel<R>],
        bounds: &mut [IntervalBox<R>],
        components: usize,
    ) {
        let order = &mut self.order[..panels.len()];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        order.sort_unstable_by(|&a, &b| domain_order(panels[a], panels[b]));
        // order[destination] = source. Visit each cycle once, marking its map.
        for start in 0..order.len() {
            if order[start] == usize::MAX {
                continue;
            }
            let mut destination = start;
            loop {
                let source = order[destination];
                order[destination] = usize::MAX;
                if source == start {
                    break;
                }
                panels.swap(destination, source);
                for component in 0..components {
                    bounds.swap(
                        destination * components + component,
                        source * components + component,
                    );
                }
                destination = source;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::NormSelection;

    #[test]
    fn component_heaps_follow_changing_relative_scales_and_removal() {
        let mut storage = Refinement::<f64>::new();
        storage.reserve(32, 3).unwrap();
        let mut heaps = storage.view(32, 3);
        let mut panels = [placeholder(); 32];
        let mut bounds = [[zero().unwrap(); 3]; 32];
        // Insert in reverse domain order; index order must not settle ties.
        for (index, panel) in panels.iter_mut().enumerate() {
            let a = (31 - index) as f64;
            *panel = QuadraturePanel {
                piece: 0,
                domain: ExtendedInterval::new(Some(a), Some(a + 1.)).unwrap(),
            };
            for (component, value) in bounds[index].iter_mut().enumerate() {
                let radius = ((index * (component + 3) + component) % 17 + 1) as f64;
                *value = pack(Box2::real(D::new(-radius, radius).unwrap()).unwrap());
            }
        }
        for (index, local) in bounds.iter().enumerate() {
            heaps.insert(index, &panels, Some(local)).unwrap();
        }
        let tolerance = ErrorTolerance::scalar(0.125, 0.25, NormSelection::Infinity).unwrap();
        let mut live = [true; 32];
        for turn in 0..32 {
            let total = core::array::from_fn::<_, 3, _>(|component| {
                pack(
                    Box2::real(D::point(if component == turn % 3 { 1. } else { 1000. }).unwrap())
                        .unwrap(),
                )
            });
            let score = |index: usize| {
                bounds[index]
                    .iter()
                    .enumerate()
                    .map(|(c, &value)| adaptive::ratio(value, total[c], &tolerance, c, 3).unwrap())
                    .fold(0_f64, f64::max)
            };
            let expected = (0..32)
                .filter(|&i| live[i])
                .max_by(|&a, &b| {
                    score(a)
                        .partial_cmp(&score(b))
                        .unwrap()
                        .then_with(|| domain_order(panels[b], panels[a]))
                })
                .unwrap();
            assert_eq!(
                heaps.select(&panels, &total, &tolerance).unwrap(),
                Some(expected)
            );
            heaps.remove(expected, &panels);
            live[expected] = false;
        }
        assert_eq!(
            heaps
                .select(&panels, &[zero().unwrap(); 3], &tolerance)
                .unwrap(),
            None
        );
    }

    #[test]
    fn panels_without_finite_bounds_precede_zero_tolerance_ties() {
        let mut storage = Refinement::<f64>::new();
        storage.reserve(4, 2).unwrap();
        let mut heaps = storage.view(4, 2);
        let panels: [_; 4] = core::array::from_fn(|index| QuadraturePanel {
            piece: 0,
            domain: ExtendedInterval::new(Some(index as f64), Some(index as f64 + 1.)).unwrap(),
        });
        let wide = pack(Box2::real(D::new(-1., 1.).unwrap()).unwrap());
        for index in [0, 2] {
            heaps.insert(index, &panels, Some(&[wide; 2])).unwrap();
        }
        for index in [3, 1] {
            heaps.insert(index, &panels, None).unwrap();
        }
        // Every finite radius has infinite priority at zero tolerance, and finite
        // panels come earlier in domain order; the total is not read meanwhile.
        let tolerance = ErrorTolerance::scalar(0., 0., NormSelection::Infinity).unwrap();
        let unread = [sentinel(); 2];
        for expected in [1, 3] {
            assert_eq!(
                heaps.select(&panels, &unread, &tolerance).unwrap(),
                Some(expected)
            );
            heaps.remove(expected, &panels);
        }
        let total = [wide; 2];
        assert_eq!(heaps.select(&panels, &total, &tolerance).unwrap(), Some(0));
    }
}
