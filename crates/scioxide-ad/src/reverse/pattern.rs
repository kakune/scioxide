//! Structural patterns of a recording by index domains.
//!
//! A forward pass propagates, for every slot, the sorted set of inputs it depends
//! on. The sets live in one arena; a unary statement shares the set of its
//! argument. Jacobian rows are the sets of the outputs. The Hessian pattern adds
//! nonlinear interaction domains (Walther 2008) over the statements the output
//! depends on.

use alloc::vec::Vec;

use scioxide_core::{SciOxideError, StructuralPattern};

use super::TapeValue;
use super::recorder::{Columns, Frozen};
use super::word::Word;
use crate::rules::Class;

fn allocation(quantity: scioxide_core::Quantity, len: usize) -> SciOxideError {
    SciOxideError::allocation_failure(quantity, len)
}

/// No set: a passive slot.
const EMPTY: usize = usize::MAX;
/// A slot read at the end: an output.
const KEEP: usize = usize::MAX;

/// Index domains of the slots of a forward pass: shared, reference-counted sorted
/// sets, released after the last statement that reads them, so the memory is that
/// of the domains alive at one time.
struct Domains {
    /// The set of each slot, an index into `sets`, or [`EMPTY`].
    of: Vec<usize>,
    sets: Vec<Vec<usize>>,
    refs: Vec<usize>,
    free: Vec<usize>,
    /// The word index of the last statement that reads each slot, or [`KEEP`].
    last: Vec<usize>,
    /// Scratch of `union`: the merged prefix and the next merge.
    merged: Vec<usize>,
    spare: Vec<usize>,
}

fn reserve<T>(vector: &mut Vec<T>, additional: usize) -> Result<(), SciOxideError> {
    vector.try_reserve(additional).map_err(|_| {
        allocation(
            scioxide_core::quantity!("pattern"),
            vector.len() + additional,
        )
    })
}

impl Domains {
    fn get(&self, slot: usize) -> &[usize] {
        match self.of.get(slot) {
            Some(&id) if id != EMPTY => &self.sets[id],
            _ => &[],
        }
    }

    fn id(&self, slot: usize) -> usize {
        self.of.get(slot).copied().unwrap_or(EMPTY)
    }

    /// Gives `slot` the set `id`, shared.
    fn push(&mut self, id: usize) -> Result<(), SciOxideError> {
        if id != EMPTY {
            self.refs[id] += 1;
        }
        reserve(&mut self.of, 1)?;
        self.of.push(id);
        Ok(())
    }

    /// Drops the set of `slot` if `index` is its last reader.
    fn release(&mut self, slot: usize, index: usize) {
        if self.last.get(slot) != Some(&index) {
            return;
        }
        let id = self.id(slot);
        if id == EMPTY {
            return;
        }
        self.of[slot] = EMPTY;
        self.refs[id] -= 1;
        if self.refs[id] == 0 {
            // Drop the buffer: a cleared one would keep its capacity on the free list.
            self.sets[id] = Vec::new();
            // `free` holds at most one entry per set, reserved with it.
            self.free.push(id);
        }
    }

    /// Returns a new set (with no references) holding the union of the sets of
    /// `slots`, or the shared set when every non-empty one is the same.
    fn union(
        &mut self,
        slots: impl Iterator<Item = usize> + Clone,
    ) -> Result<usize, SciOxideError> {
        let mut distinct = slots.clone().map(|s| self.id(s)).filter(|&id| id != EMPTY);
        let Some(first) = distinct.next() else {
            return Ok(EMPTY);
        };
        if distinct.all(|id| id == first) {
            return Ok(first);
        }
        let mut target = core::mem::take(&mut self.merged);
        let mut merged = core::mem::take(&mut self.spare);
        target.clear();
        for slot in slots {
            let id = self.id(slot);
            if id == EMPTY {
                continue;
            }
            let set = &self.sets[id];
            core::mem::swap(&mut merged, &mut target);
            target.clear();
            reserve(&mut target, merged.len() + set.len())?;
            let (mut a, mut b) = (0, 0);
            while a < merged.len() || b < set.len() {
                let next = match (merged.get(a), set.get(b)) {
                    (Some(&x), Some(&y)) if x < y => {
                        a += 1;
                        x
                    }
                    (Some(&x), Some(&y)) if x > y => {
                        b += 1;
                        y
                    }
                    (Some(&x), Some(_)) => {
                        a += 1;
                        b += 1;
                        x
                    }
                    (Some(&x), None) => {
                        a += 1;
                        x
                    }
                    (None, Some(&y)) => {
                        b += 1;
                        y
                    }
                    (None, None) => break,
                };
                target.push(next);
            }
        }
        self.spare = merged;
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                reserve(&mut self.sets, 1)?;
                reserve(&mut self.refs, 1)?;
                let additional = (self.sets.len() + 1).saturating_sub(self.free.len());
                reserve(&mut self.free, additional)?;
                self.sets.push(Vec::new());
                self.refs.push(0);
                self.sets.len() - 1
            }
        };
        // The merged buffer becomes the set; the next merge allocates its own.
        self.sets[id] = target;
        Ok(id)
    }
}

/// The argument slots of every statement word, by index.
fn arguments<V: TapeValue, C: Columns<V>>(
    columns: &C,
    index: usize,
) -> impl Iterator<Item = usize> {
    let word = Word(columns.word(index));
    [
        matches!(word.arity(), 1 | 2).then_some(word.slot0()),
        (word.arity() == 2).then_some(word.slot1()),
    ]
    .into_iter()
    .flatten()
}

/// Runs the forward pass of index domains, handing every statement to `visit`
/// (with its word index) before the domains it last reads are released. The
/// slots in `keep` keep their domains to the end.
fn domains<V: TapeValue>(
    frozen: &Frozen<'_, V>,
    inputs: usize,
    keep: &[u32],
    mut visit: impl FnMut(usize, &Domains) -> Result<(), SciOxideError>,
) -> Result<Domains, SciOxideError> {
    let words = frozen.words.len();
    let mut last = Vec::new();
    reserve(&mut last, words)?;
    last.resize(words, 0);
    let mut index = 1 + inputs;
    while index < words {
        let word = Word(frozen.word(index));
        match word.arity() {
            0 => {}
            1 => last[word.slot0().min(words - 1)] = index,
            2 => {
                last[word.slot0().min(words - 1)] = index;
                last[word.slot1().min(words - 1)] = index;
            }
            _ => {
                let count = word.count();
                let copy = index + count + 1;
                for k in 0..count {
                    let slot = Word(frozen.word(index + 1 + k)).slot0();
                    last[slot.min(words - 1)] = copy;
                }
                index = copy;
            }
        }
        index += 1;
    }
    for &slot in keep {
        if let Some(target) = last.get_mut(slot as usize) {
            *target = KEEP;
        }
    }
    let mut domains = Domains {
        of: Vec::new(),
        sets: Vec::new(),
        refs: Vec::new(),
        free: Vec::new(),
        last,
        merged: Vec::new(),
        spare: Vec::new(),
    };
    reserve(&mut domains.of, words)?;
    reserve(&mut domains.sets, inputs)?;
    reserve(&mut domains.refs, inputs)?;
    reserve(&mut domains.free, inputs)?;
    domains.of.push(EMPTY);
    for input in 0..inputs {
        let mut set = Vec::new();
        reserve(&mut set, 1)?;
        set.push(input);
        domains.sets.push(set);
        domains.refs.push(1);
        domains.of.push(input);
    }
    let mut index = 1 + inputs;
    while index < words {
        let word = Word(frozen.word(index));
        match word.arity() {
            0 => domains.push(EMPTY)?,
            1 => {
                let id = domains.id(word.slot0());
                domains.push(id)?;
                visit(index, &domains)?;
                domains.release(word.slot0(), index);
            }
            2 => {
                let id = domains.union([word.slot0(), word.slot1()].into_iter())?;
                domains.push(id)?;
                visit(index, &domains)?;
                domains.release(word.slot0(), index);
                domains.release(word.slot1(), index);
            }
            _ => {
                let count = word.count();
                let header = index;
                let copy = header + count + 1;
                let slots = (0..count).map(move |k| Word(frozen.word(header + 1 + k)).slot0());
                let id = domains.union(slots.clone())?;
                for _ in 0..=count {
                    domains.push(EMPTY)?;
                }
                domains.push(id)?;
                index = copy;
                if word.is_call() {
                    // Call outputs depend on every input of the call.
                    while index + 1 < words
                        && Word(frozen.word(index + 1)).arity() == 0
                        && Word(frozen.word(index + 1)).is_call_output()
                    {
                        domains.push(id)?;
                        index += 1;
                    }
                }
                visit(header, &domains)?;
                for slot in slots {
                    domains.release(slot, copy);
                }
            }
        }
        index += 1;
    }
    Ok(domains)
}

/// Builds a CSR pattern from sorted rows.
fn csr<'a>(
    rows: usize,
    cols: usize,
    row: impl Fn(usize) -> &'a [usize],
) -> Result<StructuralPattern, SciOxideError> {
    let mut pointers = Vec::new();
    pointers
        .try_reserve_exact(rows + 1)
        .map_err(|_| allocation(scioxide_core::quantity!("pattern"), rows + 1))?;
    let mut indices = Vec::new();
    pointers.push(0);
    for i in 0..rows {
        let entries = row(i);
        indices.try_reserve(entries.len()).map_err(|_| {
            allocation(
                scioxide_core::quantity!("pattern"),
                indices.len() + entries.len(),
            )
        })?;
        indices.extend_from_slice(entries);
        pointers.push(indices.len());
    }
    StructuralPattern::new(rows, cols, pointers, indices)
}

/// The Jacobian pattern: outputs × inputs.
pub(crate) fn jacobian<V: TapeValue>(
    frozen: &Frozen<'_, V>,
    inputs: usize,
    outputs: &[u32],
) -> Result<StructuralPattern, SciOxideError> {
    let domains = domains(frozen, inputs, outputs, |_, _| Ok(()))?;
    csr(outputs.len(), inputs, |row| {
        domains.get(outputs[row] as usize)
    })
}

fn insert_all(set: &mut Vec<usize>, values: &[usize]) -> Result<(), SciOxideError> {
    for &value in values {
        if let Err(position) = set.binary_search(&value) {
            set.try_reserve(1)
                .map_err(|_| allocation(scioxide_core::quantity!("pattern"), set.len() + 1))?;
            set.insert(position, value);
        }
    }
    Ok(())
}

/// The Hessian pattern of the output at `output`: inputs × inputs.
pub(crate) fn hessian<V: TapeValue>(
    frozen: &Frozen<'_, V>,
    inputs: usize,
    output: u32,
) -> Result<StructuralPattern, SciOxideError> {
    let words = frozen.words.len();
    // Statements the output depends on.
    let mut live = Vec::new();
    live.try_reserve_exact(words)
        .map_err(|_| allocation(scioxide_core::quantity!("pattern"), words))?;
    live.resize(words, false);
    if let Some(flag) = live.get_mut(output as usize) {
        *flag = true;
    }
    let mut index = words;
    while index > 0 {
        index -= 1;
        let word = Word(frozen.word(index));
        match word.arity() {
            0 => {
                // A live call output makes its call live.
                if word.is_call_output() && live[index] {
                    let mut copy = index;
                    while copy > 0 && Word(frozen.word(copy)).arity() == 0 {
                        copy -= 1;
                    }
                    live[copy] = true;
                }
            }
            3 => {
                let count = word.count();
                let first = index.saturating_sub(count);
                if live[index] {
                    for k in 0..count {
                        let slot = Word(frozen.word(first + k)).slot0();
                        if let Some(flag) = live.get_mut(slot) {
                            *flag = true;
                        }
                    }
                }
                index = first.saturating_sub(1);
            }
            _ => {
                if live[index] {
                    for slot in arguments(frozen, index) {
                        if let Some(flag) = live.get_mut(slot) {
                            *flag = true;
                        }
                    }
                }
            }
        }
    }
    let mut interactions: Vec<Vec<usize>> = Vec::new();
    interactions
        .try_reserve_exact(inputs)
        .map_err(|_| allocation(scioxide_core::quantity!("pattern"), inputs))?;
    interactions.resize_with(inputs, Vec::new);
    let square = |interactions: &mut Vec<Vec<usize>>, set: &[usize]| -> Result<(), SciOxideError> {
        for &i in set {
            insert_all(&mut interactions[i], set)?;
        }
        Ok(())
    };
    let cross = |interactions: &mut Vec<Vec<usize>>,
                 a: &[usize],
                 b: &[usize]|
     -> Result<(), SciOxideError> {
        for &i in a {
            insert_all(&mut interactions[i], b)?;
        }
        for &j in b {
            insert_all(&mut interactions[j], a)?;
        }
        Ok(())
    };
    let mut slots = Vec::new();
    let mut union = Vec::new();
    domains(frozen, inputs, &[], |index, domains| {
        let word = Word(frozen.word(index));
        match word.arity() {
            1 if live[index] && word.class() == Class::Nonlinear => {
                square(&mut interactions, domains.get(word.slot0()))?;
            }
            2 if live[index] => match word.class() {
                Class::Linear => {}
                Class::Bilinear => cross(
                    &mut interactions,
                    domains.get(word.slot0()),
                    domains.get(word.slot1()),
                )?,
                Class::Nonlinear => {
                    union.clear();
                    insert_all(&mut union, domains.get(word.slot0()))?;
                    insert_all(&mut union, domains.get(word.slot1()))?;
                    square(&mut interactions, &union)?;
                }
            },
            3 => {
                let count = word.count();
                let copy = index + count + 1;
                if live.get(copy).copied().unwrap_or(false) {
                    slots.clear();
                    reserve(&mut slots, count)?;
                    slots.extend((0..count).map(|k| Word(frozen.word(index + 1 + k)).slot0()));
                    match (word.is_call(), word.class()) {
                        (false, Class::Linear) => {}
                        (false, Class::Bilinear) => {
                            let half = count / 2;
                            for k in 0..half {
                                cross(
                                    &mut interactions,
                                    domains.get(slots[k]),
                                    domains.get(slots[k + half]),
                                )?;
                            }
                        }
                        _ => {
                            union.clear();
                            for &slot in &slots {
                                insert_all(&mut union, domains.get(slot))?;
                            }
                            square(&mut interactions, &union)?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    csr(inputs, inputs, |row| &interactions[row])
}
