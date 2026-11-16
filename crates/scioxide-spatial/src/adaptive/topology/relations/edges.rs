//! Exact affine-line keys and output-sensitive interval containment joins.
use super::*;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Line {
    // Root coordinate, primitive signed direction, intercept numerator.
    terms: [(usize, i64, i128); 8],
    exponent: u8,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Segment {
    line: Line,
    lower: u64,
    upper: u64,
    id: usize,
    sign: i8,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn segment(a: Key, b: Key, id: usize, work: &mut Work<'_, '_, '_>) -> Result<Segment> {
    work.tick()?;
    let exponent = a.exponent.max(b.exponent);
    let mut roots = [0; 16];
    let mut len = 0;
    for &(root, _) in a.terms[..usize::from(a.len)]
        .iter()
        .chain(&b.terms[..usize::from(b.len)])
    {
        roots[len] = root;
        len += 1;
    }
    roots[..len].sort_unstable();
    let mut count = 0;
    for i in 0..len {
        if count == 0 || roots[i] != roots[count - 1] {
            roots[count] = roots[i];
            count += 1;
        }
    }
    if count > 8 {
        return Err(work.invalid("edge exceeds root support"));
    }
    let weight = |key: Key, root: usize| -> i64 {
        key.terms[..usize::from(key.len)]
            .iter()
            .find(|term| term.0 == root)
            .map_or(0, |term| (term.1 << (exponent - key.exponent)) as i64)
    };
    let mut direction = [0i64; 8];
    let mut divisor = 0;
    for (out, &root) in direction.iter_mut().zip(&roots[..count]) {
        *out = weight(b, root) - weight(a, root);
        divisor = gcd(divisor, out.unsigned_abs());
    }
    let pivot = direction[..count]
        .iter()
        .position(|&d| d != 0)
        .ok_or_else(|| work.invalid("zero provenance edge"))?;
    let sign = if direction[pivot] > 0 { 1 } else { -1 };
    for d in &mut direction[..count] {
        *d = *d / divisor as i64 * i64::from(sign);
    }
    let mut terms = [(0, 0, 0); 8];
    let origin = weight(a, roots[pivot]);
    for (i, &root) in roots[..count].iter().enumerate() {
        // Depth <= 20 bounds aligned weights and directions by 2^60.
        // These products and their difference fit in signed 128 bits.
        let intercept = i128::from(weight(a, root)) * i128::from(direction[pivot])
            - i128::from(origin) * i128::from(direction[i]);
        terms[i] = (root, direction[i], intercept);
    }
    let mut intercept_exponent = exponent;
    while intercept_exponent > 0 && terms[..count].iter().all(|term| term.2 % 2 == 0) {
        for term in &mut terms[..count] {
            term.2 /= 2;
        }
        intercept_exponent -= 1;
    }
    // One fixed dyadic scale makes endpoint comparison exact across levels.
    let x = (origin as u64) << (60 - exponent);
    let y = (weight(b, roots[pivot]) as u64) << (60 - exponent);
    Ok(Segment {
        line: Line {
            terms,
            exponent: intercept_exponent,
        },
        lower: x.min(y),
        upper: x.max(y),
        id,
        sign,
    })
}

fn segments<S: RealScalar, const D: usize>(
    storage: &Storage<S, D>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    work.reserve(&mut segments, storage.edges.len())?;
    for (id, edge) in storage.edges.iter().enumerate() {
        segments.push(segment(
            storage.vertices[edge[0]].key,
            storage.vertices[edge[1]].key,
            id,
            work,
        )?);
    }
    work.sort(&mut segments)?;
    Ok(segments)
}

fn join(
    sources: &[Segment],
    destinations: &[Segment],
    reverse: bool,
    index: &mut Index<(u64, usize)>,
    matches: &mut Vec<Match>,
    work: &mut Work<'_, '_, '_>,
) -> Result<()> {
    index.clear();
    let mut next = 0;
    for destination in destinations {
        work.tick()?;
        while next < sources.len() && sources[next].lower <= destination.lower {
            index.insert((sources[next].upper, sources[next].id), next, work)?;
            next += 1;
        }
        index.from(&(destination.upper, 0), work, |at, work| {
            let source = &sources[at];
            let same = source.lower == destination.lower && source.upper == destination.upper;
            if reverse && same {
                return Ok(());
            }
            let (source_id, destination_id) = if reverse {
                (destination.id, source.id)
            } else {
                (source.id, destination.id)
            };
            work.push(
                matches,
                Match {
                    source: source_id,
                    destination: destination_id,
                    kind: if same {
                        0
                    } else if reverse {
                        -1
                    } else {
                        1
                    },
                    orientation: source.sign * destination.sign,
                },
            )
        })?;
    }
    Ok(())
}

pub(super) fn relations<S: RealScalar, const D: usize>(
    a: &Storage<S, D>,
    b: &Storage<S, D>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<EntityRelation>> {
    let sources = segments(a, work)?;
    let destinations = segments(b, work)?;
    let mut matches = Vec::new();
    let mut index = Index::new();
    let (mut i, mut j) = (0, 0);
    while i < sources.len() && j < destinations.len() {
        work.tick()?;
        match sources[i].line.cmp(&destinations[j].line) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                let (mut end_i, mut end_j) = (i + 1, j + 1);
                while end_i < sources.len() && sources[end_i].line == sources[i].line {
                    work.tick()?;
                    end_i += 1;
                }
                while end_j < destinations.len() && destinations[end_j].line == destinations[j].line
                {
                    work.tick()?;
                    end_j += 1;
                }
                join(
                    &sources[i..end_i],
                    &destinations[j..end_j],
                    false,
                    &mut index,
                    &mut matches,
                    work,
                )?;
                join(
                    &destinations[j..end_j],
                    &sources[i..end_i],
                    true,
                    &mut index,
                    &mut matches,
                    work,
                )?;
                i = end_i;
                j = end_j;
            }
        }
    }
    finish(a.edges.len(), b.edges.len(), matches, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn line_keys_are_exact_at_the_dyadic_exponent_bound() -> Result<()> {
        let key = |weights: [u64; 8], exponent| Key {
            terms: core::array::from_fn(|id| (id, weights[id])),
            len: 8,
            exponent,
        };
        let scale = 1u64 << 59;
        let a = [scale - 7, 1, 1, 1, 1, 1, 1, 1];
        let b = [2, scale - 10, 3, 1, 1, 1, 1, 1];
        let midpoint = key(core::array::from_fn(|i| a[i] + b[i]), 60);
        let a = key(a, 59);
        let b = key(b, 59);
        let mut work = Work::new(
            AdaptationLimits::new()
                .with_max_depth(20)
                .with_max_cells(1)
                .with_max_entities(8)
                .with_max_work(100)
                .with_max_storage_bytes(1024),
            None,
            scioxide_core::argument!("edge"),
        )?;
        let whole = segment(a, b, 0, &mut work)?;
        let left = segment(a, midpoint, 1, &mut work)?;
        let right = segment(midpoint, b, 2, &mut work)?;
        let reversed = segment(b, a, 3, &mut work)?;
        assert!(whole.line == left.line && whole.line == right.line && whole.line == reversed.line);
        assert_eq!(whole.sign, -reversed.sign);
        assert_eq!((whole.lower, whole.upper), (reversed.lower, reversed.upper));
        assert!(left.lower >= whole.lower && left.upper <= whole.upper);
        assert!(right.lower >= whole.lower && right.upper <= whole.upper);
        assert!(on_edge(midpoint, a, b));
        Ok(())
    }
}
