use ::std::hash::Hash;
use ::std::result::Result as StdResult;

// NOTE: Currently there is no "group" trait, for a couple of reasons:
//
// * Many groups will depend on some form of context, which is awkward
//   to work into a trait (but trivial to add to a closure).
// * Making the group operation as part of the type hides some
//   potentially important considerations.  A type may have multiple
//   possible choices of the group operator, and use of a homomorphism
//   requires selecting the right one.

/// Tree representation of a finite group, with generators as leaves.
pub(crate) struct GroupTree<G> {
    members: Vec<G>,
    decomps: Vec<Option<(usize, usize)>>,
}

impl<G> GroupTree<G>
{
    /// Constructs a `GroupTree<G>` given a sequence that contains
    /// each member of a finite group exactly once.
    ///
    /// A `GroupTree<G>` constructed in this manner is guaranteed to
    /// order its elements in the same order as the input `Vec`.
    ///
    /// In line with the library's predominantly row-centric design,
    /// arguments of the closure are flipped from the typical mathematical
    /// convention. `compose(a, b)` should perform *`a` followed by `b`*,
    /// (i.e. "`b` of `a`").
    pub fn from_all_members(
        members: Vec<G>,
        mut compose: impl FnMut(&G, &G) -> G,
    ) -> Self
    where G: Hash + Eq + Clone,
    {
        // we cannot construct the identity without any generators
        assert!(members.len() > 0, "empty groups do not exist!");

        let indices: ::std::collections::HashMap<G, usize> =
            members.iter().cloned()
            .enumerate().map(|(i, x)| (x, i))
            .collect();

        // Brute force O(G)^2 attempt to fill the tree.
        // I'm fairly certain this can be improved in some way by using
        // known element-inverse pairs to quickly find new members,
        // but I don't think it's worth it since this will probably only ever
        // be used on spacegroups, which are not terribly large.
        let mut decomps = vec![None; members.len()];
        for a in 0..members.len() {
            for b in 0..=a {
                let c = indices[&compose(&members[a], &members[b])];
                if c > a {
                    decomps[c] = Some((a, b));
                }
            }
        }
        GroupTree { members, decomps }
    }

    /// Compute a homomorphism of a group using the tree
    /// to elide expensive computations.
    ///
    /// Ideally, `F` should be a function that is very expensive to
    /// compute, while `HFn` should be comparatively cheaper.
    ///
    /// `compose(a, b)` should compute `b of a`.
    pub fn compute_homomorphism<H>(
        &self,
        mut compute: impl FnMut(usize, &G) -> H,
        mut compose: impl FnMut(&H, &H) -> H,
    ) -> Vec<H>
    {
        self.try_compute_homomorphism(
            |idx, g| Ok::<_, ()>(compute(idx, g)),
            |a, b| Ok::<_, ()>(compose(a, b)),
        ).unwrap()
    }

    /// `compute_homomorphism` for fallible functions.
    pub fn try_compute_homomorphism<E, H>(
        &self,
        mut compute: impl FnMut(usize, &G) -> StdResult<H, E>,
        mut compose: impl FnMut(&H, &H) -> StdResult<H, E>,
    ) -> StdResult<Vec<H>, E>
    {Ok({
        let len = self.members.len();
        let mut out = Vec::with_capacity(len);

        for (index, g, decomp) in izip!(0..len, &self.members, &self.decomps) {
            let value = match *decomp {
                None => compute(index, g)?,
                Some((a, b)) => compose(&out[a], &out[b])?,
            };
            out.push(value);
        }
        out
    })}
}

/// Generates a finite group from a non-empty set of generators.
///
/// The generators may contain duplicates or extraneous elements.
///
/// The order of the output is arbitrary, but consistent for
/// inputs that are related by a group isomorphism.
pub fn generate_finite_group<G>(
    generators: &[G],
    mut g_fn: impl FnMut(&G, &G) -> G,
) -> Vec<G>
where G: Hash + Eq + Clone,
{
    use ::std::collections::{HashSet, VecDeque};
    assert!(generators.len() > 0, "empty groups do not exist!");

    let mut seen = HashSet::new();
    let mut out = vec![];

    let mut queue: VecDeque<_> = generators.iter().cloned().collect();

    while let Some(g) = queue.pop_front() {
        if seen.insert(g.clone()) {
            queue.extend(generators.iter().map(|h| g_fn(&g, h)));
            out.push(g);
        }
    }
    out
}
