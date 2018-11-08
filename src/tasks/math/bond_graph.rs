
use ::math::bonds::{FracBonds, FracBond};

use ::rsp2_array_types::V3;

use ::rsp2_newtype_indices::index_cast;
use ::std::ops::{Deref};
use ::petgraph::prelude::*;

pub use self::periodic::PeriodicGraph;
pub mod periodic {
    use super::*;

    pub type Node = ();
    pub type Edge = V3<i32>;

    /// Constructs a finite graph representing bonds per unitcell.
    ///
    /// This is basically just a different representation of FracBonds, which is
    /// better equipped for certain types of lookup and modification.
    /// (It's *terrible* for most graph algorithms, though)
    ///
    /// # Properties
    ///
    /// * The graph will be a directed multi-graph.
    /// * Parallel edges will each represent a different `image_diff`, stored
    ///   as the edge weight.  This image diff is `dest_image - src_image`.
    /// * There should be no self edges with `image_diff == [0, 0, 0]`.
    /// * If the `FracBonds` held directed interactions, then for any edge `S -> T` with
    ///   `image_diff`, there is another edge `T -> S` with `-1 * image_diff`.
    #[derive(Debug, Clone)]
    pub struct PeriodicGraph(DiGraph<Node, Edge>);

    impl Deref for PeriodicGraph {
        type Target = DiGraph<Node, Edge>;

        fn deref(&self) -> &Self::Target { &self.0 }
    }

    impl FracBonds {
        /// Constructs a finite graph representing bonds per unitcell.
        ///
        /// This is basically just a different representation of FracBonds, which is
        /// better equipped for certain types of lookup and modification.
        /// (It's *terrible* for most graph algorithms, though)
        ///
        /// # Properties
        ///
        /// * Node indices will match site indices.
        /// * The graph will be a directed multi-graph.
        /// * Parallel edges will each represent a different `image_diff`, stored
        ///   as the edge weight.
        pub fn to_periodic_graph(&self) -> PeriodicGraph {
            let num_atoms = self.num_atoms_per_cell();

            let mut graph = DiGraph::new();
            for site in 0..num_atoms {
                assert_eq!(NodeIndex::new(site), graph.add_node(()));
            }
            for FracBond { from, to, image_diff } in self {
                graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), image_diff);
            }
            PeriodicGraph(graph)
        }
    }

    impl From<PeriodicGraph> for FracBonds {
        fn from(PeriodicGraph(g): PeriodicGraph) -> FracBonds {
            let (nodes, edges) = g.into_nodes_edges();

            FracBonds::from_iter(nodes.len(), edges.into_iter().map(|bond| {
                let from = bond.source().index();
                let to = bond.target().index();
                let image_diff = bond.weight;
                FracBond { from, to, image_diff }
            }))
        }
    }

    impl PeriodicGraph {
        /// Get keys labelling each site by the connected component it belongs to.
        ///
        /// The keys are assigned deterministically, but arbitrarily, and are **not**
        /// necessarily consecutive integers.
        pub fn connected_components_by_site(&self) -> Vec<ComponentLabel> {
            use ::petgraph::visit::NodeIndexable;

            // petgraph has a connected_components function, but it only gives the count!
            // ho hum.
            let mut vertex_sets = ::petgraph::unionfind::UnionFind::new(self.node_bound());
            for edge in self.0.edge_references() {
                let (a, b) = (edge.source(), edge.target());
                vertex_sets.union(self.to_index(a), self.to_index(b));
            }
            index_cast(vertex_sets.into_labeling())
        }

        pub fn frac_bonds_from(&self, from: usize) -> impl Iterator<Item=FracBond> + '_ {
            self.0.edges(NodeIndex::new(from)).map(move |edge| {
                let to = edge.target().index();
                let image_diff = edge.weight().clone();
                FracBond { from, to, image_diff }
            })
        }
    }
}

/// Label of a connected component, suitable for partitioning.
///
/// These are **not** indices into anything; they are merely arbitrary integers that are
/// assigned deterministically from the topology of a graph.
newtype_index!{ ComponentLabel }

//----------------------------------------------------------------
// Mostly untested functionality that was needed at one point

//pub use self::cutout::CutoutGraph;
//pub mod cutout {
//    use super::*;
//
//    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
//    pub struct Node {
//        pub site: usize,
//        pub image: V3<i32>,
//    }
//
//    pub type Edge = ();
//    pub type CutoutGraph = UnGraphMap<Node, Edge>;
//
//    impl FracBonds {
//        // FIXME needs testing, might be buggy
//        #[allow(unused)]
//        /// Constructs a finite graph covering all of the sites in each of the requested
//        /// images of the cell (which may contain duplicates).
//        ///
//        /// Properties:
//        ///
//        /// * The graph will be simple. (no parallel edges)
//        /// * The graph is undirected.
//        ///
//        /// Suitable for computing shortest graph distances between specific images of sites.
//        ///
//        /// The graph does not represent the periodic system, but only a cutout;
//        /// it does **not** attempt to connect distant atoms with "wraparound" edges,
//        /// and atoms near the edge of the included region may appear to have a lower
//        /// bond order than the correct number.
//        pub fn to_cutout_graph(
//            &self,
//            requested_images: impl IntoIterator<Item=V3<i32>>,
//        ) -> CutoutGraph {
//            let bonds_by_image_diff: BTreeMap<V3<i32>, Vec<(usize, usize)>> = {
//                let mut map = BTreeMap::default();
//                for FracBond { from, to, image_diff } in self {
//                    map.entry(image_diff)
//                        .or_insert(vec![])
//                        .push((from, to));
//                }
//                map
//            };
//
//            let requested_images: BTreeSet<_> = requested_images.into_iter().collect();
//            let all_nodes = {
//                requested_images.iter()
//                    .flat_map(|&image| {
//                        (0..self.num_atoms_per_cell()).map(move |site| Node { site, image })
//                    })
//            };
//
//            // Initialize nodes of graph
//            let mut graph = CutoutGraph::default();
//            for node in all_nodes {
//                graph.add_node(node);
//            }
//
//            // Add every possible image of every bond.
//            // ("possible" meaning both endpoints are present)
//            for (&image_diff, pairs) in &bonds_by_image_diff {
//                for &from_image in &requested_images {
//                    let to_image = from_image + image_diff;
//                    if !requested_images.contains(&to_image) {
//                        continue;
//                    }
//
//                    graph.extend(pairs.iter().map(|&(from_site, to_site)| {
//                        let from = Node { site: from_site, image: from_image };
//                        let to   = Node { site: to_site,   image: to_image   };
//                        (from, to)
//                    }))
//                }
//            }
//            graph
//        }
//    }
//}

//// FIXME this needs testing
//#[allow(unused)]
///// Exclude from the interaction list any pair of sites connected by a path of `distance`
///// or fewer edges in the given bond graph.
/////
///// A possible use case is to restrict a potential to surface-to-surface interactions,
///// while still allowing these two surfaces to "connect" at an arbitrarily far-away point.
/////
///// (**NOTE:** be warned that doing so introduces a sharp cutoff which, if not addressed,
/////  will make the potential and force discontinuous)
//pub fn exclude_short_paths(
//    interactions: &PeriodicGraph,
//    bonds: &CutoutGraph,
//    distance: u32,
//) -> PeriodicGraph {
//    let mut out = PeriodicGraph(Graph::with_capacity(interactions.node_count(), interactions.edge_count()));
//    for i in 0..interactions.node_count() {
//        assert_eq!(NodeIndex::new(i), out.add_node(()));
//    }
//
//    for prim_from in interactions.node_indices() {
//        let super_from = cutout::Node { site: prim_from.index(), image: V3::zero() };
//
//        let exclude_targets = nodes_within(bonds, super_from, distance);
//
//        for edge in interactions.edges(prim_from) {
//            let prim_to = edge.target();
//            let image_to = edge.weight().clone();
//
//            let super_to = cutout::Node { site: prim_to.index(), image: image_to };
//            if !exclude_targets.contains_key(&super_to) {
//                out.add_edge(prim_from, prim_to, image_to);
//            }
//        }
//    }
//    out
//}

//// FIXME this needs testing
///// Find all nodes within (`<=`) `max_distance` edges of a given node, and produce a node-distance map.
//#[allow(unused)]
//fn nodes_within(
//    graph: &CutoutGraph,
//    start: cutout::Node,
//    max_distance: u32,
//) -> BTreeMap<cutout::Node, u32> {
//    // handwritten BFS because petgraph::Bfs can't be used to obtain predecessors.
//    let mut distances: BTreeMap<_, _> = Default::default();
//    let mut queue: VecDeque<_> = vec![(start, 0)].into_iter().collect();
//    while let Some((source, source_dist)) = queue.pop_front() {
//        if source_dist > max_distance {
//            break;
//        }
//        distances.insert(source, source_dist);
//
//        for target in graph.neighbors(source) {
//            if distances.contains_key(&target) {
//                continue;
//            }
//            queue.push_back((target, source_dist + 1));
//        }
//    }
//    trace!("DISTANCES: {:?}", distances);
//    distances
//}
