
pub type Point = (usize, usize);
use ::std::ops::Range;
use ::std::hash::Hash;
use ::std::collections::HashSet;

use ::slice_of_array::prelude::*;
use ::rsp2_slice_math::{v, V, vdot};

pub fn integrate_two_eigenvectors<E, F>(
    dims: (usize, usize),
    init_pos: &[[f64; 3]],
    ranges: (Range<f64>, Range<f64>),
    eigenvecs: (&[[f64; 3]], &[[f64; 3]]),
    mut compute_grad: F,
) -> Result<Vec<f64>, E>
where
    F: Fn(&[[f64; 3]]) -> Result<Vec<[f64; 3]>, E> + Sync,
    E: Send,
{
    let xs = linspace(ranges.0, dims.0);
    let ys = linspace(ranges.1, dims.1);

    integrate_grid_random(
        dims,
        |(x, y)| {Ok::<_,E>({
//            println!("XY {:?} {:?}", xs[x], ys[y]);
            let V(pos) = v(init_pos.flat())
                + xs[x] * v(eigenvecs.0.flat())
                + ys[y] * v(eigenvecs.1.flat());

            let pos = pos.to_vec();
//            println!("IP {:?}", init_pos.flat().iter().sum::<f64>());
//            println!("PP {:?}", pos.iter().sum::<f64>());
            let grad = compute_grad(pos.nest())?.flat().to_vec();
            (pos, grad)
        })},

        |(_, &(ref pos1, ref grad1)), (_, &(ref pos2, ref grad2))| {Ok({
            // trapezoid
            let V(grad) = 0.5 * (v(grad1) + v(grad2));
            let V(d_pos) = v(pos2) - v(pos1);
            vdot(&grad, &d_pos)
        })},
    )
}

#[derive(Debug, Clone)]
struct Tree<V> {
    /// Which node is the root?
    root: V,
    /// Tree edges `(source, target)`. Every node except the root will appear exactly once as the
    /// `target`, and it will always appear as a `target` before it ever appears as a `source`.
    edges: Vec<(V, V)>,
}

fn random_tree<V, Vs, F>(
    vertices: Vs,
    mut out_edges: F,
) -> Tree<V>
    where
        V: Clone + Hash + Eq,
        Vs: IntoIterator<Item=V>,
        F: FnMut(V) -> Vec<V>,
{
    use ::rand::Rng;

    let mut rng = ::rand::thread_rng();

    // Begin at a random node
    let vertices: Vec<_> = vertices.into_iter().collect();
    let root = rng.choose(&vertices).expect("no vertices!?").clone();

    let mut out_edges = |v: V| out_edges(v.clone()).into_iter().map(move |t| (v.clone(), t));

    let mut edge_queue: Vec<_> = out_edges(root.clone()).collect();
    let mut edges = vec![];
    let mut done: HashSet<V> = vec![root.clone()].into_iter().collect();

    'outer:
        loop {
        // Follow a random out edge to a new point.
        rng.shuffle(&mut edge_queue[..]);

        'skip:
            while let Some((from, to)) = edge_queue.pop() {
            if done.contains(&to) {
                continue 'skip;
            }

            edges.push((from, to.clone()));
            edge_queue.extend(out_edges(to.clone()));
            done.insert(to);

            continue 'outer;
        }
        break;
    }
    assert!(vertices.into_iter().collect::<HashSet<_>>() == done,
               "mismatch between given vertices and those found through out_edges");

    Tree { root, edges }
}

pub fn integrate_grid_random<M, E, F, G>(
    (n_x, n_y): (usize, usize),
    mut compute_meta: F,
    mut integrate: G,
) -> Result<Vec<f64>, E>
    where
        F: Fn(Point) -> Result<M, E> + Sync,
        M: Send, E: Send,
        G: FnMut((Point, &M), (Point, &M)) -> Result<f64, E>,
{Ok({
    use ::rayon::prelude::*;

    let vertices = (0..n_x).flat_map(|x| (0..n_y).map(move |y| (x, y))).collect::<Vec<_>>();
    let out_edges = |(x, y)| {
        let mut out = vec![];
        if 0 < x { out.push((x - 1, y)); }
        if 0 < y { out.push((x, y - 1)); }
        if x + 1 < n_x { out.push((x + 1, y)); }
        if y + 1 < n_y { out.push((x, y + 1)); }
        out
    };

    // randomly choose starting point and ancestors of each point,
    // with the expectation that this will reduce some forms of bias.
    let Tree { root, edges } = random_tree(vertices.clone(), out_edges);

    let index = |(x, y)| (y * n_x + x);
    let metas = vertices.par_iter()
        .map(|&v| compute_meta(v))
        .collect::<Result<Vec<_>, E>>()?;

    let mut values = vec![0./0.; n_x * n_y];
    values[index(root)] = 0.0;

    for (from, to) in edges {
        values[index(to)] = values[index(from)] + integrate(
            (from, &metas[index(from)]),
            (to, &metas[index(to)]),
        )?;
    }
    values
})}

fn linspace(r: Range<f64>, n: usize) -> Vec<f64>
{
    let out: Vec<_> = (0..n as u32)
        .map(|i| i as f64 / (n as f64 - 1f64))
        .map(|a| (1.0 - a) * r.start + a * r.end)
        .collect();

    assert_eq!(out[0], r.start);
    assert_eq!(*out.last().unwrap(), r.end);
    assert_eq!(out.len(), n);
    out
}

