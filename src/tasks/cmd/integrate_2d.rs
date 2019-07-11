/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

pub type Point = (usize, usize);
use std::ops::Range;
use std::hash::Hash;
use std::collections::HashSet;

use slice_of_array::prelude::*;
use rsp2_slice_math::{v, V, vdot};
use rsp2_array_types::{V3};

// Note: directions do not need to be normalized.
// (The ranges are in units of these norms)
pub fn integrate_two_directions<E, F>(
    // dims must be greater than 1
    dims: [usize; 2],
    init_pos: &[V3],
    ranges: [Range<f64>; 2],
    extend_borders: [bool; 2],
    directions: [&[V3]; 2],
    mut compute_grad: F,
) -> Result<TwoDeeIntegrated, E>
where
    F: FnMut(&[V3]) -> Result<Vec<V3>, E>,
{
    let (ixs, xs) = linspace(ranges[0].clone(), dims[0], extend_borders[0]);
    let (iys, ys) = linspace(ranges[1].clone(), dims[1], extend_borders[1]);

    let indices = iproduct!(ixs, iys).map(|(ix, iy)| [ix, iy]).collect();
    let coords = iproduct!(&xs, &ys).map(|(&x, &y)| [x, y]).collect();

    let values = integrate_grid_random(
        (dims[0], dims[1]),
        |(x, y)| {Ok::<_,E>({
            let V(pos) = v(init_pos.flat())
                + xs[x] * v(directions[0].flat())
                + ys[y] * v(directions[1].flat());

            let pos = pos.to_vec();
            let grad = compute_grad(pos.nest())?.flat().to_vec();
            (pos, grad)
        })},

        |(_, (pos1, grad1)), (_, (pos2, grad2))| {Ok({
            // trapezoid
            let V(grad) = 0.5 * (v(grad1) + v(grad2));
            let V(d_pos) = v(pos2) - v(pos1);
            vdot(&grad[..], &d_pos)
        })},
    )?;

    Ok(TwoDeeIntegrated { indices, values, coords })
}

#[derive(Debug, Clone)]
pub struct TwoDeeIntegrated {
    pub indices: Vec<[i32; 2]>,
    pub coords: Vec<[f64; 2]>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Tree<V> {
    /// Which node is the root?
    root: V,
    /// Tree edges `(source, target)`. Every node except the root will appear exactly once as the
    /// `target`, and it will always appear as a `target` before it ever appears as a `source`.
    edges: Vec<(V, V)>,
}

fn random_tree<V>(
    vertices: impl IntoIterator<Item=V>,
    mut out_edges: impl FnMut(V) -> Vec<V>,
) -> Tree<V>
where V: Clone + Hash + Eq,
{
    use rand::Rng;

    let mut rng = rand::thread_rng();

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

pub fn integrate_grid_random<M, E>(
    (n_x, n_y): (usize, usize),
    mut compute_meta: impl FnMut(Point) -> Result<M, E>,
    mut integrate: impl FnMut((Point, &M), (Point, &M)) -> Result<f64, E>,
) -> Result<Vec<f64>, E>
{Ok({
    let vertices = iproduct!(0..n_x, 0..n_y).collect::<Vec<_>>();
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
    let metas = vertices.iter()
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

fn linspace(r: Range<f64>, n: usize, extend_borders: bool) -> (Vec<i32>, Vec<f64>)
{
    assert!(n > 1, "cannot perform linspace with n < 2");

    let (mut indices, mut values): (Vec<i32>, Vec<f64>) = (0..n as i32)
        .map(|i| (i, i as f64 / (n as f64 - 1f64)))
        .map(|(i, a)| (i, (1.0 - a) * r.start + a * r.end))
        .unzip();

    assert_eq!(values[0], r.start);
    assert_eq!(*values.last().unwrap(), r.end);
    assert_eq!(values.len(), n);

    if extend_borders {
        let step = values[1] - values[0];
        values.push(r.end + step);
        indices.push(n as i32);
        values.insert(0, r.start - step);
        indices.insert(0, -1);
    }

    (indices, values)
}

