
pub type Point = (usize, usize);
use ::std::ops::Range;

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
    F: FnMut(&[[f64; 3]]) -> Result<Vec<[f64; 3]>, E>,
{
    let xs = linspace(ranges.0, dims.0);
    let ys = linspace(ranges.1, dims.1);

    integrate_grid_random(
        dims,
        |(x, y)| {Ok::<_,E>({
            let V(pos) = v(init_pos.flat())
                + xs[x] * v(eigenvecs.0.flat())
                + ys[y] * v(eigenvecs.1.flat());

            let pos = pos.to_vec();
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

// randomly choose starting point and ancestors of each point,
// with the expectation that this will reduce some forms of bias.
pub fn integrate_grid_random<M, E, F, G>(
    (n_x, n_y): (usize, usize),
    mut compute_meta: F,
    mut integrate: G,
) -> Result<Vec<f64>, E>
where
    F: FnMut(Point) -> Result<M, E>,
    G: FnMut((Point, &M), (Point, &M)) -> Result<f64, E>,
{Ok({
    use ::rand::Rng;

    let mut rng = ::rand::thread_rng();

    let index = |(x, y)| (y * n_x + x);
    let out_edges = |(x, y)| {
        let mut out = vec![];
        if 0 < x { out.push(((x, y), (x - 1, y))); }
        if 0 < y { out.push(((x, y), (x, y - 1))); }
        if x + 1 < n_x { out.push(((x, y), (x + 1, y))); }
        if y + 1 < n_y { out.push(((x, y), (x, y + 1))); }
        out
    };

    let mut metas: Vec<_> = (0..n_x * n_y).map(|_| None).collect(); // no Clone
    let mut values = vec![0./0.; n_x * n_y];
    let mut edge_queue = vec![];

    {
        // Randomly choose zero point.
        let x = rng.gen_range(0, n_x);
        let y = rng.gen_range(0, n_y);
        metas[index((x, y))] = Some(compute_meta((x, y))?);
        values[index((x, y))] = 0.0;
        edge_queue.extend(out_edges((x, y)))
    }

    'outer:
    loop {
        // Follow a random out edge to a new point.
        rng.shuffle(&mut edge_queue[..]);

        'skip:
        while let Some((from, to)) = edge_queue.pop() {
            if metas[index(to)].is_some() {
                continue 'skip;
            }

            metas[index(to)] = Some(compute_meta(to)?);
            values[index(to)] = values[index(from)] + integrate(
                (from, metas[index(from)].as_ref().unwrap()),
                (to, metas[index(to)].as_ref().unwrap()),
            )?;
            edge_queue.extend(out_edges(to));

            continue 'outer;
        }
        break;
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

