


struct SmallCache<A, R, F> {
    max_len: usize,
    cache: ::std::collections::VecDeque<(A, R)>,
    func: F,
}

impl<A, R, F> SmallCache<A, R, F>
  where
    A: PartialEq + Clone,
    R: Clone,
    F: FnMut(&A) -> R,
{
    pub fn new(max_len: usize, func: F) -> Self
    {
        let cache = Default::default();
        SmallCache { max_len, func, cache }
    }

    pub fn evaluate(&mut self, arg: A) -> R {
        let ret = match self.lookup_index(&arg) {
            None => (&mut self.func)(&arg.clone()),
            Some(i) => self.cache.remove(i).unwrap().1,
        };
        self.cache.push_front((arg, ret.clone()));
        self.cache.truncate(self.max_len);
        ret
    }

    fn lookup_index(&self, want: &A) -> Option<usize> {
        self.cache.iter().enumerate()
            .find(|&(_, &(ref have, _))| want == have )
            .map(|(i, _)| i)
    }
}

// grad_max:     G -> V
// grad_norm:    G -> V
// potential:    V -> V
// experimental: (Option<(P, G, V)>, (P, G, V)) -> Either<V, S>
//
// all:  ((P -> (G, V)), Option<P>, P) -> Either<V, S>

// in metropolis: (Option<P>, P) -> Either<V, S>

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ObjectiveStyle { GradMax, Potential, Experimental }

struct Potential<F> {
    style: ObjectiveStyle,
    diff_fn: F,
}

impl<F> Potential<F>
where F: FnMut(&SPos) -> (f64, SGrad)
{
    fn compute(&mut self, prev: Option<&SPos>, cur: &SPos) -> ValueOrDelta {
        match self.style {
            ObjectiveStyle::GradMax => {
                let (_, g) = (&mut self.diff_fn)(cur);
                unimplemented!(); // TODO
            },
            ObjectiveStyle::Potential => { ValueOrDelta::Value((&mut self.diff_fn)(cur).0) },
            ObjectiveStyle::Experimental => {
                match prev {
                    None => ValueOrDelta::Value((&mut self.diff_fn)(cur).0),
                    Some(prev) => {
                        // TODO
                        ValueOrDelta::Delta(unimplemented!())
                    },
                }
            }
        }
    }
}

type V = f64;
type S = f64;
#[derive(Clone)] struct SMut; // structural mutation
#[derive(Clone)] struct SPos; // structural pos
#[derive(Clone)] struct SGrad;

fn structural<F>(
    settings: &Settings,
    initial_position: SPos,
    mut callbacks: Callbacks<SPos, SMut>,
    mut diff_fn: F,
) -> SPos
where F: FnMut(&SPos) -> (V, SGrad),
{
    let style = settings.objective;
    let mut pot = Potential { style, diff_fn, };
    metropolis(
        settings, initial_position, callbacks,
        |prev, cur| pot.compute(prev, cur),
    )
}

#[derive(Debug,Copy,Clone,PartialEq,PartialOrd)]
enum ValueOrDelta { Value(f64), Delta(f64) }

impl ValueOrDelta {
    pub fn value_from(self, from: f64) -> f64 { match self {
        ValueOrDelta::Value(v) => v,
        ValueOrDelta::Delta(d) => from + d,
    }}

    pub fn delta_from(self, from: f64) -> f64 { match self {
        ValueOrDelta::Value(v) => v - from,
        ValueOrDelta::Delta(d) => d,
    }}

    pub fn value_and_delta_from(self, from: f64) -> (f64, f64) {
        (self.value_from(from), self.delta_from(from))
    }
}

enum Never {}
struct Callbacks<P, D> {
    f: Box<FnMut(P,D) -> Never>,
}
impl<P, D> Callbacks<P,D> {
    fn generate(&mut self, _: &P) -> D { unimplemented!() }
    fn apply(&mut self, _: &P, _: &D) -> P { unimplemented!() }
    fn visit(&mut self, _: &P, _: f64, _: bool) { unimplemented!() }
    fn applied(&mut self, _: &P, _: &D, _: (f64, f64, f64)) { unimplemented!() }
}
struct Settings {
    objective: ObjectiveStyle,
    output_level: i32,
    improve_iteration_limit: i32,
    iteration_limit: i32,

}

fn metropolis<P, D, F>(
    settings: &Settings,
    initial_position: P,
    mut callbacks: Callbacks<P, D>,
    mut objective_fn: F,
) -> P
where F: FnMut(Option<&P>, &P) -> ValueOrDelta,
{
    let mut mutations = 0;        // total number of mutations tested
    let mut mutations_since = 0;  // number of mutations since last improvement

    // Initialize
    let mut position_cur = initial_position;
    let mut value_cur = objective_fn(None, &position_cur).value_from(0.0);

    callbacks.visit(&position_cur, value_cur, true);

    if settings.output_level > 1 {
        unimplemented!();
    }

    loop {
        let mutation = callbacks.generate(&position_cur);
        let position_new = callbacks.apply(&position_cur, &mutation);
        mutations += 1;
        mutations_since += 1;

        let (value_new, delta) =
            objective_fn(Some(&position_cur), &position_new)
            .value_and_delta_from(value_cur);

        let successful = delta < 0.0;

        callbacks.visit(&position_cur, value_new, successful);
        callbacks.applied(&position_cur, &mutation, (value_cur, delta, value_new));
        if successful {
            if settings.output_level > 1 {
                unimplemented!();
            }

            // keep modified position and value
            value_cur = value_new;
            position_cur = position_new;
            mutations_since = 0;
        }

        // for now, simply reject all inferior values
        // (this is Metropolis at 0K)

        // exit conditions
        let mut done = false;
        done |= settings.improve_iteration_limit > 0 &&
                mutations_since >= settings.improve_iteration_limit;

        done |= settings.iteration_limit > 0 &&
                mutations >= settings.iteration_limit;

        if done { break; }
    }

    return position_cur;
}
