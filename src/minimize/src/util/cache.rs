//! Various helper types for remembering output from expensive functions.

/// A map that only remembers the last inserted pair.
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub(crate) struct LastCache<K, V>(Option<(K, V)>);

#[allow(dead_code)]
impl<K, V> LastCache<K, V>
where K: PartialEq,
{
    pub(crate) fn new() -> Self { LastCache(None) }
    pub(crate) fn put(&mut self, key: K, value: V) { self.0 = Some((key, value)) }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.0.as_ref()
            .and_then(|(k, v)| if k == key { Some(v) } else { None })
    }
    // pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    /// Consumes self to look up a value.
    pub(crate) fn get_consume(self, key: &K) -> Option<V> {
        self.0.and_then(|(k, v)| if &k == key { Some(v) } else { None })
    }
    // pub(crate) fn assert_get(&self, key: &K) -> &V;
    // pub(crate) fn assert_get_mut(&mut self, key: &K) -> &mut V;

    pub(crate) fn as_option(&self) -> Option<(&K, &V)> {
        self.0.as_ref().map(|(a, b)| (a, b))
    }
    // pub(crate) fn as_option_mut(&mut self) -> Option<(&mut V, &mut K)>;
    pub(crate) fn into_option(self) -> Option<(K, V)> { self.0 }
}

/// A map that only saves the pair with minimum value according to some projection.
///
/// The API is similar to LastCache, the only real difference being which (key, value)
/// pair the map retains.
//
// NOTE: the projection is to avoid `put` having both `key` and `objective`
//       arguments which could easily get mixed up.
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub(crate) struct MinCacheBy<M, K, V, F> {
    best: LastCache<K, (M, V)>,
    projection: F,
}

#[allow(dead_code)]
impl<M, K, V, F> MinCacheBy<M, K, V, F>
where
    K: PartialEq,
    M: PartialOrd,
    F: FnMut(&K,&V) -> M,
{
    pub(crate) fn new(projection: F) -> Self { MinCacheBy { best: LastCache::new(), projection } }
    pub(crate) fn put(&mut self, key: K, value: V) {
        let objective = (&mut self.projection)(&key, &value);
        if let Some((_, (m, _))) = self.best.as_option() {
            if m < &objective { return; }
        }
        self.best.put(key, (objective, value))
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> { self.best.get(key).map(|t| &t.1) }
    pub(crate) fn get_consume(self, key: &K) -> Option<V> { self.best.get_consume(key).map(|t| t.1) }
}

#[allow(dead_code)]
/// Caches all input/output pairs of a function in a hashmap
/// using the given key function to produce hashable keys.
///
/// `key_func` must be an isomorphism (at least as far as
/// `compute` is concerned). That is, if two inputs produce
/// the same key, then they must both `compute` to the same
/// output.
pub fn hash_memoize_by_key<'a, Key, In, Out, KeyFunc, F>(
    mut key_func: KeyFunc,
    mut compute: F,
) -> Box<FnMut(In) -> Out + 'a>
where
    Out: Clone + 'a,
    Key: ::std::hash::Hash + Eq + 'a,
    KeyFunc: FnMut(&In) -> Key + 'a,
    F: FnMut(In) -> Out + 'a,
{
    let mut cache = ::std::collections::HashMap::new();
    Box::new(move |input|
        cache.entry(key_func(&input))
            .or_insert_with(|| compute(input))
            .clone())
}

/// Caches all input/output pairs of a fallible function in a hashmap
/// using the given key function to produce hashable keys.
///
/// `key_func` must be an isomorphism (at least as far as
/// `compute` is concerned). That is, if two inputs produce
/// the same key, then they must both `compute` to the same
/// output.
///
/// The advantage over 'hash_memoize_by_key' is that it
/// does not require `E: Clone`.
pub fn hash_memoize_result_by_key<'a, Key, In, Out, KeyFunc, E, F>(
    mut key_func: KeyFunc,
    mut compute: F,
) -> Box<FnMut(In) -> Result<Out, E> + 'a>
where
    Out: Clone + 'a,
    Key: ::std::hash::Hash + Eq + 'a,
    KeyFunc: FnMut(&In) -> Key + 'a,
    F: FnMut(In) -> Result<Out, E> + 'a,
{
    use ::std::collections::hash_map::{HashMap, Entry};

    let mut cache = HashMap::new();
    Box::new(move |input| -> Result<Out, E> {
        // explicit match instead of 'or_insert_with' to ensure
        // the Err gets propogated to the right place
        match cache.entry(key_func(&input)) {
            // not sure why, but type inference needs a little help here,
            // hence the 'as' cast
            Entry::Occupied(e) =>  Ok((e.get() as &Out).clone()),
            Entry::Vacant(e) => Ok(e.insert(compute(input)?).clone()),
        }
    })
}
