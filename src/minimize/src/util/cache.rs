//! Various helper types for remembering output from expensive functions.

/// A map that only remembers the last inserted pair.
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub(crate) struct LastCache<K, V>(Option<(K, V)>);

impl<K, V> LastCache<K, V>
where K: PartialEq,
{
    pub(crate) fn new() -> Self { LastCache(None) }
    pub(crate) fn put(&mut self, key: K, value: V) { self.0 = Some((key, value)) }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.0.as_ref()
            .and_then(|&(ref k, ref v)| if k == key { Some(v) } else { None })
    }
    // pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    /// Consumes self to look up a value.
    pub(crate) fn get_consume(self, key: &K) -> Option<V> {
        self.0.and_then(|(k,v)| if &k == key { Some(v) } else { None })
    }
    // pub(crate) fn assert_get(&self, key: &K) -> &V;
    // pub(crate) fn assert_get_mut(&mut self, key: &K) -> &mut V;

    pub(crate) fn as_option(&self) -> Option<(&K, &V)> {
        self.0.as_ref().map(|&(ref a, ref b)| (a,b))
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

impl<M, K, V, F> MinCacheBy<M, K, V, F>
where
    K: PartialEq,
    M: PartialOrd,
    F: FnMut(&K,&V) -> M,
{
    pub(crate) fn new(projection: F) -> Self { MinCacheBy { best: LastCache::new(), projection } }
    pub(crate) fn put(&mut self, key: K, value: V) {
        let objective = (&mut self.projection)(&key, &value);
        if let Some((_, &(ref m, _))) = self.best.as_option() {
            if m < &objective { return; }
        }
        self.best.put(key, (objective, value))
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> { self.best.get(key).map(|t| &t.1) }
    pub(crate) fn get_consume(self, key: &K) -> Option<V> { self.best.get_consume(key).map(|t| t.1) }
}