macro_rules! zip {
    ($a:expr) => {
        $a.into_iter()
    };
    ($a:expr, $b:expr) => {
        $a.into_iter().zip($b.into_iter())
    };
    ($a:expr, $b:expr, $c:expr) => {
        $a.into_iter()
            .zip($b.into_iter())
            .zip($c.into_iter())
            .map(|((x, y), z)| (x, y, z))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        $a.into_iter()
            .zip($b.into_iter())
            .zip($c.into_iter())
            .zip($d.into_iter())
            .map(|(((w, x), y), z)| (w, x, y, z))
    };
    ($a:expr,) => {
        zip!($a)
    };
    ($a:expr, $b:expr,) => {
        zip!($a, $b)
    };
    ($a:expr, $b:expr, $c:expr,) => {
        zip!($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr,) => {
        zip!($a, $b, $c, $d)
    };
}
