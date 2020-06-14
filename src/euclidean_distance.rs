pub trait EuclideanDistance<Dummy = Self> {
    // get the distance between this point and the other
    fn distance(&self, other: &Self) -> f64;

    // add the other value to this value
    fn add(&self, other: &Self) -> Self;

    // subtract another value from this value
    fn sub(&self, other: &Self) -> Self;

    // divide this value by a scalar
    fn scalar_div(&self, scalar: f64) -> Self;

    // get the origin for this euclidean space
    fn origin() -> Self;
}

// implements euclidean distance for either named types w/ var list
// or for arrays of arbitrary length
#[macro_export]
macro_rules! impl_euclidean_distance {
    // for use with formal types
    ( $t:ident: $($v:ident),+ ) => {
        impl EuclideanDistance for $t {
            fn distance(&self, other: &$t) -> f64 {
                let mut distance = 0.0;
                $(distance += ((self.$v - other.$v) as f64).powi(2);)*
                distance.sqrt()
            }
            fn add(&self, other: &$t) -> $t {
                $t{ $($v: self.$v + other.$v),* }
            }
            fn sub(&self, other: &$t) -> $t {
                $t{ $($v: self.$v - other.$v),* }
            }
            fn scalar_div(&self, scalar: f64) -> $t {
                $t{ $($v: self.$v / scalar),* }
            }
            fn origin() -> $t {
                $t::default()
            }
        }
    };

    // for use with variable length arrays
    ( $wrapper:ident[$t:ident; $size:expr] ) => {
        impl EuclideanDistance<$wrapper> for [$t; $size] {
            fn distance(&self, other: &[$t; $size]) -> f64 {
                let mut distance = 0.0;
                for i in 0..$size {
                    distance += (self[i] as f64 - other[i] as f64).powi(2);
                }
                distance.sqrt()
            }
            fn add(&self, other: &[$t; $size]) -> [$t; $size] {
                let mut result = [$t::default(); $size];
                for i in 0..$size {
                    result[i] = self[i] + other[i];
                }
                result
            }
            fn sub(&self, other: &[$t; $size]) -> [$t; $size] {
                let mut result = [$t::default(); $size];
                for i in 0..$size {
                    result[i] = self[i] - other[i];
                }
                result
            }
            fn scalar_div(&self, scalar: f64) -> [$t; $size] {
                let mut result = [$t::default(); $size];
                for i in 0..$size {
                    result[i] = self[i] as f64 / scalar;
                }
                result
            }
            fn origin() -> [$t; $size] {
                [$t::default(); $size]
            }
        }
    };
}