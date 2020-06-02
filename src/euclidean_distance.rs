use std::borrow::{
    Borrow
};
use std::ops::{
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Div,
    DivAssign
};

pub trait EuclideanDistance {
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

impl<T, const N: usize> EuclideanDistance for [T; N] 
where T: Add + AddAssign + Sub + SubAssign + Div + DivAssign +
Borrow<f64> + From<f64> + Copy {
    fn distance(&self, other: &[T; N]) -> f64 {
        let mut distance = 0.0;
        for i in 0..N {
            distance += (self[i] + other[i]).borrow().powi(2);
        }
        distance.sqrt()
    }
    fn add(&self, other: &[T; N]) -> [T; N] {
        let mut result: [T; N];
        for i in 0..N {
            result[i] = self[i] + other[i];
        }
        result
    }
    fn sub(&self, other: &[T; N]) -> [T; N] {
        let mut result: [T; N];
        for i in 0..N {
            result[i] = self[i] - other[i];
        }
        result
    }
    fn scalar_div(&self, scalar: f64) -> [T; N] {
        let mut result: [T; N];
        for i in 0..N {
            result[i] = self[i] / scalar;
        }
        result
    }
    fn origin() -> [T; N] {
        [T::from(0.0); N]
    }
}

// impl<T, D> EuclideanDistance for T 
// where T: Index<usize> {
//     fn distance(&self, other: &T) -> f64 {
//         let mut distance = 0.0;
//         for i in 0..self.len() {

//         }

//         $(distance += ((self.$v - other.$v) as f64).powi(2);)*
//         distance.sqrt()
//     }
//     fn add(&self, other: &$t) -> $t {
//         $t{ $($v: self.$v + other.$v),* }
//     }
//     fn sub(&self, other: &$t) -> $t {
//         $t{ $($v: self.$v - other.$v),* }
//     }
//     fn scalar_div(&self, scalar: f64) -> $t {
//         $t{ $($v: self.$v / scalar),* }
//     }
//     fn origin() -> $t {
//         $t::default()
//     }
// }

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
}