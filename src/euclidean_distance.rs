pub trait EuclideanDistance {
    // get the distance between this point and the other
    fn distance(&self, other: &Self) -> f64;

    // get the order of this element and the other
    // returns -1, 0, 1 if this item is smaller than, equal to, or larger
    // than the other, respctively
    // fn compare(&self, other: &Self) -> i8;
}