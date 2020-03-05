use crate::euclidean_distance::EuclideanDistance;
use crate::random::Random;

#[derive(Debug)]
pub struct KMeans<T: EuclideanDistance + Default> {
    categories: Box<Vec<T>>,
    k: usize
}

impl<T: EuclideanDistance + Random + Default + Clone> KMeans<T> {
    pub fn new(k: usize) -> KMeans<T> {
        KMeans{categories: Box::new(vec![Default::default(); k]), k: k}
    }

    fn randomize_centroids(&mut self) {
        let cats = &mut self.categories;
        for cat in cats.iter_mut() {
            *cat = T::random();
        }
    }

    fn randomize_centroids_in_range(&mut self, lower: &T, upper: &T) {
        let cats = &mut self.categories;
        for cat in cats.iter_mut() {
            *cat = T::random_in_range(lower, upper);
        }
    }
}