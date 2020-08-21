use rand::Rng;
use std::f64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt::Debug;

use crate::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

const LLOYD_LIMIT: usize = 100;

#[derive(Clone, Debug)]
enum Initiliser {
    Default,
    PP,
    Scalable
}

#[derive(Clone, Debug)]
pub struct KMeans<T: EuclideanDistance + PartialEq + Clone> {
    categories: Option<Box<Vec<T>>>,
    k: usize,
    trainer: Initiliser
}

impl<T> KMeans<T> 
where T: EuclideanDistance + PartialEq + Clone {
    // get a regular kmeans object
    pub fn new(k: usize) -> KMeans<T> {
        KMeans{categories: None, k: k, trainer: Initiliser::Default}
    }

    // get a kmeans++ object
    pub fn new_pp(k: usize) -> KMeans<T> {
        KMeans{categories: None, k: k, trainer: Initiliser::PP}
    }

    // get a kmeans parallel object
    pub fn new_scalable(k: usize) -> KMeans<T> {
        KMeans{categories: None, k: k, trainer: Initiliser::Scalable}
    }

    // returns the points that represent the centroids
    pub fn categories(&self) -> Result<&Vec<T>, ClassificationError> {
        match &self.categories {
            Some(v) => Ok(v.as_ref()),
            None => Err(ClassificationError::ClassifierNotTrained)
        }
    }

    // private methods for use during training

    // trains the classifier with the correct method
    fn initialise_with_appropriate_method(&mut self, samples: &mut Vec<(T, f64)>) {
        match self.trainer {
            Initiliser::Default => self.get_random_centroids(samples),
            Initiliser::PP => self.get_weighted_random_centroids(samples),
            Initiliser::Scalable => self.get_centoids_by_oversampling(samples)
        }
    }
    
    // gets initial centroids completely at random
    fn get_random_centroids(&mut self, samples: &mut Vec<(T, f64)>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }
        
        // reset the previous training, if any
        self.categories = Some(Box::new(Vec::new()));

        // select some random values (this will not work
        // if there are more categories than samples)
        let mut generator = rand::thread_rng();
        let in_range = samples.len();
        let mut selection = &samples[generator.gen_range(0, in_range)].0;
        let categories = self.categories.as_mut().unwrap();
        for _ in 0..self.k {
            // don't select the same category value twice
            while categories.contains(&selection) {
                selection = &samples[generator.gen_range(0, in_range)].0;
            }

            categories.push(selection.clone());
        }
    }

    fn sum_of_weights_squared(
        samples: &Vec<(T, f64)>
    ) -> f64 {
        let mut total_weight = 0.0;
        for (_, dist) in samples.iter() {
            if dist.is_nan() {
                continue;
            }
            total_weight += *dist;
        }
        total_weight
    }

    fn update_weights(
        samples: &mut Vec<(T, f64)>,
        newest_cat: &T
    ) {
        // weight the next set of values
        for (val, dist) in samples.iter_mut() {
            if dist.is_nan() {
                continue;
            }
            let distance = newest_cat.distance(val);
            if distance < *dist {
                *dist = distance;
            }
        }
    }

    // for use with plus plus and parallel selection algorithms
    // calculates the new smallest distance of the samples to the
    // thus-far-selected points and returns the sum of the weights
    fn calculate_weights(
        samples: &mut Vec<(T, f64)>,
        newest_cat: &T
    ) -> f64 {
        // weight the next set of values
        let mut weight_total = 0.0;
        for (val, dist) in samples.iter_mut() {
            if dist.is_nan() {
                continue;
            }
            let distance = newest_cat.distance(val);
            if distance < *dist {
                *dist = distance;
            }
            weight_total += dist.powi(2);
        }
        weight_total
    }

    // gets initial centroids by means of KMeans++
    fn get_weighted_random_centroids(&mut self, samples: &mut Vec<(T, f64)>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }

        // reset the previous training, if any
        self.categories = Some(Box::new(Vec::new()));

        // select one random value initially
        let mut generator = rand::thread_rng();
        let mut selection = {
            let index = generator.gen_range(0, samples.len());
            samples[index].1 = f64::NAN;
            samples[index].0.clone()
        };
        let categories = self.categories.as_mut().unwrap();

        // continue to select values based on weighted probablity
        categories.push(selection.clone());
        for _ in 0..(self.k - 1) {
            let weight_total = Self::calculate_weights(
                samples,
                &selection
            );

            // now select one at random
            let mut remainder = generator.gen_range(0.0, weight_total);
            for (val, weight) in samples.iter_mut() {
                if weight.is_nan() {
                    continue;
                }
                remainder -= weight.powi(2);
                if remainder <= 0.0 {
                    *weight = f64::NAN;
                    selection = val.clone();
                    break;
                }

            }
            categories.push(selection.clone());
        }
    }

    // gets initial centroids by oversampling (http://arxiv.org/abs/1203.6402)
    fn get_centoids_by_oversampling(&mut self, samples: &mut Vec<(T, f64)>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }

        // select one random value initially
        let mut generator = rand::thread_rng();
        let mut sampled = Vec::new();
        let initial = &samples[generator.gen_range(0, samples.len())].0.clone();

        // get initial weight to determine num. iterations
        let mut weight_total = Self::calculate_weights(
            samples,
            initial
        );

        // oversample centres (achieve approx. klog(initial_weight))
        for _ in 0..(weight_total as f64).log10() as usize {
            let mut selections = Vec::new();

            // sample independantly proportional to distance from a centre
            for (val, weight) in samples.iter_mut() {
                if weight.is_nan() {
                    continue;
                }
                let prob = generator.gen_range(0.0, 1.0);
                let select = (weight.powi(2) / weight_total as f64) * self.k as f64;
                if prob < select {
                    selections.push((val.clone(), f64::MAX));
                    *weight = f64::NAN;
                }
            }

            // update the smallest distance to any point
            for selection in selections.iter() {
                Self::update_weights(
                    samples, 
                    &selection.0
                );
            }
            weight_total = Self::sum_of_weights_squared(samples);
            sampled.append(&mut selections);
        }

        // find the real initial centroids using kmeans++
        self.get_weighted_random_centroids(&mut sampled);
    }

    fn lloyds_iteration(&mut self, data: &Vec<(T, f64)>) -> Result<(), TrainingError> {

        // until the centroids don't change, keep changing the centroids
        let mut sum_pairs = vec!((T::origin(), 0); self.k);
        let mut i = 0;
        for _ in (0..LLOYD_LIMIT) {
            // classify the data with the current centroids, then update the centroids
            self.sum_categories(data, &mut sum_pairs)?;
            if !self.update_categories(&sum_pairs)? {
                break;
            }
            i += 1;
        }
        Ok(())
    }

    fn categorise(&self, datum: &T) -> Result<usize, TrainingError> {
        let mut result = Err(TrainingError::InvalidClassifier);
        let mut distance = f64::MAX;
        let categories = match &self.categories {
            Some(c) => c,
            None => return result
        };
        for (i, cat) in categories.iter().enumerate() {
            let cur_dist = cat.distance(datum);
            if cur_dist < distance {
                distance = cur_dist;
                result = Ok(i);
            }
        }
        return result;
    }

    fn categorise_all(&self, data: &Vec<T>) -> Result<Vec<usize>, TrainingError> {
        let mut cats = Vec::new();
        for datum in data.iter() {
            cats.push(self.categorise(datum)?);
        }
        return Ok(cats);
    }

    fn sum_categories(
        &self,
        samples: &Vec<(T, f64)>,
        sum_pairs: &mut Vec<(T, usize)>
    ) -> Result<(), TrainingError> {
        for (val, _) in samples.iter() {
            let cur = &mut sum_pairs[self.categorise(val)?];
            cur.0 = cur.0.add(val);
            cur.1 += 1;
        }
        Ok(())
    }

    fn update_categories(&mut self, sum_pairs: &Vec<(T, usize)>) 
    -> Result<bool, TrainingError> {
        if self.categories == None {
            return Err(TrainingError::InvalidClassifier);
        }
        let categories = self.categories.as_mut().unwrap();
        if sum_pairs.len() != categories.len() {
            return Err(TrainingError::InvalidData);
        }

        let mut updated = false;
        for (i, (sum, count)) in sum_pairs.iter().enumerate() {
            let new_cat = sum.scalar_div(*count as f64);
            updated |= categories[i] != new_cat;
            categories[i] = new_cat;
        }
        Ok(updated)
    }

    // updates the centroids with the current classifications,
    // returns true if the centoids changed
    fn update_centroids(&mut self, data: &Vec<T>, cats: &Vec<usize>) -> Result<bool, TrainingError> {
        if data.len() != cats.len() {
            return Err(TrainingError::InvalidData);
        } else if self.categories == None {
            return Err(TrainingError::InvalidClassifier);
        }

        // take an average value from the current categorisation
        let mut updated = false;
        let mut sums = vec![T::origin(); self.k];
        let mut counts = vec![0; self.k];
        let categories = self.categories.as_mut().unwrap();
        for (i, cat) in cats.iter().enumerate() {
            sums[*cat] = sums[*cat].add(&data[i]);
            counts[*cat] += 1;
        }

        // update the old value, and keep track of it if changed
        for (i, sum) in sums.iter_mut().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            let new_val = sum.scalar_div(counts[i] as f64);
            if new_val != categories[i] {
                updated = true;
            }
            categories[i] = new_val;
        }
        return Ok(updated);
    }
}

impl<T> UnsupervisedClassifier<T> for KMeans<T>
where T: EuclideanDistance + PartialEq + Clone {
    fn train(&mut self, pre_data: Vec<T>) -> Result<Vec<T>, TrainingError> {
        // initialise the centroids randomly, initially
        let mut data: Vec<(T, f64)> = pre_data
            .into_iter().map(|v| (v, f64::MAX)).collect();
        self.initialise_with_appropriate_method(&mut data);
        self.lloyds_iteration(&mut data)?;
        return Ok(self.categories.as_ref().unwrap().clone().to_vec());
    }

    fn train_from_file(
        &mut self, 
        file: &mut File, 
        parser: &dyn Fn(&Vec<u8>) -> Result<T, TrainingError>
    ) -> Result<Vec<T>, TrainingError> {
        let mut data = Vec::new();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let buf = match line {
                Ok(l) => l.into_bytes(),
                Err(_) => return Err(TrainingError::FileReadFailed),
            };
            let val = parser(&buf)?;
            data.push(val);
        }
        return self.train(data);
    }

    fn classify(&self, datum: &T) -> Result<usize, ClassificationError> {
        if self.categories == None {
            return Err(ClassificationError::ClassifierInvalid);
        }
        let result = self.categorise(datum);
        match result {
            Ok(i) => Ok(i),
            Err(_) => Err(ClassificationError::ClassifierInvalid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_datatypes::point::Point;
    
    fn get_some_classifier() -> KMeans<Point> {
        KMeans{categories: Some(Box::new(vec![
            Point::new(0.0, 1.0),
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0)
        ])), k: 3, trainer: Initiliser::Default}
    }

    fn get_some_data_points() -> Vec<Point> {
        vec!(
            Point::new(0.0, 2.0), 
            Point::new(-2.0, -2.0), 
            Point::new(10.0, -3.0)
        )
    }

    fn get_more_data_points() -> Vec<Point> {
        let mut v = get_some_data_points();
        v.push(Point::new(-4.0, -4.0));
        v.push(Point::new(12.0, -5.0));
        v.push(Point::new(0.0, 1.0));
        v
    }

    // tests for private methods
    #[test]
    fn test_categorise() {
        // make a custom classifier
        let c = get_some_classifier();
        let data = get_some_data_points();
        assert_eq!(c.categorise(&data[0]), Ok(0));
        assert_eq!(c.categorise(&data[1]), Ok(1));
        assert_eq!(c.categorise(&data[2]), Ok(2));

        let c = KMeans::<Point>{categories: None, k: 0, trainer: Initiliser::Default};
        assert_eq!(c.categorise(&data[0]), Err(TrainingError::InvalidClassifier));

        let c = KMeans::<Point>::new(0);
        assert_eq!(c.categorise(&data[0]), Err(TrainingError::InvalidClassifier));
    }

    #[test]
    fn test_categorise_all() {
        let c = get_some_classifier();
        let data = get_some_data_points();
        assert_eq!(c.categorise_all(&data), Ok(vec!(0, 1, 2)));

        let c = KMeans::<Point>{categories: None, k: 0, trainer: Initiliser::Default};
        assert_eq!(c.categorise_all(&data), Err(TrainingError::InvalidClassifier));

        let c = KMeans::<Point>::new(0);
        assert_eq!(c.categorise_all(&data), Err(TrainingError::InvalidClassifier));
    }

    #[test]
    fn test_update_centroids() {
        let mut c = get_some_classifier();
        let data = get_more_data_points();
        let cats = vec!(0, 1, 2, 1, 2, 0);
        assert_eq!(c.categorise_all(&data), Ok(cats.clone()));

        // test the centroid update requires 2 passes for this data
        let avgs = vec!(
            Point::new(0.0, 1.5),
            Point::new(-3.0, -3.0),
            Point::new(11.0, -4.0)
        );
        assert_eq!(c.update_centroids(&data, &cats), Ok(true));
        assert_eq!(c.categories.as_ref().unwrap(), &Box::new(avgs.clone()));

        // second pass, centoids shouldn't change
        assert_eq!(c.categorise_all(&data), Ok(cats.clone()));
        assert_eq!(c.update_centroids(&data, &cats), Ok(false));
        assert_eq!(c.categories.as_ref().unwrap(), &Box::new(avgs.clone()));
    }
}