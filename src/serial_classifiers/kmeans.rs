use rand::Rng;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::io::{BufRead, BufReader};

use crate::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

#[derive(Debug)]
enum Initiliser {
    Default,
    PP
}

#[derive(Debug)]
pub struct KMeans<D, T: EuclideanDistance<D> + PartialEq + Clone> {
    categories: Option<Box<Vec<T>>>,
    k: usize,
    trainer: Initiliser,
    phantom: Option<PhantomData<D>>
}

impl<D, T> KMeans<D, T> 
where T: EuclideanDistance<D> + PartialEq + Clone {
    // get a regular kmeans object
    pub fn new(k: usize) -> KMeans<D, T> {
        KMeans{categories: None, k: k, trainer: Initiliser::Default, phantom: None}
    }

    // get a kmeans++ object
    pub fn new_pp(k: usize) -> KMeans<D, T> {
        KMeans{categories: None, k: k, trainer: Initiliser::PP, phantom: None}
    }

    // private methods for use during training

    // trains the classifier with the correct method
    fn initialise_with_appropriate_method(&mut self, samples: &Vec<T>) {
        match self.trainer {
            Initiliser::Default => self.get_random_centroids(samples),
            Initiliser::PP => self.get_weighted_random_centroids(samples)
        }
    }
    
    // gets initial centroids completely at random
    fn get_random_centroids(&mut self, samples: &Vec<T>) {
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
        let mut selection = &samples[generator.gen_range(0, in_range)];
        let categories = self.categories.as_mut().unwrap();
        for _ in 0..self.k {
            // don't select the same category value twice
            while categories.contains(&selection) {
                selection = &samples[generator.gen_range(0, in_range)];
            }

            categories.push(selection.clone());
        }
    }

    // gets initial centroids by means of KMeans++
    fn get_weighted_random_centroids(&mut self, samples: &Vec<T>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }

        // reset the previous training, if any
        self.categories = Some(Box::new(Vec::new()));

        // select one random value initially
        let mut taken = Vec::new();
        let mut generator = rand::thread_rng();
        let mut prob_list = HashMap::<usize, f64>::new();
        let mut selection: usize = generator.gen_range(0, samples.len());
        let categories = self.categories.as_mut().unwrap();

        // continue to select values based on weighted probablity
        categories.push(samples[selection].clone());
        taken.push(selection);
        for _ in 0..(self.k - 1) {
            // weight the next set of values
            let mut weight_total = 0;
            for (i, value) in samples.iter().enumerate() {
                if taken.contains(&i) {
                    continue;
                }
                let distance = categories[categories.len() - 1].distance(value);

                // use the closest of the currently selected centroids
                if let Some(v) = prob_list.get_mut(&i) {
                    if distance < *v {
                        *v = distance;
                    }
                } else {
                    prob_list.insert(i, distance);
                }
                weight_total += prob_list[&i].powi(2) as usize;
            }

            // now select one at random
            let mut remainder = generator.gen_range(0, weight_total) as f64;
            for k in prob_list.keys() {
                remainder -= prob_list[k].powi(2);
                if remainder <= 0.0 {
                    selection = *k;
                    break;
                }
            }
            prob_list.remove(&selection);
            categories.push(samples[selection].clone());
            taken.push(selection);
        }
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

impl<D, T> UnsupervisedClassifier<T> for KMeans<D, T>
where T: EuclideanDistance<D> + PartialEq + Clone {
    fn train(&mut self, data: &Vec<T>) -> Result<Vec<T>, TrainingError> {
        // initialise the centroids randomly, initially
        self.initialise_with_appropriate_method(data);

        // until the centroids don't change, keep changing the centroids
        let mut categories: Vec<usize>;
        loop {
            
            // classify the data with the current centroids, then update the centroids
            match self.categorise_all(data) {
                Ok(cats) => categories = cats,
                Err(e) => return Err(e),
            }
            match self.update_centroids(data, &categories) {
                Ok(updated) => {
                    if !updated {break;}
                },
                Err(e) => return Err(e),
            }
        }
        return Ok(self.categories.as_ref().unwrap().clone().to_vec());
    }

    fn train_from_file(
        &mut self, 
        file: &mut File, 
        parser: &Fn(&Vec<u8>) -> Result<T, TrainingError>
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
        return self.train(&data);
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
    
    fn get_some_classifier() -> KMeans<Point, Point> {
        KMeans{categories: Some(Box::new(vec![
            Point::new(0.0, 1.0),
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0)
        ])), k: 3, trainer: Initiliser::Default, phantom: None}
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

        let c = KMeans::<Point, Point>{categories: None, k: 0, trainer: Initiliser::Default, phantom: None};
        assert_eq!(c.categorise(&data[0]), Err(TrainingError::InvalidClassifier));

        let c = KMeans::<Point, Point>::new(0);
        assert_eq!(c.categorise(&data[0]), Err(TrainingError::InvalidClassifier));
    }

    #[test]
    fn test_categorise_all() {
        let c = get_some_classifier();
        let data = get_some_data_points();
        assert_eq!(c.categorise_all(&data), Ok(vec!(0, 1, 2)));

        let c = KMeans::<Point, Point>{categories: None, k: 0, trainer: Initiliser::Default, phantom: None};
        assert_eq!(c.categorise_all(&data), Err(TrainingError::InvalidClassifier));

        let c = KMeans::<Point, Point>::new(0);
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