use crate::euclidean_distance::EuclideanDistance;
use crate::random::Random;
use rand::Rng;
use std::fs::File;
use std::f64;
use std::io::{BufRead, BufReader};

use crate::classifier::{
    UnsupervisedClassifier,
    TrainingError
};

#[derive(Debug)]
pub struct KMeans<T: EuclideanDistance + PartialEq + Default + Clone> {
    categories: Box<Vec<T>>,
    k: usize
}

impl<T: EuclideanDistance + PartialEq + Default + Clone> KMeans<T> {
    pub fn new(k: usize) -> KMeans<T> {
        KMeans{categories: Box::new(vec![Default::default(); k]), k: k}
    }

    // private methods for use during training
    
    // gets initial centroids completely at random
    fn get_random_centroids(&mut self, samples: &Vec<T>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.categories.len() {
            return;
        }

        // select some random values (this will not work
        // if there are more categories than samples)
        let mut taken = Vec::new();
        let mut generator = rand::thread_rng();
        let in_range = samples.len();
        let mut selection: usize = generator.gen_range(0, in_range);
        for cat in self.categories.iter_mut() {
            // don't select the same category value twice
            while taken.contains(&selection) {
                selection = generator.gen_range(0, in_range);
            }
            println!("Selection: {}", selection);
            *cat = samples[selection].clone();
            taken.push(selection);
        }
    }

    // gets initial centroids by means of KMeans++
    fn get_weighted_random_centroids(&mut self, samples: &Vec<T>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.categories.len() {
            return;
        }

        // select some random value initially
        let mut taken = Vec::new();
        let mut generator = rand::thread_rng();
        let mut prob_list;
        let mut selection: usize = generator.gen_range(0, samples.len());

        // continue to select values based on weighted probablity
        for cat in self.categories.iter_mut() {
            *cat = samples[selection].clone();
            taken.push(selection);

            // weight the next set of values
            let mut weight_total = 0;
            prob_list = Vec::new();
            for (i, value) in samples.iter().enumerate() {
                if taken.contains(&i) {
                    continue;
                }
                let distance = cat.distance(value);
                let weight = (distance * distance) as i64;
                prob_list.push((i, weight));
                weight_total += weight;
            }

            // now select one at random
            let mut remainder = generator.gen_range(0, weight_total);
            for pair in prob_list {
                remainder -= pair.1;
                if remainder <= 0 {
                    selection = pair.0;
                    break;
                }
            }
        }
    }

    fn categorise(&self, datum: &T) -> Result<usize, TrainingError> {
        let mut result = Err(TrainingError::InvalidClassifier);
        let mut distance: f64 = f64::MAX;
        for (i, cat) in self.categories.iter().enumerate() {
            let cur_dist = cat.distance(datum);
            if cur_dist < distance {
                distance = cur_dist;
                result = Ok(i);
            }
        }
        return result;
    }

    fn categorise_all(&self, data: &Vec<T>) -> Result<Vec<usize>, TrainingError> {
        let mut cats = vec![0 as usize; data.len()];
        for (i, datum) in data.iter().enumerate() {
            cats[i] = self.categorise(datum)?;
        }
        return Ok(cats);
    }

    // updates the centroids with the current classifications,
    // returns true if the centoids changed
    fn update_centroids(&mut self, data: &Vec<T>, cats: &Vec<usize>) -> Result<bool, TrainingError> {
        if data.len() != cats.len() {
            return Err(TrainingError::InvalidData);
        }

        // take an average value from the current categorisation
        let mut updated = false;
        let mut sums = vec![T::default(); self.categories.len()];
        let mut counts = vec![0; self.categories.len()];
        for (i, cat) in cats.iter().enumerate() {
            sums[*cat] = sums[*cat].add(&data[i]);
            counts[*cat] += 1;
        }

        // update the old value, and keep track of it if changed
        for (i, sum) in sums.iter_mut().enumerate() {
            let new_val = sum.scalar_div(&(counts[i] as f64));
            if new_val != self.categories[i] {
                updated = true;
            }
            self.categories[i] = new_val;
        }
        println!("Updating centroids");
        return Ok(updated);
    }
}

impl<T: EuclideanDistance + PartialEq + Random + Default + Clone> UnsupervisedClassifier<T> for KMeans<T> {
    fn train(&mut self, data: &Vec<T>) -> Result<Vec<T>, TrainingError> {
        // initialise the centroids randomly, initially
        self.get_psuedo_random_centroids(data);

        // until the centroids don't change, keep changing the centroids
        let mut categories: Vec<usize>;
        loop {
            
            // classify the data with the current centroids, then update the centroids
            match self.categorise_all(data) {
                Ok(cats) => categories = cats,
                Err(e) => return Err(e),
            }
            match self.update_centroids(data, &categories) {
                Ok(fin) => {
                    if !fin {break;}
                },
                Err(e) => return Err(e),
            }
        }
        return Ok(self.categories.clone().to_vec());
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
}