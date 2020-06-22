use crate::serial_classifiers::kmeans::KMeans;
use crate::unsupervised_classifier::UnsupervisedClassifier;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

use rand::Rng;
use std::fs::File;
use std::collections::HashMap;
use mpi::{
    traits::*,
    topology::SystemCommunicator
};

const ROOT: i32 = 0;

pub struct MPIKMeans<T> where T: Clone + EuclideanDistance {
    categories: Option<Box<Vec<T>>>,
    k: usize
}

fn get_world_info(world: &SystemCommunicator) -> (i32, i32) {
    (world.rank(), world.size())
}

impl<T> MPIKMeans<T> where T: Default + Clone
+ EuclideanDistance + Equivalence + PartialEq {

    fn update_weights(
        samples: &Vec<T>,
        prob_list: &mut HashMap<usize, f64>,
        newest_cat: &T
    ) {
        // weight the next set of values
        for (i, value) in samples.iter().enumerate() {
            let distance = newest_cat.distance(value);

            // use the closest of the currently selected centroids
            if let Some(v) = prob_list.get_mut(&i) {
                if distance < *v {
                    *v = distance;
                }
            } else {
                prob_list.insert(i, distance);
            }
        }
    }

    fn sum_of_weights_squared(prob_list: &HashMap<usize, f64>) -> usize {
        let mut weight = 0_f64;
        for (_, distance) in prob_list {
            weight += distance.powi(2);
        }
        weight as usize
    }

    // selects the initial point from which distance
    // determines probablity of selecting other points
    fn select_initial(&mut self, world: &SystemCommunicator, data: &Vec<T>) -> T {
        
        // each process selects one point at random
        let mut generator = rand::thread_rng();
        let in_range = data.len();
        let selection = &data[generator.gen_range(0, in_range)];

        // then sends it to proc 0 to decide which is the initial
        let (my_rank, size) = get_world_info(world);
        let root_proc = world.process_at_rank(ROOT);
        let mut final_initial = if my_rank == ROOT {
            let mut options = vec!(T::default(); size as usize);
            root_proc.gather_into_root(selection, options.as_mut_slice());
            options[generator.gen_range(0, size) as usize].clone()
        } else {
            root_proc.gather_into(selection);
            T::default()
        };

        // proc 0 broadcasts to all what the point is
        root_proc.broadcast_into(&mut final_initial);
        final_initial
    }

    // perform Lloyd's iteration
    // fn lloyds_iteration(&mut self, world: &SystemCommunicator, data: &Vec<T>) {
    //
    // }

    // sends the chosen samples for this iteration
    fn send_selected(world: &SystemCommunicator, samples: &mut Vec<T>) -> Vec<T> {
        let (my_rank, size) = get_world_info(world);
        let mut selected_size = vec!(samples.len(); size as usize);
        let mut additional = Vec::new();
        for i in 0..size {
            // discover how many items were selected
            let proc = world.process_at_rank(i);
            proc.broadcast_into(&mut selected_size[i as usize]);

            // collect that many items
            let mut added = vec!(T::default(); selected_size[i as usize]);
            if my_rank == i {
                proc.broadcast_into(samples.as_mut_slice());
            } else {
                proc.broadcast_into(added.as_mut_slice());
                additional.append(&mut added);
            }
        }
        additional
    }

    // oversample points to form initial cluster approximation
    fn oversample_by_weight(&mut self, world: &SystemCommunicator, data: &Vec<T>) {
        // TODO: figure out how to deal with this in the distributed case
        // can't do anything if there is no data
        // if data.len() == 0 || data.len() < self.k {
        //     return;
        // }

        // select one random value initially
        let (my_rank, size) = get_world_info(world);
        let mut taken = Vec::new();
        let mut prob_list = HashMap::<usize, f64>::new();
        let initial = self.select_initial(world, data);

        // get initial weight to determine num. iterations
        Self::update_weights(data, &mut prob_list, &initial);
        let mut weight_total = Self::sum_of_weights_squared(&prob_list);
        let mut global_weights = vec!(0; size as usize);
        let mut generator = rand::thread_rng();
        world.all_to_all_into(
            vec!(weight_total).as_slice(),
            global_weights.as_mut_slice()
        );
        weight_total = global_weights.into_iter().sum();

        // oversample centres (achieve approx. klog(initial_weight))
        for _ in 0..(weight_total as f64).log10() as usize {
            let mut selections = Vec::new();
            let mut indices = Vec::new();

            // sample independantly proportional to distance from a centre
            for (index, dist) in prob_list.iter() {
                let prob = generator.gen_range(0.0, 1.0);
                let select = (dist.powi(2) * self.k as f64) / weight_total as f64;
                if prob < select {
                    selections.push(data[*index].clone());
                    indices.push(*index);
                }
            }
            for index in indices {
                prob_list.remove(&index);
            }

            let mut global_selections = Self::send_selected(world, &mut selections);

            // update the smallest distance to any point
            global_selections.append(&mut selections);
            for selection in global_selections.iter() {
                Self::update_weights(
                    data,
                    &mut prob_list,
                    &selection
                )
            }

            // the root will calculate the initial centroids
            if my_rank == ROOT {
                taken.append(&mut global_selections);
            }
            weight_total = Self::sum_of_weights_squared(&prob_list);
            global_weights = vec!(0; size as usize);
            world.all_to_all_into(
                vec!(weight_total).as_slice(),
                global_weights.as_mut_slice()
            );
            weight_total = global_weights.into_iter().sum();
        }

        // TODO: check if this is desirable
        // the original paper suggests doing this portion
        // of the computation on a single machine
        let root_proc = world.process_at_rank(ROOT);
        if my_rank == ROOT {
            let mut trainer = KMeans::new_pp(self.k);
            trainer.train(&taken);
            self.categories = Some(Box::new(trainer.categories().unwrap().clone()));
        } else {
            self.categories = Some(Box::new(vec!(T::default(); self.k)));
        }
        root_proc.broadcast_into(self.categories.as_mut().unwrap().as_mut_slice());
    }
}

impl<T> UnsupervisedClassifier<T> for MPIKMeans<T>
where T: Clone + EuclideanDistance {
    fn train(&mut self, _data: &Vec<T>) -> Result<Vec<T>, TrainingError> {
        Err(TrainingError::InvalidClassifier)
    }

    fn train_from_file(
        &mut self, 
        _file: &mut File,
        _parser: &dyn Fn(&Vec<u8>) -> Result<T, TrainingError>
    ) -> Result<Vec<T>, TrainingError> {
        Err(TrainingError::InvalidClassifier)
    }

    fn classify(&self, _datum: &T) -> Result<usize, ClassificationError> {
        Err(ClassificationError::ClassifierInvalid)
    }
}
