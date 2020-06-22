use crate::serial_classifiers::kmeans::KMeans;
use crate::unsupervised_classifier::UnsupervisedClassifier;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

use rand::Rng;
use std::fs::File;
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

impl<T> MPIKMeans<T> where T: Default + Clone + EuclideanDistance + Equivalence {

    // selects the initial point from which distance
    // determines probablity of selecting other points
    fn select_initial(&mut self, world: &SystemCommunicator, data: &Vec<T>) -> T {
        
        // each process selects one point at random
        let mut generator = rand::thread_rng();
        let in_range = data.len();
        let mut selection = &data[generator.gen_range(0, in_range)];

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
}

impl<T> UnsupervisedClassifier<T> for MPIKMeans<T>
where T: Clone + EuclideanDistance {
    fn train(&mut self, data: &Vec<T>) -> Result<Vec<T>, TrainingError> {
        Err(TrainingError::InvalidClassifier)
    }

    fn train_from_file(
        &mut self, 
        file: &mut File, 
        parser: &dyn Fn(&Vec<u8>) -> Result<T, TrainingError>
    ) -> Result<Vec<T>, TrainingError> {
        Err(TrainingError::InvalidClassifier)
    }

    fn classify(&self, datum: &T) -> Result<usize, ClassificationError> {
        Err(ClassificationError::ClassifierInvalid)
    }
}
