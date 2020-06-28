use crate::serial_classifiers::kmeans::KMeans;
use crate::mpi_classifiers::unsupervised_classifier::UnsupervisedClassifier;
use crate::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier as SUC;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

use rand::Rng;
use std::{
    f64,
    fs::File,
    collections::HashMap,
    mem::size_of
};
use mpi::{
    traits::*,
    topology::SystemCommunicator,
    collective::UserOperation,
    datatype::{
        UserDatatype,
        PartitionMut
    },
};
use std::borrow::Borrow;
use std::io::{BufRead, BufReader};
use std::fmt::{Display, Debug};

const ROOT: i32 = 0;

// used to transfer the
#[derive(Clone, Default)]
struct SumCountPair<T: EuclideanDistance> {
    sum: T,
    count: u64
}

impl<T: EuclideanDistance> SumCountPair<T> {
    fn plus(&mut self, val: &T) {
        self.sum = self.sum.add(val);
        self.count += 1;
    }

    fn combine(&self, other: &SumCountPair<T>) -> SumCountPair<T> {
        SumCountPair{
            sum: self.sum.add(&other.sum),
            count: self.count + other.count
        }
    }

    fn result(&self) -> T {
        self.sum.scalar_div(self.count as f64)
    }
}

unsafe impl<T> Equivalence for SumCountPair<T>
where T: Equivalence + EuclideanDistance {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            2,
            &[1, 1],
            &[size_of::<u64>() as mpi::Address, 0],
            &[&T::equivalent_datatype(), &u64::equivalent_datatype()]
        )
    }
}

pub struct MPIKMeans<T> where T: Clone + EuclideanDistance {
    categories: Option<Box<Vec<T>>>,
    k: usize
}

fn get_world_info(world: &SystemCommunicator) -> (i32, i32) {
    (world.rank(), world.size())
}

impl<T> MPIKMeans<T> where T: Default + Clone
+ EuclideanDistance + Equivalence + PartialEq + Debug {

    pub fn new(k: usize) -> Self {
        MPIKMeans{
            categories: None,
            k: k
        }
    }

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

    // perform Lloyd's iteration
    fn lloyds_iteration(
        &mut self,
        world: &SystemCommunicator,
        data: &Vec<T>
    ) -> Result<(), TrainingError> {
        let (my_rank, size) = get_world_info(world);
        let mut iters = 0;
        // continue until means no longer update
        while iters < 100 {

            // calculate local values
            let cats = self.categorise_all(data)?;
            let mut local_scs: Vec<SumCountPair<T>> =
                vec!(SumCountPair::default(); self.k);
            let mut gathered =
                vec!(SumCountPair::<T>::default(); self.k * self.k);
            let mut global_scs: Vec<SumCountPair<T>> =
                vec!(SumCountPair::default(); self.k);
            for (i, cat) in cats.iter().enumerate() {
                local_scs[*cat].plus(&data[i]);
            }

            // combine with other procs into global values
            world.all_gather_into(
                local_scs.as_slice(),
                gathered.as_mut_slice()
            );
            for i in 0..self.k {
                let base = i * self.k;
                for j in 0..self.k {
                    global_scs[i] = global_scs[i].combine(&gathered[base + j]);
                }
            }

            // check if the means updated
            let mut finish = true;
            let mut cur_cats = self.categories.as_mut().unwrap();
            for (cur_cat, scs) in cur_cats.iter_mut().zip(global_scs) {
                if scs.count == 0 {
                    continue;
                }
                let new_mean = scs.result();
                if *cur_cat != new_mean {
                    finish = false;
                }
                *cur_cat = new_mean;
            }
            if finish { break; }
            iters += 1;
        }
        Ok(())
    }

    // sends the chosen samples for this iteration
    fn send_selected(world: &SystemCommunicator, samples: &mut Vec<T>) -> Vec<T> {
        let (my_rank, size) = get_world_info(world);
        // let root_proc = world.process_at_rank(ROOT);
        let my_size = samples.len() as i32;

        // send the sizes of each process to each other
        let mut sizes: Vec<i32> = vec!(0; size as usize);
        world.all_gather_into(&my_size, sizes.as_mut_slice());
        let mut displacements = vec!(0; size as usize);
        for i in 1..size {
            let prev = (i - 1) as usize;
            displacements[i as usize] = sizes[prev] + displacements[prev];
        }

        // send the samples of each process to each other
        let mut global_samples = vec!(T::default(); sizes.iter().sum::<i32>() as usize);
        let mut buffer = PartitionMut::new(
            global_samples.as_mut_slice(),
            sizes.borrow(),
            displacements.borrow()
        );
        world.all_gather_varcount_into(samples.as_slice(), &mut buffer);
        global_samples
    }

    // initial centroid selection for kmeans++
    fn get_weighted_random_centroids(&mut self, samples: &Vec<T>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }

        // reset the previous training, if any
        self.categories = Some(Box::new(Vec::new()));

        // select one random value initially
        let mut generator = rand::thread_rng();
        let mut prob_list = HashMap::<usize, f64>::new();
        let mut selection: usize = generator.gen_range(0, samples.len());
        let categories = self.categories.as_mut().unwrap();

        // continue to select values based on weighted probablity
        categories.push(samples[selection].clone());
        for _ in 0..(self.k - 1) {
            Self::update_weights(samples, &mut prob_list, &samples[selection]);
            let weight_total = Self::sum_of_weights_squared(&prob_list);

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
        }
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
            vec!(weight_total; size as usize).as_slice(),
            global_weights.as_mut_slice()
        );
        weight_total = global_weights.into_iter().sum();

        // oversample centres (achieve approx. klog(initial_weight))
        for iter_val in 0..(weight_total as f64).log10() as usize {
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

            let mut global_selections = Self::send_selected(
                world,
                &mut selections
            );

            // update the smallest distance to any point
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
                vec!(weight_total; size as usize).as_slice(),
                global_weights.as_mut_slice()
            );
            weight_total = global_weights.into_iter().sum();
        }

        // TODO: check if this is desirable
        // the original paper suggests doing this portion
        // of the computation on a single machine
        let root_proc = world.process_at_rank(ROOT);
        if my_rank == ROOT {
            self.get_weighted_random_centroids(&taken);
        } else {
            self.categories = Some(Box::new(vec!(T::default(); self.k)));
        }
        root_proc.broadcast_into(self.categories.as_mut().unwrap().as_mut_slice());
    }
}

impl<T> UnsupervisedClassifier<T> for MPIKMeans<T>
where T: Clone + EuclideanDistance + Default + PartialEq + Equivalence + Debug {
    fn train(
        &mut self,
        world: &SystemCommunicator,
        data: &Vec<T>
    ) -> Result<Vec<T>, TrainingError> {
        self.oversample_by_weight(world, data);
        self.lloyds_iteration(world, data)?;
        Ok(*self.categories.clone().unwrap())
    }

    fn train_from_file(
        &mut self,
        world: &SystemCommunicator,
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
        self.train(world, &data)
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
