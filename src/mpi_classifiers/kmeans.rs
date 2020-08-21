use crate::mpi_classifiers::unsupervised_classifier::UnsupervisedClassifier;
use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError
};

use rand::Rng;
use large_int::large_int::LargeInt;
use std::{
    f64,
    fs::File,
    mem::size_of
};
use mpi::{
    traits::*,
    topology::SystemCommunicator,
    datatype::{
        UserDatatype,
        PartitionMut
    },
};
use std::borrow::Borrow;
use std::io::{BufRead, BufReader};
use std::fmt::Debug;
use std::io::Write;

const ROOT: i32 = 0;

// used to transfer the
#[derive(Clone, Default, Debug)]
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
        samples: &mut Vec<(T, f64)>,
        newest_cat: &T
    ) {
        // weight the next set of values
        for (i, value) in samples.iter_mut().enumerate() {
            let distance = newest_cat.distance(&value.0);

            // use the closest of the currently selected centroids
            if distance < value.1 {
                value.1 = distance;
            }
        }
    }

    fn sum_of_weights_squared(prob_list: &Vec<(T, f64)>) -> f64 {
        let mut weight = 0_f64;
        for (_, distance) in prob_list {
            if !distance.is_nan() {
                weight += distance.powi(2);
            }
        }
        weight
    }

    fn sum_of_weights_squared_large(prob_list: &Vec<(T, f64)>) -> LargeInt {
        let mut weight = LargeInt::new();
        for (_, distance) in prob_list {
            if !distance.is_nan() {
                weight += distance.powi(2).round() as u64;
            }
        }
        weight
    }

    // selects the initial point from which distance
    // determines probablity of selecting other points
    fn select_initial(&mut self, world: &SystemCommunicator, data: &Vec<(T, f64)>) -> T {
        
        // each process selects one point at random
        let mut generator = rand::thread_rng();
        let in_range = data.len();
        let selection = &data[generator.gen_range(0, in_range)].0;

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

    fn sum_categories(
        &self,
        data: &Vec<(T, f64)>,
        cat_sums: &mut Vec<SumCountPair<T>>
    ) -> Result<(), TrainingError> {
        for (datum, _) in data.iter() {
            cat_sums[self.categorise(datum)?].plus(datum);
        }
        return Ok(());
    }

    // perform Lloyd's iteration
    fn lloyds_iteration(
        &mut self,
        world: &SystemCommunicator,
        data: &Vec<(T, f64)>
    ) -> Result<(), TrainingError> {
        let mut iters = 0;
        let size = world.size();
        // continue until means no longer update
        while iters < 10 {

            // calculate local values
            let mut local_scs: Vec<SumCountPair<T>> =
                vec!(SumCountPair::default(); self.k);
            self.sum_categories(data, &mut local_scs)?;
            let mut gathered =
                vec!(SumCountPair::<T>::default(); self.k * size as usize);
            let mut global_scs: Vec<SumCountPair<T>> =
                vec!(SumCountPair::default(); self.k);

            // combine with other procs into global values
            world.all_gather_into(
                local_scs.as_slice(),
                gathered.as_mut_slice()
            );
            // if world.rank() == ROOT {println!("Gathered = {:?}", gathered);}
            for i in 0..self.k {
                for j in 0..size as usize {
                    global_scs[i] = global_scs[i].combine(&gathered[j * self.k + i]);
                }
            }

            // check if the means updated
            let mut finish = true;
            let cur_cats = self.categories.as_mut().unwrap();
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
        let (_, size) = get_world_info(world);
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
    fn get_weighted_random_centroids(&mut self, samples: &mut Vec<(T, f64)>) {
        // can't do anything if there is no data
        if samples.len() == 0 || samples.len() < self.k {
            return;
        }

        // reset the previous training, if any
        self.categories = Some(Box::new(Vec::new()));

        // select one random value initially
        let mut generator = rand::thread_rng();
        let categories = self.categories.as_mut().unwrap();

        // continue to select values based on weighted probablity
        categories.push(samples[generator.gen_range(0, samples.len()) as usize].0.clone());
        for i in 0..(self.k - 1) {
            Self::update_weights(samples, &categories[i]);
            let weight_total = Self::sum_of_weights_squared(&samples);

            // now select one at random
            let mut remainder = generator.gen_range(0, weight_total as usize) as f64;
            for (val, dist) in samples.iter_mut() {
                if dist.is_nan() { continue; }
                remainder -= dist.powi(2);
                if remainder <= 0.0 {
                    *dist = f64::NAN;
                    categories.push(val.clone());
                    break;
                }
            }
        }
    }

    // oversample points to form initial cluster approximation
    fn oversample_by_weight(&mut self, world: &SystemCommunicator, data: &mut Vec<(T, f64)>) {
        // TODO: figure out how to deal with this in the distributed case
        // can't do anything if there is no data
        // if data.len() == 0 || data.len() < self.k {
        //     return;
        // }

        // select one random value initially
        let (my_rank, size) = get_world_info(world);
        let mut taken = Vec::new();
        let initial = self.select_initial(world, data);

        // get initial weight to determine num. iterations
        Self::update_weights(data, &initial);
        let mut weight_total = Self::sum_of_weights_squared(data);
        let mut global_weights = vec!(0.0; size as usize);
        let mut generator = rand::thread_rng();
        world.all_to_all_into(
            vec!(weight_total; size as usize).as_slice(),
            global_weights.as_mut_slice()
        );
        weight_total = global_weights.into_iter().sum();
        println!("Done selecting initial");
        std::io::stdout().flush();

        // oversample centres (achieve approx. klog(initial_weight))
        for _ in 0..(weight_total as f64).log10() as usize {
            let mut selections = Vec::new();

            // sample independantly proportional to distance from a centre
            for (val, dist) in data.iter_mut() {
                let prob = generator.gen_range(0.0, 1.0);
                let select = (dist.powi(2) * self.k as f64) / weight_total as f64;
                if prob < select {
                    selections.push(val.clone());
                    *dist = f64::NAN; // exclude from sum of weights
                }
            }

            let mut global_selections = Self::send_selected(
                world,
                &mut selections
            );

            // update the smallest distance to any point
            for selection in global_selections.iter() {
                Self::update_weights(
                    data,
                    &selection
                )
            }

            // the root will calculate the initial centroids
            if my_rank == ROOT {
                taken.append(&mut global_selections);
            }
            weight_total = Self::sum_of_weights_squared(data);
            global_weights = vec!(0.0; size as usize);
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
        let mut weighted_taken = taken.into_iter().map(|v| (v, f64::MAX)).collect();
        if my_rank == ROOT {
            self.get_weighted_random_centroids(&mut weighted_taken);
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
        pre_data: Vec<T>
    ) -> Result<Vec<T>, TrainingError> {
        let mut data: Vec<(T, f64)> = pre_data.into_iter().map(|v| (v, f64::MAX)).collect();
        self.oversample_by_weight(world, &mut data);
        println!("Done sampling");
        std::io::stdout().flush();
        self.lloyds_iteration(world, &mut data)?;
        println!("Done lloyds iteration");
        std::io::stdout().flush();
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
        println!("Done reading file");
        std::io::stdout().flush();
        self.train(world, data)
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

// only allow this code in debug such that tests
// can be run with mpi from the external example
#[cfg(debug_assertions)]
pub mod tests {
    use super::MPIKMeans;
    use crate::example_datatypes::point::Point;
    use mpi::topology::{SystemCommunicator, Communicator};
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use crate::common::TrainingError;
    use std::f64;

    static INPUT_FILE: &'static str = "./data/easy_clusters_rand";

    // this data was collected from a particular run of
    // the serial version, so depends on the accuracy of that
    // version to determine the accuracy of this one
    fn get_convergence_sets() -> (Vec<Point>, Vec<Point>) {
        let mut cats = vec!();
        cats.push(Point::new(332237.00,911313.00));
        cats.push(Point::new(382419.00,110392.00));
        cats.push(Point::new(598184.00,531473.00));
        cats.push(Point::new(709303.00,897538.00));
        cats.push(Point::new(139903.00,592241.00));
        cats.push(Point::new(850959.00,157920.00));
        cats.push(Point::new(716139.00,621721.00));
        cats.push(Point::new(541435.00,206637.00));
        cats.push(Point::new(371464.00,328177.00));
        cats.push(Point::new(415895.00,775793.00));
        cats.push(Point::new(178869.00,370161.00));
        cats.push(Point::new(900829.00,590250.00));
        cats.push(Point::new(344915.00,702496.00));
        cats.push(Point::new(835128.00,212354.00));
        cats.push(Point::new(206660.00,575002.00));

        // rough values
        let mut fin = vec!();
        fin.push(Point::new(244654.89,847642.04));
        fin.push(Point::new(320602.55,161521.85));
        fin.push(Point::new(612001.26,483041.06));
        fin.push(Point::new(670929.07,862765.73));
        fin.push(Point::new(136671.89,558362.82));
        fin.push(Point::new(852058.45,157685.52));
        fin.push(Point::new(822688.98,730558.55));
        fin.push(Point::new(508062.26,176011.77));
        fin.push(Point::new(400011.48,404475.00));
        fin.push(Point::new(418183.10,786581.83));
        fin.push(Point::new(167856.14,347812.72));
        fin.push(Point::new(858529.97,546434.75));
        fin.push(Point::new(349034.24,566772.45));
        fin.push(Point::new(798783.14,321818.38));
        fin.push(Point::new(283176.86,545552.91));

        (cats, fin)
    }

    fn point_vec_from_file(file: &File) -> Result<Box<Vec<Point>>, TrainingError> {
        let mut data = Box::new(Vec::new());
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let buf = match line {
                Ok(l) => l.into_bytes(),
                Err(_) => return Err(TrainingError::FileReadFailed),
            };
            let val = Point::point_from_vec(&buf)?;
            data.push(val);
        }
        return Ok(data);
    }

    pub fn test_convergence(world: &SystemCommunicator) {
        let my_rank = world.rank();
        let size = world.size();
        let data_file = match File::open(INPUT_FILE) {
            Ok(f) => f,
            Err(_) => {
                panic!("Test data file not found");
            },
        };
        let just_data = match point_vec_from_file(&data_file) {
            Ok(d) => d,
            Err(_) => panic!("Could not load test data")
        };
        let data: Vec<(Point, f64)> = just_data.into_iter().map(|v| (v, f64::MAX)).collect();
        let portion_size = data.len() / size as usize;
        let my_data = if my_rank == size - 1 {
            data[portion_size * my_rank as usize
                ..data.len()].to_vec()
        } else {
            data[portion_size * my_rank as usize
                ..portion_size * (my_rank + 1) as usize].to_vec()
        };

        let mut km = MPIKMeans::new(15);
        let (cats, fin) = get_convergence_sets();

        km.categories = Some(Box::new(cats));
        match km.lloyds_iteration(&world, &my_data) {
            Err(_) => panic!("Lloyd's iteration failed"),
            _ => ()
        }
        let finals = km.categories.unwrap_or(Box::new(Vec::new()));

        // use the format string so that decimal resolution isn't an issue
        assert_eq!(format!("{:?}", finals), format!("{:?}", fin));
    }
}
