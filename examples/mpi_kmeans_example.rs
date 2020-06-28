extern crate rust_classifiers;

use std::fs::File;
use std::io::{BufRead, BufReader};

use rust_classifiers::example_datatypes::point::Point;
use rust_classifiers::mpi_classifiers::kmeans::MPIKMeans;
use rust_classifiers::mpi_classifiers::unsupervised_classifier::{
    UnsupervisedClassifier,
    classify_csv
};
use rust_classifiers::common::TrainingError;
use mpi::topology::Communicator;

static INPUT_FILE: &'static str = "./data/mpi/easy_clusters";
static OUTPUT_FILE: &'static str = "./data/mpi/clustered_data.csv";

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

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let my_rank = world.rank();
    let size = world.size();

    let my_file = format!("./data/mpi/{}procs/easy_clusters{}", size, my_rank);

    let mut classifier = MPIKMeans::<Point>::new(15);
    let mut f = match File::open(my_file) {
        Ok(f) => f,
        Err(_) => {
            println!("Couldn't open file containing data");
            return;
        },
    };
    let cats_res = classifier.train_from_file(
        &world,
        &mut f,
        &Point::point_from_vec
    );

    let cats = match cats_res {
        Ok(c) => c,
        Err(_) => {
            println!("Categorisation failed");
            return;
        },
    };

    // only one process should do this part
    if my_rank == 0 {
        println!("{:?}", cats);

        // print out a csv file containing the data and their clusters
        let f_out = match File::create(OUTPUT_FILE) {
            Ok(f) => f,
            Err(_) => {
                println!("Couldn't create clustered data file");
                return;
            },
        };

        let f = match File::open(INPUT_FILE) {
            Ok(f) => f,
            Err(_) => {
                println!("Couldn't open file containing data");
                return;
            },
        };
        let data = match point_vec_from_file(&f) {
            Ok(v) => v,
            Err(_) => {
                println!("Couldn't read data from file");
                return;
            }
        };
        match classify_csv(&classifier, &f_out, &data) {
            Ok(_) => return,
            Err(_) => {
                println!("Could not write csv file");
                return;
            },
        }
    }
}
