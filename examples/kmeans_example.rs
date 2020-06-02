extern crate rust_classifiers;

use std::fs::File;
use std::io::{BufRead, BufReader};

use rust_classifiers::example_datatypes::point::Point;
use rust_classifiers::serial_classifiers::kmeans::KMeans;
use rust_classifiers::impl_euclidean_distance;
use rust_classifiers::impl_attributable;
use rust_classifiers::euclidean_distance::EuclideanDistance;
use rust_classifiers::common::Attributable;
use rust_classifiers::serial_classifiers::unsupervised_classifier::{
    UnsupervisedClassifier,
    classify_csv
};
use rust_classifiers::common::TrainingError;

static INPUT_FILE: &'static str = "./data/easy_clusters_rand";
static OUTPUT_FILE: &'static str = "./data/clustered_data_pp.csv";
// static INPUT_FILE: &'static str = "./data/easy_clusters_rand";
// static OUTPUT_FILE: &'static str = "./data/clustered_data_rand.csv";
// static INPUT_FILE: &'static str = "./data/hr-diagram-data-shuffed";
// static OUTPUT_FILE: &'static str = "./data/clustered_hr_data.csv";

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

fn two_d_point_from_file(file: &File) -> Result<Box<Vec<[f64; 2]>>, TrainingError> {
    let data = point_vec_from_file(file)?;
    let mut results = Box::new(Vec::new());
    for datum in data.into_iter() {
        let mut cur = [0.0; 2];
        cur[0] = datum.get_x();
        cur[1] = datum.get_y();
        results.push(cur);
    }
    Ok(results)
}

fn two_d_point_from_vec(bytes: &Vec<u8>) -> Result<[f64; 2], TrainingError> {
    let point = Point::point_from_vec(bytes)?;
    Ok([point.get_x(), point.get_y()])
}

struct Wrapper;
impl_euclidean_distance!(Wrapper[f64; 2]);
impl_attributable!(Wrapper[f64; 2]);

fn main() {
    let mut classifier = KMeans::<Wrapper, [f64; 2]>::new_pp(15);
    let mut f = match File::open(INPUT_FILE) {
        Ok(f) => f,
        Err(_) => {
            println!("Couldn't open file containing data");
            return;
        },
    };
    let cats_res = classifier.train_from_file(
        &mut f,
        &two_d_point_from_vec
    );

    let cats = match cats_res {
        Ok(c) => c,
        Err(_) => {
            println!("Categorisation failed");
            return;
        },
    };

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
    let data = match two_d_point_from_file(&f) {
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
