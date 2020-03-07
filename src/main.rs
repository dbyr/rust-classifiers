mod point;
mod classifier;
mod euclidean_distance;
mod kmeans;
mod random;

use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::kmeans::KMeans;
use crate::point::Point;
use crate::classifier::{
    UnsupervisedClassifier,
    TrainingError
};

// static INPUT_FILE: &'static str = "./data/easy_clusters_rand";
// static OUTPUT_FILE: &'static str = "./data/clustered_data_rand.csv";
static INPUT_FILE: &'static str = "./data/hr-diagram-data-shuffed";
static OUTPUT_FILE: &'static str = "./data/clustered_hr_data.csv";

fn point_vec_from_file(file: &File) -> Result<Vec<Point>, TrainingError> {
    let mut data = Vec::new();
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
    let mut classifier = KMeans::<Point>::new(4);
    let mut f = match File::open(INPUT_FILE) {
        Ok(f) => f,
        Err(_) => {
            println!("Couldn't open file containing data");
            return;
        },
    };
    let cats_res = classifier.train_from_file(
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
    match classifier::classify_csv(&f_out, &data, &cats) {
        Ok(_) => return,
        Err(_) => {
            println!("Could not write csv file");
            return;
        },
    }
}
