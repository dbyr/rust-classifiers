mod point;
mod classifier;
mod euclidean_distance;
mod kmeans;
mod random;

use std::fs::File;
use std::io::{LineWriter, BufRead, BufReader};

use crate::euclidean_distance::EuclideanDistance;
use crate::kmeans::KMeans;
use crate::point::Point;
use crate::classifier::{
    UnsupervisedClassifier,
    TrainingError
};

fn point_from_vec(buff: &Vec<u8>) -> Result<Point, TrainingError> {
    // let buf = buff.into_bytes();
    let mut val = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    for c in buff {
        // only match numbers (and delimit on space and new line)
        if *c >= 48u8 && *c <= 57u8 {
            val *= 10.0;
            val += (*c - 48u8) as f64;
        } else if *c == 32u8 {
            x = val;
            val = 0.0;
        } else {
            return Err(TrainingError::InvalidFile);
        }
    }
    y = val;
    return Ok(Point::new(x, y));
}

fn point_vec_from_file(file: &File) -> Result<Vec<Point>, TrainingError> {
    let mut data = Vec::new();
    let mut reader = BufReader::new(file);
    for line in reader.lines() {
        let buf = match line {
            Ok(l) => l.into_bytes(),
            Err(e) => return Err(TrainingError::FileReadFailed),
        };
        let val = point_from_vec(&buf)?;
        data.push(val);
    }
    return Ok(data);
}

fn main() {
    let p1 = Point::new(1.0, 2.0);
    let p2 = Point::new(1.5, 3.0);
    let cats = vec![Point::new(5.1, 10.2), Point::new(1.0, 3.0)];
    let mut classifier = KMeans::<Point>::new(15);
    let mut f = match File::open("./easy_clusters") {
        Ok(f) => f,
        Err(e) => {
            println!("Couldn't open file containing data");
            return;
        },
    };
    let cats_res = classifier.train_from_file(
        &mut f,
        &point_from_vec
    );

    let cats = match cats_res {
        Ok(c) => c,
        Err(e) => {
            println!("Categorisation failed");
            return;
        },
    };

    println!("{:?}", cats);

    // print out a csv file containing the data and their clusters
    let mut f_out = match File::create("./clustered_data.csv") {
        Ok(f) => f,
        Err(e) => {
            println!("Couldn't create clustered data file");
            return;
        },
    };

    let mut f = match File::open("./easy_clusters") {
        Ok(f) => f,
        Err(e) => {
            println!("Couldn't open file containing data");
            return;
        },
    };
    let data = match point_vec_from_file(&f) {
        Ok(v) => v,
        Err(e) => {
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
