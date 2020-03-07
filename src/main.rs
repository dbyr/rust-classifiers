mod point;
mod classifier;
mod euclidean_distance;
mod kmeans;
mod random;

use std::fs::File;

use crate::euclidean_distance::EuclideanDistance;
use crate::kmeans::KMeans;
use crate::classifier::{
    UnsupervisedClassifier,
    TrainingError
};

fn main() {
    let p1 = point::Point::new(1.0, 2.0);
    println!("{:?}", p1);

    let p2 = point::Point::new(1.5, 3.0);
    println!("{}", p1.distance(&p2));
    println!("{:?}", p2);

    let cats = vec![point::Point::new(5.1, 10.2), point::Point::new(1.0, 3.0)];
    println!("{}", classifier::classify(&p2, &cats));

    let mut classifier = KMeans::<point::Point>::new(3);
    println!("{:?}", classifier);

    let mut f = match File::open("./easy_clusters") {
        Ok(f) => f,
        Err(e) => return,
    };
    let cats_res = classifier.train_from_file(
        &mut f,
        &|buff| {
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
            return Ok(point::Point::new(x, y));
        }
    );

    let cats = match cats_res {
        Ok(c) => c,
        Err(e) => return,
    };

    println!("{:?}", cats);
}
