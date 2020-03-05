mod point;
mod classifier;
mod euclidean_distance;
mod kmeans;
mod random;

use crate::euclidean_distance::EuclideanDistance;
use crate::kmeans::KMeans;

fn main() {
    let p1 = point::Point::new(1.0, 2.0);
    println!("{:?}", p1);

    let p2 = point::Point::new(1.5, 3.0);
    println!("{}", p1.distance(&p2));
    println!("{:?}", p2);

    let cats = vec![point::Point::new(5.1, 10.2), point::Point::new(1.0, 3.0)];
    println!("{}", classifier::classify(&p2, &cats));

    let mut classifier = KMeans::<point::Point>::new(3);
    // classifier.randomize_centroids_in_range(
    //     &point::Point::new(99569.0, 99170.0),
    //     &point::Point::new(101427.0, 101272.0)
    // );
    println!("{:?}", classifier);
}
