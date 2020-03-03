use crate::classifier::EuclideanDistance;

mod point;
mod classifier;

fn main() {
    let p1 = point::Point::new(1.0, 2.0);
    println!("{:?}", p1);

    let p2 = point::Point::new(1.5, 3.0);
    println!("{}", p1.distance(&p2));
    println!("{:?}", p2);

    let cats = vec![point::Point::new(5.1, 10.2), point::Point::new(1.0, 3.0)];
    println!("{}", classifier::classify(&p2, &cats));
}
