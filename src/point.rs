use rand::Rng;

use crate::euclidean_distance::EuclideanDistance;
use crate::random::Random;
use crate::classifier::Attributable;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct Point {
    x: f64,
    y: f64
}

impl Point{
    pub fn new(x: f64, y: f64) -> Point {
        Point{x: x, y: y}
    }
    pub fn get_x(&self) -> &f64 {
        return &self.x;
    }
    pub fn get_y(&self) -> &f64 {
        return &self.y;
    }
}

impl EuclideanDistance for Point {
    fn distance(&self, rhs: &Point) -> f64 {
        ((self.x - rhs.x).powi(2) + (self.y - rhs.y).powi(2)).sqrt()
    }

    fn add(&self, other: &Point) -> Point {
        Point{x: self.x + other.x, y: self.y + other.y}
    }

    fn scalar_div(&self, scalar: &f64) -> Point {
        Point{x: self.x / scalar, y: self.y / scalar}
    }
}

impl Attributable for Point {
    fn attribute_names() -> String {
        return "x,y".to_string();
    }
    fn attribute_values(&self) -> String {
        return format!("{},{}", self.x, self.y).to_string();
    }
}

impl Random for Point {
    fn random() -> Point {
        let mut rng = rand::thread_rng();
        Point{
            x: rng.gen(), 
            y: rng.gen()
        }
    }

    // provide a box within which to generate points
    fn random_in_range(lower: &Point, upper: &Point) -> Point {
        let mut rng = rand::thread_rng();
        Point{
            x: rng.gen_range(lower.x, upper.x), 
            y: rng.gen_range(lower.y, upper.y)
        }
    }
}