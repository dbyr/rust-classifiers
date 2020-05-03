use rand::Rng;

use crate::euclidean_distance::EuclideanDistance;
use crate::random::Random;
use crate::common::{
    TrainingError,
    Attributable
};

#[derive(Debug, PartialEq, Default, Clone)]
pub struct Point {
    x: f64,
    y: f64
}

impl Point{
    pub fn new(x: f64, y: f64) -> Point {
        Point{x: x, y: y}
    }
    pub fn get_x(&self) -> f64 {
        return self.x;
    }
    pub fn get_y(&self) -> f64 {
        return self.y;
    }


    pub fn point_from_vec(buff: &Vec<u8>) -> Result<Point, TrainingError> {
        // let buf = buff.into_bytes();
        let mut val = 0.0;
        let mut sign = 1.0;
        let mut multiplier = 1.0;
        let mut addition;
        let mut post_decimal = false;
        let mut first_char = true;
        let mut x = 0.0;
        let y;
        for c in buff.iter() {
            // only match numbers (and delimit on space)
            if *c == 45u8 && first_char {
                first_char = false;
                sign = -1.0;
            } else if *c == 46u8 && !post_decimal {
                post_decimal = true;
                continue;
            } else if *c >= 48u8 && *c <= 57u8 {
                // make sure decimals are handled correctly
                addition = (*c - 48u8) as f64;
                if post_decimal {
                    multiplier *= 10.0;
                    addition /= multiplier;
                } else {
                    val *= 10.0;
                }
                val += addition;
            } else if *c == 32u8 {
                val *= sign;
                x = val;
                val = 0.0;
                sign = 1.0;
                multiplier = 1.0;
                post_decimal = false;
                first_char = true;
            } else {
                return Err(TrainingError::InvalidData);
            }
        }
        val *= sign;
        y = val;
        return Ok(Point::new(x, y));
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