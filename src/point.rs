use crate::classifier::EuclideanDistance;

#[derive(Debug, PartialEq, Default)]
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
}