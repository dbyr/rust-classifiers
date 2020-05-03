extern crate rust_classifiers;

#[cfg(test)]
mod tests {
    use rust_classifiers::point::Point;
    use rust_classifiers::common::TrainingError;

    #[test]
    fn test_point_from_vec() {
        // test integers
        let expected = Point::new(1.0, 2.0);
        let actual = Point::point_from_vec(&"1 2".to_owned().into_bytes());
        assert_eq!(Ok(expected), actual, "Point::point_from_vec failed: test integers");
    
        // test floats
        let expected = Point::new(1.3, 2.75);
        let actual = Point::point_from_vec(&"1.3 2.75".to_owned().into_bytes());
        assert_eq!(Ok(expected), actual, "Point::point_from_vec failed: test floats");
    
        // test negatives
        let expected = Point::new(-1.0, 2.0);
        let actual = Point::point_from_vec(&"-1 2".to_owned().into_bytes());
        assert_eq!(Ok(expected), actual, "Point::point_from_vec failed: test negatives");
    
        // test failures
        let expected = TrainingError::InvalidData;
        let actual = Point::point_from_vec(&"abc".to_owned().into_bytes());
        assert_eq!(Err(expected), actual, "Point::point_from_vec failed: test failures");
    }
}