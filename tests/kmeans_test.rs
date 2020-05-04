extern crate rust_classifiers;

#[cfg(test)]
mod tests {
    use rust_classifiers::serial_classifiers::kmeans::KMeans;
    use rust_classifiers::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier;
    use rust_classifiers::example_datatypes::point::Point;
    use rust_classifiers::euclidean_distance::EuclideanDistance;

    #[test]
    fn test_train() {
        let mut classifier = train_generic_classifier();
        let sample_data = get_generic_data();

        assert_eq!(
            classifier.classify(&sample_data[0]), 
            classifier.classify(&sample_data[1])
        );
        assert_eq!(
            classifier.classify(&sample_data[2]), 
            classifier.classify(&sample_data[3])
        );
        assert_eq!(
            classifier.classify(&sample_data[4]), 
            classifier.classify(&sample_data[5])
        );
        assert_ne!(
            classifier.classify(&sample_data[0]), 
            classifier.classify(&sample_data[2])
        );
        assert_ne!(
            classifier.classify(&sample_data[0]), 
            classifier.classify(&sample_data[5])
        );
        assert_ne!(
            classifier.classify(&sample_data[2]), 
            classifier.classify(&sample_data[5])
        );
    }

    #[test]
    fn test_classify() {
        let mut classifier = train_generic_classifier();
        let sample_data = get_generic_data();

        let point1 = Point::new(-1.0, 4.0);
        let point2 = Point::new(-0.5, -0.5);
        assert_eq!(
            classifier.classify(&point1),
            classifier.classify(&sample_data[2])
        );
        assert_eq!(
            classifier.classify(&point2),
            classifier.classify(&sample_data[0])
        );
        assert_ne!(
            classifier.classify(&point1),
            classifier.classify(&point2)
        );
    }

    fn train_generic_classifier() -> KMeans<Point> {
        let mut classifier = KMeans::new(3);
        let data = get_generic_data();
        match classifier.train(&data) {
            Ok(_s) => assert!(true),
            Err(_e) => assert!(false)
        }
        classifier
    }

    // generates some data that are of 3 very obviously
    // distinct groups
    fn get_generic_data() -> Vec<Point> {
        vec!(
            Point::new(-1.0, -2.0),
            Point::new(-2.0, -1.0),
            Point::new(0.0, 3.0),
            Point::new(0.5, 2.0),
            Point::new(2.0, 1.0),
            Point::new(3.0, 0.0)
        )
    }
}