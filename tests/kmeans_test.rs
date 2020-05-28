extern crate rust_classifiers;

#[cfg(test)]
mod tests {
    use rust_classifiers::serial_classifiers::kmeans::KMeans;
    use rust_classifiers::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier;
    use rust_classifiers::example_datatypes::point::Point;
    use rust_classifiers::serial_classifiers::unsupervised_classifier::classify_csv;
    use std::fs::File;

    #[test]
    fn test_train() {
        let classifier = train_generic_classifier();
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
        classify_csv(&classifier, &File::create("test_train.csv").unwrap(), &sample_data);
    }

    #[test]
    fn test_classify() {
        let classifier = train_generic_classifier();
        let mut sample_data = get_generic_data();

        let point1 = Point::new(-1.0, 160.0);
        let point2 = Point::new(-120.0, -0.5);
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
        sample_data.push(point1);
        sample_data.push(point2);
        classify_csv(&classifier, &File::create("test_classify.csv").unwrap(), &sample_data);
    }

    fn train_generic_classifier() -> KMeans<Point> {
        let mut classifier = KMeans::new_pp(3);
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
            Point::new(-110.0, -2.0),
            Point::new(-120.0, -2.0),
            Point::new(0.0, 130.0),
            Point::new(0.5, 140.0),
            Point::new(160.0, 1.0),
            Point::new(170.0, 0.0)
        )
    }
}