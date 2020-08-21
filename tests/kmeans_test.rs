extern crate rust_classifiers;

#[cfg(test)]
mod tests {
    use rust_classifiers::serial_classifiers::kmeans::KMeans;
    use rust_classifiers::serial_classifiers::unsupervised_classifier::UnsupervisedClassifier;
    use rust_classifiers::example_datatypes::point::Point;

    use rand::Rng;
    use rand::thread_rng;
    use rand::seq::SliceRandom;
    // use std::fs::File;

    #[test]
    fn test_train_default() {
        let classifier = train_generic_classifier();
        let sample_data = get_generic_data();

        // rust_classifiers::serial_classifiers::unsupervised_classifier::classify_csv(
        //     &classifier, 
        //     &File::create("training_tester_classes.csv").unwrap(), 
        //     &sample_data
        // ).unwrap();
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
    fn test_train_pp() {
        let classifier = train_pp_classifier();
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
    fn test_train_scalable() {
        let classifier = train_scalable_classifier();
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
    fn test_default_classify() {
        let classifier = train_generic_classifier();
        let point1 = Point::new(0.0, 160.0);
        let point2 = Point::new(-120.0, -1.0);

        let sample_data = get_generic_data();
        // sample_data.push(point1.clone());
        // sample_data.push(point2.clone());
        // rust_classifiers::serial_classifiers::unsupervised_classifier::classify_csv(
        //     &classifier, 
        //     &File::create("classify_tester_classes.csv").unwrap(), 
        //     &sample_data
        // ).unwrap();
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

    #[test]
    fn test_pp_classify() {
        let classifier = train_pp_classifier();
        let mut sample_data = get_generic_data();

        let point1 = Point::new(0.0, 160.0);
        let point2 = Point::new(-120.0, -1.0);
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
    }

    #[test]
    fn test_scalable_classify() {
        let classifier = train_scalable_classifier();
        let mut sample_data = get_generic_data();

        let point1 = Point::new(0.0, 160.0);
        let point2 = Point::new(-120.0, -1.0);
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
    }

    fn train_generic_classifier() -> KMeans<Point> {
        let mut classifier = KMeans::new(3);
        let data = get_generic_data();
        match classifier.train(data) {
            Ok(_s) => assert!(true),
            Err(_e) => assert!(false)
        }
        classifier
    }

    fn train_pp_classifier() -> KMeans<Point> {
        let mut classifier = KMeans::new_pp(3);
        let data = get_generic_data();
        match classifier.train(data) {
            Ok(_s) => assert!(true),
            Err(_e) => assert!(false)
        }
        classifier
    }

    fn train_scalable_classifier() -> KMeans<Point> {
        let mut classifier = KMeans::new_scalable(3);
        let data = get_generic_data();
        match classifier.train(data) {
            Ok(_s) => assert!(true),
            Err(_e) => assert!(false)
        }
        classifier
    }

    // generates some data that are of 3 well-separated sets
    fn get_generic_data() -> Vec<Point> {
        let mut generator = rand::thread_rng();
        let mut data = vec!(
            Point::new(-110.0, -2.0),
            Point::new(-120.0, -2.0),
            Point::new(0.0, 130.0),
            Point::new(0.5, 140.0),
            Point::new(160.0, 1.0),
            Point::new(170.0, 0.0)
        );
        // generate lots of data for each well-separated set
        let mut rand_data = Vec::new();
        for _ in 0..100 {
            rand_data.push(
                Point::new(
                    generator.gen_range(-120.0, -110.0),
                    generator.gen_range(-2.0, 0.0)
                )
            )
        }
        for _ in 0..100 {
            rand_data.push(
                Point::new(
                    generator.gen_range(0.0, 1.0),
                    generator.gen_range(130.0, 140.0)
                )
            )
        }
        for _ in 0..100 {
            rand_data.push(
                Point::new(
                    generator.gen_range(160.0, 170.0),
                    generator.gen_range(0.0, 1.0)
                )
            )
        }
        rand_data.shuffle(&mut thread_rng());
        data.append(&mut rand_data);
        data
    }
}