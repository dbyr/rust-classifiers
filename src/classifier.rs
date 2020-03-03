use std::fs::File;
use std::f64;

pub trait EuclideanDistance {
    fn distance(&self, other: &Self) -> f64;
}

pub trait UnsupervisedClassifier<T: EuclideanDistance> {
    // trains the classifier using "data"
    // data: the data with which to train the classifier
    // return: the categories/centroids that are produced
    fn train(data: &Vec<T>) -> Vec<T>;

    // trains the classifier using the data in "file"
    // file: the file containing the data with which to train the classifier
    // return: the categories/centroids that are produced
    fn train_from_file(file: &File) -> Vec<T>;
}

// classifies "datum" into one of the provided "categories"
// datum: the data to classify
// categories: the categories into which to classify datum
// return: the index of the category most appropriate for datum, or -1 if categories size is 0
pub fn classify<T>(datum: &T, categories: &Vec<T>) -> isize 
where T: EuclideanDistance {
    let mut closest_dist = f64::MAX;
    let mut closest_cat: isize = -1;
    for (i, cat) in categories.iter().enumerate() {
        let cur_dist = datum.distance(cat);
        if cur_dist < closest_dist {
            closest_dist = cur_dist;
            closest_cat = i as isize;
        }
    }
    return closest_cat;
}