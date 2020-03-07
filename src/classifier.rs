use std::fs::File;
use std::f64;
use std::io::{
    Write,
    LineWriter
};
use std::io::Result as IOResult;

use crate::euclidean_distance::EuclideanDistance;

pub trait UnsupervisedClassifier<T: EuclideanDistance> {
    // trains the classifier using "data"
    // data: the data with which to train the classifier
    // return: the categories/centroids that are produced
    fn train(&mut self, data: &Vec<T>) -> Result<Vec<T>, TrainingError>;

    // trains the classifier using the data in "file"
    // file: the file containing the data with which to train the classifier
    // parser: a method to read T's from the file. returns a result with an option (which
    //     should contain a value read from the file, or none if there are no more
    //     values to read) on success, or a TrainingError if the file is invalid
    // return: the categories/centroids that are produced
    fn train_from_file(
        &mut self, 
        file: &mut File, 
        parser: &Fn(&Vec<u8>) -> Result<T, TrainingError>
    ) -> Result<Vec<T>, TrainingError>;
}

pub trait Attributable {
    // returns a string demonstrating the attributes of this struct
    // returns: attribute name string
    fn attribute_names() -> String;

    // returns this object's values in a corresponding format to attribute_names
    // returns: attribute values string
    fn attribute_values(&self) -> String;
}

pub enum TrainingError {
    InvalidData,
    InvalidClassifier,
    InvalidFile,

    FileReadFailed,
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

// classifies all the data in data, and saves the classified data to file
// file: the file to save the classifications
// data: the data to classify
// categories: the categories into which to classify the data
// return: whether the file was successfully created
pub fn classify_csv<T>(file: &File, data: &Vec<T>, categories: &Vec<T>) -> IOResult<()>
where T: EuclideanDistance + Attributable {
    let mut writer = LineWriter::new(file);
    writer.write_all(T::attribute_names().as_bytes())?;
    writer.write_all(b",cat\n")?;

    for val in data {
        writer.write_all(val.attribute_values().as_bytes())?;
        let cat = classify(val, categories);
        writer.write_all(format!(",{}\n", cat).as_bytes())?;
    }
    writer.flush()?;
    return Ok(());
}