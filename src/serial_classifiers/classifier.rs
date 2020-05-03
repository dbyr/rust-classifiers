use std::fs::File;
use std::io::{
    Error,
    ErrorKind
};
use std::io::{
    Write,
    LineWriter
};
use std::io::Result as IOResult;

use crate::euclidean_distance::EuclideanDistance;
use crate::common::{
    TrainingError,
    ClassificationError,
    Attributable
};

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

    // classifies a datum using the trained classifier
    // datum: the data one wishes to classify
    // return: the category to which this datum belongs
    fn classify(&self, datum: &T) -> Result<usize, ClassificationError>;
}

// classifies all the data in data, and saves the classified data to file
// file: the file to save the classifications
// data: the data to classify
// categories: the categories into which to classify the data
// return: whether the file was successfully created
pub fn classify_csv<T>(
    classifier: &dyn UnsupervisedClassifier<T>, 
    file: &File, 
    data: &Vec<T>
) -> IOResult<()>
where T: EuclideanDistance + Attributable {
    let mut writer = LineWriter::new(file);
    writer.write_all(T::attribute_names().as_bytes())?;
    writer.write_all(b",cat\n")?;

    for val in data {
        writer.write_all(val.attribute_values().as_bytes())?;
        let cat;
        let class = match classifier.classify(val) {
            Ok(v) => v as i64,
            Err(_) => -1
        };
        if class >= 0 {
            cat = class as usize;
        } else {
            return Err(Error::new(
                ErrorKind::Other, "CSV creation failed: could not classify some data"
            ));
        }
        writer.write_all(format!(",{}\n", cat).as_bytes())?;
    }
    writer.flush()?;
    return Ok(());
}