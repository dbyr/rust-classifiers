pub trait Attributable {
    // returns a string naming the attributes of this struct
    // returns: attribute name string
    fn attribute_names() -> String;

    // returns this object's values in a corresponding format to attribute_names
    // returns: attribute values string
    fn attribute_values(&self) -> String;
}

#[derive(PartialEq, Debug)]
pub enum TrainingError {
    InvalidData,
    InvalidClassifier,
    InvalidFile,

    FileReadFailed,
}

#[derive(PartialEq, Debug)]
pub enum ClassificationError {
    ClassifierNotTrained,
    ClassifierInvalid
}