use std::fs::File;

pub trait Attributable<T> {
    // returns a string naming the attributes of this struct
    // returns: attribute name string
    fn attribute_names() -> String;

    // returns this object's values in a corresponding format to attribute_names
    // returns: attribute values string
    fn attribute_values(&self) -> String;
}

#[macro_export]
macro_rules! impl_attributable {
    ( $wrapper:ident[$t:ident; $size:expr] ) => {
        impl Attributable<$wrapper> for [$t; $size] {
            fn attribute_names() -> String {
                let mut result = String::from("a0");
                for i in 1..$size {
                    result.push_str(&format!(",{}", i));
                }
                result
            }
            fn attribute_values(&self) -> String {
                let mut result = String::from(format!("{}", self[0]));
                for i in 1..$size {
                    result.push_str(&format!(",{}", self[i]));
                }
                result
            }
        }
    };
}

pub trait Savable {
    // saves this object to the given file
    // file: the file to which to save the object
    fn save_to_file(&self, file: File);

    // loads an object of this type from a file
    // file: the file from which to load an object
    fn load_from_file(file: File) -> Self;
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