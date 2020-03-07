This repository will contain classifiers for the Rust programming language.

# KMeans Classifier
The K Means classifier included here has been built to be used with any data type the user might desire to use. In the example of use (main.rs), I have used a 2 dimensional Euclidean space data set. 

The main program itself will read data from the specified file and create a classifier from the data. The main method also includes a section that will write the results to the speicified file. The output can be customised based on the data type desired, but I have created a CSV formatted string from the Point type, and a method in the classifier module to print data out in CSV form.

# Python Script (./data/chart_cats.py)
This simple script can be used to display the graphed results of the classification effort of the given example in main.rs. This script will of course need to be modified if the data type is changed.