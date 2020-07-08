extern crate rust_classifiers;

use rust_classifiers::mpi_classifiers::kmeans::tests;
use mpi::topology::SystemCommunicator;

macro_rules! tests_vec {
    ($($func:path),+) => {
        vec!(
            $(Box::new($func)),+
        )
    }
}

fn enum_tests() -> Vec<Box<dyn Fn(&SystemCommunicator)>> {
    tests_vec!(
        tests::test_convergence
    )
}

/// Can't run tests with cargo because it requires
/// MPI, and there's no functionality in cargo-mpirun
/// that allows running tests, but it does allow running
/// examples, so use an example to run the tests, which
/// are public and only included when debug is enabled
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    for test in enum_tests() {
        test(&world);
    }
}
