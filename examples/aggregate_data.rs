extern crate rust_classifiers;
use rust_classifiers::example_datatypes::point::Point;
use rust_classifiers::euclidean_distance::EuclideanDistance;

use timely::dataflow::operators::{Inspect, Broadcast};
use timely::dataflow::operators::map::Map;
use timely::dataflow::operators::Accumulate;
use timely::dataflow::*;
use timely::dataflow::operators::{Input, Exchange, Probe};

fn main() {
    timely::execute_from_args(std::env::args(), |worker| {
        let mut input = InputHandle::new();
        let mut probe = ProbeHandle::new();
        let index = worker.index();

        worker.dataflow(|scope| {
            scope.input_from(&mut input)
                .broadcast() // everyone sends their local totals to each other
                .accumulate( // everyone calculates the new global means
                    vec!(),
                    move |totals: &mut Vec<(Point, usize)>, 
                    locals: timely_communication::message::RefOrMut<Vec<(Point, usize, usize)>>| {
                        for pair in locals.iter() {
                            let i = pair.2;
                            if totals.len() <= i {
                                totals.push((pair.0.clone(), pair.1));
                            } else {
                                totals[i].0.add(&pair.0);
                                totals[i].1 += pair.1;
                            }
                        }
                    }
                )
                .map(|point_sums| {
                    point_sums
                    .into_iter()
                    .map(|point_sum| point_sum.0.scalar_div(&(point_sum.1 as f64)))
                    .collect::<Vec<Point>>()
                })
                .inspect(move |v| println!("worker {} sees {:?}", index, v))
                .probe_with(&mut probe);
        });

        for i in 0..10 {
            println!("worker {} sending round {}", index, i);
            let means = vec!(
                // these pairs will be the "sum" and "count" of values
                // associated with each mean (in this case, two)
                (Point::new(i as f64+5.0, i as f64), index + 1), // first mean
                (Point::new(-i as f64-5.0, i as f64), index + 1) // second mean
            );
            for (i, mean) in means.into_iter().enumerate() {
                input.send((mean.0, mean.1, i));
            }
            input.advance_to(input.epoch() + 1);
            while probe.less_than(input.time()) {
                worker.step();
            }
        }
    }).unwrap();
}
