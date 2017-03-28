//! # Introduction on mutator gene effect
//!
//! # Usage examples
//!
//! # Simulation details

// Coding conventions
#![deny(non_upper_case_globals)]
#![deny(non_camel_case_types)]
#![deny(non_snake_case)]
#![deny(unused_mut)]
//#![deny(missing_docs)]

// Features
#![feature(conservative_impl_trait)]
#![feature(custom_derive)]
#![feature(proc_macro)]

// Extern crates (libraries)
#[macro_use] extern crate clap;
extern crate pretty_env_logger;
extern crate futures;
extern crate tokio_core;

// Import required types, functions, traits etc
use futures::{Future};
use futures::future;
use tokio_core::reactor::Core;


// Conveniece macro used for specififcation of verbose only output
macro_rules! verbose {
    ( $x:expr, $is_verbose:expr ) => ( if $is_verbose { println!("{:?}", $x); } );
}

// Simulation settings and other configurable parameters
pub struct SimulationConfiguration {
    // model parameters
    mu1: f64,
    mu2: f64,
    alpha1: f64,
    alpha2: f64,

    // supported simulation settings
    timestep: Option<f64>, // timestep settings (constant value or None for auto)
}

// Simulation results representation along various derived statistics
pub struct SimulationResult {
    pub s: f64, // average fitness per capita
}

// Evolutionary model state representation, given fixed moment in time
pub struct State {
}

// Simulation instance
pub struct SimulationCore {

    // Constructor for mutator model

}

impl SimulationCore {

    // Single iteration
    fn iteration( &self, current_state : State ) -> State {
        State {} // return new state
    }

    // Computation loop logic
    fn run_simulation( &self, settings : SimulationConfiguration ) -> impl Future<Item = SimulationResult, Error = ()> {
        let timestep = settings.timestep;
        let result = SimulationResult{ s: 0.0 };
        future::ok( result ) // return result wrapped in future
    }

}

fn main() {
    // default values for model parameters
    let mu1_default: f64 = 1.0;
    let mu2_default: f64 = 2.0;
    let alpha1_default: f64 = 1.0;
    let alpha2_default: f64 = 0.0;

    // command line arguments and help
    let matches = clap_app!(genaddress =>
        (version: "1.0")
        (author: "Tatiana Iakushkina <tatiana.yakushkina@gmail.com>, Ilya Eriklintsev <erik.lite@gmail.com>")
        (about: "Runs numerical simulation of evolution process for population of specimen, which we assume to have genome that is approximated by binary string. We explicitly introduce presence of mutator gene that regulates mutation intensity for the rest of geneome.")
        (@arg mutation_rate_wild: --mu1 +takes_value "Sets mutation rate for wild type (lower mutation rate) replication, the rate is per capita per gene per second.")
        (@arg mutation_rate_mutator: --mu2 +takes_value "Sets mutation rate for mutator type (higher mutation rate).")
        (@arg transition_rate_wild_to_mutator: --alpha1 +takes_value "Sets transition rate for wild->mutator genotype change process, per capita per second.")
        (@arg transition_rate_mutator_to_wild: --alpha2 +takes_value "Sets transition rate for mutator->wild genotype change process.")
        (@arg verbose: -v --verbose "Enables intermidiate evolution iterations summary and makes output more verbose.")
    ).get_matches();
    let is_verbose = matches.is_present("verbose");

    // configure
    let setting = SimulationConfiguration {
        // extract arguments
        mu1 : matches.value_of("mutation_rate_wild").map_or( mu1_default, |x| x.parse().expect("Invalid mutation rate specified") ),
        mu2 : matches.value_of("mutation_rate_mutator").map_or( mu2_default, |x| x.parse().expect("Invalid mutation rate specified") ),
        alpha1: matches.value_of("transition_rate_wild_to_mutator").map_or( alpha1_default, |x| x.parse().expect("Invalid transition rate specified") ),
        alpha2: matches.value_of("transition_rate_mutator_to_wild").map_or( alpha2_default, |x| x.parse().expect("Invalid transition rate specified") ),
        timestep: None,
    };

    // run simulation
    verbose!( "Running in verbose mode", is_verbose);

    // Initialize logger for prettier console output and discard instantiated object afterwards
    let _ = pretty_env_logger::init();

    // Start tokio::Core reactor, object that will manage all concurrecy and threads in our application
    let mut core = Core::new().unwrap();

    // Start concurrent task doing actual computation
    let simulation_core = SimulationCore{};
    let result = core.run(simulation_core.run_simulation(setting)).unwrap();
}


// TO DO write tests
#[cfg(test)]
mod tests {
    use exchanges::bitmex::auth::generate_signature;
    use std::str;

    //TO DO
    #[test]
    fn iteration_test() {
        assert!(true);
    }

}
