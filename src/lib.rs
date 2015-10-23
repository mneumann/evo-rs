#![feature(num_bits_bytes)]
extern crate bit_vec;
extern crate rand;
use rand::{Rng, sample};

pub mod bit_string;

/// Represents a fitness value.
pub trait Fitness {
    fn fitter_than(&self, other: &Self) -> bool;
}

/// Represents an individual in a Population.
pub trait Individual {
}

/// Evaluates the fitness of an Individual.
trait Evaluator<I:Individual, F:Fitness> {
    fn fitness(&mut self, individual: &I) -> F;
}

/// Caches the `fitness` value for an individual.
struct EvaluatedIndividual<I:Individual, F:Fitness> {
    individual: I,
    fitness: Option<F>,
}

/// Manages a population of individuals.
pub struct Population<I:Individual, F:Fitness> {
    population: Vec<EvaluatedIndividual<I,F>>
}

/// Select the best individual out of `k` randomly choosen.
/// `n` is the total number of individuals.
pub fn tournament_selection<R:Rng, F, E>(rng: &mut R, eval: E, n: usize, k: usize) -> Option<(usize, F)>
where F:Fitness,
      E:Fn(usize) -> F {
    assert!(n >= k);

    let mut best: Option<(usize, F)> = None;

    let sample = sample(rng, 0..n, k);
    for i in sample {
        let fitness = eval(i);
        let better = match best {
            Some((_, ref current_best_fitness)) => {
                fitness.fitter_than(current_best_fitness)
            }
            None => { true }
       };
       if better {
           best = Some((i, fitness));
       }
    }
    return best;
}
