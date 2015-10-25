// XXX: Use a ref-counted gene Pool.
// A population is then only a collection of pool item ids.
// This prevents from duplicating potential large genes.
//
// Use typing to make sure that individuals are evaluated when required.
// fn evaluate(self, evaluator) -> EvaluatedPopulation
// fn add(self, ind) -> Population

#![feature(num_bits_bytes)]
extern crate bit_vec;
extern crate rand;
use rand::{Rng, Rand};
use std::cmp::PartialOrd;

pub mod bit_string;

/// A probability is a value in the range [0.0, 1.0].
/// A probability of 1.0 means `always`, that of 0.0
/// means `never`.
#[derive(Copy, Clone, Debug)]
pub struct Probability(f32);

impl Probability {
    #[inline]
    pub fn new(pb: f32) -> Probability {
        assert!(pb >= 0.0 && pb <= 1.0);
        Probability(pb)
    }
}

impl std::ops::Add<Probability> for Probability {
    type Output = Probability;
    #[inline]
    fn add(self, rhs: Probability) -> Probability {
        Probability::new(self.0 + rhs.0)
    }
}

/// A probability value is a random value in the
/// range [0.0, 1.0). 
#[derive(Copy, Clone, Debug)]
pub struct ProbabilityValue(f32);

impl Rand for ProbabilityValue {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> ProbabilityValue {
        let val = rng.gen::<f32>();
        assert!(val >= 0.0 && val < 1.0);
        ProbabilityValue(val)
    }
}

impl ProbabilityValue {
    pub fn is_probable_with(self, pb: Probability) -> bool {
        self.0 < pb.0
    }
}

/// Represents a fitness value.
pub trait Fitness: Clone {
    fn fitter_than(&self, other: &Self) -> bool;
}

/// Maximizes the fitness value as objective.
#[derive(Copy, Clone, Debug)]
pub struct MaxFitness<T:PartialOrd+Clone>(pub T);
impl<T:PartialOrd+Clone> Fitness for MaxFitness<T> {
    fn fitter_than(&self, other: &MaxFitness<T>) -> bool {
        self.0 > other.0
    }
}

/// Minimizes the fitness value as objective.
#[derive(Copy, Clone, Debug)]
pub struct MinFitness<T:PartialOrd+Clone>(pub T);
impl<T:PartialOrd+Clone> Fitness for MinFitness<T> {
    fn fitter_than(&self, other: &MinFitness<T>) -> bool {
        self.0 < other.0
    }
}

/// Represents an individual in a Population.
pub trait Individual: Clone {
}

/// Evaluates the fitness of an Individual.
pub trait Evaluator<I:Individual, F:Fitness> {
    fn fitness(&self, individual: &I) -> F;
}

/// Caches the `fitness` value for an individual.
#[derive(Clone, Debug)]
pub struct EvaluatedIndividual<I:Individual, F:Fitness> {
    individual: I,
    fitness: Option<F>,
}

impl<I:Individual, F:Fitness>  EvaluatedIndividual<I,F> {
    pub fn new(ind: I) -> EvaluatedIndividual<I,F> {
        EvaluatedIndividual{individual: ind, fitness: None}
    }
    pub fn fitness(&self) -> Option<F> {
        self.fitness.clone()
    }
    pub fn individual<'a>(&'a self) -> &'a I {
        &self.individual
    }
}

/// Manages a population of individuals.
#[derive(Clone, Debug)]
pub struct Population<I:Individual, F:Fitness> {
    population: Vec<EvaluatedIndividual<I,F>>
}

impl<I:Individual, F:Fitness> Population<I,F> {
    pub fn new() -> Population<I,F> {
        Population{population: Vec::new()}
    }

    pub fn with_capacity(capa: usize) -> Population<I,F> {
        Population{population: Vec::with_capacity(capa)}
    }

    pub fn len(&self) -> usize {
        self.population.len()
    }

    /// Return individual with best fitness.
    pub fn fittest(&self) -> Option<(usize, F)> {
        let mut fittest: Option<(usize, F)> = None;

        for (i, ref ind) in self.population.iter().enumerate() {
            if let Some(ref f) = ind.fitness {
                let mut is_better = true;
                if let Some((_, ref best_f)) = fittest {
                    if !f.fitter_than(best_f) {
                        is_better = false;
                    }
                }
                if is_better {
                    fittest = Some((i, f.clone()));
                }
            }
        }
        return fittest;
    }

    pub fn get_ref<'a>(&'a self, idx: usize) -> &'a EvaluatedIndividual<I, F> {
        &self.population[idx]
    }

    pub fn add_individual(&mut self, ind: I) {
        self.population.push(EvaluatedIndividual::new(ind));
    }

    pub fn add(&mut self, ind: EvaluatedIndividual<I,F>) {
        self.population.push(ind);
    }

    /// Evaluates the whole population, i.e. determines the fitness of
    /// each `individual` (unless already calculated).
    /// Returns the number of evaluations performed.
    pub fn evaluate<E>(&mut self, evaluator: &E) -> usize where E:Evaluator<I,F> {
        let mut nevals = 0;
        for i in self.population.iter_mut() {
            if i.fitness.is_some() { continue; }
            i.fitness = Some(evaluator.fitness(&i.individual));
            nevals += 1;
        }
        return nevals;
    }

    fn add_population(&mut self, p: &Population<I,F>) {
        for ind in p.population.iter() {
            self.add(ind.clone());
        }
    }
}

/// Select the best individual out of `k` randomly choosen.
/// This gives individuals with better fitness a higher chance to reproduce.
/// `n` is the total number of individuals.
///
/// NOTE: We are not using `sample(rng, 0..n, k)` as it is *very* expensive.
/// Instead we call `rng.gen_range()` k-times. The drawn items could be the same,
/// but the probability is very low if `n` high compared to `k`.
pub fn tournament_selection<R:Rng, F, E>(rng: &mut R, evaluate: E, n: usize, k: usize) -> Option<(usize, F)>
  where F:Fitness,
        E:Fn(usize) -> F {
    assert!(n >= k);

    let mut best: Option<(usize, F)> = None;

    for _ in 0..k {
        let i = rng.gen_range(0, n);
        let fitness = evaluate(i);
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

pub enum VariationMethod {
    Crossover,
    Mutation,
    Reproduction,
}

/// Mates two individual, producing two children.
pub trait OpCrossover<I: Individual> {
    fn crossover(&mut self, parent1: &I, parent2: &I) -> (I, I);
}

/// Mutates an individual.
pub trait OpMutate<I: Individual> {
    fn mutate(&mut self, ind: &I) -> I;
}

/// Selects a variation method to use.
pub trait OpVariation {
    fn variation(&mut self) -> VariationMethod;
}

/// Selects a random individual from the population.
pub trait OpSelectRandomIndividual<I: Individual, F: Fitness> {
    fn select_random_individual<'a>(&mut self, population: &'a Population<I, F>) -> &'a EvaluatedIndividual<I, F>;
}

/// Produce new generation through selection of \mu individuals from population.
pub trait OpSelect<I: Individual, F: Fitness> {
    fn select(&mut self, population: &Population<I,F>, mu: usize) -> Population<I,F>;
}

pub fn variation_or<I, F, T>(toolbox: &mut T, population: &Population<I,F>, lambda: usize) -> Population<I,F>
where I: Individual,
      F: Fitness,
      T: OpCrossover<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I,F>
{
    let mut offspring = Population::with_capacity(lambda);
    // Each step produces exactly one child.
    for _ in 0..lambda {
        let method = toolbox.variation();
        let child = match method {
            VariationMethod::Crossover => {
                // select two individuals and mate them.
                // only the first offspring is used, the second is thrown away.
                let parent1 = toolbox.select_random_individual(population);
                let parent2 = toolbox.select_random_individual(population);
                let (child1, _child2) = toolbox.crossover(&parent1.individual, &parent2.individual);
                EvaluatedIndividual::new(child1)
            }
            VariationMethod::Mutation => {
                // select a single individual and mutate it. 
                let ind = toolbox.select_random_individual(population);
                let child = toolbox.mutate(&ind.individual);
                EvaluatedIndividual::new(child)
            }
            VariationMethod::Reproduction => {
                let ind = toolbox.select_random_individual(population);
                ind.clone()
            }
        };
        offspring.add(child);
    }
    return offspring;
}

// The (\mu + \lambda) algorithm.
// From `population`, \lambda offspring is produced, through either
// mutation, crossover or random reproduction.
// For the next generation, \mu individuals are selected from the \mu + \lambda (parents and offspring).
pub fn ea_mu_plus_lambda<I,F,T,E,S>(toolbox: &mut T, evaluator: &E, population: &Population<I,F>, mu: usize, lambda: usize, num_generations: usize, stat: S)
    -> Population<I,F>
where I: Individual,
      F: Fitness,
      T: OpCrossover<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I,F> + OpSelect<I,F>,
      E: Evaluator<I,F>,
      S: Fn(usize, &Population<I,F>)
{
    let mut p = population.clone();
    p.evaluate(evaluator);
    stat(0, &p);

    for gen in 0..num_generations {
        // evaluate population. make sure that every individual has been rated.
        let mut offspring = variation_or(toolbox, &p, lambda);
        offspring.add_population(&p);
        offspring.evaluate(evaluator);
        // select from offspring the `best` individuals
        p = toolbox.select(&offspring, mu); 
        stat(gen+1, &p);
    }

    return p; 
}
