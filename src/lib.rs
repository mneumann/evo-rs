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
extern crate simple_parallel;
use rand::{Rand, Rng};
use std::cmp::PartialOrd;
use simple_parallel::Pool;

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
    #[inline]
    pub fn is_probable_with(self, pb: Probability) -> bool {
        self.0 < pb.0
    }
}

/// Represents a fitness value.
pub trait Fitness: Clone+Send {
    fn fitter_than(&self, other: &Self) -> bool;
}

/// Maximizes the fitness value as objective.
#[derive(Copy, Clone, Debug)]
pub struct MaxFitness<T: PartialOrd + Clone + Send>(pub T);
impl<T:PartialOrd+Clone+Send> Fitness for MaxFitness<T> {
    #[inline(always)]
    fn fitter_than(&self, other: &MaxFitness<T>) -> bool {
        self.0 > other.0
    }
}

/// Minimizes the fitness value as objective.
#[derive(Copy, Clone, Debug)]
pub struct MinFitness<T: PartialOrd + Clone + Send>(pub T);
impl<T:PartialOrd+Clone+Send> Fitness for MinFitness<T> {
    #[inline(always)]
    fn fitter_than(&self, other: &MinFitness<T>) -> bool {
        self.0 < other.0
    }
}

/// Represents an individual in a Population.
pub trait Individual: Clone+Send {
}

/// Evaluates the fitness of an Individual.
pub trait Evaluator<I:Individual, F:Fitness>: Sync {
    fn fitness(&self, individual: &I) -> F;
}

/// Caches the `fitness` value for an individual.
#[derive(Clone, Debug)]
pub struct EvaluatedIndividual<I: Individual, F: Fitness> {
    individual: I,
    fitness: Option<F>,
}

impl<I:Individual, F:Fitness>  EvaluatedIndividual<I,F> {
    pub fn new(ind: I) -> EvaluatedIndividual<I, F> {
        EvaluatedIndividual {
            individual: ind,
            fitness: None,
        }
    }
}

/// Manages a population of individuals.
#[derive(Clone, Debug)]
pub struct Population<I: Individual, F: Fitness> {
    population: Vec<EvaluatedIndividual<I, F>>,
}

// XXX: Have Population and EvaluatedPopulation. This avoids having two arrays.
impl<I:Individual, F:Fitness> Population<I,F>
{
    pub fn new() -> Population<I, F> {
        Population { population: Vec::new() }
    }

    pub fn with_capacity(capa: usize) -> Population<I, F> {
        Population { population: Vec::with_capacity(capa) }
    }

    #[inline(always)]
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

    #[inline]
    pub fn get_individual(&self, idx: usize) -> &I {
        &self.population[idx].individual
    }

    #[inline]
    pub fn get_fitness(&self, idx: usize) -> F {
        (&self.population[idx]).fitness.clone().unwrap()
    }

    pub fn add_individual(&mut self, ind: I) {
        self.population.push(EvaluatedIndividual::new(ind));
    }

    pub fn add_individual_with_fitness(&mut self, ind: I, fitness: F) {
        self.population.push(EvaluatedIndividual{individual: ind, fitness: Some(fitness)});
    }

    /// Evaluates the whole population, i.e. determines the fitness of
    /// each `individual` (unless already calculated).
    /// Returns the number of evaluations performed.
    pub fn evaluate<E>(&mut self, evaluator: &E) -> usize
        where E: Evaluator<I, F>
    {
        let mut nevals = 0;
        for i in self.population.iter_mut() {
            if i.fitness.is_some() {
                continue;
            }
            i.fitness = Some(evaluator.fitness(&i.individual));
            nevals += 1;
        }
        return nevals;
    }

    /// Evaluate the population in parallel using the threadpool `pool`.
    pub fn evaluate_in_parallel<E>(&mut self, evaluator: &E, pool: &mut Pool, chunk_size: usize) -> usize
        where E: Evaluator<I, F>
    {
        let mut nevals = 0;
        for i in self.population.iter() {
            if i.fitness.is_none() {
                nevals += 1;
            }
        }

        // XXX split population into two arrays. one evaluated, one not evaluated. this should speed up parallel evaluation a lot.
        pool.for_(self.population.chunks_mut(chunk_size), |chunk| {
            for ind in chunk.iter_mut() {
                if ind.fitness.is_some() { continue; }
                ind.fitness = Some(evaluator.fitness(&ind.individual));
            }
        });

        return nevals;
    }

    fn extend_with(&mut self, p: Population<I, F>) {
        self.population.extend(p.population);
    }
}

/// Select the best individual out of `k` randomly choosen.
/// This gives individuals with better fitness a higher chance to reproduce.
/// `n` is the total number of individuals.
///
/// NOTE: We are not using `sample(rng, 0..n, k)` as it is *very* expensive.
/// Instead we call `rng.gen_range()` k-times. The drawn items could be the same,
/// but the probability is very low if `n` high compared to `k`.
#[inline]
pub fn tournament_selection<R: Rng, F, E>(rng: &mut R,
                                          fitness: E,
                                          n: usize,
                                          k: usize)
                                          -> Option<(usize, F)>
    where F: Fitness,
          E: Fn(usize) -> F
{
    assert!(n >= k);

    let mut best: Option<(usize, F)> = None;

    for _ in 0..k {
        let i = rng.gen_range(0, n);
        let fitness = fitness(i);
        let better = match best {
            Some((_, ref current_best_fitness)) => {
                fitness.fitter_than(current_best_fitness)
            }
            None => {
                true
            }
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

/// Mates two individual, producing one child.
pub trait OpCrossover1<I: Individual> {
    fn crossover1(&mut self, parent1: &I, parent2: &I) -> I;
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
    fn select_random_individual<'a>(&mut self,
                                    population: &'a Population<I, F>)
                                    -> usize; // IndividualIndex
}

/// Produce new generation through selection of \mu individuals from population.
pub trait OpSelect<I: Individual, F: Fitness> {
    fn select(&mut self, population: &Population<I, F>, mu: usize) -> Population<I, F>;
}

pub fn variation_or<I, F, T>(toolbox: &mut T,
                             population: &Population<I, F>,
                             lambda: usize)
                             -> (Population<I,F>, Population<I,F>)
    where I: Individual,
          F: Fitness,
          T: OpCrossover1<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I, F>
{
    // We assume that most offspring is unrated and only a small amount of already rated offspring.
    let mut unrated_offspring = Population::with_capacity(lambda);
    let mut rated_offspring = Population::new();

    // Each step produces exactly one child.
    for _ in 0..lambda {
        let method = toolbox.variation();
        match method {
            VariationMethod::Crossover => {
                // select two individuals and mate them.
                // only the first offspring is used, the second is thrown away.
                let parent1_idx = toolbox.select_random_individual(population);
                let parent2_idx = toolbox.select_random_individual(population);
                let child1 = toolbox.crossover1(population.get_individual(parent1_idx), population.get_individual(parent2_idx));
                unrated_offspring.add_individual(child1);
            }
            VariationMethod::Mutation => {
                // select a single individual and mutate it.
                let ind_idx = toolbox.select_random_individual(population);
                let child = toolbox.mutate(population.get_individual(ind_idx));
                unrated_offspring.add_individual(child);
            }
            VariationMethod::Reproduction => {
                let ind_idx = toolbox.select_random_individual(population);
                rated_offspring.add_individual_with_fitness(population.get_individual(ind_idx).clone(), population.get_fitness(ind_idx));
            }
        }
    }
    return (unrated_offspring, rated_offspring);
}

// The (\mu + \lambda) algorithm.
// From `population`, \lambda offspring is produced, through either
// mutation, crossover or random reproduction.
// For the next generation, \mu individuals are selected from the \mu + \lambda
// (parents and offspring).
#[inline]
pub fn ea_mu_plus_lambda<I,F,T,E,S>(toolbox: &mut T, evaluator: &E, mut population: Population<I,F>, mu: usize, lambda: usize, num_generations: usize, stat: S, numthreads: usize, chunksize: usize)
    -> Population<I,F>
where I: Individual,
      F: Fitness,
      T: OpCrossover1<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I,F> + OpSelect<I,F>,
      E: Evaluator<I,F>+Sync,
      S: Fn(usize, usize, &Population<I,F>)
{
    let mut pool = simple_parallel::Pool::new(numthreads);

    let nevals = population.evaluate_in_parallel(evaluator, &mut pool, chunksize);
    stat(0, nevals, &population);

    for gen in 0..num_generations {
        // evaluate population. make sure that every individual has been rated.
        let (mut unrated_offspring, rated_offspring) = variation_or(toolbox, &population, lambda);

        let nevals = unrated_offspring.evaluate_in_parallel(evaluator, &mut pool, chunksize);

        population.extend_with(unrated_offspring); // this is now rated.
        population.extend_with(rated_offspring);

        // select from offspring the `best` individuals
        let p = toolbox.select(&population, mu);
        stat(gen + 1, nevals, &p);
        population = p;
    }

    return population;
}
