// XXX: Use a ref-counted gene Pool.
// A population is then only a collection of pool item ids.
// This prevents from duplicating potential large genes.

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
#[derive(Copy, Clone, Debug, Default)]
pub struct MaxFitness<T: PartialOrd + Clone + Send + Default>(pub T);
impl<T:PartialOrd+Clone+Send+Default> Fitness for MaxFitness<T> {
    #[inline(always)]
    fn fitter_than(&self, other: &MaxFitness<T>) -> bool {
        self.0 > other.0
    }
}

/// Minimizes the fitness value as objective.
#[derive(Copy, Clone, Debug, Default)]
pub struct MinFitness<T: PartialOrd + Clone + Send + Default>(pub T);
impl<T:PartialOrd+Clone+Send+Default> Fitness for MinFitness<T> {
    #[inline(always)]
    fn fitter_than(&self, other: &MinFitness<T>) -> bool {
        self.0 < other.0
    }
}

/// Represents an individual in a Population.
pub trait Individual: Clone+Send {
}

/// Manages a population of (unrated) individuals.
#[derive(Clone, Debug)]
pub struct UnratedPopulation<I: Individual> {
    population: Vec<I>,
}

/// Manages a population of rated individuals.
#[derive(Clone, Debug)]
pub struct RatedPopulation<I: Individual, F:Fitness> {
    rated_population: Vec<(I,F)>,
}

impl<I:Individual> UnratedPopulation<I>
{
    pub fn new() -> UnratedPopulation<I> {
        UnratedPopulation { population: Vec::new() }
    }

    pub fn with_capacity(capa: usize) -> UnratedPopulation<I> {
        UnratedPopulation { population: Vec::with_capacity(capa) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.population.len()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &I {
        &self.population[idx]
    }

    pub fn add(&mut self, ind: I) {
        self.population.push(ind);
    }

    fn extend_with(&mut self, p: UnratedPopulation<I>) {
        self.population.extend(p.population);
    }

    /// Evaluates the whole population, i.e. determines the fitness of
    /// each `individual` (unless already calculated).
    /// Returns the rated population.
    pub fn rate<E, F>(self, evaluator: &E) -> RatedPopulation<I, F>
        where E: Evaluator<I, F>,
              F: Fitness
    {
        let len = self.population.len();
        let rated_population : Vec<(I,F)> = self.population.into_iter().map(|ind| {
            let fitness = evaluator.fitness(&ind);
            (ind, fitness)
        }).collect();
        debug_assert!(rated_population.len() == len);
        RatedPopulation {rated_population: rated_population}
    }

    /// Evaluate the population in parallel using the threadpool `pool`.
    pub fn rate_in_parallel<E, F>(self, evaluator: &E, pool: &mut Pool, chunk_size: usize) -> RatedPopulation<I, F>
        where E: Evaluator<I, F>,
              F: Fitness+Default,
    {
        let len = self.population.len();
        let mut rated_population : Vec<(I,F)> = self.population.into_iter().map(|ind| {
            let fitness = F::default(); 
            (ind, fitness)
        }).collect();

        pool.for_(rated_population.chunks_mut(chunk_size), |chunk| {
            for &mut (ref ind, ref mut fitness) in chunk.iter_mut() {
                *fitness = evaluator.fitness(ind);
            }
        });

        debug_assert!(rated_population.len() == len);
        RatedPopulation {rated_population: rated_population}
    }

}

impl<I:Individual, F:Fitness> RatedPopulation<I, F>
{
    pub fn new() -> RatedPopulation<I, F> {
        RatedPopulation { rated_population: Vec::new() }
    }

    pub fn with_capacity(capa: usize) -> RatedPopulation<I, F> {
        RatedPopulation { rated_population: Vec::with_capacity(capa) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.rated_population.len()
    }

    pub fn add(&mut self, ind: I, fitness: F) {
        self.rated_population.push((ind, fitness));
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &I {
        &self.rated_population[idx].0
    }

    #[inline]
    pub fn get_individual(&self, idx: usize) -> &I {
        self.get(idx)
    }

    #[inline]
    pub fn get_fitness(&self, idx: usize) -> &F {
        &self.rated_population[idx].1
    } 

    fn extend_with(&mut self, p: RatedPopulation<I, F>) {
        self.rated_population.extend(p.rated_population);
    }

    #[inline]
    pub fn fitter_than(&self, i1: usize, i2: usize) -> bool {
        self.get_fitness(i1).fitter_than(self.get_fitness(i2))
    }

    /// Return index of individual with best fitness.
    pub fn fittest(&self) -> usize {
        assert!(self.len() > 0);
        let mut fittest = 0;

        for i in 1 .. self.rated_population.len() {
            if self.rated_population[i].1.fitter_than(&self.rated_population[fittest].1) {
                fittest = i;
            }
        }

        return fittest;
    }

}

/// Evaluates the fitness of an Individual.
pub trait Evaluator<I:Individual, F:Fitness>: Sync {
    fn fitness(&self, individual: &I) -> F;
}


/// Select the best individual out of `k` randomly choosen.
/// This gives individuals with better fitness a higher chance to reproduce.
/// `n` is the total number of individuals.
///
/// NOTE: We are not using `sample(rng, 0..n, k)` as it is *very* expensive.
/// Instead we call `rng.gen_range()` k-times. The drawn items could be the same,
/// but the probability is very low if `n` is high compared to `k`.
#[inline]
pub fn tournament_selection_fast<R: Rng, F>(rng: &mut R,
                                          better_than: F,
                                          n: usize,
                                          k: usize)
                                          -> usize
    where F: Fn(usize, usize) -> bool
{
    assert!(n > 0);
    assert!(k > 0);
    assert!(n >= k);

    let mut best: usize = rng.gen_range(0, n);

    for _ in 1..k {
        let i = rng.gen_range(0, n);
        if better_than(i, best) {
            best = i;
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
pub trait OpSelectRandomIndividual<I: Individual,F: Fitness> {
    fn select_random_individual<'a>(&mut self,
                                    population: &'a RatedPopulation<I, F>)
                                    -> usize; // IndividualIndex
}

/// Produce new generation through selection of \mu individuals from population.
pub trait OpSelect<I: Individual, F: Fitness> {
    fn select(&mut self, population: &RatedPopulation<I, F>, mu: usize) -> RatedPopulation<I, F>;
}

pub fn variation_or<I, F, T>(toolbox: &mut T,
                             population: &RatedPopulation<I, F>,
                             lambda: usize)
                             -> (UnratedPopulation<I>, RatedPopulation<I,F>)
    where I: Individual,
          F: Fitness,
          T: OpCrossover1<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I, F>
{
    // We assume that most offspring is unrated and only a small amount of already rated offspring.
    let mut unrated_offspring = UnratedPopulation::with_capacity(lambda);
    let mut rated_offspring = RatedPopulation::new();

    // Each step produces exactly one child.
    for _ in 0..lambda {
        let method = toolbox.variation();
        match method {
            VariationMethod::Crossover => {
                // select two individuals and mate them.
                // only the first offspring is used, the second is thrown away.
                let parent1_idx = toolbox.select_random_individual(population);
                let parent2_idx = toolbox.select_random_individual(population);
                let child1 = toolbox.crossover1(population.get(parent1_idx), population.get(parent2_idx));
                unrated_offspring.add(child1);
            }
            VariationMethod::Mutation => {
                // select a single individual and mutate it.
                let ind_idx = toolbox.select_random_individual(population);
                let child = toolbox.mutate(population.get(ind_idx));
                unrated_offspring.add(child);
            }
            VariationMethod::Reproduction => {
                let ind_idx = toolbox.select_random_individual(population);
                rated_offspring.add(population.get_individual(ind_idx).clone(), population.get_fitness(ind_idx).clone());
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
pub fn ea_mu_plus_lambda<I,F,T,E,S>(toolbox: &mut T, evaluator: &E, mut population: RatedPopulation<I,F>, mu: usize, lambda: usize, num_generations: usize, stat: S, numthreads: usize, chunksize: usize)
    -> RatedPopulation<I,F>
where I: Individual,
      F: Fitness+Default,
      T: OpCrossover1<I> + OpMutate<I> + OpVariation + OpSelectRandomIndividual<I,F> + OpSelect<I,F>,
      E: Evaluator<I,F>+Sync,
      S: Fn(usize, usize, &RatedPopulation<I,F>)
{
    let mut pool = simple_parallel::Pool::new(numthreads);

    //let nevals = population.rate_in_parallel(evaluator, &mut pool, chunksize);
    //stat(0, nevals, &population);

    for gen in 0..num_generations {
        // evaluate population. make sure that every individual has been rated.
        let (unrated_offspring, rated_offspring) = variation_or(toolbox, &population, lambda);
        let nevals = unrated_offspring.len();
        population.extend_with(rated_offspring);
        population.extend_with(unrated_offspring.rate_in_parallel(evaluator, &mut pool, chunksize));

        // select from offspring the `best` individuals
        let p = toolbox.select(&population, mu);
        stat(gen + 1, nevals, &p);
        population = p;
    }

    return population;
}
