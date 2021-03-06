/// Optimizes the one-max problem (maximizing the number of "ones" in a
/// bit-string)

extern crate rand;
extern crate evo;

use rand::Rng;
use std::cmp;

use evo::{Individual, RatedPopulation, UnratedPopulation, Evaluator, OpCrossover1, OpMutate,
          OpSelectRandomIndividual, OpSelect, OpVariation, VariationMethod, ea_mu_plus_lambda,
          Probability, ProbabilityValue, MaxFitness};
use evo::selection::tournament_selection_fast;
use evo::bit_string::{BitString, crossover_one_point};


#[derive(Clone, Debug)]
struct MyGenome {
    bits: BitString,
}

impl MyGenome {
    fn new(bs: BitString) -> MyGenome {
        MyGenome { bits: bs }
    }
}

impl Individual for MyGenome {}

struct MyEval;

impl Evaluator<MyGenome, MaxFitness<usize>> for MyEval {
    fn fitness(&self, ind: &MyGenome) -> MaxFitness<usize> {
        MaxFitness(ind.bits.count(true))
    }
}

struct Toolbox {
    rng: Box<Rng>,
    prob_crossover: Probability,
    prob_mutation: Probability,
    prob_bitflip: Probability,
    tournament_size: usize,
}

impl OpCrossover1<MyGenome> for Toolbox {
    fn crossover1(&mut self, male: &MyGenome, female: &MyGenome) -> MyGenome {
        let point = self.rng.gen_range(0, cmp::min(male.bits.len(), female.bits.len()));
        let (c1, _c2) = crossover_one_point(point, (&male.bits, &female.bits));
        MyGenome::new(c1)
    }
}

impl OpMutate<MyGenome> for Toolbox {
    fn mutate(&mut self, ind: &MyGenome) -> MyGenome {
        let mut bs = ind.bits.clone();
        bs.flip_bits_randomly(&mut self.rng, self.prob_bitflip);
        MyGenome::new(bs)
    }
}

impl OpVariation for Toolbox {
    fn variation(&mut self) -> VariationMethod {
        let r = self.rng.gen::<ProbabilityValue>();
        if r.is_probable_with(self.prob_crossover) {
            VariationMethod::Crossover
        } else if r.is_probable_with(self.prob_crossover + self.prob_mutation) {
            VariationMethod::Mutation
        } else {
            VariationMethod::Reproduction
        }
    }
}

// XXX: No need for Fitness
impl<I: Individual, F:PartialOrd+Clone+Send+Default> OpSelectRandomIndividual<I, F> for Toolbox {
    fn select_random_individual(&mut self, population: &RatedPopulation<I,F>) -> usize {
        self.rng.gen_range(0, population.len())
    }
}

impl<I: Individual, F: PartialOrd + Clone + Send + Default> OpSelect<I, F> for Toolbox {
    fn select(&mut self, population: &RatedPopulation<I, F>, mu: usize) -> RatedPopulation<I, F> {
        let mut pop: RatedPopulation<I, F> = RatedPopulation::with_capacity(mu);
        for _ in 0..mu {
            let choice = tournament_selection_fast(&mut self.rng,
                                                   |i1, i2| population.fitter_than(i1, i2),
                                                   population.len(),
                                                   self.tournament_size);
            pop.add(population.get_individual(choice).clone(),
                    population.get_fitness(choice).clone());
        }
        assert!(pop.len() == mu);
        return pop;
    }
}

fn print_stat(p: &RatedPopulation<MyGenome, MaxFitness<usize>>) {
    let mut fitnesses = Vec::new();
    for i in 0..p.len() {
        let f = p.get_fitness(i).0;
        fitnesses.push(f);
    }
    let min = fitnesses.iter().fold(fitnesses[0], |b, &i| cmp::min(b, i));
    let max = fitnesses.iter().fold(fitnesses[0], |b, &i| cmp::max(b, i));
    let sum = fitnesses.iter().fold(0, |b, &i| b + i);

    println!("min: {}, max: {}, sum: {}, avg: {}",
             min,
             max,
             sum,
             sum as f32 / fitnesses.len() as f32);
}

fn main() {
    const BITS: usize = 100;
    const MU: usize = 600;
    const LAMBDA: usize = 300;
    const NGEN: usize = 100;

    let mut rng = rand::isaac::Isaac64Rng::new_unseeded();

    let mut initial_population: UnratedPopulation<MyGenome> = UnratedPopulation::with_capacity(MU);
    for _ in 0..MU {
        let iter = rng.gen_iter::<bool>().take(BITS);
        initial_population.add(MyGenome { bits: BitString::from_iter(iter) });
    }
    let evaluator = MyEval;
    let rated_population: RatedPopulation<MyGenome, MaxFitness<usize>> =
        initial_population.rate(&evaluator);

    let mut toolbox = Toolbox {
        rng: Box::new(rng),
        prob_crossover: Probability::new(0.5),
        prob_mutation: Probability::new(0.2),
        prob_bitflip: Probability::new(0.05),
        tournament_size: 3,
    };

    fn stat(gen: usize, nevals: usize, pop: &RatedPopulation<MyGenome, MaxFitness<usize>>) {
        print!("{:04} {:04}", gen, nevals);
        print_stat(pop);
    }

    let _optimum = evo::ea_mu_plus_lambda(&mut toolbox,
                                          &evaluator,
                                          rated_population,
                                          MU,
                                          LAMBDA,
                                          NGEN,
                                          stat,
                                          8,
                                          10);
}
