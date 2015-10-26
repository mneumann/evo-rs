/// Optimizes the one-max problem (maximizing the number of "ones" in a
/// bit-string)

extern crate rand;
extern crate evo;

use rand::{Rng};
use std::cmp;

use evo::{
    Fitness,
    Individual,
    Population,
    EvaluatedIndividual,
    Evaluator,
    OpCrossover,
    OpMutate,
    OpSelectRandomIndividual,
    OpSelect,
    OpVariation,
    VariationMethod,
    tournament_selection,
    ea_mu_plus_lambda,
    Probability,
    ProbabilityValue,
    MaxFitness,
};
use evo::bit_string::{
    BitString,
    crossover_one_point,
};


#[derive(Clone, Debug)]
struct MyGenome {
    bits: BitString
}

impl MyGenome {
   fn new(bs: BitString) -> MyGenome {
      MyGenome{bits: bs}
   }
}

impl Individual for MyGenome {
}

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

impl OpCrossover<MyGenome> for Toolbox {
    fn crossover(&mut self, male: &MyGenome, female: &MyGenome) -> (MyGenome, MyGenome) { 
           let point = self.rng.gen_range(0, cmp::min(male.bits.len(), female.bits.len()));
           let (c1, c2) = crossover_one_point(point, (&male.bits, &female.bits));
           (MyGenome::new(c1), MyGenome::new(c2))
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
        }
        else if r.is_probable_with(self.prob_crossover + self.prob_mutation) {
            VariationMethod::Mutation
        }
        else {
            VariationMethod::Reproduction
        }
    }
}

// XXX: No need for Fitness
impl<I: Individual, F: Fitness> OpSelectRandomIndividual<I, F> for Toolbox {
    fn select_random_individual<'a>(&mut self, population: &'a Population<I,F>) -> &'a EvaluatedIndividual<I, F> {
        population.get_ref(self.rng.gen_range(0, population.len()))
    }
}

impl<I: Individual, F: Fitness> OpSelect<I, F> for Toolbox {
    fn select(&mut self, population: &Population<I,F>, mu: usize) -> Population<I,F> {
        let mut pop = Population::with_capacity(mu);
        for _ in 0..mu {
            let x = tournament_selection(&mut self.rng, |idx| { population.get_ref(idx).fitness().unwrap() }, population.len(), self.tournament_size).unwrap();
            pop.add(population.get_ref(x.0).clone());
        }
        assert!(pop.len() == mu);
        return pop;
    }
}

fn print_stat(p: &Population<MyGenome, MaxFitness<usize>>) {
    let mut fitnesses = Vec::new();
    for i in 0..p.len() {
        let f = p.get_ref(i).fitness().unwrap().0;
        fitnesses.push(f);
    }
    let min = fitnesses.iter().fold(fitnesses[0], |b, &i| cmp::min(b, i));
    let max = fitnesses.iter().fold(fitnesses[0], |b, &i| cmp::max(b, i));
    let sum = fitnesses.iter().fold(0, |b, &i| b + i);

    println!("min: {}, max: {}, sum: {}, avg: {}", min, max, sum, sum as f32 / fitnesses.len() as f32);
}

fn main() {
   const BITS: usize = 100;
   const MU: usize = 300;
   const LAMBDA: usize = 300;
   const NGEN: usize = 100;

   let mut rng = rand::isaac::Isaac64Rng::new_unseeded();

   let mut initial_population: Population<MyGenome, MaxFitness<usize>> = Population::with_capacity(MU);
   for _ in 0..MU {
        let iter = rng.gen_iter::<bool>().take(BITS);
        initial_population.add_individual(MyGenome{bits:BitString::from_iter(iter)});
   }
   let evaluator = MyEval;
   initial_population.evaluate(&evaluator);
   //print_stat(&initial_population);
   //println!("{:?}", initial_population);

   let mut toolbox = Toolbox {
        rng: Box::new(rng),
        prob_crossover: Probability::new(0.5),
        prob_mutation: Probability::new(0.2),
        prob_bitflip: Probability::new(0.05),
        tournament_size: 3,
   };

   fn stat(gen:usize, pop: &Population<MyGenome, MaxFitness<usize>>) {
       print!("{} ", gen);
       print_stat(&pop);
   }

   let _optimum = evo::ea_mu_plus_lambda(&mut toolbox, &evaluator, initial_population, MU, LAMBDA, NGEN, stat);
}
