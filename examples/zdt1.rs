/// Optimizes the zdt1 function using NSGA-II

extern crate rand;
extern crate evo;

use rand::{Rng, Closed01};

use evo::nsga2::{self, MultiObjective2};
use evo::crossover::sbx_single_var_bounded;

/// optimal pareto front (f_1, 1 - sqrt(f_1))
/// 0 <= x[i] <= 1.0
fn zdt1(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    assert!(n >= 2);

    let f1 = x[0];
    let g = 1.0 + (9.0 / (n - 1) as f32) * x[1..].iter().fold(0.0, |b, &i| b + i);
    let f2 = g * (1.0 - (f1 / g).sqrt());

    (f1, f2)
}

#[derive(Clone, Debug)]
struct MyGenome {
    xs: Vec<f32>,
}

impl MyGenome {
    fn new(xs: Vec<f32>) -> MyGenome {
        assert!(xs.len() >= 2);
        for &x in xs.iter() {
            assert!(x >= 0.0 && x <= 1.0);
        }
        MyGenome { xs: xs }
    }

    fn random<R: Rng>(rng: &mut R, n: usize) -> MyGenome {
        MyGenome::new((0..n).map(|_| rng.gen::<Closed01<f32>>().0).collect())
    }

    fn fitness(&self) -> MultiObjective2<f32> {
        MultiObjective2::from(zdt1(&self.xs[..]))
    }

    fn len(&self) -> usize {
        self.xs.len()
    }

    fn crossover1<R: Rng>(rng: &mut R, p: (&MyGenome, &MyGenome), eta: f32) -> MyGenome {
        assert!(p.0.len() == p.1.len());
        let xs: Vec<_> = p.0
                          .xs
                          .iter()
                          .zip(p.1.xs.iter())
                          .map(|(&x1, &x2)| {
                              let (c1, _c2) = sbx_single_var_bounded(rng,
                                                                     (x1, x2),
                                                                     (0.0, 1.0),
                                                                     eta);
                              c1
                          })
                          .collect();
        MyGenome::new(xs)
    }
}

fn main() {
    const MU: usize = 600;
    const N: usize = 2;
    const ETA: f32 = 2.0;
    // const NGEN: usize = 100;

    let mut rng = rand::isaac::Isaac64Rng::new_unseeded();

    // create initial random population
    let initial_population: Vec<MyGenome> = (0..MU)
                                                .map(|_| MyGenome::random(&mut rng, N))
                                                .collect();

    // evaluate fitness
    let fitness: Vec<_> = initial_population.iter().map(|ind| ind.fitness()).collect();
    assert!(fitness.len() == initial_population.len());

    // evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fitness[..], MU);
    assert!(fitness.len() == rank_dist.len());

    for rd in rank_dist.iter() {
        println!("-------------------------------------------");
        println!("rd: {:?}", rd);
        println!("fitness: {:?}", fitness[rd.idx]);
        println!("genome: {:?}", initial_population[rd.idx]);
    }

    // create MU new offspring using binary tournament (randomly select two mating
    // partners)
    let mut offspring: Vec<MyGenome> = Vec::with_capacity(MU);
    for _ in 0..MU {
        // first parent. two candidates
        let p1cand = (rng.gen_range(0, initial_population.len()),
                      rng.gen_range(0, initial_population.len()));

        // second parent. two candidates
        let p2cand = (rng.gen_range(0, initial_population.len()),
                      rng.gen_range(0, initial_population.len()));

        // choose the better candiate (first parent)
        let p1 = if rank_dist[p1cand.0] < rank_dist[p1cand.1] {
            p1cand.0
        } else {
            p1cand.1
        };

        // choose the better candiate (second parent)
        let p2 = if rank_dist[p2cand.0] < rank_dist[p2cand.1] {
            p2cand.0
        } else {
            p2cand.1
        };


        // cross-over the two parents and produce one child (throw away second child)
        let child = MyGenome::crossover1(&mut rng,
                                         (&initial_population[p1], &initial_population[p2]),
                                         ETA);
        offspring.push(child);
    }

    assert!(offspring.len() == MU);

    // evaluate fitness of offspring
    let fitness_offspring: Vec<_> = offspring.iter().map(|ind| ind.fitness()).collect();
    assert!(fitness_offspring.len() == offspring.len());

    // merge population and offspring, then select
    let mut new_pop = initial_population;
    new_pop.extend(offspring);
    let mut new_fitness = fitness;
    new_fitness.extend(fitness_offspring);

    assert!(new_pop.len() == 2 * MU);
    assert!(new_fitness.len() == 2 * MU);

    // evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&new_fitness[..], MU);
    assert!(rank_dist.len() == MU);

    println!("===========================================================");
    println!("===========================================================");
    for rd in rank_dist.iter() {
        println!("-------------------------------------------");
        println!("rd: {:?}", rd);
        println!("fitness: {:?}", new_fitness[rd.idx]);
        println!("genome: {:?}", new_pop[rd.idx]);
    }
}
