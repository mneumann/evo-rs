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

fn iterate<R: Rng>(rng: &mut R,
                   population: Vec<MyGenome>,
                   fitness: Vec<MultiObjective2<f32>>,
                   pop_size: usize,
                   offspring_size: usize,
                   eta: f32)
                   -> (Vec<MyGenome>, Vec<MultiObjective2<f32>>) {
    assert!(population.len() == fitness.len());

    // evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fitness[..], pop_size);
    assert!(rank_dist.len() == pop_size);

    //
    // for rd in rank_dist.iter() {
    // println!("-------------------------------------------");
    // println!("rd: {:?}", rd);
    // println!("fitness: {:?}", fitness[rd.idx]);
    // println!("genome: {:?}", population[rd.idx]);
    // }
    //

    // create `offspring_size` new offspring using binary tournament (randomly
    // select two mating
    // partners)
    let offspring: Vec<_> = (0..offspring_size)
                                .map(|_| {
                                    // first parent. two candidates
                                    let p1cand = (rng.gen_range(0, rank_dist.len()),
                                                  rng.gen_range(0, rank_dist.len()));

                                    // second parent. two candidates
                                    let p2cand = (rng.gen_range(0, rank_dist.len()),
                                                  rng.gen_range(0, rank_dist.len()));

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

                                    // cross-over the two parents and produce one child (throw away
                                    // second child)
                                    MyGenome::crossover1(rng,
                                                         (&population[p1], &population[p2]),
                                                         eta)
                                })
                                .collect();

    assert!(offspring.len() == offspring_size);

    // evaluate fitness of offspring
    let fitness_offspring: Vec<_> = offspring.iter().map(|ind| ind.fitness()).collect();
    assert!(fitness_offspring.len() == offspring.len());

    // merge population and offspring, then select
    let mut new_pop = Vec::with_capacity(rank_dist.len() + offspring.len());
    let mut new_fit = Vec::with_capacity(rank_dist.len() + offspring.len());
    for rd in rank_dist {
        new_pop.push(population[rd.idx].clone());
        new_fit.push(fitness[rd.idx].clone());
    }

    new_pop.extend(offspring);
    new_fit.extend(fitness_offspring);

    assert!(new_pop.len() == pop_size + offspring_size);
    assert!(new_fit.len() == pop_size + offspring_size);

    return (new_pop, new_fit);
}

fn main() {
    const N: usize = 2; // ZDT1 order
    const MU: usize = 600; // size of population
    const LAMBDA: usize = 300; // size of offspring population
    const ETA: f32 = 2.0; // cross-over variance
    const NGEN: usize = 10; // number of generations

    let mut rng = rand::isaac::Isaac64Rng::new_unseeded();

    // create initial random population
    let initial_population: Vec<MyGenome> = (0..MU)
                                                .map(|_| MyGenome::random(&mut rng, N))
                                                .collect();

    // evaluate fitness
    let fitness: Vec<_> = initial_population.iter().map(|ind| ind.fitness()).collect();
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for i in 0..NGEN {
        println!("===========================================================");
        println!("Iteration: {}", i);
        println!("===========================================================");

        let (new_pop, new_fit) = iterate(&mut rng, pop, fit, MU, LAMBDA, ETA);
        pop = new_pop;
        fit = new_fit;
    }
    println!("===========================================================");
    println!("END");
    println!("===========================================================");

    // finally evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fit[..], MU);
    assert!(rank_dist.len() == MU);

    for rd in rank_dist.iter() {
        println!("-------------------------------------------");
        println!("rd: {:?}", rd);
        println!("fitness: {:?}", fit[rd.idx]);
        println!("genome: {:?}", pop[rd.idx]);
    }
}
