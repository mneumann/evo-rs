/// Optimizes the zdt1 function using NSGA-II

extern crate rand;
extern crate evo;

use rand::{Rng, Closed01};

use evo::nsga2::{self, Mate, FitnessEval};
use evo::mo::MultiObjective2;
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

struct Toolbox {
    mating_eta: f32,
}

impl Mate<MyGenome> for Toolbox {
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &MyGenome, p2: &MyGenome) -> MyGenome {
        MyGenome::crossover1(rng, (p1, p2), self.mating_eta)
    }
}

impl FitnessEval<MyGenome, MultiObjective2<f32>> for Toolbox {
    fn fitness(&mut self, pop: &[MyGenome]) -> Vec<MultiObjective2<f32>> {
        pop.iter().map(|ind| ind.fitness()).collect()
    }
}

fn main() {
    const N: usize = 2; // ZDT1 order
    const MU: usize = 600; // size of population
    const LAMBDA: usize = 300; // size of offspring population
    const ETA: f32 = 2.0; // cross-over variance
    const NGEN: usize = 100; // number of generations

    let mut rng = rand::isaac::Isaac64Rng::new_unseeded();

    let mut toolbox = Toolbox { mating_eta: ETA };

    // create initial random population
    let initial_population: Vec<MyGenome> = (0..MU)
                                                .map(|_| MyGenome::random(&mut rng, N))
                                                .collect();

    // evaluate fitness
    let fitness: Vec<_> = toolbox.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for _ in 0..NGEN {
        let (new_pop, new_fit) = nsga2::iterate(&mut rng, pop, fit, MU, LAMBDA, 2, 2, &mut toolbox);
        pop = new_pop;
        fit = new_fit;
    }
    println!("===========================================================");

    // finally evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fit[..], MU, 2);
    assert!(rank_dist.len() == MU);

    for rd in rank_dist.iter() {
        println!("-------------------------------------------");
        println!("rd: {:?}", rd);
        println!("fitness: {:?}", fit[rd.idx]);
        println!("genome: {:?}", pop[rd.idx]);
    }
}
