use std::cmp::{self, Ordering};
use std::f32;
use std::convert::From;
use rand::Rng;
use std::ops::Sub;
use super::selection::tournament_selection_fast;

pub trait Mate<T> {
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &T, p2: &T) -> T;
}

pub trait Dominate<Rhs=Self> {
    fn dominates(&self, other: &Rhs) -> bool;
}

pub trait MultiObjective {
    fn num_objectives() -> usize;

    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering;

    // Distance between self and other
    fn dist_objective(&self, other: &Self, objective: usize) -> f32;

}

impl<T: MultiObjective> Dominate<T> for T {
    fn dominates(&self, other: &Self) -> bool {
        let mut less_cnt = 0;
        for i in 0..Self::num_objectives() {
            match self.cmp_objective(other, i) {
                Ordering::Greater => {
                    return false;
                }
                Ordering::Less => {
                    less_cnt += 1;
                }
                Ordering::Equal => {
                }
            }
        }
        return less_cnt > 0;
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjective2<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    pub objectives: [T; 2],
}

impl<T: Sized + PartialOrd + Copy + Clone> From<(T, T)> for MultiObjective2<T> {
    fn from(t: (T, T)) -> MultiObjective2<T> {
        MultiObjective2 { objectives: [t.0, t.1] }
    }
}

impl<T, R> MultiObjective for MultiObjective2<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    fn num_objectives() -> usize {
        2
    }
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjective3<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    pub objectives: [T; 3],
}

impl<T: Sized + PartialOrd + Copy + Clone> From<(T, T, T)> for MultiObjective3<T> {
    fn from(t: (T, T, T)) -> MultiObjective3<T> {
        MultiObjective3 { objectives: [t.0, t.1, t.2] }
    }
}

impl<T, R> MultiObjective for MultiObjective3<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    fn num_objectives() -> usize {
        3
    }
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

/// Stop after we have found `n` solutions (we include the whole pareto front, so it are probably more solutions).
fn fast_non_dominated_sort<P: Dominate>(solutions: &[P], n: usize) -> Vec<Vec<usize>> {
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    let mut domination_count: Vec<usize> = (0..solutions.len()).map(|_| 0).collect();
    let mut dominated_solutions: Vec<Vec<usize>> = (0..solutions.len())
                                                       .map(|_| Vec::new())
                                                       .collect();
    let mut found_solutions: usize = 0;

    for (i, p) in solutions.iter().enumerate() {
        for (j, q) in solutions.iter().enumerate() {
            if i == j {
                continue;
            }
            if p.dominates(q) {
                // Add `q` to the set of solutions dominated by `p`.
                dominated_solutions[i].push(j);
            } else if q.dominates(p) {
                // Increment domination counter of `p`.
                domination_count[i] += 1;
            }
        }

        if domination_count[i] == 0 {
            // `p` belongs to the first front as it is not dominated by any
            // other solution.
            current_front.push(i);
        }
    }

    while !current_front.is_empty() {
        found_solutions += current_front.len();
        if found_solutions >= n {
            fronts.push(current_front);
            break;
        } else {
            // we haven't found enough solutions yet.
            let mut next_front = Vec::new();
            for &p_i in current_front.iter() {
                for &q_i in dominated_solutions[p_i].iter() {
                    domination_count[q_i] -= 1;
                    if domination_count[q_i] == 0 {
                        // q belongs to the next front
                        next_front.push(q_i);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }
    }

    return fronts;
}

#[derive(Debug)]
pub struct SolutionRankDist {
    pub idx: usize,
    pub rank: u32,
    pub dist: f32,
}

impl PartialEq for SolutionRankDist {
    #[inline]
    fn eq(&self, other: &SolutionRankDist) -> bool {
        self.rank == other.rank && self.dist == other.dist
    }
}

// Implement the crowding-distance comparison operator.
impl PartialOrd for SolutionRankDist {
    #[inline]
    // compare on rank first (ASC), then on dist (DESC)
    fn partial_cmp(&self, other: &SolutionRankDist) -> Option<Ordering> {
        match self.rank.partial_cmp(&other.rank) {
            Some(Ordering::Equal) => {
                // first criterion equal, second criterion decides
                // reverse ordering
                self.dist.partial_cmp(&other.dist).map(|i| i.reverse())
            }
            other => other,
        }
    }
}

fn crowding_distance_assignment<P: MultiObjective>(solutions: &[P],
                                                   common_rank: u32,
                                                   individuals_idx: &[usize])
                                                   -> Vec<SolutionRankDist> {
    let l = individuals_idx.len();
    let mut distance: Vec<f32> = (0..l).map(|_| 0.0).collect();
    let mut indices: Vec<usize> = (0..l).map(|i| i).collect();

    for m in 0..P::num_objectives() {
        // sort using objective `m`
        indices.sort_by(|&a, &b| {
            solutions[individuals_idx[a]].cmp_objective(&solutions[individuals_idx[b]], m)
        });
        distance[indices[0]] = f32::INFINITY;
        distance[indices[l - 1]] = f32::INFINITY;

        let min_idx = individuals_idx[indices[0]];
        let max_idx = individuals_idx[indices[l - 1]];

        let dist_max_min = solutions[max_idx].dist_objective(&solutions[min_idx], m);
        if dist_max_min != 0.0 {
            let norm = P::num_objectives() as f32 * dist_max_min;
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                let next_idx = individuals_idx[indices[i + 1]];
                let prev_idx = individuals_idx[indices[i - 1]];
                distance[indices[i]] += solutions[next_idx]
                                            .dist_objective(&solutions[prev_idx], m) /
                                        norm;
            }
        }
    }

    return indices.iter()
                  .map(|&i| {
                      SolutionRankDist {
                          idx: individuals_idx[i],
                          rank: common_rank,
                          dist: distance[i],
                      }
                  })
                  .collect();
}


/// Select `n` out of the `solutions`, assigning rank and distance.
pub fn select<P: Dominate + MultiObjective>(solutions: &[P], n: usize) -> Vec<SolutionRankDist> {
    let mut selection = Vec::with_capacity(cmp::min(solutions.len(), n));

    let pareto_fronts = fast_non_dominated_sort(solutions, n);

    for (rank, front) in pareto_fronts.iter().enumerate() {
        if selection.len() >= n {
            break;
        }
        let missing: usize = n - selection.len();

        let mut solution_rank_dist = crowding_distance_assignment(solutions,
                                                                  rank as u32,
                                                                  &front[..]);
        if solution_rank_dist.len() <= missing {
            // whole front fits into result.
            selection.extend(solution_rank_dist);
            assert!(selection.len() <= n);
        } else {
            // choose only best from this front, according to the crowding distance.
            solution_rank_dist.sort_by(|a, b| {
                debug_assert!(a.rank == b.rank);
                a.partial_cmp(b).unwrap()
            });
            selection.extend(solution_rank_dist.into_iter().take(missing));
            assert!(selection.len() == n);
            break;
        }
    }

    return selection;
}

pub trait FitnessEval<I, F:Dominate+MultiObjective+Clone> {
    fn fitness(&mut self, &[I]) -> Vec<F>;
}

pub fn iterate<R: Rng,
               I: Clone,
               F: Dominate + MultiObjective + Clone,
               T: Mate<I> + FitnessEval<I, F>>
    (rng: &mut R,
     population: Vec<I>,
     fitness: Vec<F>,
     pop_size: usize,
     offspring_size: usize,
     tournament_k: usize,
     toolbox: &mut T)
     -> (Vec<I>, Vec<F>) {
    assert!(tournament_k > 0);
    assert!(population.len() == fitness.len());

    // evaluate rank and crowding distance (using select()).
    let rank_dist = select(&fitness[..], pop_size);
    assert!(rank_dist.len() == pop_size);

    // create `offspring_size` new offspring using k-tournament (
    // select the best individual out of k randomly choosen individuals)
    let offspring: Vec<_> = (0..offspring_size)
                                .map(|_| {

                                    // first parent. k candidates
                                    let p1 = tournament_selection_fast(rng,
                                                                       |i1, i2| {
                                                                           rank_dist[i1] <
                                                                           rank_dist[i2]
                                                                       },
                                                                       rank_dist.len(),
                                                                       tournament_k);

                                    // second parent. k candidates
                                    let p2 = tournament_selection_fast(rng,
                                                                       |i1, i2| {
                                                                           rank_dist[i1] <
                                                                           rank_dist[i2]
                                                                       },
                                                                       rank_dist.len(),
                                                                       tournament_k);

                                    // cross-over the two parents and produce one child (throw away
                                    // second child XXX)

                                    // The potentially dominating individual is gives as first
                                    // parameter.
                                    let (p1, p2) = if rank_dist[p1] < rank_dist[p2] {
                                        (p1, p2)
                                    } else if rank_dist[p2] < rank_dist[p1] {
                                        (p2, p1)
                                    } else {
                                        (p1, p2)
                                    };

                                    toolbox.mate(rng, &population[p1], &population[p2])
                                })
                                .collect();

    assert!(offspring.len() == offspring_size);

    // evaluate fitness of offspring
    let fitness_offspring = toolbox.fitness(&offspring[..]);
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

#[test]
fn test_dominates() {
    let a = MultiObjective2 { objectives: [1.0f32, 0.1] };
    let b = MultiObjective2 { objectives: [0.1f32, 0.1] };
    let c = MultiObjective2 { objectives: [0.1f32, 1.0] };

    assert_eq!(false, a.dominates(&a));
    assert_eq!(false, a.dominates(&b));
    assert_eq!(false, a.dominates(&c));

    assert_eq!(true, b.dominates(&a));
    assert_eq!(false, b.dominates(&b));
    assert_eq!(true, b.dominates(&c));

    assert_eq!(false, c.dominates(&a));
    assert_eq!(false, c.dominates(&b));
    assert_eq!(false, c.dominates(&c));
}

#[test]
fn test_abc() {
    let mut solutions: Vec<MultiObjective2<f32>> = Vec::new();
    solutions.push(MultiObjective2 { objectives: [1.0, 0.1] });
    solutions.push(MultiObjective2 { objectives: [0.1, 0.1] });
    solutions.push(MultiObjective2 { objectives: [0.1, 1.0] });
    solutions.push(MultiObjective2 { objectives: [0.5, 0.5] });
    solutions.push(MultiObjective2 { objectives: [0.5, 0.5] });

    println!("solutions: {:?}", solutions);
    let selection = select(&solutions[..], 5);
    println!("selection: {:?}", selection);

    let fronts = fast_non_dominated_sort(&solutions[..], 10);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for (rank, front) in fronts.iter().enumerate() {
        let distances = crowding_distance_assignment(&solutions[..], rank as u32, &front[..]);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
