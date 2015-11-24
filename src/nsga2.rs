use std::cmp::{self, Ordering};
use std::f32;
use std::convert::From;

pub trait Dominate<Rhs=Self> {
    fn dominates(&self, other: &Rhs) -> bool;
}

pub trait MultiObjective {
    fn num_objectives() -> usize;
    fn get_f32_objective(&self, i: usize) -> f32;
}

#[derive(Debug, Clone)]
pub struct MultiObjective2<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    objectives: [T; 2],
}

impl<T:Sized+PartialOrd+Copy+Clone> From<(T, T)> for MultiObjective2<T> {
    fn from(t: (T, T)) -> MultiObjective2<T> {
        MultiObjective2 {objectives: [t.0, t.1]}
    }
}

impl MultiObjective for MultiObjective2<f32> {
    fn num_objectives() -> usize {
        2
    }
    fn get_f32_objective(&self, i: usize) -> f32 {
        self.objectives[i]
    }
}

#[derive(Debug)]
pub struct MultiObjective3<T>
    where T: Sized + PartialOrd
{
    objectives: [T; 3],
}

fn dominates_slices<T: Sized + PartialOrd>(sa: &[T], sb: &[T]) -> bool {
    assert!(sa.len() == sb.len());
    assert!(sa.len() > 0);

    let mut less_cnt = 0;
    for (a, b) in sa.iter().zip(sb.iter()) {
        if a > b {
            return false;
        } else if a < b {
            less_cnt += 1;
        }
    }

    return less_cnt > 0;
}

impl<T:Sized+PartialOrd> Dominate for MultiObjective2<T> {
    fn dominates(&self, other: &Self) -> bool {
        dominates_slices(&self.objectives[..], &other.objectives[..])
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
            other => { other }
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
            solutions[individuals_idx[a]]
                .get_f32_objective(m)
                .partial_cmp(&solutions[individuals_idx[b]].get_f32_objective(m))
                .unwrap()
        });
        distance[indices[0]] = f32::INFINITY;
        distance[indices[l - 1]] = f32::INFINITY;
        let min_f = solutions[individuals_idx[indices[0]]].get_f32_objective(m);
        let max_f = solutions[individuals_idx[indices[l - 1]]].get_f32_objective(m);
        if min_f != max_f {
            let norm = P::num_objectives() as f32 * (max_f - min_f);
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                distance[indices[i]] += (solutions[individuals_idx[indices[i + 1]]]
                                             .get_f32_objective(m) -
                                         solutions[individuals_idx[indices[i - 1]]]
                                             .get_f32_objective(m)) /
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

#[test]
fn test_dominates() {
    let a = MultiObjective2 { objectives: [1.0, 0.1] };
    let b = MultiObjective2 { objectives: [0.1, 0.1] };
    let c = MultiObjective2 { objectives: [0.1, 1.0] };

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

fn main() {
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
