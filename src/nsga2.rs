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

trait Dominate<Rhs=Self> {
    fn dominates(&self, other: &Rhs) -> bool;
}

trait MultiObjective {
    fn num_objectives() -> usize;
    fn get_f32_objective(&self, i: usize) -> f32;
}

#[derive(Debug)]
struct MultiObjective2<T>
    where T: Sized + PartialOrd
{
    objectives: [T; 2],
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
struct MultiObjective3<T>
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

/// `n` stop after we have found this number of solutions (we include the whole pareto front).
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

/// returns: (individual_idx, distance)
fn crowding_distance_assignment<P: MultiObjective>(solutions: &[P],
                                                   individuals_idx: &[usize])
                                                   -> Vec<(usize, f32)> {
    let l = individuals_idx.len();
    let mut distance: Vec<f32> = (0..l).map(|_| 0.0).collect();

    // maps index into distance array to index into solutions.
    let mut map: Vec<(usize, usize)> = (0..l).map(|i| (i, individuals_idx[i])).collect();

    for m in 0..P::num_objectives() {
        // sort using objective `m`
        map.sort_by(|a, b| {
            solutions[a.1]
                .get_f32_objective(m)
                .partial_cmp(&solutions[b.1].get_f32_objective(m))
                .unwrap()
        });
        distance[map[0].0] = std::f32::INFINITY;
        distance[map[l - 1].0] = std::f32::INFINITY;
        let min_f = solutions[map[0].1].get_f32_objective(m);
        let max_f = solutions[map[l - 1].1].get_f32_objective(m);
        if min_f != max_f {
            let norm = P::num_objectives() as f32 * (max_f - min_f);
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                distance[map[i].0] += (solutions[map[i + 1].1].get_f32_objective(m) -
                                       solutions[map[i - 1].1].get_f32_objective(m)) /
                                      norm;
            }
        }
    }

    let mut res: Vec<(usize, f32)> = (0..l).map(|_| (0, -1.0)).collect();
    for &(i, idx) in map.iter() {
        debug_assert!(res[i].1 == -1.0);
        debug_assert!(res[i].0 == 0);
        res[i] = (idx, distance[i]);
    }

    return res;
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

    let fronts = fast_non_dominated_sort(&solutions[..], 10);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for front in fronts.iter() {
        let distances = crowding_distance_assignment(&solutions[..], &front[..]);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
