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

#[derive(Debug)]
struct MultiObjective2<T>
    where T: Sized + PartialOrd
{
    objectives: [T; 2],
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

fn fast_non_dominated_sort<P: Dominate>(solutions: &[P]) -> Vec<Vec<usize>> {
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    let mut domination_count: Vec<usize> = (0..solutions.len()).map(|_| 0).collect();
    let mut dominated_solutions: Vec<Vec<usize>> = (0..solutions.len())
                                                       .map(|_| Vec::new())
                                                       .collect();
    let mut rank: Vec<Option<usize>> = (0..solutions.len()).map(|_| None).collect();

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
            assert!(rank[i].is_none());
            rank[i] = Some(fronts.len());
        }
    }

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p_i in current_front.iter() {
            for &q_i in dominated_solutions[p_i].iter() {
                domination_count[q_i] -= 1;
                if domination_count[q_i] == 0 {
                    // q belongs to the next front
                    assert!(rank[q_i].is_none());
                    rank[q_i] = Some(fronts.len() + 1);
                    next_front.push(q_i);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    for i in 0..solutions.len() {
        assert!(rank[i].is_some());
    }

    return fronts;
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

    let fronts = fast_non_dominated_sort(&solutions[..]);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);
}
