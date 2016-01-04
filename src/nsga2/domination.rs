pub trait Dominate<Rhs=Self> {
    fn dominates(&self, other: &Rhs) -> bool;
}

/// Stop after we have found `n` solutions. As we include the whole pareto
/// front, there are probably more solutions returned.
pub fn fast_non_dominated_sort<P: Dominate>(solutions: &[P], n: usize) -> Vec<Vec<usize>> {
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
