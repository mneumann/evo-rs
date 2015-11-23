#[cfg(test)]
use rand::Rand;

use rand::Rng;

#[inline]
fn sbx_beta(u: f32, eta: f32) -> f32 {
    debug_assert!(u >= 0.0 && u < 1.0);

    if u <= 0.5 {
        2.0 * u
    } else {
        1.0 / (2.0 * (1.0 - u))
    }
    .powf(1.0 / (eta + 1.0))
}

/// Modifies ind1 and ind2 in-place.
/// Reference: http://www.iitk.ac.in/kangal/papers/k2011017.pdf
pub fn simulated_binary_crossover<R: Rng>(rng: &mut R,
                                          ind1: &mut [f32],
                                          ind2: &mut [f32],
                                          eta: f32) {
    assert!(ind1.len() == ind2.len());

    let mut iter = ind1.iter_mut().zip(ind2.iter_mut());
    for (x1, x2) in iter {
        let (p1, p2) = (*x1, *x2);

        let u = rng.gen::<f32>();
        let beta = sbx_beta(u, eta);

        let c1 = 0.5 * (((1.0 + beta) * p1) + ((1.0 - beta) * p2));
        let c2 = 0.5 * (((1.0 - beta) * p1) + ((1.0 + beta) * p2));

        *x1 = c1;
        *x2 = c2;
    }
}

#[test]
fn test_sbx() {
    let mut rng = ::rand::isaac::Isaac64Rng::new_unseeded();

    let parent1 = vec![1.0, 2.0];
    let parent2 = vec![100.0, 1.0];

    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();

    simulated_binary_crossover(&mut rng, &mut child1[..], &mut child2[..], 2.0);

    let beta: Vec<_> = (0..2)
                           .map(|i| (child2[i] - child1[i]) / (parent2[i] - parent1[i]))
                           .collect();

    println!("({:?}, {:?}) => ({:?}, {:?})",
             parent1,
             parent2,
             child1,
             child2);
    println!("beta: {:?}", beta);
}
