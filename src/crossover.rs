use rand::Rng;

#[inline]
fn sbx_beta(u: f32, eta: f32) -> f32 {
    assert!(u >= 0.0 && u < 1.0);

    if u <= 0.5 {
        2.0 * u
    } else {
        1.0 / (2.0 * (1.0 - u))
    }
    .powf(1.0 / (eta + 1.0))
}

#[inline]
fn sbx_beta_bounded(u: f32, eta: f32, gamma: f32) -> f32 {
    assert!(u >= 0.0 && u < 1.0);

    let g = 1.0 - gamma;
    let ug = u * g;

    if u <= 0.5 / g {
        2.0 * ug
    } else {
        1.0 / (2.0 * (1.0 - ug))
    }
    .powf(1.0 / (eta + 1.0))
}

#[inline]
pub fn sbx_single_var<R: Rng>(rng: &mut R, p: (f32, f32), eta: f32) -> (f32, f32) {
    let u = rng.gen::<f32>();
    let beta = sbx_beta(u, eta);

    (0.5 * (((1.0 + beta) * p.0) + ((1.0 - beta) * p.1)),
     0.5 * (((1.0 - beta) * p.0) + ((1.0 + beta) * p.1)))
}

#[inline]
fn _sbx_single_var_bounded<R: Rng>(rng: &mut R,
                                   p: (f32, f32),
                                   bounds: (f32, f32),
                                   eta: f32)
                                   -> (f32, f32) {
    let (a, b) = bounds;
    let p_diff = p.1 - p.0;

    assert!(a <= b);
    assert!(p_diff > 0.0);
    assert!(p.0 >= a && p.0 <= b);
    assert!(p.1 >= a && p.1 <= b);

    let beta_a = 1.0 + (p.0 - a) / p_diff;
    let beta_b = 1.0 + (b - p.1) / p_diff;

    fn gamma(beta: f32, eta: f32) -> f32 {
        1.0 / (2.0 * beta.powf(eta + 1.0))
    }

    let gamma_a = gamma(beta_a, eta);
    let gamma_b = gamma(beta_b, eta);

    let u = rng.gen::<f32>();
    let beta_ua = sbx_beta_bounded(u, eta, gamma_a);
    let beta_ub = sbx_beta_bounded(u, eta, gamma_b);

    let c = (0.5 * (((1.0 + beta_ua) * p.0) + ((1.0 - beta_ua) * p.1)),
             0.5 * (((1.0 - beta_ub) * p.0) + ((1.0 + beta_ub) * p.1)));

    assert!(c.0 >= a && c.0 <= b);
    assert!(c.1 >= a && c.1 <= b);

    return c;
}

#[inline]
pub fn sbx_single_var_bounded<R: Rng>(rng: &mut R,
                                      p: (f32, f32),
                                      bounds: (f32, f32),
                                      eta: f32)
                                      -> (f32, f32) {
    if p.0 < p.1 {
        _sbx_single_var_bounded(rng, (p.0, p.1), bounds, eta)
    } else if p.0 > p.1 {
        let r = _sbx_single_var_bounded(rng, (p.1, p.0), bounds, eta);
        (r.1, r.0)
    } else {
        debug_assert!(p.0 == p.1);
        (p.0, p.1)
    }
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
        let (c1, c2) = sbx_single_var(rng, (*x1, *x2), eta);
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

pub fn linear_2point_crossover<T:Clone>(p1: &[T], p2: &[T], cut_pt1: (usize, usize), cut_pt2: (usize, usize)) -> Vec<T> {
    assert!(cut_pt1.0 <= cut_pt1.1);
    assert!(cut_pt2.0 <= cut_pt2.1);
    assert!(cut_pt1.0 <= p1.len());
    assert!(cut_pt1.1 <= p1.len());
    assert!(cut_pt2.0 <= p2.len());
    assert!(cut_pt2.1 <= p2.len());

    let len1 = cut_pt1.1 - cut_pt1.0;
    let len2 = cut_pt2.1 - cut_pt2.0;

    let len = p1.len() - /* cut out */ len1 + /* insert */ len2;

    let mut res = Vec::with_capacity(len);
    for e in &p1[..cut_pt1.0] { res.push(e.clone()); }
    for e in &p2[cut_pt2.0..cut_pt2.1] { res.push(e.clone()); }
    for e in &p1[cut_pt1.1..] { res.push(e.clone()); }

    assert!(res.len() == len);
    return res;
}

#[test]
fn test_linear_2point_crossover() {
    let p1 = vec![1,2,3,4];
    //             ^ ^
    let p2 = vec![5,6,7,8,9];
    //             ^     ^

    let c = linear_2point_crossover(&p1[..], &p2[..], (1,2), (1, 4));
    assert_eq!(&[1, 6,7,8, 3,4], &c[..]);

    let c = linear_2point_crossover(&p1[..], &p2[..], (1,2), (1, 5));
    assert_eq!(&[1, 6,7,8,9, 3,4], &c[..]);

    let c = linear_2point_crossover(&p1[..], &p2[..], (1,2), (0, 5));
    assert_eq!(&[1, 5,6,7,8,9, 3,4], &c[..]);

    let c = linear_2point_crossover(&p1[..], &p2[..], (1,1), (0, 5));
    assert_eq!(&[1, 5,6,7,8,9, 2,3,4], &c[..]);

    let c = linear_2point_crossover(&p1[..], &p2[..], (0,4), (0, 5));
    assert_eq!(&[5,6,7,8,9], &c[..]);
}
