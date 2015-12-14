use std::ops::Add;
use rand::{Rand, Rng};

/// A probability is a value in the range [0.0, 1.0].
/// A probability of 1.0 means `always`, that of 0.0
/// means `never`.
#[derive(Copy, Clone, Debug)]
pub struct Probability(f32);

impl Probability {
    #[inline]
    pub fn new(pb: f32) -> Probability {
        assert!(pb >= 0.0 && pb <= 1.0);
        Probability(pb)
    }
}

impl Add<Probability> for Probability {
    type Output = Probability;
    #[inline]
    fn add(self, rhs: Probability) -> Probability {
        Probability::new(self.0 + rhs.0)
    }
}

/// A probability value is a random value in the
/// range [0.0, 1.0).
#[derive(Copy, Clone, Debug)]
pub struct ProbabilityValue(f32);

impl Rand for ProbabilityValue {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> ProbabilityValue {
        let val = rng.gen::<f32>();
        assert!(val >= 0.0 && val < 1.0);
        ProbabilityValue(val)
    }
}

impl ProbabilityValue {
    #[inline]
    pub fn is_probable_with(self, pb: Probability) -> bool {
        self.0 < pb.0
    }
}
