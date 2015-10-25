use super::bit_vec::BitVec;
use std::usize;
use rand::Rng;
use super::{Probability, ProbabilityValue};

pub trait BitRepr {
    fn nbits(&self) -> usize;
    fn get_bit(&self, pos: usize) -> bool;
}

impl BitRepr for usize {
    #[inline(always)]
    fn nbits(&self) -> usize {
        usize::BITS
    }
    #[inline(always)]
    fn get_bit(&self, pos: usize) -> bool {
        assert!(pos < self.nbits());
        (*self >> pos) & 1 == 1
    }
}

impl BitRepr for bool {
    #[inline(always)]
    fn nbits(&self) -> usize {
        1
    }
    #[inline(always)]
    fn get_bit(&self, pos: usize) -> bool {
        assert!(pos == 0);
        *self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitString {
    bits: BitVec,
}

impl BitString {
    /// Append the lowest `nbits` bits of `value` to the BitString, msb first.
    pub fn append_bits_msb_from<T: BitRepr>(&mut self, value: &T, nbits: usize) {
        assert!(nbits <= value.nbits());
        self.bits.reserve(nbits);
        for i in (0..nbits).rev() {
            self.bits.push(value.get_bit(i));
        }
    }

    /// Flip bits with probability `prob`.
    /// The default rng for floats returns numbers in [0, 1)
    pub fn flip_bits_randomly<R: Rng>(&mut self, rng: &mut R, prob: Probability) {
        for i in 0..self.len() {
            if rng.gen::<ProbabilityValue>().is_probable_with(prob) {
                self.flip(i);
            }
        }
    }

    #[inline]
    pub fn flip(&mut self, pos: usize) {
        let old = self.get(pos);
        self.set(pos, !old);
    }

    #[inline(always)]
    pub fn set(&mut self, pos: usize, val: bool) {
        self.bits.set(pos, val);
    }

    #[inline(always)]
    pub fn get(&self, pos: usize) -> bool {
        self.bits[pos]
    }

    #[inline(always)]
    pub fn push(&mut self, bit: bool) {
        self.bits.push(bit);
    }

    pub fn count(&self, val: bool) -> usize {
        let mut cnt = 0;
        for v in self.bits.iter() {
            if v == val {
                cnt += 1;
            }
        }
        return cnt;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> BitString {
        BitString { bits: BitVec::with_capacity(cap) }
    }

    #[inline]
    pub fn new() -> BitString {
        BitString { bits: BitVec::new() }
    }

    #[inline]
    pub fn from_elem(len: usize, val: bool) -> BitString {
        BitString { bits: BitVec::from_elem(len, val) }
    }

    pub fn from_iter<I>(iter: I) -> BitString
        where I: Iterator<Item = bool>
    {
        let mut bs;
        match iter.size_hint() {
            (lower, Some(upper)) if upper > lower => {
                bs = BitString::with_capacity(upper);
            }
            (lower, _) => {
                bs = BitString::with_capacity(lower);
            }
        }
        for val in iter {
            bs.push(val);
        }
        return bs;
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<bool> {
        self.bits.iter().collect()
    }
}

/// One-point crossover
pub fn crossover_one_point(point: usize,
                           parents: (&BitString, &BitString))
                           -> (BitString, BitString) {
    let (pa, pb) = parents;
    assert!(point <= pa.len());
    assert!(point <= pb.len());

    // `ca` will contain pa[..point], pb[point..]
    // `cb` will contain pb[..point], pa[point..]
    let mut ca = BitString::with_capacity(pb.len());
    let mut cb = BitString::with_capacity(pa.len());

    for i in 0..point {
        ca.push(pa.get(i));
        cb.push(pb.get(i));
    }
    for i in point..pb.len() {
        ca.push(pb.get(i));
    }
    for i in point..pa.len() {
        cb.push(pa.get(i));
    }

    assert!(ca.len() == pb.len());
    assert!(cb.len() == pa.len());

    return (ca, cb);
}


#[test]
fn test_bitstring() {
    let bs = BitString::new();
    assert_eq!(0, bs.len());

    let bs = BitString::from_elem(4, true);
    assert_eq!(4, bs.len());
    assert_eq!(vec![true, true, true, true], bs.to_vec());

    let mut bs = BitString::new();
    bs.append_bits_msb_from(&0b11001, 4);
    assert_eq!(4, bs.len());
    assert_eq!(vec![true, false, false, true], bs.to_vec());

}

#[test]
fn test_crossover_one_point() {
    {
        let mut pa = BitString::new();
        pa.append_bits_msb_from(&0b1011usize, 4);
        let mut pb = BitString::new();
        pb.append_bits_msb_from(&0b0010usize, 4);

        let (ca, cb) = crossover_one_point(2, (&pa, &pb));
        assert_eq!(vec![true, false, true, false], ca.to_vec());
        assert_eq!(vec![false, false, true, true], cb.to_vec());
    }
    {
        let mut pa = BitString::new();
        pa.append_bits_msb_from(&0b1011usize, 4);
        let pb = BitString::new();

        let (ca, cb) = crossover_one_point(0, (&pa, &pb));
        assert!(ca.to_vec().is_empty());
        assert_eq!(vec![true, false, true, true], cb.to_vec());

    }
}
