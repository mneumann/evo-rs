use super::bit_vec::BitVec;
use std::usize;

pub trait BitRepr {
    fn nbits(&self) -> usize;
    fn get_bit(&self, pos: usize) -> bool;
}

impl BitRepr for usize {
    fn nbits(&self) -> usize { usize::BITS }
    fn get_bit(&self, pos: usize) -> bool {
        assert!(pos < self.nbits());
        (*self >> pos) & 1 == 1
    }
}

impl BitRepr for bool {
    fn nbits(&self) -> usize { 1 }
    fn get_bit(&self, pos: usize) -> bool {
        assert!(pos == 0);
        *self
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct BitString {
    bits: BitVec,
}

impl BitString {
    /// Append the lowest `nbits` bits of `value` to the BitString, msb first.
    pub fn append_bits_msb_from<T:BitRepr>(&mut self, value: &T, nbits: usize) {
        assert!(nbits <= value.nbits());
        self.bits.reserve(nbits);
        for i in (0 .. nbits).rev() {
            self.bits.push(value.get_bit(i));
        }
    }

    pub fn get(&self, pos: usize) -> bool {
        self.bits[pos]
    }

    pub fn push(&mut self, bit: bool) {
        self.bits.push(bit);
    }

    pub fn len(&self) -> usize { self.bits.len() }

    pub fn with_capacity(cap: usize) -> BitString {
        BitString {bits: BitVec::with_capacity(cap)}
    }

    pub fn new() -> BitString {
        BitString {bits: BitVec::new()}
    }

    pub fn from_elem(len: usize, val: bool) -> BitString {
        BitString {bits: BitVec::from_elem(len, val)}
    }

    pub fn to_vec(&self) -> Vec<bool> {
        self.bits.iter().collect()
    }
}

#[test]
fn test_bitstring() {
    let bs = BitString::new();
    assert_eq!(0, bs.len());

    let bs = BitString::from_elem(4, true);
    assert_eq!(4, bs.len());
    assert_eq!(vec!(true, true, true, true), bs.to_vec());

    let mut bs = BitString::new();
    bs.append_bits_msb_from(&0b11001, 4);
    assert_eq!(4, bs.len());
    assert_eq!(vec!(true, false, false, true), bs.to_vec());
}
