#![feature(num_bits_bytes)]
extern crate bit_vec;

use bit_string::BitString;

pub mod bit_string;

/// One-point crossover
pub fn crossover_one_point(point: usize, parents: (&BitString, &BitString)) -> (BitString, BitString) {
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
fn test_crossover_one_point() {
    {
        let mut pa = BitString::new();
        pa.append_bits_msb_from(&0b1011usize, 4);
        let mut pb = BitString::new();
        pb.append_bits_msb_from(&0b0010usize, 4);

        let (ca, cb) = crossover_one_point(2, (&pa, &pb));
        assert_eq!(vec!(true, false, true, false), ca.to_vec());
        assert_eq!(vec!(false, false, true, true), cb.to_vec());
    }
    {
        let mut pa = BitString::new();
        pa.append_bits_msb_from(&0b1011usize, 4);
        let pb = BitString::new();

        let (ca, cb) = crossover_one_point(0, (&pa, &pb));
        assert!(ca.to_vec().is_empty());
        assert_eq!(vec!(true, false, true, true), cb.to_vec());

    }
}
