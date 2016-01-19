use std::cmp::Ordering;
use std::ops::Sub;
use std::convert::From;

pub trait MultiObjective {
    fn num_objectives(&self) -> usize;

    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering;

    /// Calculates the distance of objective between self and other
    fn dist_objective(&self, other: &Self, objective: usize) -> f32;
}

#[derive(Debug, Clone)]
pub struct MultiObjective2<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    pub objectives: [T; 2],
}

impl<T: Sized + PartialOrd + Copy + Clone> From<(T, T)> for MultiObjective2<T> {
    #[inline]
    fn from(t: (T, T)) -> MultiObjective2<T> {
        MultiObjective2 { objectives: [t.0, t.1] }
    }
}

impl<T, R> MultiObjective for MultiObjective2<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    #[inline]
    fn num_objectives(&self) -> usize {
        2
    }
    #[inline]
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    #[inline]
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

#[derive(Debug, Clone, Default)]
pub struct MultiObjective3<T>
    where T: Sized + PartialOrd + Copy + Clone + Default
{
    pub objectives: [T; 3],
}

impl<T, I> From<I> for MultiObjective3<T>
    where T: Sized + PartialOrd + Copy + Clone + Default,
          I: Iterator<Item = T>
{
    #[inline]
    fn from(iter: I) -> MultiObjective3<T> {
        let mut mo = Self::default();
        for (i, elm) in iter.into_iter().enumerate() {
            assert!(i < 3);
            mo.objectives[i] = elm;
        }
        mo
    }
}


impl<T, R> MultiObjective for MultiObjective3<T>
    where T: Copy + PartialOrd + Sub<Output = R> + Default,
          R: Into<f32>
{
    #[inline]
    fn num_objectives(&self) -> usize {
        3
    }
    #[inline]
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    #[inline]
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjective4<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    pub objectives: [T; 4],
}

impl<T: Sized + PartialOrd + Copy + Clone> From<(T, T, T, T)> for MultiObjective4<T> {
    #[inline]
    fn from(t: (T, T, T, T)) -> MultiObjective4<T> {
        MultiObjective4 { objectives: [t.0, t.1, t.2, t.3] }
    }
}

impl<T, R> MultiObjective for MultiObjective4<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    #[inline]
    fn num_objectives(&self) -> usize {
        4
    }
    #[inline]
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    #[inline]
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjectiveVec<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    objectives: Vec<T>,
}

impl<T: Sized + PartialOrd + Copy + Clone> From<Vec<T>> for MultiObjectiveVec<T> {
    #[inline]
    fn from(t: Vec<T>) -> MultiObjectiveVec<T> {
        MultiObjectiveVec { objectives: t }
    }
}

impl<T: Sized + PartialOrd + Copy + Clone> AsRef<[T]> for MultiObjectiveVec<T> {
    fn as_ref(&self) -> &[T] {
        &self.objectives
    }
}

impl<T, R> MultiObjective for MultiObjectiveVec<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    #[inline]
    fn num_objectives(&self) -> usize {
        self.objectives.len()
    }
    #[inline]
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    #[inline]
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}
