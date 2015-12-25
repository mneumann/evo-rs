use std::cmp::Ordering;
use std::ops::Sub;
use std::convert::From;

pub trait MultiObjective {
    fn num_objectives() -> usize;

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
    fn from(t: (T, T)) -> MultiObjective2<T> {
        MultiObjective2 { objectives: [t.0, t.1] }
    }
}

impl<T, R> MultiObjective for MultiObjective2<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    fn num_objectives() -> usize {
        2
    }
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjective3<T>
    where T: Sized + PartialOrd + Copy + Clone
{
    pub objectives: [T; 3],
}

impl<T: Sized + PartialOrd + Copy + Clone> From<(T, T, T)> for MultiObjective3<T> {
    fn from(t: (T, T, T)) -> MultiObjective3<T> {
        MultiObjective3 { objectives: [t.0, t.1, t.2] }
    }
}

impl<T, R> MultiObjective for MultiObjective3<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    fn num_objectives() -> usize {
        3
    }
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
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
    fn from(t: (T, T, T, T)) -> MultiObjective4<T> {
        MultiObjective4 { objectives: [t.0, t.1, t.2, t.3] }
    }
}

impl<T, R> MultiObjective for MultiObjective4<T>
    where T: Copy + PartialOrd + Sub<Output = R>,
          R: Into<f32>
{
    fn num_objectives() -> usize {
        4
    }
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}
