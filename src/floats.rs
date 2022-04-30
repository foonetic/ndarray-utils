use ndarray::{Array, ArrayBase, Axis, Data, DataMut, Dimension, RemoveAxis};
use num_traits::Float;

pub trait FillInPlaceExt<A, S, D>
where
    S: DataMut<Elem = A>,
    A: Float,
{
    /// Fills non-finite floating point values (NaN, infinity, and negative
    /// infinity) with the given replacement.
    fn fill_non_finite_inplace(&mut self, with: A);
}

impl<A, S, D> FillInPlaceExt<A, S, D> for ArrayBase<S, D>
where
    A: Float,
    S: DataMut<Elem = A>,
    D: Dimension,
{
    fn fill_non_finite_inplace(&mut self, with: A) {
        self.map_inplace(|x| {
            if !x.is_finite() {
                *x = with;
            }
        });
    }
}

pub trait CountExt<A, S, D>
where
    S: Data<Elem = A>,
    A: Float,
{
    /// Returns the number of finite values.
    fn count_finite(&self) -> usize;

    /// Returns the number of non-finite values.
    fn count_non_finite(&self) -> usize;
}

impl<A, S, D> CountExt<A, S, D> for ArrayBase<S, D>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn count_finite(&self) -> usize {
        self.fold(0, |a, b| a + if b.is_finite() { 1 } else { 0 })
    }

    fn count_non_finite(&self) -> usize {
        self.fold(0, |a, b| a + if !b.is_finite() { 1 } else { 0 })
    }
}

pub trait CountAxisExt<A, S, D>
where
    S: Data<Elem = A>,
    A: Float,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Smaller: Dimension,
{
    /// Returns the number of finite values for each index along the given axis.
    /// For example, in a matrix, specifying Axis(0) will give the number of
    /// finite values per row.
    fn count_finite_axis(&self, axis: Axis) -> Array<usize, D::Smaller>;

    /// Returns the number of non-finite values for each index along the given
    /// axis.  For example, in a matrix, specifying Axis(0) will give the number
    /// of non-finite values per row.
    fn count_non_finite_axis(&self, axis: Axis) -> Array<usize, D::Smaller>;
}

impl<A, S, D> CountAxisExt<A, S, D> for ArrayBase<S, D>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Smaller: Dimension,
    Array<usize, <D as Dimension>::Smaller>: FromIterator<usize>,
{
    fn count_finite_axis(&self, axis: Axis) -> Array<usize, D::Smaller> {
        self.axis_iter(axis)
            .map(|view| view.count_finite())
            .collect()
    }

    fn count_non_finite_axis(&self, axis: Axis) -> Array<usize, D::Smaller> {
        self.axis_iter(axis)
            .map(|view| view.count_non_finite())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::NAN;

    #[test]
    fn count_and_fill() {
        let mut vals = array![1., 2., NAN, 3.];
        assert_eq!(3, vals.count_finite());
        assert_eq!(1, vals.count_non_finite());
        vals.fill_non_finite_inplace(42.);
        assert_eq!(vals, array![1., 2., 42., 3.]);
        assert_eq!(4, vals.count_finite());
        assert_eq!(0, vals.count_non_finite());
    }

    #[test]
    fn count_matrix() {
        let vals = array![[1., 2., NAN, 3.], [NAN, 4., 5., NAN]];
        assert_eq!(5, vals.count_finite());
        assert_eq!(array![3, 2], vals.count_finite_axis(Axis(0)));
        assert_eq!(array![1, 2], vals.count_non_finite_axis(Axis(0)));
        assert_eq!(array![1, 2, 1, 1], vals.count_finite_axis(Axis(1)));
        assert_eq!(array![1, 0, 1, 1], vals.count_non_finite_axis(Axis(1)));
    }
}
