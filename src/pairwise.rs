use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, NdIndex};

pub trait PairwiseInplaceExt<A, S, SS, D>
where
    S: DataMut<Elem = A>,
    SS: Data<Elem = A>,
{
    /// Takes the elementwise maximum with another array.
    fn maximum_with_inplace(&mut self, other: &ArrayBase<SS, D>);

    /// Takes the elementwise minimum with another array.
    fn minimum_with_inplace(&mut self, other: &ArrayBase<SS, D>);
}

pub trait PairwiseExt<A, S, D>
where
    S: Data<Elem = A>,
{
    /// Returns the elementwise maximum with another array.
    fn maximum_with(&self, other: &ArrayBase<S, D>) -> Array<A, D>;

    /// Returns the elementwise minimum with another array.
    fn minimum_with(&self, other: &ArrayBase<S, D>) -> Array<A, D>;
}

impl<A, S, D> PairwiseExt<A, S, D> for ArrayBase<S, D>
where
    A: PartialOrd + Copy,
    S: Data<Elem = A>,
    D: Dimension,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    fn maximum_with(&self, other: &ArrayBase<S, D>) -> Array<A, D> {
        let mut array = self.to_owned();
        array.maximum_with_inplace(other);
        array
    }

    fn minimum_with(&self, other: &ArrayBase<S, D>) -> Array<A, D> {
        let mut array = self.to_owned();
        array.minimum_with_inplace(other);
        array
    }
}

impl<A, S, SS, D> PairwiseInplaceExt<A, S, SS, D> for ArrayBase<S, D>
where
    A: PartialOrd + Copy,
    S: DataMut<Elem = A>,
    SS: Data<Elem = A>,
    D: Dimension,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    fn maximum_with_inplace(&mut self, other: &ArrayBase<SS, D>) {
        for (i, val) in self.indexed_iter_mut() {
            let o = &other[i];
            if *val < *o {
                *val = *o;
            }
        }
    }

    fn minimum_with_inplace(&mut self, other: &ArrayBase<SS, D>) {
        for (i, val) in self.indexed_iter_mut() {
            let o = &other[i];
            if *val > *o {
                *val = *o;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn pairwise() {
        let lhs = array![1., 2., 3.];
        let rhs = array![-1., 2., 5.];

        assert_eq!(lhs.maximum_with(&rhs), array![1., 2., 5.]);
        assert_eq!(lhs.minimum_with(&rhs), array![-1., 2., 3.]);
    }

    #[test]
    fn inplace() {
        let mut lhs = array![1, 2, 3];
        let rhs = array![-1, 2, 5];

        lhs.maximum_with_inplace(&rhs);
        assert_eq!(lhs, array![1, 2, 5]);

        lhs.minimum_with_inplace(&rhs);
        assert_eq!(lhs, array![-1, 2, 5]);
    }
}
