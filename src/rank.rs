use ndarray::{Array, ArrayBase, Axis, Data, Dimension, NdIndex, RemoveAxis};

/// Method for breaking ties among ranks. Either the minimum, maximum, or
/// average rank can be used.
#[derive(Clone, Copy)]
pub enum RankMethod {
    Minimum,
    Maximum,
    Average,
}

pub trait RankExt<A, S, D>
where
    S: Data<Elem = A>,
{
    /// Returns an array of the same size as the original, where each value is
    /// replaced with a rank. Zero is reserved for elements whose rank cannot be
    /// computed (i.e. NaN values in floating point matrices). The lowest rank
    /// is one.
    fn rank(&self, method: RankMethod) -> Array<usize, D>;
}

pub trait RankAxisExt<A, S, D>
where
    S: Data<Elem = A>,
{
    /// Returns an array of the same size as the original, where each value is
    /// replaced with a rank across all values sharing that element's position
    /// along the given axis. For example, in a 2d matrix, setting Axis(0) will
    /// rank elements within rows.
    fn rank_axis(&self, axis: Axis, method: RankMethod) -> Array<usize, D>;
}

impl<A, S, D> RankExt<A, S, D> for ArrayBase<S, D>
where
    A: PartialOrd + Default,
    S: Data<Elem = A>,
    D: Dimension,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    fn rank(&self, method: RankMethod) -> Array<usize, D> {
        let mut index_and_value = Vec::new();
        for (index, element) in self.indexed_iter() {
            if element.partial_cmp(&A::default()).is_none() {
                continue;
            }
            index_and_value.push((index, element));
        }
        index_and_value.sort_unstable_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        let mut rank: usize = 1;
        let mut index: usize = 0;

        let mut ranks = Array::zeros(self.dim());
        while index < index_and_value.len() {
            let start_index = index;
            let current_value = index_and_value.get(index).unwrap().1;
            while index < index_and_value.len()
                && index_and_value.get(index).unwrap().1 == current_value
            {
                index += 1;
            }

            let assign_rank = match method {
                RankMethod::Minimum => rank,
                RankMethod::Maximum => rank + index - start_index - 1,
                RankMethod::Average => rank + (index - start_index - 1) / 2,
            };
            for (key, _) in index_and_value[start_index..index].iter() {
                ranks[key.clone()] = assign_rank;
            }
            rank += index - start_index;
        }

        return ranks;
    }
}

impl<A, S, D> RankAxisExt<A, S, D> for ArrayBase<S, D>
where
    A: PartialOrd + Default,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Pattern: NdIndex<D>,
    <D as Dimension>::Smaller: Dimension,
    <<D as Dimension>::Smaller as Dimension>::Pattern: NdIndex<<D as Dimension>::Smaller>,
{
    fn rank_axis(&self, axis: Axis, method: RankMethod) -> Array<usize, D> {
        let mut ranks = Array::zeros(self.dim());
        for (i, subarray) in self.axis_iter(axis).enumerate() {
            let ranked = subarray.rank(method);
            ranked.assign_to(ranks.index_axis_mut(axis, i));
        }
        ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::NAN;

    #[test]
    fn rank_vector_no_ties() {
        let arr = array![4, 3, 2, 1];
        let ranks = arr.rank(RankMethod::Minimum);
        assert_eq!(ranks, array![4, 3, 2, 1]);
    }

    #[test]
    fn rank_vector_missing_values() {
        let arr = array![4., 3., NAN, 1.];
        let ranks = arr.rank(RankMethod::Minimum);
        assert_eq!(ranks, array![3, 2, 0, 1]);
    }

    #[test]
    fn rank_vector_ties_minimum() {
        let arr = array![4, 2, 2, 1];
        let ranks = arr.rank(RankMethod::Minimum);
        assert_eq!(ranks, array![4, 2, 2, 1]);
    }

    #[test]
    fn rank_vector_ties_maximum() {
        let arr = array![4, 2, 2, 1];
        let ranks = arr.rank(RankMethod::Maximum);
        assert_eq!(ranks, array![4, 3, 3, 1]);
    }

    #[test]
    fn rank_vector_ties_average() {
        let arr = array![4, 1, 1, 1];
        let ranks = arr.rank(RankMethod::Average);
        assert_eq!(ranks, array![4, 2, 2, 2]);
    }

    #[test]
    fn rank_matrix_full() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.rank(RankMethod::Minimum);
        assert_eq!(ranks, array![[6, 5, 4], [3, 2, 1]]);
    }

    #[test]
    fn rank_matrix_rows() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.rank_axis(Axis(0), RankMethod::Minimum);
        assert_eq!(ranks, array![[3, 2, 1], [3, 2, 1]]);
    }

    #[test]
    fn rank_matrix_cols() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.rank_axis(Axis(1), RankMethod::Minimum);
        assert_eq!(ranks, array![[2, 2, 2], [1, 1, 1]]);
    }
}
