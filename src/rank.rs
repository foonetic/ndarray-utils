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

    /// Returns an array of the same size of the original, where each value is
    /// replaced with a bucket identifier. Zero is reserved for elements whose
    /// bucket cannot be computed (i.e. NaN values in floating point matrices).
    /// The lowest bucket is one and the maximum bucket is the given number.
    fn discretize(&self, method: RankMethod, buckets: usize) -> Array<usize, D>;
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

    /// Returns an array of the same size as the original, where each value is
    /// replaced with a bucket across all values sharing that element's position
    /// along the given axis. For example, in a 2d matrix, setting Axis(0) will
    /// bucket elements within rows.
    fn discretize_axis(&self, axis: Axis, method: RankMethod, buckets: usize) -> Array<usize, D>;
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

    fn discretize(&self, method: RankMethod, buckets: usize) -> Array<usize, D> {
        let mut ranks = self.rank(method);
        if let Some(max_rank) = ranks.iter().reduce(|a, b| if *a > *b { a } else { b }) {
            let ranks_per_bucket = *max_rank / buckets;

            // As a special case, there isn't enough data to cover all the buckets.
            let (buckets, ranks_per_bucket) = if ranks_per_bucket == 0 {
                (*max_rank, 1)
            } else {
                (buckets, ranks_per_bucket)
            };

            let remainder = *max_rank % buckets;

            let mut rank_cut_points = Vec::new();
            let mut low_rank: usize = 1;
            // Separate handling of the remainder and non-remainder cases to
            // allocate the ranks that don't evenly divide the buckets.
            for _ in 0..remainder {
                // For example: if the low rank of this bucket is 1 and there
                // are normally 2 ranks per bucket, then in the remainder case
                // the first bucket should include the extra element, i.e. the
                // range [1, 2, 3].
                let high_rank = low_rank + ranks_per_bucket;
                rank_cut_points.push(low_rank);
                low_rank = high_rank + 1;
            }
            for _ in remainder..buckets {
                let high_rank = low_rank + ranks_per_bucket - 1;
                rank_cut_points.push(low_rank);
                low_rank = high_rank + 1;
            }
            ranks.map_inplace(|x| {
                if *x == 0 {
                    return;
                }
                let mut bucket = 0;
                for cut in rank_cut_points.iter() {
                    if *x >= *cut {
                        bucket += 1;
                    } else {
                        break;
                    }
                }
                *x = bucket;
            });
        }
        ranks
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

    fn discretize_axis(&self, axis: Axis, method: RankMethod, buckets: usize) -> Array<usize, D> {
        let mut ranks = Array::zeros(self.dim());
        for (i, subarray) in self.axis_iter(axis).enumerate() {
            let ranked = subarray.discretize(method, buckets);
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

    #[test]
    fn discretize_matrix_full() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.discretize(RankMethod::Minimum, 3);
        assert_eq!(ranks, array![[3, 3, 2], [2, 1, 1]]);
    }

    #[test]
    fn discretize_matrix_rows() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.discretize_axis(Axis(0), RankMethod::Minimum, 2);
        assert_eq!(ranks, array![[2, 1, 1], [2, 1, 1]]);
    }

    #[test]
    fn discretize_matrix_cols() {
        let arr = array![[6, 5, 4], [3, 2, 1]];
        let ranks = arr.discretize_axis(Axis(1), RankMethod::Minimum, 2);
        assert_eq!(ranks, array![[2, 2, 2], [1, 1, 1]]);
    }

    #[test]
    fn discretize_matrix_with_missing_values() {
        let arr = array![[6., 5., NAN], [3., NAN, 1.]];
        let ranks = arr.discretize(RankMethod::Minimum, 2);
        assert_eq!(ranks, array![[2, 2, 0], [1, 0, 1]]);
    }
}
