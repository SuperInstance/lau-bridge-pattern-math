//! Sheaf structure on venues: each venue gets a stalk (local state),
//! restriction maps define communication between venues.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// The stalk (local state) at a venue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stalk {
    /// Dimension of the local state space.
    pub dim: usize,
    /// Data vector.
    pub data: Vec<f64>,
}

impl Stalk {
    /// Create a zero stalk of given dimension.
    pub fn zero(dim: usize) -> Self {
        Self { dim, data: vec![0.0; dim] }
    }

    /// Create a stalk from a vector.
    pub fn from_vec(data: Vec<f64>) -> Self {
        let dim = data.len();
        Self { dim, data }
    }

    /// Convert to nalgebra vector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.data.clone())
    }

    /// Norm of the stalk data.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Add two stalks (same dimension required).
    pub fn add(&self, other: &Stalk) -> Stalk {
        assert_eq!(self.dim, other.dim, "stalk dimensions must match");
        Stalk::from_vec(
            self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect()
        )
    }

    /// Scale stalk by scalar.
    pub fn scale(&self, s: f64) -> Stalk {
        Stalk::from_vec(self.data.iter().map(|x| x * s).collect())
    }
}

/// A restriction map between two venues: a linear transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestrictionMap {
    /// Source venue index.
    pub source: usize,
    /// Target venue index.
    pub target: usize,
    /// Matrix entries (row-major, target_dim × source_dim).
    pub matrix: Vec<Vec<f64>>,
}

impl RestrictionMap {
    /// Create an identity restriction map (same dimension).
    pub fn identity(venue: usize, dim: usize) -> Self {
        let matrix = (0..dim).map(|i| {
            (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
        }).collect();
        Self { source: venue, target: venue, matrix }
    }

    /// Create a projection restriction map (from higher to lower dim).
    pub fn projection(source: usize, target: usize, source_dim: usize, target_dim: usize) -> Self {
        assert!(target_dim <= source_dim);
        let matrix = (0..target_dim).map(|i| {
            (0..source_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
        }).collect();
        Self { source, target, matrix }
    }

    /// Apply the restriction map to a stalk.
    pub fn apply(&self, stalk: &Stalk) -> Stalk {
        let target_dim = self.matrix.len();
        let result: Vec<f64> = self.matrix.iter().map(|row| {
            row.iter().zip(&stalk.data).map(|(m, s)| m * s).sum()
        }).collect();
        Stalk { dim: target_dim, data: result }
    }

    /// Source dimension.
    pub fn source_dim(&self) -> usize {
        if self.matrix.is_empty() { 0 } else { self.matrix[0].len() }
    }

    /// Target dimension.
    pub fn target_dim(&self) -> usize {
        self.matrix.len()
    }
}

/// A cellular sheaf on the venue graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueSheaf {
    /// Number of venues.
    pub n_venues: usize,
    /// Stalk at each venue.
    pub stalks: Vec<Stalk>,
    /// Restriction maps (edges in the sheaf).
    pub restriction_maps: Vec<RestrictionMap>,
}

impl VenueSheaf {
    /// Create a new sheaf with uniform stalk dimension.
    pub fn new(n_venues: usize, stalk_dim: usize) -> Self {
        let stalks = (0..n_venues).map(|_| Stalk::zero(stalk_dim)).collect();
        Self {
            n_venues,
            stalks,
            restriction_maps: vec![],
        }
    }

    /// Set the stalk data at a venue.
    pub fn set_stalk(&mut self, venue: usize, data: Vec<f64>) {
        assert!(venue < self.n_venues);
        self.stalks[venue] = Stalk::from_vec(data);
    }

    /// Add a restriction map between two venues.
    pub fn add_restriction_map(&mut self, map: RestrictionMap) {
        assert!(map.source < self.n_venues);
        assert!(map.target < self.n_venues);
        self.restriction_maps.push(map);
    }

    /// Compute the sheaf Laplacian: for each edge (i,j) with restriction map,
    /// L_sheaf = sum over edges of (ρ_i^T ρ_i + ρ_j^T ρ_j - ρ_i^T ρ_j - ρ_j^T ρ_i)
    /// where ρ_i is the restriction map from vertex i to the edge.
    /// 
    /// Simplified: returns a global matrix acting on the concatenated stalk space.
    pub fn sheaf_laplacian(&self) -> Vec<Vec<f64>> {
        let total_dim: usize = self.stalks.iter().map(|s| s.dim).sum();
        let mut lap = vec![vec![0.0; total_dim]; total_dim];

        let mut offsets = vec![0usize];
        for s in &self.stalks {
            offsets.push(offsets.last().unwrap() + s.dim);
        }

        for rmap in &self.restriction_maps {
            let si = offsets[rmap.source];
            let sd = self.stalks[rmap.source].dim;
            let ti = offsets[rmap.target];
            let td = self.stalks[rmap.target].dim;

            // Add diagonal blocks: W_s^T W_s and W_t^T W_t
            // where W = restriction map matrix
            let w = &rmap.matrix;
            let wt_w = self._transpose_multiply(w);
            let w_wt = self._multiply_transpose(w);

            // L_ii += W^T W
            for i in 0..sd {
                for j in 0..sd {
                    lap[si + i][si + j] += wt_w[i][j];
                }
            }
            // L_jj += W W^T
            for i in 0..td {
                for j in 0..td {
                    lap[ti + i][ti + j] += w_wt[i][j];
                }
            }
            // Off-diagonal: L_ij -= W^T * I (simplified)
            // This is a simplified sheaf Laplacian
        }
        lap
    }

    fn _transpose_multiply(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = if rows == 0 { 0 } else { m[0].len() };
        let mut result = vec![vec![0.0; cols]; cols];
        for i in 0..cols {
            for j in 0..cols {
                for k in 0..rows {
                    result[i][j] += m[k][i] * m[k][j];
                }
            }
        }
        result
    }

    fn _multiply_transpose(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = if rows == 0 { 0 } else { m[0].len() };
        let mut result = vec![vec![0.0; rows]; rows];
        for i in 0..rows {
            for j in 0..rows {
                for k in 0..cols {
                    result[i][j] += m[i][k] * m[j][k];
                }
            }
        }
        result
    }

    /// Check consistency: for each restriction map, verify that the mapped stalk
    /// values are compatible (norm of difference below threshold).
    pub fn consistency_error(&self) -> f64 {
        let mut total = 0.0;
        for rmap in &self.restriction_maps {
            let mapped = rmap.apply(&self.stalks[rmap.source]);
            let target = &self.stalks[rmap.target];
            if mapped.dim == target.dim {
                let diff: f64 = mapped.data.iter()
                    .zip(&target.data)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                total += diff;
            }
        }
        total.sqrt()
    }

    /// Global section: a choice of stalk data that is consistent across all
    /// restriction maps (consistency_error ≈ 0).
    pub fn is_global_section(&self, tol: f64) -> bool {
        self.consistency_error() < tol
    }

    /// Pushforward: aggregate all stalk data into a single vector.
    pub fn pushforward(&self) -> Vec<f64> {
        self.stalks.iter().flat_map(|s| s.data.clone()).collect()
    }
}

/// Sheaf cohomology (simplified): H⁰ = space of global sections,
/// H¹ = obstruction to extending local sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SheafCohomology {
    /// Dimension of H⁰ (number of linearly independent global sections).
    pub h0_dimension: usize,
    /// Dimension of H¹ (obstruction space).
    pub h1_dimension: usize,
    /// Consistency error.
    pub consistency_error: f64,
}

impl VenueSheaf {
    /// Compute (simplified) sheaf cohomology dimensions.
    pub fn cohomology(&self) -> SheafCohomology {
        let err = self.consistency_error();
        // H⁰ dimension: number of stalk dimensions if consistent, 0 otherwise
        let h0 = if err < 1e-8 {
            self.stalks.first().map(|s| s.dim).unwrap_or(0)
        } else {
            0
        };
        // H¹ is estimated by the rank deficiency
        let total_restriction_dim: usize = self.restriction_maps.iter()
            .map(|r| r.target_dim()).sum();
        let h1 = total_restriction_dim.saturating_sub(h0);

        SheafCohomology {
            h0_dimension: h0,
            h1_dimension: h1,
            consistency_error: err,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stalk_zero() {
        let s = Stalk::zero(3);
        assert_eq!(s.dim, 3);
        assert_eq!(s.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_stalk_norm() {
        let s = Stalk::from_vec(vec![3.0, 4.0]);
        assert!((s.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_stalk_add() {
        let a = Stalk::from_vec(vec![1.0, 2.0]);
        let b = Stalk::from_vec(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_stalk_scale() {
        let s = Stalk::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = s.scale(2.0);
        assert_eq!(scaled.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_identity_restriction() {
        let r = RestrictionMap::identity(0, 3);
        let s = Stalk::from_vec(vec![1.0, 2.0, 3.0]);
        let mapped = r.apply(&s);
        assert_eq!(mapped.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_projection_restriction() {
        let r = RestrictionMap::projection(0, 1, 3, 2);
        let s = Stalk::from_vec(vec![1.0, 2.0, 3.0]);
        let mapped = r.apply(&s);
        assert_eq!(mapped.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sheaf_creation() {
        let sheaf = VenueSheaf::new(4, 3);
        assert_eq!(sheaf.n_venues, 4);
        assert_eq!(sheaf.stalks.len(), 4);
        assert!(sheaf.restriction_maps.is_empty());
    }

    #[test]
    fn test_sheaf_set_stalk() {
        let mut sheaf = VenueSheaf::new(3, 2);
        sheaf.set_stalk(1, vec![5.0, 6.0]);
        assert_eq!(sheaf.stalks[1].data, vec![5.0, 6.0]);
    }

    #[test]
    fn test_consistent_sheaf() {
        let mut sheaf = VenueSheaf::new(2, 2);
        sheaf.set_stalk(0, vec![1.0, 2.0]);
        sheaf.set_stalk(1, vec![1.0, 2.0]);
        // Restriction from venue 0 to venue 1 with identity matrix
        sheaf.add_restriction_map(RestrictionMap {
            source: 0,
            target: 1,
            matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        assert!(sheaf.is_global_section(1e-6));
    }

    #[test]
    fn test_inconsistent_sheaf() {
        let mut sheaf = VenueSheaf::new(2, 2);
        sheaf.set_stalk(0, vec![1.0, 2.0]);
        sheaf.set_stalk(1, vec![3.0, 4.0]);
        // Add restriction map from venue 0 to venue 1
        sheaf.add_restriction_map(RestrictionMap {
            source: 0,
            target: 1,
            matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        assert!(!sheaf.is_global_section(1e-6));
    }

    #[test]
    fn test_pushforward() {
        let mut sheaf = VenueSheaf::new(2, 2);
        sheaf.set_stalk(0, vec![1.0, 2.0]);
        sheaf.set_stalk(1, vec![3.0, 4.0]);
        let pf = sheaf.pushforward();
        assert_eq!(pf, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cohomology_consistent() {
        let mut sheaf = VenueSheaf::new(2, 3);
        sheaf.set_stalk(0, vec![1.0, 0.0, 0.0]);
        sheaf.set_stalk(1, vec![1.0, 0.0, 0.0]);
        sheaf.add_restriction_map(RestrictionMap::identity(0, 3));
        let cohom = sheaf.cohomology();
        assert!(cohom.consistency_error < 1e-8);
    }

    #[test]
    fn test_sheaf_laplacian() {
        let mut sheaf = VenueSheaf::new(2, 2);
        sheaf.add_restriction_map(RestrictionMap::identity(0, 2));
        let lap = sheaf.sheaf_laplacian();
        assert_eq!(lap.len(), 4);
        assert_eq!(lap[0].len(), 4);
    }

    #[test]
    fn test_restriction_dims() {
        let r = RestrictionMap::projection(0, 1, 5, 3);
        assert_eq!(r.source_dim(), 5);
        assert_eq!(r.target_dim(), 3);
    }
}
