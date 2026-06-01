//! JEPA embeddings live on a Riemannian manifold.
//! Fisher information metric and geometric analysis.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A JEPA embedding: a point on the information manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaEmbedding {
    /// Dimension of the embedding space.
    pub dim: usize,
    /// Embedding coordinates.
    pub coords: Vec<f64>,
}

impl JepaEmbedding {
    /// Create a zero embedding.
    pub fn zero(dim: usize) -> Self {
        Self { dim, coords: vec![0.0; dim] }
    }

    /// Create from a vector.
    pub fn from_vec(coords: Vec<f64>) -> Self {
        let dim = coords.len();
        Self { dim, coords }
    }

    /// Norm of the embedding.
    pub fn norm(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return self.clone();
        }
        Self::from_vec(self.coords.iter().map(|x| x / n).collect())
    }

    /// Euclidean distance to another embedding.
    pub fn euclidean_distance(&self, other: &JepaEmbedding) -> f64 {
        self.coords.iter().zip(&other.coords)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Inner product with another embedding.
    pub fn dot(&self, other: &JepaEmbedding) -> f64 {
        self.coords.iter().zip(&other.coords)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Convert to nalgebra vector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.coords.clone())
    }
}

/// The Fisher information metric tensor at a point on the manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherMetric {
    pub dim: usize,
    /// Metric tensor as a flat matrix (dim × dim).
    pub metric: Vec<Vec<f64>>,
}

impl FisherMetric {
    /// Identity metric (Euclidean space).
    pub fn identity(dim: usize) -> Self {
        let metric = (0..dim).map(|i| {
            (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
        }).collect();
        Self { dim, metric }
    }

    /// Diagonal metric from variances (inverse of covariance diagonal).
    pub fn from_variances(variances: &[f64]) -> Self {
        let dim = variances.len();
        let metric = (0..dim).map(|i| {
            (0..dim).map(|j| {
                if i == j { 1.0 / variances[i].max(1e-10) } else { 0.0 }
            }).collect()
        }).collect();
        Self { dim, metric }
    }

    /// Full Fisher metric from Jacobian of the embedding map.
    pub fn from_jacobian(jacobian: &DMatrix<f64>) -> Self {
        let dim = jacobian.nrows();
        // G = J^T J
        let g = &jacobian.transpose() * jacobian;
        let metric = (0..dim).map(|i| {
            (0..dim).map(|j| g[(i, j)]).collect()
        }).collect();
        Self { dim, metric }
    }

    /// Compute the geodesic distance between two embeddings under this metric.
    pub fn geodesic_distance(&self, a: &JepaEmbedding, b: &JepaEmbedding) -> f64 {
        // d(x,y)² = (x-y)^T G (x-y)
        let diff: Vec<f64> = a.coords.iter().zip(&b.coords).map(|(x, y)| x - y).collect();
        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                dist_sq += self.metric[i][j] * diff[i] * diff[j];
            }
        }
        dist_sq.max(0.0).sqrt()
    }

    /// Volume element: sqrt(det(G)).
    pub fn volume_element(&self) -> f64 {
        let g = DMatrix::from_row_slice(
            self.dim, self.dim,
            &self.metric.iter().flat_map(|r| r.iter().copied()).collect::<Vec<_>>()
        );
        // Determinant via LU decomposition
        let det = g.determinant();
        det.abs().sqrt()
    }

    /// Compute Christoffel symbols (Levi-Civita connection).
    /// Returns Γ^k_{ij} for all i,j,k (simplified: assumes constant metric).
    pub fn christoffel_symbols(&self) -> Vec<Vec<Vec<f64>>> {
        let n = self.dim;
        // For constant metric, all Christoffel symbols are zero
        vec![vec![vec![0.0; n]; n]; n]
    }
}

/// A collection of JEPA embeddings forming a dataset on the manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaManifold {
    pub embeddings: Vec<JepaEmbedding>,
    pub metric: FisherMetric,
}

impl JepaManifold {
    /// Create a new manifold with identity metric.
    pub fn new(embeddings: Vec<JepaEmbedding>) -> Self {
        let dim = embeddings.first().map(|e| e.dim).unwrap_or(0);
        Self {
            embeddings,
            metric: FisherMetric::identity(dim),
        }
    }

    /// Compute pairwise geodesic distance matrix.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.embeddings.len();
        (0..n).map(|i| {
            (0..n).map(|j| {
                self.metric.geodesic_distance(&self.embeddings[i], &self.embeddings[j])
            }).collect()
        }).collect()
    }

    /// Compute the Riemannian center of mass (Frechet mean) under identity metric.
    pub fn frechet_mean(&self) -> JepaEmbedding {
        if self.embeddings.is_empty() {
            return JepaEmbedding::zero(0);
        }
        let dim = self.embeddings[0].dim;
        let n = self.embeddings.len() as f64;
        let mean: Vec<f64> = (0..dim).map(|k| {
            self.embeddings.iter().map(|e| e.coords[k]).sum::<f64>() / n
        }).collect();
        JepaEmbedding::from_vec(mean)
    }

    /// Compute sectional curvature (simplified: 0 for flat metric).
    pub fn sectional_curvature(&self, _i: usize, _j: usize) -> f64 {
        0.0
    }

    /// Compute the exponential map at a point (identity for flat space).
    pub fn exp_map(&self, point: &JepaEmbedding, tangent: &JepaEmbedding) -> JepaEmbedding {
        assert_eq!(point.dim, tangent.dim);
        JepaEmbedding::from_vec(
            point.coords.iter().zip(&tangent.coords).map(|(p, t)| p + t).collect()
        )
    }

    /// Compute the logarithmic map (inverse of exponential).
    pub fn log_map(&self, base: &JepaEmbedding, point: &JepaEmbedding) -> JepaEmbedding {
        assert_eq!(base.dim, point.dim);
        JepaEmbedding::from_vec(
            base.coords.iter().zip(&point.coords).map(|(b, p)| p - b).collect()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_zero() {
        let e = JepaEmbedding::zero(4);
        assert_eq!(e.dim, 4);
        assert_eq!(e.norm(), 0.0);
    }

    #[test]
    fn test_embedding_normalize() {
        let e = JepaEmbedding::from_vec(vec![3.0, 4.0]);
        let n = e.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = JepaEmbedding::from_vec(vec![0.0, 0.0]);
        let b = JepaEmbedding::from_vec(vec![3.0, 4.0]);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = JepaEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let b = JepaEmbedding::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_metric() {
        let m = FisherMetric::identity(3);
        let a = JepaEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let b = JepaEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        assert!((m.geodesic_distance(&a, &b) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_metric() {
        let m = FisherMetric::from_variances(&[1.0, 4.0]);
        let a = JepaEmbedding::from_vec(vec![1.0, 0.0]);
        let b = JepaEmbedding::from_vec(vec![0.0, 2.0]);
        // d² = (1)² * 1.0 + (2)² * 0.25 = 1 + 1 = 2
        assert!((m.geodesic_distance(&a, &b) - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_volume_element() {
        let m = FisherMetric::identity(3);
        assert!((m.volume_element() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manifold_distance_matrix() {
        let manifold = JepaManifold::new(vec![
            JepaEmbedding::from_vec(vec![0.0, 0.0]),
            JepaEmbedding::from_vec(vec![1.0, 0.0]),
            JepaEmbedding::from_vec(vec![0.0, 1.0]),
        ]);
        let dm = manifold.distance_matrix();
        assert_eq!(dm.len(), 3);
        assert!((dm[0][1] - 1.0).abs() < 1e-10);
        assert!((dm[0][2] - 1.0).abs() < 1e-10);
        assert!((dm[1][2] - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_mean() {
        let manifold = JepaManifold::new(vec![
            JepaEmbedding::from_vec(vec![0.0, 0.0]),
            JepaEmbedding::from_vec(vec![2.0, 4.0]),
        ]);
        let mean = manifold.frechet_mean();
        assert!((mean.coords[0] - 1.0).abs() < 1e-10);
        assert!((mean.coords[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_map_inverse() {
        let manifold = JepaManifold::new(vec![JepaEmbedding::zero(3)]);
        let base = JepaEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent = JepaEmbedding::from_vec(vec![0.5, -0.5, 1.0]);
        let exp = manifold.exp_map(&base, &tangent);
        let log = manifold.log_map(&base, &exp);
        for i in 0..3 {
            assert!((log.coords[i] - tangent.coords[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_christoffel_symbols_zero() {
        let m = FisherMetric::identity(2);
        let gamma = m.christoffel_symbols();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert!(gamma[k][i][j].abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_from_jacobian() {
        let j = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let m = FisherMetric::from_jacobian(&j);
        // G = J^T J = [[1,0],[0,4]]
        assert!((m.metric[0][0] - 1.0).abs() < 1e-10);
        assert!((m.metric[1][1] - 4.0).abs() < 1e-10);
    }
}
