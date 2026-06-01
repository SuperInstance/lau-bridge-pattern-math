//! Optimal transport of "vibes" between venues.
//! Wasserstein distance on venue state distributions.

use serde::{Serialize, Deserialize};

/// A discrete probability distribution over venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeDistribution {
    /// Venue labels.
    pub venues: Vec<String>,
    /// Probability masses (must sum to 1).
    pub masses: Vec<f64>,
}

impl VibeDistribution {
    /// Create a uniform distribution over n venues.
    pub fn uniform(n: usize) -> Self {
        let mass = 1.0 / n as f64;
        Self {
            venues: (0..n).map(|i| format!("venue_{}", i)).collect(),
            masses: vec![mass; n],
        }
    }

    /// Create from raw masses (normalizes automatically).
    pub fn from_masses(masses: Vec<f64>) -> Self {
        let total: f64 = masses.iter().sum();
        let n = masses.len();
        let normalized = if total > 0.0 {
            masses.iter().map(|m| m / total).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
        Self {
            venues: (0..n).map(|i| format!("venue_{}", i)).collect(),
            masses: normalized,
        }
    }

    /// Check if distribution is valid (non-negative, sums to ~1).
    pub fn is_valid(&self) -> bool {
        let total: f64 = self.masses.iter().sum();
        self.masses.iter().all(|&m| m >= 0.0) && (total - 1.0).abs() < 1e-8
    }

    /// Number of venues.
    pub fn n(&self) -> usize {
        self.masses.len()
    }
}

/// Cost matrix for transport between venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMatrix {
    pub n: usize,
    pub costs: Vec<Vec<f64>>,
}

impl CostMatrix {
    /// Create from a distance matrix.
    pub fn from_distances(distances: Vec<Vec<f64>>) -> Self {
        let n = distances.len();
        Self { n, costs: distances }
    }

    /// Uniform cost matrix (all costs = 1).
    pub fn uniform(n: usize) -> Self {
        Self {
            n,
            costs: vec![vec![1.0; n]; n],
        }
    }

    /// Euclidean distance cost matrix from 1D positions.
    pub fn from_positions_1d(positions: &[f64]) -> Self {
        let n = positions.len();
        let costs: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..n).map(|j| (positions[i] - positions[j]).abs()).collect()
        }).collect();
        Self { n, costs }
    }

    /// Euclidean distance cost matrix from 2D positions.
    pub fn from_positions_2d(positions: &[(f64, f64)]) -> Self {
        let n = positions.len();
        let costs: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..n).map(|j| {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                (dx * dx + dy * dy).sqrt()
            }).collect()
        }).collect();
        Self { n, costs }
    }
}

/// Transport plan (coupling matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportPlan {
    /// Transport matrix T[i][j] = amount transported from i to j.
    pub plan: Vec<Vec<f64>>,
    /// Total transport cost.
    pub cost: f64,
}

/// Compute Wasserstein-1 distance (Earth Mover's Distance) between two 1D distributions.
/// Uses the closed-form solution for 1D: integral of |CDF_1 - CDF_2|.
pub fn wasserstein_1d(source: &VibeDistribution, target: &VibeDistribution) -> f64 {
    assert_eq!(source.n(), target.n(), "distributions must have same support size");
    let n = source.n();
    let mut cdf_diff = 0.0;
    let mut source_cdf = 0.0;
    let mut target_cdf = 0.0;
    for i in 0..n {
        source_cdf += source.masses[i];
        target_cdf += target.masses[i];
        cdf_diff += (source_cdf - target_cdf).abs();
    }
    cdf_diff
}

/// Compute Wasserstein-1 distance with arbitrary cost matrix.
/// Uses a simple greedy/ Sinkhorn-free approach for small instances.
pub fn wasserstein_1(source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> f64 {
    assert_eq!(source.n(), target.n());
    let n = source.n();
    
    // For small n, solve via greedy transport
    let mut remaining_source = source.masses.clone();
    let mut remaining_target = target.masses.clone();
    let mut total_cost = 0.0;

    // Iteratively move mass from cheapest to most expensive
    for _ in 0..(2 * n) {
        // Find source with most remaining mass
        let (si, &sm) = remaining_source.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        // Find target with most remaining deficit
        let (ti, &tm) = remaining_target.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        if sm < 1e-15 || tm < 1e-15 { break; }
        
        let transported = sm.min(tm);
        total_cost += transported * cost.costs[si][ti];
        remaining_source[si] -= transported;
        remaining_target[ti] -= transported;
    }

    total_cost
}

/// Compute a transport plan between source and target distributions.
pub fn compute_transport_plan(source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> TransportPlan {
    let n = source.n();
    let mut plan = vec![vec![0.0; n]; n];
    let mut remaining_source = source.masses.clone();
    let mut remaining_target = target.masses.clone();
    let mut total_cost = 0.0;

    // Simple greedy: sort edges by cost, fill cheapest first
    let mut edges: Vec<(usize, usize, f64)> = vec![];
    for i in 0..n {
        for j in 0..n {
            edges.push((i, j, cost.costs[i][j]));
        }
    }
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (i, j, c) in edges {
        let amount = remaining_source[i].min(remaining_target[j]);
        if amount > 1e-15 {
            plan[i][j] = amount;
            total_cost += amount * c;
            remaining_source[i] -= amount;
            remaining_target[j] -= amount;
        }
    }

    TransportPlan { plan, cost: total_cost }
}

/// Barycenter of distributions (weighted average of masses).
pub fn barycenter(distributions: &[VibeDistribution], weights: &[f64]) -> VibeDistribution {
    assert!(!distributions.is_empty());
    let n = distributions[0].n();
    let mut masses = vec![0.0; n];
    let w_sum: f64 = weights.iter().sum();
    
    for (dist, w) in distributions.iter().zip(weights) {
        let nw = w / w_sum;
        for i in 0..n {
            masses[i] += nw * dist.masses[i];
        }
    }
    VibeDistribution::from_masses(masses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let d = VibeDistribution::uniform(4);
        assert_eq!(d.n(), 4);
        assert!(d.is_valid());
        for &m in &d.masses {
            assert!((m - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_masses_normalizes() {
        let d = VibeDistribution::from_masses(vec![1.0, 2.0, 3.0]);
        assert!(d.is_valid());
        assert!((d.masses[0] - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_same() {
        let d = VibeDistribution::uniform(3);
        let w = wasserstein_1d(&d, &d);
        assert!(w < 1e-10, "same distribution should have 0 distance");
    }

    #[test]
    fn test_wasserstein_1d_shifted() {
        let a = VibeDistribution::from_masses(vec![1.0, 0.0, 0.0]);
        let b = VibeDistribution::from_masses(vec![0.0, 0.0, 1.0]);
        let w = wasserstein_1d(&a, &b);
        assert!(w > 0.0, "different distributions should have positive distance");
    }

    #[test]
    fn test_wasserstein_1_with_cost() {
        let a = VibeDistribution::from_masses(vec![1.0, 0.0]);
        let b = VibeDistribution::from_masses(vec![0.0, 1.0]);
        let cost = CostMatrix::from_distances(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let w = wasserstein_1(&a, &b, &cost);
        assert!((w - 1.0).abs() < 1e-8, "transport cost should be 1.0");
    }

    #[test]
    fn test_transport_plan() {
        let a = VibeDistribution::from_masses(vec![1.0, 0.0]);
        let b = VibeDistribution::from_masses(vec![0.0, 1.0]);
        let cost = CostMatrix::from_distances(vec![vec![0.0, 2.0], vec![2.0, 0.0]]);
        let plan = compute_transport_plan(&a, &b, &cost);
        assert!((plan.cost - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_barycenter() {
        let a = VibeDistribution::from_masses(vec![1.0, 0.0]);
        let b = VibeDistribution::from_masses(vec![0.0, 1.0]);
        let bc = barycenter(&[a, b], &[1.0, 1.0]);
        assert!((bc.masses[0] - 0.5).abs() < 1e-8);
        assert!((bc.masses[1] - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_cost_matrix_from_positions_1d() {
        let cost = CostMatrix::from_positions_1d(&[0.0, 1.0, 3.0]);
        assert!((cost.costs[0][1] - 1.0).abs() < 1e-10);
        assert!((cost.costs[0][2] - 3.0).abs() < 1e-10);
        assert!((cost.costs[1][2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_matrix_from_positions_2d() {
        let cost = CostMatrix::from_positions_2d(&[(0.0, 0.0), (3.0, 4.0)]);
        assert!((cost.costs[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_triangle_inequality() {
        let a = VibeDistribution::from_masses(vec![1.0, 0.0, 0.0]);
        let b = VibeDistribution::from_masses(vec![0.0, 1.0, 0.0]);
        let c = VibeDistribution::from_masses(vec![0.0, 0.0, 1.0]);
        let w_ab = wasserstein_1d(&a, &b);
        let w_bc = wasserstein_1d(&b, &c);
        let w_ac = wasserstein_1d(&a, &c);
        assert!(w_ac <= w_ab + w_bc + 1e-10, "triangle inequality violated");
    }

    #[test]
    fn test_uniform_cost_matrix() {
        let cost = CostMatrix::uniform(3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((cost.costs[i][j] - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_is_valid_zero_masses() {
        let d = VibeDistribution::from_masses(vec![0.0, 0.0]);
        // Should handle gracefully - normalization produces uniform
        assert!(d.is_valid());
    }
}
