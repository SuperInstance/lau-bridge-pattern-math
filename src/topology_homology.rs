//! Homology groups of the Grand Pattern topology.
//! Cycles = redundant paths, boundaries = trivial.

use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

/// A simplicial complex representing the Grand Pattern topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternComplex {
    /// Vertices.
    pub vertices: Vec<String>,
    /// Edges: pairs of vertex indices.
    pub edges: Vec<(usize, usize)>,
    /// Triangles: triples of vertex indices.
    pub triangles: Vec<(usize, usize, usize)>,
}

impl PatternComplex {
    /// Create an empty complex.
    pub fn new() -> Self {
        Self {
            vertices: vec![],
            edges: vec![],
            triangles: vec![],
        }
    }

    /// Add a vertex.
    pub fn add_vertex(&mut self, name: &str) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(name.to_string());
        idx
    }

    /// Add an edge.
    pub fn add_edge(&mut self, i: usize, j: usize) {
        assert!(i < self.vertices.len() && j < self.vertices.len());
        self.edges.push((i.min(j), i.max(j)));
    }

    /// Add a triangle.
    pub fn add_triangle(&mut self, i: usize, j: usize, k: usize) {
        assert!(i < self.vertices.len() && j < self.vertices.len() && k < self.vertices.len());
        let mut t = [i, j, k];
        t.sort();
        self.triangles.push((t[0], t[1], t[2]));
    }

    /// Compute the boundary matrix ∂₁: edges → vertices.
    /// ∂₁(e) = v_j - v_i for edge (i,j).
    pub fn boundary_1(&self) -> DMatrix<f64> {
        let n_v = self.vertices.len();
        let n_e = self.edges.len();
        let mut mat = vec![0.0; n_v * n_e];
        for (col, (i, j)) in self.edges.iter().enumerate() {
            mat[i + col * n_v] = -1.0;
            mat[j + col * n_v] = 1.0;
        }
        DMatrix::from_vec(n_v, n_e, mat)
    }

    /// Compute the boundary matrix ∂₂: triangles → edges.
    /// ∂₂(Δ) = e_jk - e_ik + e_ij for triangle (i,j,k).
    pub fn boundary_2(&self) -> DMatrix<f64> {
        let n_e = self.edges.len();
        let n_t = self.triangles.len();
        let mut mat = vec![0.0; n_e * n_t];

        // Create edge index lookup
        let mut edge_idx = std::collections::HashMap::new();
        for (idx, &(i, j)) in self.edges.iter().enumerate() {
            edge_idx.insert((i, j), idx);
        }

        for (col, &(i, j, k)) in self.triangles.iter().enumerate() {
            // ∂(ijk) = (jk) - (ik) + (ij)
            if let Some(&idx) = edge_idx.get(&(i.min(j), i.max(j))) {
                mat[idx + col * n_e] += 1.0;
            }
            if let Some(&idx) = edge_idx.get(&(i.min(k), i.max(k))) {
                mat[idx + col * n_e] -= 1.0;
            }
            if let Some(&idx) = edge_idx.get(&(j.min(k), j.max(k))) {
                mat[idx + col * n_e] += 1.0;
            }
        }
        DMatrix::from_vec(n_e, n_t, mat)
    }

    /// Compute Betti numbers: β_k = dim(H_k) = dim(ker ∂_k) - dim(im ∂_{k+1}).
    /// Returns (β₀, β₁, β₂).
    pub fn betti_numbers(&self) -> (usize, usize, usize) {
        let d1 = self.boundary_1();
        
        // β₀ = dim(ker ∂₁) ... but actually β₀ = # connected components
        // = n_vertices - rank(∂₁)
        let rank_d1 = self._rank(&d1);
        let beta0 = self.vertices.len().saturating_sub(rank_d1);

        let d2 = self.boundary_2();
        let rank_d2 = self._rank(&d2);
        
        // β₁ = dim(ker ∂₁) - dim(im ∂₂) = (n_edges - rank_d1) - rank_d2
        let null_d1 = self.edges.len().saturating_sub(rank_d1);
        let beta1 = null_d1.saturating_sub(rank_d2);

        // β₂ = dim(ker ∂₂) = n_triangles - rank_d2
        let beta2 = self.triangles.len().saturating_sub(rank_d2);

        (beta0, beta1, beta2)
    }

    /// Compute rank of a matrix via row reduction.
    fn _rank(&self, mat: &DMatrix<f64>) -> usize {
        let mut m = mat.clone();
        let nrows = m.nrows();
        let ncols = m.ncols();
        let mut rank = 0;
        let mut row = 0;

        for col in 0..ncols {
            if row >= nrows { break; }
            // Find pivot
            let mut pivot_row = None;
            for r in row..nrows {
                if m[(r, col)].abs() > 1e-10 {
                    pivot_row = Some(r);
                    break;
                }
            }
            if let Some(pr) = pivot_row {
                // Swap rows
                if pr != row {
                    for c in 0..ncols {
                        let tmp = m[(row, c)];
                        m[(row, c)] = m[(pr, c)];
                        m[(pr, c)] = tmp;
                    }
                }
                // Eliminate below
                for r in (row + 1)..nrows {
                    let factor = m[(r, col)] / m[(row, col)];
                    for c in col..ncols {
                        m[(r, c)] -= factor * m[(row, c)];
                    }
                }
                rank += 1;
                row += 1;
            }
        }
        rank
    }

    /// Euler characteristic: χ = β₀ - β₁ + β₂.
    pub fn euler_characteristic(&self) -> i64 {
        let (b0, b1, b2) = self.betti_numbers();
        b0 as i64 - b1 as i64 + b2 as i64
    }

    /// Check if a cycle is a boundary (trivial in homology).
    pub fn is_boundary_cycle(&self, edge_indices: &[usize]) -> bool {
        // A cycle is a boundary if it's the image of some triangle chain
        // Simplified: check if all edges belong to some triangle
        if self.triangles.is_empty() {
            return false;
        }
        
        // Build edge set for each triangle
        for &(i, j, k) in &self.triangles {
            let tri_edges: std::collections::HashSet<usize> = {
                let mut set = std::collections::HashSet::new();
                for (idx, &(a, b)) in self.edges.iter().enumerate() {
                    if (a == i && b == j) || (a == i && b == k) || (a == j && b == k) {
                        set.insert(idx);
                    }
                }
                set
            };
            if edge_indices.iter().all(|e| tri_edges.contains(e)) {
                return true;
            }
        }
        false
    }
}

/// Homology computation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomologyResult {
    pub betti: (usize, usize, usize),
    pub euler_characteristic: i64,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_triangles: usize,
}

impl PatternComplex {
    /// Full homology analysis.
    pub fn homology(&self) -> HomologyResult {
        let betti = self.betti_numbers();
        HomologyResult {
            betti,
            euler_characteristic: self.euler_characteristic(),
            n_vertices: self.vertices.len(),
            n_edges: self.edges.len(),
            n_triangles: self.triangles.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_complex() {
        let c = PatternComplex::new();
        let (b0, b1, _b2) = c.betti_numbers();
        assert_eq!(b0, 0);
        assert_eq!(b1, 0);
    }

    #[test]
    fn test_single_vertex() {
        let mut c = PatternComplex::new();
        c.add_vertex("v0");
        let (b0, b1, _) = c.betti_numbers();
        assert_eq!(b0, 1, "single vertex: β₀=1");
        assert_eq!(b1, 0, "single vertex: β₁=0");
    }

    #[test]
    fn test_two_vertices_edge() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        c.add_edge(v0, v1);
        let (b0, b1, _) = c.betti_numbers();
        assert_eq!(b0, 1, "two vertices connected: β₀=1");
        assert_eq!(b1, 0, "two vertices: β₁=0");
    }

    #[test]
    fn test_triangle_no_fill() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        c.add_edge(v1, v2);
        c.add_edge(v0, v2);
        let (b0, b1, _) = c.betti_numbers();
        assert_eq!(b0, 1);
        assert_eq!(b1, 1, "triangle without fill: β₁=1 (one cycle)");
    }

    #[test]
    fn test_triangle_filled() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        c.add_edge(v1, v2);
        c.add_edge(v0, v2);
        c.add_triangle(v0, v1, v2);
        let (b0, b1, _) = c.betti_numbers();
        assert_eq!(b0, 1);
        assert_eq!(b1, 0, "filled triangle: β₁=0 (cycle is a boundary)");
    }

    #[test]
    fn test_disconnected_components() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        // v2 is isolated
        let (b0, _, _) = c.betti_numbers();
        assert_eq!(b0, 2, "disconnected: β₀=2");
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        c.add_edge(v1, v2);
        c.add_edge(v0, v2);
        c.add_triangle(v0, v1, v2);
        let chi = c.euler_characteristic();
        // χ = V - E + F = 3 - 3 + 1 = 1
        assert_eq!(chi, 1);
    }

    #[test]
    fn test_euler_characteristic_unfilled() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        c.add_edge(v1, v2);
        c.add_edge(v0, v2);
        // χ = β₀ - β₁ = 1 - 1 = 0
        let chi = c.euler_characteristic();
        assert_eq!(chi, 0);
    }

    #[test]
    fn test_boundary_matrices() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        c.add_edge(v0, v1);
        let d1 = c.boundary_1();
        assert_eq!(d1.nrows(), 2);
        assert_eq!(d1.ncols(), 1);
    }

    #[test]
    fn test_homology_result() {
        let mut c = PatternComplex::new();
        c.add_vertex("v0");
        let result = c.homology();
        assert_eq!(result.n_vertices, 1);
        assert_eq!(result.betti.0, 1);
    }

    #[test]
    fn test_complex_cycle_detection() {
        let mut c = PatternComplex::new();
        let v0 = c.add_vertex("v0");
        let v1 = c.add_vertex("v1");
        let v2 = c.add_vertex("v2");
        c.add_edge(v0, v1);
        c.add_edge(v1, v2);
        c.add_edge(v0, v2);
        // The cycle {0,1,2} (all edges) is not a boundary without the triangle
        assert!(!c.is_boundary_cycle(&[0, 1, 2]));
    }
}
