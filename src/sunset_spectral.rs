//! Spectral decomposition of the sunset lifecycle.
//! Ethos, pathos, logos as eigenvectors of the sunset operator.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// The three sunset modes: ethos, pathos, logos.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SunsetMode {
    Ethos,   // Character/credibility
    Pathos,  // Emotion/feeling
    Logos,   // Logic/structure
}

impl SunsetMode {
    /// All modes.
    pub fn all() -> &'static [SunsetMode] {
        &[SunsetMode::Ethos, SunsetMode::Pathos, SunsetMode::Logos]
    }

    /// Name of the mode.
    pub fn name(&self) -> &'static str {
        match self {
            SunsetMode::Ethos => "ethos",
            SunsetMode::Pathos => "pathos",
            SunsetMode::Logos => "logos",
        }
    }
}

/// A sunset state: weighted combination of the three modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunsetState {
    pub ethos: f64,
    pub pathos: f64,
    pub logos: f64,
}

impl SunsetState {
    /// Pure mode.
    pub fn pure(mode: SunsetMode) -> Self {
        match mode {
            SunsetMode::Ethos => Self { ethos: 1.0, pathos: 0.0, logos: 0.0 },
            SunsetMode::Pathos => Self { ethos: 0.0, pathos: 1.0, logos: 0.0 },
            SunsetMode::Logos => Self { ethos: 0.0, pathos: 0.0, logos: 1.0 },
        }
    }

    /// Balanced state (equal weights).
    pub fn balanced() -> Self {
        Self { ethos: 1.0 / 3.0, pathos: 1.0 / 3.0, logos: 1.0 / 3.0 }
    }

    /// Convert to vector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.ethos, self.pathos, self.logos])
    }

    /// Create from vector.
    pub fn from_vector(v: &DVector<f64>) -> Self {
        Self { ethos: v[0], pathos: v[1], logos: v[2] }
    }

    /// Norm of the state.
    pub fn norm(&self) -> f64 {
        (self.ethos.powi(2) + self.pathos.powi(2) + self.logos.powi(2)).sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 { return self.clone(); }
        Self {
            ethos: self.ethos / n,
            pathos: self.pathos / n,
            logos: self.logos / n,
        }
    }

    /// Dominant mode.
    pub fn dominant_mode(&self) -> SunsetMode {
        if self.ethos >= self.pathos && self.ethos >= self.logos {
            SunsetMode::Ethos
        } else if self.pathos >= self.logos {
            SunsetMode::Pathos
        } else {
            SunsetMode::Logos
        }
    }
}

/// The sunset operator: a 3×3 matrix that governs mode evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunsetOperator {
    pub matrix: [[f64; 3]; 3],
}

impl SunsetOperator {
    /// Identity operator (modes don't change).
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Symmetric mixing operator (all modes equally coupled).
    pub fn symmetric_mixing(coupling: f64) -> Self {
        Self {
            matrix: [
                [1.0 - 2.0 * coupling, coupling, coupling],
                [coupling, 1.0 - 2.0 * coupling, coupling],
                [coupling, coupling, 1.0 - 2.0 * coupling],
            ],
        }
    }

    /// Create from a nalgebra matrix.
    pub fn from_matrix(m: &DMatrix<f64>) -> Self {
        let mut matrix = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = m[(i, j)];
            }
        }
        Self { matrix }
    }

    /// Apply to a sunset state.
    pub fn apply(&self, state: &SunsetState) -> SunsetState {
        let v = state.to_vector();
        let m = self.to_matrix();
        let result = &m * v;
        SunsetState::from_vector(&result)
    }

    /// Convert to nalgebra matrix.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 3, &[
            self.matrix[0][0], self.matrix[0][1], self.matrix[0][2],
            self.matrix[1][0], self.matrix[1][1], self.matrix[1][2],
            self.matrix[2][0], self.matrix[2][1], self.matrix[2][2],
        ])
    }

    /// Compute eigenvalues (using characteristic polynomial for 3×3).
    pub fn eigenvalues(&self) -> [f64; 3] {
        let m = self.to_matrix();
        let _trace = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        
        // For symmetric matrices, use Jacobi iteration
        let mut a = m.clone();
        for _ in 0..200 {
            let mut max_val = 0.0_f64;
            let (mut p, mut q) = (0, 1);
            for i in 0..3 {
                for j in (i + 1)..3 {
                    if a[(i, j)].abs() > max_val {
                        max_val = a[(i, j)].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-12 { break; }

            let app = a[(p, p)];
            let aqq = a[(q, q)];
            let apq = a[(p, q)];
            let theta = 0.5 * (2.0 * apq / (app - aqq)).atan();
            let c = theta.cos();
            let s = theta.sin();

            let mut new_a = a.clone();
            for i in 0..3 {
                if i != p && i != q {
                    let aip = a[(i, p)];
                    let aiq = a[(i, q)];
                    new_a[(i, p)] = c * aip + s * aiq;
                    new_a[(p, i)] = new_a[(i, p)];
                    new_a[(i, q)] = -s * aip + c * aiq;
                    new_a[(q, i)] = new_a[(i, q)];
                }
            }
            new_a[(p, p)] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            new_a[(q, q)] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            new_a[(p, q)] = 0.0;
            new_a[(q, p)] = 0.0;
            a = new_a;
        }

        let mut evals = [a[(0, 0)], a[(1, 1)], a[(2, 2)]];
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        evals
    }

    /// Power iteration to find dominant eigenvector.
    pub fn dominant_eigenvector(&self, n_iters: usize) -> SunsetState {
        let m = self.to_matrix();
        let mut v = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        
        for _ in 0..n_iters {
            v = &m * &v;
            let norm = v.norm();
            if norm > 1e-15 {
                v = v / norm;
            }
        }
        
        SunsetState::from_vector(&v)
    }

    /// Compose two operators.
    pub fn compose(&self, other: &SunsetOperator) -> SunsetOperator {
        let a = self.to_matrix();
        let b = other.to_matrix();
        SunsetOperator::from_matrix(&(a * b))
    }

    /// Power of the operator: apply n times.
    pub fn power(&self, n: usize) -> SunsetOperator {
        let m = self.to_matrix();
        let mut result = DMatrix::identity(3, 3);
        let mut base = m.clone();
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result * &base;
            }
            base = &base * &base;
            exp /= 2;
        }
        SunsetOperator::from_matrix(&result)
    }
}

/// Spectral decomposition result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    pub eigenvalues: [f64; 3],
    pub dominant_eigenvector: SunsetState,
    pub spectral_gap: f64,
    pub dominant_mode: SunsetMode,
    pub entropy: f64,
}

impl SunsetOperator {
    /// Full spectral decomposition.
    pub fn decompose(&self) -> SpectralDecomposition {
        let evals = self.eigenvalues();
        let dom_ev = self.dominant_eigenvector(100);
        let gap = if evals.len() >= 2 {
            (evals[0] - evals[1]).abs()
        } else {
            0.0
        };
        
        // Spectral entropy: -sum λ_i log λ_i (normalized eigenvalues)
        let total: f64 = evals.iter().map(|x| x.abs()).sum();
        let entropy = if total > 1e-15 {
            let probs: Vec<f64> = evals.iter().map(|x| x.abs() / total).collect();
            -probs.iter().map(|&p| if p > 1e-15 { p * p.ln() } else { 0.0 }).sum::<f64>()
        } else {
            0.0
        };

        SpectralDecomposition {
            eigenvalues: evals,
            dominant_eigenvector: dom_ev.normalize(),
            spectral_gap: gap,
            dominant_mode: dom_ev.dominant_mode(),
            entropy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_states() {
        let e = SunsetState::pure(SunsetMode::Ethos);
        assert_eq!(e.ethos, 1.0);
        assert_eq!(e.pathos, 0.0);
        assert_eq!(e.logos, 0.0);
    }

    #[test]
    fn test_balanced_state() {
        let b = SunsetState::balanced();
        assert!((b.ethos - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let s = SunsetState { ethos: 3.0, pathos: 4.0, logos: 0.0 };
        let n = s.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dominant_mode() {
        let s = SunsetState { ethos: 0.1, pathos: 0.8, logos: 0.1 };
        assert_eq!(s.dominant_mode(), SunsetMode::Pathos);
    }

    #[test]
    fn test_identity_operator() {
        let op = SunsetOperator::identity();
        let s = SunsetState::pure(SunsetMode::Ethos);
        let result = op.apply(&s);
        assert!((result.ethos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_mixing() {
        let op = SunsetOperator::symmetric_mixing(0.1);
        let evals = op.eigenvalues();
        // Should have one eigenvalue = 1.0 (uniform eigenvector)
        assert!(evals.iter().any(|&e| (e - 1.0).abs() < 1e-6),
            "symmetric mixing should have eigenvalue 1");
    }

    #[test]
    fn test_eigenvalues_identity() {
        let op = SunsetOperator::identity();
        let evals = op.eigenvalues();
        for &e in &evals {
            assert!((e - 1.0).abs() < 1e-6, "identity eigenvalues should be 1.0");
        }
    }

    #[test]
    fn test_power_iteration() {
        let op = SunsetOperator::symmetric_mixing(0.2);
        let dom = op.dominant_eigenvector(1000);
        assert!(dom.norm() > 0.0);
    }

    #[test]
    fn test_operator_compose() {
        let a = SunsetOperator::identity();
        let b = SunsetOperator::identity();
        let c = a.compose(&b);
        let s = SunsetState::pure(SunsetMode::Pathos);
        let result = c.apply(&s);
        assert!((result.pathos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_operator_power() {
        let op = SunsetOperator::identity();
        let op2 = op.power(5);
        let s = SunsetState::pure(SunsetMode::Logos);
        let result = op2.apply(&s);
        assert!((result.logos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_decomposition() {
        let op = SunsetOperator::symmetric_mixing(0.15);
        let decomp = op.decompose();
        assert_eq!(decomp.eigenvalues.len(), 3);
        assert!(decomp.spectral_gap >= 0.0);
    }

    #[test]
    fn test_spectral_entropy() {
        let op = SunsetOperator::identity();
        let decomp = op.decompose();
        // Identity has all eigenvalues equal → max entropy
        assert!(decomp.entropy > 0.0);
    }

    #[test]
    fn test_mode_names() {
        assert_eq!(SunsetMode::Ethos.name(), "ethos");
        assert_eq!(SunsetMode::Pathos.name(), "pathos");
        assert_eq!(SunsetMode::Logos.name(), "logos");
    }
}
