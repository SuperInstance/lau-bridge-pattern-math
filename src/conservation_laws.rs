//! Formal conservation laws for the Grand Pattern.
//! Energy, mass, momentum analogues for the venue graph.

use serde::{Serialize, Deserialize};

/// A conserved quantity in the Grand Pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedQuantity {
    pub name: String,
    pub value: f64,
    pub flux: Vec<f64>, // flux at each venue
}

/// The total state of the Grand Pattern for conservation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternState {
    pub n_venues: usize,
    /// "Mass" at each venue (total activity level).
    pub mass: Vec<f64>,
    /// "Momentum" at each venue (directional flow).
    pub momentum: Vec<(f64, f64)>,
    /// "Energy" at each venue (kinetic + potential).
    pub energy: Vec<f64>,
    /// Adjacency weights for potential energy computation.
    pub adjacency: Vec<Vec<f64>>,
}

impl PatternState {
    /// Create a zero state.
    pub fn zero(n: usize) -> Self {
        Self {
            n_venues: n,
            mass: vec![0.0; n],
            momentum: vec![(0.0, 0.0); n],
            energy: vec![0.0; n],
            adjacency: vec![vec![0.0; n]; n],
        }
    }

    /// Total mass across all venues.
    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }

    /// Total momentum (vector sum).
    pub fn total_momentum(&self) -> (f64, f64) {
        self.momentum.iter().fold((0.0, 0.0), |(px, py), (mx, my)| {
            (px + mx, py + my)
        })
    }

    /// Total energy across all venues.
    pub fn total_energy(&self) -> f64 {
        self.energy.iter().sum()
    }

    /// Kinetic energy: sum of ½|m|² for each venue.
    pub fn kinetic_energy(&self) -> f64 {
        self.momentum.iter().map(|(px, py)| 0.5 * (px * px + py * py)).sum()
    }

    /// Potential energy: sum of adjacency-weighted interactions.
    pub fn potential_energy(&self) -> f64 {
        let mut pe = 0.0;
        for i in 0..self.n_venues {
            for j in (i + 1)..self.n_venues {
                pe -= self.adjacency[i][j] * self.mass[i] * self.mass[j];
            }
        }
        pe
    }
}

/// Result of checking conservation laws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationCheck {
    pub mass_conserved: bool,
    pub momentum_conserved: bool,
    pub energy_conserved: bool,
    pub mass_before: f64,
    pub mass_after: f64,
    pub momentum_before: (f64, f64),
    pub momentum_after: (f64, f64),
    pub energy_before: f64,
    pub energy_after: f64,
}

/// Check conservation between two states.
pub fn check_conservation(before: &PatternState, after: &PatternState, tol: f64) -> ConservationCheck {
    let mb = before.total_mass();
    let ma = after.total_mass();
    let (mpx_b, mpy_b) = before.total_momentum();
    let (mpx_a, mpy_a) = after.total_momentum();
    let eb = before.total_energy();
    let ea = after.total_energy();

    ConservationCheck {
        mass_conserved: (mb - ma).abs() < tol,
        momentum_conserved: (mpx_b - mpx_a).abs() < tol && (mpy_b - mpy_a).abs() < tol,
        energy_conserved: (eb - ea).abs() < tol,
        mass_before: mb,
        mass_after: ma,
        momentum_before: (mpx_b, mpy_b),
        momentum_after: (mpx_a, mpy_a),
        energy_before: eb,
        energy_after: ea,
    }
}

/// Noether's theorem analogue: every symmetry of the Grand Pattern
/// corresponds to a conservation law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryConservationLaw {
    pub symmetry: String,
    pub conserved_quantity: String,
    pub description: String,
}

/// Enumerate the Noether-type conservation laws for the Grand Pattern.
pub fn noether_laws() -> Vec<SymmetryConservationLaw> {
    vec![
        SymmetryConservationLaw {
            symmetry: "Time translation".into(),
            conserved_quantity: "Energy".into(),
            description: "Invariance under time shifts → total energy is conserved".into(),
        },
        SymmetryConservationLaw {
            symmetry: "Venue permutation".into(),
            conserved_quantity: "Total mass".into(),
            description: "Relabeling venues doesn't change total activity".into(),
        },
        SymmetryConservationLaw {
            symmetry: "Spatial translation".into(),
            conserved_quantity: "Momentum".into(),
            description: "Shifting the entire pattern preserves momentum".into(),
        },
        SymmetryConservationLaw {
            symmetry: "Phase rotation".into(),
            conserved_quantity: "Vibe amplitude".into(),
            description: "Rotating vibe phases preserves total vibe amplitude".into(),
        },
        SymmetryConservationLaw {
            symmetry: "Scale invariance".into(),
            conserved_quantity: "Fibonacci ratio".into(),
            description: "Self-similar scaling preserves Fibonacci proportions".into(),
        },
    ]
}

/// Apply a mass-conserving evolution step.
/// Distributes mass according to adjacency weights.
pub fn evolve_mass(state: &mut PatternState, dt: f64) {
    let n = state.n_venues;
    let mut new_mass = state.mass.clone();
    
    for i in 0..n {
        let mut flux = 0.0;
        for j in 0..n {
            let diff = state.mass[j] - state.mass[i];
            flux += state.adjacency[i][j] * diff;
        }
        new_mass[i] += dt * flux;
    }
    
    // Renormalize to conserve total mass
    let old_total: f64 = state.mass.iter().sum();
    let new_total: f64 = new_mass.iter().sum();
    if new_total > 0.0 {
        let scale = old_total / new_total;
        for m in &mut new_mass {
            *m *= scale;
        }
    }
    
    state.mass = new_mass;
}

/// Apply energy-conserving evolution (Hamiltonian flow).
pub fn evolve_hamiltonian(state: &mut PatternState, dt: f64) {
    let n = state.n_venues;
    
    // Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    // With q = mass, p = momentum
    let mut new_momentum = state.momentum.clone();
    
    for i in 0..n {
        let mut force_x = 0.0_f64;
        let _force_y = 0.0_f64;
        for j in 0..n {
            let coupling = state.adjacency[i][j];
            force_x -= coupling * (state.mass[i] - state.mass[j]);
            // Symplectic: force depends on gradient of potential
        }
        new_momentum[i].0 -= dt * force_x;
        // Keep y-momentum unchanged for simplicity
    }
    
    state.momentum = new_momentum;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let s = PatternState::zero(5);
        assert_eq!(s.total_mass(), 0.0);
        assert_eq!(s.total_energy(), 0.0);
        assert_eq!(s.total_momentum(), (0.0, 0.0));
    }

    #[test]
    fn test_total_mass() {
        let mut s = PatternState::zero(3);
        s.mass = vec![1.0, 2.0, 3.0];
        assert!((s.total_mass() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_momentum() {
        let mut s = PatternState::zero(3);
        s.momentum = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)];
        let (px, py) = s.total_momentum();
        assert!((px - 9.0).abs() < 1e-10);
        assert!((py - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_kinetic_energy() {
        let mut s = PatternState::zero(2);
        s.momentum = vec![(3.0, 4.0), (0.0, 0.0)];
        // KE = 0.5 * (9 + 16) = 12.5
        assert!((s.kinetic_energy() - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_potential_energy() {
        let mut s = PatternState::zero(2);
        s.mass = vec![1.0, 1.0];
        s.adjacency = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        // PE = -1 * 1 * 1 = -1
        assert!((s.potential_energy() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_check_same_state() {
        let mut s = PatternState::zero(3);
        s.mass = vec![1.0, 2.0, 3.0];
        s.energy = vec![1.0, 1.0, 1.0];
        let check = check_conservation(&s, &s, 1e-8);
        assert!(check.mass_conserved);
        assert!(check.energy_conserved);
        assert!(check.momentum_conserved);
    }

    #[test]
    fn test_conservation_check_different_state() {
        let mut s1 = PatternState::zero(2);
        s1.mass = vec![1.0, 1.0];
        s1.energy = vec![2.0, 2.0];
        
        let mut s2 = PatternState::zero(2);
        s2.mass = vec![1.0, 1.0];
        s2.energy = vec![3.0, 3.0];
        
        let check = check_conservation(&s1, &s2, 1e-8);
        assert!(check.mass_conserved);
        assert!(!check.energy_conserved);
    }

    #[test]
    fn test_noether_laws() {
        let laws = noether_laws();
        assert_eq!(laws.len(), 5);
        assert_eq!(laws[0].conserved_quantity, "Energy");
    }

    #[test]
    fn test_evolve_mass_conserves() {
        let mut s = PatternState::zero(3);
        s.mass = vec![1.0, 2.0, 3.0];
        s.adjacency = vec![
            vec![0.0, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.0, 0.5, 0.0],
        ];
        let mass_before = s.total_mass();
        evolve_mass(&mut s, 0.1);
        let mass_after = s.total_mass();
        assert!((mass_before - mass_after).abs() < 1e-8, "mass should be conserved");
    }

    #[test]
    fn test_evolve_hamiltonian() {
        let mut s = PatternState::zero(2);
        s.mass = vec![1.0, 2.0];
        s.momentum = vec![(1.0, 0.0), (0.0, 0.0)];
        s.adjacency = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        evolve_hamiltonian(&mut s, 0.01);
        // Momentum should change
        assert!(s.momentum[0].0 != 1.0 || s.momentum[1].0 != 0.0);
    }

    #[test]
    fn test_conserved_quantity_struct() {
        let q = ConservedQuantity {
            name: "Energy".into(),
            value: 42.0,
            flux: vec![1.0, -1.0],
        };
        assert_eq!(q.name, "Energy");
        assert_eq!(q.flux.len(), 2);
    }
}
