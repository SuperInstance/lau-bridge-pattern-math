//! Fleet-level thermodynamic laws.
//! Entropy always increases; Landauer bound on fleet updates.

use serde::{Serialize, Deserialize};

/// Thermodynamic state of a fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetThermoState {
    /// Number of agents in the fleet.
    pub n_agents: usize,
    /// Internal energy of each agent.
    pub energies: Vec<f64>,
    /// Entropy of each agent.
    pub entropies: Vec<f64>,
    /// Temperature of the system.
    pub temperature: f64,
    /// Total information bits stored.
    pub information_bits: f64,
}

impl FleetThermoState {
    /// Create a zero state.
    pub fn zero(n_agents: usize) -> Self {
        Self {
            n_agents,
            energies: vec![0.0; n_agents],
            entropies: vec![0.0; n_agents],
            temperature: 1.0,
            information_bits: 0.0,
        }
    }

    /// Total internal energy.
    pub fn total_energy(&self) -> f64 {
        self.energies.iter().sum()
    }

    /// Total entropy.
    pub fn total_entropy(&self) -> f64 {
        self.entropies.iter().sum()
    }

    /// Free energy: F = E - TS.
    pub fn free_energy(&self) -> f64 {
        self.total_energy() - self.temperature * self.total_entropy()
    }

    /// Thermodynamic beta: β = 1/(kT).
    pub fn beta(&self) -> f64 {
        if self.temperature > 0.0 {
            1.0 / self.temperature
        } else {
            f64::INFINITY
        }
    }

    /// Partition function: Z = sum_i exp(-β E_i).
    pub fn partition_function(&self) -> f64 {
        let beta = self.beta();
        self.energies.iter().map(|&e| (-beta * e).exp()).sum()
    }

    /// Boltzmann distribution over agents.
    pub fn boltzmann_probabilities(&self) -> Vec<f64> {
        let z = self.partition_function();
        let beta = self.beta();
        self.energies.iter().map(|&e| (-beta * e).exp() / z).collect()
    }

    /// Gibbs entropy: S = -sum p_i ln(p_i).
    pub fn gibbs_entropy(&self) -> f64 {
        let probs = self.boltzmann_probabilities();
        probs.iter().map(|&p| {
            if p > 1e-15 { -p * p.ln() } else { 0.0 }
        }).sum()
    }

    /// Heat capacity: C = d<E>/dT.
    pub fn heat_capacity(&self) -> f64 {
        let beta = self.beta();
        let probs = self.boltzmann_probabilities();
        let mean_e: f64 = probs.iter().zip(&self.energies).map(|(&p, &e)| p * e).sum();
        let mean_e2: f64 = probs.iter().zip(&self.energies).map(|(&p, &e)| p * e * e).sum();
        // C = β² * (<E²> - <E>²)
        beta * beta * (mean_e2 - mean_e * mean_e)
    }
}

/// Landauer's principle: erasing one bit of information requires kT ln(2) energy.
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
pub const LANDAUER_FACTOR: f64 = BOLTZMANN_CONSTANT * std::f64::consts::LN_2;

/// Compute the Landauer bound: minimum energy to erase `n_bits` at temperature T.
pub fn landauer_bound(n_bits: f64, temperature: f64) -> f64 {
    n_bits * BOLTZMANN_CONSTANT * temperature * std::f64::consts::LN_2
}

/// A thermodynamic process on the fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoProcess {
    pub name: String,
    pub delta_energy: f64,
    pub delta_entropy: f64,
    pub heat: f64,
    pub work: f64,
}

impl ThermoProcess {
    /// Create an adiabatic process (no heat exchange).
    pub fn adiabatic(work: f64) -> Self {
        Self {
            name: "adiabatic".into(),
            delta_energy: work,
            delta_entropy: 0.0,
            heat: 0.0,
            work,
        }
    }

    /// Create an isothermal process (constant temperature).
    pub fn isothermal(heat: f64) -> Self {
        Self {
            name: "isothermal".into(),
            delta_energy: 0.0,
            delta_entropy: heat, // ΔS = Q/T, assuming T=1
            heat,
            work: -heat,
        }
    }

    /// Check first law: ΔE = Q + W (physics convention: W done on system is positive).
    pub fn check_first_law(&self) -> bool {
        (self.delta_energy - (self.heat + self.work)).abs() < 1e-10
    }

    /// Check second law: ΔS ≥ 0 for isolated system.
    pub fn check_second_law(&self) -> bool {
        self.delta_entropy >= -1e-10
    }
}

/// Result of applying a process to the fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetEvolution {
    pub process: ThermoProcess,
    pub entropy_change: f64,
    pub energy_change: f64,
    pub irreversible: bool,
    pub landauer_cost: f64,
}

/// Evolve the fleet thermodynamic state.
pub fn evolve_fleet(state: &mut FleetThermoState, process: &ThermoProcess) -> FleetEvolution {
    let old_entropy = state.total_entropy();
    let old_energy = state.total_energy();

    // Distribute energy change equally among agents
    let de_per_agent = process.delta_energy / state.n_agents as f64;
    for e in &mut state.energies {
        *e += de_per_agent;
    }

    // Distribute entropy change
    let ds_per_agent = process.delta_entropy / state.n_agents as f64;
    for s in &mut state.entropies {
        *s += ds_per_agent;
    }

    let new_entropy = state.total_entropy();
    let new_energy = state.total_energy();

    // Landauer cost for information processing
    let landauer = landauer_bound(state.information_bits, state.temperature);

    FleetEvolution {
        process: process.clone(),
        entropy_change: new_entropy - old_entropy,
        energy_change: new_energy - old_energy,
        irreversible: (new_entropy - old_entropy) > 1e-10,
        landauer_cost: landauer,
    }
}

/// Compute the thermodynamic efficiency of fleet updates.
pub fn thermodynamic_efficiency(useful_work: f64, total_energy: f64) -> f64 {
    if total_energy.abs() < 1e-15 {
        return 0.0;
    }
    (useful_work / total_energy).min(1.0)
}

/// Carnot efficiency: η = 1 - T_cold / T_hot.
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> f64 {
    if t_hot <= t_cold || t_hot <= 0.0 {
        return 0.0;
    }
    1.0 - t_cold / t_hot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let s = FleetThermoState::zero(5);
        assert_eq!(s.total_energy(), 0.0);
        assert_eq!(s.total_entropy(), 0.0);
    }

    #[test]
    fn test_total_energy() {
        let mut s = FleetThermoState::zero(3);
        s.energies = vec![1.0, 2.0, 3.0];
        assert!((s.total_energy() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_energy() {
        let mut s = FleetThermoState::zero(2);
        s.energies = vec![10.0, 10.0];
        s.entropies = vec![1.0, 1.0];
        s.temperature = 2.0;
        // F = 20 - 2 * 2 = 16
        assert!((s.free_energy() - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_function() {
        let mut s = FleetThermoState::zero(2);
        s.energies = vec![0.0, 0.0];
        s.temperature = 1.0;
        let z = s.partition_function();
        assert!((z - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_boltzmann_probabilities() {
        let mut s = FleetThermoState::zero(2);
        s.energies = vec![0.0, 0.0];
        s.temperature = 1.0;
        let probs = s.boltzmann_probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-8);
        assert!((probs[1] - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_gibbs_entropy() {
        let mut s = FleetThermoState::zero(2);
        s.energies = vec![0.0, 0.0];
        s.temperature = 1.0;
        let s_gibbs = s.gibbs_entropy();
        // Max entropy for 2 states: ln(2)
        assert!((s_gibbs - std::f64::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn test_landauer_bound() {
        let energy = landauer_bound(1.0, 300.0);
        assert!(energy > 0.0);
        assert!((energy - BOLTZMANN_CONSTANT * 300.0 * std::f64::consts::LN_2).abs() < 1e-30);
    }

    #[test]
    fn test_adiabatic_process() {
        let p = ThermoProcess::adiabatic(10.0);
        assert_eq!(p.heat, 0.0);
        assert!((p.delta_entropy).abs() < 1e-10);
        assert!(p.check_first_law());
    }

    #[test]
    fn test_isothermal_process() {
        let p = ThermoProcess::isothermal(5.0);
        assert!((p.delta_energy).abs() < 1e-10);
        assert!(p.check_second_law());
    }

    #[test]
    fn test_evolve_fleet() {
        let mut state = FleetThermoState::zero(3);
        state.information_bits = 100.0;
        let process = ThermoProcess::adiabatic(3.0);
        let result = evolve_fleet(&mut state, &process);
        assert!((result.energy_change - 3.0).abs() < 1e-10);
        assert!(result.landauer_cost > 0.0);
    }

    #[test]
    fn test_thermodynamic_efficiency() {
        let eta = thermodynamic_efficiency(5.0, 10.0);
        assert!((eta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_efficiency() {
        let eta = carnot_efficiency(600.0, 300.0);
        assert!((eta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_invalid() {
        let eta = carnot_efficiency(300.0, 600.0);
        assert_eq!(eta, 0.0);
    }

    #[test]
    fn test_heat_capacity() {
        let mut s = FleetThermoState::zero(3);
        s.energies = vec![0.0, 1.0, 2.0];
        s.temperature = 1.0;
        let c = s.heat_capacity();
        assert!(c >= 0.0);
    }

    #[test]
    fn test_first_law_consistent() {
        let p = ThermoProcess {
            name: "test".into(),
            delta_energy: 5.0,
            delta_entropy: 1.0,
            heat: 8.0,
            work: -3.0,
        };
        assert!(p.check_first_law()); // ΔE = Q + W = 8 + (-3) = 5 ✓
    }
}
