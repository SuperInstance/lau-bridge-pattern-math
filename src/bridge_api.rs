//! Unified bridge API: accepts Grand Pattern events and returns math crate analysis.

use serde::{Serialize, Deserialize};

use crate::graph_spectral::SpectralAnalysis;
use crate::venue_sheaf::{VenueSheaf, SheafCohomology};
use crate::vibe_transport::{VibeDistribution, CostMatrix, TransportPlan, wasserstein_1, compute_transport_plan};
use crate::conservation_laws::{PatternState, ConservationCheck, check_conservation, noether_laws, SymmetryConservationLaw};
use crate::jepa_geometry::{JepaEmbedding, JepaManifold};
use crate::fibonacci_categorification::FibonacciSequence;
use crate::topology_homology::{PatternComplex, HomologyResult};
use crate::sunset_spectral::{SunsetOperator, SpectralDecomposition};
use crate::fleet_thermodynamics::{FleetThermoState, landauer_bound, evolve_fleet, ThermoProcess};

/// A Grand Pattern event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEvent {
    pub event_type: String,
    pub venue_id: Option<usize>,
    pub data: serde_json::Value,
    pub timestamp: f64,
}

/// The unified analysis result from all math modules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub spectral: Option<SpectralAnalysis>,
    pub cohomology: Option<SheafCohomology>,
    pub transport_cost: Option<f64>,
    pub conservation: Option<ConservationCheck>,
    pub homology: Option<HomologyResult>,
    pub sunset_decomp: Option<SpectralDecomposition>,
    pub fleet_entropy: Option<f64>,
    pub fibonacci_ratio: Option<f64>,
    pub landauer_cost: Option<f64>,
    pub warnings: Vec<String>,
}

/// The main bridge API.
pub struct BridgeApi {
    pub graph: crate::graph_spectral::CellularGraph,
    pub sheaf: VenueSheaf,
    pub pattern_state: PatternState,
    pub complex: PatternComplex,
    pub sunset_op: SunsetOperator,
    pub fleet_state: FleetThermoState,
    fib: FibonacciSequence,
}

impl BridgeApi {
    /// Create a new bridge API with the given number of venues.
    pub fn new(n_venues: usize) -> Self {
        Self {
            graph: crate::graph_spectral::CellularGraph::new(n_venues),
            sheaf: VenueSheaf::new(n_venues, 3),
            pattern_state: PatternState::zero(n_venues),
            complex: PatternComplex::new(),
            sunset_op: SunsetOperator::identity(),
            fleet_state: FleetThermoState::zero(n_venues),
            fib: FibonacciSequence::new(20),
        }
    }

    /// Process a Grand Pattern event and return analysis.
    pub fn process_event(&mut self, event: &PatternEvent) -> AnalysisResult {
        let mut warnings = vec![];

        match event.event_type.as_str() {
            "edge_added" => {
                if let (Some(i), Some(j), Some(w)) = (
                    event.data.get("source").and_then(|v| v.as_u64()).map(|v| v as usize),
                    event.data.get("target").and_then(|v| v.as_u64()).map(|v| v as usize),
                    event.data.get("weight").and_then(|v| v.as_f64()),
                ) {
                    self.graph.add_edge(i, j, w);
                }
            }
            "venue_updated" => {
                if let (Some(id), Some(mass)) = (
                    event.data.get("venue_id").and_then(|v| v.as_u64()).map(|v| v as usize),
                    event.data.get("mass").and_then(|v| v.as_f64()),
                ) {
                    if id < self.pattern_state.n_venues {
                        self.pattern_state.mass[id] = mass;
                    }
                }
            }
            _ => {
                warnings.push(format!("Unknown event type: {}", event.event_type));
            }
        }

        AnalysisResult {
            spectral: Some(self.graph.analyze()),
            cohomology: Some(self.sheaf.cohomology()),
            transport_cost: None,
            conservation: None,
            homology: None,
            sunset_decomp: Some(self.sunset_op.decompose()),
            fleet_entropy: Some(self.fleet_state.gibbs_entropy()),
            fibonacci_ratio: Some(self.fib.golden_ratio()),
            landauer_cost: Some(landauer_bound(self.fleet_state.information_bits, self.fleet_state.temperature)),
            warnings,
        }
    }

    /// Run full analysis of current state.
    pub fn full_analysis(&self) -> AnalysisResult {
        AnalysisResult {
            spectral: Some(self.graph.analyze()),
            cohomology: Some(self.sheaf.cohomology()),
            transport_cost: None,
            conservation: None,
            homology: None,
            sunset_decomp: Some(self.sunset_op.decompose()),
            fleet_entropy: Some(self.fleet_state.gibbs_entropy()),
            fibonacci_ratio: Some(self.fib.golden_ratio()),
            landauer_cost: Some(landauer_bound(self.fleet_state.information_bits, self.fleet_state.temperature)),
            warnings: vec![],
        }
    }

    /// Compute vibe transport between two distributions.
    pub fn analyze_transport(&self, source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> (f64, TransportPlan) {
        let dist = wasserstein_1(source, target, cost);
        let plan = compute_transport_plan(source, target, cost);
        (dist, plan)
    }

    /// Check conservation laws between two states.
    pub fn check_conservation(&self, before: &PatternState, after: &PatternState) -> ConservationCheck {
        check_conservation(before, after, 1e-6)
    }

    /// Compute homology of the current complex.
    pub fn compute_homology(&self) -> HomologyResult {
        self.complex.homology()
    }

    /// Get the Noether conservation laws.
    pub fn noether_laws(&self) -> Vec<SymmetryConservationLaw> {
        noether_laws()
    }

    /// Compute JEPA manifold distances.
    pub fn jepa_distances(&self, embeddings: &[JepaEmbedding]) -> Vec<Vec<f64>> {
        let manifold = JepaManifold::new(embeddings.to_vec());
        manifold.distance_matrix()
    }

    /// Fibonacci analysis.
    pub fn fibonacci_analysis(&self, n: usize) -> (Vec<u64>, f64) {
        let values: Vec<u64> = (0..=n).map(|i| self.fib.get(i)).collect();
        let ratio = self.fib.golden_ratio();
        (values, ratio)
    }

    /// Fleet thermodynamic evolution.
    pub fn evolve_fleet(&mut self, process: &ThermoProcess) -> crate::fleet_thermodynamics::FleetEvolution {
        evolve_fleet(&mut self.fleet_state, process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let api = BridgeApi::new(5);
        assert_eq!(api.graph.n, 5);
        assert_eq!(api.sheaf.n_venues, 5);
    }

    #[test]
    fn test_process_edge_event() {
        let mut api = BridgeApi::new(3);
        let event = PatternEvent {
            event_type: "edge_added".into(),
            venue_id: None,
            data: serde_json::json!({"source": 0, "target": 1, "weight": 1.5}),
            timestamp: 0.0,
        };
        let result = api.process_event(&event);
        assert!(result.spectral.is_some());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_process_venue_event() {
        let mut api = BridgeApi::new(3);
        let event = PatternEvent {
            event_type: "venue_updated".into(),
            venue_id: Some(0),
            data: serde_json::json!({"venue_id": 0, "mass": 5.0}),
            timestamp: 1.0,
        };
        let result = api.process_event(&event);
        assert!(result.spectral.is_some());
    }

    #[test]
    fn test_unknown_event() {
        let mut api = BridgeApi::new(3);
        let event = PatternEvent {
            event_type: "unknown_event".into(),
            venue_id: None,
            data: serde_json::json!({}),
            timestamp: 0.0,
        };
        let result = api.process_event(&event);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_full_analysis() {
        let api = BridgeApi::new(4);
        let result = api.full_analysis();
        assert!(result.spectral.is_some());
        assert!(result.cohomology.is_some());
        assert!(result.sunset_decomp.is_some());
        assert!(result.fleet_entropy.is_some());
        assert!(result.fibonacci_ratio.is_some());
        assert!(result.landauer_cost.is_some());
    }

    #[test]
    fn test_transport_analysis() {
        let api = BridgeApi::new(3);
        let source = VibeDistribution::from_masses(vec![1.0, 0.0, 0.0]);
        let target = VibeDistribution::from_masses(vec![0.0, 0.0, 1.0]);
        let cost = CostMatrix::from_positions_1d(&[0.0, 1.0, 2.0]);
        let (dist, plan) = api.analyze_transport(&source, &target, &cost);
        assert!(dist >= 0.0);
        assert!(plan.cost >= 0.0);
    }

    #[test]
    fn test_conservation_check() {
        let api = BridgeApi::new(3);
        let before = PatternState::zero(3);
        let after = PatternState::zero(3);
        let check = api.check_conservation(&before, &after);
        assert!(check.mass_conserved);
    }

    #[test]
    fn test_jepa_distances() {
        let api = BridgeApi::new(3);
        let embeddings = vec![
            JepaEmbedding::from_vec(vec![0.0, 0.0]),
            JepaEmbedding::from_vec(vec![1.0, 0.0]),
            JepaEmbedding::from_vec(vec![0.0, 1.0]),
        ];
        let dists = api.jepa_distances(&embeddings);
        assert_eq!(dists.len(), 3);
        assert!((dists[0][0]).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci_analysis() {
        let api = BridgeApi::new(5);
        let (values, ratio) = api.fibonacci_analysis(10);
        assert_eq!(values.len(), 11);
        assert!(ratio > 1.5 && ratio < 2.0);
    }

    #[test]
    fn test_fleet_evolution() {
        let mut api = BridgeApi::new(3);
        api.fleet_state.information_bits = 1000.0;
        let process = ThermoProcess::adiabatic(5.0);
        let result = api.evolve_fleet(&process);
        assert!((result.energy_change - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_laws() {
        let api = BridgeApi::new(3);
        let laws = api.noether_laws();
        assert_eq!(laws.len(), 5);
    }

    #[test]
    fn test_serialization() {
        let api = BridgeApi::new(3);
        let result = api.full_analysis();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("spectral"));
    }
}
