//! `lau-bridge-pattern-math` — Bridges the Grand Pattern system with deep math crates.
//!
//! The Grand Pattern has: cellular graphs, Fibonacci growth, JEPA embedding,
//! venue-as-agent, topology sweep, conservation laws.
//!
//! The Math crates have: spectral theory, sheaves, category theory, optimal transport,
//! free probability, PDEs.

pub mod graph_spectral;
pub mod venue_sheaf;
pub mod vibe_transport;
pub mod conservation_laws;
pub mod jepa_geometry;
pub mod fibonacci_categorification;
pub mod topology_homology;
pub mod sunset_spectral;
pub mod fleet_thermodynamics;
pub mod bridge_api;

pub use bridge_api::BridgeApi;
