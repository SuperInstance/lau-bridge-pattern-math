# lau-bridge-pattern-math

> Bridges the Grand Pattern system with deep math crates: spectral theory, sheaves, category theory, optimal transport, free probability, PDEs

## What This Does

Bridges the Grand Pattern system with deep math crates: spectral theory, sheaves, category theory, optimal transport, free probability, PDEs. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-bridge-pattern-math
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_bridge_pattern_math::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct FibonacciSequence 
    pub fn new(n: usize) -> Self 
    pub fn get(&self, k: usize) -> u64 
    pub fn golden_ratio(&self) -> f64 
pub struct FibMorphism 
    pub fn new(source: usize, target: usize, fib: &FibonacciSequence) -> Self 
    pub fn compose(&self, other: &FibMorphism) -> FibMorphism 
pub struct FibonacciCategory 
    pub fn new(n_objects: usize) -> Self 
    pub fn morphism(&self, n: usize, m: usize) -> FibMorphism 
    pub fn identity(&self, n: usize) -> FibMorphism 
    pub fn check_associativity(&self, f: &FibMorphism, g: &FibMorphism, h: &FibMorphism) -> bool 
    pub fn zeckendorf(&self, n: u64) -> Vec<u64> 
pub struct FreeFibonacciMonoid 
    pub fn new(max_len: usize) -> Self 
    pub fn count(&self, n: usize) -> usize 
    pub fn verify_fibonacci_growth(&self) -> bool 
pub struct CategorifiedFibonacci 
    pub fn euler_characteristic(&self, n: usize) -> i64 
    pub fn generating_function_coefficients(&self, n: usize) -> Vec<f64> 
    pub fn binet(&self, n: usize) -> f64 
pub struct PatternEvent 
pub struct AnalysisResult 
pub struct BridgeApi 
    pub fn new(n_venues: usize) -> Self 
    pub fn process_event(&mut self, event: &PatternEvent) -> AnalysisResult 
    pub fn full_analysis(&self) -> AnalysisResult 
    pub fn analyze_transport(&self, source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> (f64, TransportPlan) 
    pub fn check_conservation(&self, before: &PatternState, after: &PatternState) -> ConservationCheck 
    pub fn compute_homology(&self) -> HomologyResult 
    pub fn noether_laws(&self) -> Vec<SymmetryConservationLaw> 
    pub fn jepa_distances(&self, embeddings: &[JepaEmbedding]) -> Vec<Vec<f64>> 
    pub fn fibonacci_analysis(&self, n: usize) -> (Vec<u64>, f64) 
    pub fn evolve_fleet(&mut self, process: &ThermoProcess) -> crate::fleet_thermodynamics::FleetEvolution 
pub struct VibeDistribution 
    pub fn uniform(n: usize) -> Self 
    pub fn from_masses(masses: Vec<f64>) -> Self 
    pub fn is_valid(&self) -> bool 
    pub fn n(&self) -> usize 
pub struct CostMatrix 
    pub fn from_distances(distances: Vec<Vec<f64>>) -> Self 
    pub fn uniform(n: usize) -> Self 
    pub fn from_positions_1d(positions: &[f64]) -> Self 
    pub fn from_positions_2d(positions: &[(f64, f64)]) -> Self 
pub struct TransportPlan 
pub fn wasserstein_1d(source: &VibeDistribution, target: &VibeDistribution) -> f64 
pub fn wasserstein_1(source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> f64 
pub fn compute_transport_plan(source: &VibeDistribution, target: &VibeDistribution, cost: &CostMatrix) -> TransportPlan 
pub fn barycenter(distributions: &[VibeDistribution], weights: &[f64]) -> VibeDistribution 
pub struct ConservedQuantity 
pub struct PatternState 
    pub fn zero(n: usize) -> Self 
    pub fn total_mass(&self) -> f64 
    pub fn total_momentum(&self) -> (f64, f64) 
    pub fn total_energy(&self) -> f64 
    pub fn kinetic_energy(&self) -> f64 
    pub fn potential_energy(&self) -> f64 
pub struct ConservationCheck 
pub fn check_conservation(before: &PatternState, after: &PatternState, tol: f64) -> ConservationCheck 
pub struct SymmetryConservationLaw 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**127 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
