# lau-bridge-pattern-math

**Bridges the Grand Pattern system with deep math crates: spectral theory, sheaves, category theory, optimal transport, Fibonacci categorification, and thermodynamic laws.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests: 127](https://img.shields.io/badge/tests-127-brightgreen.svg)]()

---

## What This Does

`lau-bridge-pattern-math` is the bridge between two worlds:

1. **The Grand Pattern** — a venue/agent system with cellular graphs, Fibonacci growth, JEPA embeddings, conservation laws, and fleet thermodynamics
2. **The Math Crates** — 320+ crates covering spectral theory, sheaves, category theory, optimal transport, free probability, and PDEs

This crate provides 10 mathematical modules that translate Grand Pattern events into deep mathematical analysis:

| Module | What It Does |
|--------|-------------|
| `graph_spectral` | Spectral analysis of venue connectivity graphs (Laplacian eigenvalues, clustering) |
| `venue_sheaf` | Sheaf structure on venues: stalks, restriction maps, sheaf Laplacian, cohomology |
| `vibe_transport` | Optimal transport of "vibes" (distributions) between venues via Wasserstein distances |
| `conservation_laws` | Energy/mass/momentum conservation + Noether's theorem for the Grand Pattern |
| `jepa_geometry` | Riemannian geometry of JEPA embedding space: Fisher metric, geodesics, Fréchet mean |
| `fibonacci_categorification` | Fibonacci sequence as a categorical construction: free monoid, Zeckendorf, Binet's formula |
| `topology_homology` | Simplicial homology of the pattern topology: Betti numbers, Euler characteristic |
| `sunset_spectral` | Spectral decomposition of the sunset lifecycle: ethos/pathos/logos as eigenvectors |
| `fleet_thermodynamics` | Thermodynamic laws for fleet evolution: entropy, Landauer bound, Carnot efficiency |
| `bridge_api` | Unified API that wires all modules together into a single `process_event()` call |

---

## Key Idea

The Grand Pattern generates **events** (edges added, venues updated, vibes shifted). Each event triggers a cascade of mathematical analysis across all 10 modules simultaneously.

A single call to `BridgeApi::process_event()` returns:
- **Spectral analysis** of the updated graph
- **Sheaf cohomology** of venue data
- **Conservation law checks** (did mass/energy/momentum change?)
- **Homology groups** of the current topology
- **Sunset decomposition** (ethos/pathos/logos eigenvalues)
- **Fleet entropy** and **Landauer cost** (minimum energy for information erasure)
- **Fibonacci golden ratio** of the current state

It's a mathematical "everything detector" for the Grand Pattern.

---

## Install

```toml
[dependencies]
lau-bridge-pattern-math = "0.1.0"
```

Requires **Rust 2021 edition**. Dependencies: `nalgebra`, `serde`, `serde_json`.

---

## Quick Start

```rust
use lau_bridge_pattern_math::{BridgeApi, PatternEvent};
use serde_json::json;

// Create the bridge with 5 venues
let mut api = BridgeApi::new(5);

// Process a Grand Pattern event: add an edge between venues 0 and 1
let event = PatternEvent {
    event_type: "edge_added".into(),
    venue_id: None,
    data: json!({"source": 0, "target": 1, "weight": 1.5}),
    timestamp: 0.0,
};
let result = api.process_event(&event);

// Inspect the unified analysis
println!("Connected: {}", result.spectral.unwrap().is_connected);
println!("Fleet entropy: {:.4}", result.fleet_entropy.unwrap());
println!("Landauer cost: {:.2e} J", result.landauer_cost.unwrap());
println!("Golden ratio: {:.6}", result.fibonacci_ratio.unwrap());

// Run full analysis of current state
let full = api.full_analysis();

// Check conservation between two states
let before = PatternState::zero(5);
let after = PatternState::zero(5);
let check = api.check_conservation(&before, &after);
assert!(check.mass_conserved);
```

---

## API Reference

### `BridgeApi` — Unified Entry Point

```rust
let mut api = BridgeApi::new(n_venues);

// Process events
api.process_event(&event) → AnalysisResult

// Full analysis
api.full_analysis() → AnalysisResult

// Transport analysis
api.analyze_transport(&source, &target, &cost) → (distance, TransportPlan)

// Conservation
api.check_conservation(&before, &after) → ConservationCheck

// Homology
api.compute_homology() → HomologyResult

// Noether's laws
api.noether_laws() → Vec<SymmetryConservationLaw>

// JEPA geometry
api.jepa_distances(&embeddings) → Vec<Vec<f64>>

// Fibonacci
api.fibonacci_analysis(n) → (Vec<u64>, f64)

// Thermodynamics
api.evolve_fleet(&process) → FleetEvolution
```

### `graph_spectral` — Cellular Graph Analysis

```rust
let mut graph = CellularGraph::new(n);
graph.add_edge(i, j, weight);

graph.laplacian()              // D - A (unnormalized)
graph.normalized_laplacian()   // I - D^{-1/2} A D^{-1/2}
graph.laplacian_eigenvalues()  // Sorted eigenvalues via Jacobi iteration
graph.algebraic_connectivity() // 2nd smallest eigenvalue (Fiedler value)
graph.is_connected()           // algebraic_connectivity > 0
graph.spectral_cluster(k)      // k-way clustering
graph.analyze()                // Full SpectralAnalysis struct
```

### `venue_sheaf` — Sheaves on Venues

```rust
let mut sheaf = VenueSheaf::new(n_venues, stalk_dim);
sheaf.set_stalk(venue_id, data);
sheaf.add_restriction_map(map);

sheaf.sheaf_laplacian()       // Global sheaf Laplacian matrix
sheaf.consistency_error()     // √Σ ||ρ(s_i) - s_j||²
sheaf.is_global_section(tol)  // consistency_error < tol
sheaf.pushforward()           // Concatenated stalk data
sheaf.cohomology()            // SheafCohomology { h0, h1, error }
```

### `vibe_transport` — Optimal Transport

```rust
let source = VibeDistribution::from_masses(vec![1.0, 0.0, 0.0]);
let target = VibeDistribution::from_masses(vec![0.0, 0.0, 1.0]);
let cost = CostMatrix::from_positions_1d(&[0.0, 1.0, 2.0]);

let w1 = wasserstein_1(&source, &target, &cost);  // Wasserstein-1 distance
let w1d = wasserstein_1d(&source, &target);        // 1D closed-form: ∫|CDF₁ - CDF₂|
let plan = compute_transport_plan(&source, &target, &cost); // Greedy OT
let bary = barycenter(&[dist_a, dist_b], &[1.0, 1.0]);     // Weighted average
```

### `conservation_laws` — Physics of the Pattern

```rust
let state = PatternState::zero(n_venues);
state.total_mass();
state.total_momentum();
state.kinetic_energy();    // Σ ½|p|²
state.potential_energy();  // -Σ A_{ij} m_i m_j

check_conservation(&before, &after, tol) → ConservationCheck

// Noether's theorem: symmetry → conservation law
noether_laws() → Vec<SymmetryConservationLaw>;
// Time translation → Energy conservation
// Venue permutation → Mass conservation
// Spatial translation → Momentum conservation
// Phase rotation → Vibe amplitude conservation
// Scale invariance → Fibonacci ratio conservation

// Conservative evolution
evolve_mass(&mut state, dt);        // Diffusion with mass conservation
evolve_hamiltonian(&mut state, dt); // Hamiltonian flow
```

### `jepa_geometry` — Information Geometry of JEPA Embeddings

```rust
let emb = JepaEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
emb.normalize();
emb.euclidean_distance(&other);
emb.dot(&other);

let metric = FisherMetric::identity(3);
metric.geodesic_distance(&a, &b);       // Fisher-Rao distance
metric.christoffel_symbols();            // Connection coefficients
metric.volume_element();                 // √det(G)

let manifold = JepaManifold::new(embeddings);
manifold.distance_matrix();              // Pairwise distances
manifold.frechet_mean();                 // Riemannian center of mass
manifold.exp_map(&base, &tangent);       // Exponential map
manifold.log_map(&base, &point);         // Logarithmic map
```

### `fibonacci_categorification` — Fibonacci as Category Theory

```rust
let fib = FibonacciSequence::new(20);
fib.get(n);              // F(n)
fib.golden_ratio();      // F(n)/F(n-1) → φ ≈ 1.618

let cat = FibonacciCategory::new(10);
cat.morphism(n, m);      // Ratio F(m)/F(n)
cat.identity(n);          // Ratio 1.0
cat.zeckendorf(100);      // [89, 8, 3] — non-consecutive Fibonacci sum

let monoid = FreeFibonacciMonoid::new(8);
monoid.count(n);          // F(n+1)
monoid.verify_fibonacci_growth();  // F(n) = F(n-1) + F(n-2)

let c = CategorifiedFibonacci { fib };
c.euler_characteristic(n);           // Σ(-1)^i F(i)
c.binet(n);                          // (φⁿ - ψⁿ)/√5
c.generating_function_coefficients(n); // [F(0), F(1), ..., F(n)]
```

### `topology_homology` — Simplicial Homology

```rust
let mut complex = PatternComplex::new();
let v0 = complex.add_vertex("venue-0");
let v1 = complex.add_vertex("venue-1");
let v2 = complex.add_vertex("venue-2");
complex.add_edge(v0, v1);
complex.add_edge(v1, v2);
complex.add_edge(v0, v2);
// Hollow triangle → β₀=1, β₁=1

complex.add_triangle(v0, v1, v2);
// Filled triangle → β₀=1, β₁=0

complex.betti_numbers();         // (β₀, β₁, β₂)
complex.euler_characteristic();  // β₀ - β₁ + β₂
complex.boundary_1();            // ∂₁: edges → vertices
complex.boundary_2();            // ∂₂: triangles → edges
complex.homology();              // Full HomologyResult
```

### `sunset_spectral` — Ethos/Pathos/Logos Decomposition

```rust
let op = SunsetOperator::symmetric_mixing(0.1);
let state = SunsetState::pure(SunsetMode::Ethos);
let evolved = op.apply(&state);

op.eigenvalues();               // [λ₁, λ₂, λ₃] via Jacobi iteration
op.dominant_eigenvector(100);   // Power iteration
op.compose(&other);             // Operator composition
op.power(n);                    // Apply n times
op.decompose();                 // Full SpectralDecomposition
```

### `fleet_thermodynamics` — Thermodynamic Laws for Fleets

```rust
let state = FleetThermoState::zero(n_agents);
state.total_energy();
state.free_energy();            // E - TS
state.partition_function();     // Z = Σ exp(-β Eᵢ)
state.boltzmann_probabilities(); // exp(-β Eᵢ) / Z
state.gibbs_entropy();          // -Σ pᵢ ln(pᵢ)
state.heat_capacity();          // β²(⟨E²⟩ - ⟨E⟩²)

landauer_bound(bits, temperature);  // E ≥ nkT ln(2)

let process = ThermoProcess::adiabatic(work);
evolve_fleet(&mut state, &process);

carnot_efficiency(t_hot, t_cold);  // η = 1 - T_c/T_h
```

---

## How It Works

### Architecture

```
         Grand Pattern Events
                │
                ▼
        ┌───────────────┐
        │   BridgeApi    │  ← Unified entry point
        └───────┬───────┘
                │
     ┌──────────┼──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Graph   │ │ Venue  │ │  Vibe  │ │ Conser-│ │ JEPA   │
│Spectral │ │ Sheaf  │ │Trans-  │ │ vation │ │Geometry│
│(Laplacian)│(cohom) │ │port(OT)│ │ Laws   │ │(Fisher)│
└─────────┘ └────────┘ └────────┘ └────────┘ └────────┘
     │          │          │          │          │
     └──────────┼──────────┴──────────┘          │
                ▼                                ▼
     ┌──────────────────┐              ┌──────────────────┐
     │  Topology/       │              │ Fleet             │
     │  Homology        │              │ Thermodynamics    │
     │  (Betti numbers) │              │ (Entropy, Landauer)│
     └──────────────────┘              └──────────────────┘
                │                                │
                ▼                                ▼
     ┌──────────────────┐              ┌──────────────────┐
     │  Fibonacci       │              │ Sunset            │
     │  Categorification│              │ Spectral          │
     │  (φ, Zeckendorf) │              │ (Ethos/Pathos/    │
     └──────────────────┘              │  Logos eigenvectors)│
                                       └──────────────────┘
                │
                ▼
        AnalysisResult
        (all modules combined)
```

### Event Processing Pipeline

1. `PatternEvent` arrives (edge added, venue updated, etc.)
2. `BridgeApi` dispatches to the relevant internal state
3. All 10 modules update their state
4. An `AnalysisResult` is returned containing outputs from all modules
5. Conservation laws are checked: did the event violate mass/energy/momentum conservation?

---

## The Math

### Spectral Graph Theory

The **graph Laplacian** `L = D - A` where `D` is the degree matrix and `A` is the adjacency matrix. Its eigenvalues encode:
- `λ₁ = 0` always (constant eigenvector)
- `λ₂ > 0` iff the graph is connected (**Fiedler value**)
- The **spectral gap** `λ₂ - λ₁` controls mixing time

### Sheaf Theory

A **cellular sheaf** assigns vector spaces (stalks) to each cell of a graph, with linear maps (restriction maps) between stalks on adjacent cells. The **sheaf Laplacian** generalizes the graph Laplacian:

$$L_{\text{sheaf}} = \sum_{\text{edges}} \rho_i^T \rho_i + \rho_j^T \rho_j - \rho_i^T \rho_j - \rho_j^T \rho_i$$

**Cohomology**: `H⁰` = global sections (consistent data), `H¹` = obstructions to gluing.

### Optimal Transport

The **Wasserstein-1 distance** (Earth Mover's Distance) between distributions `μ` and `ν`:

$$W_1(\mu, \nu) = \inf_{\gamma \in \Pi(\mu, \nu)} \int c(x, y) \, d\gamma(x, y)$$

For 1D distributions, this simplifies to `W₁ = ∫|CDF_μ - CDF_ν| dx`.

### Noether's Theorem

Every continuous symmetry of the Grand Pattern corresponds to a conserved quantity:
- Time translation → Energy
- Space translation → Momentum
- Phase rotation → Vibe amplitude
- Scale invariance → Fibonacci ratio

### Fibonacci Categorification

Fibonacci numbers arise as dimensions of Hom-spaces in a category. The **free Fibonacci monoid** on generators `{a, b}` with the rule "after `b`, can only append `a`" gives word counts `F(n+1)`.

**Zeckendorf's theorem**: Every positive integer has a unique representation as a sum of non-consecutive Fibonacci numbers.

**Binet's formula**: `F(n) = (φⁿ - ψⁿ) / √5` where `φ = (1+√5)/2` and `ψ = (1-√5)/2`.

### Information Geometry

The **Fisher information metric** on the manifold of probability distributions:

$$g_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]$$

This induces geodesics (information-optimal paths), Christoffel symbols (connection), and volume elements (information capacity).

### Thermodynamics

**Landauer's principle**: Erasing one bit of information at temperature `T` requires at least `kT ln(2)` energy.

**Gibbs entropy**: `S = -Σ pᵢ ln pᵢ`

**Carnot efficiency**: `η = 1 - T_cold / T_hot` — the maximum possible efficiency of any heat engine.

---

## Test Coverage

**127 tests** across all 10 modules:

| Module | Tests | Key Verifications |
|--------|-------|-------------------|
| `graph_spectral` | 15 | Eigenvalues, connectivity, clustering, PSD check |
| `venue_sheaf` | 14 | Stalks, restrictions, sheaf Laplacian, cohomology |
| `vibe_transport` | 12 | Wasserstein distances, transport plans, barycenters |
| `conservation_laws` | 12 | Mass/energy/momentum, Noether, Hamiltonian evolution |
| `jepa_geometry` | 14 | Embeddings, Fisher metric, geodesics, exp/log maps |
| `fibonacci_categorification` | 14 | Sequence, category, Zeckendorf, Binet, monoid |
| `topology_homology` | 11 | Betti numbers, Euler characteristic, boundary matrices |
| `sunset_spectral` | 14 | Eigenvalues, power iteration, decomposition, entropy |
| `fleet_thermodynamics` | 16 | Entropy, Landauer, Boltzmann, Carnot, heat capacity |
| `bridge_api` | 12 | Event processing, full analysis, serialization |

---

## License

MIT
