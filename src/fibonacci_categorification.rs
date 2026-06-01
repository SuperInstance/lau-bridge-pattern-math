//! Fibonacci growth as a categorical construction.
//! Fibonacci numbers arise as the free monoid on 2 generators
//! with the relation F(n+2) = F(n+1) + F(n).

use serde::{Serialize, Deserialize};

/// Fibonacci sequence generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciSequence {
    /// Cached values.
    values: Vec<u64>,
}

impl FibonacciSequence {
    /// Create a new Fibonacci sequence, precomputing up to `n` terms.
    pub fn new(n: usize) -> Self {
        let mut values = Vec::with_capacity(n.max(2));
        values.push(0);
        values.push(1);
        while values.len() < n {
            let len = values.len();
            values.push(values[len - 1] + values[len - 2]);
        }
        Self { values }
    }

    /// Get the k-th Fibonacci number (0-indexed).
    pub fn get(&self, k: usize) -> u64 {
        if k < self.values.len() {
            self.values[k]
        } else {
            // Extend the sequence up to k
            let mut vals = self.values.clone();
            while vals.len() <= k {
                let len = vals.len();
                vals.push(vals[len - 1] + vals[len - 2]);
            }
            vals[k]
        }
    }

    /// Golden ratio approximation from the sequence.
    pub fn golden_ratio(&self) -> f64 {
        let n = self.values.len();
        if n < 3 {
            return 1.618033988749895; // exact enough
        }
        self.values[n - 1] as f64 / self.values[n - 2] as f64
    }
}

/// A morphism in the Fibonacci category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibMorphism {
    /// Source object (index in the Fibonacci sequence).
    pub source: usize,
    /// Target object.
    pub target: usize,
    /// The morphism is given by F(target) / F(source).
    pub ratio: f64,
}

impl FibMorphism {
    /// Create a morphism between two Fibonacci objects.
    pub fn new(source: usize, target: usize, fib: &FibonacciSequence) -> Self {
        let fs = fib.get(source).max(1) as f64;
        let ft = fib.get(target) as f64;
        Self { source, target, ratio: ft / fs }
    }

    /// Compose two morphisms: g ∘ f.
    pub fn compose(&self, other: &FibMorphism) -> FibMorphism {
        assert_eq!(self.source, other.target, "composition: source must match target");
        FibMorphism {
            source: other.source,
            target: self.target,
            ratio: self.ratio * other.ratio,
        }
    }
}

/// The Fibonacci category: objects are natural numbers,
/// morphisms n → m are given by F(m)/F(n).
#[derive(Debug, Clone)]
pub struct FibonacciCategory {
    pub fib: FibonacciSequence,
    pub n_objects: usize,
}

impl FibonacciCategory {
    /// Create a Fibonacci category with `n` objects.
    pub fn new(n_objects: usize) -> Self {
        Self {
            fib: FibonacciSequence::new(n_objects + 2),
            n_objects,
        }
    }

    /// Get the morphism n → m.
    pub fn morphism(&self, n: usize, m: usize) -> FibMorphism {
        assert!(n < self.n_objects && m < self.n_objects);
        FibMorphism::new(n, m, &self.fib)
    }

    /// Identity morphism at object n.
    pub fn identity(&self, n: usize) -> FibMorphism {
        FibMorphism { source: n, target: n, ratio: 1.0 }
    }

    /// Check associativity: h ∘ (g ∘ f) == (h ∘ g) ∘ f.
    pub fn check_associativity(&self, f: &FibMorphism, g: &FibMorphism, h: &FibMorphism) -> bool {
        let fg = g.compose(f);
        let h_fg = h.compose(&fg);
        let hg = h.compose(g);
        let hg_f = hg.compose(f);
        (h_fg.ratio - hg_f.ratio).abs() < 1e-10
    }

    /// Zeckendorf representation: express a number as sum of non-consecutive Fibonacci numbers.
    pub fn zeckendorf(&self, n: u64) -> Vec<u64> {
        let mut fibs: Vec<u64> = self.fib.values.iter().filter(|&&f| f > 0 && f <= n).copied().collect();
        fibs.sort_by(|a, b| b.cmp(a));
        
        let mut representation = vec![];
        let mut remaining = n;
        for &f in &fibs {
            if f <= remaining {
                remaining -= f;
                representation.push(f);
            }
            if remaining == 0 { break; }
        }
        representation
    }
}

/// Free monoid on 2 generators {a, b} with Fibonacci growth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeFibonacciMonoid {
    /// Words of length n: count = F(n+1).
    pub words_by_length: Vec<Vec<String>>,
}

impl FreeFibonacciMonoid {
    /// Generate all words up to length `max_len`.
    pub fn new(max_len: usize) -> Self {
        let mut words_by_length = vec![vec!["".to_string()]]; // length 0: empty word
        
        // Generator words
        words_by_length.push(vec!["a".to_string(), "b".to_string()]);
        
        for len in 2..=max_len {
            let mut words = vec![];
            let prev = &words_by_length[len - 1];
            for word in prev {
                // Rule: after 'a', can append 'a' or 'b'
                //        after 'b', can only append 'a'
                // This gives Fibonacci growth
                let last_char = word.chars().last();
                match last_char {
                    Some('a') => {
                        words.push(word.clone() + "a");
                        words.push(word.clone() + "b");
                    }
                    Some('b') => {
                        words.push(word.clone() + "a");
                    }
                    _ => {}
                }
            }
            words_by_length.push(words);
        }
        
        Self { words_by_length }
    }

    /// Count of words of length n (should be F(n+1)).
    pub fn count(&self, n: usize) -> usize {
        self.words_by_length.get(n).map(|w| w.len()).unwrap_or(0)
    }

    /// Verify Fibonacci growth.
    pub fn verify_fibonacci_growth(&self) -> bool {
        for n in 2..self.words_by_length.len() {
            let expected = self.count(n - 1) + self.count(n - 2);
            if self.count(n) != expected {
                return false;
            }
        }
        true
    }
}

/// Categorified Fibonacci: the Fibonacci numbers as dimensions of
/// Hom-spaces in a category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorifiedFibonacci {
    pub fib: FibonacciSequence,
}

impl CategorifiedFibonacci {
    /// The Euler characteristic of the Fibonacci complex.
    pub fn euler_characteristic(&self, n: usize) -> i64 {
        // Alternating sum of Fibonacci numbers: F(0) - F(1) + F(2) - ...
        let mut chi = 0i64;
        for i in 0..=n {
            let sign = if i % 2 == 0 { 1 } else { -1 };
            chi += sign * self.fib.get(i) as i64;
        }
        chi
    }

    /// The Fibonacci generating function: sum F(n) x^n = x / (1 - x - x²).
    pub fn generating_function_coefficients(&self, n: usize) -> Vec<f64> {
        (0..=n).map(|i| self.fib.get(i) as f64).collect()
    }

    /// Binet's formula: F(n) = (φ^n - ψ^n) / √5.
    pub fn binet(&self, n: usize) -> f64 {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let psi = (1.0 - 5.0_f64.sqrt()) / 2.0;
        (phi.powi(n as i32) - psi.powi(n as i32)) / 5.0_f64.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_basic() {
        let fib = FibonacciSequence::new(10);
        assert_eq!(fib.get(0), 0);
        assert_eq!(fib.get(1), 1);
        assert_eq!(fib.get(2), 1);
        assert_eq!(fib.get(3), 2);
        assert_eq!(fib.get(5), 5);
        assert_eq!(fib.get(10), 55);
    }

    #[test]
    fn test_golden_ratio() {
        let fib = FibonacciSequence::new(20);
        let phi = fib.golden_ratio();
        assert!((phi - 1.6180339887).abs() < 0.01, "golden ratio should be ~1.618");
    }

    #[test]
    fn test_fibonacci_morphism() {
        let fib = FibonacciSequence::new(10);
        let m = FibMorphism::new(4, 6, &fib);
        // F(6)/F(4) = 8/3
        assert!((m.ratio - 8.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_morphism_composition() {
        let fib = FibonacciSequence::new(10);
        let f = FibMorphism::new(3, 4, &fib);
        let g = FibMorphism::new(4, 5, &fib);
        let h = g.compose(&f);
        assert_eq!(h.source, 3);
        assert_eq!(h.target, 5);
        // Should equal direct morphism 3→5
        let direct = FibMorphism::new(3, 5, &fib);
        assert!((h.ratio - direct.ratio).abs() < 1e-8);
    }

    #[test]
    fn test_identity_morphism() {
        let cat = FibonacciCategory::new(10);
        let id = cat.identity(5);
        assert!((id.ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_associativity() {
        let cat = FibonacciCategory::new(10);
        let f = cat.morphism(1, 3);
        let g = cat.morphism(3, 5);
        let h = cat.morphism(5, 7);
        assert!(cat.check_associativity(&f, &g, &h));
    }

    #[test]
    fn test_zeckendorf() {
        let cat = FibonacciCategory::new(15);
        let rep = cat.zeckendorf(100);
        let sum: u64 = rep.iter().map(|&x| x as u64).sum();
        assert_eq!(sum, 100);
    }

    #[test]
    fn test_free_monoid_growth() {
        let fm = FreeFibonacciMonoid::new(8);
        assert!(fm.verify_fibonacci_growth());
    }

    #[test]
    fn test_free_monoid_counts() {
        let fm = FreeFibonacciMonoid::new(6);
        assert_eq!(fm.count(0), 1); // empty word
        assert_eq!(fm.count(1), 2); // a, b
        assert_eq!(fm.count(2), 3); // aa, ab, ba
        assert_eq!(fm.count(3), 5); // F(4) = 5
        assert_eq!(fm.count(4), 8); // F(5) = 8
    }

    #[test]
    fn test_categorified_euler() {
        let cat = CategorifiedFibonacci { fib: FibonacciSequence::new(10) };
        let chi = cat.euler_characteristic(5);
        // F(0) - F(1) + F(2) - F(3) + F(4) - F(5) = 0 - 1 + 1 - 2 + 3 - 5 = -4
        assert_eq!(chi, -4);
    }

    #[test]
    fn test_binet_formula() {
        let cat = CategorifiedFibonacci { fib: FibonacciSequence::new(15) };
        for n in 0..=10 {
            let binet = cat.binet(n);
            let actual = cat.fib.get(n) as f64;
            assert!((binet - actual).abs() < 0.01, "Binet mismatch at n={}", n);
        }
    }

    #[test]
    fn test_generating_function() {
        let cat = CategorifiedFibonacci { fib: FibonacciSequence::new(6) };
        let coeffs = cat.generating_function_coefficients(5);
        assert_eq!(coeffs.len(), 6);
        assert!((coeffs[0] - 0.0).abs() < 1e-10);
        assert!((coeffs[5] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci_recurrence() {
        let fib = FibonacciSequence::new(15);
        for n in 2..15 {
            assert_eq!(fib.get(n), fib.get(n - 1) + fib.get(n - 2),
                "Fibonacci recurrence failed at n={}", n);
        }
    }
}
