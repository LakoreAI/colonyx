/// Bounds for optimization variables
#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl Bounds {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, String> {
        if lower.len() != upper.len() {
            return Err("Lower and upper bounds must have the same length".to_string());
        }

        for (i, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
            if l > u {
                return Err(format!(
                    "Lower bound {} > upper bound {} at index {}",
                    l, u, i
                ));
            }
        }

        Ok(Self { lower, upper })
    }

    /// Create uniform bounds for all dimensions
    pub fn uniform(dimensions: usize, lower: f64, upper: f64) -> Result<Self, String> {
        if lower > upper {
            return Err("Lower bound must be <= upper bound".to_string());
        }

        Ok(Self {
            lower: vec![lower; dimensions],
            upper: vec![upper; dimensions],
        })
    }

    /// Check if a solution is within bounds
    pub fn contains(&self, solution: &[f64]) -> bool {
        if solution.len() != self.lower.len() {
            return false;
        }

        for (i, &value) in solution.iter().enumerate() {
            if value < self.lower[i] || value > self.upper[i] {
                return false;
            }
        }

        true
    }

    /// Clamp a solution to be within bounds.
    /// Panics if the solution has more dimensions than the bounds.
    pub fn clamp(&self, solution: &mut [f64]) {
        assert!(
            solution.len() <= self.lower.len(),
            "solution has {} dimensions but bounds have {}",
            solution.len(),
            self.lower.len()
        );
        for (i, value) in solution.iter_mut().enumerate() {
            *value = value.max(self.lower[i]).min(self.upper[i]);
        }
    }

    /// Get the range (upper - lower) for each dimension
    pub fn ranges(&self) -> Vec<f64> {
        self.upper
            .iter()
            .zip(self.lower.iter())
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Get the midpoint of bounds
    pub fn midpoint(&self) -> Vec<f64> {
        self.upper
            .iter()
            .zip(self.lower.iter())
            .map(|(u, l)| (u + l) / 2.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Bounds;

    #[test]
    fn uniform_bounds_validate_and_contain() {
        let bounds = Bounds::uniform(3, -1.0, 1.0).expect("valid bounds");
        assert!(bounds.contains(&[0.0, 0.5, -0.5]));
        assert!(!bounds.contains(&[0.0, 0.5]));
        assert!(!bounds.contains(&[0.0, 2.0, -0.5]));
    }

    #[test]
    fn midpoint_is_center_of_bounds() {
        let bounds = Bounds::new(vec![-2.0, 0.0], vec![2.0, 4.0]).expect("valid bounds");
        assert_eq!(bounds.midpoint(), vec![0.0, 2.0]);
    }
}
