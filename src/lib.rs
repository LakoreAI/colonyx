use pyo3::prelude::*;

// Core modules
pub mod core;
pub mod algorithms;

// Python bindings
mod bindings;

/// A Python module implemented in Rust.
#[pymodule]
fn _colonyx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::PyAntColony>()?;
    m.add_class::<bindings::PyParticleSwarm>()?;
    m.add_class::<bindings::PyBeeColony>()?;
    Ok(())
}
