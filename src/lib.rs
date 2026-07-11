use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Core modules
pub mod algorithms;
pub mod core;

// Python bindings
mod bindings;

/// A Python module implemented in Rust.
#[pymodule]
fn _colonyx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bindings::two_opt_py, m)?)?;
    m.add_class::<bindings::PyAntColony>()?;
    m.add_class::<bindings::PyParticleSwarm>()?;
    m.add_class::<bindings::PyBeeColony>()?;
    m.add_class::<bindings::PyGreyWolfOptimizer>()?;
    m.add_class::<bindings::PyFireflyOptimizer>()?;
    m.add_class::<bindings::PySimulatedAnnealing>()?;
    m.add_class::<bindings::PyCuckooSearch>()?;
    m.add_class::<bindings::PyBatAlgorithm>()?;
    m.add_class::<bindings::PyGlowwormOptimizer>()?;
    m.add_class::<bindings::PyBacterialForagingOptimizer>()?;
    m.add_class::<bindings::PyDifferentialEvolution>()?;
    m.add_class::<bindings::PyCmaEsOptimizer>()?;
    Ok(())
}
