# **colonyx — Comprehensive TODO List for Scikit-Learn Compatible Swarm Intelligence Library**

> **Note**: This library uses a unified interface called `AutoColony` (similar to HuggingFace's `AutoModel`) as the main entry point, with individual algorithms (ACO, PSO, ABC, GWO, FA, SA, CS, BA, GSO, DE) selected via the `mode` parameter.

## **Architecture Overview**
- **Main Interface**: `AutoColony` class with `mode` parameter for algorithm selection
- **Algorithm Modes**: 'aco', 'pso', 'abc', 'gwo', 'fa', 'sa', 'cs', 'ba', 'gso', 'de', 'auto' (auto-selects based on problem type)
- **Scikit-Learn Compatible**: Follows sklearn's `BaseEstimator` interface
- **HuggingFace-style**: Single class with mode switching like `AutoModel`

## **Phase 0: Foundation & Architecture (Sprint 0)**

### **0.1 Core Infrastructure Setup**
- [ ] **Rust Core Library Structure**
  - [ ] Create `src/algorithms/` module structure
  - [ ] Implement base `SwarmOptimizer` trait with required methods (Rust core)
  - [ ] Add `src/problems/` for problem definitions (TSP, continuous functions)
  - [ ] Create `src/utils/` for common utilities (random, math functions)
  - [ ] Add comprehensive error handling with custom error types
  - [ ] Implement logging infrastructure using `log` crate
  - [ ] Add benchmarking utilities using `criterion` crate

- [ ] **Python Package Structure (scikit-learn compatible)**
  - [ ] Create `python/colonyx/` directory structure
  - [ ] Implement `AutoColony` class as main interface (HuggingFace-style)
  - [ ] Implement `BaseOptimizer` class following scikit-learn's `BaseEstimator`
  - [ ] Add `__init__.py` files with proper module exports
  - [ ] Create `python/colonyx/base.py` for base classes
  - [ ] Create `python/colonyx/algorithms/` for internal algorithm implementations
  - [ ] Create `python/colonyx/utils/` for validation and utilities
  - [ ] Add `python/colonyx/metrics/` for scoring functions
  - [ ] Create `python/colonyx/datasets/` for benchmark problems
  - [ ] Add `python/colonyx/auto.py` for auto-selection logic

### **0.2 Development Environment & Tools**
- [ ] **Testing Infrastructure**
  - [ ] Set up `pytest` configuration with coverage
  - [ ] Add `tox.ini` for multi-Python version testing
  - [ ] Configure `pytest-benchmark` for performance testing
  - [ ] Add property-based testing with `hypothesis`
  - [ ] Set up Rust unit tests with `cargo test`
  - [ ] Add integration tests between Rust and Python

- [ ] **Code Quality Tools**
  - [ ] Configure `ruff` for Python linting and formatting
  - [ ] Add `mypy` configuration for type checking
  - [ ] Set up `clippy` for Rust linting
  - [ ] Add `rustfmt` configuration
  - [ ] Configure `pre-commit` hooks
  - [ ] Add `bandit` for security checks

- [ ] **Documentation Setup**
  - [ ] Set up `sphinx` with scikit-learn theme
  - [ ] Configure `autodoc` for API documentation
  - [ ] Add `numpydoc` style docstring validation
  - [ ] Set up `jupyter-book` for tutorials
  - [ ] Add example gallery using `sphinx-gallery`

---

## **Phase 1: Base Classes & Scikit-Learn Interface (Sprint 1)**

### **1.1 Unified Interface Implementation**
- [x] **AutoColony Class (Python) - Main Interface**
  - [x] Implement `__init__(mode='auto', **kwargs)` with mode parameter
  - [x] Add algorithm selection via `mode`: 'aco', 'pso', 'abc', 'auto'
  - [x] Implement automatic algorithm selection based on problem type
  - [x] Add parameter validation specific to each algorithm mode
  - [x] Implement dynamic parameter routing to underlying algorithms
  - [x] Add mode-specific parameter constraints and validation
  - [x] Support algorithm-specific parameters with clear documentation

- [ ] **BaseOptimizer Abstract Class (Python)**
  - [ ] Implement `__init__` with parameter validation
  - [ ] Add `get_params()` and `set_params()` methods
  - [ ] Implement `fit(X, y=None)` method signature
  - [ ] Add `score(X, y=None)` method for optimization quality
  - [ ] Implement `_validate_params()` using sklearn's validation
  - [ ] Add `_more_tags()` method for estimator metadata
  - [ ] Implement `__repr__()` with proper parameter display

- [ ] **Parameter Validation & Constraints**
  - [ ] Create parameter constraint classes (Interval, StrOptions, etc.)
  - [ ] Add input validation for optimization problems
  - [ ] Implement bounds checking for continuous problems
  - [ ] Add graph validation for discrete problems
  - [ ] Create custom exception classes for optimization errors

### **1.2 Problem Definition Classes**
- [ ] **BaseProblem Abstract Class**
  - [ ] Define interface for fitness/objective functions
  - [ ] Add problem dimension and bounds properties
  - [ ] Implement problem type classification (continuous/discrete)
  - [ ] Add problem metadata (name, description, optimal_value)

- [ ] **ContinuousProblem Class**
  - [ ] Support for multidimensional continuous functions
  - [ ] Bounds handling (box constraints)
  - [ ] Gradient information (if available)
  - [ ] Built-in test functions (Sphere, Rosenbrock, Rastrigin, etc.)

- [ ] **DiscreteProblem Class**
  - [ ] Graph representation for TSP-like problems
  - [ ] Adjacency matrix and distance matrix support
  - [ ] Constraint handling for discrete optimization
  - [ ] Support for custom discrete problems

---

## **Phase 2: Ant Colony Optimization (ACO) Implementation (Sprint 2)**

### **2.1 Rust Core Implementation**
- [ ] **ACO Algorithm Structure**
  - [ ] Implement `AntColonyCore` struct with pheromone matrix
  - [ ] Add `Ant` struct for individual ant behavior
  - [ ] Implement pheromone update rules (local and global)
  - [ ] Add evaporation mechanism
  - [ ] Implement construction graph traversal
  - [ ] Add elitist ant strategy
  - [ ] Implement max-min ant system (MMAS) variant

- [ ] **Performance Optimizations**
  - [ ] Use sparse matrices for large graphs
  - [ ] Implement parallel ant execution
  - [ ] Add memory-efficient pheromone storage
  - [ ] Optimize distance calculations
  - [ ] Add early stopping criteria

### **2.2 Python Interface**
- [ ] **AntColonyOptimizer Class (Internal)**
  - [ ] Inherit from `BaseOptimizer`
  - [ ] Parameters: `n_ants`, `n_iterations`, `alpha`, `beta`, `rho`, `q0`
  - [ ] Implement `fit(distance_matrix)` method
  - [ ] Add `predict()` method returning best tour
  - [ ] Implement `score()` method (negative tour length)
  - [ ] Add `transform()` method for tour representation
  - [ ] **Not exposed directly - accessed via AutoColony(mode='aco')**

- [ ] **Advanced Features**
  - [ ] Support for asymmetric TSP problems
  - [ ] Add callback system for monitoring convergence
  - [ ] Implement warm start functionality
  - [ ] Add multiple colony support
  - [ ] Support for constrained TSP variants

### **2.3 Testing & Validation**
- [ ] **Unit Tests**
  - [ ] Test pheromone matrix operations
  - [ ] Validate ant construction behavior
  - [ ] Test parameter validation
  - [ ] Benchmark against known TSP instances
  - [ ] Test convergence properties

- [ ] **Integration Tests**
  - [ ] Test with TSPLIB instances
  - [ ] Cross-validation with different problem sizes
  - [ ] Performance benchmarks vs. other implementations
  - [ ] Memory usage profiling

---

## **Phase 3: Particle Swarm Optimization (PSO) Implementation (Sprint 3)**

### **3.1 Rust Core Implementation**
- [ ] **PSO Algorithm Structure**
  - [ ] Implement `ParticleSwarmCore` struct
  - [ ] Add `Particle` struct with position, velocity, and personal best
  - [ ] Implement velocity update equations
  - [ ] Add inertia weight strategies (constant, linear, adaptive)
  - [ ] Implement constriction factor method
  - [ ] Add boundary handling strategies (reflection, absorption, etc.)
  - [ ] Support for different topology structures (ring, star, etc.)

- [ ] **Advanced PSO Variants**
  - [ ] Implement adaptive PSO (APSO)
  - [ ] Add comprehensive learning PSO (CLPSO)
  - [ ] Implement multi-swarm PSO
  - [ ] Add opposition-based learning
  - [ ] Support for discrete PSO variants

### **3.2 Python Interface**
- [ ] **ParticleSwarmOptimizer Class (Internal)**
  - [ ] Inherit from `BaseOptimizer`
  - [ ] Parameters: `n_particles`, `n_iterations`, `w`, `c1`, `c2`, `bounds`
  - [ ] Implement `fit(objective_function)` method
  - [ ] Add `predict()` method returning best solution
  - [ ] Implement `score()` method (negative objective value)
  - [ ] Add `transform()` method for solution representation
  - [ ] **Not exposed directly - accessed via AutoColony(mode='pso')**

- [ ] **Objective Function Handling**
  - [x] Support for callable Python functions
  - [x] Add built-in benchmark functions
  - [ ] Implement function evaluation caching
  - [ ] Add gradient-based acceleration (if available)
  - [ ] Support for noisy objective functions

### **3.3 Testing & Validation**
- [ ] **Comprehensive Test Suite**
  - [x] Test on standard benchmark functions
  - [x] Validate convergence behavior
  - [ ] Test parameter sensitivity
  - [ ] Benchmark against scipy.optimize
  - [x] Test with different problem dimensions

---

## **Phase 4: Artificial Bee Colony (ABC) Implementation (Sprint 4)**

### **4.1 Rust Core Implementation**
- [ ] **ABC Algorithm Structure**
  - [ ] Implement `BeeColonyCore` struct
  - [ ] Add `EmployedBee`, `OnlookerBee`, and `ScoutBee` structs
  - [ ] Implement food source representation
  - [ ] Add waggle dance mechanism for information sharing
  - [ ] Implement abandonment criteria
  - [ ] Add local search procedures

### **4.2 Python Interface**
- [ ] **ArtificialBeeColonyOptimizer Class (Internal)**
  - [ ] Inherit from `BaseOptimizer`
  - [ ] Parameters: `n_bees`, `n_iterations`, `limit`, `bounds`
  - [ ] Implement `fit(objective_function)` method
  - [ ] Add `predict()` method returning best solution
  - [ ] Implement `score()` method (negative objective value)
  - [ ] Add `transform()` method for solution representation
  - [ ] **Not exposed directly - accessed via AutoColony(mode='abc')**

---

## **Phase 5: Scikit-Learn Integration & Compatibility (Sprint 5)**

### **5.1 Estimator Compliance**
- [x] **Scikit-Learn Estimator Interface**
  - [x] Implement `sklearn.base.BaseEstimator` compliance
  - [x] Add `sklearn.base.OptimizerMixin` (custom mixin)
  - [x] Ensure parameter validation follows sklearn patterns
  - [x] Add `_check_is_fitted()` method
  - [x] Implement proper `__sklearn_tags__()` method

- [x] **Pipeline Integration**
  - [x] Test integration with `sklearn.pipeline.Pipeline`
  - [x] Add transformer interface for problem preprocessing
  - [x] Implement `sklearn.model_selection` compatibility
  - [x] Add support for `sklearn.compose` utilities

### **5.2 Cross-Validation & Model Selection**
- [x] **Custom Scoring Functions**
  - [x] Implement optimization-specific scoring metrics
  - [x] Add convergence rate scoring
  - [x] Create diversity metrics for population-based algorithms
  - [x] Add robustness scoring for noisy problems

- [ ] **Grid Search Integration**
  - [x] Ensure `GridSearchCV` compatibility with unified interface
  - [x] Add `RandomizedSearchCV` support for all modes
  - [x] Implement custom parameter distributions for each algorithm
  - [x] Add optimization-specific cross-validation strategies
  - [x] Support mode-specific parameter grids
  - [ ] **Example Grid Search Usage:**
    ```python
    from sklearn.model_selection import GridSearchCV
    
    # Grid search across algorithms and parameters
    param_grid = [
        {'mode': ['aco'], 'n_ants': [20, 50], 'alpha': [0.5, 1.0, 2.0]},
        {'mode': ['pso'], 'n_particles': [20, 50], 'w': [0.5, 0.9]},
        {'mode': ['abc'], 'n_bees': [20, 50], 'limit': [5, 10]}
    ]
    
    optimizer = AutoColony()
    grid_search = GridSearchCV(optimizer, param_grid, cv=5)
    grid_search.fit(X, y)
    ```

### **5.3 Utility Functions**
- [x] **Validation Utilities**
  - [x] Implement `check_optimization_problem()` function
  - [x] Add `check_bounds()` validation
  - [x] Create `check_objective_function()` utility
  - [x] Add dimension consistency checks

### **1.3 Unified Interface Design & Auto-Selection**
- [x] **Algorithm Auto-Selection Logic**
  - [x] Implement problem type detection (continuous vs discrete)
  - [x] Add dimensionality analysis for algorithm suitability
  - [x] Create heuristics for automatic algorithm selection
  - [x] Support user override of auto-selection
  - [x] Add performance-based algorithm recommendation

- [x] **Parameter Mapping & Validation**
  - [x] Create parameter mapping between unified and algorithm-specific interfaces
  - [x] Implement parameter conflict resolution
  - [x] Add parameter suggestion system for different algorithms
  - [x] Support algorithm-specific parameter validation
  - [x] Add parameter documentation and help system

- [ ] **Unified Interface Examples**
  ```python
  # Auto-select algorithm based on problem type
  optimizer = AutoColony(mode='auto', n_iterations=100)
  optimizer.fit(problem_data)
  
  # Explicit algorithm selection
  optimizer = AutoColony(mode='aco', n_ants=50, alpha=1.0, beta=2.0)
  optimizer.fit(distance_matrix)
  
  optimizer = AutoColony(mode='pso', n_particles=30, w=0.9, c1=2.0, c2=2.0)
  optimizer.fit(objective_function, bounds=bounds)
  
  optimizer = AutoColony(mode='abc', n_bees=50, limit=10)
  optimizer.fit(objective_function, bounds=bounds)
  ```

---

## **Phase 6: Metrics & Evaluation Framework (Sprint 6)**

### **6.1 Optimization Metrics**
- [x] **Performance Metrics**
  - [x] Implement convergence rate calculation
  - [x] Add success rate metrics
  - [x] Create optimization gap metrics
  - [x] Add computational efficiency metrics

- [x] **Statistical Analysis**
  - [x] Implement statistical significance tests
  - [x] Add distribution analysis for multiple runs
  - [ ] Create performance profiling tools
  - [x] Add robustness analysis utilities

### **6.2 Benchmark Suite**
- [ ] **Standard Benchmark Problems**
  - [ ] Implement CEC benchmark functions
  - [ ] Add TSPLIB problem instances
  - [x] Create custom benchmark suite
  - [ ] Add real-world optimization problems

- [ ] **Benchmarking Framework**
  - [x] Create automated benchmarking pipeline
  - [x] Add performance comparison tools
  - [x] Implement result visualization
  - [x] Add statistical reporting

---

## **Phase 7: Documentation & Examples (Sprint 7)**

### **7.1 API Documentation**
- [ ] **Sphinx Documentation**
  - [ ] Complete API reference with examples
  - [ ] Add user guide with scikit-learn style
  - [ ] Create developer documentation
  - [ ] Add troubleshooting guide

- [ ] **Docstring Standards**
  - [ ] Follow numpydoc style consistently
  - [ ] Add comprehensive parameter descriptions
  - [ ] Include usage examples in docstrings
  - [ ] Add cross-references to related methods

### **7.2 Tutorials & Examples**
- [ ] **Jupyter Notebook Tutorials**
  - [ ] Basic optimization tutorial
  - [ ] Advanced parameter tuning guide
  - [ ] Multi-objective optimization examples
  - [ ] Integration with other libraries

- [ ] **Example Gallery**
  - [ ] **Unified Interface Examples:**
         ```python
     # Example 1: Auto-selection for different problem types
     optimizer = AutoColony(mode='auto', n_iterations=100)
     
     # Discrete problem (TSP) - auto-selects ACO
     optimizer.fit(distance_matrix)
     tour = optimizer.predict()
     
     # Continuous problem - auto-selects PSO or ABC
     optimizer.fit(rosenbrock_function, bounds=bounds)
     solution = optimizer.predict()
     
     # Example 2: Explicit algorithm selection
     aco_optimizer = AutoColony(mode='aco', n_ants=50, alpha=1.0, beta=2.0)
     pso_optimizer = AutoColony(mode='pso', n_particles=30, w=0.9)
     abc_optimizer = AutoColony(mode='abc', n_bees=50, limit=10)
     
     # Example 3: Pipeline integration
     from sklearn.pipeline import Pipeline
     from sklearn.preprocessing import StandardScaler
     
     pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('optimizer', AutoColony(mode='pso', n_particles=30))
     ])
     pipeline.fit(X, y)
     ```
  - [ ] TSP solving with ACO via unified interface
  - [ ] Function optimization with PSO via unified interface
  - [ ] Engineering optimization problems using auto-selection
  - [ ] Machine learning hyperparameter tuning with grid search

---

## **Phase 8: Advanced Features (Sprint 8)**

### **8.1 Multi-Objective Optimization**
- [ ] **NSGA-II Implementation**
  - [ ] Implement non-dominated sorting
  - [ ] Add crowding distance calculation
  - [ ] Create Pareto front visualization
  - [ ] Add hypervolume indicator

### **8.2 Parallel & Distributed Computing**
- [ ] **Parallelization**
  - [ ] Implement thread-based parallelism in Rust
  - [ ] Add multiprocessing support in Python
  - [ ] Create distributed optimization framework
  - [ ] Add GPU acceleration (optional)

### **8.3 Visualization & Monitoring**
- [ ] **Real-time Monitoring**
  - [ ] Implement convergence plotting
  - [ ] Add population diversity visualization
  - [ ] Create interactive parameter tuning
  - [ ] Add 3D search space visualization

---

## **Phase 9: Testing & Quality Assurance (Sprint 9)**

### **9.1 Comprehensive Testing**
- [ ] **Unit Testing**
  - [ ] Achieve >95% code coverage
  - [ ] Add property-based testing
  - [ ] Implement parametric testing
  - [ ] Add regression testing

- [ ] **Integration Testing**
  - [ ] Test all algorithm combinations
  - [ ] Validate scikit-learn compatibility
  - [ ] Test with different Python versions
  - [ ] Add performance regression tests

### **9.2 Continuous Integration**
- [ ] **GitHub Actions Setup**
  - [ ] Multi-platform testing (Linux, macOS, Windows)
  - [ ] Multi-Python version testing (3.8-3.12)
  - [ ] Rust testing with multiple toolchains
  - [ ] Performance benchmarking in CI

- [ ] **Code Quality Gates**
  - [ ] Add automated code review
  - [ ] Implement security scanning
  - [ ] Add dependency vulnerability checks
  - [ ] Create automated documentation builds

---

## **Phase 10: Packaging & Distribution (Sprint 10)**

### **10.1 Package Preparation**
- [ ] **Wheel Building**
  - [ ] Configure `cibuildwheel` for cross-platform builds
  - [ ] Add ARM64 support for Apple Silicon
  - [ ] Create manylinux wheels
  - [ ] Add Windows wheel building

- [ ] **Package Metadata**
  - [ ] Complete `pyproject.toml` configuration
  - [ ] Add comprehensive package classifiers
  - [ ] Create detailed package description
  - [ ] Add proper license information

### **10.2 Release Process**
- [ ] **Release Automation**
  - [ ] Create automated release workflow
  - [ ] Add changelog generation
  - [ ] Implement semantic versioning
  - [ ] Add release notes template

- [ ] **Distribution**
  - [ ] Publish to TestPyPI
  - [ ] Create official PyPI release
  - [ ] Add conda-forge package
  - [ ] Create GitHub releases

---

## **Phase 11: Community & Maintenance (Sprint 11)**

### **11.1 Community Building**
- [ ] **Contributing Guidelines**
  - [ ] Create detailed CONTRIBUTING.md
  - [ ] Add code of conduct
  - [ ] Create issue templates
  - [ ] Add pull request templates

- [ ] **Community Features**
  - [ ] Set up discussion forums
  - [ ] Add contributor recognition
  - [ ] Create roadmap documentation
  - [ ] Add feature request process

### **11.2 Long-term Maintenance**
- [ ] **Sustainability**
  - [ ] Add deprecation policy
  - [ ] Create backward compatibility guidelines
  - [ ] Implement security update process
  - [ ] Add performance monitoring

---

## **Acceptance Criteria for Scikit-Learn Compatibility**

### **Must-Have Features**
- [x] **Unified Interface (`AutoColony`)** 
  - [x] Single class inherits from `BaseEstimator`
  - [x] Support `mode` parameter for algorithm selection: 'aco', 'pso', 'abc', 'auto'
  - [x] Automatic algorithm selection based on problem type
  - [x] Dynamic parameter routing to underlying algorithms
  - [x] Mode-specific parameter validation and constraints
- [x] **Scikit-Learn Compatibility**
  - [x] Implement `fit()`, `predict()`, `score()` methods
  - [x] Support `get_params()` and `set_params()` with mode-specific parameters
  - [x] Pass `check_estimator()` tests for all modes
  - [x] Work with `Pipeline` and `GridSearchCV`
  - [x] Follow sklearn naming conventions
  - [x] Implement proper parameter validation
  - [x] Add comprehensive docstrings
  - [x] Support pickle serialization
  - [x] Include reproducibility features (random_state)

### **Quality Metrics**
- [ ] >95% test coverage
- [ ] <100ms import time
- [ ] Memory usage <10x scipy.optimize
- [ ] Performance >2x pure Python implementations
- [ ] Support Python 3.8+ and major platforms
- [ ] Zero critical security vulnerabilities
- [ ] Documentation completeness >90%

---

## **Algorithm Backlog & Point Estimates**

> **Status:** ACO, PSO and ABC are implemented end-to-end (Rust core → pyo3
> bindings → `AutoColony` → tests). The items below are the next algorithms to
> add, using the same pattern.

**Point scale (Fibonacci, relative effort).** Each point ≈ one focused unit of
work. Every algorithm estimate already includes the full slice: Rust core
(`Optimizer` impl) + pyo3 binding + `AutoColony` wiring + Rust & pytest tests.

| Points | Meaning |
| ------ | ------------------------------------------------------ |
| 1 | Trivial, mechanical |
| 2 | Easy — a small self-contained update rule |
| 3 | Moderate — some fiddly bits or a new sub-mechanism |
| 5 | Involved — a real algorithm with several phases |
| 8 | Hard — nontrivial math or new infrastructure |
| 13 | Large — a whole subsystem across multiple files |

Continuous algorithms reuse the existing Python-callable objective bridge and
`Bounds`, so they are cheaper than the discrete or multi-objective work.

### **Continuous (reuse objective bridge) — total 35 pts**
- [x] **Grey Wolf Optimizer (GWO)** — 2 pts · popular, compact update rule
- [x] **Differential Evolution (DE)** — 2 pts · simple and genuinely strong; a good default
- [x] **Firefly (FA)** — 2 pts · roadmap algorithm, PSO-shaped
- [x] **Simulated Annealing (SA)** — 2 pts · single-solution, fits the interface cleanly
- [x] **Cuckoo Search (CS)** — 3 pts · Lévy-flight steps
- [x] **Bat Algorithm (BA)** — 3 pts · frequency/loudness/pulse-rate tuning
- [x] **Glowworm (GSO)** — 5 pts · roadmap algorithm, luciferin + dynamic neighborhoods
- [x] **Bacterial Foraging (BFO)** — 8 pts · roadmap algorithm, chemotaxis/reproduction/elimination
- [ ] **CMA-ES** — 8 pts · strongest continuous optimizer, real covariance-adaptation math

### **Discrete / combinatorial (moderate new work) — total 15 pts**
- [x] **2-opt local search** — 2 pts · cheap ACO tour-improvement hybrid
- [ ] **Binary/Discrete PSO** — 3 pts · reuses PSO structure
- [ ] **ACO variants** (Ant Colony System, Max-Min AS, Elitist) — 5 pts · extend existing `aco.rs`
- [ ] **Genetic Algorithm (GA) for permutations/TSP** — 5 pts · order-crossover + mutation on tours

### **Multi-objective (new infrastructure) — total 18 pts**
- [ ] **Pareto infrastructure** (non-dominated sorting, crowding distance, archive) — 8 pts · prerequisite
- [ ] **NSGA-II** — 5 pts · builds on the Pareto infrastructure
- [ ] **MOPSO** — 5 pts · multi-objective PSO on the same infrastructure

### **Notes on priority**
- **Best value first:** DE, GWO (continuous) and 2-opt (discrete) — high impact, low points.
- Several trendy "metaphor" algorithms (Whale, Moth-Flame, Salp Swarm, Grasshopper)
  are largely PSO/DE re-dressed; implementable at ~2 pts each but **low priority** —
  they add breadth without much real capability.
- **Grand total for everything above: 68 pts.** A sensible first milestone is the
  35-pt continuous set (it needs no new infrastructure).
