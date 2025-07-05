
## Algorithms

| Order | Algorithm | Type       | Use Case                     | Notes                 |
| ----- | --------- | ---------- | ---------------------------- | --------------------- |
| 1     | ACO       | Discrete   | TSP, routing, scheduling     | Good starter          |
| 2     | PSO       | Continuous | Function optimization        | Widely used           |
| 3     | ABC       | Continuous | Optimization, feature select | Slightly more complex |
| 4     | Firefly   | Continuous | Multimodal optimization      | Easy to add after PSO |
| 5     | Glowworm  | Continuous | Multi-solution discovery     | Experimental          |
| 6     | Bacterial | Mixed      | Bio-inspired optimization    | Complex               |


## Directory

```

colonyx/
├── colonyx/          # Python package directory 
│   └── __init__.py   # Python module initialization
├── src/              # Rust core implementation
│   └── lib.rs        # Rust library code
├── Cargo.toml        # Rust package configuration
├── pyproject.toml    # Python package configuration
└── docs/
    └── TODO.md       # Your detailed TODO list

```