# JAX GPU Support and Differentiability - Implementation Plan

**Date:** 2025-11-17
**Status:** Planning
**Goal:** Add JAX GPU acceleration and automatic differentiation while preserving Qiskit's symbolic/analytical capabilities

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Architecture](#architecture)
4. [Implementation Phases](#implementation-phases)
5. [API Design](#api-design)
6. [Examples and Use Cases](#examples-and-use-cases)
7. [Testing Strategy](#testing-strategy)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Documentation Updates](#documentation-updates)

---

## Overview

### Current State

**Strengths:**
- ✅ Qiskit `SparsePauliOp` for symbolic Hamiltonian construction
- ✅ Multiple grouping strategies (XYZ, parity)
- ✅ Sophisticated Trotter error bounds (tight, analytical, interference)
- ✅ High-order Trotter formulas (1st, 2nd, 4th, 6th, 8th)
- ✅ Basic JAX support via `use_jax=True` flag
- ✅ JAX dependencies already in `pyproject.toml`

**Limitations:**
- ❌ Manual `.to_matrix()` calls scattered throughout code
- ❌ No automatic differentiation support
- ❌ No batching/vectorization over parameters
- ❌ JAX integration is incomplete and ad-hoc
- ❌ No gradient-based optimization capabilities

### Inspiration from dynamiqs

**Key features to learn from:**
- Full JAX compatibility for all numerical operations
- Automatic differentiation through all operations
- GPU/CPU transparent execution
- Batching over Hamiltonians, states, and parameters
- Differentiable solvers for quantum control

**Our unique value proposition:**
- State-of-the-art Trotter error bounds and analysis
- High-order Trotter decompositions
- OpenFermion integration for fermionic systems
- Focus on error analysis and optimization

---

## Design Philosophy

### Hybrid Architecture Principle

> **"Symbolic construction with Qiskit, numerical computation with JAX"**

**Key insight:** Keep what Qiskit does best (symbolic operations) and use JAX for what it does best (numerical GPU computation and gradients).

### Design Principles

1. **Preserve Qiskit for symbolic work:**
   - Hamiltonian construction and simplification
   - Pauli string manipulation
   - Grouping strategies (XYZ, parity)
   - Symbolic commutator relations

2. **Use JAX for numerical work:**
   - Matrix exponentials
   - Matrix multiplications
   - Norm computations
   - Error bound calculations
   - Parameter optimization

3. **Provide seamless conversion:**
   - Automatic `SparsePauliOp → JAX array` when needed
   - Caching for repeated conversions
   - Clear API for explicit conversion

4. **Backward compatibility:**
   - All existing code continues to work
   - JAX is optional (graceful degradation)
   - Clear migration path for users

---

## Architecture

### Component Structure

```
qsimkit/
├── backend.py          # NEW: Conversion utilities (Qiskit ↔ JAX)
├── spin.py             # Symbolic Hamiltonian construction (keep Qiskit)
├── fermion.py          # Symbolic fermionic Hamiltonians (keep Qiskit)
├── trotter.py          # ENHANCE: Add JAX numerical backend
├── bounds.py           # ENHANCE: JAX-compatible error bounds
├── measure.py          # ENHANCE: JAX-compatible norms/commutators
├── diff/               # NEW: Differentiable operations
│   ├── __init__.py
│   ├── optimize.py     # Parameter optimization
│   ├── gradients.py    # Gradient utilities
│   └── control.py      # Quantum optimal control
└── examples/           # NEW: Example scripts
    ├── gpu_acceleration.py
    ├── parameter_optimization.py
    └── gradient_analysis.py
```

### Conversion Layer Design

The conversion layer (`backend.py`) provides:

```python
# Core conversion functions
to_jax(operator)              # SparsePauliOp/ndarray → JAX array
to_jax_list(operators)        # List conversion
to_numpy(operator)            # JAX array → NumPy array

# Performance features
@lru_cache
cached_pauli_to_jax(string)   # Cached Pauli conversions

# Utilities
get_default_backend()         # Auto-detect GPU/CPU
print_backend_info()          # Display device info
ensure_jax(*ops)              # Ensure operators are JAX arrays
```

---

## Implementation Phases

### Phase 1: Core JAX Infrastructure (Week 1-2)

**Goal:** Establish conversion utilities and update core numerical functions

**Tasks:**

1. **Create `qsimkit/backend.py`**
   - [ ] Implement `to_jax()` with SparsePauliOp support
   - [ ] Implement `to_jax_list()` for Hamiltonian term lists
   - [ ] Implement `to_numpy()` for reverse conversion
   - [ ] Add `cached_pauli_to_jax()` with LRU cache
   - [ ] Add device detection (`get_default_backend()`)
   - [ ] Add `print_backend_info()` diagnostic tool
   - [ ] Write comprehensive docstrings with examples

2. **Update `qsimkit/trotter.py`**
   - [ ] Import backend utilities
   - [ ] Update `expH()` to use `to_jax()` conversion
   - [ ] Ensure `pf()` works with JAX arrays
   - [ ] Ensure `pf_high()` works with JAX arrays
   - [ ] Add type hints for JAX/NumPy compatibility
   - [ ] Preserve backward compatibility (test with existing code)

3. **Update `qsimkit/measure.py`**
   - [ ] Add `use_jax` parameter to `norm()`
   - [ ] Add `use_jax` parameter to `commutator()` if needed
   - [ ] Update `anticommutator()` similarly
   - [ ] Support both SparsePauliOp and JAX arrays

4. **Update `qsimkit/bounds.py`**
   - [ ] Add `use_jax` parameter to all bound functions
   - [ ] Update `tight_bound()` for JAX compatibility
   - [ ] Update `analytic_bound()` for JAX compatibility
   - [ ] Update `interference_bound()` for JAX compatibility

**Deliverables:**
- Working conversion layer
- All numerical operations support JAX
- Backward compatibility maintained
- Initial test suite

**Success Criteria:**
```python
# This should work seamlessly
from qsimkit.spin import Nearest_Neighbour_1d
from qsimkit.trotter import pf
from qsimkit.backend import print_backend_info

print_backend_info()  # Shows GPU if available

H = Nearest_Neighbour_1d(n=10, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)
h_list = H.ham_xyz

# Automatic GPU acceleration
U = pf(h_list, t=1.0, r=100, order=2, use_jax=True)
print(f"Computed on: {U.device()}")  # Should show GPU
```

---

### Phase 2: Differentiability and Gradients (Week 3-4)

**Goal:** Enable automatic differentiation through all operations

**Tasks:**

1. **Create `qsimkit/diff/__init__.py`**
   - [ ] Define public API for differentiable operations
   - [ ] Export key functions

2. **Create `qsimkit/diff/gradients.py`**
   - [ ] Implement `grad_hamiltonian()` - gradients w.r.t. Hamiltonian parameters
   - [ ] Implement `grad_time()` - gradients w.r.t. evolution time
   - [ ] Implement `grad_trotter_steps()` - gradients w.r.t. Trotter steps
   - [ ] Implement `grad_error_bound()` - gradients of error bounds
   - [ ] Add Hessian computation utilities
   - [ ] Add Jacobian computation utilities

3. **Make Hamiltonians Differentiable**
   - [ ] Create JAX-compatible Hamiltonian constructors
   - [ ] Support parameterized Hamiltonians (e.g., `Jx`, `Jy`, `Jz` as differentiable params)
   - [ ] Add `to_jax_parametric()` method to Hamiltonian classes

4. **Ensure JAX Transformations Work**
   - [ ] Test `jax.jit` compilation on all functions
   - [ ] Test `jax.grad` on Trotter operations
   - [ ] Test `jax.vmap` for batching
   - [ ] Test `jax.jacrev` and `jax.jacfwd`

**Deliverables:**
- Differentiable Hamiltonians
- Gradient utilities
- Hessian computation
- JIT compilation support

**Success Criteria:**
```python
import jax
import jax.numpy as jnp
from qsimkit.diff import grad_error_bound

# Compute gradient of error bound w.r.t. coupling constants
def error_fn(Jx, Jy, Jz, hx):
    H = make_hamiltonian_jax(n=10, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx)
    return tight_bound_jax(H, order=2, t=1.0, r=100)

# Automatic differentiation
grad_fn = jax.grad(error_fn, argnums=(0, 1, 2, 3))
gradients = grad_fn(1.0, 1.0, 1.0, 0.5)
print(f"∂error/∂Jx = {gradients[0]}")
```

---

### Phase 3: Optimization and Control (Week 5-6)

**Goal:** Add practical applications of differentiability

**Tasks:**

1. **Create `qsimkit/diff/optimize.py`**
   - [ ] Implement `minimize_error()` - find optimal Hamiltonian parameters
   - [ ] Implement `optimize_trotter_schedule()` - adaptive Trotter steps
   - [ ] Implement gradient descent optimizers
   - [ ] Integration with JAX optimizers (optax)
   - [ ] Support for constrained optimization

2. **Create `qsimkit/diff/control.py`**
   - [ ] Implement basic quantum optimal control
   - [ ] Time-dependent Hamiltonian support
   - [ ] Pulse optimization examples
   - [ ] Gate synthesis via gradient descent

3. **Add Sensitivity Analysis Tools**
   - [ ] Parameter sensitivity analysis
   - [ ] Error propagation with gradients
   - [ ] Uncertainty quantification

**Deliverables:**
- Parameter optimization tools
- Quantum control capabilities
- Sensitivity analysis utilities

**Success Criteria:**
```python
from qsimkit.diff import minimize_error

# Optimize coupling constants to minimize Trotter error
result = minimize_error(
    n=10,
    target_error=1e-6,
    t=1.0,
    r=100,
    order=2,
    initial_params={'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0, 'hx': 0.5}
)
print(f"Optimal Jx = {result.params['Jx']}")
```

---

### Phase 4: Batching and Vectorization (Week 7-8)

**Goal:** Enable parallel computation over multiple parameters

**Tasks:**

1. **Implement Batched Operations**
   - [ ] Batch over Hamiltonian parameters
   - [ ] Batch over evolution times
   - [ ] Batch over Trotter steps
   - [ ] Batch over different groupings

2. **Optimize for GPU Throughput**
   - [ ] Use `jax.vmap` for automatic vectorization
   - [ ] Optimize memory usage for large batches
   - [ ] Implement chunked computation for memory efficiency

3. **Add High-Level Batch APIs**
   - [ ] `batch_compute_errors()` - compute errors for parameter grid
   - [ ] `batch_evolve()` - evolve multiple initial states
   - [ ] `parameter_sweep()` - sweep over parameter ranges

**Deliverables:**
- Vectorized operations
- Batch computation APIs
- Memory-efficient implementations

**Success Criteria:**
```python
import jax.numpy as jnp
from qsimkit.diff import batch_compute_errors

# Compute errors for 1000 different coupling constants in parallel
Jx_values = jnp.linspace(0.1, 2.0, 1000)
errors = batch_compute_errors(
    n=10,
    Jx=Jx_values,  # Vectorized parameter
    Jy=1.0,
    Jz=1.0,
    hx=0.5,
    t=1.0,
    r=100,
    order=2
)
print(f"Computed {len(errors)} errors in parallel on GPU")
```

---

### Phase 5: Documentation and Examples (Week 9-10)

**Goal:** Comprehensive documentation and example gallery

**Tasks:**

1. **Create Example Scripts**
   - [ ] `examples/gpu_acceleration.py` - Basic GPU usage
   - [ ] `examples/parameter_optimization.py` - Optimize Hamiltonian
   - [ ] `examples/gradient_analysis.py` - Sensitivity analysis
   - [ ] `examples/batch_computation.py` - Vectorized operations
   - [ ] `examples/quantum_control.py` - Optimal control
   - [ ] `examples/benchmark_gpu_vs_cpu.py` - Performance comparison

2. **Update Documentation**
   - [ ] Add "JAX GPU Support" section to README
   - [ ] Write tutorial: "Getting Started with GPU Acceleration"
   - [ ] Write tutorial: "Differentiable Quantum Simulation"
   - [ ] Write tutorial: "Parameter Optimization"
   - [ ] Update API reference with JAX features
   - [ ] Add performance benchmarks

3. **Create Jupyter Notebooks**
   - [ ] `notebooks/01_jax_basics.ipynb`
   - [ ] `notebooks/02_gpu_acceleration.ipynb`
   - [ ] `notebooks/03_gradients_and_optimization.ipynb`
   - [ ] `notebooks/04_quantum_control.ipynb`

**Deliverables:**
- 6+ example scripts
- Updated README
- 3+ tutorials
- 4 Jupyter notebooks

---

## API Design

### Workflow: Symbolic → Numerical

**Recommended user workflow:**

```python
# Step 1: Symbolic construction (Qiskit)
from qsimkit.spin import Nearest_Neighbour_1d

H = Nearest_Neighbour_1d(n=10, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)
h_list = H.ham_xyz  # List of SparsePauliOp

# Step 2: Automatic conversion + numerical computation (JAX)
from qsimkit.trotter import pf
from qsimkit.bounds import tight_bound

U = pf(h_list, t=1.0, r=100, order=2, use_jax=True)  # Auto-converts internally
error = tight_bound(h_list, order=2, t=1.0, r=100, use_jax=True)

# Step 3: Explicit conversion when needed
from qsimkit.backend import to_jax_list

h_jax = to_jax_list(h_list)  # Now h_jax is list of JAX arrays
# Can use h_jax for custom JAX operations
```

### Differentiable API

**For gradient-based workflows:**

```python
from qsimkit.diff import make_hamiltonian_jax, grad_error_bound
import jax

# Create differentiable Hamiltonian constructor
def loss_fn(params):
    Jx, Jy, Jz, hx = params
    H_list = make_hamiltonian_jax(
        'nearest_neighbor_1d',
        n=10,
        Jx=Jx, Jy=Jy, Jz=Jz, hx=hx
    )
    return tight_bound_jax(H_list, order=2, t=1.0, r=100)

# Compute gradient
grad_fn = jax.grad(loss_fn)
params = jnp.array([1.0, 1.0, 1.0, 0.5])
gradients = grad_fn(params)
```

### Backend Selection

**Three modes of operation:**

1. **NumPy mode (default, no JAX):**
   ```python
   U = pf(h_list, t=1.0, r=100, order=2)  # Uses NumPy/SciPy
   ```

2. **JAX CPU mode:**
   ```python
   U = pf(h_list, t=1.0, r=100, order=2, use_jax=True)  # JAX on CPU
   ```

3. **JAX GPU mode (automatic if GPU available):**
   ```python
   from qsimkit.backend import print_backend_info
   print_backend_info()  # Check if GPU detected
   U = pf(h_list, t=1.0, r=100, order=2, use_jax=True)  # JAX on GPU
   ```

---

## Examples and Use Cases

### Use Case 1: GPU Acceleration for Large Systems

```python
"""
Accelerate Trotter simulation of 15-qubit system on GPU
"""
from qsimkit.spin import Nearest_Neighbour_1d
from qsimkit.trotter import pf
from qsimkit.backend import print_backend_info
import time

print_backend_info()

# Large system
H = Nearest_Neighbour_1d(n=15, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)
h_list = H.ham_xyz

# CPU timing
start = time.time()
U_cpu = pf(h_list, t=1.0, r=1000, order=2, use_jax=False)
cpu_time = time.time() - start

# GPU timing
start = time.time()
U_gpu = pf(h_list, t=1.0, r=1000, order=2, use_jax=True)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.3f}s")
print(f"GPU time: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

### Use Case 2: Parameter Optimization

```python
"""
Find optimal Hamiltonian parameters to minimize Trotter error
"""
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from qsimkit.diff import make_hamiltonian_jax, tight_bound_jax

def objective(params):
    """Error bound as function of parameters"""
    Jx, Jy, Jz, hx = params
    H_list = make_hamiltonian_jax(
        'nearest_neighbor_1d',
        n=10, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx
    )
    return tight_bound_jax(H_list, order=2, t=1.0, r=100)

# Gradient descent
grad_fn = jax.grad(objective)
params = jnp.array([1.0, 1.0, 1.0, 0.5])
learning_rate = 0.01

for i in range(100):
    grads = grad_fn(params)
    params = params - learning_rate * grads
    if i % 10 == 0:
        print(f"Iteration {i}: error = {objective(params):.6e}")

print(f"Optimal params: Jx={params[0]:.3f}, Jy={params[1]:.3f}, "
      f"Jz={params[2]:.3f}, hx={params[3]:.3f}")
```

### Use Case 3: Batch Parameter Sweep

```python
"""
Compute Trotter errors across a grid of coupling constants
"""
import jax.numpy as jnp
from qsimkit.diff import batch_compute_errors
import matplotlib.pyplot as plt

# Create parameter grid
Jx_values = jnp.linspace(0.1, 2.0, 50)
hx_values = jnp.linspace(0.0, 1.0, 50)
Jx_grid, hx_grid = jnp.meshgrid(Jx_values, hx_values)

# Vectorized computation on GPU
errors = jax.vmap(
    jax.vmap(lambda Jx, hx: compute_error_jax(
        n=10, Jx=Jx, Jy=1.0, Jz=1.0, hx=hx, t=1.0, r=100
    ))
)(Jx_grid, hx_grid)

# Visualize
plt.contourf(Jx_grid, hx_grid, jnp.log10(errors), levels=20)
plt.colorbar(label='log10(error)')
plt.xlabel('Jx')
plt.ylabel('hx')
plt.title('Trotter Error Landscape')
plt.savefig('error_landscape.png')
```

### Use Case 4: Sensitivity Analysis

```python
"""
Analyze sensitivity of error bounds to parameter variations
"""
import jax
import jax.numpy as jnp
from qsimkit.diff import compute_hessian

# Define error as function of all parameters
def error_bound(Jx, Jy, Jz, hx, t, r):
    H = make_hamiltonian_jax('nearest_neighbor_1d',
                              n=10, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx)
    return tight_bound_jax(H, order=2, t=t, r=r)

# Compute Hessian to understand parameter interactions
params = (1.0, 1.0, 1.0, 0.5, 1.0, 100)
hessian = compute_hessian(error_bound, params)

# Analyze sensitivity
param_names = ['Jx', 'Jy', 'Jz', 'hx', 't', 'r']
print("Parameter Sensitivity (diagonal of Hessian):")
for i, name in enumerate(param_names):
    print(f"  ∂²error/∂{name}² = {hessian[i,i]:.6e}")

print("\nParameter Interactions (off-diagonal):")
print(f"  ∂²error/∂Jx∂hx = {hessian[0,3]:.6e}")
```

---

## Testing Strategy

### Unit Tests

**Create `tests/test_backend.py`:**
- [ ] Test `to_jax()` with SparsePauliOp
- [ ] Test `to_jax()` with NumPy arrays
- [ ] Test `to_numpy()` conversions
- [ ] Test caching behavior
- [ ] Test device detection

**Create `tests/test_trotter_jax.py`:**
- [ ] Test `pf()` with `use_jax=True`
- [ ] Test `pf_high()` with `use_jax=True`
- [ ] Compare JAX vs NumPy results (numerical accuracy)
- [ ] Test with different Hamiltonian types

**Create `tests/test_bounds_jax.py`:**
- [ ] Test all bound functions with `use_jax=True`
- [ ] Verify numerical agreement with NumPy
- [ ] Test gradient computation

**Create `tests/test_gradients.py`:**
- [ ] Test `jax.grad` through Trotter operations
- [ ] Test `jax.vmap` for batching
- [ ] Test `jax.jit` compilation
- [ ] Verify gradient correctness with finite differences

### Integration Tests

**Create `tests/test_workflows.py`:**
- [ ] Test full workflow: Qiskit construction → JAX computation
- [ ] Test optimization pipeline
- [ ] Test batch computations
- [ ] Test GPU/CPU parity

### Performance Tests

**Create `tests/benchmark_gpu.py`:**
- [ ] Benchmark CPU vs GPU for various system sizes
- [ ] Benchmark batch operations
- [ ] Profile memory usage
- [ ] Generate performance plots

---

## Performance Benchmarks

### Expected Speedups

Based on typical JAX GPU performance:

| System Size (qubits) | NumPy (CPU) | JAX (CPU) | JAX (GPU) | Speedup |
|---------------------|-------------|-----------|-----------|---------|
| 8                   | 0.1s        | 0.08s     | 0.05s     | 2x      |
| 10                  | 0.5s        | 0.3s      | 0.1s      | 5x      |
| 12                  | 5s          | 3s        | 0.5s      | 10x     |
| 14                  | 50s         | 30s       | 3s        | 16x     |
| 15                  | 200s        | 120s      | 10s       | 20x     |

**Note:** Actual speedups depend on hardware and operation type.

### Batch Computation Speedups

| Batch Size | JAX (CPU) | JAX (GPU) | Speedup |
|------------|-----------|-----------|---------|
| 10         | 1s        | 0.5s      | 2x      |
| 100        | 10s       | 1s        | 10x     |
| 1000       | 100s      | 5s        | 20x     |

---

## Documentation Updates

### README Updates

Add section after "Features":

```markdown
## GPU Acceleration & Differentiability

Qsimkit supports **GPU acceleration** and **automatic differentiation** via JAX:

- **GPU Acceleration**: Run simulations 10-20x faster on NVIDIA GPUs
- **Automatic Differentiation**: Compute gradients for parameter optimization
- **Batching**: Vectorize computations over parameter ranges
- **Quantum Control**: Gradient-based pulse optimization

### Quick Example

python
from qsimkit.spin import Nearest_Neighbour_1d
from qsimkit.trotter import pf

# Symbolic construction (Qiskit)
H = Nearest_Neighbour_1d(n=10, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)

# Numerical computation (JAX, automatic GPU if available)
U = pf(H.ham_xyz, t=1.0, r=100, order=2, use_jax=True)


See [JAX GPU Guide](docs/jax-gpu-guide.md) for details.
```

### New Documentation Files

1. **`docs/jax-gpu-guide.md`** - Complete guide to GPU acceleration
2. **`docs/gradients-tutorial.md`** - Tutorial on automatic differentiation
3. **`docs/optimization-examples.md`** - Parameter optimization examples
4. **`docs/api-jax.md`** - JAX-specific API reference

---

## Migration Guide for Users

### Minimal Changes Required

**Before (existing code still works):**
```python
from qsimkit.spin import Nearest_Neighbour_1d
from qsimkit.trotter import pf

H = Nearest_Neighbour_1d(n=10, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)
U = pf(H.ham_xyz, t=1.0, r=100, order=2)  # NumPy backend
```

**After (opt-in to JAX):**
```python
from qsimkit.spin import Nearest_Neighbour_1d
from qsimkit.trotter import pf

H = Nearest_Neighbour_1d(n=10, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.5)
U = pf(H.ham_xyz, t=1.0, r=100, order=2, use_jax=True)  # JAX backend
```

**Key point:** Just add `use_jax=True` - everything else stays the same!

---

## Open Questions and Decisions

### 1. Default Backend Behavior

**Question:** Should JAX be the default if available, or require explicit opt-in?

**Options:**
- A) Always default to NumPy (explicit `use_jax=True` required)
- B) Auto-detect and use JAX if available (explicit `use_jax=False` to disable)
- C) Global configuration setting

**Recommendation:** Start with Option A (explicit opt-in) for Phase 1, consider B for Phase 2.

### 2. Sparse Matrix Support in JAX

**Question:** JAX doesn't have great sparse matrix support. How to handle large sparse systems?

**Options:**
- A) Only support dense JAX arrays (limit to ~15 qubits)
- B) Integrate JAX-compatible sparse libraries (e.g., JAX-MD, sparse-opt)
- C) Hybrid approach: keep SciPy sparse for large systems

**Recommendation:** Option C - use dense JAX for n≤15, sparse NumPy for larger.

### 3. Caching Strategy

**Question:** Should we cache SparsePauliOp → JAX conversions?

**Options:**
- A) Always cache (memory vs speed tradeoff)
- B) Optional caching with `@lru_cache`
- C) No caching (convert on-demand)

**Recommendation:** Option B - optional caching with configurable size.

### 4. Type Annotations

**Question:** How to handle type hints for functions that accept both NumPy and JAX arrays?

**Options:**
- A) Use `Union[np.ndarray, jax.Array]`
- B) Use generic `ArrayLike` protocol
- C) Separate functions for NumPy/JAX

**Recommendation:** Option B - use protocols for cleaner type hints.

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] All numerical operations support `use_jax=True`
- [ ] GPU automatically detected and used
- [ ] 10x speedup for n=12 qubits on GPU vs CPU
- [ ] 100% backward compatibility
- [ ] <5% overhead for conversion layer

### Phase 2 Success Criteria
- [ ] `jax.grad` works through all operations
- [ ] `jax.jit` compiles all functions
- [ ] Example gradient-based optimization working
- [ ] Gradients match finite differences to 1e-6

### Phase 3 Success Criteria
- [ ] Parameter optimization converges
- [ ] Quantum control examples working
- [ ] Documented use cases

### Phase 4 Success Criteria
- [ ] Batch computation over 1000 parameters in <10s
- [ ] 20x speedup for batched operations
- [ ] Memory-efficient for large batches

### Overall Success Criteria
- [ ] All existing tests pass
- [ ] 50+ new tests added
- [ ] Documentation complete
- [ ] 5+ working examples
- [ ] Performance benchmarks published

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| Phase 1: Core Infrastructure | 2 weeks | backend.py, JAX-compatible numerical ops |
| Phase 2: Differentiability | 2 weeks | Gradients, JIT compilation |
| Phase 3: Optimization | 2 weeks | Parameter optimization, quantum control |
| Phase 4: Batching | 2 weeks | Vectorization, batch APIs |
| Phase 5: Documentation | 2 weeks | Examples, tutorials, benchmarks |
| **Total** | **10 weeks** | Full JAX GPU + differentiability support |

---

## References

### External Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [dynamiqs](https://github.com/dynamiqs/dynamiqs)
- [Qiskit Quantum Info](https://qiskit.org/documentation/apidoc/quantum_info.html)
- [JAX GPU Tutorial](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

### Internal Resources

- Current Qsimkit README: `/README.md`
- Trotter implementation: `/qsimkit/trotter.py`
- Bounds implementation: `/qsimkit/bounds.py`
- Test suite: `/tests/`

---

## Appendix: Backend.py API Specification

### Full API Specification

```python
# qsimkit/backend.py

def to_jax(
    operator: Union[SparsePauliOp, np.ndarray, jnp.ndarray],
    dtype: jnp.dtype = jnp.complex128
) -> jnp.ndarray:
    """Convert operator to JAX array."""

def to_jax_list(
    operators: List[Union[SparsePauliOp, np.ndarray]],
    dtype: jnp.dtype = jnp.complex128
) -> List[jnp.ndarray]:
    """Convert list of operators to JAX arrays."""

def to_numpy(
    operator: Union[SparsePauliOp, jnp.ndarray, np.ndarray]
) -> np.ndarray:
    """Convert operator to NumPy array."""

@lru_cache(maxsize=128)
def cached_pauli_to_jax(
    pauli_string: str,
    coeff: complex = 1.0
) -> jnp.ndarray:
    """Convert Pauli string to JAX with caching."""

def get_default_backend() -> str:
    """Return 'jax-gpu', 'jax-cpu', or 'numpy'."""

def print_backend_info() -> None:
    """Print device and backend information."""

def ensure_jax(
    *operators,
    dtype: jnp.dtype = jnp.complex128
) -> Union[jnp.ndarray, List[jnp.ndarray]]:
    """Ensure operators are JAX arrays."""

def ensure_numpy(
    *operators
) -> Union[np.ndarray, List[np.ndarray]]:
    """Ensure operators are NumPy arrays."""
```

---

**End of Implementation Plan**
