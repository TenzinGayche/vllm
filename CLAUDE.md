# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It's a complex Python project with C++/CUDA extensions, designed for production-scale LLM serving with advanced features like PagedAttention, quantization, and distributed inference.

## Development Commands

### Installation & Setup
```bash
# Install from source (recommended for development)
pip install -e .

# Install development dependencies
pip install -r requirements/dev.txt

# Install linting and formatting tools
pip install -r requirements/lint.txt

# Setup pre-commit hooks for automatic linting
pre-commit install
```

### Building
```bash
# Build the project with CMake (automatic during pip install -e .)
python setup.py build_ext --inplace

# Environment variables for build configuration:
# VLLM_TARGET_DEVICE=cuda|cpu|rocm|neuron|tpu|xpu
# MAX_JOBS=<number>  # Control parallel compilation
# NVCC_THREADS=<number>  # CUDA compilation threads
# CMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/basic_correctness/
pytest tests/distributed/ -v
pytest tests/models/ -k "test_model_name"

# Run tests with specific markers
pytest -m "not distributed"  # Skip distributed tests
pytest -m "core_model"       # Only core model tests

# Test with specific configurations
pytest tests/ --forked      # Run tests in separate processes
pytest tests/ --tb=short    # Shorter traceback format
```

### Linting & Formatting
```bash
# Modern linting is handled by pre-commit hooks
# Legacy format.sh now redirects to pre-commit setup

# Run specific linters manually
ruff check vllm/
ruff format vllm/
mypy vllm/

# Type checking with mypy
./tools/mypy.sh
```

### Benchmarking
```bash
# Throughput benchmarking
python benchmarks/benchmark_throughput.py

# Latency benchmarking  
python benchmarks/benchmark_latency.py

# Serving benchmarks
python benchmarks/benchmark_serving.py
```

## Architecture Overview

### Core Components

**Engine Layer (`vllm/engine/`)**
- `LLMEngine`: Main synchronous inference engine
- `AsyncLLMEngine`: Asynchronous wrapper for concurrent request handling
- `EngineArgs`/`AsyncEngineArgs`: Configuration classes for engine initialization

**Execution Layer (`vllm/executor/`)**
- `uniproc_executor.py`: Single-process execution
- `mp_distributed_executor.py`: Multi-process distributed execution
- `ray_distributed_executor.py`: Ray-based distributed execution

**Model Execution (`vllm/model_executor/`)**
- `models/`: Implementations for 100+ supported model architectures
- `layers/`: Custom layers including attention, MoE, quantization
- `model_loader/`: Weight loading strategies (default, sharded, quantized)

**Attention System (`vllm/attention/`)**
- `backends/`: Multiple attention implementations (FlashAttention, FlashInfer, etc.)
- `ops/`: Low-level attention kernels and operations
- PagedAttention implementation for memory efficiency

**Worker System (`vllm/worker/`)**
- `worker.py`: Main worker implementation for model execution
- `model_runner.py`: Manages model forward passes and batching
- `cache_engine.py`: KV cache management

**Entrypoints (`vllm/entrypoints/`)**
- `llm.py`: Simple Python API for inference
- `api_server.py`: FastAPI-based server
- `openai/`: OpenAI-compatible API implementation
- `cli/`: Command-line interfaces

### Key Architectural Patterns

**V1 vs Legacy Architecture**
- `vllm/v1/`: New modular architecture (alpha) with improved performance
- Legacy codebase: Current stable implementation
- V1 provides 1.7x speedup with cleaner separation of concerns

**Distributed Computing**
- Tensor parallelism for large models across GPUs
- Pipeline parallelism for memory efficiency
- Expert parallelism for MoE models
- Data parallelism for serving multiple requests

**Memory Management**
- PagedAttention: Variable-length attention with paged memory
- Block-based KV cache allocation and eviction
- CPU offloading for memory optimization

**Quantization Support**
- FP8, INT8, INT4 quantization schemes
- Multiple backends: GPTQ, AWQ, AutoRound, SmoothQuant
- Hardware-specific optimizations (CUTLASS, Marlin kernels)

## Model Support

**Supported Model Families**
- Transformer models: Llama, Mistral, Qwen, Gemma, etc.
- Mixture-of-Experts: Mixtral, DeepSeek-V2/V3, etc.
- Multimodal: LLaVA, CLIP, Qwen-VL, etc.
- Embedding models: E5-Mistral, BGE, etc.
- Specialized: Mamba (SSM), BERT variants

**Model Registration**
- Models auto-registered in `vllm/model_executor/models/`
- `registry.py` manages model discovery and instantiation
- Support for custom/OOT (out-of-tree) model implementations

## Configuration

**Key Configuration Files**
- `pyproject.toml`: Python package configuration, build settings, linting rules
- `requirements/`: Platform-specific dependency files (cuda.txt, cpu.txt, etc.)
- `vllm/config.py`: Core configuration classes

**Environment Variables**
- `VLLM_TARGET_DEVICE`: Target hardware platform
- `VLLM_USE_PRECOMPILED`: Use precompiled wheels vs source build
- `CUDA_HOME`, `ROCM_HOME`: Hardware SDK paths
- `MAX_JOBS`, `NVCC_THREADS`: Build parallelism control

## Testing Strategy

**Test Organization**
- `tests/basic_correctness/`: Core functionality validation
- `tests/distributed/`: Multi-GPU and multi-node testing
- `tests/models/`: Model-specific correctness tests
- `tests/quantization/`: Quantization method validation
- `tests/kernels/`: Low-level kernel testing

**Performance Testing**
- Accuracy regression tests against reference implementations
- Performance benchmarks for throughput and latency
- Memory usage validation

## Platform Support

**Hardware Platforms**
- NVIDIA GPUs (CUDA): Primary platform with full feature support
- AMD GPUs (ROCm): Growing support with optimized kernels  
- Intel CPUs/GPUs (XPU): CPU inference and Intel GPU support
- AWS Neuron: Support for Inferentia/Trainium chips
- Google TPU: TPU inference via JAX/Pallas
- Apple Silicon: CPU-only inference on macOS

**Build Targets**
Each platform has specific build requirements and optimized kernels managed through CMake and the setup.py build system.