# FPGN

**Ultra-Fast Programmable Gate-based Neural Acceleration with Differentiable LUTs**

FPGN is a research prototype for mapping LUT-based neural networks to FPGA hardware. It provides a training implementation built with PyTorch and a compiler that searches for a resource/latency-efficient execution schedule before emitting synthesizable Verilog templates.

## Pipeline

```text
PyTorch model
    │  train differentiable LUT layers
    ▼
trained checkpoint
    │  trace LUT weights and layer metadata
    ▼
model description JSON
    │  optimize W/H parallelism under LUT and FF limits
    ▼
optimized execution JSON
    │  instantiate RTL templates
    ▼
generated Verilog
```

The repository contains three main pieces:

- **Training** — differentiable 6-input LUT layers, LUT convolution/fully-connected layers, quantization, and a CIFAR-10 training entry point.
- **Compiler** — JSON parsing, FPGA resource and latency estimation, optimization with CVXPY/Gurobi, and RTL generation.
- **Pruning/support code** — LUT input-sensitivity/pruning utilities and related experiments.

## Repository layout

```text
training/
  lut_layer.py       Differentiable LUT operators
  train.py           CIFAR-10 distributed training entry point
  setup.py           CUDA extension build configuration

compiler/src/
  json_save_model.py Export a traced model to JSON
  json_parser.py     Parse and enrich model JSON
  solver.py          Resource/latency optimization
  compiler.py        Optimize a model and generate RTL
  all.py             Configuration evaluation utilities

compiler/template/   Verilog generator templates and RTL helpers
```

## Requirements

The code targets Linux with an NVIDIA GPU. The versions used by the project are:

- Python 3.10
- PyTorch 2.2.1
- torchvision 0.17.1
- CVXPY 1.5.2
- Gurobi (required by `compiler/src/solver.py`)

Training uses CUDA distributed processing (`nccl`) and starts one worker per visible GPU. A CUDA-capable PyTorch installation and at least one visible GPU are therefore required for the current `train.py` entry point. CIFAR-10 is downloaded automatically to `../data` on the first run.

Gurobi must be installed and licensed separately. The solver explicitly requests the Gurobi backend through CVXPY; installing CVXPY alone is not sufficient.

## Training

From the repository root, install the Python dependencies in your environment and run:

```bash
cd training
python train.py --epochs 200 --batch-size 200 --test-batch-size 200
```

Useful options include:

```bash
python train.py --dry-run
python train.py --epochs 200 --lr 1.0 --gamma 0.5 --seed 1
```

The script uses all GPUs returned by `torch.cuda.device_count()` and trains with PyTorch distributed data parallelism. It loads `base_3layer_small_quantbn.pt` as an initial checkpoint and periodically writes the best checkpoint as `base_3layer_small_quantbn_nores.pt` from rank 0. These checkpoint files are not part of this repository and must be supplied separately, or the checkpoint-loading code must be adapted for training from scratch.

Before running the training entry point, verify that the model-specific imports and optional CUDA extension expected by your checkout are available. In particular, `train.py` imports `resnet`, while `setup.py` expects a `lut6.cu` source file; neither file is currently present in this repository snapshot.

## Export a trained model

`compiler/src/json_save_model.py` provides `save_model_info_to_json(model, model_name, input_tensor, file_path)`. It registers forward hooks, executes one example input, and records the LUT weights and layer metadata in execution order.

The exported JSON should contain at least the model name and a `layers` array. The parser enriches this description with hardware-related fields such as LUT size, bit width, tensor dimensions, register estimates, and layer counts.

Because the exporter imports project-specific model modules and is not currently a standalone CLI, use it from a small project script after loading your trained model. Conceptually:

```python
from compiler.src.json_save_model import save_model_info_to_json

# model = ...       # load the trained FPGN model
# sample = ...      # one input tensor with the model's expected shape
save_model_info_to_json(model, "my_model", sample, "model_execution_info.json")
```

Use an input tensor with the same shape and preprocessing as the deployed model. The forward pass is required because spatial dimensions and execution order are inferred from runtime tensors.

## Optimize and generate RTL

The current compiler entry point is configured in `compiler/src/compiler.py`. Set `json_path` to the exported model description and update the target FPGA resource limits:

```python
limit = {
    "LUT": 2414378,
    "FF": 5000000,
}
json_path = "model_execution_info.json"
```

Then run it from `compiler/src` so that the relative input and output paths resolve as expected:

```bash
cd compiler/src
python compiler.py
```

The compiler will:

1. parse and normalize the model JSON;
2. search valid per-layer parallelism configurations;
3. solve the resource-constrained latency problem with Gurobi;
4. write the optimized configuration to `model_execution_info_optimized.json`;
5. recreate `./verilog/` and emit the generated Verilog modules there.

The generated RTL is a set of Verilog modules, not a complete vendor project. Create your own Vivado (or other FPGA toolchain) project, add the generated files and the required top-level constraints, and validate synthesis/timing for the target device.

## Configuration and optimization notes

- `W` and `H` describe the spatial/parallel execution configuration searched by the compiler.
- `LUT` and `FF` limits are user-supplied target-device constraints; replace the example values with the resources available on your FPGA.
- `force_no_packing` is supported by the parser/solver when packing must be disabled.
- `heuristic_h_search_best()` performs a layer-wise heuristic search over `H` configurations and invokes the solver for each candidate.
- `exhaustive_pareto_plot()` can be used to explore latency/resource trade-offs and generate a Pareto plot.

The JSON is the interface between training and compilation. If the network topology, quantization scheme, or target device changes, regenerate the JSON and review the resource model before generating RTL.

## Reproducibility and limitations

This is research code rather than a packaged application. Paths, checkpoint names, model imports, compiler limits, and some model-specific assumptions are currently hard-coded in the entry points. Results can also depend on the installed PyTorch/CUDA/Gurobi versions and on the target FPGA timing/resource model.

For a reproducible experiment, record:

- the checkpoint and model definition used for export;
- the input tensor shape and preprocessing;
- the JSON model description;
- FPGA LUT/FF limits and solver settings;
- Python, PyTorch, CUDA, CVXPY, and Gurobi versions.

## License and citation

No license or citation entry is currently included in this repository. Add the project’s intended license and publication citation here before redistributing the code.
