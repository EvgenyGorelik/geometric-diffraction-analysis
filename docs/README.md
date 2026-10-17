# geometric-diffraction-analysis

Create datastructure:
```
./run.sh split --data_path <path/to/data> --labels 2DZone 3DLaueIntersections MultipleCrystals --output_folder data
```

Training:
```
./run.sh train --root_dir data --model_path models/diff_fibonacci_model.pth --use_class_weights
```


Inference:
```
./run.sh infer models/diff_fibonacci_model.pth ./data --output results.json
```


## Preprocessing Experimental Data

The data should look close to synthetic data, so we need to make it look synthetic.

First we need some statistics on the synthetic data. They should look something like:
```
{
    "img_width": 812,
    "img_height": 781,
    "img_max": 255.0,
    "img_min": 0.0,
    "num_dots": 88,
    "dot_radius_median": 3.1915382432114616,
    "dot_radius_max": 5.641895835477563
}
```

We can create a json file using the following command:
```
./run.sh stats <path/to/synthetic/img>
```

Then we use this file for converting all experimental data to synthetic using
```
./run.sh synthetic ./data img_stats.json --output_folder ./data_processed --threshold 0.01 --nms_size 20
```



## Experimental Data Simulator

Simulate experiment data using the `simulation/simulate_experiment.py` script. It will automatically create simulated experiment data.


## Segmentation Data Preparation

Simulate experiment multiple times (e.g. 10):
```
count=10 for i in $(seq 1 ${count}); do ./run.sh simulate-random --output_folder data; done
```

Prepare data for training:
```
./run.sh prepare-segmentation --data_path data/volume_500/ --output_path data/segmentation
```

Train segmentation
```
./run.sh train-segmentation --num_epochs 8 > segmentation_"$(date +%s)".log
```

## Configured rotation experiments

Install the lightweight simulator dependencies (no trained models or PyTorch needed):

```bash
python -m pip install -r requirements/requirements-simulation.txt
```

The example config contains the custom cell `a=9.68934, b=8.41686,
c=35.6672, alpha=90, beta=89.2448, gamma=90` and wavelength `0.0251 Å`.
It preserves all provided cell digits. Run either through the existing script or
`python -m simulation.rotation_experiment`:

```bash
# Generate all reflections within the configured resolution; intensity 1000 each.
./run.sh simulate --config configs/rotation_experiment.json \
  --intensity-mode constant --output results/constant

# Use only supplied HKL rows, with intensities from column 4.
./run.sh simulate --config configs/rotation_experiment.json \
  --hkl-file /path/to/output.hkl --intensity-mode hkl --output results/measured

# Same supplied HKL reflection list, but equal intensities for comparison.
./run.sh simulate --config configs/rotation_experiment.json \
  --hkl-file /path/to/output.hkl --intensity-mode constant --output results/equal

# Override scan geometry on the command line.
./run.sh simulate --config configs/rotation_experiment.json \
  --axis 0 1 0 --start-deg -60 --stop-deg 60 --step-deg 0.5 \
  --output results/half_degree
```

Use a new or empty output directory for each run. `rotation_axis` is any nonzero
Cartesian direction in the fixed laboratory frame, through the reciprocal origin.
Angles are right-handed degrees; each frame is an absolute rotation from the
initial orientation, avoiding accumulated drift. `initial_euler_deg` applies
extrinsic xyz rotations before the scan. The stop angle is included only when it
falls on the step grid; descending scans use a negative step. These are snapshots
at each angle, not intensity integrations over angular exposure windows.

JSON configs support the fields shown in `configs/rotation_experiment.json`.
Simple `key=value` files are also accepted, including blank lines, `#` comments,
and `lambda`, `wavelength`, or `lambda / wavelength` as wavelength keys:

```text
a = 9.68934
b = 8.41686
c = 35.6672
alpha = 90
beta = 89.2448
gamma = 90
lambda / wavelength = 0.0251
rotation_axis = [0, 1, 0]
start_deg = -60
stop_deg = 60
step_deg = 1
```

### Ewald association

The radial error is `abs(norm(g - [0,0,1/wavelength]) - 1/wavelength)`.
A reflection is accepted when this error is strictly smaller than:

```text
excitation_error + excitation_error_slope * max(0, norm(g) - reference_radius)
```

The two threshold parameters are the base `excitation_error` (Å⁻¹) and the
nonnegative, dimensionless `excitation_error_slope`. The default reference radius
is zero. A zero slope reproduces the original static threshold. Distance is
measured in physical reciprocal space (1/d, without 2π), not in index space.
The legacy random-orientation CLI also accepts `--excitation_error_slope`.

For exactly `threshold(111)=0.01` and `threshold(333)=0.05`, set
`excitation_reference_hkl` to `[1,1,1]`, `excitation_error` to `0.01`, and
`excitation_error_slope` to `0.04 / (2 * norm([1,1,1] @ basis))`.
Compute the numeric slope for your cell with:

```python
import numpy as np
from simulation.rotation_experiment import load_config, reciprocal_basis
cfg = load_config('configs/rotation_experiment.json')
r111 = np.linalg.norm(np.array([1, 1, 1]) @ reciprocal_basis(cfg))
print(0.04 / (2 * r111))
```

This holds the threshold at 0.01 below the reference radius and grows linearly
above it. It does not cap growth beyond 333.

### Intensity and detector output

Each frame produces:

- `frame_00000.png`: 8-bit grayscale visualization series, saved with PIL.
- `frame_00000.npy`: authoritative float64 detector array, black background 0.
- `frame_00000.tif`: float32 TIFF, preserving dynamic range for image software.
- `frame_00000.csv`: on-detector HKL, input intensity, continuous projected x/y,
  pixel column/row, camera depth, Ewald error, and applied threshold.
- `frames.json`: frame angles and spot counts; `config.json`: resolved configuration.
- `intrinsics.json`: 3×3 pinhole K; `extrinsics.json`: per-frame 3×4 world-to-camera matrices.

No normalization, amplitude squaring, Gaussian blur, partiality weighting,
noise, or artificial central beam is applied. An isolated intensity 1000 gives
pixel value 1000 throughout its circular spot. Spots use the notebook’s integer-center
discs (`spot_radius_px=5` by default); overlapping spots sum. Set
`spot_radius_px=0` for single-pixel impulses.
Negative measured intensities are retained. TIFF may round decimal intensities
to float32; use NPY when exact float64 values matter.

HKL columns are signed integers h, k, l followed by intensity; extra columns are
ignored. `000` is omitted as the usual terminator. Only explicitly supplied
reflections are used: there is no automatic Friedel/symmetry expansion. Supplied
rows outside `resolution` (minimum d-spacing in Å) are filtered. Duplicate HKLs
raise an error by default; set `duplicate_policy` to `mean`, `sum`, or `first`
when merging is intended. A config's `hkl_file` is relative to the config;
`--hkl-file` is relative to the working directory. In `constant` mode,
`constant_intensity` sets the value (default 1000).

Projection follows the pinhole camera in the `vggt` branch notebook
`notebooks/extract_camera_positions.ipynb`: homogeneous pixel coordinates are
`K @ [R|t] @ [x,y,z,1]`, divided by camera depth. Points at or behind the camera
are excluded. The fixed camera is on +z looking toward the reciprocal origin;
its default distance is `1/wavelength`, focal length is `10 * image_size` pixels,
and principal point is `[image_size//2, image_size//2]`. Camera +y points downward.
Set `focal_length_px`, `principal_point_px`, and `camera_distance` to override
these defaults. `image_size` is the square image dimension. Legacy `edge` and
`detector_distance` keys are rejected in configured scans to avoid silently
changing their meaning. The Ewald sphere remains fixed while the lattice rotates.

The camera pose is the notebook’s look-at pose at the fixed +z Ewald center.
A rotation of the crystal is exported as the equivalent changing extrinsic
relative to the original unrotated reciprocal lattice: each matrix in
`extrinsics.json` projects `hkl @ basis`, including the initial Euler orientation
and that frame’s scan rotation. Camera depth uses reciprocal-space units.
Images are saved directly using PIL, with no Matplotlib rendering in configured
experiments. Unlike the notebook’s black binary dots on white, these images use
actual reflection intensities on a zero background.

Original random training-data generation remains available without `--config`.

Run regression checks with `python -m unittest discover -s tests -v`.

## Docker and unified command runner

Run all command-line tools through `run.sh`. It builds the Docker image on first
use, bind-mounts this checkout at `/workspace`, and forwards the command arguments.
Outputs and checkpoints remain in your checkout. On Linux, the container runs
with your UID/GID. Relative paths are relative to the repository root even when
calling the script from another directory.

```bash
# Optional explicit build; otherwise the first command builds automatically.
./run.sh build

# Simulate with the supplied cell config (used by default).
./run.sh simulate --output results/constant
./run.sh --mount /data:/data:ro simulate \
  --hkl-file /data/output.hkl --intensity-mode hkl --output results/measured

# Custom config and scan; external config paths need a corresponding mount too.
./run.sh simulate --config configs/rotation_experiment.json \
  --axis 0 1 0 --start-deg -60 --stop-deg 60 --step-deg 0.5 \
  --output results/custom

# Original classification/segmentation data workflow.
./run.sh simulate-random --output_folder data
./run.sh split --data_path data/volume_500 \
  --labels 2DZone 3DLaueIntersections --output_folder data/processed
./run.sh train --root_dir data/processed \
  --model_path models/trained.pth --num_epochs 8 --use_class_weights
./run.sh infer models/trained.pth data/processed/val --output results.json
./run.sh prepare-segmentation --data_path data/volume_500 --output_path data/segmentation
./run.sh train-segmentation --data_dir data/segmentation --num_epochs 8
./run.sh infer-segmentation --model_path models/segmentation.pth --input_dir data/images

./run.sh test
./run.sh shell
./run.sh help
# Per-command options:
./run.sh simulate --help
./run.sh train --help
```

The default Docker image includes CPU simulation and the repository's PyTorch
2.5.1 / torchvision 0.20.1 pair with CUDA 12.4 wheels, using the
[official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/).
Training/inference auto-enables GPUs when `nvidia-smi` works on the host;
NVIDIA Container Toolkit must also be installed. Use `--gpus all` to request
GPUs explicitly or `--gpus none` to force CPU execution. Simulation needs no GPU.
For a smaller CPU-only image:

```bash
GDA_IMAGE=geometric-diffraction:cpu ./run.sh build --build-arg TORCH_BACKEND=cpu
GDA_IMAGE=geometric-diffraction:cpu ./run.sh --gpus none train --root_dir data/processed
```

Build again after dependency or `docker/Dockerfile` changes. Source edits are visible through
the bind mount immediately. `GDA_IMAGE` selects the image name; `GDA_SHM_SIZE`
controls container shared memory (default `2g`). Downloaded pretrained weights
are cached in `.cache/torch` in your checkout. Initial training can require
internet access to download torchvision weights. Existing model files are used
from the mounted checkout; datasets, model weights, HKL files, secrets, and
results are excluded from the image build context.

For direct image use, mount a checkout containing configs/data/models:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" geometric-diffraction:latest simulate --output results/direct
```

`./run.sh --local COMMAND ...` skips Docker and uses the installed Python
interpreter (`PYTHON` overrides `python3`). Full local functionality requires
`requirements/requirements.txt` and `requirements/requirements-simulation.txt`; configured simulation alone
needs only `requirements/requirements-simulation.txt`. Docker is the default execution mode.

Additional preprocessing commands are `convert-tif`, `synthetic`, `stats`,
`overlay`, and `extract-peaks`; each accepts the underlying script's `--help`.
`python` forwards arbitrary Python arguments for diagnostics or custom work.
Configured rotation output is quantitative diffraction data; the existing
classification training workflow expects the original labeled directory split.


## Repository layout

`run.sh` is the only visible file at the repository root. Run the commands above
from that root; moving this README into `docs/` does not change command paths.

| Folder | Contents |
| --- | --- |
| `simulation/` | Configured and random experiment simulation, geometry utilities |
| `training/` | Classification and segmentation training entry points |
| `inference/` | Classification and segmentation inference entry points |
| `data_processing/` | Image conversion, splitting, statistics and overlays |
| `utils/` | Shared utilities and segmentation data preparation |
| `docker/` | Dockerfile and its Docker-specific build-context ignore file |
| `requirements/` | Full and simulation-only dependency files |
| `configs/` | Experiment configurations |
| `docs/` | This documentation |
| `tests/` | Geometry and launcher regression tests |
| `notebooks/` | Interactive analysis notebooks |
| `models/` | Model checkpoints |

Python entry points can also be invoked as modules from the root, for example
`python -m training.train --help` or `python -m inference.run_inference --help`.
Use `./run.sh build` for Docker builds, or explicitly run
`docker build -f docker/Dockerfile -t geometric-diffraction:latest .`. The context
remains the repository root, and Docker uses `docker/Dockerfile.dockerignore`.


### PNG visualization series

Every configured simulation also saves `frame_00000.png`, `frame_00001.png`, etc.
Open these in a standard image viewer. PNGs use bright spots on a black background
and a shared scale across the complete experiment so intensities remain comparable
when stepping through frames. The default logarithmic mapping makes weak reflections
easier to see: `255 * log(1 + max(I, 0)) / log(1 + series_max)`.

```bash
./run.sh simulate --output results/visualization
# Optional linear brightness mapping instead of logarithmic:
./run.sh simulate --png-scale linear --output results/linear_preview
```

Set `png_scale` to `log` or `linear` in your config, or override it with
`--png-scale`. `visualization.json` records the scale and series maximum;
`frames.json` links each frame to its PNG. Negative intensities are displayed
as zero; empty/all-nonpositive series produce black PNGs. Preview files are
normalized for viewing; NPY and TIFF retain the quantitative intensity values.
PNGs are written after all raw frames have been simulated so the common maximum
is known. The conversion reads one raw frame at a time.
