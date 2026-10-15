# geometric-diffraction-analysis

Create datastructure:
```
python create_data_split.py --data_path <path/to/data> --labels 2DZone 3DLaueIntersections MultipleCrystals --output_folder data
```

Training:
```
python train.py --root_dir data --model_path models/diff_fibonacci_model.pth --use_class_weights
```


Inference:
```
python run_inference.py models/diff_fibonacci_model.pth ./data --output results.json
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
python data_processing/get_img_stats.py <path/to/synthetic/img>
```

Then we use this file for converting all experimental data to synthetic using
```
python data_processing/create_synthetic_images.py ./data img_stats.json --output_folder ./data_processed --threshold 0.01 --nms_size 20
```



## Experimental Data Simulator

Simulate experiment data using the `simulate_experiment.py` script. It will automatically create simulated experiment data.


## Segmentation Data Preparation

Simulate experiment multiple times (e.g. 10):
```
count=10 for i in $(seq 1 ${count}); do python simulate_experiment.py --output_folder data; done
```

Prepare data for training:
```
python utils/prepare_segmentation_data.py --data_path data/volume_500/ --output_path data/segmentation
```

Train segmentation
```
python train_segmentation.py --num_epochs 8 > segmentation_"$(date +%s)".log
```

## Configured rotation experiments

Install the lightweight simulator dependencies (no trained models or PyTorch needed):

```bash
python -m pip install -r requirements-simulation.txt
```

The example config contains the custom cell `a=9.68934, b=8.41686,
c=35.6672, alpha=90, beta=89.2448, gamma=90` and wavelength `0.0251 Å`.
It preserves all provided cell digits. Run either through the existing script or
`python -m simulation.rotation_experiment`:

```bash
# Generate all reflections within the configured resolution; intensity 1000 each.
python simulate_experiment.py --config configs/rotation_experiment.json \
  --intensity-mode constant --output results/constant

# Use only supplied HKL rows, with intensities from column 4.
python simulate_experiment.py --config configs/rotation_experiment.json \
  --hkl-file /path/to/output.hkl --intensity-mode hkl --output results/measured

# Same supplied HKL reflection list, but equal intensities for comparison.
python simulate_experiment.py --config configs/rotation_experiment.json \
  --hkl-file /path/to/output.hkl --intensity-mode constant --output results/equal

# Override scan geometry on the command line.
python simulate_experiment.py --config configs/rotation_experiment.json \
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

- `frame_00000.npy`: authoritative float64 detector array, black background 0.
- `frame_00000.tif`: float32 TIFF, preserving dynamic range for image software.
- `frame_00000.csv`: on-detector HKL, input intensity, continuous projected x/y,
  pixel column/row, Ewald error, and applied threshold.
- `frames.json`: frame angles and spot counts; `config.json`: configuration.

No normalization, amplitude squaring, Gaussian blur, partiality weighting,
noise, or artificial central beam is applied. An isolated intensity 1000 gives
pixel value 1000. Spots are nearest-pixel impulses and overlapping spots sum.
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

Projection retains the existing simulator convention: sphere center +z/wavelength,
projection from that center to z=-detector_distance. At detector_distance=0,
x/y and `edge` have reciprocal-space units. `edge` is the detector half-width;
`image_size` is its square pixel dimension. Rows increase downward, +y points up.
Original random training-data generation remains available without `--config`.

Run regression checks with `python -m unittest discover -s tests -v`.
