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

## Reconstruct a lattice from diffraction images

`utils/reconstruct_lattice.py` estimates a reciprocal lattice and relative camera
rotations from a batch of grayscale images, without input orientations or Miller
indices. Run the reproducible image-only example:

```bash
python -m examples.reconstruct_lattice_demo
python -m unittest discover -s tests -v
```

For your own images:

```python
import numpy as np
from PIL import Image
from utils.reconstruct_lattice import Calibration, reconstruct_images

images = [np.asarray(Image.open(path).convert('L')) for path in image_paths]
calibrations = [Calibration(center=(512, 512), pixel_size=0.002,
                            wavelength=0.0407, detector_distance=0.0)
                for image in images]  # replace with your instrument calibration
result = reconstruct_images(images, calibrations, tolerance=0.003,
                            peak_options={'dark': True, 'min_area': 3})
basis = result['basis']  # three ROW vectors: hkl @ basis = reciprocal coordinates
angles = result['azimuth_elevation_roll']  # degrees, relative to image 0
```

The calibration above is an example, not an inferred measurement. The detector
model is exactly `simulation/utils/project_reflections_2D.py`: let `R=1/wavelength`,
`x,y=(u,v-center)*pixel_size`, and `d=(x,y,-(detector_distance+R))`. Then the
reciprocal vector is `q=R*d/|d|+(0,0,R)`. Pixel v increases with detector y.
`pixel_size` must undo any resize or scaling used to rasterize the detector
coordinates. `tolerance` is a reciprocal-space distance, not pixels; include
pixel localization error and excitation thickness. Absolute units require known
scale and wavelength. The general Ewald-sphere conversion is described in
[IUCr's autoindexing overview](https://journals.iucr.org/d/issues/2013/07/00/ba5191/).

The perspective render in `notebooks/spherical_intersections.ipynb` is a different
model: it rotates the lattice and then views the selected points using an
independent off-axis camera. Do not pass those canvases to this detector inverse.
Generate detector images with `project_reflections_2D` instead, as in the new
`notebooks/lattice_reconstruction.ipynb` example. Reconstruction from that arbitrary
perspective setup would require a different projection/selection model and joint
camera calibration; equal pixel radii are not invariant there.

The algorithm extracts weighted component centroids, lifts them to the Ewald
sphere, proposes correspondences with matching vector lengths (shorter vectors
get higher weights), and uses pairs of non-collinear vectors to hypothesize
proper 3D rotations. It checks the hypotheses against additional spots using
one-to-one matches, then connects images through a maximum-score spanning tree.
Finally, it searches short vectors and differences for an independent basis,
alternating integer indexing and least-squares refinement. Known detector
geometry supplies depth; merely matching one radial line in two flat planes
would leave their relative tilt undetermined.

Outputs include `uv_points`, `points`, `image_ids`, `rotations`, scored `links`,
the chosen `tree`, `basis`, `hkl`, `inliers`, and per-point `residuals`. The direct
beam is removed before fitting. Rows of `points`, `hkl`, `inliers`, `residuals`,
and `image_ids` correspond; `uv_points` retains all extracted spots. A camera
vector maps to the reference frame as `q @ rotations[i]`. Angles describe the
same active column rotation as `Rz(azimuth) Ry(elevation) Rx(roll)`. Global
orientation and crystallographically equivalent basis choices are arbitrary.
The direct-space basis under the simulator's no-2π convention is
`np.linalg.inv(basis).T`.

This is a bounded initial estimator, not a general autoindexing solver. It assumes
one crystal, separated spots, known centers and calibration, and enough shared
non-collinear reflections for all images to connect. It raises an error for
disconnected batches, effectively coplanar observations, or unsuccessful basis
fits. Symmetry can produce incorrect or equivalent registrations; missing
reflections can make the primitive lattice unidentifiable. A high indexed
fraction alone is not a uniqueness guarantee. There is no global rotation or
camera refinement, so long registration chains can drift. Pair matching costs
O(N²) view comparisons; use a smaller overlapping batch for the notebook's 1000
views. `max_trials` and `candidate_limit` bound the correspondence and basis
searches. Experimental backgrounds and merged spots require preprocessing.
