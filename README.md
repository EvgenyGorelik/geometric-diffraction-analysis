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
