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

