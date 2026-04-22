# Road Network Extraction

This project trains a road segmentation model on the DeepGlobe Road Extraction Dataset and then fine-tunes it on multi-city Indian satellite imagery. It includes utilities to download map tiles, build ground-truth masks, prepare DeepGlobe patches, train the model, fine-tune on additional cities, and run inference on Indore.

## Results

Result images are available here:

https://drive.google.com/drive/folders/1uy4F5W_RTpzjlbhLYX4HsUAs7m_8Jp2y?usp=drive_link

## 👥 Team Members

| Name               | Roll Number |
|--------------------|------------|
| Vikash Kumar       | 230004049  |
| Ayush Kumar       | 230001012  |


## Requirements

Use Python 3.10+ and install the project dependencies from `requirements.txt`.

For a CPU-only environment:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

For a CUDA 12.6 GPU environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

If you already have a compatible PyTorch build installed, you can still run:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### 1) Download DeepGlobe

Download the DeepGlobe Road Extraction Dataset from Kaggle, extract it, and move the train, val, and test folders into:

```text
data/datasets/deepglobe/
```

The preprocessing script expects the raw DeepGlobe files to be available under that directory, including the `*_sat.jpg` images and matching `*_mask.png` files.

Kaggle dataset:

https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

### 2) Download the city data used by this project

Run the download scripts from the project root:

```bash
python download_map.py
python download_gt.py
python download_cities.py
```

If some tiles are still missing after running `download_cities.py`, repair them with:

```bash
python repair_missing_tiles.py
```

These scripts generate:

```text
indore/indore_map.tif
indore/real_export.geojson
data/maps/<city>.tif
data/masks/<city>.geojson
data/manifest.json
```

The `data/manifest.json` file is used by `finetune_multicity.py`.

## Workflow

### 1) Prepare DeepGlobe patches

Before training, convert the raw DeepGlobe dataset into train/validation/test patches:

```bash
python scripts/prepare_deepglobe.py --data-dir data/datasets/deepglobe
```


The script writes processed data to:

```text
data/processed/train/
data/processed/val/
data/processed/test/
```

### 2) Train the base model

Run the main training script:

```bash
python train.py
```

Useful options:

```bash
python train.py --config configs/config.yaml
python train.py --model attention_unet --encoder resnet50
python train.py --batch-size 2
python train.py --loss focal_dice
python train.py --dry-run
```

The dry run is recommended first because it checks that data loading and the forward pass work correctly.

Training outputs are written to `runs/`, including checkpoints and logs.

### 3) Fine-tune on multi-city data

After the base model is trained, run the multi-city fine-tuning script:

```bash
python finetune_multicity.py
```

This script loads `runs/checkpoints/best_model.pth`, builds patches from the downloaded city data, and saves the fine-tuned model to:

```text
runs/final_model_v3.pth
runs/best_model.pth
```

### 4) Run inference

After fine-tuning, evaluate the model on the Indore map:

```bash
python run_inference.py
```

The inference script reads:

```text
indore/indore_map.tif
indore/real_export.geojson
```

and saves predictions and metrics in `results/`.

## Configuration

Training and data paths are configured in `configs/config.yaml`.

Key settings include:

- model architecture and encoder
- learning rate, batch size, epochs, and scheduler
- loss function settings
- threshold optimization
- post-processing parameters

## Project Structure

```text
download_cities.py            # Download satellite maps and OSM road masks for multiple cities
download_gt.py                # Download Indore road ground truth from Overpass
download_map.py               # Download and stitch the Indore satellite map
scripts/prepare_deepglobe.py  # Convert DeepGlobe into train/val/test patches
train.py                      # Base training entry point
finetune_multicity.py         # Multi-city fine-tuning pipeline
run_inference.py              # Indore inference and evaluation
configs/config.yaml           # Main configuration file
```

## Notes

- Make sure the directory names match the expected paths exactly.
- If `download_cities.py` leaves a few blank or missing tiles, run `python repair_missing_tiles.py` before training.
- If you want to verify the full pipeline quickly, start with `python train.py --dry-run`.
- The multi-city fine-tuning stage expects a trained base checkpoint in `runs/checkpoints/best_model.pth`.
