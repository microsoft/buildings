# TEMPO

Reproduce the full training configuration from the paper "[TEMPO: Global Temporal Building Density and Height Estimation from Satellite Imagery](https://arxiv.org/abs/2511.12104)" to train a deep learning model that predicts building density and height from RGB quarterly mosaics using weak supervision from [Open Buildings Temporal](https://arxiv.org/abs/2310.11622) and [Overture](https://overturemaps.org/) data—this repository contains only the main training pipeline, not ablation studies or evaluation protocols.

## Before You Start

**You need Planet imagery quarterly mosaics to use this repository.** This is the primary requirement for training. If you don't have access to Planet data, training is not possible with this codebase as-is. 

*Alternative:* creating Sentinel-2 quarterly mosaics could be a potential replacement, but we cannot guarantee the training will work as described in [the paper](https://arxiv.org/abs/2511.12104).

If you have Planet imagery, this guide will walk you through acquiring and processing the supporting datasets (Overture building priors and Google 2.5D weak labels) and training your model.

## Setup

### 1. Set Your Data Root Directory

First, choose a root directory where all your data will live:

```bash
export DATA_ROOT=/path/to/your/data
mkdir -p $DATA_ROOT
```

### 2. Create Environment and Install Dependencies

```bash
mamba create -n tempo python=3.12.3
conda activate tempo
mamba install -c conda-forge gdal
pip install -r requirements-training.txt
pip install -e .
```

### 3. Index Files

Download the required index files from Azure:

```bash
mkdir -p data/
wget -O data/planet_index.gpkg https://opendata.aiforgood.ai/building-density/tile_index.gpkg
wget -O data/google_index.feather https://opendata.aiforgood.ai/building-density/google_index.feather
```

These provide:

1. **`planet_index.gpkg`**: Planet quad geometries (will be read to create columns `quad` and `geometry`)
2. **`google_index.feather`**: Google tile index with columns `tile_path`, `geometry`, and `crs`

You need to create the following:
3. **`available_planet_images.csv`**: List of Planet imagery mosaics you have. Columns: `file` (quad name) and `date` (format: `YYYY-qQ`)

Example:
```csv
file,date
L15-1202E-1200N,2020-q1
L15-1453E-1200N,2020-q1
```

## Data Acquisition & Processing

Now we'll acquire and process the supporting datasets needed for training.

### Step 1: Download Overture Building Priors

Create a directory for Overture data:

```bash
mkdir -p $DATA_ROOT/overture/feathers
mkdir -p $DATA_ROOT/overture/masks
```

Download Overture building polygons for your Planet quads:

```bash
python scripts/acquisition/overture.py \
  --planet_index data/planet_index.gpkg \
  --available_imagery data/available_planet_images.csv \
  --release 2024-10-23.0 \
  --output_dir $DATA_ROOT/overture/feathers
```

Rasterize to 512×512 density masks:

```bash
python scripts/processing/overture.py \
  --overture_feathers_path $DATA_ROOT/overture/feathers \
  --output_path $DATA_ROOT/overture/masks
```

### Step 2: Generate Google 2.5D Training Labels

Create directories for Google data:

```bash
mkdir -p $DATA_ROOT/google/masks
```

List all Google Open Buildings Temporal GeoTIFF URLs for your year of interest:

```bash
python scripts/acquisition/build_google_index.py \
  --url-list $DATA_ROOT/google_urls.txt \
  --year 2020
```

Process Google tiles to create 512×512 density and height masks aligned to Planet quads:

```bash
python scripts/processing/open_buildings.py \
  --available-planet-csv data/available_planet_images.csv \
  --google-index data/google_index.feather \
  --planet-index data/planet_index.gpkg \
  --urls-file $DATA_ROOT/google_urls.txt \
  --output-dir $DATA_ROOT/google/masks
```

Output structure: `$DATA_ROOT/google/masks/{year}/{quad_name}.tif` with 2 bands (density, height).

### Step 3: Create Training Index CSV

Create a CSV file at `$DATA_ROOT/training_index.csv` linking your Planet images, Overture priors, and Google labels:

```csv
image,mask,overture,val_weight,weight
$DATA_ROOT/planet/2020/q1/L15-1440E-1202N.tif,$DATA_ROOT/google/masks/2020/L15-1440E-1202N.tif,$DATA_ROOT/overture/masks/L15-1440E-1202N.tif,1.0,3493.8360
$DATA_ROOT/planet/2020/q1/L15-1468E-1206N.tif,$DATA_ROOT/google/masks/2020/L15-1468E-1206N.tif,$DATA_ROOT/overture/masks/L15-1468E-1206N.tif,1.0,1460.2109
```

**Columns:**
- `image`: Path to Planet imagery (4-band GeoTIFF)
- `mask`: Path to Google 2.5D labels (2-band: density, height)
- `overture`: Path to Overture density prior (1-band)
- `weight`: Training sampling weight (sum of valid density pixels in mask)
- `val_weight`: Validation sampling weight

## Training Your Model

Once your data is prepared, you're ready to train.

### Configure Training

Create a training configuration file or use the FULL configuration from the paper (see `configs/FULL.yaml`). Update these key parameters:

- `index`: Path to your training CSV (e.g., `$DATA_ROOT/training_index.csv`)
- `in_channels`: 4 (RGB + Overture prior)
- `target_bands`: [0, 1] (density and height)
- `band_normalizers`: [[min, max], ...] for each input band
- `target_normalizers`: [1.0, 100] (density scaled by 1.0, height by 100)
- `output_dir`: Where to save checkpoints
- `log_dir`: TensorBoard logs directory

### Run Training

```bash
python scripts/train.py --config configs/FULL.yaml
```

## Running Inference

After training, use your model to make predictions on new imagery.

### Prepare Input File

Create an input file (e.g., `$DATA_ROOT/inference_inputs.txt`) listing Overture priors and RGB imagery pairs, one per line:

```
$DATA_ROOT/overture/masks/L15-1440E-1202N.tif,$DATA_ROOT/planet/2020/q1/L15-1440E-1202N.tif
$DATA_ROOT/overture/masks/L15-1468E-1206N.tif,$DATA_ROOT/planet/2020/q1/L15-1468E-1206N.tif
```

### Run Predictions

```bash
python scripts/infer.py \
  --checkpoint <CHECKPOINT_PATH> \
  --input-file $DATA_ROOT/inference_inputs.txt \
  --output-dir $DATA_ROOT/predictions \
  --gpu 0 \
  --batch-size 4
```

**Parameters:**
- `--checkpoint`: Path to trained model checkpoint (`.ckpt` file)
- `--input-file`: Text file with `overture_path,rgb_path` pairs (one per line)
- `--output-dir`: Directory to save predictions
- `--gpu`: GPU ID (omit for CPU inference)
- `--batch-size`: Number of images to process simultaneously (default: 1)
- `--num-workers`: Data loading workers (default: 4)

**Output:** Predictions saved as `{output_dir}/{quad}.tif` with 2 bands (density, height) in EPSG:3857.
