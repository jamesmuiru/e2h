# Embed2Heights

Embed2Heights is a dual-branch multi-task deep learning model designed for Earth Observation applications. It simultaneously performs semantic segmentation for land-cover classes (buildings, vegetation, water) and predicts normalized Digital Surface Model (nDSM) height maps from multi-modal satellite data.

---

## 🏗️ Architecture Overview

The model employs a custom dual-branch architecture to fuse high-resolution spatial data with dense patch embeddings.

* **Branch A (High-Resolution):** Concatenates AlphaEarth (64 channels) and Tessera (128 channels) into a 192-channel input.
* **High-Resolution Backbone:** Utilizes a pretrained ConvNeXt-Base model (via `timm`) with a modified stem to handle the 192-channel input, outputting 5 multi-scale feature maps.
* **Branch B (Patch Token):** Concatenates TerraMind (Sentinel-1 and Sentinel-2) and THOR (Sentinel-1 and Sentinel-2) embeddings to form a 3072-channel input.
* **Token Backbone:** Processes embeddings through a linear projection and a 4-layer Transformer Encoder.
* **Fusion Module:** Employs Cross-Attention at a 16x16 spatial resolution, treating the high-resolution features as the query and the token features as the keys and values.
* **Decoder:** Uses a SegFormer-style decoder that projects all features using 1x1 convolutions, bilinear-upsamples them to 256x256, and fuses them via a 3x3 convolution.
* **Heads:** Features a segmentation head outputting a 3-class sigmoid tensor and a height head outputting a softplus tensor in log1p space.

---

## 📊 Loss and Evaluation Metrics

The model optimizes for a custom composite loss function combining segmentation and height estimation objectives.

$$Loss_{total} = 0.45 \times (Loss_{BCE} + Loss_{Dice}) + 0.55 \times Loss_{Huber}$$

Performance is evaluated using the official competition metric weights:

* mIoU Buildings accounts for 25% of the total score.
* RMSE Building Height accounts for 25% of the total score.
* RMSE Vegetation Height accounts for 20% of the total score.
* mIoU Trees accounts for 15% of the total score.
* mIoU Water accounts for 15% of the total score.

---

## 🗂️ Project Structure

* `config.py`: Acts as the single source of truth for all root paths, dataset directories, and model hyperparameters.
* `dataset.py`: Defines the `GeoFMDataset` and `GeoFMTestDataset` classes, handling defensive NaN sanitization, array shape enforcement, and parquet catalog parsing.
* `losses.py`: Implements numerical-safe formulations of BCE-Dice and Huber losses, alongside the evaluation metrics logic.
* `model.py`: Contains the raw PyTorch implementation of the complete dual-branch architecture, including pure-PyTorch fallbacks if `timm` is unavailable.
* `predict.py`: Handles inference on the test set, features Test-Time Augmentation (TTA), and clamps height outputs to a realistic physical threshold of 100 meters.
* `train.py`: Provides the main training loop with Automatic Mixed Precision (AMP), multi-GPU DataParallel support, and Cosine Annealing learning rate scheduling.

---

## 🚀 Usage

### Prerequisites

Ensure your data is organized exactly as defined in the configuration, primarily under `/scratch/lustre/users/hmwangi/embed2heights/data/data/`. You will need PyTorch alongside `timm`, `rasterio`, `tifffile`, and `pandas`.

### Training the Model

To launch training using the default configurations specified in `config.py` (80 epochs, batch size 16, learning rate 1e-4):

```bash
python train.py

```

To override default training parameters:

```bash
python train.py --epochs 100 --batch_size 32 --lr 5e-5

```

The script will automatically use all available GPUs via `DataParallel` and save the best checkpoint to the `runs/exp01_convnext/checkpoints/` directory.

### Running Inference

To generate predictions on the test dataset using the best trained checkpoint:

```bash
python predict.py

```

By default, inference utilizes Test-Time Augmentation (TTA) with horizontal and vertical flips. To disable TTA for faster, un-augmented inference:

```bash
python predict.py --no_tta

```

Predictions will be saved as raw `.npy` arrays containing the segmentation fractions and absolute heights in meters to the `runs/exp01_convnext/predictions/` folder.
