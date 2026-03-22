# High-Sensitivity Pan-Cancer AI Assessment of Lymph Node Metastasis via Uncertainty Quantification

[Link to paper](https://www.nature.com/articles/s41746-026-02564-y)

This repository contains the official implementation of **UPATHLN**, a novel deep learning framework proposed in our paper. UPATHLN is an AI system specifically designed for predicting lymph node metastasis tumors in Whole Slide Images (WSIs), while simultaneously providing uncertainty estimations for its predictions. It features an automated pipeline that generates tumor prediction heatmaps, uncertainty maps, and extracts structured quantitative results for high-sensitivity assessment.

## Environment Requirements

- python 3.12.7
- PyTorch 2.5.1
- torchvision 0.20.1
- timm 1.0.11
- openslide-bin 4.0.0.6
- openslide-python 1.4.1
- opencv-python 4.10.0.84
- numpy 1.26.4
- einops 0.8.0

## Project Structure

- `classifier.py` - Core model architectures for patch classification
- `infer_methods.py` - Implementation of various inference methodologies
- `infer_wsi.py` - Main script for WSI-level inference pipeline
- `multiscale_infer_dataset.py` - Dataset handling and preprocessing for multiscale patches

## Usage

### Executing Inference

Run the following command to start the inference process:

```bash
python infer_wsi.py \
    --ckpt_path ./model_ckpt.pth.tar \
    --wsi_dirpath ./case/wsi \
    --mask_dirpath ./case/mask \
    --thumb_dirpath ./case/thumb \
    --batch_size 256 \
    --mask_resize 6
```

**Parameter Description:**

- `mask_resize`: Key parameter controlling inference granularity. Set to 1 for fine-grained inference; larger values result in coarser granularity. Setting it to 6 achieves an optimal balance between high accuracy and reduced computational complexity.
- `batch_size`: It is recommended to adjust based on your GPU VRAM. Increasing `batch_size` can help reduce inference time if ample memory is available.


## Citation

```bibtex
@article{wang2026highsensitivity,
  title   = {High-sensitivity pan-cancer {AI} assessment of lymph node metastasis via uncertainty quantification},
  author  = {Wang, Xiaodong and Chen, Ying and Liu, Xiaohong and Qiu, Cen and Tang, Hong and Huang, Tinggui and Guo, Siqi and Ma, Sainan and Cai, Mengjiao and Sun, Qingyun and Chang, Zichen and Liu, Jinge and Wang, Xiongjun and Li, Jinda and Qian, Wulei and Wang, Biyu and Zhang, Boan and Bai, Chenguang and Shi, Min and Zhang, Xinlei and Li, Meng and Wang, Jiahai and Wang, Bin and Ma, Jinlu and Ai, Lirong and Yu, Shaoqing and Wang, Liming and Feng, Ninghan and Liu, Xiyang and Yu, Guanzhen},
  journal = {npj Digital Medicine},
  year    = {2026},
  month   = mar,
  doi     = {10.1038/s41746-026-02564-y},
  url     = {https://doi.org/10.1038/s41746-026-02564-y},
  note    = {Published online 2026-03-21}
}
