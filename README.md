# Image Deblurring with U-Net and GAN

## Abstract
This project explores deep learning-based image deblurring using paired blurred and sharp images. Two approaches are implemented and compared: a U-Net architecture optimized with pixel-wise reconstruction loss, and a GAN-based model trained with adversarial and perceptual losses. Performance is evaluated quantitatively using PSNR and SSIM metrics and qualitatively through visual inspection.

## Problem
Motion blur frequently occurs in real-world image acquisition scenarios such as handheld photography, fast object motion, and low-light conditions. Removing blur while preserving fine details and realistic textures remains a challenging inverse problem. This project aims to restore sharp images from blurred inputs using convolutional neural networks.

## Approach
- **Dataset**: Paired blurred and sharp images derived from the GoPro (GOPRO_Large) dataset.
- **Preprocessing**: Images are resized to 256×256 and normalized prior to training.
- **U-Net Model**: An encoder–decoder architecture with skip connections trained using Mean Squared Error (MSE) loss.
- **GAN Model**: A generator–discriminator framework incorporating Wasserstein-style adversarial loss and VGG-based perceptual loss to encourage visually realistic outputs.
- **Evaluation**: Image quality is assessed using PSNR and SSIM on held-out test samples.

> Due to computational constraints, training was conducted on a limited subset of the dataset.

## Key Findings
- The **U-Net** model achieved higher quantitative performance (PSNR and SSIM) in this experimental setting.
- The **GAN** model produced visually plausible results but showed lower numerical scores under limited training data.
- Overall, reconstruction-based models demonstrated more stable performance given constrained resources.

## Project Structure
```text
src/
├── preprocessing/   # Data loading and preprocessing
├── models/           # U-Net and GAN architectures
├── train/            # Training scripts
├── eval/             # Evaluation and metrics
└── inference/        # Inference on new images
```

## Code
- `src/preprocessing/paired_dataset.py`: Loads paired blurred/sharp images and applies resizing and normalization.
- `src/models/UNet.py`: Defines the U-Net architecture with encoder–decoder blocks and skip connections.
- `src/models/GAN.py`: Implements the GAN generator and discriminator, including reflection padding and perceptual loss.
- `src/train/train_UNet.py`: Training pipeline for the U-Net model.
- `src/train/train_GAN.py`: Training pipeline for the GAN-based model.
- `src/eval/metrics.py`: PSNR and SSIM evaluation utilities.
- `src/eval/eval_UNet.py`: Quantitative evaluation of the trained U-Net.
- `src/eval/eval_GAN.py`: Quantitative evaluation of the trained GAN.
- `src/inference/`: Scripts for single-image inference and visualization.

## Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-image
- Pillow

## Contribution
- Independently designed and implemented the full image deblurring pipeline.
- Developed and compared two deep learning approaches (U-Net and GAN).
- Implemented preprocessing, training, evaluation, and inference workflows.
- Analyzed results using standard image quality metrics and documented limitations.

## Dataset
Downloads link: [GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view)

The dataset is **not included** in this repository due to size constraints.  
Please download it separately and place it under the `data/` directory following the structure described in `data/README.md`.

## References
- [Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
- [Nah, S., Kim, T. H., & Lee, K. M. (2017). *Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring*. CVPR](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)
- [Kupyn, O., Budzan, V., Mykhailych, M., Mishkin, D., & Matas, J. (2018). *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks*. CVPR](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf)

