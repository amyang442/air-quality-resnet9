# Image-Based Deep Learning for Air Quality Classification

## Awards & Recognition

- **Regeneron Science Talent Search (STS) Scholar** - Top 300 nationally (2023)
- **Winner - New York State Science & Engineering Fair**
- **Winner - Westchester Science & Engineering Fair**
- **Student Award of Geoscience Excellence - Association of Women Geoscientists**
- **Taking the Pulse of the Planet Award - National Oceanic and Atmospheric Association (NOAA)**

---

## Project Overview

Traditional air quality monitoring systems cost thousands of dollars and are often broken or nonexistent in many locations, including national parks. This project develops a **9-layer Residual Neural Network (ResNet9)** to estimate PM2.5 air pollution levels directly from webcam images, providing a cost-effective alternative.

**Key Achievement:** Achieved **83% classification accuracy**, outperforming state-of-the-art CNN models (69%) and Random Forest classifiers (64%) by over 20%.

### The Problem
- 124 U.S. national parks have no air quality monitors
- Traditional monitors cost $10,000+ and require sophisticated setup
- 7 million people die annually from air pollution worldwide

### The Solution
A deep learning model that classifies images into three PM2.5 concentration levels:
- **Good:** <35.4 µg/m³
- **Unhealthy:** 35.5-150.4 µg/m³  
- **Hazardous:** >150.5 µg/m³

## Results

| Model | Accuracy | Training Loss | Validation Loss |
|-------|----------|---------------|-----------------|
| **ResNet9 (This Project)** | **83%** | 0.38 | 0.42 |
| CNN Baseline | 69% | 0.51 | 0.58 |
| Random Forest | 64% | - | - |

## Dataset

**862 images** spanning 2014-2021 from two diverse locations:
- **456 images:** Beijing (urban environment, multiple webcams)
- **406 images:** Yosemite National Park (natural environment, Turtleback Dome)

Each image is paired with PM2.5 concentration data from the AirNow database.

**[Access Dataset on Kaggle](https://www.kaggle.com/datasets/amyyang442/airqualityyosemitebeijing)**

## Technical Details

**Architecture:**
- 9-layer Residual Neural Network with skip connections
- 4 convolutional blocks with batch normalization
- 2 residual blocks to prevent vanishing gradients
- Dropout (0.2) for regularization

**Training Configuration:**
- Framework: PyTorch
- Batch size: 128 images
- Multi-stage learning: 0.001 → 0.0001 → 0.00001
- Optimizer: Adam (handles noisy images effectively)
- Total epochs: 30 (5 + 10 + 15)
- GPU acceleration: CUDA-enabled

**Data Processing:**
- Original resolution: 320×213 pixels
- Resized to: 64×64 pixels
- Train/validation split: 90%/10% (776 train, 86 validation)

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation
#### Clone the repository
- git clone https://github.com/amyang442/air-quality-resnet9.git
- cd air-quality-resnet9

#### Install dependencies
pip install -r requirements.txt

### Run the Notebook
1. Open `air-quality-resnet9.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to:
   - Download and explore the dataset
   - Train the ResNet9 model
   - Evaluate performance metrics
   - Test predictions on individual images

## Research Paper

This project is based on peer-reviewed research presented at multiple science competitions.

**Read the full paper:** [`Image-based-Deep-Learning-on-Air-Quality.pdf`](Image-based-Deep-Learning-on-Air-Quality.pdf)

## Applications

This technology has the potential to enable:
- **National Parks:** Low-cost monitoring for 124 parks without air quality sensors
- **Mobile Apps:** Real-time PM2.5 estimation from smartphone photos
- **Government Agencies:** Scalable air quality monitoring across large regions
- **Researchers:** Historical air quality analysis using webcam archives

## Future Work

- Expand dataset to additional geographic regions (Europe, Asia, South America)
- Incorporate weather variables (humidity, temperature) for PM2.5 forecasting
- Test alternative architectures (ResNet18, EfficientNet, Vision Transformers)
- Deploy as mobile application for iOS/Android
- Integrate with existing webcam networks in national parks

## Contact

**Amy Yang**
- LinkedIn: https://www.linkedin.com/in/amy-yang-44bw/ 
- Email: amyyang442@gmail.com

## Acknowledgments

- National Park Service for providing Yosemite webcam archives
- AirNow database for PM2.5 concentration data
- My research mentor for guidance throughout this project
