---
title: Crack Segmentation
emoji: 🗿
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.1.1
app_file: app.py
pinned: false
license: mit
---

# Corrosion Segmentation Web Application for Cawil.AI

Welcome to the Corrosion Segmentation Web Application! This tool is designed to segment corrosion in images. It's powered by Gradio and deployed on HuggingFace's Spaces platform. The underlying model and utilities are built using Python 3.10.9.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Interactive UI:** Powered by Gradio for easy upload and visualization of corrosion segmentation results.
- **High-Quality Segmentation:** Uses state-of-the-art machine learning techniques to provide accurate segmentation results.
- **Deployed on HuggingFace Spaces:** Access the application anytime from anywhere!

## Installation

To run this application locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/cawil-ai/Cawil-Corrosion-Segmentation.git 
    cd Cawil-Corrosion-Segmentation
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Once the dependencies are installed, run the application using the command:
    ```bash
    python app.py
    ```

2. Navigate to the URL provided in the console to access the web application.

3. Upload an image and click on 'Segment' to view the corrosion segmentation results.

## Dependencies

The application requires the following Python packages:

```
gradio==4.1.1
numpy==1.25.2
pandas==2.0.3
Pillow==10.0.0
torch==2.0.1
ultralytics==8.0.149
opencv-python>=4.6.0
shortuuid>=1.0.11
```

These dependencies can also be found in the `requirements.txt` file in the repository.
