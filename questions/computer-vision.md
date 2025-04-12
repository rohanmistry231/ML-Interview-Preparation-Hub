# Computer Vision Questions

This file contains computer vision questions commonly asked in machine learning interviews at companies like **Google**, **Amazon**, **Meta**, and others. These questions assess your **understanding** of image processing, deep learning models, and vision-specific techniques, covering topics like convolutional neural networks (CNNs), object detection, and evaluation metrics.

Below are the questions with detailed answers, including explanations, technical details, and practical insights for interviews.

---

## Table of Contents

1. [What is the role of convolutional neural networks (CNNs) in computer vision?](#1-what-is-the-role-of-convolutional-neural-networks-cnns-in-computer-vision)
2. [Explain the difference between convolution and pooling layers](#2-explain-the-difference-between-convolution-and-pooling-layers)
3. [What is transfer learning, and how is it used in computer vision?](#3-what-is-transfer-learning-and-how-is-it-used-in-computer-vision)
4. [What is the difference between object detection and image classification?](#4-what-is-the-difference-between-object-detection-and-image-classification)
5. [Explain the YOLO algorithm for object detection](#5-explain-the-yolo-algorithm-for-object-detection)
6. [What is semantic segmentation, and how does it differ from instance segmentation?](#6-what-is-semantic-segmentation-and-how-does-it-differ-from-instance-segmentation)
7. [What are some common data augmentation techniques for computer vision?](#7-what-are-some-common-data-augmentation-techniques-for-computer-vision)
8. [How do you evaluate the performance of a computer vision model?](#8-how-do-you-evaluate-the-performance-of-a-computer-vision-model)

---

## 1. What is the role of convolutional neural networks (CNNs) in computer vision?

**Answer**:

**Convolutional Neural Networks (CNNs)** are deep learning models designed to process structured grid-like data, such as images, and are foundational to computer vision tasks.

- **Role**:
  - **Feature Extraction**: Learn hierarchical features (e.g., edges, textures, objects) via convolutional layers.
  - **Spatial Invariance**: Capture patterns regardless of position (e.g., a cat anywhere in the image).
  - **Task Versatility**: Used for classification, detection, segmentation, and more.
  - **End-to-End Learning**: Map raw pixels to outputs without manual feature engineering.

- **Components**:
  - **Convolutional Layers**: Apply filters to extract features (e.g., 3x3 kernels).
  - **Pooling Layers**: Downsample to reduce dimensionality, enhance robustness.
  - **Fully Connected Layers**: Aggregate features for final predictions.
  - **Activation Functions**: Introduce non-linearity (e.g., ReLU).

- **Why Effective**:
  - Exploit local correlations (pixels nearby are related).
  - Reduce parameters via weight sharing (unlike dense layers).
  - Scale to large images with deep architectures.

**Example**:
- Task: Classify cats vs. dogs.
- CNN: Learns edges (layer 1), shapes (layer 2), cat faces (layer 5).

**Interview Tips**:
- Emphasize features: “CNNs learn patterns automatically.”
- Highlight invariance: “Handles translations well.”
- Be ready to sketch: “Show conv → pool → dense layers.”

---

## 2. Explain the difference between convolution and pooling layers

**Answer**:

- **Convolutional Layer**:
  - **Purpose**: Extract features by applying learnable filters to input.
  - **How**:
    - Filter (e.g., 3x3) slides over image, computing dot products.
    - Output: Feature map highlighting patterns (e.g., edges).
    - **Math**: `output[i,j] = Σ_m Σ_n input[i+m,j+n] * filter[m,n] + bias`.
  - **Parameters**: Filter weights, biases (learned via backprop).
  - **Pros**: Captures local patterns (e.g., corners, textures).
  - **Cons**: Computationally intensive, many parameters for deep layers.
  - **Example**: Detect vertical edges in an image.

- **Pooling Layer**:
  - **Purpose**: Downsample feature maps to reduce size, improve robustness.
  - **How**:
    - Apply operation (e.g., max, average) over a region (e.g., 2x2).
    - Output: Smaller map (e.g., 2x2 max-pooling halves dimensions).
    - **Types**:
      - Max Pooling: Take maximum value.
      - Average Pooling: Compute mean.
  - **Parameters**: None (fixed operation).
  - **Pros**: Reduces compute, prevents overfitting, adds invariance.
  - **Cons**: Loses some spatial info.
  - **Example**: Max-pool a 4x4 feature map to 2x2, keeping strongest signals.

- **Key Differences**:
  - **Function**: Convolution extracts features; pooling downsamples.
  - **Learnable**: Convolution has weights; pooling is fixed.
  - **Output**: Convolution preserves depth; pooling reduces dimensions.
  - **Use**: Convolution for pattern detection; pooling for efficiency.

**Example**:
- Input: 28x28 image.
- Conv: 3x3 filter → 26x26 feature map (32 filters).
- Pool: 2x2 max-pool → 13x13 map.

**Interview Tips**:
- Clarify roles: “Conv learns, pooling simplifies.”
- Mention trade-offs: “Pooling loses detail but speeds up.”
- Be ready to compute: “Show 3x3 conv on 5x5 input.”

---

## 3. What is transfer learning, and how is it used in computer vision?

**Answer**:

**Transfer learning** involves using a pretrained model (trained on a large dataset) as a starting point for a new task, adapting it with minimal training.

- **How It Works**:
  - **Pretrained Model**: Trained on large dataset (e.g., ImageNet with 1M images).
    - Example: ResNet, VGG, EfficientNet.
  - **Feature Extraction**:
    - Use pretrained layers to extract general features (e.g., edges, shapes).
    - Freeze early layers, keeping weights fixed.
  - **Fine-Tuning** (optional):
    - Train later layers or entire model on new data.
    - Adjust with small learning rate to avoid overfitting.
  - **Output Layer**: Replace final layer (e.g., 1000-class softmax → 2-class).

- **Why Used in Vision**:
  - **Data Scarcity**: Small datasets (e.g., 1000 medical images) benefit from pretrained features.
  - **Speed**: Reduces training time (hours vs. days).
  - **Performance**: Leverages learned patterns (e.g., textures), boosts accuracy.
  - **Accessibility**: Pretrained models are widely available (e.g., PyTorch Hub).

- **Process**:
  1. Load pretrained model (e.g., ResNet50).
  2. Freeze convolutional base.
  3. Replace head (e.g., for 10 classes).
  4. Train on new data, optionally unfreeze layers.

**Example**:
- Task: Classify X-ray images (normal vs. pneumonia).
- Transfer: Use ImageNet-pretrained ResNet, fine-tune on 5000 X-rays.

**Interview Tips**:
- Highlight efficiency: “Saves time and data.”
- Discuss layers: “Early layers are generic, later task-specific.”
- Be ready to code: “Show PyTorch transfer learning.”

---

## 4. What is the difference between object detection and image classification?

**Answer**:

- **Image Classification**:
  - **Goal**: Assign a single label to an entire image.
  - **Input**: Image (e.g., 224x224 pixels).
  - **Output**: Class probability (e.g., “dog” with 0.9).
  - **How**: CNN predicts one label via softmax.
  - **Use Case**: Identify scene type (e.g., beach vs. forest).
  - **Example**: “This is a cat image.”
  - **Models**: ResNet, VGG, EfficientNet.

- **Object Detection**:
  - **Goal**: Identify and localize multiple objects in an image.
  - **Input**: Image.
  - **Output**: Bounding boxes + class labels + confidence scores (e.g., “cat at [x,y,w,h], 0.95”).
  - **How**: Models predict boxes and classes (e.g., via anchors, grids).
  - **Use Case**: Autonomous driving (detect cars, pedestrians).
  - **Example**: “Cat at top-left, dog at bottom-right.”
  - **Models**: YOLO, Faster R-CNN, SSD.

- **Key Differences**:
  - **Scope**: Classification labels whole image; detection localizes objects.
  - **Output**: Classification gives one class; detection gives boxes + classes.
  - **Complexity**: Detection is harder (spatial + classification).
  - **ML Context**: Classification for simple tasks; detection for localization.

**Example**:
- Classification: Image → “positive for tumor.”
- Detection: Image → “tumor at [100,150,50,50].”

**Interview Tips**:
- Clarify output: “Detection adds spatial info.”
- Mention models: “YOLO for detection, ResNet for classification.”
- Be ready to sketch: “Show image with box vs. single label.”

---

## 5. Explain the YOLO algorithm for object detection

**Answer**:

**YOLO (You Only Look Once)** is a real-time object detection algorithm that predicts bounding boxes and class probabilities in a single pass, optimized for speed and accuracy.

- **How It Works**:
  - **Input**: Image (e.g., 416x416).
  - **Grid Division**: Split image into `S x S` grid (e.g., 13x13).
  - **Predictions per Cell**:
    - `B` bounding boxes with coordinates `[x, y, w, h]`.
    - Confidence score: `P(object) * IoU` (Intersection over Union).
    - `C` class probabilities (e.g., dog, cat).
  - **Output**: Tensor of predictions (e.g., `S x S x (B * 5 + C)`).
  - **Post-Processing**:
    - Non-Max Suppression (NMS): Remove overlapping boxes.
    - Threshold confidence to filter weak detections.
  - **Architecture**:
    - CNN backbone (e.g., DarkNet, EfficientNet).
    - Multi-scale predictions (e.g., YOLOv3 detects at different resolutions).

- **Key Features**:
  - **Single Pass**: Unlike two-stage models (e.g., Faster R-CNN), YOLO is fast.
  - **Global Context**: Considers entire image, reducing background errors.
  - **Versions**: YOLOv1 (2015) to YOLOv8 (2023), improving accuracy/speed.

- **Pros**:
  - Fast (e.g., 60 FPS for YOLOv5).
  - Unified model, easy to train.
  - Good for real-time (e.g., video).
- **Cons**:
  - Struggles with small objects or dense scenes.
  - Lower precision than two-stage models.

**Example**:
- Task: Detect cars in traffic cam.
- YOLO: Outputs boxes for each car with class “car” and confidence.

**Interview Tips**:
- Emphasize speed: “YOLO’s single pass is key.”
- Compare: “Vs. Faster R-CNN: faster but less precise.”
- Be ready to sketch: “Show grid and boxes.”

---

## 6. What is semantic segmentation, and how does it differ from instance segmentation?

**Answer**:

- **Semantic Segmentation**:
  - **Goal**: Assign a class label to every pixel in an image.
  - **Input**: Image.
  - **Output**: Pixel-wise class map (e.g., “sky,” “tree,” “road”).
  - **How**: CNNs with upsampling (e.g., encoder-decoder like U-Net).
    - Predict class per pixel via softmax.
  - **Use Case**: Autonomous driving (label road vs. sidewalk).
  - **Example**: Color all cars red, all roads gray.
  - **Models**: U-Net, DeepLab, FCN.

- **Instance Segmentation**:
  - **Goal**: Assign a class label and unique ID to each object instance per pixel.
  - **Input**: Image.
  - **Output**: Pixel-wise map with instance IDs (e.g., “car1,” “car2”).
  - **How**: Combines detection and segmentation (e.g., predict boxes + masks).
  - **Use Case**: Robotics (distinguish individual objects).
  - **Example**: Separate two cars with different colors.
  - **Models**: Mask R-CNN, YOLACT.

- **Key Differences**:
  - **Granularity**: Semantic labels classes; instance labels objects.
  - **Output**: Semantic has one class per pixel; instance has class + ID.
  - **Task**: Semantic is simpler (no instance separation); instance is detection + segmentation.
  - **ML Context**: Semantic for scene understanding; instance for object interaction.

**Example**:
- Image: Two cats.
- Semantic: All cat pixels = “cat.”
- Instance: Cat1 pixels = “cat1,” Cat2 = “cat2.”

**Interview Tips**:
- Clarify instances: “Instance segmentation tracks individuals.”
- Mention models: “U-Net for semantic, Mask R-CNN for instance.”
- Be ready to sketch: “Show pixel labels vs. instance masks.”

---

## 7. What are some common data augmentation techniques for computer vision?

**Answer**:

**Data augmentation** artificially increases dataset size by applying transformations to images, improving model robustness and generalization.

- **Geometric Transformations**:
  - **Rotation**: Rotate image (e.g., ±30°).
    - **Why**: Handles tilted objects.
  - **Translation**: Shift image (e.g., ±10 pixels).
    - **Why**: Objects not centered.
  - **Scaling/Zoom**: Resize (e.g., 0.8x to 1.2x).
    - **Why**: Varying object sizes.
  - **Flipping**: Horizontal/vertical flip.
    - **Why**: Symmetry (e.g., left/right faces).
  - **Shearing**: Skew image.
    - **Why**: Perspective changes.
- **Color Transformations**:
  - **Brightness**: Adjust intensity (e.g., ±20%).
    - **Why**: Lighting variations.
  - **Contrast**: Stretch intensity range.
    - **Why**: Different exposures.
  - **Hue/Saturation**: Alter colors.
    - **Why**: Color shifts (e.g., sunlight).
- **Noise and Blur**:
  - **Gaussian Noise**: Add random noise.
    - **Why**: Sensor imperfections.
  - **Blur**: Apply Gaussian blur.
    - **Why**: Out-of-focus images.
- **Cutout/Dropout**:
  - Randomly mask patches.
    - **Why**: Forces focus on other regions.
- **Mixup/Cutmix**:
  - Blend images/labels (Mixup) or paste patches (Cutmix).
    - **Why**: Improves generalization.
- **Task-Specific**:
  - Crop for detection (focus on objects).
  - Elastic distortions for OCR (mimic handwriting).

**Example**:
- Task: Classify dogs.
- Augmentation: Rotate, flip, adjust brightness → model handles varied poses/lighting.

**Interview Tips**:
- Link to robustness: “Augmentation mimics real-world noise.”
- Balance: “Too much augmentation distorts data.”
- Be ready to code: “Show PIL/PyTorch augmentation.”

---

## 8. How do you evaluate the performance of a computer vision model?

**Answer**:

Evaluating computer vision models depends on the task, using metrics to measure accuracy, robustness, and generalization:

- **Image Classification**:
  - **Accuracy**: Fraction of correct predictions.
  - **Precision/Recall/F1**:
    - Precision: `TP / (TP + FP)` (correct positives).
    - Recall: `TP / (TP + FN)` (captured positives).
    - F1: `2 * (P * R) / (P + R)`.
  - **Top-k Accuracy**: Correct class in top k predictions.
  - **Use**: F1 for imbalanced data (e.g., rare diseases).
- **Object Detection**:
  - **mAP (Mean Average Precision)**:
    - Compute AP per class at IoU threshold (e.g., 0.5).
    - Average across classes.
  - **IoU**: `Intersection / Union` of predicted vs. true boxes.
  - **Use**: mAP@0.5 for standard detection (e.g., COCO).
- **Semantic Segmentation**:
  - **Pixel Accuracy**: Fraction of correct pixels.
  - **mIoU (Mean Intersection over Union)**:
    - Average IoU per class: `IoU = TP / (TP + FP + FN)`.
  - **Use**: mIoU for class imbalance (e.g., Cityscapes).
- **Instance Segmentation**:
  - **mAP with Masks**: Like detection, but for mask IoU.
  - **Use**: COCO-style mAP for Mask R-CNN.
- **General Metrics**:
  - **Confusion Matrix**: Analyze errors (e.g., false positives).
  - **ROC-AUC**: For binary tasks, measure threshold trade-offs.
- **Non-Metric Evaluation**:
  - **Visualization**: Inspect predictions (e.g., bounding boxes, masks).
  - **Error Analysis**: Identify failure modes (e.g., small objects missed).
- **Practical Considerations**:
  - **Latency**: Ensure real-time performance (e.g., <100ms).
  - **Robustness**: Test on augmented/noisy data.
  - **Domain-Specific**: Align metrics with goals (e.g., safety for autonomous driving).

**Example**:
- Task: Detect pedestrians.
- Metrics: mAP@0.5 = 0.85, inference time = 50ms.
- Analysis: Check missed small pedestrians.

**Interview Tips**:
- Match metric to task: “mAP for detection, mIoU for segmentation.”
- Discuss trade-offs: “Accuracy vs. latency matters.”
- Be ready to compute: “Show IoU formula.”

---

## Notes

- **Focus**: Answers cover core computer vision concepts, ideal for ML interviews.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Depth**: Includes technical details (e.g., YOLO grid, CNN math) and practical tips (e.g., augmentation pipelines).
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, implement vision models (see [ML Coding](ml-coding.md)) or explore [Deep Learning](deep-learning.md) for CNN foundations. 🚀

---

**Next Steps**: Build on these skills with [Natural Language Processing](natural-language-processing.md) for multimodal tasks or revisit [Production MLOps](production-mlops.md) for deploying vision models! 🌟