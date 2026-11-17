
# MSDS565 — Project 2: Image Classification

**Team:** TeamKAI — KeiTonia, Ashley, India  
**Course:** MSDS 565 — Predictive Modeling and Analytics  
**Project:** Image Classification Using Transfer Learning and Explainable AI

---

## Project Overview

TeamKAI developed an end-to-end computer vision pipeline to classify five types of clouds using a custom web-scraped image dataset.

Following the CRISP-DM framework, the project incorporates image collection, data cleaning, exploratory data analysis (EDA), preprocessing, transfer learning, model evaluation, explainable artificial intelligence (XAI), and deployment through a Streamlit web application.

Four pretrained Keras convolutional neural network architectures were evaluated:

- MobileNetV2
- EfficientNetV2S
- ResNet50
- EfficientNetV2M

The project also applies Saliency, SmoothGrad, and Grad-CAM to investigate model predictions and identify potentially misleading visual features.

**Best Recorded Model:** ResNet50 achieved 81.0% test accuracy and a macro-F1 score of 0.811.

---

## Repository Structure

```text
MSDS565--Images--TeamKAI/
│
├── README.md
├── environment.yml
│
├── data/
│   └── P2_manifest.csv
│
├── models/
│   ├── P2_N3_mobilenetv2.keras
│   ├── P2_N3_effnetv2s.keras
│   ├── P2_N3_resnet50.keras
│   ├── P2_N3_effnetv2m.keras
│   └── P2_label_map.json
│
├── notebooks/
│   ├── ddgs_scraper.py
│   ├── image_explainers.py
│   ├── P2_Images-1_Scrape_TeamKAI.ipynb
│   ├── P2_Images-2_Preprocessing_TeamKAI.ipynb
│   ├── P2_Images-3_TransferLearning_TeamKAI.ipynb
│   └── P2_Images-4_xAI_TeamKAI.ipynb
│
└── app/
    ├── image_explainers.py
    └── P2_Images-5_App_TeamKAI.py
```

The repository contains the project notebooks, dataset manifest, trained classification probes, label mapping, application code, and environment configuration.

Raw images, cleaned images, and preprocessed NumPy arrays are maintained locally and are not included in the GitHub repository.

---

## 1. Chosen Classes and Rationale

Five cloud categories were selected based on their recognizable visual characteristics and potential classification challenges.

| Cloud Class | Visual Characteristics |
|---|---|
| Altocumulus | Small, rounded cloud patches that frequently appear in groups |
| Cirrus | Thin, wispy, feather-like cloud formations |
| Cumulonimbus | Large, vertically developed clouds commonly associated with thunderstorms |
| Cumulus | Puffy clouds with relatively distinct shapes and boundaries |
| Stratus | Broad, flat cloud layers that frequently produce overcast skies |

These categories allow the models to distinguish differences in cloud shape, texture, density, and structure.

However, variations in lighting, perspective, camera angle, and cloud formations create realistic classification challenges.

### Expected Confusion Pair

Before modeling, we predicted that **Cumulus and Altocumulus** would be the most frequently confused classes because both exhibit rounded, puffy formations.

Differences in altitude and scale are not always apparent in individual photographs.

Notebook 3 did not identify this combination as the most frequently confused pair for any of the four models.

---

## 2. Dataset Construction and Search Terms

Images were collected using the instructor-provided `ddgs_scraper.py` module with the Bing image-search backend.

Three search queries were used for each class to increase image diversity.

### Image Collection Summary

| Class | Search Queries | Images Collected |
|---|---|---:|
| Cumulus | `cumulus clouds`; `fair weather cumulus sky`; `cumulus humilis clouds` | 400 |
| Stratus | `stratus clouds overcast`; `stratus cloud layer sky`; `low stratus fog clouds sky` | 400 |
| Cirrus | `cirrus clouds`; `wispy cirrus clouds sky`; `cirrus fibratus clouds` | 400 |
| Cumulonimbus | `cumulonimbus clouds`; `towering cumulonimbus thunderstorm`; `cumulonimbus incus cloud` | 400 |
| Altocumulus | `altocumulus clouds`; `altocumulus cloud patches sky`; `altocumulus stratiformis clouds` | 400 |
| **Total** | **15 search queries** | **2,000** |

### Scraper Configuration

- Minimum image resolution: 224 × 224 pixels
- Acceptable aspect ratio: 0.5–2.0
- Maximum requested images per class: 400
- Per-image timeout: 10 seconds
- Moderate safe search
- US-English search region

Notebook 1 records 2,000 downloaded and readable images across the five categories.

### Dataset Manifest

The committed file `data/P2_manifest.csv` contains image metadata, including:

- Image filenames
- Cloud class labels
- Search queries
- Image and source URLs
- Image dimensions
- Readability indicators
- Cleaning and retention status
- Training and testing split assignments

### Cleaning and Retention Summary

The committed manifest documents the following dataset counts:

| Dataset Stage | Image Count |
|---|---:|
| Initial image collection | 2,000 |
| Records in committed manifest | 1,976 |
| Retained images | 1,603 |
| Excluded images | 373 |
| Training images | 1,282 |
| Testing images | 321 |

The committed manifest contains 24 fewer records than the initial collection. The available artifacts do not establish the reason for this difference.

Of the 1,976 manifest records, 1,603 were retained and 373 were marked as excluded.

The retained images were divided into training and testing sets using an approximately 80/20 split.

**Important:** Notebook 3's saved evaluation used 400 test images, whereas the committed manifest documents 321 test images. These represent different recorded dataset states and should not be treated as identical evaluation sets.

---

## 3. Project Pipeline

The project consists of four Jupyter notebooks documenting dataset construction, preprocessing, transfer learning, and explainability.

### Notebook 1 — Dataset Construction

**File:** `notebooks/P2_Images-1_Scrape_TeamKAI.ipynb`

**Purpose:** Collect a custom dataset representing five cloud categories.

**Methodology:**

1. Define the five image classes.
2. Establish three search queries per class.
3. Collect images using the instructor-provided scraper.
4. Validate image readability and dimensions.
5. Record image metadata and source information.
6. Export a dataset manifest.

**Primary Output:** `P2_manifest.csv`

The notebook documents the initial collection of 2,000 readable images.

### Notebook 2 — Cleaning, Preprocessing, and EDA

**File:** `notebooks/P2_Images-2_Preprocessing_TeamKAI.ipynb`

**Purpose:** Clean and prepare images for consistent model training and evaluation.

**Methodology:**

- Review images for unsuitable content and duplicates.
- Examine image dimensions and class distributions.
- Convert images to three-channel RGB.
- Resize images to 224 × 224 pixels.
- Preserve aspect ratios using letterbox resizing.
- Center images on a gray canvas with padding value 128.
- Perform a stratified training/testing split.
- Use `random_state=42` for reproducibility.
- Save processed image arrays and labels.

**Locally Generated Outputs:**

```text
x_train.npy
y_train.npy
x_test.npy
y_test.npy
label_map.json
```

These NumPy arrays are stored locally rather than committed to the repository.

The committed dataset manifest documents 1,603 retained images, including 1,282 training images and 321 testing images.

### Notebook 3 — Transfer Learning and Model Evaluation

**File:** `notebooks/P2_Images-3_TransferLearning_TeamKAI.ipynb`

**Purpose:** Compare pretrained CNN architectures using transfer learning.

Four pretrained Keras applications were evaluated.

| Model | Architecture Size |
|---|---|
| MobileNetV2 | Small |
| EfficientNetV2S | Medium |
| ResNet50 | Medium |
| EfficientNetV2M | Large |

**Methodology:**

1. Load preprocessed training and testing arrays.
2. Verify class distributions.
3. Apply training-only augmentation.
4. Use pretrained Keras models as frozen feature extractors.
5. Extract image embeddings.
6. Train a single-layer softmax classification probe.
7. Evaluate predictions on held-out test images.
8. Compare accuracy, macro-F1, and confusion matrices.

Training augmentation includes:

- Horizontal flipping
- Rotation
- Zoom
- Brightness adjustments
- Contrast adjustments

The notebook also includes an oversampling procedure to address potential class imbalance.

For the recorded Notebook 3 run, the training data contained 320 images per class, so oversampling did not alter the balanced class distribution.

### Model Performance

Notebook 3 reports the following results using 400 test images, with 80 images per class.

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| MobileNetV2 | 77.2% | 0.772 |
| EfficientNetV2S | 76.7% | 0.766 |
| **ResNet50** | **81.0%** | **0.811** |
| EfficientNetV2M | 73.0% | 0.728 |

**Key Findings:**

- ResNet50 achieved the highest recorded test accuracy and macro-F1.
- MobileNetV2 performed competitively despite its smaller architecture.
- EfficientNetV2M, the largest architecture evaluated, produced the lowest recorded accuracy.
- Increasing model size did not consistently improve classification performance.

These results demonstrate that larger pretrained architectures do not necessarily outperform smaller alternatives on a particular dataset.

### Evaluation Limitation

The saved Notebook 3 analysis identified 10 test images identical to training images.

This overlap creates a potential data-leakage concern and may influence the reported evaluation metrics.

Therefore, the 81.0% accuracy should be interpreted as the best recorded result from this experiment rather than a fully leakage-free estimate of generalization performance.

A future evaluation should remove overlapping images before creating the train/test split.

### Saved Models

The following trained classification probes are included in the repository:

```text
models/P2_N3_mobilenetv2.keras
models/P2_N3_effnetv2s.keras
models/P2_N3_resnet50.keras
models/P2_N3_effnetv2m.keras
models/P2_label_map.json
```

The label mapping assigns class indices as follows:

```json
{
  "altocumulus": 0,
  "cirrus": 1,
  "cumulonimbus": 2,
  "cumulus": 3,
  "stratus": 4
}
```

### Notebook 4 — Explainable AI and Shortcut Audit

**File:** `notebooks/P2_Images-4_xAI_TeamKAI.ipynb`

**Purpose:** Investigate how the trained models make predictions and whether they rely on meaningful cloud features or potentially misleading visual patterns.

Three explainability techniques are used.

**1. Raw Saliency**

Highlights image regions where changes in pixel values influence the model's predictions.

**2. SmoothGrad**

Averages saliency maps across multiple noisy versions of an image to produce more stable visual explanations.

The notebook configures SmoothGrad to use 32 samples.

**3. Grad-CAM**

Uses convolutional feature activations to highlight image regions contributing to a predicted class.

### Additional Explainability Analyses

Notebook 4 includes investigations of:

- Correctly classified examples
- Misclassified examples
- Cross-model explanations
- Background and border effects
- Potential watermarks and banners
- Occlusion-based ablation tests
- Potential classification shortcuts

### Explainability Limitations

The saved Notebook 4 analysis uses 321 test images, compared with the 400 test images recorded in Notebook 3.

Because these evaluations use different recorded test sets, their results should not be interpreted as directly comparable without reconciling the underlying data.

The notebook also contains discussion prompts that require final interpretation.

Explainability maps provide useful diagnostic information but do not establish definitive causal explanations for model predictions.

---

## 4. Streamlit Web Application

**File:** `app/P2_Images-5_App_TeamKAI.py`

The Streamlit application provides an interactive interface for cloud image classification and model explanation.

### Application Features

**Image Upload**

Users can upload cloud photographs for classification.

Uploaded images are converted to RGB and resized to 224 × 224 pixels using aspect-ratio-preserving letterbox resizing.

**Model Comparison**

The application uses the pretrained feature extractors and saved classification probes to generate predictions.

It provides:

- Predicted cloud categories
- Class probability distributions
- Comparisons across the four models
- Raw Saliency explanations
- SmoothGrad explanations
- Grad-CAM explanations

**Prediction Diagnostics**

The application also includes:

- Model agreement and disagreement information
- Low-confidence warnings
- Prediction uncertainty summaries
- Information about classifier limitations

Confidence warnings use heuristic thresholds and should not be interpreted as guarantees of calibrated prediction probabilities.

---

## 5. Environment Setup and Reproducibility

The repository includes `environment.yml`, which documents the Conda environment used for the project.

The committed environment is named `msds565`.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/IKScruggs/MSDS565--Images--TeamKAI.git
```

```bash
cd MSDS565--Images--TeamKAI
```

**Note:** The repository is private. Cloning requires authorized GitHub access.

### Step 2 — Create the Conda Environment

```bash
conda env create -f environment.yml
```

### Step 3 — Activate the Environment

```bash
conda activate msds565
```

The environment file is a full environment export and may include platform-specific dependencies.

Environment creation on another computer has not been independently verified.

### Step 4 — Launch the Streamlit Application

From the repository root directory:

```bash
streamlit run app/P2_Images-5_App_TeamKAI.py
```

Streamlit should open the application in a browser when the environment and required dependencies are correctly configured.

### Reproducibility Notes

The repository includes:

- Four Jupyter notebooks
- Dataset manifest and metadata
- Four saved Keras classification probes
- Class label mapping
- Streamlit application
- Explainability support code
- Conda environment configuration

The repository does not include the raw image collection or locally generated NumPy arrays.

Therefore, reproducing the entire training pipeline requires access to the original image data or reconstruction of the dataset through the documented scraping and preprocessing procedures.

Because web image-search results can change, a new scrape may not reproduce the original images.

The committed model files support inference workflows, but complete application functionality and cross-platform environment compatibility have not been independently verified.

---

## 6. Project Findings

TeamKAI developed a computer vision workflow that integrates image scraping, preprocessing, transfer learning, model comparison, explainable AI, and interactive deployment.

The principal findings are:

1. **Transfer learning was effective.** Pretrained CNN feature extractors and simple classification probes produced meaningful performance across five cloud categories.

2. **ResNet50 performed best.** ResNet50 achieved the highest recorded test accuracy of 81.0% and macro-F1 of 0.811.

3. **Larger models were not always better.** EfficientNetV2M underperformed the smaller architectures in the recorded evaluation.

4. **Cloud classification presents visual ambiguity.** Differences in cloud shape, lighting, texture, and scale create challenges for automated classification.

5. **Explainability adds diagnostic value.** Saliency, SmoothGrad, and Grad-CAM provide complementary ways to examine model attention and potential visual shortcuts.

6. **Dataset quality and evaluation consistency matter.** The recorded train/test overlap and differences between notebook evaluation sets highlight the importance of maintaining consistent, leakage-free datasets.

---

## 7. Project Limitations and Future Improvements

Several limitations should be considered when interpreting the results.

### Dataset Limitations

The images were obtained through web searches and may contain differences in image quality, lighting, perspective, and background composition.

### Train/Test Overlap

Notebook 3 identified 10 identical images across training and testing sets, creating a potential data-leakage concern.

### Evaluation Dataset Differences

Notebook 3 reports 400 test images, while the committed manifest and Notebook 4 document 321 test images.

These differences should be reconciled before performing direct comparisons across evaluation stages.

### Limited Class Coverage

The classifier recognizes five cloud categories and does not include an explicit unknown-cloud or out-of-distribution class.

### Model Confidence

Softmax probabilities should not automatically be interpreted as calibrated confidence estimates.

### Future Improvements

Potential extensions include:

- Rebuilding a consistent, leakage-free dataset
- Re-evaluating all four models on the same held-out test set
- Expanding the number of cloud categories
- Improving image quality and dataset diversity
- Evaluating probability calibration
- Conducting additional shortcut and robustness analyses
- Testing application portability across environments

---

## 8. Conclusion

TeamKAI completed an image-classification project combining custom dataset construction, preprocessing, pretrained CNN architectures, model evaluation, explainable AI, and Streamlit application development.

Among the four evaluated architectures, ResNet50 achieved the strongest recorded performance, with 81.0% accuracy and a macro-F1 score of 0.811.

The results demonstrate the usefulness of transfer learning for custom image-classification tasks while emphasizing the importance of data quality, consistent evaluation procedures, and model interpretability.

The project also illustrates how explainable AI techniques can support further investigation of classification decisions and potential model limitations.

---

## TeamKAI

**Course:** MSDS 565 — Predictive Modeling and Analytics

**Team Members:**

- KeiTonia
- Ashley
- India

---

**Project Repository:**  
https://github.com/IKScruggs/MSDS565--Images--TeamKAI

**Repository Visibility:** Private — authorized access required.
