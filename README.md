# MSDS565--Images--TeamKAI

## Overview
This project implements an end-to-end CRISP-DM computer vision pipeline using a custom web-scraped image dataset. It covers data collection, manual cleaning and exploratory data analysis (EDA), transfer learning across pretrained Keras applications of varying sizes, model interpretability (Saliency, SmoothGrad, and Grad-CAM), and deployment via a live Streamlit web application.

---

## Repository Structure
```text
MSDS565-Images-<TeamKAI>/
├── README.md
├── environment.yml
├── data/
│   └── P2_manifest.csv                  # Record of scraped URLs, labels, and cleaning status (NO images)
├── models/
│   ├── P2_N3_<small_app>.keras          # Trained linear probe for small model
│   ├── P2_N3_<medium_app>.keras         # Trained linear probe for medium model
│   ├── P2_N3_<large_app>.keras          # Trained linear probe for large model
│   └── [label_mapping_file]             # Saved class index mapping
├── notebooks/
│   ├── ddgs_scraper.py                  # Provided scraping module
│   ├── image_explainers.py              # Provided xAI module
│   ├── P2_Images-1_Scrape_<TeamName>.ipynb
│   ├── P2_Images-2_Preprocessing_<TeamName>.ipynb
│   ├── P2_Images-3_TransferLearning_<TeamName>.ipynb
│   └── P2_Images-4_xAI_<TeamName>.ipynb
└── app/
    ├── image_explainers.py              # Copy of xAI module for Streamlit deployment
    └── P2_Images-5_App_<TeamName>.py    # Streamlit web application
