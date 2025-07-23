# **Intrusion Detection System for 5G Networks**

## **Objective**

This project focuses on building an effective Intrusion Detection System (IDS) for 5G networks. With 5G enabling massive device connectivity and faster communication, it also opens the door to a wide range of security threats. Traditional IDS solutions often struggle to keep up with the volume and variety of data traffic in such networks. This project aims to develop an IDS that can detect attacks accurately, adapt to unknown threats and also provide meaningful explanations for its decisions, making it practical for real-time use and analysis.

## **Motivation**

5G technology has revolutionized connectivity, enabling ultra-fast internet, IoT communication, and smart infrastructure. However, this also introduces unique security challenges like massive data throughput, low latency demands, and complex heterogeneous devices. With these increased vulnerabilities, it becomes crucial to adopt intelligent systems that can not only detect attacks quickly but also explain their decisions.

Motivation behind using both deep learning and explainable AI techniques is to bridge the gap between high model accuracy and human understanding, ensuring that the IDS is trustworthy, efficient, and applicable in real-world use cases.

## **Methodology**

### **Pipeline 1: Explainable Deep Learning with Gradient Boosting**

1. **Feature Selection**: Used Boruta with a Random Forest base estimator to filter out the most significant features from the dataset.
2. **Data Balancing**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
3. **Feature Extraction using DNN**:

   *Model architecture: Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Output (Softmax)
   * The model extracts high-level representations from the tabular input data.
4. **Classification**: Gradient Boosting classifier is trained on the extracted DNN features.
5. **Explainability**: SHAP (SHapley Additive exPlanations) is used to interpret the model predictions and assess feature importance.

### **Pipeline 2: CNN-Based Classification with SHAP**

1. **Feature Selection**: Used Boruta to reduce feature dimensionality.
2. **Data Balancing**: Handled using SMOTE.
3. **Classification using CNN**:

   *Model architecture: Reshaped Input → Conv1D (32 filters, ReLU) → MaxPooling → Conv1D (64 filters, ReLU) → MaxPooling → Flatten → Dense(128, ReLU) → Dropout(0.1) → Output (Sigmoid)
   
   * Adapted CNN for tabular data to leverage its ability to extract spatial correlations in structured inputs.
5. **Explainability**: SHAP is again applied to evaluate and visualize the model’s predictions.

## **Results**

| Pipeline | Classifier        | Accuracy | Precision | Recall | F1-Score |
| -------- | ----------------- | -------- | --------- | ------ | -------- |
| 1        | Gradient Boosting | 99.97%   | 99.94%    | 100%   | 99.97%   |
| 2        | CNN Classifier    | 99.91%   | 99.82%    | 100%   | 99.91%   |

## **Tech Stack and Tools**

* Python
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* SHAP
* SMOTE
* Matplotlib, Seaborn

## **Dataset**

**Name**: TII-SSRC-23 Dataset

**Source**: [Kaggle - TII-SSRC-23](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23)

**Description**:
The dataset contains network traffic data in two formats:

* Tabular format (CSV) with pre-extracted features
* Raw network traffic (PCAP files)

In this project, the **CSV file** is used for building and evaluating the intrusion detection system.

## **Explainability**

SHAP was used in both pipelines to provide transparency in model predictions. It generates visualizations that highlight how much each feature contributes to the final decision. The use of SHAP increases trust and allows network analysts to understand the behavior of the model in real-world scenarios.

## **Conclusion**

This project demonstrates the use of hybrid deep learning models and explainability techniques to create a high-performing and interpretable IDS for 5G environments. Through two robust pipelines, it combines the strengths of neural networks, ensemble models, and XAI methods to address security needs of next-gen networks. The modularity of the pipelines also allows easy adaptation to different datasets and evolving threats.

## **Contact**

For questions, feedback, or collaboration related to this project, feel free to reach out:

**Aashuti Gambhir**

Email: [atechtrek@gmail.com](mailto:atechtrek@gmail.com)

GitHub: [Aashuti-Tech-Trek](https://github.com/Aashuti-Tech-Trek)
