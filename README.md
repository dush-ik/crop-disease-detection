**Capstone Project Proposal**

**Title: Crop Disease Classification Using Transfer Learning**

**Data Efficiency, Confidence Calibration and Explainable Disease Localisation**

**Group Number: 5**

Team Members:

1\. Deepthi Prakash

2\. Shahina Pathan

3\. Venkata S. Sarma

4\. Dushyant Kumar

# **Abstract**

Crop diseases cause substantial agricultural losses and disproportionately affect smallholder farmers who may have limited access to timely agronomic expertise. Smartphone-based image classification can provide a scalable and comparatively low-cost method for preliminary disease screening. This project proposes a convolutional neural network classifier for identifying 38 crop-disease categories from leaf photographs in the PlantVillage dataset. A ResNet-50 model pretrained on ImageNet will be fine-tuned through transfer learning and evaluated primarily using macro-F1 score, with a target of at least 0.88.

The project goes beyond a single benchmark score. It will investigate the amount of labelled data required to approach full-dataset performance by training the same model on stratified subsets of 10,000, 20,000, 30,000 and all available training images. Reliability will be examined through confidence calibration, reliability diagrams, Expected Calibration Error and temperature scaling. Grad-CAM heatmaps will be used to assess whether the classifier focuses on disease symptoms rather than backgrounds. Field-oriented augmentation, including variable lighting, Gaussian and motion blur, occlusion, perspective distortion and possible background replacement, will be used to reduce the gap between controlled laboratory images and real smartphone photographs.

The expected outputs are a fine-tuned ResNet-50 classifier, a baseline comparison, exploratory data analysis, a data-efficiency curve, calibration analysis, Grad-CAM visualisations, robustness results and a deployment brief for an agricultural extension service.

# **Table of Contents**

1\. Introduction

2\. Problem Statement

3\. Literature Review

4\. Proposed Methodology

5\. Tools and Deployment

6\. Expected Outcomes

7\. Applicability in the Real World

8\. Challenges and Risk Mitigation

9\. Week-wise Plan

10\. References

# 

# **1\. Introduction**

Plant diseases threaten food security by reducing crop quality and yield. The problem is especially serious for smallholder farmers because diagnosis often depends on the availability of trained agricultural officers or plant pathologists. Delayed or incorrect identification can lead to crop damage, avoidable expenditure and inappropriate pesticide use.

The increasing availability of smartphones creates an opportunity for image-based decision-support systems. A farmer or agricultural extension worker can photograph a symptomatic leaf and receive a preliminary prediction within seconds. Deep convolutional neural networks can learn visual patterns such as leaf spots, discolouration, lesions, rust and blight directly from labelled images.

This project will use the PlantVillage dataset, containing approximately 54,306 images distributed across 38 crop-disease classes involving 14 crop species. The proposed classifier will be based on ResNet-50 and transfer learning. The project will also address a major limitation of laboratory datasets: photographs collected under controlled backgrounds and lighting may not represent field conditions. Therefore, benchmark performance will be supplemented with robustness, calibration and explainability analyses.

# **2\. Problem Statement**

Although deep-learning models can achieve very high accuracy on controlled leaf-image datasets, their practical value depends on more than test accuracy. A model trained on uniform backgrounds may learn shortcuts that fail under natural lighting, blur, clutter, occlusion and different camera angles. It may also be overconfident when it is wrong.

The central problem is therefore to develop an accurate crop disease classifier while determining: (i) how much labelled data is required to reach deployment-oriented performance, (ii) whether the reported confidence scores can be trusted, (iii) whether the model attends to genuine disease symptoms, and (iv) how strongly performance degrades under field-like image variation.

## **2.1 Objectives**

Objective one: Develop and train a baseline classification pipeline by fine-tuning a pre-trained ResNet-50 model on the PlantVillage dataset, successfully mapping all 54,306 images to their respective 38 crop-disease categories.

Objective two: Achieve a macro-F1 score of ≥0.88 on the held-out test set and plot a data efficiency curve to explicitly quantify the minimum amount of labelled data required to maintain deployment-ready accuracy.

Objective three: Generate Grad-CAM heatmaps to visually verify accurate disease lesion localization and deliver a confidence calibration analysis alongside a structured field deployment brief for a mobile agricultural extension service.

# **3\. Literature Review**

## **3.1 Prior Work**

**1\. Hughes & Salathé (2015)**

* **What they did:** They created the open-access PlantVillage dataset, assembling over 50,000 expertly curated, laboratory-controlled images of healthy and diseased crop leaves.  
* **Relevance:** This dataset serves as the foundational data source used to train and test the ResNet-50 classifier in this capstone project.

**2\. Mohanty, Hughes, & Salathé (2016)**

* **What they did:** They trained deep convolutional neural networks on the PlantVillage dataset, achieving over 99% accuracy and proving the technical feasibility of smartphone-based disease diagnosis.  
* **Relevance:** This provides the baseline proof-of-concept for our approach, which we will build upon by moving beyond simple accuracy to tackle real-world deployment challenges.

**3\. Singh et al. (2020)**

* **What they did:** They introduced the PlantDoc dataset consisting of real-world, noisy field images to demonstrate how models trained purely on controlled lab data suffer massive performance drops in practical environments.  
* **Relevance:** This paper directly justifies our project's requirement to confront distribution shift and evaluate how well lab-trained models generalize to variable-lighting field conditions.

## **3.2 Methods to build upon**

**4\. Lin et al. (2017)**

* **What they did:** They proposed Focal Loss, a loss function that dynamically scales based on prediction confidence to force models to learn hard, misclassified examples in imbalanced datasets.  
* **Relevance:** We will utilize Focal Loss to prevent majority classes (like healthy tomatoes) from dominating training, ensuring we hit our strict target of macro-F1 ≥ 0.88 across all 38 categories.

**5\. Guo et al. (2017)**

* **What they did:** They discovered that modern neural networks are frequently overconfident in their predictions and provided methods like temperature scaling to align predicted probabilities with actual accuracy.  
* **Relevance:** This research forms the exact methodological basis for the confidence calibration analysis required before we can recommend our model to an agricultural extension service.

# 

# **4\. Proposed Methodology**

## **4.1 Data Preprocessing**

* Load and organise the PlantVillage dataset containing approximately 54,306 images across 38 crop-disease classes.  
* Check the dataset for corrupted, duplicate and near-duplicate images using image hashing techniques.  
* Analyse class distribution, image dimensions, aspect ratios and background uniformity.  
* Split the dataset into training, validation and testing sets using stratified sampling.  
* Resize all images to a fixed input size, such as 512 × 512 pixels.  
* Normalise the images using the mean and standard deviation calculated from the training set.  
* Apply data augmentation techniques such as brightness and contrast variation, Gaussian blur, motion blur, horizontal flipping, perspective transformation and partial occlusion.

### **4.2 Model Architecture**

* Implement a basic Convolutional Neural Network as the baseline model.  
* Use a pretrained ResNet-50 model for crop disease classification.  
* Replace the final classification layer of ResNet-50 with a new layer containing 38 output classes.  
* Use transfer learning to make use of features learned from the ImageNet dataset.  
* Apply Grad-CAM to highlight the regions of the leaf that influence the model’s prediction.

### **4.3 Training**

* Initially freeze the pretrained ResNet-50 layers and train only the new classification layer.  
* Unfreeze the final ResNet layers and fine-tune the model using a lower learning rate.  
* Use categorical cross-entropy as the initial loss function.  
* Apply weighted cross-entropy or focal loss if significant class imbalance is identified.  
* Use AdamW as the optimiser with early stopping and learning-rate scheduling.  
* Train the model using different quantities of labelled data, including 10,000, 20,000, 30,000 and the complete training dataset.

### **4.4 Evaluation**

* ## Evaluate the model using accuracy, precision, recall, per-class F1 score and macro-F1 score.

* ## Calculate the F1 score for each class using:

        F1ci \= (2 x Pci x Rci) / (Pci \+ Rci)

* ## Calculate macro-F1 as the average F1 score across all 38 classes:

        Macro-F1 \= (F1c1 \+ F1c2 \+ ... \+ F1c38) / 38

* ## The expected target performance is:

        Macro-F1 \>= 0.88

* ## Generate a confusion matrix to identify commonly misclassified crop diseases.

* ## Plot a data-efficiency curve comparing model performance with 10,000, 20,000, 30,000 and all available training samples.

* ## Perform confidence-calibration analysis using reliability diagrams and Expected Calibration Error.

* ## Use Grad-CAM visualisations to verify whether the model focuses on diseased leaf regions.

* ## Test the model on images with lighting changes, blur, occlusion and background variation to assess its robustness under field-like conditions.


# **5\. Tools and Deployment**

## **5.1 Tools**

* Python, NumPy and Pandas for data handling.  
* PyTorch or TensorFlow/Keras for model development.  
* OpenCV and Pillow for image inspection and preprocessing.  
* torchvision or Albumentations for augmentation.  
* scikit-learn for splits, metrics and calibration support.  
* Matplotlib for EDA and result visualisation.  
* Google Colab Pro for GPU training.  
* GitHub for version control, collaboration and experiment tracking.

## **5.2 Prototype Deployment**

A lightweight demonstration may be created using Streamlit or Gradio. The interface will accept a leaf image and display the predicted class, confidence, top alternatives and a Grad-CAM overlay. The prototype will be presented as decision support rather than a treatment-prescription system.

## **6\. Possible Outcomes in Stages**

### **Stage 1: Data Preparation and Preliminary Analysis**

* Acquire and organise the PlantVillage dataset.  
* Perform exploratory data analysis on class distribution, image dimensions, aspect ratios and background uniformity.  
* Identify corrupted, duplicate and near-duplicate images.  
* Create stratified training, validation and test datasets.  
* Complete image preprocessing and augmentation.  
* Conduct preliminary experiments using a baseline CNN model.

### **Stage 2: Model Development**

* Implement a pretrained ResNet-50 model using transfer learning.  
* Replace the original classification layer with a new output layer for the 38 crop-disease classes.  
* Train the classification head while keeping the pretrained layers frozen.  
* Fine-tune the final ResNet-50 layers using a lower learning rate.  
* Compare the performance of the baseline CNN, frozen ResNet-50 and fine-tuned ResNet-50 models.

### 

### **Stage 3: Model Evaluation and Tuning**

* Evaluate model performance on the validation set using accuracy, precision, recall and macro-F1 score.  
* Fine-tune hyperparameters such as learning rate, batch size, number of epochs and augmentation settings.  
* Apply class weights or focal loss if class imbalance affects model performance.  
* Generate confusion matrices and per-class precision, recall and F1 scores.  
* Select the best-performing model based on validation macro-F1 score.

### **Stage 4: Data-Efficiency and Advanced Analysis**

* Train the selected ResNet-50 model using 10,000, 20,000, 30,000 and the complete training dataset.  
* Generate a data-efficiency curve showing the relationship between the number of labelled images and model performance.  
* Conduct confidence-calibration analysis using reliability diagrams and Expected Calibration Error.  
* Apply temperature scaling if the model produces overconfident predictions.  
* Generate Grad-CAM visualisations to identify the leaf regions influencing disease predictions.  
* Evaluate model robustness using images with lighting variations, Gaussian blur, motion blur, occlusion and perspective changes.

### **Stage 5: Final Model Testing and Documentation**

* Evaluate the final selected model on the untouched PlantVillage test set.  
* Verify whether the model achieves the target macro-F1 score of at least 0.88.  
* Compare clean-image performance with performance under field-like transformations.  
* Analyse incorrectly classified images and commonly confused crop-disease classes.  
* Prepare the final project report, results, graphs and visualisations.  
* Organise the source code and experiment notebooks in the shared GitHub repository.  
* Prepare the final project presentation and demonstration of model predictions.

# **7\. Applicability in the Real World**

The proposed system can support agricultural extension workers by prioritising cases, providing a preliminary disease category and identifying low-confidence images that require expert review. A future mobile application could guide the user to capture multiple photographs, verify crop type and transmit uncertain cases to a regional agronomist.

The project does not assume that a model trained only on PlantVillage is ready for autonomous diagnosis. Real deployment would require region-specific field images, crop-stage information, expert validation and careful separation between disease prediction and pesticide recommendations.

## **8\. Challenges**

Following are some possible challenges that could be encountered during the project:

1. ### Class Imbalance

   The PlantVillage dataset may contain a different number of images for each crop-disease class. Classes with fewer samples may receive lower recall and F1 scores. This challenge can be addressed using macro-F1 evaluation, class weights, focal loss or oversampling of minority classes within the training set.

2. ### Duplicate and Near-Duplicate Images

   The dataset may contain exact or visually similar images of the same leaf. If duplicate images appear in both the training and test sets, the model may produce artificially high performance. Duplicate and near-duplicate images will therefore be identified and grouped before splitting the dataset.

3. ### Controlled Image Backgrounds

   Many PlantVillage images are captured under controlled conditions with simple and uniform backgrounds. The model may learn background patterns instead of genuine disease symptoms. Background uniformity will be analysed, and field-oriented augmentation will be used to increase variation.

4. ### Image Size and Computational Cost

   Training a deep-learning model using 512 × 512 images can require significant GPU memory and longer training time. This challenge can be managed using smaller batch sizes, mixed-precision training or a lower image resolution if necessary.

5. ### Transfer Learning and Fine-Tuning

   Although transfer learning from a pretrained ResNet-50 model can reduce training time, fine-tuning the model for 38 crop-disease classes may require careful experimentation. Parameters such as learning rate, batch size, number of epochs and the number of unfrozen layers must be selected carefully.

6. ### Confidence and Overconfidence

   Deep-learning models may produce high confidence scores even when their predictions are incorrect. Confidence calibration will therefore be analysed using reliability diagrams and Expected Calibration Error. Temperature scaling may be applied if the model is found to be overconfident.

7. ### Field Generalisation

   A model trained on controlled laboratory images may not perform equally well on real field images containing shadows, complex backgrounds, uneven lighting and different camera angles. The model will be tested using field-like transformations such as blur, lighting variation, occlusion and perspective changes.

8. ### Unrealistic Data Augmentation

   Excessive or unrealistic augmentation may change important disease characteristics and reduce classification performance. Augmentation techniques will therefore be kept biologically realistic and visually inspected before model training.

9. ### Handling False Positives and False Negatives

   False-positive predictions may incorrectly classify a healthy leaf as diseased, while false-negative predictions may fail to identify an actual disease. Both types of errors can affect practical decision-making. Per-class precision, recall, F1 scores and confusion matrices will be used to analyse these errors.

Addressing these challenges will require systematic data analysis, careful experimentation and repeated validation to develop an accurate and reliable crop disease classification model.

# **9\. Week-wise Plan**

| Week | Activity |
| :---- | :---- |
| 1 | Prepare the project proposal, conduct the literature survey, finalise the methodology and perform initial exploratory data analysis. Create the shared GitHub repository and set up Google Colab Pro. |
| 2 | Complete image preprocessing, implement data augmentation techniques and generate exploratory data visualisations. |
| 3 | Develop and train the baseline CNN model using a representative subset of the PlantVillage dataset. |
| 4 | Compare the baseline and transfer-learning models, select the final model architecture and provide justification for the selected model. |
| 5 | Train and fine-tune the selected ResNet-50 model using the complete training dataset. |
| 6 | Evaluate the model and generate the macro-F1 score, data-efficiency curve, Grad-CAM visualisations and confidence-calibration results. |
| 7 | Complete model deployment, organise the source code, prepare the final project report and create the presentation slides. |
| 8 | Conduct the final project presentation and demonstrate the crop disease classification model. |

# 

# **10\. References**

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of the 34th International Conference on Machine Learning, 1321-1330.

Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. Proceedings of the IEEE International Conference on Computer Vision, 2980-2988.

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, Article 1419\.

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, 249-253.

