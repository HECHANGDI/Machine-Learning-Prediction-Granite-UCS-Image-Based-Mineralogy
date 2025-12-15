# Machine-Learning-for-Predicting-the-UCS-of-Granite-from-Image-Derived-Mineralogical-Features
This study uses machine learning (ML) models to predict the numerically simulated uniaxial compressive strength (UCS) of granite directly from digital images. Specifically, the images were processed using an in-house Digital Image Processing (DIP) tool to derive mineralogical features (used as input features for the ML models), including mineral content, grain size, and spatial distribution. Mineral content and distribution were quantified using $m$-harmonic Fourier series equations, whereas mineral grain size was determined using the 4-connectivity method. The target UCS values were derived from the 2D physically informed Subspring Network Breakable Voronoi (SNBV) microstructural models, replicating the mineralogical features observed in the granite images. Extreme Gradient Boosting (XGBoost) models with different input combinations and hyperparameter optimization methods were trained and evaluated on 126 granite images using a single train/test split and repeated 5-fold cross-validation.

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/ba421ffe-bc31-4cf9-8b42-5a75e6a4555c" />


# Image dataset
Find the images in the folder of: Granite_Images_G1-G126 <br>

<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/da2c3bc3-7dfa-4d57-b6f5-46f5ec677211" />

# DIP Web Tool
Open the DIP Tool_V1.0.html with browser. <br>
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/7a71a78f-c634-4cef-b97d-50b00cfaade7" />


# Training the Optuna-XGBoost model using the features (without SHAP-based feature selection)




# Training the Optuna-XGBoost model using the features (with SHAP-based feature selection for Cases 4 -- 7)





# Roubstness verification (using Case 7 as an example)



# Effect of the hyperparameter optimization techniques


# Please cite the following papers while using the follwing datasets:
1. He, C., Mishra, B., & Potyondy, D. O. (2026). Impact of mineralogical features on the mechanical behaviors of granite: A study using physically informed 3D microstructural model. International Journal of Rock Mechanics and Mining Sciences, 197, 106355.
@article{he2026impact,<br>
  title={Impact of mineralogical features on the mechanical behaviors of granite: A study using physically informed 3D microstructural model}, <br>
  author={He, Changdi and Mishra, Brijes and Potyondy, David Oskar},<br>
  journal={International Journal of Rock Mechanics and Mining Sciences},<br>
  volume={197},<br>
  pages={106355},<br>
  year={2026},<br>
  publisher={Elsevier}<br>
}<br>
=====================================================
2. 
