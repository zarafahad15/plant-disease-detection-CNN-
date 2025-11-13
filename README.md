# plant-disease-detection-CNN-
Project Overview
Plant diseases affect crop yield and food security globally. This project uses Convolutional Neural Networks (CNNs) to automatically detect diseases in plant leaves. The system classifies images into healthy or diseased categories, making it a practical tool for precision agriculture and AI-based farming solutions.

⸻

Live Demo / Screenshots

Training Accuracy & Loss

Prediction on Sample Leaf Images

⸻

Features
	•	Multi-class classification of plant leaf diseases.
	•	Data augmentation to enhance model generalization.
	•	Custom CNN architecture with dropout and pooling layers.
	•	Early stopping & model checkpointing to prevent overfitting.
	•	Training and validation visualization.
	•	Test evaluation for performance measurement.

⸻

Technologies Used
	•	Python 3.x
	•	TensorFlow & Keras
	•	NumPy, Pandas
	•	Matplotlib & Seaborn
	•	OpenCV (for future real-time detection)

⸻

Dataset
	•	Source: PlantVillage Dataset
	•	Classes: Healthy + multiple disease categories (e.g., tomato, potato, corn).
	•	Directory Structure:

dataset/
│
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
└── test/


⸻

Installation
	1.	Clone the repository:

git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection

	2.	Install dependencies:

pip install -r requirements.txt

	3.	Place dataset in the dataset/ folder as per structure above.

⸻

Usage
	1.	Train the model:

jupyter notebook plant_disease_detection.ipynb

	2.	Evaluate on test data:

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

	3.	Use the saved model plant_disease_model.h5 for predictions.

⸻

Model Architecture

Layer	Output Shape	Activation
Conv2D (32)	(128,128,32)	ReLU
MaxPooling2D	(64,64,32)	-
Conv2D (64)	(64,64,64)	ReLU
MaxPooling2D	(32,32,64)	-
Conv2D (128)	(32,32,128)	ReLU
MaxPooling2D	(16,16,128)	-
Flatten	32768	-
Dense (256)	256	ReLU
Dropout	256	-
Output Dense	num_classes	Softmax


⸻

Results
	•	Validation Accuracy: ~95%
	•	Model generalizes well due to data augmentation and early stopping.
	•	Ready for real-time leaf disease detection in agriculture applications.

⸻

Future Improvements
	•	Use Transfer Learning (ResNet50 / EfficientNet) for better accuracy.
	•	Add real-time predictions via webcam or smartphone camera.
	•	Deploy as a web app (Streamlit/Flask) or mobile app for farmers.

⸻

Project Tags

#MachineLearning #DeepLearning #ComputerVision #CNN #PlantDiseaseDetection #Python #TensorFlow

⸻

License

This project is licensed under the MIT License – see the LICENSE file for details.

⸻

References
	•	PlantVillage Dataset on Kaggle
	•	TensorFlow Keras Documentation
	•	CS231n CNN Guide

⸻
