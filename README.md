\# DeepTrace: Deepfake Detection System



DeepTrace is a deep learning-based system designed to detect deepfake images using spatial and frequency-based features.



\## Overview



Deepfake technology can create highly realistic fake images and videos, making it difficult to distinguish real from manipulated media. DeepTrace aims to detect such manipulations using deep learning techniques.



The system trains a neural network to classify images as \*\*Real\*\* or \*\*Fake\*\* and evaluates the model using standard machine learning metrics.



---



\## Features



\- Deepfake image detection

\- Hybrid spatial-frequency model

\- Model training and evaluation pipeline

\- Confusion matrix visualization

\- ROC curve analysis

\- Performance metrics (Accuracy, Precision, Recall, F1 Score)



---



\## Project Structure



DeepTrace/

│

├── train.py

├── test.py

├── config.py

├── requirements.txt

├── README.md

│

├── models/

├── utils/

├── datasets/

│

├── checkpoints/

│

└── results/

├── confusion\_matrix.png

└── roc\_curve.png



---



\## Installation



Clone the repository:





git clone https://github.com/mohammedsalahuddin1313/DeepTrace.git





Navigate to the project folder:





cd DeepTrace





Install dependencies:





pip install -r requirements.txt





---



\## Training the Model



Run the following command:





python train.py





This will train the deepfake detection model and save the trained weights.



---



\## Testing the Model



Run:





python test.py





This will evaluate the model and generate performance metrics.



---



\## Results



Model Performance:



\- Accuracy: \*\*67.75%\*\*

\- Precision: \*\*66.98%\*\*

\- Recall: \*\*52.59%\*\*

\- F1 Score: \*\*58.92%\*\*

\- AUC Score: \*\*0.73\*\*



\### Confusion Matrix



Stored in:





results/confusion\_matrix.png





\### ROC Curve



Stored in:





results/roc\_curve.png





---



\## Notes



Due to GitHub file size limitations, trained model weights (`.pth` files) are not included in this repository.  

Run `train.py` to generate the model locally.



---



\## Authors



DeepTrace – Deepfake Detection System



Contributors:

\- Mohammed Salahuddin

