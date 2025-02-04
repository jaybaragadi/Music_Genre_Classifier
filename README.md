# Music_Genre_Classifier

🚀 Overview
This project implements a high-performing Convolutional Neural Network (CNN) model for accurate music genre classification using the Librosa and TensorFlow Python libraries. It demonstrates expertise in audio signal processing, deep learning model development, and real-time application deployment.

The model processes audio signals, extracts relevant features, and classifies them into predefined genres, making it useful for music recommendation systems, content filtering, and automated music categorization.

📌 Features
✅ Audio Feature Extraction using Librosa
✅ Convolutional Neural Network (CNN) for classification
✅ Genre Prediction with High Accuracy
✅ Real-time & Batch Processing Support
✅ Scalable & Optimized Deep Learning Model

📂 Project Structure
graphql
Copy
Edit
📁 Music-Genre-Classification
│── 📂 dataset               # Processed audio dataset
│── 📂 models                # Saved trained models
│── 📂 notebooks             # Jupyter Notebooks for EDA & training
│── 📂 src                   # Source code
│   │── data_preprocessing.py  # Audio processing and feature extraction
│   │── train_model.py         # CNN model training script
│   │── predict.py             # Genre classification script
│── 📜 requirements.txt       # Required dependencies
│── 📜 README.md              # Project documentation
│── 📜 config.yaml            # Configurations for the model
│── 📜 app.py                 # Web app for real-time genre classification

📊 Data Processing
The project uses Librosa to extract Mel Spectrograms and other relevant audio features, which serve as input for the CNN model. The dataset is preprocessed and split into training and testing sets.

🔥 Model Architecture
The CNN model consists of multiple convolutional layers, batch normalization, dropout layers, and dense layers to accurately classify music genres. The architecture is optimized for generalization and robust performance.


🛠 Installation & Setup
1️⃣ Clone the repository
bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
2️⃣ Install dependencies
bash
pip install -r requirements.txt
3️⃣ Run the training script
bash
python src/train_model.py
4️⃣ Predict genre for an audio file
bash
python src/predict.py --file example.wav
🚀 Real-Time Genre Classification App
A Flask-based web application is provided to allow users to upload an audio file and get real-time genre classification.

Start the Web App
bash
python app.py
Access the web interface at: http://localhost:5000

📈 Results
The trained model achieves high accuracy on benchmark datasets and effectively classifies music genres.

Training Accuracy: ~95%
Validation Accuracy: ~92%

💬 Contact
For any queries or suggestions, reach out via:
📧 b22095@iitmandi.ac.in
🌐 GitHub: jaybaragadi
