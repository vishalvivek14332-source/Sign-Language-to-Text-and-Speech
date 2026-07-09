# Sign Language to Text and Speech 🤟➡️📝🔊

A real-time system that recognizes American Sign Language (ASL) hand gestures using computer vision and deep learning, then converts them into text and speech — helping bridge communication between the hearing/speech impaired and others.

## ✨ Features

- **Real-time hand gesture recognition** via webcam using computer vision
- **CNN-based classification** of the 26 ASL alphabet gestures, grouped into 8 similar-gesture clusters for higher accuracy
- **Text formation** from recognized letters into words and sentences
- **Text-to-speech conversion** so recognized text can be spoken aloud
- **Custom data collection scripts** to build your own gesture dataset
- **Pre-trained models included** for immediate use
- **GUI and non-GUI prediction modes**

## 🧠 How It Works

1. **Data Collection** — Hand images are captured via webcam and preprocessed (e.g. skeletonized/binarized) to build a training dataset for each letter/gesture.
2. **Training** — A Convolutional Neural Network (CNN) is trained on the collected dataset to classify hand gestures. The 26 alphabets are split into 8 groups of visually similar gestures to improve accuracy, with additional geometric/landmark-based logic used to distinguish between letters within the same group.
3. **Prediction** — The trained model performs real-time inference on webcam frames, predicting the most likely gesture and assembling recognized letters into words and sentences.
4. **Speech Output** — The final recognized text is converted into speech, making the output accessible audibly as well as visually.

## 📁 Project Structure

```
Sign-Language-to-Text-and-Speech/
├── AtoZ_3.1/B                       # Gesture dataset (organized by letter)
├── best_cnn8grps_rad1_model.h5      # Best-performing trained CNN model
├── cnn8grps_rad1_model.h5           # Trained CNN model (8-group classification)
├── data_collection_binary.py        # Script to collect binary/thresholded hand images
├── data_collection_final.py         # Script to collect the final training dataset
├── evaluate_model.py                # Evaluates trained model performance/accuracy
├── final_pred.py                    # Main application — GUI + real-time gesture recognition
├── prediction_wo_gui.py             # Real-time prediction without the GUI
├── train_model.py                   # Trains the CNN model on the collected dataset
└── README.md
```

## 🛠️ Tech Stack

- **Python**
- **OpenCV** — image capture and processing
- **MediaPipe / cvzone** — hand detection and landmark tracking
- **TensorFlow / Keras** — CNN model training and inference
- **NumPy**
- **pyttsx3 / gTTS** — text-to-speech conversion
- **Tkinter** — GUI (in `final_pred.py`)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A working webcam

### Installation

```bash
git clone https://github.com/vishalvivek14332-source/Sign-Language-to-Text-and-Speech.git
cd Sign-Language-to-Text-and-Speech
conda activate signlang
pip install opencv-python mediapipe cvzone tensorflow keras numpy pyttsx3
```

> 💡 If a `requirements.txt` isn't present in the repo yet, consider adding one so dependencies can be installed with a single `pip install -r requirements.txt`.

### Usage

**Run the full application (GUI + speech output):**
```bash
python final_pred.py
```

**Run prediction without the GUI:**
```bash
python prediction_wo_gui.py
```

**Collect your own gesture data:**
```bash
python data_collection_final.py
# or
python data_collection_binary.py
```

**Train the model on your dataset:**
```bash
python train_model.py
```

**Evaluate model performance:**
```bash
python evaluate_model.py
```

## 🎯 Model Details

- Gestures are grouped into **8 clusters** of visually similar ASL letters to simplify the classification problem for the CNN.
- Within each cluster, additional logic (e.g. hand landmark geometry) is used to distinguish between individual letters.
- Two model checkpoints are provided: `cnn8grps_rad1_model.h5` and `best_cnn8grps_rad1_model.h5` (the higher-accuracy checkpoint).

## 📄 License

No license has been specified for this repository yet. Consider adding one (e.g. MIT) so others know how they can use your code.

## 🙏 Acknowledgements

Built using open-source computer vision and deep learning tools to make sign language communication more accessible.
