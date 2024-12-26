# Face Recognition System

## Overview
This is a Face Recognition System that utilizes Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machine (SVM) for classification. It supports adding and removing new faces, training a model, and real-time face recognition using a webcam.

This project is built with Python and utilizes libraries such as **scikit-learn**, **OpenCV**, and **PyQt5** for creating the graphical user interface (GUI). The system recognizes faces from a dataset and supports adding/removing new faces dynamically.

## Features
- **Train on a dataset**: Uses PCA for feature extraction and SVM for classification.
- **Add new faces**: You can add new faces to the model.
- **Remove faces**: Allows for the removal of specific faces from the model.
- **Real-time recognition**: Uses a webcam to perform live face recognition.
- **View added faces**: Displays images of all faces added to the system.
- **Accuracy Display**: Shows the model’s test accuracy and predicted vs actual labels for a random selection of faces.

## Installation

### Requirements
- Python 3.x
- **PyQt5**: For the GUI
- **OpenCV**: For image and video processing
- **scikit-learn**: For machine learning models and PCA
- **NumPy**: For numerical operations
- **Matplotlib**: For visualization

### Steps to Install
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/AhmadAbdullah22/Face_Recognition_System.git
   cd Face_Recognition_System
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the program:
   ```bash
   python Face_Recognition_System.py
   ```

## Usage

1. **Training the Model**:
   - The system is pre-configured to use the **Labeled Faces in the Wild (LFW)** dataset, which contains images of celebrities.
   - Upon running the program, the model is trained using PCA and SVM on this dataset.
   - If you want to add new faces, you can do so through the GUI.

2. **Adding a New Face**:
   - Click on **"Add New Face"** to upload an image and provide a label (e.g., a person's name or an ID).
   - The face is added to the system, and the model is retrained to include the new data.

3. **Removing a Face**:
   - Click on **"Remove Face"** to remove an image from the dataset. You can specify the index of the face to remove.

4. **Real-time Face Recognition**:
   - Click on **"Start Recognition"** to start real-time face recognition using your webcam.
   - The system will recognize faces from the webcam stream, display the predicted labels, and highlight the faces.

5. **Viewing Added Faces**:
   - Click on **"View New Faces"** to display the added faces from the **new_faces** directory.

6. **Displaying Model Accuracy and Predicted Faces**:
   - Click on **"Predicted Faces"** to see a few random faces from the test dataset and their predicted and true labels.

## File Structure

```
Face_Recognition_System/
│
├── Face_Recognition_System.py   # Main Python script with GUI and logic
├── requirements.txt             # Required Python libraries
├── models/                      # Directory to store the trained models
│   ├── pca_model.pkl
│   ├── svc_model.pkl
│   ├── scaler_model.pkl
│   └── feature_vectors.npy
├── new_faces/                   # Directory to store images of newly added faces
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
├── README.md                    # Project documentation
└── ...
```

## Example

### Adding a New Face
After running the script and opening the GUI, you will see the option to add a new face. Select an image file, and provide a label for the new face. The system will automatically add the face to the dataset and retrain the model.

### Real-time Recognition
Click **Start Recognition** to open the webcam feed. The system will detect faces in the webcam feed and label them in real-time.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make improvements, and create a pull request. Below are some areas where contributions are welcome:

- Improving face recognition accuracy.
- Optimizing the GUI design.
- Adding support for additional classifiers or feature extraction methods.
- Writing unit tests and improving code coverage.

## License

This project is licensed under the MIT License - see the (LICENSE) file for details.

## Contact

For any inquiries or issues related to this project, feel free to contact:

**Ahmad Abdullah**  
- Email: ahmadabdullahbrw@gmail.com  
- Phone: +923066357357
