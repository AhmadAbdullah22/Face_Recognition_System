import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import cv2
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QInputDialog, QStatusBar, QDialog, QDialogButtonBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import sys

def load_lfw_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    return X, y, target_names

def preprocess_data(X, y):
    n_samples, h, w = X.shape
    X = X.reshape(n_samples, -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def perform_pca(X_train, n_components=150):
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    return pca, X_train_pca

def train_svm(X_train_pca, y_train):
    svc = SVC(kernel="rbf", class_weight="balanced", C=1)
    svc.fit(X_train_pca, y_train)
    return svc

def evaluate_model(svc, pca, X_test, y_test, target_names, X_train=None, y_train=None):
    train_accuracy = 0.0
    test_accuracy = 0.0
    if X_train is not None and y_train is not None:
        X_train_pca = pca.transform(X_train)
        y_train_pred = svc.predict(X_train_pca)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    X_test_pca = pca.transform(X_test)
    y_test_pred = svc.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return train_accuracy, test_accuracy

def plot_eigenfaces(pca, h=50, w=37, n_row=5, n_col=10):
    eigenfaces = pca.components_.reshape((150, h, w))
    n_eigenfaces = min(10, len(eigenfaces))
    titles = [f"Eigenface {i+1}" for i in range(n_eigenfaces)]
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90)
    for i in range(n_eigenfaces):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenfaces[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=10, fontweight='bold', color='black', backgroundcolor='white', pad=2)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def add_new_face(image_path, new_label, pca, X_train, y_train, scaler, images_dir):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (50, 37))
    img_flattened = img_resized.flatten() / 255.0
    img_scaled = scaler.transform(img_flattened.reshape(1, -1))

    X_train = np.vstack([X_train, img_scaled])
    y_train = np.append(y_train, new_label)

    os.makedirs(images_dir, exist_ok=True)
    img_name = f"face_{len(y_train)}.jpg"
    cv2.imwrite(os.path.join(images_dir, img_name), img_resized)

    pca = PCA(n_components=150, whiten=True)
    X_train_pca = pca.fit_transform(X_train)

    return pca, X_train, y_train

def remove_face(image_index, X_train, y_train, pca, images_dir):
    if image_index < 0 or image_index >= len(X_train):
        raise ValueError("Invalid index")

    X_train = np.delete(X_train, image_index, axis=0)
    y_train = np.delete(y_train, image_index)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('jpg', 'png', 'bmp'))]
    if image_index < len(image_files):
        img_to_remove = image_files[image_index]
        os.remove(os.path.join(images_dir, img_to_remove))
        print(f"Removed image {img_to_remove} from the directory.")

    pca = PCA(n_components=150, whiten=True)
    X_train_pca = pca.fit_transform(X_train)

    return pca, X_train, y_train

def save_model(pca, svc, scaler, feature_vectors, labels):
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca, 'models/pca_model.pkl')
    joblib.dump(svc, 'models/svc_model.pkl')
    joblib.dump(scaler, 'models/scaler_model.pkl')
    joblib.dump(feature_vectors, 'models/feature_vectors.npy')
    joblib.dump(labels, 'models/labels.npy')

def load_model():
    pca = joblib.load('models/pca_model.pkl')
    svc = joblib.load('models/svc_model.pkl')
    scaler = joblib.load('models/scaler_model.pkl')
    feature_vectors = joblib.load('models/feature_vectors.npy')
    labels = joblib.load('models/labels.npy')
    return pca, svc, scaler, feature_vectors, labels

def real_time_recognition(pca, svc, scaler, target_names):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (50, 37))
            face_flattened = face_resized.flatten() / 255.0
            face_scaled = scaler.transform(face_flattened.reshape(1, -1))

            face_pca = pca.transform(face_scaled)
            prediction = svc.predict(face_pca)

            label = target_names[prediction[0]]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Real-time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class FaceRecognitionUI(QWidget):
    def __init__(self, pca, svc, scaler, X_train, y_train, X_test, y_test, images_dir, target_names):
        super().__init__()
        self.pca = pca
        self.svc = svc
        self.scaler = scaler
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.images_dir = images_dir
        self.target_names = target_names
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Face Recognition')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: grey;")
        self.layout = QVBoxLayout()
        self.header_label = QLabel("Face Recognition System", self)
        self.header_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.header_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header_label)

        self.button_layout = QVBoxLayout()

        self.add_face_button = QPushButton("Add New Face", self)
        self.add_face_button.setFont(QFont("Arial", 12))
        self.add_face_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px;")
        self.add_face_button.clicked.connect(self.add_new_face)
        self.button_layout.addWidget(self.add_face_button)

        self.remove_face_button = QPushButton("Remove Face", self)
        self.remove_face_button.setFont(QFont("Arial", 12))
        self.remove_face_button.setStyleSheet("background-color: #f44336; color: white; border-radius: 5px; padding: 10px;")
        self.remove_face_button.clicked.connect(self.remove_face)
        self.button_layout.addWidget(self.remove_face_button)

        self.recognize_button = QPushButton("Start Recognition", self)
        self.recognize_button.setFont(QFont("Arial", 12))
        self.recognize_button.setStyleSheet("background-color: #008CBA; color: white; border-radius: 5px; padding: 10px;")
        self.recognize_button.clicked.connect(self.start_recognition)
        self.button_layout.addWidget(self.recognize_button)

        self.info_button = QPushButton("Info", self)
        self.info_button.setFont(QFont("Arial", 12))
        self.info_button.setStyleSheet("background-color: #FFD700; color: white; border-radius: 5px; padding: 10px;")
        self.info_button.clicked.connect(self.show_info)
        self.button_layout.addWidget(self.info_button)

        self.view_pictures_button = QPushButton("View New Faces", self)
        self.view_pictures_button.setFont(QFont("Arial", 12))
        self.view_pictures_button.setStyleSheet("background-color: #32CD32; color: white; border-radius: 5px; padding: 10px;")
        self.view_pictures_button.clicked.connect(self.view_added_faces)
        self.button_layout.addWidget(self.view_pictures_button)

        self.contact_button = QPushButton("Contact Info", self)
        self.contact_button.setFont(QFont("Arial", 12))
        self.contact_button.setStyleSheet("background-color: #FF8C00; color: white; border-radius: 5px; padding: 10px;")
        self.contact_button.clicked.connect(self.show_contact_info)
        self.button_layout.addWidget(self.contact_button)

        self.show_accuracy_and_faces_button = QPushButton("Predicted Faces", self)
        self.show_accuracy_and_faces_button.setFont(QFont("Arial", 12))
        self.show_accuracy_and_faces_button.setStyleSheet("background-color: #FF6347; color: white; border-radius: 5px; padding: 10px;")
        self.show_accuracy_and_faces_button.clicked.connect(self.show_accuracy_and_random_faces)
        self.button_layout.addWidget(self.show_accuracy_and_faces_button)

        self.layout.addLayout(self.button_layout)

        self.status_label = QLabel("Welcome to the Face Recognition System!", self)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        self.status_bar = QStatusBar(self)
        self.layout.addWidget(self.status_bar)

        self.setLayout(self.layout)

    def update_status(self, message):
        self.status_label.setText(message)
        self.status_bar.showMessage(message, 5000)

    def add_new_face(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            label, ok = QInputDialog.getText(self, "Enter New Label", "Label for this face:")
            if ok and label.isdigit():
                self.pca, self.X_train, self.y_train = add_new_face(file_path, int(label), self.pca, self.X_train, self.y_train, self.scaler, self.images_dir)
                self.update_status(f"Face added with label {label}.")
            else:
                self.update_status("Invalid label entered!")
        else:
            self.update_status("No file selected!")

    def remove_face(self):
        index, ok = QInputDialog.getInt(self, "Enter Index", "Index of face to remove:")
        if ok and 0 <= index < len(self.X_train):
            self.pca, self.X_train, self.y_train = remove_face(index, self.X_train, self.y_train, self.pca, self.images_dir)
            self.update_status(f"Face at index {index} removed.")
        else:
            self.update_status("Invalid index!")

    def start_recognition(self):
        self.update_status("Starting real-time recognition...")
        real_time_recognition(self.pca, self.svc, self.scaler, self.target_names)
        self.update_status("Real-time recognition stopped.")

    def show_info(self):
        info_dialog = QDialog(self)
        info_dialog.setWindowTitle("System Information")
        info_dialog.setFixedSize(400, 200)

        layout = QVBoxLayout()
        info_label = QLabel(f"Total Faces: {len(self.X_train)}\nTotal Labels: {len(set(self.y_train))}\n"
                            "This system recognizes faces using PCA and SVM.", self)
        layout.addWidget(info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(info_dialog.accept)
        layout.addWidget(buttons)

        info_dialog.setLayout(layout)
        info_dialog.exec_()

    def show_contact_info(self):
        contact_dialog = QDialog(self)
        contact_dialog.setWindowTitle("Contact Information")
        contact_dialog.setFixedSize(400, 200)

        layout = QVBoxLayout()
        contact_label = QLabel("For support or inquiries, please contact:\n\n"
                               "Name: Ahmad Abdullah\n"
                               "Email: ahmadabdullahbrw@gmail.com\n"
                               "Phone: +923066357357", self)
        layout.addWidget(contact_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(contact_dialog.accept)
        layout.addWidget(buttons)

        contact_dialog.setLayout(layout)
        contact_dialog.exec_()

    def view_added_faces(self):
        if not os.path.exists(self.images_dir):
            self.update_status("No images directory found!")
            return

        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('jpg', 'png', 'bmp'))]

        if not image_files:
            self.update_status("No images found!")
            return

        plt.figure(figsize=(10, 10))

        for i, image_file in enumerate(image_files[:20]):
            img = cv2.imread(os.path.join(self.images_dir, image_file), cv2.IMREAD_GRAYSCALE)
            plt.subplot(5, 4, i + 1)
            plt.imshow(img, cmap=plt.cm.gray)
            plt.title(f"Image {i+1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def show_accuracy_and_random_faces(self):
        X_test_pca = self.pca.transform(self.X_test)
        y_pred = self.svc.predict(X_test_pca)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        self.update_status(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        num_faces_to_show = 20
        random_indices = random.sample(range(len(self.X_test)), num_faces_to_show)

        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(random_indices):
            plt.subplot(5, 4, i + 1)
            img = self.X_test[idx].reshape(50, 37)
            plt.imshow(img, cmap=plt.cm.gray)

            label = self.target_names[y_pred[idx]]
            true_label = self.target_names[self.y_test[idx]]
            plt.title(f"Pred: {label}\nTrue: {true_label}", fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    X, y, target_names = load_lfw_data()
    print(f"Dataset loaded with {len(X)} images of {len(target_names)} people.")
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    pca, X_train_pca = perform_pca(X_train)
    
    svc = train_svm(X_train_pca, y_train)
    print("SVM model trained.")
    
    save_model(pca, svc, scaler, X_train, y_train)
    print("Model saved.")

    pca, svc, scaler, feature_vectors, labels = load_model()
    print("Model loaded.")

    app = QApplication(sys.argv)
    window = FaceRecognitionUI(pca, svc, scaler, X_train, y_train, X_test, y_test, images_dir='./new_faces', target_names=target_names)
    window.show()
    
    sys.exit(app.exec_())
