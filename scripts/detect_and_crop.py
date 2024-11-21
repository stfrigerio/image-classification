import cv2
import os

def detect_and_crop_faces(input_dir, output_dir, face_cascade_path='haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    if not face_cascade.empty():
        print("Haar Cascade loaded successfully.")
    else:
        raise IOError("Haar Cascade XML file not found.")

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read {image_path}. Skipping.")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                print(f"No face detected in {image_path}. Skipping.")
                continue
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                save_path = os.path.join(output_label_dir, image_name)
                cv2.imwrite(save_path, face)
                break  # Assume one face per image

if __name__ == "__main__":
    detect_and_crop_faces(input_dir='../dataset', output_dir='../cropped_dataset')