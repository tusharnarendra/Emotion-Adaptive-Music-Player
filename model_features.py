import dlib
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm 
import re

#Function used to normalize the feature landmarks to remove irrelevant variation and improve generalization when training the model

def normalize_landmarks(landmarks):

    #Create an array with all the landmark points
    point_list = []
    for i in range(68):
        point_list.append([landmarks.part(i).x, landmarks.part(i).y])
    points = np.array(point_list, dtype=np.float32)

    #Center the tip of the nose at (0,0), so that the position of the face on the screen in the original frame has minimal impact
    nose_tip = points[30]
    points -= nose_tip

    #Compute the angle of a vector representing the eyes to the horizontal axis, to rotate the face to remove tilting
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)

    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    angle = np.arctan2(delta_y, delta_x)

    #Use a rotation matrix to rotate all points
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a, cos_a]])

    points = points.dot(R)

    #Re-center in case rotation caused center to move
    nose_tip_after_rotation = points[30]
    points -= nose_tip_after_rotation

    # Scale so distance between eyes is 1
    left_eye_rot = np.mean(points[36:42], axis=0)
    right_eye_rot = np.mean(points[42:48], axis=0)
    eye_dist = np.linalg.norm(right_eye_rot - left_eye_rot)
    points /= eye_dist

    return points

# Calculate the eybrow raise as an additional feature to the raw normalized features
def eyebrow_raise(norm_points):
    left_eyebrow_y = np.mean(norm_points[17:22, 1])
    left_eye_y = np.mean(norm_points[36:40, 1])
    right_eyebrow_y = np.mean(norm_points[22:27, 1])
    right_eye_y = np.mean(norm_points[42:46, 1])
    return (left_eyebrow_y - left_eye_y, right_eyebrow_y - right_eye_y)

# Similarly calculate some smile features
def smile_features(norm_points):
    mouth_left = norm_points[48]
    mouth_right = norm_points[54]
    mouth_top = norm_points[51]
    mouth_bottom = norm_points[57]

    mouth_width = np.linalg.norm(mouth_right - mouth_left)
    mouth_openness = mouth_bottom[1] - mouth_top[1]

    delta_y = mouth_right[1] - mouth_left[1]
    delta_x = mouth_right[0] - mouth_left[0]
    smile_angle = np.arctan2(delta_y, delta_x)

    return mouth_width, mouth_openness, smile_angle

#Load the models

cnn_face_detector = dlib.cnn_face_detection_model_v1("prebuilt_models/mmod_human_face_detector.dat")
shape_predictor = dlib.shape_predictor("prebuilt_models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("prebuilt_models/lib_face_recognition_resnet_model_v1.dat")

directory_path = 'dataset'

# Create and empty array to store the features for each image
X = []


def sort_key(filename):
    # Extract the numeric parts before and after underscore
    match = re.match(r"(\d+)_(\d+)", filename)
    if match:
        folder_num = int(match.group(1))
        image_num = int(match.group(2))
        return (folder_num, image_num)
    else:
        return (float('inf'), float('inf'))

file_list = sorted(os.listdir(directory_path), key=sort_key)[:5000]
Failed_detections = []


# Iterate through every image in the dataset with progress bar
for entry in tqdm(file_list, desc="Processing images"):
    img_path = os.path.join(directory_path, entry)
    if os.path.isfile(img_path):
        print(f"File: {img_path}")
    
    #Run the face detection on the image
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = cnn_face_detector(rgb_img, 1)
    #Check if a face is detected
    if len(detections) == 0:
        print("No faces detected!")
        Failed_detections.append(entry)
        continue
    print(f"Detected {len(detections)} face(s)")

    # Process only the first detected face in the image
    detection = detections[0]
    #Create a bounding box around the face and generate the 68 landmarks from this
    rect = detection.rect
    shape = shape_predictor(rgb_img, rect)
    #Run the normalization function
    norm_points = normalize_landmarks(shape)
    #Convert to 1D array
    flat_features = norm_points.flatten()

    #Add a couple additional features
    brow_raise = eyebrow_raise(norm_points)
    mouth_width, mouth_openness, smile_angle = smile_features(norm_points)

    additional_features = np.array([mouth_width, mouth_openness, smile_angle, *brow_raise])
    combined_features = np.concatenate([flat_features, additional_features])

    face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, shape)
    face_embedding = np.array(face_descriptor)  
    combined_features = np.concatenate([flat_features, additional_features, face_embedding])

    X.append(combined_features)

X = np.array(X)

# Save failed detections to a text file for later processing
with open("failed_detections.txt", "w") as f:
    for filename in Failed_detections:
        f.write(filename + "\n")
np.save('normalized_features.npy', X)
