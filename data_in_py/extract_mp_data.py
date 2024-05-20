import mediapipe as mp
import cv2 as cv
import math
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
import keyboard
import time
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import os
from PIL import Image
from itertools import islice

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv.VideoCapture('abc2.mp4')
with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            cap.release()
            cv.destroyAllWindows()
        else:
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

            image.flags.writeable = False
            image = image
            cv.imshow('frame', cv.flip(image, 1))
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', cv.flip(image, 1))

            frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    cap.release()
    cv.destroyAllWindows()

def make_folders(actions, no_sequences, DATA_PATH):
    for action in actions: 
        for frame_num in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(frame_num)))
            except:
                pass

def extract_keypoints(results, width, height):
    keypoint_values = []
    hand_no = 0

    # as many results as hands
    # as many times through loop as hands
    if results.multi_hand_landmarks:
        for hand_no, handLandmarks in enumerate(results.multi_hand_landmarks):
            hand_no =+ 1
            for point in mp.solutions.hands.HandLandmark:
                # by default, landmarks are returned in their normalized format
                normalizedLandmark = handLandmarks.landmark[point]
                # function returns tuple of x and y coordinates 
                pixelCoordinatesLandmark = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, width, height)
                keypoint_values.append([normalizedLandmark.x, normalizedLandmark.y, handLandmarks.landmark[point].z])
                keypoint_array = np.array(keypoint_values)
                keypoint_array_flat = keypoint_array.flatten()
        
        if (hand_no == 1) and (len(keypoint_array_flat) < 126): 
            zero_array = np.zeros(63)
            if results.multi_handedness[0].classification[0].label == 'Right':
                keypoint_array_flat = np.append(keypoint_array_flat, zero_array)
            elif results.multi_handedness[0].classification[0].label == 'Left':
                keypoint_array_flat = np.append(zero_array, keypoint_array_flat)

        return (keypoint_array_flat)

DATA_PATH = os.path.join('MP_Data_test') 
actions = [ 'hola']
key = [1f]
key_action = dict(zip(key,actions))
no_sequences = 15
sequence_length = 20

make_folders(actions, no_sequences, DATA_PATH)

cap = cv.VideoCapture('abc2.mp4')
with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands =2 ,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('error')
            cap.release()
            cv.destroyAllWindows()
        else:
            if cv.waitKey(5) & 0xFF == ord('q'):
                break
            image.flags.writeable = False
            image = image
            cv.imshow('frame', cv.flip(image, 1))
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', cv.flip(image, 1))
            frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            number = 1
            count = 2
            if keyboard.is_pressed(str(number)):
                print('keyboard pressed: {}'.format(number))
                for frame_num in range(sequence_length):
                    #cv.imwrite('Frame'+str(frame_num)+'.jpg', image)
                    keypoints = extract_keypoints(results, frameWidth, frameHeight)
                    action = key_action[number]
                    npy_path = os.path.join(DATA_PATH, action, str(count), str(frame_num))
                    print(os.path.join(DATA_PATH, action, str(count), str(frame_num)))
                    np.save(npy_path, keypoints)
                    success, image = cap.read()
                    frame_num = frame_num +1

    cap.release()
    cv.destroyAllWindows()

results.multi_hand_landmarks

array_results = np.load('MP_Data_test/c/0/9.npy', allow_pickle=True)
print(np.array(array_results).shape)
