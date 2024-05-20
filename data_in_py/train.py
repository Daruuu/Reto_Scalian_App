# import necessary libraries
import numpy as np 
import mediapipe as mp 
import os
import cv2
import keyboard
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

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
    if results.multi_hand_landmarks:
        for hand_no, handLandmarks in enumerate(results.multi_hand_landmarks):
            hand_no =+ 1
            for point in mp.solutions.hands.HandLandmark:
                normalizedLandmark = handLandmarks.landmark[point]
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

def import_solutions():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return(mp_hands, mp_drawing, mp_drawing_styles)


def frame_colection(key_action, key_to_press, count, input=0):
    mp_hands, mp_drawing, mp_drawing_styles = import_solutions()
    cap = cv2.VideoCapture(input)
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
                cv2.destroyAllWindows()
            else:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                image.flags.writeable = False
                image = image
                cv2.imshow('frame', cv2.flip(image, 1))
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

                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                sequence_length = 20
                DATA_PATH = os.path.join('MP_Data') 

                if keyboard.is_pressed(str(key_to_press)):
                    print('keyboard pressed: {}'.format(key_to_press))
                    for frame_num in range(sequence_length):
                        results = hands.process(image)
                        keypoints = extract_keypoints(results, frameWidth, frameHeight)
                        action = key_action[key_to_press]
                        npy_path = os.path.join(DATA_PATH, action, str(count), str(frame_num))
                        print(os.path.join(DATA_PATH, action, str(count), str(frame_num)))
                        cv2.imwrite(DATA_PATH+str(frame_num)+'.jpg', image)
                        np.save(npy_path, keypoints)
                        success, image = cap.read()
                        frame_num = frame_num +1
                        image.flags.writeable = False
                        image = image
                        cv2.imshow('frame', cv2.flip(image, 1))
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
                        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        cap.release()
        cv2.destroyAllWindows()


def make_variables(DATA_PATH, no_sequences, labels_dict, actions, sequence_length=10):
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                try:
                    window.append(list(res))
                except:
                    continue
            sequences.append(window)
            labels.append(labels_dict[action])
    return(sequences, labels)


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
    return output_frame


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return(model)


def show_cm(X, y, best_model):
    y_pred = best_model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    class_names = ['hola', 'a', 'b', 'c', 'i', 'n', 'bien']  # Reemplaza con los nombres reales de tus clases

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


DATA_PATH = os.path.join('MP_Data') 
actions = ['hola', 'a', 'b', 'c', 'i', 'n', 'bien']
no_sequences = 16
labels_dict = {label:num for num, label in enumerate(actions)}
sequences, labels = make_variables(DATA_PATH, no_sequences, labels_dict, actions)

np.asarray(sequences, dtype=object).shape

# for seq, label in zip(sequences, labels):
#     print(len(seq))
#     for s in seq:
#         sequences_clean.append(s)
#         labels_clean.append(label)
# X = X.reshape((X.shape[0], 1, 126))

# save as x and y variables 
sequences_clean, labels_clean = [], []

for seq, label in zip(sequences, labels):
    if len(seq) > 7:
        while len(seq) < 10:
            seq.append([0]*126)

        sequences_clean.append(seq[:7])
        labels_clean.append(label)

X = np.array(sequences_clean)
y = to_categorical(labels_clean).astype(int)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

## allows to monitor accuracy while its training
# check on terminal- into Logs folder:  tensorboard --logdir=. 
# copy link 
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

learning_rate = 1e-4 
batch_size = 32
epochs = 400

steps_per_epoch = len(X) // batch_size
checkpoint_callback = ModelCheckpoint('my_model_manos16.keras', monitor='val_loss', save_best_only=True, mode='min')

optimizer = Adam(learning_rate=learning_rate)

model = Sequential()
# Long short-term memory 
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(7,126))) # returning sequences for next layer 
model.add(LSTM(128, return_sequences=True, activation='relu')) # returning sequences for next layer 
model.add(LSTM(64, return_sequences=False, activation='relu'))
# Dense layer: each neuron receives input from all the neurons of previous layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# loss categorical_crossentropy because it is a multiclass classification
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(X, y, validation_data=(X_val, y_val), batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[tb_callback, checkpoint_callback])

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='train_categorical_accuracy')
plt.plot(val_acc, label='val_categorical_accuracy')
plt.title('Categorical Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.legend()
plt.show()

best_model = tf.keras.models.load_model('my_model_manos16.keras')
show_cm(X_val, y_val, best_model)

show_cm(X, y, best_model)
