import cv2
import os
import sys
import csv
import numpy as np
from collections import deque
import pyautogui
import argparse
import configparser
from ast import literal_eval
import errno
import tensorflow as tf
import time
from keras.optimizers.schedules import ExponentialDecay

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from data.data_loader import FrameQueue

def load_model(model_path):
    print("MODEL PATH:", model_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ExponentialDecay": ExponentialDecay}
    )
    print("Loaded CNN model from disk")
    return model

def main(config):
    nb_frames = config.getint('general', 'nb_frames')
    target_size = literal_eval(config.get('general', 'target_size'))
    root_data = config.get('path', 'root_data')
    csv_labels = config.get('path', 'csv_labels')
    gesture_keyboard_mapping = config.get('path', 'gesture_keyboard_mapping')
    model_path = config.get('path', 'model_path')

    labels_path = os.path.join(root_data, csv_labels)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    with open(labels_path) as f:
        f_csv = csv.reader(f)
        labels_list = tuple([row for row in f_csv]) 

    mapping = configparser.ConfigParser()
    action = {}
    if os.path.isfile(gesture_keyboard_mapping):
        mapping.read(gesture_keyboard_mapping)
        for m in mapping['MAPPING']:
            val = mapping['MAPPING'][m].split(',')
            action[m] = {'fn': val[0], 'keys': val[1:]}
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), gesture_keyboard_mapping)

    act = deque(['No gesture', "No gesture"], maxlen=3)

    net = load_model(model_path)

    frame_queue = FrameQueue(nb_frames, target_size)

    while cap.isOpened():
        on, frame = cap.read()
        if on:
            b, g, r = cv2.split(frame)
            frame_calibrated = cv2.merge([r, g, b])
            batch_x = frame_queue.img_in_queue(frame_calibrated)
            res = net.predict(batch_x)

        predicted_class = labels_list[np.argmax(res)]
        gesture = predicted_class[0]
        confidence = np.amax(res)

        if confidence >= 0.4:
            print(f'Gesture = {predicted_class}; Accuracy = {confidence * 100:.2f}%')
            
            gesture = gesture.lower()
            act.append(gesture)

            if act[0] != act[1] and len(set(list(act)[1:])) == 1:
                if gesture in action.keys():
                    t = action[gesture]['fn']
                    k = action[gesture]['keys']

                    print('[DEBUG]', gesture, '-- ', t, str(k))

                    if t == 'typewrite':
                        pyautogui.typewrite(k)
                    elif t == 'press':
                        pyautogui.press(k)
                    elif t == 'hotkey':
                        for key in k:
                            pyautogui.keyDown(key)
                        for key in k[::-1]:
                            pyautogui.keyUp(key)
        else:
            print(f'[INFO] Gesto ignorado. Confiança muito baixa: {confidence * 100:.2f}%')

            cv2.imshow('camera0', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
