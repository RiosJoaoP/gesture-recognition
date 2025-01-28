import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pandas as pd

class DataFrameVideoSequence(Sequence):
    def __init__(self, dataframe, directory, x_col="filenames", y_col="class",
                 target_size=(256, 256), nb_frames=16, skip=1, batch_size=32,
                 shuffle=True, class_mode="categorical", image_data_generator=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.target_size = target_size
        self.nb_frames = nb_frames
        self.skip = skip
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_mode = class_mode
        self.image_data_generator = image_data_generator

        # Processar classes
        self.classes = sorted(self.dataframe[y_col].unique())
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}
        self.dataframe[y_col] = self.dataframe[y_col].map(self.class_indices)

        # Armazenar IDs e labels
        self.ids = self.dataframe[x_col].tolist()
        self.labels = self.dataframe[y_col].tolist()

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # Gerar índices para este lote
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.ids[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        # Carregar os dados
        batch_x, batch_y = self._generate_batch(batch_ids, batch_labels)
        return batch_x, batch_y

    def _generate_batch(self, batch_ids, batch_labels):
        batch_x = []
        batch_y = []

        for video_id, label_int in zip(batch_ids, batch_labels):
            video_path = os.path.join(self.directory, video_id)
            frames = self._load_frames(video_path)
            batch_x.append(frames)
            batch_y.append(label_int)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y)

        if self.class_mode == "categorical":
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

        return batch_x, batch_y

    def _load_frames(self, video_path):
        frame_files = sorted(os.listdir(video_path))
        frame_files = frame_files[::self.skip][:self.nb_frames]

        # Garantir número consistente de frames
        if len(frame_files) < self.nb_frames:
            frame_files += [frame_files[-1]] * (self.nb_frames - len(frame_files))

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            img = tf.keras.preprocessing.image.load_img(frame_path, target_size=self.target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            frames.append(img_array)

        return np.stack(frames, axis=0)
