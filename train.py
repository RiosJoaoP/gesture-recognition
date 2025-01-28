import os
import argparse
import configparser
from ast import literal_eval

import math

from models.resnet3d import Resnet3DBuilder
from data.data_loader import DataLoader
import data.image as kmg
from data.iterator import DataFrameVideoSequence
from utils.dir import mkdirs
from utils.save_history import save_training_history

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def train(args):

    nb_frames = config.getint('general', 'nb_frames')
    target_size = literal_eval(config.get('general', 'target_size'))
    nb_classes = config.getint('general', 'nb_classes')
    batch_size = config.getint('general', 'batch_size')
    epochs = config.getint('general', 'epochs')
    skip = config.getint('general', 'skip')

    root_data = config.get('path', 'root_data')
    vid_data = config.get('path', 'vid_data')

    model_name = config.get('path', 'model_name')
    data_model = config.get('path', 'data_model')

    weights_path = config.get('path', 'weights_path')

    csv_labels = config.get('path', 'csv_labels')
    csv_train = config.get('path', 'csv_train')
    csv_val = config.get('path', 'csv_val')

    vid_path = os.path.join(root_data, vid_data)
    model_path = os.path.join(root_data, data_model, model_name)
    labels_path = os.path.join(root_data, csv_labels)
    train_path = os.path.join(root_data, csv_train)
    val_path = os.path.join(root_data, csv_val)

    inp_shape = (nb_frames,) + target_size + (3,)

    inp_shape = (16, 64, 96, 3,)

    data = DataLoader(vid_path, labels_path, train_path, val_path)

    mkdirs(model_path, 0o755)

    gen = kmg.ImageDataGenerator()
    gen_train = gen.flow_video_from_dataframe(data.train_df, vid_path, path_classes=labels_path, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
    gen_val = gen.flow_video_from_dataframe(data.val_df, vid_path, path_classes=labels_path, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)

    net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes, drop_rate=0.5)

    lr_schedule = ExponentialDecay(
		initial_learning_rate=0.005,
		decay_steps=5000,
		decay_rate=0.96,  # Taxa de decaimento (ajuste conforme necessário)
		staircase=True  # Use True para decaimento em degraus
	)

    optimizer = SGD(learning_rate=lr_schedule, momentum=0.99, nesterov=True)
    net.compile(optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    if(weights_path != "None"):
        print("Loading weights from : " + weights_path)
        net.load_weights(weights_path)

    best_model_path = os.path.join(model_path, "model.best.h5")

    checkpointer_best = ModelCheckpoint(
        best_model_path, 
        monitor='val_accuracy',
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )

    nb_sample_train = data.train_df["video_id"].size
    nb_sample_val = data.val_df["video_id"].size

    history = net.fit(
        gen_train,
        steps_per_epoch=math.ceil(nb_sample_train / batch_size),
        epochs=epochs,
        validation_data=gen_val,
        validation_steps=math.ceil(nb_sample_val / batch_size),
        shuffle=True,
        verbose=1,
        callbacks=[checkpointer_best],
        workers=4,
        use_multiprocessing=True,
    )

    save_training_history(history, model_path)

    model_json = net.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)

    net.save_weights(model_name + ".h5")
    print(f"Saved model to disk on: {model_name}.json and {model_name}.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    train(config)
