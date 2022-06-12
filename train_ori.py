
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_model
from utils import *
from metrics import *
import argparse
def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([args.img_size, args.img_size, 3])
    y.set_shape([args.img_size, args.img_size, 2])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py', description='training') 
    parser.add_argument('--vggpretrain', '-vgg_pre', default='./vgg_pretrain.h5', type=str, required=False,  help='vggpretrain for transfer learning')
    parser.add_argument('--img_size', default=256, type=int, required=False,  help='img_size for training')
    parser.add_argument('--train_path', '-tp', default='stas/train/', type=str, required=False,  help='train_path for training')
    parser.add_argument('--valid_path', '-vp', default='stas/valid/', type=str, required=False,  help='valid_path for training')
    parser.add_argument('--store_folder', '-sf', default='files', type=str, required=False,  help='store weight in folder while training')
    parser.add_argument('--model_path', '-wp', default="files/model_withvgg19.h5", type=str, required=False,  help='store weight in folder while training')

    parser.add_argument('--batch_size', '-v', default=4, type=int, required=False,  help='batchsize for training')
    parser.add_argument('--epoch', '-e', default=100, type=int, required=False,  help='epoch for training')
    parser.add_argument('--EarlyStopping', '-es', default=50, type=int, required=False,  help='EarlyStopping for training')
    parser.add_argument('--learning_rate', '-lr', default=8e-5, type=int, required=False,  help='learning_rate for training')
    args = parser.parse_args()
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir(args.store_folder)

    train_path = args.train_path
    valid_path = args.valid_path

    ## Training
    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.png")))

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Validation
    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.png")))

    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epoch
    lr = args.learning_rate
    shape = (args.img_size, args.img_size, 3)

    model = build_model(shape,args.vggpretrain)
    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]
    
    train_dataset = tf_dataset(train_x, train_y, batch=args.batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=args.batch_size)
    
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=args.EarlyStopping, restore_best_weights=True)
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)
