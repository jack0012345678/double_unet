
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from train_ori import tf_dataset

def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred
def evaluate_normal(model, x_data):
    THRESHOLD = 0.5
    total = []
    for i, x in tqdm(enumerate(x_data), total=len(x_data)):
        n = x


        x = read_image(x)
        _, h, w, _ = x.shape

        y_pred1 = parse(model.predict(x)[0][..., -2])
        y_pred2 = parse(model.predict(x)[0][..., -1])
        
        line = np.ones((h, 10, 3)) * 255.0
        
        all_images = [
            x[0] * 255.0, line,

            mask_to_3d(y_pred1) * 255.0, line,
            mask_to_3d(y_pred2) * 255.0
        ]
        mask1 = np.concatenate([mask_to_3d(y_pred1)], axis=1)
        mask2 = np.concatenate([mask_to_3d(y_pred2)], axis=1)
        mask = mask1 + mask2
        mask = cv2.resize(mask, (1716, 942), interpolation=cv2.INTER_AREA)
        mask[mask > 0.15] = 255
        mask[mask < 0.15] = 0
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./results/" + n[10:-4]+'.png', mask)
        print("./results/" + n[10:-4]+'.png')

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results/")

    batch_size = 2

    test_path = "stas/test/"
    test_x = sorted(glob(os.path.join(test_path, "*.jpg")))
    model = load_model_weight("files/model_b12_stas_aug_new_final_512.h5")
    #model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_x)
