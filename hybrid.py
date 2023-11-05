# coding: utf-8
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory('E:\Desktop\deep learning\deep learning project\mixeddataset/test/',
                                            target_size=IMG_SIZE,
                                            class_mode='binary'
                                            )
"""
    ---------Load Models---------
"""


model_1 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_1.h5')
model_2 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_2.h5')
model_3 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_resnet.h5')
model_4 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_xception.h5')
models = [model_1, model_2, model_3, model_4]
MODEL_WEIGHTING = [0.1, 0.16, 0.35, 0.39]

"""
    ---------Summary---------
"""
accuracydata = []
for m in range(len(models)):
    modelacc = models[m].evaluate(test_set, batch_size=512)[1]
    print(f"Model_{m} accuracy is :", modelacc * 100, '%')
    accuracydata.append(modelacc)


"""
    ---------Hybrid Models---------
"""

"""         进行模型杂交，使用的是集成学习中软投票的方法，并进行了一些改动。输入单张图片，使用模型返回的预期值来确定图片类型        """

# 模型准确率的平均值
averageacc = sum(accuracydata) / len(models)


def predict_single_img(model, imagedata):
    # 会返回0-1之间的值，1为肺炎，0为正常
    result = np.squeeze(model.predict(imagedata))

    return result


def go_through_images():
    # 数据集test文件夹路径
    normpath = "E:\Desktop\deep learning\deep learning project\mixeddataset/test/NORMAL"
    pneupath = "E:\Desktop\deep learning\deep learning project\mixeddataset/test/PNEUMONIA"
    normimgs = os.listdir(normpath)
    pneuimgs = os.listdir(pneupath)

    norm_predict = []
    pneu_predict = []

    for allimg in normimgs:
        img_path = os.path.join('%s/%s' % (normpath, allimg))
        print(img_path)
        res = []
        # 输入图片，返回模型的预期值
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.array(img) / 255.
        img = np.expand_dims(img, axis=3)
        img = img.reshape((-1, 224, 224, 3))

        for m in range(len(models)):
            res.append(predict_single_img(models[m], img))
        print(res)
        norm_predict.append(res)

    for allimg in pneuimgs:
        img_path = os.path.join('%s/%s' % (pneupath, allimg))
        print(img_path)
        res = []
        # 输入图片，返回模型的预期值
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.array(img) / 255.
        img = np.expand_dims(img, axis=3)
        img = img.reshape((-1, 224, 224, 3))

        for m in range(len(models)):
            res.append(predict_single_img(models[m], img))
        print(res)
        pneu_predict.append(res)

    return norm_predict, pneu_predict


def soft_voting(result):
    predict = []
    for r in range(len(result)):
        predict.append(result[r] * MODEL_WEIGHTING[r])

    if sum(predict) < 0.5:
        return 0
    else:
        return 1


# sp: without model weights
def soft_voting_un(result):
    predict = []
    for r in range(len(result)):
        predict.append(result[r])

    if ((sum(predict)) / 5.0) < 0.5:
        return 0
    else:
        return 1


def hard_voting(result):
    predict = 0
    for r in range(len(result)):
        if result[r] > 0.5:
            predict += 1 * MODEL_WEIGHTING[r]

    if predict >= 0.5:
        return 1
    else:
        return 0


def hard_voting_un(result):
    predict = 0
    for r in range(len(result)):
        if result[r] > 0.5:
            predict += 1

    if predict >= len(result) / 2.0:
        return 1
    else:
        return 0


def stage_voting(res, k):
    stage_value = []
    for r in range(len(res)):
        if res[r] > 0.5:
            stage_value.append(1 * math.pow(1 - math.sin(math.pi * res[r]), k) * MODEL_WEIGHTING[r])
        else:
            stage_value.append(-1 * math.pow(1 - math.sin(math.pi * res[r]), k) * MODEL_WEIGHTING[r])
    if sum(stage_value) > 0:
        return 1
    else:
        return 0


def stage_voting_un(res, k):
    stage_value = []
    for r in range(len(res)):
        if res[r] > 0.5:
            stage_value.append(1 * k * math.pow(1 - math.sin(math.pi * res[r]), k))
        else:
            stage_value.append(-1 * k * math.pow(1 - math.sin(math.pi * res[r]), k))
    if sum(stage_value) > 0:
        return 1
    else:
        return 0


norm_predict, pneu_predict = go_through_images()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Hard-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = hard_voting(norm_predict[i])
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = hard_voting(pneu_predict[i])
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Hard voting final accuracy: ", finalacc * 100, '%')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Un model weight Hard-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = hard_voting_un(norm_predict[i])
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = hard_voting_un(pneu_predict[i])
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Un model weight Hard voting final accuracy: ", finalacc * 100, '%')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Normal soft-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = soft_voting(norm_predict[i])
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = soft_voting(pneu_predict[i])
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Soft voting final accuracy: ", finalacc * 100, '%')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Un model weight Normal soft-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = soft_voting_un(norm_predict[i])
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = soft_voting_un(pneu_predict[i])
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Un model weight Soft voting final accuracy: ", finalacc * 100, '%')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Stage-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = stage_voting(norm_predict[i], 0.5)
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = stage_voting(pneu_predict[i], 0.5)
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Stage voting final accuracy: ", finalacc * 100, '%')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Un model weight Stage-voting: ")
TP, FN, FP, TN = 0, 0, 0, 0
for i in range(len(norm_predict)):
    predict_label = stage_voting_un(norm_predict[i], 0.5)
    if predict_label == 0:
        TP += 1
    else:
        FN += 1

for i in range(len(pneu_predict)):
    predict_label = stage_voting_un(pneu_predict[i], 0.5)
    if predict_label == 1:
        TN += 1
    else:
        FP += 1

normacc = TP / (TP + FN)
pneuacc = TN / (TN + FP)
total = TP + FN + FP + TN
finalacc = (TP + TN) / total
print("Normal set accuracy:", normacc * 100, '%')
print("Pneumonia set accuracy:", pneuacc * 100, '%')
print("Un model weight Stage voting final accuracy: ", finalacc * 100, '%')

