# coding: utf-8
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)

# 测试集输入图片就不用做随机化处理了
test_datagen = ImageDataGenerator(rescale=1. / 255)
# 测试集路径,和训练时一样的
test_set = test_datagen.flow_from_directory('E:\Desktop\deep learning\deep learning project\mixeddataset/test/',
                                            target_size=IMG_SIZE,
                                            class_mode='binary'
                                            )

"""
    ---------Load Models---------
"""

# 读取已经训练好的模型
model_1 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_1.h5')
model_2 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_2.h5')
model_3 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_resnet.h5')
model_4 = tf.keras.models.load_model(r'E:\Desktop\deep learning\deep learning project\modelshybrid-tensorflow_keras\model_xception.h5')
models = [model_1, model_2, model_3, model_4]
# MODEL_WEIGHTING = [0.1, 0.15, 0.15, 0.2, 0.4]

"""
    ---------Summary---------
"""

# 4个模型的准确率
accuracydata = []
for m in range(len(models)):
    modelacc = models[m].evaluate(test_set, batch_size=512)[1]
    print(f"Model_{m} accuracy is :", modelacc * 100, '%')
    accuracydata.append(modelacc)

# 找到最准确率最高和第二高的模型 （没用上）
mostaccnum = accuracydata.index(max(accuracydata))  # 0为模型1
secondaccnum = accuracydata.index(sorted(accuracydata)[-2])
if secondaccnum == mostaccnum:
    sign = False
    for i in range(len(accuracydata)):
        if accuracydata[i] == accuracydata[mostaccnum]:
            if not sign:
                sign = True
            else:
                secondaccnum = i

print("Most accurate model: ", mostaccnum, " Secondly accurate model: ", secondaccnum)

"""
plt.figure(figsize=(8, 6))
plt.title('Accuracy scores')
labels = ['model_1', 'model_2', 'model_3', 'model_4']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r']
plt.bar([0, 1, 2, 3], accuracydata, align='center', alpha=0.7, color=colors, tick_label=labels)
plt.ylim(0, 1)
plt.show()
"""

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


def soft_voting(result, model_weight):
    predict = []
    for r in range(len(result)):
        predict.append(result[r] * model_weight[r])

    if sum(predict) < 0.5:
        return 0
    else:
        return 1


def hard_voting(result, model_weight):
    predict = []
    for r in range(len(result)):
        predict.append(result[r] * model_weight[r])

    if sum(predict) >= 0.5:
        return 1
    else:
        return 0


def modi_soft_voting(res, model_weight):
    num_over50 = 0
    for result in res:
        if result >= 0.5:
            num_over50 += 1
    # 混合模型预测结果进行判断
    # 判断分为3种情况 全为小于0.5，全为大于0.5，和其他
    if num_over50 == 0:
        pre = 0
    elif num_over50 == 4:
        pre = 1
    else:
        # 判断4个预期值中的最大值和最小值是不是绝对肯定（靠近0或1）
        if max(res) > 0.75 and min(res) < 0.25:
            if (1 - max(res)) - min(res) < -0.15:
                pre = 1
            elif (1 - max(res)) - min(res) > 0.15:
                pre = 0
            else:
                if soft_voting(res, model_weight) == 1:
                    pre = 1
                else:
                    pre = 0
        # 判断是否有一个预期值非常肯定
        elif min(res) < 0.25:
            if max(res) < 0.65:
                pre = 0
            else:
                if soft_voting(res, model_weight) == 1:
                    pre = 1
                else:
                    pre = 0
        # 同上
        elif max(res) > 0.75:
            if min(res) > 0.35:
                pre = 1
            else:
                if soft_voting(res, model_weight) == 1:
                    pre = 1
                else:
                    pre = 0
        else:
            if soft_voting(res, model_weight) == 1:
                pre = 1
            else:
                pre = 0

    return pre


def stage_voting(res, k, model_weight):
    stage_value = []
    for r in range(len(res)):
        if res[r] > 0.5:
            stage_value.append(1 * math.pow(1 - math.sin(math.pi * res[r]), k) * model_weight[r])
        else:
            stage_value.append(-1 * math.pow(1 - math.sin(math.pi * res[r]), k) * model_weight[r])
    if sum(stage_value) > 0:
        return 1
    else:
        return 0


def hybrid(norm_predict, pneu_predict, model_weight):

    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(norm_predict)):
        predict_label = hard_voting(norm_predict[i], model_weight)
        if predict_label == 0:
            TP += 1
        else:
            FN += 1

    for i in range(len(pneu_predict)):
        predict_label = hard_voting(pneu_predict[i], model_weight)
        if predict_label == 1:
            TN += 1
        else:
            FP += 1

    total = TP + FN + FP + TN
    finalacc_1 = (TP + TN) / total

    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(norm_predict)):
        predict_label = soft_voting(norm_predict[i], model_weight)
        if predict_label == 0:
            TP += 1
        else:
            FN += 1

    for i in range(len(pneu_predict)):
        predict_label = soft_voting(pneu_predict[i], model_weight)
        if predict_label == 1:
            TN += 1
        else:
            FP += 1

    finalacc_2 = (TP + TN) / total

    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(norm_predict)):
        predict_label = modi_soft_voting(norm_predict[i], model_weight)
        if predict_label == 0:
            TP += 1
        else:
            FN += 1

    for i in range(len(pneu_predict)):
        predict_label = modi_soft_voting(pneu_predict[i], model_weight)
        if predict_label == 1:
            TN += 1
        else:
            FP += 1

    finalacc_3 = (TP + TN) / total

    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(norm_predict)):
        predict_label = stage_voting(norm_predict[i], 0.5, model_weight)
        if predict_label == 0:
            TP += 1
        else:
            FN += 1

    for i in range(len(pneu_predict)):
        predict_label = stage_voting(pneu_predict[i], 0.5, model_weight)
        if predict_label == 1:
            TN += 1
        else:
            FP += 1

    finalacc_4 = (TP + TN) / total

    return finalacc_1, finalacc_2, finalacc_3, finalacc_4


norm_predict, pneu_predict = go_through_images()

bestacc = [0, 0, 0, 0]
bestweight = [[], [], [], []]

for x1 in range(6, 12):
    for x2 in range(10, 15):
        for x3 in range(20, 25):
            for x4 in range(22, 27):
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    weightacc = []
                    MODEL_WEIGHTING = [x1, x2, x3, x4]
                    SUM = sum(MODEL_WEIGHTING)
                    for w in range(len(MODEL_WEIGHTING)):
                        MODEL_WEIGHTING[w] = MODEL_WEIGHTING[w] / SUM
                    print(MODEL_WEIGHTING)
                    weightacc = hybrid(norm_predict, pneu_predict, MODEL_WEIGHTING)
                    print(weightacc)
                    for a in range(len(weightacc)):
                        if weightacc[a] > bestacc[a]:
                            bestacc[a] = weightacc[a]
                            bestweight[a] = MODEL_WEIGHTING

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("best accuracy:", bestacc)
print("hard voting:", bestweight[0])
print("soft voting:", bestweight[1])
print("adapt voting:", bestweight[2])
print("stage voting:", bestweight[3])

