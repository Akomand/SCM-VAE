# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

count = 0
if not os.path.exists('./causal_data/flow_noise/'):
    os.makedirs('./causal_data/flow_noise/train/')
    os.makedirs('./causal_data/flow_noise/test/')



def generate(r, hole, h, flow, i_r=None, i_hole=None, i_h=None, i_flow=None):
    X = np.zeros((r.shape[0], 4, 96, 96))

    r = r.numpy()
    h = h.numpy()
    flow = flow.numpy()
    hole = hole.numpy()

    if i_r != None:
        r = i_r.numpy()
    ball_r = r / 30.0
    h_raw = 10 * (h - np.power(ball_r, 3))

    if i_h != None:
        h = i_h.numpy()
    else:
        h = np.power(ball_r, 3) + h_raw / 10.0

    if i_hole != None:
        hole = i_hole.numpy()

    deep = hole / 3.0
    # plt.rcParams['figure.figsize'] = (1.0, 1.0)
    # ax = plt.gca()


    for e in range(r.shape[0]):
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        ax = plt.gca()

        # water in cup
        rect = plt.Rectangle(([3, 0]), 5, 5 + h[e], color='lightskyblue')
        ax.add_artist(rect)
        ball = plt.Circle((5.5, +ball_r[e] + 0.5), ball_r[e], color='firebrick')
        ## cup
        left = plt.Polygon(([3, 0], [3, 19]), color='black', linewidth=2)
        right_1 = plt.Polygon(([8, 0], [8, deep[e]]), color='black', linewidth=2)
        right_2 = plt.Polygon(([8, deep[e] + 0.4], [8, 19]), color='black', linewidth=2)
        ax.add_artist(left)
        ax.add_artist(right_1)
        ax.add_artist(right_2)
        ax.add_artist(ball)

        # water line
        y = np.linspace(deep[e], 0.5)

        epsilon = 0.01 * np.max([np.abs(np.random.randn(1)), 1])
        # x = np.sqrt(2 * (0.98 + epsilon) * h[e] * (deep[e] - y)) + 8
        if i_flow != None:
            x = np.linspace(i_flow[e], 0.5)
        else:
            x = np.sqrt(2 * (0.98 + epsilon) * h[e] * (deep[e] - y)) + 8

        # x_max = x[-1] - 8
        x_true = np.sqrt(2 * (0.98) * h[e] * (deep[e] - 0.5))

        plt.plot(x, y, color='lightskyblue', linewidth=2)

        ##ground
        x = np.linspace(0, 20, num=50)
        y = np.zeros(50) + 0.2
        plt.plot(x, y, color='black', linewidth=2)

        ax.set_xlim((0, 20))
        ax.set_ylim((0, 20))

        plt.axis('off')
        plt.savefig('./causal_data/generators/flow/a_' + str(int(r[e])) + "_" + str(int(hole[e])) + "_" + str(int(h[e])) + "_" + str(
            int(x_true*10)) + '.png', dpi=96)

        pil_img = Image.open('./causal_data/generators/flow/a_' + str(int(r[e])) + "_" + str(int(hole[e])) + "_" + str(int(h[e])) + "_" + str(
            int(x_true*10)) + '.png')

        pil_img = np.asarray(pil_img)
        transform = transforms.Compose([transforms.ToTensor()])
        pil_img = transform(pil_img)

        X[e] = pil_img

        plt.clf()

    return X, np.array([r, h, x_true, hole]).T

#
# def generate(r, h, flow, hole, i_r=None, i_h=None, i_flow=None, i_hole=None):
#     X = np.zeros((r.shape[0], 4, 96, 96))
#     # print(h.shape)
#
#     r = r.numpy()
#     h = h.numpy()
#     flow = flow.numpy()
#     hole = hole.numpy()
#
#     ball_r = r / 30.0
#
#     h_raw = 10 * (h - pow(ball_r, 3))
#
#     if i_r != None:
#         i_r = i_r.numpy()
#         r = i_r
#         ball_r = i_r / 30.0
#
#     h = pow(ball_r, 3) + h_raw / 10.0
#
#     if i_h != None:
#         h = i_h
#
#     if i_hole != None:
#         hole = i_hole
#
#     deep = hole / 3.0
#
#     if i_flow != None:
#         x_true = i_flow
#     else:
#         x_true = np.sqrt(2 * (0.98) * h * (deep - 0.5)) * 10
#
#     plt.rcParams['figure.figsize'] = (1.0, 1.0)
#     ax = plt.gca()
#
#
#     for e in range(r.shape[0]):
#         # water in cup
#         rect = plt.Rectangle(([3, 0]), 5, 5 + h[e], color='lightskyblue')
#         ax.add_artist(rect)
#         ball = plt.Circle((5.5, +ball_r[e] + 0.5), ball_r[e], color='firebrick')
#         ## cup
#         left = plt.Polygon(([3, 0], [3, 19]), color='black', linewidth=2)
#         right_1 = plt.Polygon(([8, 0], [8, deep[e]]), color='black', linewidth=2)
#         right_2 = plt.Polygon(([8, deep[e] + 0.4], [8, 19]), color='black', linewidth=2)
#         ax.add_artist(left)
#         ax.add_artist(right_1)
#         ax.add_artist(right_2)
#         ax.add_artist(ball)
#
#         # water line
#         y = np.linspace(deep[e], 0.5)
#         epsilon = 0.01 * np.max([np.abs(np.random.randn(1)), 1])
#         x = np.sqrt(2 * (0.98 + epsilon) * h[e] * (deep[e] - y)) + 8
#         x_max = x[-1] - 8
#
#         # if i_flow != None:
#         #     x_true = i_flow
#         # else:
#         #     x_true = np.sqrt(2 * (0.98) * h[e] * (deep[e] - 0.5)) * 10
#
#         plt.plot(x, y, color='lightskyblue', linewidth=2)
#
#         ##ground
#         x = np.linspace(0, 20, num=50)
#         y = np.zeros(50) + 0.2
#         plt.plot(x, y, color='black', linewidth=2)
#
#         ax.set_xlim((0, 20))
#         ax.set_ylim((0, 20))
#
#         plt.axis('off')
#
#         plt.savefig('./causal_data/generators/flow/a_' + str(int(r[e])) + "_" + str(int(h[e])) + "_" + str(
#             int(x_true[e])) + "_" + str(int(hole[e])) + '.png', dpi=96)
#
#         pil_img = Image.open('./causal_data/generators/flow/a_' + str(int(r[e])) + "_" + str(int(h[e])) + "_" + str(
#             int(x_true[e])) + "_" + str(int(hole[e])) + '.png')
#         pil_img = np.asarray(pil_img)
#         transform = transforms.Compose([transforms.ToTensor()])
#         pil_img = transform(pil_img)
#         # print(pil_img.shape)
#         # sys.exit(0)
#
#         X[e] = pil_img
#
#         plt.clf()
#
#     return X, np.array([r, h, x_true, hole]).T



# for r in range(5, 35):
#     for h_raw in range(10, 40):
#         for hole in range(6, 15):
#             ball_r = r / 30.0
#             h = pow(ball_r, 3) + h_raw / 10.0
#             deep = hole / 3.0
#             plt.rcParams['figure.figsize'] = (1.0, 1.0)
#             ax = plt.gca()
#
#             # water in cup
#             rect = plt.Rectangle(([3, 0]), 5, 5 + h, color='lightskyblue')
#             ax.add_artist(rect)
#             ball = plt.Circle((5.5, +ball_r + 0.5), ball_r, color='firebrick')
#             ## cup
#             left = plt.Polygon(([3, 0], [3, 19]), color='black', linewidth=2)
#             right_1 = plt.Polygon(([8, 0], [8, deep]), color='black', linewidth=2)
#             right_2 = plt.Polygon(([8, deep + 0.4], [8, 19]), color='black', linewidth=2)
#             ax.add_artist(left)
#             ax.add_artist(right_1)
#             ax.add_artist(right_2)
#             ax.add_artist(ball)
#
#             # water line
#             y = np.linspace(deep, 0.5)
#             epsilon = 0.01 * np.max([np.abs(np.random.randn(1)), 1])
#             x = np.sqrt(2 * (0.98 + epsilon) * h * (deep - y)) + 8
#             x_max = x[-1] - 8
#             x_true = np.sqrt(2 * (0.98) * h * (deep - 0.5))
#             plt.plot(x, y, color='lightskyblue', linewidth=2)
#
#             ##ground
#             x = np.linspace(0, 20, num=50)
#             y = np.zeros(50) + 0.2
#             plt.plot(x, y, color='black', linewidth=2)
#
#             ax.set_xlim((0, 20))
#             ax.set_ylim((0, 20))
#
#             plt.axis('off')
#             if count == 4:
#                 plt.savefig('./causal_data/flow_noise/test/a_' + str(int(r)) + "_" + str(int(h)) + "_" + str(
#                     int(x_true * 10)) + "_" + str(int(hole)) + '.png', dpi=96)
#                 count = 0
#             else:
#                 plt.savefig('./causal_data/flow_noise/train/a_' + str(int(r)) + "_" + str(int(h)) + "_" + str(
#                     int(x_true * 10)) + "_" + str(int(hole)) + '.png', dpi=96)
#                 count += 1
#             plt.clf()







