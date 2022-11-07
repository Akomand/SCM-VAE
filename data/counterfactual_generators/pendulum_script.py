# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
import sys

import matplotlib.pyplot as plt
import os
from torchvision import transforms

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
# if not os.path.exists('./causal_data/pendulum_3/'):
#     os.makedirs('./causal_data/pendulum_3/train/')
#     os.makedirs('./causal_data/pendulum_3/test/')


def projection(theta, phi, x, y, base=-0.5):
    b = y - x * np.tan(phi)
    shade = (base - b) / np.tan(phi)
    return shade



# Input arrays
def generate(i, j, shadow_len=None, shadow_pos=None):
    # print(i)
    # sys.exit(0)
    X = np.zeros((i.shape[0], 4, 96, 96))
    scale = np.array([[0, 44], [100, 40], [7, 7.5], [10, 10]])
    count = 0
    empty = pd.DataFrame(columns=['i', 'j', 'shade', 'mid'])

    # i = 0
    # j = 68
    plt.rcParams['figure.figsize'] = (1.0, 1.0)
    theta = i * math.pi / 200.0
    phi = j * math.pi / 200.0
    x = 10 + 8 * np.sin(theta)
    y = 10.5 - 8 * np.cos(theta)

    # ball = plt.Circle((x, y), 1.5, color='firebrick')
    # gun = plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3)
    #
    # light = projection(theta, phi, 10, 10.5, 20.5)
    # sun = plt.Circle((light, 20.5), 3, color='orange')

    # calculate the mid index of
    ball_x = 10 + 9.5 * np.sin(theta)
    ball_y = 10.5 - 9.5 * np.cos(theta)

    if shadow_pos != None:
        mid = np.ones(i.shape[0]) * shadow_pos
    else:
        mid = (projection(theta, phi, 10.0, 10.5) + projection(theta, phi, ball_x, ball_y)) / 2


    if shadow_len != None:
        shade = np.ones(i.shape[0]) * shadow_len
    else:
        shade = np.maximum(3, np.abs(projection(theta, phi, 10.0, 10.5) - projection(theta, phi, ball_x, ball_y)))



    # shadow = plt.Polygon(([mid[e] - shade[e] / 2.0, -0.5], [mid[e] + shade[e] / 2.0, -0.5]), color='black', linewidth=3)

    for e in range(i.shape[0]):
        ball = plt.Circle((x[e], y[e]), 1.5, color='firebrick')
        gun = plt.Polygon(([10, 10.5], [x[e], y[e]]), color='black', linewidth=3)

        light = projection(theta[e], phi[e], 10, 10.5, 20.5)
        sun = plt.Circle((light, 20.5), 3, color='orange')

        shadow = plt.Polygon(([mid[e] - shade[e] / 2.0, -0.5], [mid[e] + shade[e] / 2.0, -0.5]), color='black',
                             linewidth=3)

        ax = plt.gca()
        ax.add_artist(gun)
        ax.add_artist(ball)
        ax.add_artist(sun)
        ax.add_artist(shadow)
        ax.set_xlim((0, 20))
        ax.set_ylim((-1, 21))

        new = pd.DataFrame({
            'i': (i[e] - scale[0][0]) / (scale[0][1] - 0),
            'j': (j[e] - scale[1][0]) / (scale[1][1] - 0),
            'shade': (shade[e] - scale[2][0]) / (scale[2][1] - 0),
            'mid': (mid[e] - scale[2][0]) / (scale[2][1] - 0)
        },

            index=[1])
        empty = empty.append(new, ignore_index=True)
        plt.axis('off')

        plt.savefig(
            './causal_data/generators/pendulum/a_' + str(int(i[e])) + '_' + str(int(j[e])) + '_' + str(int(shade[e])) + '_' + str(
                int(mid[e])) + '.png', dpi=96)

        pil_img = Image.open('./causal_data/generators/pendulum/a_' + str(int(i[e])) + '_' + str(int(j[e])) + '_' + str(int(shade[e])) + '_' + str(
                int(mid[e])) + '.png')
        pil_img = np.asarray(pil_img)
        transform = transforms.Compose([transforms.ToTensor()])
        pil_img = transform(pil_img)
        # print(pil_img.shape)
        # sys.exit(0)

        X[e] = pil_img

        plt.clf()
        count += 1

    return X, np.array([i, j, shade, mid]).T


# X, a = generate(np.array([0, 1, 2, 3]), np.array([68, 74, 89, 100]))
# print(X)


#
#
# for i in range(-40, 44):  # pendulum
#     for j in range(60, 148):  # light
#         if j == 100:
#             continue
#         plt.rcParams['figure.figsize'] = (1.0, 1.0)
#         theta = i * math.pi / 200.0
#         phi = j * math.pi / 200.0
#         x = 10 + 8 * math.sin(theta)
#         y = 10.5 - 8 * math.cos(theta)
#
#         ball = plt.Circle((x, y), 1.5, color='firebrick')
#         gun = plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3)
#
#         light = projection(theta, phi, 10, 10.5, 20.5)
#         sun = plt.Circle((light, 20.5), 3, color='orange')
#
#         # calculate the mid index of
#         ball_x = 10 + 9.5 * math.sin(theta)
#         ball_y = 10.5 - 9.5 * math.cos(theta)
#         mid = (projection(theta, phi, 10.0, 10.5) + projection(theta, phi, ball_x, ball_y)) / 2
#         shade = max(3, abs(projection(theta, phi, 10.0, 10.5) - projection(theta, phi, ball_x, ball_y)))
#
#         shadow = plt.Polygon(([mid - shade / 2.0, -0.5], [mid + shade / 2.0, -0.5]), color='black', linewidth=3)
#
#         ax = plt.gca()
#         ax.add_artist(gun)
#         ax.add_artist(ball)
#         ax.add_artist(sun)
#         ax.add_artist(shadow)
#         ax.set_xlim((0, 20))
#         ax.set_ylim((-1, 21))
#         new = pd.DataFrame({
#             'i': (i - scale[0][0]) / (scale[0][1] - 0),
#             'j': (j - scale[1][0]) / (scale[1][1] - 0),
#             'shade': (shade - scale[2][0]) / (scale[2][1] - 0),
#             'mid': (mid - scale[2][0]) / (scale[2][1] - 0)
#         },
#
#             index=[1])
#         empty = empty.append(new, ignore_index=True)
#         plt.axis('off')
#
#         plt.savefig(
#             './pendulum/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(
#                 int(mid)) + '.png', dpi=96)
#
#         plt.clf()
#         count += 1