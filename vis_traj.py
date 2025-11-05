import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2
import cairosvg


def cal_distance(gt, pred):
    return math.sqrt((float(gt[0]) - float(pred[0])) ** 2 + (
                float(gt[1]) - float(pred[1])) ** 2 + (
                          float(gt[2]) - float(pred[2])) ** 2)


def caldir(yaw, pitch, roll):
    # 定义方向向量
    direction_vector = np.array([1, 0, 0])  # 初始方向向量

    # 根据欧拉角进行旋转
    rotation_matrix = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    # 旋转方向向量
    rotated_vector = np.dot(rotation_matrix, direction_vector)
    return rotated_vector


test_obj = [folder.split('/')[0] for folder in open('vis_file_v1.txt')]
k = 0
for i, obj in enumerate(test_obj):
    traj_gt = []
    pose_file = pd.read_excel('/home/slj/EndoTraj/NEPose-main/data/{}/traj.xlsx'.format(obj), header=None)
    for pose in pose_file.values:
        if str(pose[4]) != 'OK':
            continue
        traj_gt.append(pose[9:12])
    length = 0
    for i in range(1, len(traj_gt)):
        length += cal_distance(traj_gt[i], traj_gt[i - 1])

    d_gt = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono_500_4/results_{}/directions_gt.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_gt.append(pos.split(' '))
    d_pred_offset = []
    d_pred_endoslam = []
    d_pred_mono = []
    d_pred_simcol = []
    d_pred_ours = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_offset_500_4/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_offset.append(pos.split(' '))
    with open(
            '/home/slj/EndoTraj/NEPose-main/results/results_endoslam_500_4/results_{}/directions_pred.txt'.format(obj),
            'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_endoslam.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono_500_4/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_mono.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_simcol_500_4/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_simcol.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ours_500_4/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_ours.append(pos.split(' '))

    traj_gt = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ours_500_4/results_{}/traj_gt.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_gt.append(pos.split(' '))
    traj_pred_offset = []
    traj_pred_endoslam = []
    traj_pred_mono = []
    traj_pred_simcol = []
    traj_pred_ours = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_offset_500_4/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_offset.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_endoslam_500_4/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_endoslam.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono_500_4/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_mono.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_simcol_500_4/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_simcol.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ab2_500_4/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_ours.append(pos.split(' '))

    x_gt = [0]
    y_gt = [0]
    z_gt = [0]
    x_pred_offset = [0]
    y_pred_offset = [0]
    z_pred_offset = [0]
    x_pred_endoslam = [0]
    y_pred_endoslam = [0]
    z_pred_endoslam = [0]
    x_pred_mono = [0]
    y_pred_mono = [0]
    z_pred_mono = [0]
    x_pred_simcol = [0]
    y_pred_simcol = [0]
    z_pred_simcol = [0]
    x_pred_ours = [0]
    y_pred_ours = [0]
    z_pred_ours = [0]
    for j in range(1, len(traj_gt)):
        x_gt.append((float(traj_gt[j][0])-float(traj_gt[0][0])))
        y_gt.append((float(traj_gt[j][1])-float(traj_gt[0][1])))
        z_gt.append((float(traj_gt[j][2])-float(traj_gt[0][2])))
        x_pred_offset.append((float(traj_pred_offset[j][0])-float(traj_gt[0][0])))
        y_pred_offset.append((float(traj_pred_offset[j][1])-float(traj_gt[0][1])))
        z_pred_offset.append((float(traj_pred_offset[j][2])-float(traj_gt[0][2])))
        x_pred_endoslam.append((float(traj_pred_endoslam[j][0])-float(traj_gt[0][0])))
        y_pred_endoslam.append((float(traj_pred_endoslam[j][1])-float(traj_gt[0][1])))
        z_pred_endoslam.append((float(traj_pred_endoslam[j][2])-float(traj_gt[0][2])))
        x_pred_mono.append((float(traj_pred_mono[j][0])-float(traj_gt[0][0])))
        y_pred_mono.append((float(traj_pred_mono[j][1])-float(traj_gt[0][1])))
        z_pred_mono.append((float(traj_pred_mono[j][2])-float(traj_gt[0][2])))
        x_pred_simcol.append((float(traj_pred_simcol[j][0])-float(traj_gt[0][0])))
        y_pred_simcol.append((float(traj_pred_simcol[j][1])-float(traj_gt[0][1])))
        z_pred_simcol.append((float(traj_pred_simcol[j][2])-float(traj_gt[0][2])))
        x_pred_ours.append((float(traj_pred_ours[j][0])-float(traj_gt[0][0])))
        y_pred_ours.append((float(traj_pred_ours[j][1])-float(traj_gt[0][1])))
        z_pred_ours.append((float(traj_pred_ours[j][2])-float(traj_gt[0][2])))
    # for j in range(1, len(traj_gt)):
    #     x_gt.append((float(traj_gt[j][0])-float(traj_gt[j-1][0])))
    #     y_gt.append((float(traj_gt[j][1])-float(traj_gt[j-1][1])))
    #     z_gt.append((float(traj_gt[j][2])-float(traj_gt[j-1][2])))
    #     x_pred_offset.append((float(traj_pred_offset[j][0])-float(traj_pred_offset[j-1][0])))
    #     y_pred_offset.append((float(traj_pred_offset[j][1])-float(traj_pred_offset[j-1][1])))
    #     z_pred_offset.append((float(traj_pred_offset[j][2])-float(traj_pred_offset[j-1][2])))
    #     x_pred_endoslam.append((float(traj_pred_endoslam[j][0])-float(traj_pred_endoslam[j-1][0])))
    #     y_pred_endoslam.append((float(traj_pred_endoslam[j][1])-float(traj_pred_endoslam[j-1][1])))
    #     z_pred_endoslam.append((float(traj_pred_endoslam[j][2])-float(traj_pred_endoslam[j-1][2])))
    #     x_pred_mono.append((float(traj_pred_mono[j][0])-float(traj_pred_mono[j-1][0])))
    #     y_pred_mono.append((float(traj_pred_mono[j][1])-float(traj_pred_mono[j-1][1])))
    #     z_pred_mono.append((float(traj_pred_mono[j][2])-float(traj_pred_mono[j-1][2])))
    #     x_pred_simcol.append((float(traj_pred_simcol[j][0])-float(traj_pred_simcol[j-1][0])))
    #     y_pred_simcol.append((float(traj_pred_simcol[j][1])-float(traj_pred_simcol[j-1][1])))
    #     z_pred_simcol.append((float(traj_pred_simcol[j][2])-float(traj_pred_simcol[j-1][2])))
    #     x_pred_ours.append((float(traj_pred_ours[j][0])-float(traj_pred_ours[j-1][0])))
    #     y_pred_ours.append((float(traj_pred_ours[j][1])-float(traj_pred_ours[j-1][1])))
    #     z_pred_ours.append((float(traj_pred_ours[j][2])-float(traj_pred_ours[j-1][2])))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', labelsize=14)
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    ax.set_title('The whole length: {:.2f}cm'.format(length / 10), fontsize=20)
    ax.plot(x_gt, y_gt, z_gt, label='GT', color='k')
    ax.plot(x_pred_offset, y_pred_offset, z_pred_offset, label='OffsetNet', color='c', linewidth=2)
    ax.plot(x_pred_iros, y_pred_iros, z_pred_iros, label='Fried et al.', color='c', linewidth=2)
    ax.plot(x_pred_dapose, y_pred_dapose, z_pred_dapose, label='DualAttention PoseNet.', color='c', linewidth=2)
    ax.plot(x_pred_tmi, y_pred_tmi, z_pred_tmi, label='Yang et al.', color='c', linewidth=2)
    ax.plot(x_pred_endoslam, y_pred_endoslam, z_pred_endoslam, label='Attention PoseNet', color='g', linewidth=2)
    ax.plot(x_pred_mono, y_pred_mono, z_pred_mono, label='PoseResNet', color='b', linewidth=2)
    ax.plot(x_pred_simcol, y_pred_simcol, z_pred_simcol, label='Rau et al.', color='y', linewidth=2)
    ax.plot(x_pred_ours, y_pred_ours, z_pred_ours, label='Ours', color='r', linewidth=2)
    ax.set_xlabel('X axis/mm', labelpad=12, fontsize=16)
    ax.set_ylabel('Y axis/mm', labelpad=12, fontsize=16)
    ax.set_zlabel('Z axis/mm', labelpad=12, fontsize=16)
    ax.legend(fontsize=16)
    plt.show()
    # for i in range(0, len(x_gt)-1, 2):
    #     j = (i+1) // 2
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.tick_params(axis='both', labelsize=14)
    #     plt.rcParams['font.family'] = ['serif']
    #     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    #     # ax.set_title('The whole length: {:.2f}cm'.format(length / 10), fontsize=20)
    #     ax.plot(x_gt[i:i+2], y_gt[i:i+2], z_gt[i:i+2], '--',label='GT', color='k', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_offset[i+1]], [y_gt[i]]+[y_pred_offset[i+1]], [z_gt[i]]+[z_pred_offset[i+1]], '--', label='OffsetNet', color='c', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_endoslam[i+1]], [y_gt[i]]+[y_pred_endoslam[i+1]], [z_gt[i]]+[z_pred_endoslam[i+1]], '--', label='Endo-SfM', color='g', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_mono[i+1]], [y_gt[i]]+[y_pred_mono[i+1]], [z_gt[i]]+[z_pred_mono[i+1]], '--', label='Monodepth2', color='b', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_simcol[i+1]], [y_gt[i]]+[y_pred_simcol[i+1]], [z_gt[i]]+[z_pred_simcol[i+1]], '--', label='SimCol', color='y', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_ours[i+1]], [y_gt[i]]+[y_pred_ours[i+1]], [z_gt[i]]+[z_pred_ours[i+1]], '--', label='Ours', color='r', linewidth=2)
    #     # d_0 = caldir(float(d_gt[j+1][0]), float(d_gt[j+1][1]), float(d_gt[j+1][2]))
    #     # d_offset = caldir(float(d_pred_offset[j+1][0]), float(d_pred_offset[j+1][1]), float(d_pred_offset[j+1][2]))
    #     # d_mono = caldir(float(d_pred_mono[j+1][0]), float(d_pred_mono[j+1][1]), float(d_pred_mono[j+1][2]))
    #     # d_endoslam = caldir(float(d_pred_endoslam[j+1][0]), float(d_pred_endoslam[j+1][1]), float(d_pred_endoslam[j+1][2]))
    #     # d_simcol = caldir(float(d_pred_simcol[j+1][0]), float(d_pred_simcol[j+1][1]), float(d_pred_simcol[j+1][2]))
    #     # d_ours = caldir(float(d_pred_ours[j+1][0]), float(d_pred_ours[j+1][1]), float(d_pred_ours[j+1][2]))
    #     #
    #     # ax.quiver(x_gt[i+1], y_gt[i+1], z_gt[i+1], d_0[0], d_0[1], d_0[2], color='k', label='Direction of GT', linewidth=2, length=0.05)
    #     # ax.quiver(x_pred_offset[i+1], y_pred_offset[i+1], z_pred_offset[i+1], d_offset[0], d_offset[1], d_offset[2], color='c', label='Direction of OffsetNet', linewidth=2, length=0.05)
    #     # ax.quiver(x_pred_mono[i+1], y_pred_mono[i+1], z_pred_mono[i+1], d_mono[0], d_mono[1], d_mono[2], color='b', label='Direction of MonoDepth2', linewidth=2, length=0.05)
    #     # ax.quiver(x_pred_endoslam[i+1], y_pred_endoslam[i+1], z_pred_endoslam[i+1], d_endoslam[0], d_endoslam[1], d_endoslam[2], color='g', label='Direction of Endo-SfM', linewidth=2, length=0.05)
    #     # ax.quiver(x_pred_simcol[i+1], y_pred_simcol[i+1], z_pred_simcol[i+1], d_simcol[0], d_simcol[1], d_simcol[2], color='y', label='Direction of SimCol', linewidth=2, length=0.05)
    #     # ax.quiver(x_pred_ours[i+1], y_pred_ours[i+1], z_pred_ours[i+1], d_ours[0], d_ours[1], d_ours[2], color='r', label='Direction of Ours', linewidth=2, length=0.05)
    #     ax.set_xlabel('X axis/mm', labelpad=12, fontsize=16)
    #     ax.set_ylabel('Y axis/mm', labelpad=12, fontsize=16)
    #     ax.set_zlabel('Z axis/mm', labelpad=12, fontsize=16)
    #     ax.axis('off')
    #     ax.legend(fontsize=16)
    #     plt.show()

#
# for i in range(1, 7):
#     cairosvg.svg2png(url='vis_res/comp{}.svg'.format(i), write_to='vis_res/comp{}.png'.format(i), dpi=300)
# img1 = cv2.imread('vis_res/comp1.png')[280:2900, 1000:3800]
# img2 = cv2.imread('vis_res/comp4.png')[280:2900, 1000:3800]
# for k in range(2, 4):
#     img = cv2.imread('vis_res/comp{}.png'.format(k))[280:2900, 1000:3800]
#     img1 = cv2.hconcat([img1, img])
# for k in range(5, 7):
#     img = cv2.imread('vis_res/comp{}.png'.format(k))[280:2900, 1000:3800]
#     img2 = cv2.hconcat([img2, img])
# out = cv2.vconcat([img1, img2])
# out = cv2.resize(out, (0,0), fx=0.5, fy=0.5)
# cv2.imwrite('vis_res/comp.png', out)


