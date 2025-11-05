import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import cairosvg


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
#
#
test_obj = [folder.split('/')[0] for folder in open('vis_file_v1.txt')]
# k = 0
for i, obj in enumerate(test_obj):
    traj_gt = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono_500_4/results_{}/traj_gt.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_gt.append(pos.split(' '))

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
    with open('/home/slj/EndoTraj/NEPose-main/results/results_endoslam_500_4/results_{}/directions_pred.txt'.format(obj),
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
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ab6_500_4/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_ours.append(pos.split(' '))
#
    for i in range(len(d_gt)):
        d_0 = caldir(float(d_gt[i][0]), float(d_gt[i][1]), float(d_gt[i][2]))
        d_offset = caldir(float(d_pred_offset[i][0]), float(d_pred_offset[i][1]), float(d_pred_offset[i][2]))
        d_mono = caldir(float(d_pred_mono[i][0]), float(d_pred_mono[i][1]), float(d_pred_mono[i][2]))
        d_endoslam = caldir(float(d_pred_endoslam[i][0]), float(d_pred_endoslam[i][1]), float(d_pred_endoslam[i][2]))
        d_simcol = caldir(float(d_pred_simcol[i][0]), float(d_pred_simcol[i][1]), float(d_pred_simcol[i][2]))
        d_ours = caldir(float(d_pred_ours[i][0]), float(d_pred_ours[i][1]), float(d_pred_ours[i][2]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, d_0[0], d_0[1], d_0[2], label='GT', color='k', linewidth=2)
        ax.quiver(0, 0, 0, d_offset[0], d_offset[1], d_offset[2], label='OffsetNet', color='c', linewidth=2)
        ax.quiver(0, 0, 0, d_mono[0], d_mono[1], d_mono[2], label='Monodepth2', color='b')
        ax.quiver(0, 0, 0, d_endoslam[0], d_endoslam[1], d_endoslam[2], label='Endo-SfM', color='g', linewidth=2)
        ax.quiver(0, 0, 0, d_simcol[0], d_simcol[1], d_simcol[2], label='SimCol', color='y', linewidth=2)
        ax.quiver(0, 0, 0, d_ours[0], d_ours[1], d_ours[2], label='Ours', color='r', linewidth=2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X', fontsize=18)
        ax.set_ylabel('Y', fontsize=18)
        ax.set_zlabel('Z', fontsize=18)
        # ax.axis('off')
        # ax.legend(fontsize=18)
        plt.show()

# for i in range(1, 19):
#     cairosvg.svg2png(url='./results/quali_res/d{}.svg'.format(i), write_to='./results/quali_res/d{}.png'.format(i), dpi=300)
# img1 = cv2.imread('./results/quali_res/d1.png')[280:2874, 1138:3800]
# img2 = cv2.imread('./results/quali_res/d7.png')[280:2874, 1138:3800]
# img3 = cv2.imread('./results/quali_res/d13.png')[280:2874, 1138:3800]
# for k in range(2, 7):
#     img = cv2.imread('./results/quali_res/d{}.png'.format(k))[280:2874, 1138:3800]
#     img1 = cv2.hconcat([img1, img])
# for k in range(8, 13):
#     img = cv2.imread('./results/quali_res/d{}.png'.format(k))[280:2874, 1138:3800]
#     img2 = cv2.hconcat([img2, img])
# for k in range(14, 19):
#     img = cv2.imread('./results/quali_res/d{}.png'.format(k))[280:2874, 1138:3800]
#     img3 = cv2.hconcat([img3, img])
# out = cv2.vconcat([img1, img2, img3])
# out = cv2.resize(out, (0, 0), fx=0.35, fy=0.35)
# cv2.imwrite('./results/quali_res/dir_vis_comp.png', out)



