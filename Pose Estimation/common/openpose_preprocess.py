import cv2
import numpy as np
import math
import os, sys
sys.path.append('../..')

JOINT_LIMB = [[0, 7], [7, 8], [8, 9], [9, 10], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
COLOR = [[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0],\
        [0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255]]

def plot_2d_skeleton(joint_list, image_original):
    # plot the pose on original image
    canvas = image_original
    for idx, limb in enumerate(JOINT_LIMB):
        joint_from, joint_to = joint_list[limb[0]], joint_list[limb[1]]
        canvas = cv2.line(canvas, tuple(joint_from.astype(int)), tuple(joint_to.astype(int)), color=COLOR[idx], thickness=4)
    
    return canvas   

def open_pose_interpolation(open_pose_keypoints):
    frame_num,_,_ = open_pose_keypoints.shape
    x = np.arange(frame_num)
    for i in range(14):
        for j in range(2):
            x_mask = np.nonzero(open_pose_keypoints[:,i,j])
            if len(x_mask[0]) != frame_num:
                y_mask = open_pose_keypoints[x_mask[0],i,j]
                open_pose_keypoints[:,i,j] = np.interp(x,x_mask[0],y_mask)
    return open_pose_keypoints


def open_pose_to_h36(open_pose_keypoints):
    # Open Pose Order
    # Nose = 0, Neck = 1, RShoulder = 2, RElbow = 3, RWrist = 4, LShoulder = 5, LElbow = 6
    # LWrist = 7, RHip = 8, RKnee = 9, RAnkle = 10, LHip = 11, LKnee = 12, LAnkle = 13
    # REye = 14, LEye = 15, REar = 16, LEar = 17, Background = 18 

    # H36M Order
    # PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    # L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    # L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16
    h36_keypoints = []
    frame_num,_,_ = open_pose_keypoints.shape 
    for i in range(frame_num):
        keypoints = np.zeros([17, 2])
        
        # Pelvis
        keypoints[0][0] = (open_pose_keypoints[i][8][0] + open_pose_keypoints[i][11][0]) / 2
        keypoints[0][1] = (open_pose_keypoints[i][8][1] + open_pose_keypoints[i][11][1]) / 2

        # Right Hip
        keypoints[1] = open_pose_keypoints[i][8]

        # Right Knee
        keypoints[2] = open_pose_keypoints[i][9]

        # Right Foot
        keypoints[3] = open_pose_keypoints[i][10]

        # Left Hip
        keypoints[4] = open_pose_keypoints[i][11]

        # Left Knee
        keypoints[5] = open_pose_keypoints[i][12]

        # Left Foot
        keypoints[6] = open_pose_keypoints[i][13]

        # Spine
        keypoints[7][0] = (open_pose_keypoints[i][2][0] + open_pose_keypoints[i][5][0] + open_pose_keypoints[i][8][0] + open_pose_keypoints[i][11][0]) / 4
        keypoints[7][1] = (open_pose_keypoints[i][2][1] + open_pose_keypoints[i][5][1] + open_pose_keypoints[i][8][1] + open_pose_keypoints[i][11][1]) / 4

        # Thorax
        keypoints[8] = open_pose_keypoints[i][1]

        # Nose
        keypoints[9][0] = (open_pose_keypoints[i][0][0] + open_pose_keypoints[i][1][0]) / 2
        keypoints[9][1] = (open_pose_keypoints[i][0][1] + open_pose_keypoints[i][1][1]) / 2

        # Head
        keypoints[10] = open_pose_keypoints[i][0]

        # Left Shoulder
        keypoints[11] = open_pose_keypoints[i][5]

        # Left Elbow
        keypoints[12] = open_pose_keypoints[i][6]

        # Left Wrist
        keypoints[13] = open_pose_keypoints[i][7]

        # Right Shoulder
        keypoints[14] = open_pose_keypoints[i][2]

         # Right Elbow
        keypoints[15] = open_pose_keypoints[i][3]

         # Right Wrist
        keypoints[16] = open_pose_keypoints[i][4]

        h36_keypoints.append(keypoints)

    return np.asarray(h36_keypoints)
