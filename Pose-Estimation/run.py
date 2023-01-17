import argparse
import cv2
import numpy as np
import os
import sys
import vg
sys.path.append("./acllite")

from acllite.acllite_resource import AclLiteResource 

from processor_2d_3d import ModelProcessor as Processor_2D_to_3D
from processor_img_2d_open import ModelProcessor as Processor_Img_to_2D
from common.openpose_preprocess import open_pose_interpolation, open_pose_to_h36, plot_2d_skeleton

# MODEL_IMG_2D_PATH = "model/OpenPose_light.om"
MODEL_IMG_2D_PATH = 'model/OpenPose_for_TensorFlow_BatchSize_1.om'
# MODEL_2D_3D_PATH = "model/pose3d_rie_sim.om"
MODEL_2D_3D_PATH = "model/video_pose_3d.om"
INPUT_VIDEO = "data/pose3d_test_10s.mp4"

def run_img_to_2d(model_path, input_video_path):
    model_parameters = {
        'model_dir': model_path,
        'width': 656, 
        'height': 368, 
    }
    
    model_processor = Processor_Img_to_2D(model_parameters)
    cap = cv2.VideoCapture(input_video_path)
    keypoints = []
    output_canvases = []

    ret, img_original = cap.read()
    
    img_shape = img_original.shape
    cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(ret):
        cnt += 1; print(end=f"\rImg to 2D Prediction: {cnt} / {total_frames}")
        joint_list = model_processor.predict(img_original)
        joint_list = joint_list.squeeze(0)
        joint_list[:,0] = joint_list[:,0]*img_shape[1]
        joint_list[:,1] = joint_list[:,1]*img_shape[0]
        keypoints.append(joint_list)

        ret, img_original = cap.read()
    print()
    keypoints = np.asarray(keypoints)
    # keypoints = keypoints.squeeze(1)
    keypoints = open_pose_interpolation(keypoints)
    keypoints = open_pose_to_h36(keypoints)
    cap.release()

    cap = cv2.VideoCapture(input_video_path)
    output_canvases = []
    all_frames = []

    ret, img_original = cap.read()
    
    img_shape = img_original.shape
    cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(ret):
        cnt += 1; print(end=f"\rPlot 2D Skeleton: {cnt} / {total_frames}")

        canvas = plot_2d_skeleton(keypoints[cnt-1,:,:],img_original)
        output_canvases.append(canvas)
        
        all_frames.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

        ret, img_original = cap.read()
    print()
    cap.release()

    
    model_processor.model.destroy()

    return keypoints, img_shape, all_frames


def run_2d_to_3d(model_path, keypoints, input_video_path, output_video_dir, output_format, img_shape, all_frames):
    
    model_parameters = {
        'model_dir': model_path,
        'cam_h': img_shape[0],
        'cam_w': img_shape[1]
    }

    model_processor = Processor_2D_to_3D(model_parameters)
    
    output = model_processor.predict(keypoints)

    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_path = f'{output_video_dir}/output-{input_filename}.{output_format}'

    model_processor.generate_visualization(keypoints, output, input_video_path, video_output_path, all_frames)
    print("Output exported to {}".format(video_output_path))
    return output

def classify(keypoints_3d):
    '''
    Stop sign: compute angels between left hand and shoulder (best projection in front plane)
    Turn right: compute angles between left hand and spine = 90° 
                and angle between left hand and right hand <45°(best right hand locate in the down right side)
    Turn left: compute angles between right hand and spine = 90° 
                and angle between left hand and right hand <45°(best left hand locate in the down left side)
    H36M Order
    PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16
    '''
    for frame in range(len(keypoints_3d)):
        left_wrist = keypoints_3d[frame,13,:]
        left_elbow = keypoints_3d[frame,12,:]
        left_shoulder = keypoints_3d[frame,11,:]
        right_shoulder = keypoints_3d[frame,14,:]
        right_wrist = keypoints_3d[frame,16,:]
        right_elbow = keypoints_3d[frame,15,:]
        spine = keypoints_3d[frame,7,:]
        thorax = keypoints_3d[frame,8,:]

        left_hand_v = left_wrist - left_elbow
        shoulder_v = left_shoulder - right_shoulder
        right_hand_v = right_wrist - right_elbow
        spine_v = thorax - spine
        
        # stop sign angle
        angel_stop = vg.angle(left_hand_v,shoulder_v,look=vg.basis.z)
        # turn right sign angle
        angle_lh_spine = vg.angle(left_hand_v,spine_v)
        angle_lh_rh = vg.angle(left_hand_v,right_hand_v)
        # turn left sign angle
        angle_rh_spine = vg.angle(right_hand_v,spine_v)
        angle_rh_lh = vg.angle(right_hand_v,left_hand_v,look=vg.basis.z)       

        if left_wrist[1]<left_shoulder[1] and 88<=angel_stop<=92:
            return 'STOP'
        elif 85<=angle_lh_spine<=90 and angle_lh_rh<=60:
            return 'TURN RIGHT'
        if 85<=angle_rh_spine<=90 and angle_rh_lh<=60:
            return 'TURN LEFT'
    return None

if __name__ == "__main__":
    
    description = '3D Pose Lifting'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--model2D', type=str, default=MODEL_IMG_2D_PATH)
    parser.add_argument('--model3D', type=str, default=MODEL_2D_3D_PATH)
    parser.add_argument('--input', type=str, default=INPUT_VIDEO)
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output Path")
    parser.add_argument('--output_format', type=str, default='gif', help="Either gif or mp4")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    keypoints, img_shape, all_frames = run_img_to_2d(args.model2D, args.input)

    keypoints_3d = run_2d_to_3d(args.model3D, keypoints, args.input, args.output_dir, args.output_format, img_shape, all_frames)
    sign = classify(keypoints_3d)
    if sign:
        print('='*20 + 'Gesture' + '='*20)
        print('Traffic police perform ' + f'{sign}'+ ' gesture')
        print('='*20 + 'Gesture' + '='*20)
    else:
        print('='*20 + 'Gesture' + '='*20)
        print('No traffic gesture performed')
        print('='*20 + 'Gesture' + '='*20)