
import os
import cv2
import numpy as np

from common.pose_decode import decode_pose
from acllite.acllite_model import AclLiteModel

from run_openpose_tf import post_process

heatmap_width = 92
heatmap_height = 92

class ModelProcessor:
    def __init__(self, params):
        self.params = params
        self._model_width = params['width']
        self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, img_original):
        
     
        model_input = self.pre_process(img_original)

      
        result = self.model.execute([model_input]) 


        JOINT_LIMB = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]
        COLOR = [[0, 255, 255], [0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0], [0, 0, 255], [255, 0, 0],[255, 0, 0],[255, 0, 0], [255, 0, 0]]
       
        
        out = post_process(result[0][0])[0]

        joint_list = [None] * 18
        for key, value in out.body_parts.items(): 
            a = np.array([self._model_width* value.x, self._model_height* value.y])
            joint_list[key] = a

        #do nothing here
        last_frame=None
        if last_frame is None:
            pass
            ##
          
            if joint_list[0] is None:
                ind = None
                if joint_list[14] is not None:
                    ind = 14
                elif joint_list[15] is not None:
                    ind = 15
                elif joint_list[16] is not None:
                    ind = 16
                elif joint_list[17] is not None:
                    ind = 17
                joint_list[0] = joint_list[ind]

            ## heart
            if joint_list[1] is None:
                if joint_list[5] is not None: joint_list[1] = joint_list[5]
                elif joint_list[2] is not None: joint_list[1] = joint_list[2]

            ## missing one arm/leg, copy from the other arm/leg
            if joint_list[8] is None and joint_list[9] is None and joint_list[10] is None:
                joint_list[8] = joint_list[11]; joint_list[9] = joint_list[12]; joint_list[10] = joint_list[13]
            if joint_list[11] is None and joint_list[12] is None and joint_list[13] is None:
                joint_list[11] = joint_list[8]; joint_list[12] = joint_list[9]; joint_list[13] = joint_list[10] 
            if joint_list[2] is None and joint_list[3] is None and joint_list[4] is None:
                joint_list[2] = joint_list[5]; joint_list[3] = joint_list[6]; joint_list[4] = joint_list[7]
            if joint_list[5] is None and joint_list[6] is None and joint_list[7] is None:
                joint_list[5] = joint_list[2]; joint_list[6] = joint_list[3]; joint_list[7] = joint_list[4]

            ## get two shoulder
            if joint_list[2] is None: joint_list[2] = joint_list[5]
            if joint_list[5] is None: joint_list[5] = joint_list[2]

            ## when one side of forearm is missing
            if joint_list[3] is None and joint_list[4] is None: 
                joint_list[4] = joint_list[8]; joint_list[3] = (joint_list[2]+joint_list[4])/2
            if joint_list[6] is None and joint_list[7] is None: 
                joint_list[7] = joint_list[11]; joint_list[6] = (joint_list[5]+joint_list[7])/2

            ## when one point on forearm is missing
            if joint_list[3] is None: joint_list[3] = 2* joint_list[1] - joint_list[6]
            if joint_list[4] is None: joint_list[4] = 2* joint_list[1] - joint_list[7]
            if joint_list[6] is None: joint_list[6] = 2* joint_list[1] - joint_list[3]
            if joint_list[7] is None: joint_list[7] = 2* joint_list[1] - joint_list[4]

           ## when both legs are missing
            if joint_list[8] is None and joint_list[9] is None and joint_list[10] is None\
                and joint_list[11] is None and joint_list[12] is None and joint_list[13] is None:
                
                joint_list[8] = joint_list[4]
                joint_list[11] = joint_list[7]
               

            ## when one point on legs is missing
            if joint_list[9] is None: joint_list[9] = np.array([joint_list[8][0], joint_list[8][1]*2 - joint_list[3][1]])
            if joint_list[10] is None: joint_list[10] = np.array([joint_list[9][0], joint_list[9][1]*2 - joint_list[8][1]])
            if joint_list[12] is None: joint_list[12] = np.array([joint_list[11][0], joint_list[11][1]*2 - joint_list[6][1]])
            if joint_list[13] is None: joint_list[13] = np.array([joint_list[12][0], joint_list[12][1]*2 - joint_list[11][1]])

        else:
           pass


        temp = [np.array([0,0])]*14
        

        temp[0] = joint_list[5]
        temp[1] = joint_list[6]
        temp[2] = joint_list[7]
        temp[3] = joint_list[2]
        temp[4] = joint_list[3]
        temp[5] = joint_list[4]
        temp[6] = joint_list[11]
        temp[7] = joint_list[12]
        temp[8] = joint_list[13]
        temp[9] = joint_list[8]
        temp[10] = joint_list[9]
        temp[11] = joint_list[10]
        temp[12] = joint_list[0]
        temp[13] = joint_list[1]
        
        joint_list = temp

        for ind, joint in enumerate(joint_list):
            if joint is None:
                print(ind)
                joint_list[ind] = last_frame[ind]

        canvas = img_original
        for idx, limb in enumerate(JOINT_LIMB):
            joint_from, joint_to = joint_list[limb[0]].copy(), joint_list[limb[1]].copy()
            joint_from *= 3
            joint_to *= 3
            canvas = cv2.line(canvas, tuple(joint_from.astype(int)), tuple(joint_to.astype(int)), color=COLOR[idx], thickness=4)
        

        return canvas, joint_list

    def preprocess(self,img_original):
        '''
        preprocessing: resize image to model required size, and normalize value between [0,1]
        '''
        scaled_img_data = cv2.resize(img_original, (self._model_width, self._model_height))
        preprocessed_img = np.asarray(scaled_img_data, dtype=np.float32) / 255.
        
        return preprocessed_img

    def pre_process(self, img):
        model_input = cv2.resize(img, (self._model_width, self._model_height))
        return model_input[None].astype(np.float32).copy()
    


