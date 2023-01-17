import os
import cv2
import numpy as np

from numpy.lib.stride_tricks import as_strided
import tf_pose.pafprocess.pafprocess as pafprocess
from acllite.acllite_model import AclLiteModel
from common.quaternion import qrot
from common.visualization import render_animation
from common.skeleton import Skeleton


class ModelProcessor:
    def __init__(self, params):
        self.params = params
        self._model_width = params['width']
        self._model_height = params['height']
        self._h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15],
                                        joints_left=[4, 5, 6, 11, 12, 13],
                                        joints_right=[1, 2, 3, 14, 15, 16])

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, img_original):
        
        #preprocess image to get 'model_input'
        model_input = self.preprocess(img_original)
        
        # execute model inference
        result = self.model.execute([model_input]) 
        humans = post_process(result[0][0])
        
        return humans

    def preprocess(self, img):
        # np.set_printoptions(threshold=np.inf)
        scaled_img_data = cv2.resize(img, (self._model_width, self._model_height))
        pre_img = scaled_img_data[None].astype(np.float32).copy()
        # print(pre_img)
        # preprocessed_img = np.asarray(scaled_img_data, dtype=np.float32) / 255.
        # preprocessed_img = np.expand_dims(preprocessed_img,axis=0)
        # print(preprocessed_img.shape)
        # raise Exception
        return pre_img

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    
def nms(heatmaps):
    results = np.empty_like(heatmaps)
    for i in range(heatmaps.shape[-1]):
        heat = heatmaps[:,:,i]
        hmax = pool2d(heat, 3, 1, 1)
        keep = (hmax == heat).astype(float)

        results[:, :, i] = heat * keep
    return results

def estimate_paf(peaks, heat_mat, paf_mat):
    pafprocess.process_paf(peaks, heat_mat, paf_mat)

    humans = []
    humans_joint = []
    for part_idx in range(18):
        c_idx = int(pafprocess.get_part_cid(0, part_idx))
        if c_idx < 0:
            humans_joint.append([0,0])
            continue
        humans_joint.append([float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],\
                             float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0]])
    humans.append(humans_joint)

    return np.asarray(humans)

def post_process(heat):
    heatMat = heat[:,:,:19]
    pafMat = heat[:,:,19:]
    
    ''' Visualize Heatmap '''
    # print(heatMat.shape, pafMat.shape)
    # for i in range(19):
    #     plt.imshow(heatMat[:,:,i])
    # plt.savefig("outputs/heatMat.png")

    # blur = cv2.GaussianBlur(heatMat, (25, 25), 3)

    peaks = nms(heatMat)
    humans = estimate_paf(peaks, heatMat, pafMat)
    return humans


