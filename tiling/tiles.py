import numpy as np
from scipy.misc import imread
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2

class tiles(object):
    def __init__(self, image_path, tile_size):
        self.image=cv2.imread(image_path)
        self.tile_size=tile_size
        tempimg=image.load_img(image_path)
        tempimg=image.img_to_array(tempimg)
        self.image_tiles=[]
        self.tile_offsets=[]
        m,n,_=self.image.shape
        for i in range(int(np.ceil(m/float(tile_size[0])))):
            for j in range(int(np.ceil(n/float(tile_size[1])))):
                self.tile_offsets.append((i*tile_size[0],j*tile_size[1]))
                temp_img_tile=np.zeros((tile_size[0],tile_size[1],3))
                bottom_tile_boundary=min(tile_size[0]*(i+1),m)
                right_tile_boundary=min(tile_size[1]*(j+1),n)
                temp_img_tile[0:(bottom_tile_boundary-tile_size[0]*i),0:(right_tile_boundary-tile_size[1]*j),:]=tempimg[tile_size[0]*i:bottom_tile_boundary,tile_size[1]*j:right_tile_boundary,:]
                self.image_tiles.append(temp_img_tile)

def draw_boundingBoxes(tile_object, results, threshold):
    img=np.copy(tile_object.image)
    for i in range(len(tile_object.image_tiles)):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
        # Get detections with confidence higher than threshold.
        top_indices = [x for x, conf in enumerate(det_conf) if conf >= threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        for j in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[j] * tile_object.tile_size[1])) + tile_object.tile_offsets[i][1]
            ymin = int(round(top_ymin[j] * tile_object.tile_size[0])) + tile_object.tile_offsets[i][0]
            xmax = int(round(top_xmax[j] * tile_object.tile_size[1])) + tile_object.tile_offsets[i][1]
            ymax = int(round(top_ymax[j] * tile_object.tile_size[0])) + tile_object.tile_offsets[i][0]
            score = top_conf[i]
            label = int(top_label_indices[i])
            if label > 0:
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0))
    return img
         
