import numpy as np
import cv2
import torch
import torch.nn.functional as m_func


csize = 224
net = cv2.dnn.readNetFromONNX(r'model/sbd_mask.onnx')

def classify(img_arr=None, thresh=0.5):
    blob = cv2.dnn.blobFromImage(img_arr, scalefactor=1 / 255, size=(csize, csize), mean=(0, 0, 0),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    heatmap = net.forward(['349'])
    match = m_func.log_softmax(torch.from_numpy(heatmap[0][0]), dim=0).data.numpy()
    index = np.argmax(match)

    return (0 if index > thresh else 1, match[0])

