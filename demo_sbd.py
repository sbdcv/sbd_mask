import cv2
import time
import numpy as np

from deploy import classify
from lib.CenterFace.centerface import CenterFace


def draw(bboxs, img=None, thresh=0.5, max_size=0):
    img_cp = img.copy()

    len_line = int(img_cp.shape[1] / 5)
    pad_percent = int(img_cp.shape[1] / 2)
    x = int(img_cp.shape[1] / 25)
    y = int(img_cp.shape[0] / 25)
    pad_x = int(img_cp.shape[1] / 50)
    pad_y = int(img_cp.shape[0] / 25)
    pad_text = 5
    font_scale = (img_cp.shape[0] * img_cp.shape[1]) / (750 * 750)
    font_scale = max(font_scale, 0.25)
    font_scale = min(font_scale, 0.75)

    font_thickness = 1
    if max(img_cp.shape[0], img_cp.shape[1]) > 750: font_thickness = 2

    if bboxs.shape[0] == 0: return img

    bboxs = bboxs[np.where(bboxs[:, -1] > thresh)[0]]
    bboxs = bboxs.astype(int)

    cnt_mask = 0
    cnt_nomask = 0

    for bbox in bboxs:
        img_bbox = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if img_bbox.shape[0] * img_bbox.shape[1] < max_size:
            continue

        (ftype, prob) = classify(img_arr=img_bbox)
        prob_font_scale = (img_bbox.shape[0] * img_bbox.shape[1]) / (100 * 100)
        prob_font_scale = max(prob_font_scale, 0.25)
        prob_font_scale = min(prob_font_scale, 0.75)

        cv2.putText(img_cp, '{0:.2f}'.format(prob), (bbox[0] + 7, bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, prob_font_scale, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        if ftype == 0: cnt_mask += 1
        else: cnt_nomask += 1

        color = (0, 0, 255) if ftype else (0, 255, 0)

        cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    cv2.line(img_cp, (x, y), (x + len_line, y), (0, 255, 0), 2)
    cv2.putText(img_cp, 'Mask', (x + len_line + pad_x, y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    cv2.line(img_cp, (x, y + pad_y), (x + len_line, y + pad_y), (0, 0, 255), 2)
    cv2.putText(img_cp, 'face', (x + len_line + pad_x, y + pad_y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)
    
    mask_percent = (0 if cnt_mask == 0 else (cnt_mask / (cnt_mask + cnt_nomask))) * 100
    cv2.putText(img_cp, 'Mask percent: {:.0f}%'.format(mask_percent), (x + pad_percent, y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    return img_cp


def detect(centerface, height=360, width=640, video_path='data/0.mp4', visualize=False):
    cap = cv2.VideoCapture(video_path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            frame = cv2.resize(frame, (width, height))
            start_time = time.time()
            dets, lms = centerface(frame, threshold=0.5)

            frame = draw(dets, img=frame)

            print("FPS: ", 1.0 / (time.time() - start_time))


            if visualize:
                max_size = 1024

                if max(frame.shape[0], frame.shape[1]) > max_size:
                    scale = max_size / max(frame.shape[0], frame.shape[1])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.imshow('sbd', frame)
                # cv2.waitKeyEx()
                if cv2.waitKey() & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im_width = 640
    im_height = 360

    net = CenterFace(im_height, im_width)
    print('Finished loading model!')

    detect(net, im_height, im_width, visualize=True)
