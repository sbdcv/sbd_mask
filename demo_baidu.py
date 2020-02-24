import time

import cv2

import paddlehub as hub



module = hub.Module(name="pyramidbox_lite_mobile_mask")
# 将模型保存在test_program文件夹之中
module.processor.save_inference_model(dirname="test_program")

def detect_img():
    img_data = cv2.imread(r"test_program/6e1c74e85298b5120f9f884ad348c243.jpg")

    input_dict = {"data": [img_data]}

    for i in range(10):
        start = time.time()
        # execute predict and print the result
        results = module.face_detection(data=input_dict)
        for result in results:
            left = float(result['data']['left'])
            right = float(result['data']['right'])
            bottom = float(result['data']['bottom'])
            top = float(result['data']['top'])
            conf = float(result['data']['confidence'])
            color = (0, 0, 255)
            if result['data']['label'] != 'MASK':
                color = (255, 0, 0)
            cv2.rectangle(img_data, (int(left), int(top)), (int(right), int(bottom)), color, 1)
        print("time", time.time() - start)
        cv2.imshow("asdf", img_data)
        cv2.waitKey(1)


def detct_video():
    vc = cv2.VideoCapture(r"G:\zhaozhengwei\sbd_project\onnx_mask\data\0.mp4")  # 读入视频文件

    index = 0
    while True:  # 循环读取视频帧
        index += 1
        rval, img_o = vc.read()

        if img_o is None:
            break
        # if index % 2 == 0:
        #     continue
        input_dict = {"data": [img_o]}
        start = time.time()
        # execute predict and print the result
        results = module.face_detection(data=input_dict)
        for result in results:
            left = float(result['data']['left'])
            right = float(result['data']['right'])
            bottom = float(result['data']['bottom'])
            top = float(result['data']['top'])
            conf = float(result['data']['confidence'])
            color = (0, 255, 0)
            if result['data']['label'] != 'MASK':
                color = (0, 0, 255)
            cv2.rectangle(img_o, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        print("time", time.time() - start)

        cv2.imshow("baidu",  cv2.resize(img_o, (640, 360)))
        cv2.waitKey(1)


if __name__ == '__main__':
    detct_video()
# input_dict = {"image": test_img_path}

# execute predict and print the result
# results = module.face_detection(data=input_dict)
# for result in results:
#     print(result)
#
# # 预测结果展示
# img = mpimg.imread("detection_result/test_mask_detection.jpg")
