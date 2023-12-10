# import math
# from ultralytics import YOLO
# import cv2
#
# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#
#     # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
#
#     model = YOLO('D:/detectionProject/best.pt')
#
#     classNames = ['sleeping']
#
#     while True:
#         success, img = cap.read()
#         results = model(img, stream=True)
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 print(x1, y1, x2, y2)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 # print(box.conf[0])
#                 conf = math.ceil((box.conf[0]*100))/100
#                 cls = int(box.cls[0])
#                 class_name = classNames[cls]
#                 label = f'{class_name}{conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
#                 cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
#
#         yield img
#
#
#         # out.write(img)
#         # cv2.imshow("Image", img)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#     # out.release()
# cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # while True:
# #     success, img = cap.read()
# #     out.write(img)
# #     cv2.imshow("Image", img)
# #     # plt.show()
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#
# # out.release()
#


import math
import os
from ultralytics import YOLO
import cv2

def video_detection(path_x):
    # Check if path_x is a valid path to a video file
    if path_x is None or not os.path.exists(path_x):
        print(f"Invalid video path: {path_x}")
        return

    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO('D:/detectionProject/yolov8s.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    while True:
        success, img = cap.read()
        # Check if cap.read() has returned a valid frame
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

    # Release the video capture object
    cap.release()

    cv2.destroyAllWindows()
