import pyrealsense2 as rs
import numpy as np
import random
import torch
import time
import pydarknet
from pydarknet import Detector, Image
import cv2

pydarknet.set_cuda_device(0)

#net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))


def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [box[0], box[1]] #确定索引深度的中心像素位置
    min_val = min(abs(box[2]), abs(box[3])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)
def dectshow(org_img, class_ids, boxs, depth_data):
    img = org_img.copy()
    if len(boxs)>0:
        for box, text_class in zip(boxs, class_ids):
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
            dist = get_mid_pos(org_img, box, depth_data, 24)
            txt = str(text_class) + str(dist / 1000)[:4] + 'm'
            cv2.putText(img, txt, (int(box[0]+box[2]//2), int(box[1]+box[3]//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if text_class == 'bowl':
                print( text_class + str(dist / 1000)[:4] + 'm')
            if text_class == 'sports ball':
                print( text_class + str(dist / 1000)[:4] + 'm')
                    
    cv2.imshow('dec_img', img)
    

if __name__ == "__main__":
    # Configure depth and color streams
    print("debug starts")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    start = time.time()
    print("hello")
    end = time.time()
    print(end - start)
        
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            print("hello")
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())
            
            
            boxs = []
            class_ids = []
            
            img_darknet = Image(color_image)
            results = net.detect(img_darknet)
            for category, score, bounds in results:
                x, y, w, h = bounds
                boxs.append([x, y, w, h])
                class_ids.append(category)

            dectshow(color_image, class_ids, boxs, depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            end = time.time()
            print(end - start)
            start = time.time()
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
