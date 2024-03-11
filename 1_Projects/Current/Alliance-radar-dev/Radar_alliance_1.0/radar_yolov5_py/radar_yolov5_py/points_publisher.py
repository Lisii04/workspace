# !/usr/bin/env python3
try:
    # [ROS2]
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError


    # [USER_IMPORT]
    import array
    import time
    import cv2 as cv
    import logging


    # [YOLOv5]
    import argparse
    import csv
    import os
    import platform
    import sys
    from pathlib import Path
    import torch

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from ultralytics.utils.plotting import Annotator, colors, save_one_box
    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
    from utils.torch_utils import select_device, smart_inference_mode
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    from utils.plots import Annotator, colors
except Exception as e:
    print(">\033[31m[ERROR]\033[0m[Import载入错误，请检查环境是否存在问题]")
    print(">\033[31m[ERROR]\033[0m[错误信息]" + str(e) + "\n")

# [坐标数据发送类]>
class PointsPublisher(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("[start]%s" % name)
        self.command_publisher_ = self.create_publisher(Float32MultiArray,"points_data", 10) 
        
    def send_points(self,points):
        msg = Float32MultiArray()
        msg.data = points
        self.command_publisher_.publish(msg) 
# [坐标数据发送类]<

# [测试图像接收类]>
class ImageSubscriber(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("[start]%s" % name)
        self.subscription = self.create_subscription(Image,'Radar_Sim', self.image_callback,10)

    def image_callback(self, msg):
        bridge = CvBridge()
        try:
            # 将 ROS Image 转换为 OpenCV Mat
            self.get_logger().info(f"Image Message Header: {msg.header}")
            self.get_logger().info(f"Image Encoding: {msg.encoding}")
            self.get_logger().info(f"Image Height: {msg.height}")
            self.get_logger().info(f"Image Width: {msg.width}")
            self.get_logger().info(f"Image Step: {msg.step}")
            

            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

            self.get_logger().info(f"Image shape: {cv_image.shape}")
            
            # 在这里可以对图像进行处理或显示
            cv.imshow('Image', cv_image)
            cv.waitKey(1)
        except Exception as e:
            self.get_logger().error('Error converting image: %s' % str(e))
# [测试图像接收类]<

# [YOLOv5识别网络主程序]>>>>>>
@smart_inference_mode()
def run(
        PointsPublisher_obj,
        car_weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        armor_weights = ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        identfy_car=True,  # identfy which kind of robot 
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # [载入输入图像/视频流]>
    try:
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
    except Exception as e:
        logging.error("[ERROR][摄像头/视频加载失败，请检查文件路径是否正确]")
        logging.error("[ERROR][错误信息]" + str(e) + "\n")
    # [载入输入图像/视频流]<

    # [载入模型]>
    try:
        # [Car model load]>
        device = select_device(device)
        model = DetectMultiBackend(car_weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # [Car model load]<

        # [Armor model load]>
        armor_model = attempt_load(armor_weights,0)  # load FP32 armor_model
        armor_names = ["armor_1_red","armor_2_red","armor_3_red","armor_4_red","armor_5_red","watcher_red","armor_1_blue","armor_2_blue","armor_3_blue","armor_4_blue","armor_5_blue","watcher_blue"]
        # [Armor model load]<
    except Exception as e:
        logging.error("[ERROR][模型加载失败，请检查文件路径是否正确]")
        logging.error("[ERROR][错误信息]" + str(e) + "\n")
    # [载入模型]<

    # [载入数据]>
    try:
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
    except Exception as e:
        logging.error("[ERROR][数据加载失败]")
        logging.error("[ERROR][错误信息]" + str(e) + "\n")
    # [载入数据]<


    # [Run inference]>
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        try:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image

                start_time = time.time()

                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if identfy_car else im0  # for identfy_car
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                pointslist = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]}'
                        confidence = float(conf)

                        if identfy_car or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            pointslist.append(1)
                            pointslist.append((xyxy[0] + xyxy[2]) / 2)
                            pointslist.append(xyxy[3])
                        if identfy_car and label == "car":

                            # [model2:装甲板识别网络]>>>>>>
                            try:
                                # [裁减装甲板图片]
                                croped_image = imc[int(xyxy[1])+int((int(xyxy[3])-int(xyxy[1]))/3):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]

                                if croped_image.shape[0] > 0 and croped_image.shape[1] > 0:
                                    # [转换格式]
                                    croped_image = cv.resize(croped_image,( ( round(croped_image.shape[1]/32) )* 32, ( round(croped_image.shape[1]/32) )* 32 ))
                                    car_image = torch.from_numpy(croped_image).to(device)
                                    car_image = car_image.to(torch.float32) # uint8 to fp32
                                    car_image = car_image / 255.0  # 0 - 255 to 0.0 - 1.0
                                    if len(car_image.shape) == 3:
                                        car_image = car_image[None]  # expand for batch dim
                                    car_image = car_image.transpose(2, 3)
                                    car_image = car_image.transpose(1, 2)

                                    # Inference
                                    armor_pred = armor_model(car_image)[0]

                                    # NMS
                                    armor_pred = non_max_suppression(armor_pred, conf_thres = 0.25, iou_thres = 0.45, classes = None, max_det = 1000)

                                    # Process predictions
                                    for i, det in enumerate(armor_pred):  # detections per image
                                        armor_s = ''
                                        armor_annotator = Annotator(croped_image, line_width=line_thickness, example=str(names))
                                        if len(det):
                                            # Rescale boxes from img_size to im0 size
                                            det[:, :4] = scale_boxes(car_image.shape[2:], det[:, :4], croped_image.shape).round()

                                            # Print results
                                            for c in det[:, -1].unique():
                                                n = (det[:, -1] == c).sum()  # detections per class
                                                armor_s += str(n.item()) + ' ' + str(armor_names[int(c)]) + ' '  # add to string

                                            # [对装甲板分类]>
                                            score = {}
                                            px_blue = 0
                                            px_red = 0
                                            
                                            for *armor_xyxy, conf, cls in reversed(det):
                                                c = int(cls)  # integer class

                                                # [计算每种识别结果的总置信度]>
                                                if armor_names[c] in score:
                                                    score[armor_names[c]] += conf
                                                else:
                                                    score[armor_names[c]] = conf
                                                # [计算每种识别结果的总置信度]<

                                                # [筛选颜色]
                                                armor_image = croped_image[int(armor_xyxy[1])+int((int(armor_xyxy[3])-int(armor_xyxy[1]))/3):int(armor_xyxy[3]),int(armor_xyxy[0]):int(armor_xyxy[2])]
                                            
                                                for x in range(armor_image.shape[0]):   # 图片的高
                                                    for y in range(armor_image.shape[1]):   # 图片的宽
                                                        px = armor_image[x,y]
                                                        px_blue += px[0]
                                                        px_red += px[2]
                                                

                                            armor_color = ""
                                            if px_blue >= px_red:
                                                armor_color = "blue"  
                                            else:
                                                armor_color = "red"  

                                                                                    
                                            conf_max = 0
                                            armor_max = ''
                                            for armor_name in score:
                                                if score[armor_name] > conf_max:
                                                    armor_max = armor_name
                                                    conf_max = score.get(armor_name)
                                            # [对装甲板分类]<

                                            # if (armor_color == "blue" and armor_names.index(armor_max) > 5) or (armor_color == "red" and armor_names.index(armor_max) <= 5):
                                             
                                            center_x = (xyxy[0] + xyxy[2]) / 2

                                            armor_label = armor_max
                                            # annotator.box_label(xyxy, armor_label, color=colors(c, True))
                                            # armor_annotator.box_label(armor_xyxy, armor_label, color=colors(c, True))
                                            # cv.putText(croped_image,str(px_blue)+"|"+str(px_red),(0,20),1,1,(0,255,0),2)
                                            # cv.imshow("armor_image",croped_image)
                                            # cv.waitKey()
                                                            
                                            # [识别结果添加]
                                            # pointslist.append(armor_names.index(armor_label))
                                            # pointslist.append(center_x)
                                            # pointslist.append(xyxy[3])
                            except Exception as e:
                                logging.error("[ERROR][装甲板识别过程运行错误]")
                                logging.error("[ERROR][错误信息]" + str(e) + "\n")
                            # [model2:装甲板识别网络]<<<<<<
                    
                    # [填装数据]>
                    points = array.array('f',pointslist)
                    PointsPublisher_obj.send_points(points)
                    # [填装数据]<
                # [计算帧率]
                end_time = time.time()
                process_time = end_time - start_time
                FPS = round(1.0/process_time) 
                
                # [显示结果]
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p),0)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), 800, 500)

                    logging.info("[识别结果数量]" + str(len(pointslist)/3) 
                        + "[每一帧用时]" + str(round(process_time*1000))+"ms"
                        + "[FPS]" + str(FPS))

                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
        except Exception as e:
            logging.error("[ERROR][模型识别过程运行错误]")
            logging.error("[ERROR][错误信息]" + str(e) + "\n")
            continue
        

    # [识别效果总结]
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if update:
        strip_optimizer(car_weights[0])  # update model (to fix SourceChangeWarning)

# [YOLOv5识别网络主程序]<<<<<<<


def main(args=None):
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(filename=os.getcwd()+'/logs/py.log', level=logging.DEBUG, format=LOG_FORMAT)

    rclpy.init(args=args) # 初始化rclpy

    # [提示信息界面]>
    logging.info("[正在启动神经网络识别节点]")
    logging.info("[初始化rclpy]")
    
    node = PointsPublisher("points_publisher")  # 新建一个节点

    # image_subscriber = ImageSubscriber("image_subscriber")
    # rclpy.spin(image_subscriber)

    logging.info("[初始化完成]")
    logging.info("[节点启动]")
    logging.info("[正在启动YOLOv5]")

    try:
        run(node,
            car_weights=os.getcwd()+'/user_models/car_identfy.pt',
            armor_weights=os.getcwd()+'/user_models/armor_identfy.pt',
            source=os.getcwd()+'/videos/2.mp4',
            view_img=True,
            identfy_car=True)
    except Exception as e:
        logging.fatal("[运行错误，请检查文件路径是否正确或环境是否存在问题]")
        logging.fatal("[错误信息]" + str(e))

    # [保持节点运行，检测是否收到退出指令(Ctrl+C)]
    rclpy.spin(node)
    rclpy.shutdown()