from imageai.Detection.Custom import CustomObjectDetection
import glob
import os
import cv2


def process(detector, image_in, image_out):
    detections = detector.detectObjectsFromImage(input_image=image_in,
                                                 output_image_path=image_out,
                                                 minimum_percentage_probability=30)
    print(image_in, ":")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


if __name__ == "__main__":
    model_dir = "C:\\Projects\\tag\\"
    model = model_dir + "models\\" + "detection_model-ex-004--loss-0016.161.h5"
    image_in = "C:\\Projects\\tag\\test\\WIN_20190912_14_27_52_Pro.jpg"
    #
    # model_dir = "C:\\Projects\\hololens\\"
    # model = model_dir + "models\\" + "detection_model-ex-001--loss-0031.707.h5"
    # image_in = "C:\\Projects\\hololens\\validation\\images\\image (297).jpg"

    json = model_dir + "json\\detection_config.json"

    # model = "C:\\Users\\weiha\\Downloads\\hololens-ex-60--loss-2.76.h5"
    # json = "C:\\Users\\weiha\\Downloads\\detection_config.json"

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model)
    detector.setJsonPath(json)
    detector.loadModel()

    # image_in = "C:\\Projects\\tmp\\test.jpg"
    # image_in = "C:\\Projects\\hololens\\validation\\images\\image (297).jpg"
    image_out = "C:\\Projects\\tmp\\image_out"

    if image_in.endswith('jpg'):
        process(detector, image_in, image_out + ".jpg")
    else:
        nt = 0
        for f in glob.glob(os.path.join(image_in, '*.jpg')):
            process(f, "{}_{}.jpg".format(image_out, nt))
            nt += 1