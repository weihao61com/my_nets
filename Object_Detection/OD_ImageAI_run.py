from imageai.Detection.Custom import CustomObjectDetection
import glob
import os
import cv2
import shutil

def process(detector, image_in, image_out):
    img = cv2.imread(image_in)
    sz = img.shape
    s1 = sz[0]
    s2 = int(sz[1]/2)
    nt = 0
    for a in range(0, s1, s2):
        ig = img[a:a+s2*2, :, :]
        if ig.shape[0]>s2:
            cv2.imwrite('t.jpg', ig)
            out_name = '{}_{}.jpg'.format(image_out, nt)
            detections = detector.detectObjectsFromImage(input_image='t.jpg',
                                output_image_path='s.jpg',
                                minimum_percentage_probability=30)
            if len(detections)>0:
                print(image_in, out_name, a, ":")
                for detection in detections:
                    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
                shutil.copy('s.jpg', out_name)
            nt += 1

if __name__ == "__main__":

    HOME = '/home/weihao/Projects'

    model_dir = "C:\\tmp\\hololens"
    model = model_dir + "/models/" + "detection_model-ex-002--loss-0009.175" + ".h5"
    image_in = "/home/weihao/Projects/tmp/images"
    #
    # model_dir = "C:\\Projects\\hololens\\"
    # model = model_dir + "models\\" + "detection_model-ex-001--loss-0031.707.h5"
    image_in = "C:\\tmp\\validation\\images\\image (297).jpg"

    json = model_dir + "/json/detection_config.json"

    # model = "C:\\Users\\weiha\\Downloads\\hololens-ex-60--loss-2.76.h5"
    # json = "C:\\Users\\weiha\\Downloads\\detection_config.json"

    detector = CustomObjectDetection()
    detector.__input_size = 104
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model)
    detector.setJsonPath(json)
    detector.loadModel()

    # image_in = "C:\\Projects\\tmp\\test.jpg"
    # image_in = "C:\\Projects\\hololens\\validation\\images\\image (297).jpg"
    image_out = "C:\\tmp\image_out"

    if image_in.endswith('jpg'):
        process(detector, image_in, image_out + ".jpg")
    else:
        nt = 0
        for f in glob.glob(os.path.join(image_in, '*.jpg')):
            process(detector, f, "{}_{}".format(image_out, nt))
            nt += 1