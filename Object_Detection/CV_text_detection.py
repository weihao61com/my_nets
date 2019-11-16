# Import required modules
import cv2 as cv
import math
import sys
import glob
import os
import pytesseract
import shutil

sys.path.append("/media/weihao/DISK0/GITHUB/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN-master")  # To find local version
from combi_models import find_box_and_predict_digit


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


if __name__ == "__main__":
    # Read and store arguments
    confThreshold = 0.5  # args.thr
    nmsThreshold = 0.4  # args.nms
    detect_model = None
    classification_model = None
    # inpWidth = args.width
    # inpHeight = args.height
    model_path = '/media/weihao/DISK0/Object_detection'
    # model ="C:\\GITHUB\\learnopencv-master\\TextDetectionEAST\\frozen_east_text_detection.pb"

    model = os.path.join(model_path, "frozen_east_text_detection.pb")
    print(confThreshold, nmsThreshold, model)

    # Create a new named window
    #kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    #cv.namedWindow(kWinName)  # , cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    config = ('-l eng --oem 1 --psm 3')

    # Open a video file or an image file or a camera stream
    # cap = cv.VideoCapture(sys.argv[1])
    location = '/home/weihao/tmp/ps/1'  #
    if len(sys.argv)>1:
        location = sys.argv[1]
    #  sys.argv[1]
    # location = '/media/weihao/DISK0/flickr_images/persons'  # sys.argv[1]
    # if len(sys.argv) > 2:
    #    scale = float(sys.argv[2])
    #location = "/media/weihao/DISK0/flickr_images/testing_persons"

    max_width = 200
    out_path = os.path.join(location, 'texts')
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path
              )
    for filename in glob.glob(os.path.join(location, '*')):
        if filename[-3:] in ['jpg', 'JPG']:
            # frame = cv.imread(filename)
            print(filename)
            cap = cv.VideoCapture(filename)
            hasFrame, frame = cap.read()

            basename = os.path.basename(filename)
            scale = 1.0

            # Get frame height and width
            height_ = frame.shape[0]
            width_ = frame.shape[1]

            if width_ > max_width:
                scale = max_width / width_

            inpWidth = int(width_ / 32 * scale) * 32
            inpHeight = int(height_ / 32 * scale) * 32

            rW = width_ / float(inpWidth)
            rH = height_ / float(inpHeight)

            # Create a 4D blob from frame.
            blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

            # Load network
            net = cv.dnn.readNet(model)
            # Run the model
            net.setInput(blob)
            output = net.forward(outputLayers)

            # Get scores and geometry
            scores = output[0]
            geometry = output[1]
            [boxes, confidences] = decode(scores, geometry, confThreshold)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

            # Apply NMS
            indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
            for i in indices:
                # get 4 corners of the rotated rect
                vertices = cv.boxPoints(boxes[i[0]])
                # scale the bounding box coordinates based on the respective ratios
                xmin = 1e6
                ymin = 1e6
                xmax = -1
                ymax = -1
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                    if vertices[j][0] < xmin:
                        xmin = vertices[j][0]
                    if vertices[j][0] > xmax:
                        xmax = vertices[j][0]
                    if vertices[j][1] < ymin:
                        ymin = vertices[j][1]
                    if vertices[j][1] > ymax:
                        ymax = vertices[j][1]
                xmin = int(xmin)# - 10
                ymin = int(ymin)# - 10
                xmax = int(xmax)# + 10
                ymax = int(ymax)# + 10
                print(xmin, xmax, ymin, ymax)
                if xmin<0:
                    xmin = 0
                if ymin<0:
                    ymin = 0
                ig = frame[ymin:ymax, xmin:xmax, :]
                if ig.shape[0]==0 or ig.shape[1]==0:
                    continue
                text = pytesseract.image_to_string(ig, config=config)
                val,detect_model, classification_model\
                    = find_box_and_predict_digit(ig, detect_model, classification_model)

                print('Detection {}, {}, {}'.format(i[0], text, val))
                #cv.imshow(kWinName, ig)
                # cv.waitKey()
                text = val
                if len(text)==0:
                    text = 'None'
                print(basename, ':', height_, width_, inpHeight, inpWidth, label, text)

                for j in range(4):
                    p1 = (vertices[j][0], vertices[j][1])
                    p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                    cv.line(frame, p1, p2, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, "{}".format(text), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

            # Put efficiency information
            # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


            # Display the frame
            # cv.imshow(kWinName,frame)
            cv.imwrite(os.path.join(location, 'texts', basename), frame)
