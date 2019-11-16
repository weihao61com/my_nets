import sys
import glob
import os
from sortedcontainers import SortedDict
import cv2 as cv
from CV_text_detection import decode
import wx
import numpy as np

def GetBitmap(array ):
    height, width, _ = array.shape
    a = array.copy()
    a[:, :, 0] = array[:, :, 2]
    a[:, :, 2] = array[:, :, 0]
    image = wx.EmptyImage(width, height)
    image.SetData( a.tostring())
    # wxBitmap = image.ConvertToBitmap()       # OR:  wx.BitmapFromImage(image)
    return image

def get_files(location):
    output = []
    for dirs in glob.glob(os.path.join(location, '*')):
        o = SortedDict()
        for f in glob.glob(os.path.join(dirs, "*")):
            if f[-3:].endswith('jpg'):
                sz = os.path.getsize(f)
                o[-sz] = f
        output.append((dirs, o))
    return output


class Detection:

    def __init__(self, location='/home/weihao/tmp/ps'):

        # Read and store arguments
        self.confThreshold = 0.5  # args.thr
        self.nmsThreshold = 0.4  # args.nms
        detect_model = None
        classification_model = None
        # inpWidth = args.width
        # inpHeight = args.height
        model_path = '/media/weihao/DISK0/Object_detection'
        # model ="C:\\GITHUB\\learnopencv-master\\TextDetectionEAST\\frozen_east_text_detection.pb"

        self.model = os.path.join(model_path, "frozen_east_text_detection.pb")
        # print(confThreshold, nmsThreshold, model)

        # Create a new named window
        # kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
        # cv.namedWindow(kWinName)  # , cv.WINDOW_NORMAL)
        self.outputLayers = []
        self.outputLayers.append("feature_fusion/Conv_7/Sigmoid")
        self.outputLayers.append("feature_fusion/concat_3")

        config = ('-l eng --oem 1 --psm 3')

        self.files = get_files(location)
        print('total dir', len(self.files))
        self.idx = 0
        self.idx_photo = 0
        self.idx_box = 0
        self.current_list = []
        self.boxes = []
        self.frame = []

    def get_next_person(self):
        if self.idx == len(self.files):
            return None

        _, self.current_list = self.files[self.idx]
        self.idx += 1
        self.idx_photo = 0
        return self.get_next_photo()

    def set_value(self, value):
        A = self.files[self.idx-1]
        print('Label', A[0], value)


    def get_next_photo(self):
        if self.idx_photo==len(self.current_list):
            return self.get_next_person()

        key = self.current_list.keys()[self.idx_photo]
        f = self.current_list[key]
        self.idx_photo += 1

        max_width = 200

        cap = cv.VideoCapture(f)
        hasFrame, frame = cap.read()
        self.frame = frame
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
        net = cv.dnn.readNet(self.model)
        # Run the model
        net.setInput(blob)
        output = net.forward(self.outputLayers)

        # Get scores and geometry
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, self.confThreshold)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        self.boxes = []

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self.confThreshold, self.nmsThreshold)
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
            xmin = int(xmin) - 10
            ymin = int(ymin) - 10
            xmax = int(xmax) + 10
            ymax = int(ymax) + 10
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0

            ig = frame[ymin:ymax, xmin:xmax, :]
            if ig.shape[0] == 0 or ig.shape[1] == 0:
                continue
            self.boxes.append(ig)

        self.idx_box = 0
        return self.get_next_box()

    def get_next_box(self):
        if self.idx_box==len(self.boxes):
            return self.get_next_photo()

        self.idx_box += 1
        return self.frame, self.boxes[self.idx_box-1]


class PhotoCtrl(wx.App):

    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.frame = wx.Frame(None, title='Photo Control')

        self.panel = wx.Panel(self.frame)

        self.PhotoMaxSize = 512

        self.createWidgets()
        self.data = Detection()

        self.frame.Show()

        self.set_next_person()

    def set_next_person(self):
        A, B = self.data.get_next_person()
        self.add_image(A, self.imageCtrl, 1)
        self.add_image(B, self.imageCtrl1, 2)

        self.panel.Refresh()

    def add_image(self, B, imageCtrl, scale):
        A = GetBitmap(B)

        W = A.GetWidth()
        H = A.GetHeight()
        if W > H:
            NewW = self.PhotoMaxSize
            NewH = self.PhotoMaxSize * H / W
        else:
            NewH = self.PhotoMaxSize
            NewW = self.PhotoMaxSize * W / H
        img = A.Scale(NewW, NewH/scale)

        imageCtrl.SetBitmap(wx.BitmapFromImage(img))

    def save_and_next(self, A):
        value = self.photoTxt.GetValue()
        self.photoTxt.SetValue("")
        self.data.set_value(value)
        self.set_next_person()


    def createWidgets(self):
        instructions = 'Browse for an image'
        img = wx.EmptyImage(self.PhotoMaxSize, self.PhotoMaxSize)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                         wx.BitmapFromImage(img))
        img1 = wx.EmptyImage(self.PhotoMaxSize, self.PhotoMaxSize/2)
        self.imageCtrl1 = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                          wx.BitmapFromImage(img1))

        # instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, style=wx.TE_PROCESS_ENTER, size=(200, -1))

        self.photoTxt.Bind(wx.EVT_TEXT_ENTER, self.save_and_next)

        browseBtn = wx.Button(self.panel, label='Next image')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)

        # self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
        #                   0, wx.ALL | wx.EXPAND, 5)
        # self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.imageCtrl1, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)

        self.panel.Layout()

    def onBrowse(self, event):
        A, B = self.data.get_next_box()
        self.add_image(A, self.imageCtrl, 1)
        self.add_image(B, self.imageCtrl1, 2)
        self.panel.Refresh()


if __name__ == '__main__':
    app = PhotoCtrl()
    app.MainLoop()
