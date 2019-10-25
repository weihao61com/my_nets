# Standard imports
import cv2
import pytesseract

def cal_overlap(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[0]+a[2], b[0]+b[2])
    y1 = min(a[1]+a[3], b[1]+b[3])
    if x1<x0 or y1<y0:
        return 0
    area = float(x1-x0)*(y1-y0)
    return area/a[2]/a[3]

# Read image
img = '/media/weihao/DISK0/Object_detection/learnopencv-master/TextDetectionEAST/stop2.jpg'
#img = '/media/weihao/DISK0/Object_detection/learnopencv-master/BlobDetector/blob.jpg'
img = '/home/weihao/Downloads/BolderBoulder-2002.jpg'
img = '/home/weihao/Downloads/IMG_1134.JPG'
img = cv2.imread(img)
kWinName = "blob Detection"
cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)

config = ("-l eng --oem 1 --psm 7")

mser = cv2.MSER_create(_min_area=1000, _max_area=4000)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
gray_img = img.copy()

regions = mser.detectRegions(gray)
print('Total region', len(regions[0]))
region0_list = []
region1_list = []
boxes = []
for a in range(len(regions[1])):
    overlap = False
    for l in region1_list:
        if cal_overlap(l, regions[1][a])>0.95:
            overlap = True
            break
    if not overlap:
        region0_list.append(regions[0][a])
        v = regions[1][a]
        region1_list.append(regions[1][a])
print('Total good region', len(region0_list))

f = 10
nt = 0
for r in region1_list:
    roi = gray_img[r[1]-f:r[1]+r[3]+f, r[0]-f:r[0]+r[2]+f]
    sz = roi.shape
    if sz[0]==0 or sz[1]==0:
        continue
    text = pytesseract.image_to_string(roi, config=config)
    cv2.imwrite('out_{}.jpg'.format(nt), roi)
    print(nt, r, text)
    nt += 1

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in region0_list]
cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
#cv2.imwrite('/home/mis/Text_Recognition/amit.jpg', gray_img)


# Show keypoints
cv2.imshow(kWinName, gray_img)
cv2.waitKey(0)
cv2.imwrite('out.jpg', gray_img)