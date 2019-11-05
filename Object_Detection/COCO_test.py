from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import matplotlib 
matplotlib.use('TkAgg')

annFile='/media/weihao/DISK0/Object_detection/COCO/COCO_Text.json'
annFile='/media/weihao/DISK0/Object_detection/COCO/annotations/instances_train2017.json'

coco=COCO(annFile)


# display COCO categories and supercategories# displ
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
#

catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds )
print(len(imgIds))
print(imgIds)

imgIds = coco.getImgIds(imgIds = [438915])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# load and display image# load
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
#plt.axis('off')
#plt.imshow(I)
#plt.show()


plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

#cv2.imshow("foo",I)
# cv2.waitKey(0)
annFile='/media/weihao/DISK0/Object_detection/COCO/annotations/person_keypoints_train2017.json'
coco_kps=COCO(annFile)

# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)

annFile='/media/weihao/DISK0/Object_detection/COCO/annotations/captions_train2017.json'
coco_caps=COCO(annFile)


# load and display caption annotations# load a
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()
