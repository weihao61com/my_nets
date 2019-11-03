import os
import glob


location = "C:\\Projects\\tag\\train"

images = glob.glob(os.path.join(location, 'images', '*.jpg'))
annes = glob.glob(os.path.join(location, 'annotations', '*.xml'))

print(len(images), len(annes))