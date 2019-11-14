import xml.etree.ElementTree as ET
import glob
import os

location = '/media/weihao/DISK0/flickr_images/square/train'

for f in glob.glob(os.path.join(location, 'annotations_org', '*.xml')):

    tree = ET.parse(f)
    root = tree.getroot()
    #for o in tree.find('object'):
    #    print(o.tag, o.text)
    #    if o.tag=='name':
    #        o.text = 't'
    # tree.find('.//object').text = 't'

    for child in root:
        if child.tag=='object':
            for c in child:
                if c.tag=='name':
                    #print(c.tag, c.text)
                    c.text = 't'
                if c.tag == 'bndbox':
                    for x in c:
                        if x.tag=='xmin':
                            xmin = int(x.text)
                        if x.tag=='xmax':
                            xmax = int(x.text)
                        if x.tag == 'ymin':
                            ymin = int(x.text)
                        if x.tag == 'ymax':
                            ymax = int(x.text)
                    dx = xmax-xmin
                    dy = ymax-ymin
                    dx = int(dx*0.1)
                    dy = int(dy*0.1)
                    xmin -= dx
                    xmax += dx
                    ymin -= dy
                    ymax += dy
                    for x in c:
                        if x.tag=='xmin':
                            x.text = '{}'.format(xmin)
                        if x.tag=='xmax':
                            x.text = '{}'.format(xmax)
                        if x.tag == 'ymin':
                            x.text = '{}'.format(ymin)
                        if x.tag == 'ymax':
                            x.text = '{}'.format(ymax)

    # for child in root:
    #     if child.tag=='object':
    #         for c in child:
    #             if c.tag=='name':
    #                 print(c.tag, c.text)
                    #c.tag = 't'

    basename = os.path.basename(f)
    filename = os.path.join(location, 'annotations', basename)
    tree.write(filename)