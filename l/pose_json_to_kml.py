import json
import simplekml.kml
import random
from collections import defaultdict
from bluenotelib.common.coordinate_transforms import CoordinateTransforms as ct
from bluenotelib.kml.kml import KmlGenerator
from image_pose_adjustment.utils import Utils
import csv


def read_csv(filename):
    data = []
    header = ''
    with open(filename, 'r') as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            if header is None:
                print row
                header = row
            else:
                data.append(row)
    print 'Total Truth', len(data)
    return data


class PlotPoseFile():
    def __init__(self):
        self.bundle_point_styles = defaultdict(lambda: None)
        self.bundle_point_styles[0] = PlotPoseFile.make_point_style(simplekml.Color.red)
        random.seed(240273)

    def plotPoseFiles(self, json_file_list, output_filename):
        kml_file = simplekml.Kml()

        min_lon = None
        min_lat = None
        max_lon = None
        max_lat = None
        total_poses = 0
        total_poses_FC = 0
        for json_file in json_file_list:
            poses = read_csv(json_file)
            for pose in poses:
                total_poses += 1

                total_poses_FC += 1

                lon = float(pose[6])
                lat = float(pose[5])
                alt = float(pose[7])

                image_name = "" # "{0}_{1}_{2}".format(pose["RunId"], pose["CameraId"], pose["SequenceNumber"])

                #if "BundleId" in pose:
                #    bundle_id = pose["BundleId"]
                #else:
                bundle_id = 0

                kml_pt = kml_file.newpoint(description=image_name, coords=[(lon, lat)])
                kml_pt.style = self.bundle_point_style(bundle_id)

                #if show_headings:
                #    quat = pose['StartQuaternion']
                #    r, p, h = ct.camera_quat_to_rph(lon, lat, alt, quat['W'], quat['X'], quat['Y'], quat['Z'])
                #    KmlGenerator.create_heading_line(kml_file, simplekml.Color.green, lon, lat, alt, h, arrow_length=5)
                #    print('{},{},{},{},{}'.format(pose['RunId'], lon, lat, h))

        kml_file.save(output_filename)
        #print 'polygon',min_lon,min_lat,max_lon,max_lat
        #print 'POLYGON ({} {}, {} {}, {} {}, {} {}, {} {}) \nfor https://arthur-e.github.io/Wicket/sandbox-gmaps3.html'.format(min_lon,min_lat,min_lon,max_lat,max_lon,max_lat,max_lon,min_lat,min_lon,min_lat )
        print 'Total poses', total_poses, total_poses_FC

    def bundle_point_style(self, bundle_id):
        style = self.bundle_point_styles.get(bundle_id)
        if style is None:
            color = simplekml.Color.rgb(random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
            style = PlotPoseFile.make_point_style(color)
            self.bundle_point_styles[bundle_id] = style

        return style

    @staticmethod
    def make_point_style(color):
        kml_pt_style = simplekml.Style()
        kml_pt_style.labelstyle.scale = 1
        kml_pt_style.labelstyle.color = color
        kml_pt_style.iconstyle.color = color
        kml_pt_style.iconstyle.scale = 0.25
        kml_pt_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pal2/icon26.png'
        return kml_pt_style


def main():

    import sys
    rgo = PlotPoseFile()
    output = sys.argv[2]
    inputs = [sys.argv[1]]#args.input.split(',')
    rgo.plotPoseFiles(inputs, output)



if __name__ == "__main__":
    main()
