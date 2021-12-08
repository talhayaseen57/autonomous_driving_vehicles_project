import cv2
import numpy as np
from transform import four_point_transform
from gradient import generate_gradient


class Track:

    def __init__(self, image, bbox, shape: list):
        """takes image of a track,
        an array of coordinates for a bounding box,
        and the shape of the track"""

        self.image = image
        self.rows, self.columns = shape
        self.shape = shape
        self.bbox = bbox
        self.map = self.find_map()
        self.regions = self.find_regions()
        self.types = [None]*len(self.regions)

    def find_map(self):
        """returns a birds eye view of the map"""

        return four_point_transform(self.image, self.bbox, self.shape)

    def find_regions(self):
        """returns a list of bounding boxes for each region in the map"""

        dims = self.map.shape[:2]
        y_increment = x_increment = 100

        y = 0
        points = np.empty((dims[0]+1, dims[1]+1), dtype=object)
        regions = []
        while True:

            x = 0
            while True:

                if x <= dims[0]:
                    cv2.circle(self.map, (x, y), 3, (255, 0, 0), -1)
                    points[y//y_increment][x//x_increment] = (x, y)
                    x += x_increment
                else:
                    break
            if y < dims[0]:
                y += y_increment
            else:
                break

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                regions.append(points[i:i+2, j:j+2])

        return regions

    def set_type(self, type,  i):
        """takes the element type and an index (i) for the region,
        and sets the element type for that region of the map"""

        self.types[i] = type

    def generate_heat_map(self, data):
        """takes data dict and draws a heatmap"""

        gradiant = generate_gradient("green", "red")
        heat_map = np.copy(self.map)

        for i, region in enumerate(self.regions):
            error_rate = data[self.types[i]][2]
            color = gradiant[round(error_rate*len(gradiant))]
            c = (int(color.blue*255), int(color.green*255), int(color.red*255))
            cv2.rectangle(heat_map, region[0][0], region[1][1], c, -1)
        return heat_map
