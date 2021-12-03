import cv2
from transform import four_point_transform

class Track:
    def __init__(self,image,shape:list):
        """takes image path 
        and the shape of the track as a list where the first element is the coordinates for a bbox for the track 
        and the second is a list of y coordinates for rows 
        and the third is a list of x coordinates for columns"""
        self.image = cv2.imread(image)
        self.bbox, self.rows, self.columns = shape
        self.map = None

    def find_map(self):
        return four_point_transform(self.image, self.bbox)