import cv2
import numpy as np
from dataParser import parser
from transform import four_point_transform

transform_points = np.array([(222, 222), (124, 76), (0, 222), (94, 78)])

data = parser("element_new/classes.txt")

images = []
classes = []

orb = cv2.ORB_create(nfeatures = 3000)

for image_path, image_class in data:
    image = cv2.imread("element_new/" + image_path, cv2.IMREAD_GRAYSCALE)    
    scale_percent = 10 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    image = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)

    _, image = cv2.threshold(image, 150, 250, cv2.THRESH_BINARY)

    images.append(image)
    classes.append(image_class)

def get_des(images:list):
    kp_list = []
    des_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        des_list.append(des)
        kp_list.append(kp)

    return des_list, kp_list

def get_class(img,desList):
    kp, frame_des = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    match_list = []
    match_lengths = []
    for des in des_list:
        matches = bf.knnMatch(des, frame_des, k=2)
        good = []
        for m,n in matches:
            if m.distance <0.75 * n.distance:
                good.append([m])
        match_list.append(good)
        match_lengths.append(len(good))
    best_match = max(match_lengths)
    match_index = match_lengths.index(best_match)
    return classes[match_index], kp, frame_des, match_index, match_list[match_index]


des_list, kp_list = get_des(images)


records = parser("records/record_classify.txt")

for record_path, record_class in records:
    record_img = cv2.imread("records/" + record_path, cv2.IMREAD_GRAYSCALE)
    record_img = four_point_transform(record_img,transform_points)
    predection, kp2, des2, index, matches = get_class(record_img, des_list)
    kp = kp_list[index]
    des = des_list[index]

    match_img = cv2.drawMatchesKnn(images[index], kp, record_img, kp2, matches, None, flags=2)


    #print(record_class, predection)
    cv2.imshow("match", match_img)



    _, threshold = cv2.threshold(record_img, 120, 250, cv2.THRESH_BINARY)
    predection, kp2, des2, index, matches = get_class(threshold, des_list)
    kp = kp_list[index]
    des = des_list[index]

    match_img = cv2.drawMatchesKnn(images[index], kp, threshold, kp2, matches, None, flags=2)


    print(record_class, predection)
    cv2.imshow("threshhold_match", match_img)
    cv2.waitKey(0)


