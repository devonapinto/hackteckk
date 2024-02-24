import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
import pygame
import psycopg2

conn = psycopg2.connect(
    dbname="eyes",
    user="postgres",
    password="Maqwin@95",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

def cat():
    def grey_scale(rgb_image):
        grey_Image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('grey',grey_Image)
        #cv2.waitKey(0)
        return grey_Image

    def red_scale(rgb_image):
        red_image = rgb_image.copy()
        red_image[:, :, 1] = 0
        red_image[:, :, 0] = 0
        #cv2.imshow('red image', red_image)
        #cv2.waitKey(0)
        return red_image

    def k_means(grey_img):

        img2 = grey_img  # .cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = img2.reshape(-1, 1)
        # k-meins algorithum
        kmeans = KMeans(n_clusters=6, n_init=20)
        kmeans.fit(x)
        segim = kmeans.cluster_centers_[kmeans.labels_]
        segim = segim.reshape(img2.shape)
        segim = segim / 255
        segim_2d = (segim * 255).astype(np.uint8)
        segim_2d_reshaped = segim_2d.reshape((250, 250, 1))
        blur = cv2.GaussianBlur(grey, (11, 11), 0)
        mask = cv2.equalizeHist(blur)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(segim_2d_reshaped)
        #cv2.imshow('k_means',segim_2d_reshaped)
        #cv2.waitKey(0)
        return segim_2d_reshaped,maxVal

    def morph(k_means_img):
        kernal = np.ones((3, 3), np.uint8)
        mask = cv2.erode(k_means_img, kernal, iterations=1)
        mask = cv2.dilate(mask, kernal, iterations=3)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
        #cv2.imshow('morph',mask)
        #cv2.waitKey(0)
        return mask,maxVal
    def circle(morph_img,rgb_img,max_val):

        #cv2.imshow('morphed image',morph_img)
        #cv2.waitKey(0)
        thresh = cv2.threshold(morph_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours and filter using contour area and aspect ratio
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        circle_count=0
        min_area=1000
        max_area=0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) > 1 and area > 1000 and area < 20000:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(rgb_img, (int(x), int(y)), int(r), (36, 255, 12), 2)
                circle_count+=1
                if area<min_area:
                    min_area=area
                if area>max_area:
                    max_area=area
        '''print("circle count = ",circle_count)
        print("min area",min_area)
        print("max area",max_area)
        cv2.imshow('thresh', thresh)
        cv2.imshow('opening', opening)
        cv2.imshow('image', rgb_image)
        cv2.waitKey()'''
        print("apple=",circle_count)
        if circle_count == 0:
            cursor.execute("INSERT INTO result (disorders,final) VALUES ('ct','false');")
            conn.commit()
        else:
            cursor.execute("INSERT INTO result (disorders,final) VALUES ('ct','true');")
            conn.commit()
        cursor.close()
        conn.close()

    rgb_image = cv2.imread("C:\\Users\\maqwi\\Desktop\\hack\\archive (1)\processed_images\\train\\cataract\\image_4.png")
    rgb_image = cv2.resize(rgb_image, (500, 500))
    y = 150
    x = 150
    h = 250
    w = 250
    rgb_image = rgb_image[y:y + h, x:x + w]
    #cv2.imshow('Image', rgb_image)
    #cv2.imshow('dr',rgb_image)
    #cv2.waitKey()
    green=red_scale(rgb_image)
    grey=grey_scale(green)
    grey = cv2.convertScaleAbs(grey, alpha=5, beta=20)
    k_means_img,k_max_val=k_means(grey)
    morph_img,max_val=morph(k_means_img)
    number_of_white_pix = np.sum(morph_img == max_val)
    number_of_black_pix = np.sum(morph_img != max_val)
    total=number_of_black_pix+number_of_white_pix
    print('Number of white pixels:', number_of_white_pix)
    print('Number of black pixels:', number_of_black_pix)
    print("percentage = ",((number_of_white_pix/total)*100))
    circle(k_means_img,rgb_image,k_max_val)
    save_path = "C:\\Users\\maqwi\\Desktop\\hack\\ct\\final.png"
    cv2.imwrite(save_path, rgb_image)
    pass
