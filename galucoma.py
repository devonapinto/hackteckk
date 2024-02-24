import cv2
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
import pygame
import psycopg2
# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="eyes",
    user="postgres",
    password="Maqwin@95",
    host="localhost",
    port="5432"
)


cursor = conn.cursor()
def glaucoma():

    def redimage(rgb_image):
        rgb_image = cv2.resize(rgb_image, (500, 500))
        red_image = rgb_image.copy()
        red_image[:, :, 0] = 0
        red_image[:, :, 1] = 0
        #cv2.imshow('yooo',red_image)
        #cv2.waitKey(0)
        return red_image

    def grey_img(green_img):
        img = cv2.cvtColor(green_img, cv2.COLOR_BGR2GRAY)
        new_image = cv2.convertScaleAbs(img, alpha=2.5, beta=10)
        return new_image

    class disk:
        def __init__(self):
            pass

        def morphing(self, grey):
            kernal = np.ones((6, 6), np.uint8)
            mask = cv2.erode(grey, kernal, iterations=5)
            mask = cv2.dilate(mask, kernal, iterations=5)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            return mask

        def kemins(self, img, grey_imgg):
            def veins(img):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
                erosion = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
                erosion = cv2.erode(erosion, kernel, iterations=1)
            veins(img)
            img2 = grey_imgg  # .cvtColor(img, cv2.COLOR_BGR2GRAY)
            x = img2.reshape(-1, 1)

            # croped image
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img2)
            crop_mask = np.zeros(img2.shape[:2], dtype="uint8")
            cv2.circle(crop_mask, maxLoc, 120, (255, 255, 0), -1)
            masked_crooop = cv2.bitwise_and(img2, img2, mask=crop_mask)

            # k-meins algorithum
            kmeans = KMeans(n_clusters=5, n_init=10)
            kmeans.fit(x)
            segim = kmeans.cluster_centers_[kmeans.labels_]
            segim = segim.reshape(masked_crooop.shape)
            segim = segim / 255
            segim_2d = (segim * 255).astype(np.uint8)
            segim_2d_reshaped = segim_2d.reshape((500, 500, 1))
            blur = cv2.GaussianBlur(grey, (11, 11), 0)
            mask = cv2.equalizeHist(blur)
            edge = cv2.Canny(mask, 10, 1)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(segim_2d_reshaped)

            masked_img = cv2.inRange(segim_2d_reshaped, maxVal, 255)
            masked_crooop = cv2.bitwise_and(masked_img, masked_img, mask=crop_mask)
            return masked_crooop,segim_2d_reshaped

        def centroid_calc(self, morph_img):
            # centroid calculation
            try:
                imageMoments = cv2.moments(morph_img)
                cx = int(imageMoments['m10'] / imageMoments['m00'])
                cy = int(imageMoments['m01'] / imageMoments['m00'])
                bgrImage = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)
                f = open("gl\\gl.txt", "w")
                f.write("True")
                f.close()
                return cx, cy
            except Exception as e:
                print("does not have gloaucoma")
                cursor.execute("INSERT INTO result (disorders,final) VALUES ('gl','false');")
                conn.commit()
                f=open("gl//gl.txt","w")
                f.write("False")
                f.close()
                return 0,0

        def vertical_diamter_cal(self, morph_img, cx, cy):
            x1 = cx
            y1 = cy
            x2 = cx
            y2 = cx
            while True:
                temp_image_val = morph_img[y1, x1]
                if temp_image_val < 30:
                    break
                y1 += 1
            while True:
                temp_image_val = morph_img[y2, x2]
                if temp_image_val < 30:
                    break
                y2 -= 1
            diamter = y1 - y2
            radius = int(diamter / 2)
            morph_img = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)
            cv2.circle(morph_img, (cx, cy), radius, (255, 255, 0), 1)

            return radius


    class cup:
        def __init__(self):
            pass

        def green(self,rgb_image):
            rgb_image = cv2.resize(rgb_image, (500, 500))
            green_image = rgb_image.copy()
            green_image[:, :, 2] = 0
            green_image[:, :, 0] = 0
            return green_image

        def region_calc(self,red_img,grey,cx,cy,disk_radius):
            crop_mask = np.zeros(grey.shape[:2], dtype="uint8")
            cv2.circle(crop_mask, (cx,cy), disk_radius, (255, 255, 0), -1)
            masked_crooop = cv2.bitwise_and(red_img, red_img, mask=crop_mask)
            grey_masked_crooop = cv2.bitwise_and(grey, grey, mask=crop_mask)
            return masked_crooop,grey_masked_crooop

        def cup_kemin(self,img,grey_img):
            def veins(img):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
                erosion = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
                erosion = cv2.erode(erosion, kernel, iterations=1)
            veins(img)
            img2 = grey_img  # .cvtColor(img, cv2.COLOR_BGR2GRAY)
            x = img2.reshape(-1, 1)
            # k-meins algorithum
            kmeans = KMeans(n_clusters=6, n_init=10)
            kmeans.fit(x)
            segim = kmeans.cluster_centers_[kmeans.labels_]
            segim = segim.reshape(img2.shape)
            segim = segim / 255
            segim_2d = (segim * 255).astype(np.uint8)
            segim_2d_reshaped = segim_2d.reshape((500, 500, 1))
            blur = cv2.GaussianBlur(grey, (11, 11), 0)
            mask = cv2.equalizeHist(blur)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(segim_2d_reshaped)

            crop_mask = np.zeros(img2.shape[:2], dtype="uint8")
            masked_img = cv2.inRange(segim_2d_reshaped, maxVal, 255)
            cv2.circle(crop_mask, maxLoc, 120, (255, 255, 0), -1)
            masked_crooop = cv2.bitwise_and(masked_img, masked_img, mask=crop_mask)
            return masked_crooop

        def cup_morphing(self,cup_keimen_image):
            kernal = np.ones((6, 6), np.uint8)
            mask = cv2.erode(cup_keimen_image, kernal, iterations=1)
            mask = cv2.dilate(mask, kernal, iterations=3)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            return mask

        def cup_centroid_calc(self,cup_morph_img):
            imageMoments = cv2.moments(cup_morph_img)
            cx = int(imageMoments['m10'] / imageMoments['m00'])
            cy = int(imageMoments['m01'] / imageMoments['m00'])
            bgrImage = cv2.cvtColor(cup_morph_img, cv2.COLOR_GRAY2BGR)
            return cx, cy

        def cup_vertical_diameter(self, cup_morph_img, cx, cy):
            x1 = cx
            y1 = cy
            x2 = cx
            y2 = cx
            while True:
                temp_image_val = cup_morph_img[y1, x1]
                if temp_image_val < 20:
                    break
                y1 += 1
            while True:

                temp_image_val = cup_morph_img[y2, x2]
                if temp_image_val < 20:
                    break
                y2 -= 1
            diamter = y1 - y2
            radius = int(diamter / 2)
            morph_img = cv2.cvtColor(cup_morph_img, cv2.COLOR_GRAY2BGR)
            cv2.circle(morph_img, (cx, cy), radius, (255, 255, 0), 1)
            return radius

    rgb_image = cv2.imread("C:\\Users\\maqwi\\Desktop\\hack\\archive (1)\processed_images\\train\\cataract\\image_4.png")

    rgb_image = cv2.resize(rgb_image, (500, 500))
    org2=rgb_image
    org_img=rgb_image
    red_img = redimage(rgb_image)
    grey = grey_img(red_img)
    #cv2.imshow("orignal image",org2)
    #cv2.imshow("orignal image",grey)
    #cv2.waitKey(0)

    # disk calculation
    d = disk()
    kemin_img_mask,kemmins_img = d.kemins(red_img, grey)
    morph_img = d.morphing(kemin_img_mask)
    cx, cy = d.centroid_calc(morph_img)
    f = open("gl\\gl.txt", "r")
    val=f.readline()
    f.close()
    if val=='True':
        print("yeepppp")
        disk_radius = d.vertical_diamter_cal(morph_img, cx, cy)

        # cup calculation
        c = cup()
        green_img=c.green(rgb_image)
        region_img,grey_cup_kemin_img=c.region_calc(green_img,grey,cx,cy,disk_radius)
        cup_kemin_img=c.cup_kemin(region_img,grey_cup_kemin_img)
        cup_morph_img=c.cup_morphing(cup_kemin_img)
        c2x,c2y=c.cup_centroid_calc(cup_morph_img)
        cup_radius=c.cup_vertical_diameter(cup_morph_img,c2x,c2y)

        cv2.circle(rgb_image, (cx, cy), disk_radius, (255, 255, 0), 1)
        cv2.circle(rgb_image, (c2x, c2y), cup_radius, (0, 255, 5), 1)

        print("DISK VERTICAL DIAMETER : ",disk_radius*2)
        print("CUP VERTICAL DIAMETER : ",cup_radius*2)
        ratio=((cup_radius*2)/(disk_radius*2))
        print("Disk/CUP RATIO  :",ratio)

        '''
        cv2.imshow("disk_kemins image",kemmins_img)
        cv2.imshow("disk_kemins mask image",kemin_img_mask)
        cv2.imshow("disk morphed image",morph_img)

        cv2.imshow("cup kemins image",cup_kemin_img)
        cv2.imshow("cup morphed image",cup_morph_img)
        cv2.imshow("final result",rgb_image)

        cv2.waitKey(0)'''

        def final_result():
            print("there is glaucoma")
            cursor.execute("INSERT INTO result (disorders,final) VALUES ('gl','true');")
            conn.commit()
            save_path = "C:\\Users\\maqwi\\Desktop\\hack\\gl\\final.png"
            cv2.imwrite(save_path, rgb_image)
            f=open("gl\\gl.txt","w")
            f.write(str(disk_radius*2)+"\n")
            f.write(str(cup_radius*2)+"\n")
            f.write(str(ratio)+"\n")
            f.write("Pressence of Glaucoma detected")
        final_result()

    cursor.close()
    conn.close()

