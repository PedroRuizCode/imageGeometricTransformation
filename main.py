'''         Image processing and computer vision
              Alejandra Avendaño y Pedro Ruiz
               Electronic engineering students
              Pontificia Universidad Javeriana
                      Bogotá - 2020
'''
import cv2 #import openCV library
import numpy as np #import numpy library
import sys #import sys library
import os #import os library

def click(event, x, y, flags, param):
    global i, pos #global variables
    if event == cv2.EVENT_LBUTTONDOWN: #left button pressed
        pos[i, 0] = x #save x position
        pos[i, 1] = y #save y position
        i = i + 1 #Add 1 to i
i = 0 #Create global varible
pos = np.zeros((3, 2), dtype=int) #Create global varible

if __name__ == '__main__':
    path = sys.argv[1]#Path of images
    image1_name = sys.argv[2] #name of the first image
    image2_name = sys.argv[3] #name of the second image
    path_file1 = os.path.join(path, image1_name)
    path_file2 = os.path.join(path, image2_name)
    image1 = cv2.imread(path_file1) #Upload the image
    image2 = cv2.imread(path_file2) #Upload the image
    while(1):
        cv2.imshow('Image', image1)#Show the image
        cv2.setMouseCallback('Image', click)#Read mouse status
        cv2.waitKey(1)
        if i == 3: #Point counter
            pts1_I1 = pos
            break
    cv2.destroyAllWindows()
    i = 0
    pos = np.zeros((3, 2), dtype=int)
    while (1):
        cv2.imshow('Image', image2)#Show the image
        cv2.setMouseCallback('Image', click)#Read mouse status
        cv2.waitKey(1)
        if i == 3: #Point counter
            pts2_I2 = pos
            break
    cv2.destroyAllWindows()
    pts1 = np.float32(pts1_I1)#Dot matrix of the first image
    pts2 = np.float32(pts2_I2)#Dot matrix of the second image
    M_affine = cv2.getAffineTransform(pts1, pts2) #Matrix H for affine transformation
    image_affine = cv2.warpAffine(image1, M_affine, image1.shape[:2])#Image transformation

    # scaling
    sx = np.sqrt(M_affine[0, 0] ** 2 + M_affine[1, 0] ** 2)#Calculation of sx
    sy = np.sqrt(M_affine[0, 1] ** 2 + M_affine[1, 1] ** 2)#Calculation of sy
    # rotation
    theta = -np.arctan(np.divide(M_affine[1, 0], M_affine[0, 0]))#Calculation of theta
    theta_rad = theta * np.pi / 180#Calculation of theta in rads
    # translation
    tx=np.divide(((M_affine[0, 2]*np.cos(theta_rad))-(M_affine[1, 2]*np.sin(theta_rad))),sx)#Calculation of tx
    ty=np.divide(((M_affine[0, 2]*np.sin(theta_rad))+(M_affine[1, 2]*np.cos(theta_rad))),sy)#Calculation of ty

    # similarity
    M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                        [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])# similarity matrix
    image_similarity = cv2.warpAffine(image1, M_sim, image1.shape[:2]) # similarity transformation
    i = 0
    A = [0, 0, 0]
    for i in range(3):
        A[i] = sum(abs(a - b) for a, b in zip(pts1[i, :], pts2[i, :]))#calculation of L1 norm of the error

    print('Norm L1 of error between the first point of the 2 images is ', A[0])
    print('Norm L1 of error between the second point of the 2 images is ', A[1])
    print('Norm L1 of error between the third point of the 2 images is ', A[2])
    #Show resulting images
    cv2.imshow("Original", image1)
    cv2.imshow("warped", image2)
    cv2.imshow("Affine", image_affine)
    cv2.imshow("Similarity", image_similarity)
    cv2.waitKey(0)