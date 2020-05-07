# Feature Point matching 
# artifical target: Circles
# no optical flow

import cv2
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# path
video_dir = './Data/'
video_path = os.path.join(video_dir,'2020-05-06-00-57-15-8QLCX.mp4')

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(video_path)
scale_percent = 90 # percent of original size

# ydisp_plot_imCrop = np.array([])
ydisp_global_plot = np.array([])
ii = 0

while(cap.isOpened()):
    ii = ii + 1
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()

    if ret == True:
        if ii == 1:
            roi = cv2.selectROI(frame)
            print("The Coordinate of LeftTop Point (x0, y0) = " + "(" + str(int(roi[0])) + ", " + str(int(roi[1])) + ")")
            print("The Coordinate of RightBottom Point (x1, y1) = " + "(" + str(int(roi[0]+roi[2])) + ", " + str(int(roi[1]+roi[3])) + ")")
            
        # Opens a new window and displays the input frame
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("input", resized)

        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]), ...]
        copy_imCrop = imCrop.copy()
        # cv2.imshow("Current Cropped Frame", imCrop)
        gray_imCrop = cv2.cvtColor(copy_imCrop, cv2.COLOR_BGR2GRAY)

        # binary image
        ret, thresh1 = cv2.threshold(gray_imCrop, 127, 255, cv2.THRESH_BINARY)
        # print(thresh1)
        
        # draw the reference lines
        # horizontal direction
        # every 3 pixels gets a line
        # calculate the number of lines in horizontal and vertical directions
        nH = np.int0(roi[3] / 3)
        nW = np.int0(roi[2] / 3)
        # get the samples of lines in H- and V directions
        hLine = np.array([])
        vLine = np.array([])

        # horizontal direction
        # initialize the cordinates of all sample points in H- direction
        hCoor = np.empty((0, 2), int)
        xPlus1HLine = np.array([]) # f(x+1) for calculate the gradient, <class 'numpy.ndarray'>
        for i in range(nH):
            # calculate the gradient of each line
            hLine = thresh1[i*3, :] # f(x)
            xPlus1HLine = np.concatenate(([hLine[0]], hLine[0: -1])) # f(x+1)
            hD = xPlus1HLine - hLine # d = f(x+1) - f(x)
            hRes = [idx for idx, val in enumerate(hD) if val != 0] # index of non-zero value
            hSubCoor = np.zeros((len(hRes), 2)) # initialize the sample points for each line
            for idx, val in enumerate(hRes): # coordinate
                hSubCoor[idx][0] = val # x
                hSubCoor[idx][1] = i*3 # y
            hCoor = np.concatenate((hCoor, hSubCoor), axis=0)
        
        # linear model estimation using RANSAC
        XHCoor = hCoor[:, 0]
        TXHCoor = XHCoor.reshape(-1, 1)
        yHCoor = hCoor[:, 1]  

        # Robustly fit linear model with RANSAC algorithm
        hRansac = linear_model.RANSACRegressor()
        hRansac.fit(TXHCoor, yHCoor)
        hInlierMask = hRansac.inlier_mask_
        hOutlierMask = np.logical_not(hInlierMask)

        # Predict data of estimated models
        XHLine = np.arange(TXHCoor.min(), TXHCoor.max())[:, np.newaxis]
        yHLineRansac = hRansac.predict(XHLine)
        hX0 = np.int0(XHLine[0][0])
        hy0 = np.int0(yHLineRansac[1])
        hX1 = np.int0(XHLine[-1][0])
        hy1 = np.int0(yHLineRansac[-1])

        # Estimated coefficients
        # print("Estimated RANSAC coefficients: " + float(ransac.estimator_.coef_))

        # Draw line
        hColor = (0, 255, 0)
        cv2.line(copy_imCrop, (hX0, hy0), (hX1, hy1), hColor, 2)

        
        # vertical direction
        # initialize the cordinates of all sample points in V- direction
        vCoor = np.empty((0, 2), float)
        xPlus1VLine = np.array([]) # f(x+1) for calculate the gradient, <class 'numpy.ndarray'>
        for i in range(nW):
            # calculate the gradient of each line
            vLine = thresh1[:, i*3] # f(x)
            xPlus1VLine = np.concatenate(([vLine[0]], vLine[0: -1])) # f(x+1)
            vd1 = xPlus1VLine - vLine # d = f(x+1) - f(x)
            vRes = [idx for idx, val in enumerate(vd1) if val != 0] # index of non-zero value
            vSubCoor = np.zeros((len(vRes), 2)) # initialize the sample points for each line
            for idx, val in enumerate(vRes): # coordinate
                vSubCoor[idx][0] = i*3 # y
                vSubCoor[idx][1] = val # x
            vCoor = np.concatenate((vCoor, vSubCoor), axis=0)

        # linear model estimation using RANSAC
        XVCoor = vCoor[:, 0]
        TXVCoor = XVCoor.reshape(-1, 1)
        yVCoor = vCoor[:, 1]

        # Robustly fit linear model with RANSAC algorithm
        vRansac = linear_model.RANSACRegressor()
        vRansac.fit(TXVCoor, yVCoor)
        vInlierMask = vRansac.inlier_mask_
        vOutlierMask = np.logical_not(vInlierMask)

        # Predict data of estimated models
        XVLine = np.arange(TXVCoor.min(), TXVCoor.max())[:, np.newaxis]
        yVLineRansac = vRansac.predict(XVLine)
        vX0 = np.int0(XVLine[0][0])
        vy0 = np.int0(yVLineRansac[1])
        vX1 = np.int0(XVLine[-1][0])
        vy1 = np.int0(yVLineRansac[-1])

        # Estimated coefficients
        # print("Estimated RANSAC coefficients: " + float(ransac.estimator_.coef_))

        # Draw line
        vColor = (255, 0, 0)
        cv2.line(copy_imCrop, (vX0, vy0), (vX1, vy1), vColor, 2)

        # Calculate the intersection point
        # iP = lineLineIntersection([hX0, hy0], [hX1, hy1], [vX0, vy0], [vX1, vy1])
        
        # Line AB represented as m1x + b1 = y
        m1 = (hy1 - hy0)/(hX1 - hX0)
        b1 = hy1 - m1 * hX1
    
        # Line CD represented as m2x + b2 = y
        m2 = (vy1 - vy0)/(hX1 - hX0)
        b2 = vy1 - m2 * vX1      

        xi = (b1-b2) / (m2-m1)
        yi = m1 * xi + b1
                
        # print("[" + str(xi) + ", " + str(yi) + "]")
        cv2.rectangle(copy_imCrop, (np.int0(xi) - 2, np.int0(yi) - 2), (np.int0(xi) + 2, np.int0(yi) + 2), (0, 128, 255), -1)

        if ii == 1:
            y0 = yi
        ydisp_global = y0 - yi
        ydisp_global_plot = np.append(ydisp_global_plot, ydisp_global)       
        
        cv2.imshow("IntersectionPoint is Detected", np.hstack([imCrop, copy_imCrop]))
        
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('SquareMatching' + str(ii) + '.png', copy_imCrop)
        # index of frame
    else:
        break

np.savetxt('data_I_S_57.csv', ydisp_global_plot, delimiter=',')

# plot-yflow
plt.plot(ydisp_global_plot)
# plt.axis([0, 800, -10, 70]) # 49
# plt.axis([0, 700, -5, 30]) # 48
plt.ylabel('Displcement (pixels)')
plt.xlabel('Time (Frame)')
plt.title('Vertical Displacement')
plt.savefig('I_vDisp_57.png')
plt.show()

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()