# Feature Point matching 
# artifical target: Circles
# no optical flow

import cv2
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

# path
video_dir = './Data/'
video_path = os.path.join(video_dir,'C0038.MP4')

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(video_path)
scale_percent = 90 # percent of original size
# Set min/max Redius for circle detector
# left 38/41
# middle 68/72
# right 76/78
minR = 0
maxR = 0

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
        # cv2.imshow("Current Cropped Frame", imCrop)
        
        # Apply Circle Hough Transform to detect the circle
        # Blur using 5 * 5 kernel
        imCrop_blurred = cv2.GaussianBlur(imCrop, (5, 5), 0)
        copy_imCrop = imCrop_blurred.copy()
        gray_imCrop = cv2.cvtColor(copy_imCrop, cv2.COLOR_BGR2GRAY)
        
        # detect circles in the image
        # circles = cv2.HoughCircles(gray_imCrop, cv2.HOUGH_GRADIENT, 1.2, 100)
        circles = cv2.HoughCircles(gray_imCrop, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=minR, maxRadius=maxR)
        if ii == 1:
            y0 = circles[0][0][1]
        # print(len(circles))
        
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # print(circles) # [[131 100  56]]
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # print(y)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(copy_imCrop, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(copy_imCrop, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                ydisp_global = y0 - y
                ydisp_global_plot = np.append(ydisp_global_plot, ydisp_global)
            # show the output image
            cv2.imshow("Circle is Detected", np.hstack([imCrop, copy_imCrop]))
        else:
            circles = prev_circles
            print("no circle in " + str(ii))
            # continue
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # print(y)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(copy_imCrop, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(copy_imCrop, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                # ydisp_global = frame.shape[0] - roi[1] - roi[3] + y
                ydisp_global = y0 - y
                ydisp_global_plot = np.append(ydisp_global_plot, ydisp_global)
            # show the output image
            cv2.imshow("Circle is Detected", np.hstack([imCrop, copy_imCrop]))

        prev_circles = circles
        
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('targetMatching' + str(ii) + '.png', copy_imCrop)
        # index of frame
        # print(ii)
    else:
        break
# np.savetxt('data_O_C.csv', ydisp_global_plot, delimiter=',')

# plot-yflow
plt.plot(ydisp_global_plot)
plt.axis([0, 700, -20, 60])
plt.ylabel('Displcement (pixels)')
plt.xlabel('Time(sec)')
plt.title('Vertical Displacement')
plt.savefig('vDisp.png')
plt.show()

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()