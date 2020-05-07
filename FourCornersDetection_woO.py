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
video_path = os.path.join(video_dir,'C0047.MP4')

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
 
        # find Harris corners
        gray = np.float32(gray_imCrop)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
        print(corners)
        print(centroids)

        # Now draw them
        res = np.hstack((centroids, corners))
        res_show = np.int0(res)
        # print(res.shape[0])
        # print(res.shape[1])
        print(res_show)
        copy_imCrop[res_show[:,1], res_show[:,0]] = [0, 0, 255] # Red
        copy_imCrop[res_show[:,3], res_show[:,2]] = [0, 255, 0] # Green
        cv2.imshow("Corner is Detected", np.hstack([imCrop, copy_imCrop]))

        if ii == 1:
            y0 = (res[1][1] + res[1][3]) * 0.5
        y = (res[1][1] + res[1][3]) * 0.5

        ydisp_global = y0 - y
        ydisp_global_plot = np.append(ydisp_global_plot, ydisp_global)
        
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('SquareMatching' + str(ii) + '.png', copy_imCrop)
        # index of frame
    else:
        break

np.savetxt('data_O_S_47.csv', ydisp_global_plot, delimiter=',')

# plot-yflow
plt.plot(ydisp_global_plot)
# plt.axis([0, 700, -5, 30]) # 48
plt.ylabel('Displcement (pixels)')
plt.xlabel('Time (Frame)')
plt.title('Vertical Displacement')
plt.savefig('S_vDisp_48.png')
plt.show()

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()