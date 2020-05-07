import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# path
video_dir = './Data/'
video_path = os.path.join(video_dir,'2020-05-06-13-27-58-XT1BI.mp4')

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(video_path)
# Variable for color to draw optical flow track
color = (0, 255, 0)

scale_percent = 90 # percent of original size
ydisp_global_plot = np.array([])
ii = 1

while(cap.isOpened()):
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # prev_imCrop = prev_gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        gray_imCrop = gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        frame_imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]), ...]
        if ii == 1:
            mask = np.zeros_like(frame_imCrop)
            prev_gray_imCrop = gray_imCrop
            prev_imCrop = cv2.goodFeaturesToTrack(prev_gray_imCrop, mask = None, **feature_params)

        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv2.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray_imCrop, gray_imCrop, prev_imCrop, None, **lk_params)
        # Selects good feature points for previous position
        good_old_imCrop = prev_imCrop[status == 1]
        # Selects good feature points for next position
        good_new_imCrop = next[status == 1]
        # Draws the optical flow tracks
        ydisp_total = 0
        iii = 0
        for i, (new, old) in enumerate(zip(good_new_imCrop, good_old_imCrop)):
            iii = iii + 1
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            ydisp_total = ydisp_total + b
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame_imCrop = cv2.circle(frame_imCrop, (a, b), 3, color, -1)
        if ii == 1:
            y0 = np.int0(ydisp_total / iii)
        ii = ii + 1
        print(iii)
        ydisp_total_mean = np.int0(ydisp_total / iii)
        ydisp_global_plot = np.append(ydisp_global_plot, (y0 - ydisp_total_mean))
        # print(ydisp_total_mean)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame_imCrop, mask)
        # Updates previous frame
        prev_gray_imCrop = gray_imCrop.copy()
        # Updates previous good feature points
        prev_imCrop = good_new_imCrop.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        # cv2.imshow("sparse optical flow", output)
        cv2.imshow("Sparse Optical Flow of the Cropped Frame", np.hstack([frame_imCrop, output]))
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
# np.savetxt('data_O_Circle.csv', ydisp_global_plot, delimiter=',')
np.savetxt('data_O_Square_58_OpticalFlow.csv', ydisp_global_plot, delimiter=',')

# plot-yflow
plt.plot(ydisp_global_plot)
# plt.axis([0, 500, -20, 70])
plt.ylabel('Displcement (pixels)')
plt.xlabel('Time (Frame)')
plt.title('Vertical Displacement')
plt.savefig('vDisp_58_OpticalFlow_WF.png')
plt.show()

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()
