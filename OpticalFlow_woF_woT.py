# optical flow
# no feature matching, no target

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
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Select Region of Interest (ROI)
# cv2.imshow('image',prev_gray)
roi = cv2.selectROI(first_frame)
# r = [left, top, width, height]
# Crop image
# imCrop = prev_gray[top: bottom, left: right]
prev_imCrop = prev_gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
print("The Coordinate of LeftTop Point (x0, y0) = " + "(" + str(int(roi[0])) + ", " + str(int(roi[0]+roi[2])) + ")")
print("The Coordinate of RightBottom Point (x1, y1) = " + "(" + str(int(roi[1])) + ", " + str(int(roi[1]+roi[3])) + ")")

# Display cropped image
first_frame_imCrop = first_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]), ...]
cv2.imshow("Region of Interest", first_frame_imCrop)

# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255
prev_vflow_imCrop = np.zeros_like(prev_imCrop)

# Creates an image filled with zero intensities with the same dimensions as the frame
mask_imCrop = np.zeros_like(first_frame_imCrop)
# Sets image saturation to maximum
mask_imCrop[..., 1] = 255

ii = 1
scale_percent = 90 # percent of original size
total_yflow_imCrop = 0
ydisp_plot_imCrop = np.array([])

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if ret == True:
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
        imCrop = gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        frame_imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]), ...]
        # cv2.imshow("Current Gray Cropped Frame", first_frame)
        
        # Calculates dense optical flow by Farneback method
        # cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
        flow_imCrop = cv2.calcOpticalFlowFarneback(prev_imCrop, imCrop, None, 0.5, 7, 25, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude_imCrop, angle_imCrop = cv2.cartToPolar(flow_imCrop[..., 0], flow_imCrop[..., 1])
        # print(flow_imCrop.shape[1])
        # print(flow_imCrop.shape[0])
        curr_vflow_imCrop = prev_vflow_imCrop + flow_imCrop[..., 1]

        # Sets image hue according to the optical flow direction
        mask_imCrop[..., 0] = angle_imCrop * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask_imCrop[..., 2] = cv2.normalize(magnitude_imCrop, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb_imCrop = cv2.cvtColor(mask_imCrop, cv2.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        # cv2.imshow("Dense Optical Flow of the Cropped Frame", rgb_imCrop)
        cv2.imshow("Dense Optical Flow of the Cropped Frame", np.hstack([frame_imCrop, rgb_imCrop]))

        # Show the vertical displacement
        yflow_imCrop = flow_imCrop[..., 1]
        # yflow_flatten_imCrop = yflow_imCrop.flatten()
        # yflow_flatten_imCrop.sort()
        # selected_top_disp = int(roi[2] * roi[3] * 20 / 100)
        # print("total number of pixels in ROI is: " + str(int(roi[2] * roi[3])))
        # yflow_selected_imCrop = yflow_flatten_imCrop[int(roi[2] * roi[3] - selected_top_disp - 1):int(roi[2] * roi[3] - 1)]
        # print("selected number of pixels for calculate vertical displacement is: " + str(len(yflow_selected_imCrop)))

        '''
        lower_threshold_indices = yflow_imCrop < 0.5
        yflow_imCrop[lower_threshold_indices] = 0
        yflow_flatten_imCrop = yflow_imCrop.flatten()
        yflow_flatten_imCrop.sort()
        yflow_selected_imCrop = yflow_flatten_imCrop[yflow_flatten_imCrop!=0]
        if len(yflow_selected_imCrop) == 0:
            yflow_avg_imCrop = 0
        else:
            yflow_avg_imCrop = -stats.trim_mean(yflow_selected_imCrop, 0.1)
        '''

        # yflow_avg_imCrop = -stats.trim_mean(yflow_selected_imCrop, 0.1)
        # print("displacement is: " + str(yflow_avg_imCrop))
        lower_threshold_indices = abs(flow_imCrop[..., 1]) < 0.5
        # print(lower_threshold_indices) True
        curr_vflow_imCrop[lower_threshold_indices] = 0
        yflow_flatten_imCrop = curr_vflow_imCrop.flatten()
        yflow_flatten_imCrop.sort()
        yflow_selected_imCrop = yflow_flatten_imCrop[yflow_flatten_imCrop!=0]
        if len(yflow_selected_imCrop) == 0:
            yflow_median_imCrop = 0
        else:
            yflow_median_imCrop = np.median(yflow_selected_imCrop)
        # print("displacement is: " + str(yflow_median_imCrop))
        # total_yflow_imCrop = total_yflow_imCrop + yflow_median_imCrop
        total_yflow_imCrop = yflow_median_imCrop
        ydisp_plot_imCrop = np.append(ydisp_plot_imCrop, total_yflow_imCrop)

        '''
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        cv2.imshow("dense optical flow", rgb)
        '''

        # Updates previous frame
        prev_imCrop = imCrop
        prev_vflow_imCrop = curr_vflow_imCrop
        
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('optFlow' + str(ii) + '.png', output)
        # index of frame
        # print(ii)
        ii = ii + 1
    else:
        break
savetxt('data_O_Circle.csv', ydisp_plot_imCrop, delimiter=',')
# plot-yflow
plt.plot(ydisp_plot_imCrop)
plt.axis([0, 700, -20, 60])
plt.ylabel('Displcement (mm)')
plt.xlabel('Time(sec)')
plt.title('Vertical Displacement')
plt.savefig('O_vDisp.png')
plt.show()
        
# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()
