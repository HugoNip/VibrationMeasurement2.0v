#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("C0015_Pipe.MP4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1) # hsv.shape -> (1080, 1920, 3)
hsv[...,1] = 255            # shape -> (1080, 1920)

mag_median = 0
mag_plot = np.array([])

yflow_median = 0
total_yflow = 0
ydisp_plot = np.array([])

ii = 1
while(1):
    print(ii)
    ii = ii + 1
    if ii == 384:
        break

    blur_prvs = cv2.GaussianBlur(prvs, (5, 5), 0)
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blur_next = cv2.GaussianBlur(next, (5, 5), 0)

    # Computes a dense optical flow using the Gunnar Farnebacks algorithm
    # cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
    flow = cv2.calcOpticalFlowFarneback(blur_prvs, blur_next, None, 0.5, 7, 25, 3, 5, 1.2, 0) # shape -> (1080, 1920, 2)
    yflow = flow[...,1]
    # lower_threshold_indices_yflow = yflow < 0.5
    # yflow[lower_threshold_indices_yflow] = 0
    
    # Calculates the magnitude and angle of 2D vectors
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) # x, y
    lower_threshold_indices = mag < 0.5
    mag[lower_threshold_indices] = 0

    mag_flatten = mag.flatten()
    mag_flatten.sort()
    mag_slected = mag_flatten[mag_flatten!=0]
    if len(mag_slected) == 0:
        mag_median = 0
    else:
        mag_median = np.median(mag_slected)
    mag_plot = np.append(mag_plot, mag_median)
    # print(mag_plot)

    yflow[lower_threshold_indices] = 0
    yflow_flatten = yflow.flatten()
    yflow_flatten.sort()
    yflow_slected = yflow_flatten[yflow_flatten!=0]
    if len(yflow_slected) == 0:
        yflow_avg = 0
    else:
        yflow_avg = -np.sum(yflow_slected)/len(yflow_slected)
    total_yflow = total_yflow + yflow_avg
    ydisp_plot = np.append(ydisp_plot, total_yflow)

    print(yflow_avg)

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    

    cv2.imshow('frame2', rgb)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = blur_next

# plot-mag
plt.plot(mag_plot)
plt.ylabel('Disp(pixel)')
plt.xlabel('Time(sec)')
plt.title('Measurement of Magnitude along Velocity Vector for Each Motion')
plt.savefig('Disp.png')
plt.show()

# plot-yflow
plt.plot(ydisp_plot)
plt.ylabel('y(pixel)')
plt.xlabel('Time(sec)')
plt.title('Measurement of Magnitude along y-axis')
plt.savefig('yDisp.png')
plt.show()

cap.release()
cv2.destroyAllWindows()