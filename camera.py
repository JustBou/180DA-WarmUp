import cv2
import numpy as np

#question:
#HSV is typically better; the threshold range is pretty large for HSV; based on saturation and brightness, the range is pretty high, but the difference in hue is pretty low
#lighting conditions do effect the tracking ability; at the lowest levels, it doesn't get picked up by my values
#Changing the phone brightness mainly hurts the code espeically at the lowest levels of brightness

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while(True):
        ret, fram = cap.read()
        frame = cv2.GaussianBlur(fram,(11,11),5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90,100,100])
        upper_blue = np.array([130,255,255])
        white_mask_low = np.array([0,0,250])
        white_mask_high = np.array([255,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(gray, lower_blue, upper_blue)
        # mask2 = cv2.inRange(gray, white_mask_low, white_mask_high)
        # mask = cv2.bitwise_or (mask1,mask2)
        res = cv2.bitwise_and(fram,fram, mask= mask)
        contours,hierarchy = cv2.findContours(mask, 1, 2)
        if contours:
            for cnt in contours:
                # M = cv2.moments(cnt)
                # area = cv2.contourArea(M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)


        
        cv2.imshow('frame', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
