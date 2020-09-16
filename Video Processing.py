import cv2

# Opens the Video file
cap = cv2.VideoCapture('Video/p3.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    key = cv2.waitKey(1)
    cv2.imwrite('training data/p3/' + str(i) + '.jpg', frame)
    i += 1
    cv2.imshow('gg', frame)
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()