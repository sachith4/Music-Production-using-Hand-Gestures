import handy
import cv2
import numpy as np
import winsound
import time
import math
import threading
from threading import Thread

FREEZE_DURATION = 10
start = 0
class SoundPlayer(Thread):
    global start
    def __init__(self, file):
        Thread.__init__(self)
        self.filename = file
    def run(self):
        print('Playing')
        start = 1
        winsound.PlaySound(self.filename, winsound.SND_FILENAME)
        start = 0
        print('Playing Done')

def overlay_transparent(background, overlay, x, y):
  
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# getting video feed from webcam
cap = cv2.VideoCapture(0)

hist = handy.capture_histogram(source=0)

img = cv2.imread('drum.png', cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (400,200))
# cv2.imshow('actual', img)
print(img.shape)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_white = np.array([0,15,0], dtype=np.uint8)
upper_white = np.array([255,60,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)
mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask', mask)

img2gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# print(h)
_, thresh = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY)
thresh_inv = cv2.bitwise_not(thresh)
# cv2.imshow('thresh', thresh_inv)

# Find contours:
contact_points, hierrarchy = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print('Contact points :', len(contact_points))
# print(contact_points)
ctr = np.array(contact_points).astype(np.int)
# print(len(ctr))
img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
# print(mask.shape)
for_contour = img.copy()
cv2.drawContours(for_contour, contact_points , -1, (255,0,0,255), 4)
for_contour = cv2.cvtColor(for_contour, cv2.COLOR_BGR2BGRA)
# cv2.imshow('mask', for_contour)
# Dispalce it by 250 points because you do the same when adding it as overlay
for i in range(len(contact_points[0])):
    contact_points[0][i][0][1] += 250

trigger_points = []
background_points = []
receive_input = True

class ProcessPoints(Thread):
    global trigger_points, receive_input, background_points
    
    def __init__(self):
        Thread.__init__(self)
        self.running = True
    
    def run(self):
        while True:
            if self.running:
                if len(trigger_points) > 0:
                    print(('*' * 40) + 'Sound' + ('*' * 40))
                    # print(len(background_points))
                    start_point = min(background_points, key=lambda x: x[0][1])
                    end_point = trigger_points.pop()
                    print('Start point', start_point)
                    print('End point', end_point)
                    distance = abs(end_point[0][1] - start_point[0][1])
                    acceleration = abs(distance/(end_point[1] - start_point[1]))
                    print('Distance', distance)
                    print('Acceleration', acceleration)
                    filename = ''
                    if distance < 200:
                        filename = 'F:\HCI\dhi.wav'
                    elif distance < 250:                        
                        filename = 'F:\HCI\dha-slide.wav'
                    elif distance < 300:
                        filename = 'F:\HCI\dha-noslide.wav'
                    else:
                        filename = 'F:\HCI\dhin-noslide.wav'
                    for point in background_points:
                        if point[1] < end_point[1] or point[1] < start_point[1]:
                            background_points.remove(point)
                    # print(len(background_points))
                    if filename != '':
                        winsound.PlaySound(filename, winsound.SND_APPLICATION)
                        # sound_thread = SoundPlayer(file=filename)
                        # sound_thread.start()
            else:
                break
    
    def stop(self):
        self.running = False

process_points = ProcessPoints()
process_points.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand = handy.detect_hand(frame, hist)

    quick_outline = hand.outline

    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    com = hand.get_center_of_mass()
    # print(com)
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)
        dist = cv2.pointPolygonTest(contact_points[0], com, True)
        # print(dist)
        if (dist > -50 and dist < 50):
            # print('Sound')
            if receive_input:
                print('Received input')
                receive_input = False
                trigger_points.append((com, time.time()))
        else:
            # print('Left contour')
            receive_input = True
            background_points.append((com, time.time()))

    quick_outline = cv2.resize(quick_outline, (400,400))
    quick_outline = overlay_transparent(quick_outline, for_contour, 0, 250)
    
    pts = np.array(contact_points, np.int32)
    pts = pts.reshape((-1,1,2))
    # print(pts.shape)
    # print(pts.shape)
    cv2.polylines(quick_outline,[pts],True,(0,255,0))
    # quick_outline = cv2.cvtColor(quick_outline, cv2.COLOR_BGR2BGRA)
    
    # quick_outline[275:400, 0:400] = img[50:175, 0:400]
    cv2.imshow('Output', quick_outline)
    k = cv2.waitKey(1) & 0xFF
    # Press 'q' to exit
    if k == ord('q'):
        break
    # time.sleep(0.1)

process_points.stop()
process_points.join()
cap.release()
cv2.destroyAllWindows()
