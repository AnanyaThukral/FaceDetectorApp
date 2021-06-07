import cv2
from random import randrange

# Load pre-trained data on face frontals (front facing faces) from opencv (haarcascade algorithm)
# CascadecClassifier: classifier - it classify something like as a face
# pass the training data and create a classifier
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect faces in imread- image read func
#img = cv2.imread('TwoPeople.jpg')
# To capture video from webcam, 0 means deffault webcam
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # read the current frame, returns tuple
    sucessful_frame_read, frame = webcam.read()

    # convet images to grey scale
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # PLUG grey image to our trained algorithm to detect
    # DETECT FACES, detectMultiScale - detects objects of multiple sizes
    # coordinates of the rectangle surrounding the face
    # location of face in the image
    # returns a lost of coordinates
    face_coordinates = trained_face_data.detectMultiScale(greyscale_img)
    # print(face_coordinates)

# Draw rectangle, 2 is the thickness of rectangle
# tuple unpacking
# loop through all faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(128, 256), randrange(256), randrange(256)), 2)

    # display the image with the faces
    cv2.imshow('Ananya Thukral Face Detector', frame)
    # wait until any key is pressed, otherwise the window closes immidiately
    # wait for 1ms, wait for 1ms go to next iteration, we are getting one frame every mili seconds
    key = cv2.waitKey(1)

    # quit when Q is pressed, in ASCII 81 or 113
    if key == 81 or key == 113:
        break

    print('Code Completed')
