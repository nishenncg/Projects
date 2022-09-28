import cv2
import face_recognition
import numpy as np

img_bgr = face_recognition.load_image_file('Ranil.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
cv2.imshow('bgr', img_bgr)
cv2.imshow('rgb', img_rgb)
cv2.waitKey(0)

img =face_recognition.load_image_file('Ranil.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#----------Finding face Location for drawing bounding boxes-------
face = face_recognition.face_locations(img_rgb)[0]
copy = img.copy()
#-------------------Drawing the Rectangle-------------------------
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
cv2.imshow('copy', copy)
cv2.imshow('Ranil', img)
cv2.waitKey(0)

train_Ranil_encode = face_recognition.face_encodings(img)[0]

test = face_recognition.load_image_file('Ranil_2.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_Ranil_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_Ranil_encode],test_Ranil_encode))