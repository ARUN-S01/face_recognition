import cv2
import imutils.paths as paths

import face_recognition
import pickle
import os
dataset = r"C:\Users\a2r0u\Desktop\face_recognition\ARUN"# path of the data set 
dataset1 = r"C:\Users\a2r0u\Desktop\face_recognition\AMMA"
module = "trained_faces" # were u want to store the pickle file 

imagepaths = list(paths.list_images(dataset))
imagepaths1 = list(paths.list_images(dataset1))

knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagepaths):
    print("please wait during processing..... {}/{}".format(i + 1,len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    boxes = face_recognition.face_locations(rgb, model= "hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
       knownEncodings.append(encoding)
       knownNames.append(name)
       print("please wait during encodings...")
       data = {"encodings": knownEncodings, "names": knownNames}
       output = open(module, "wb") 
       pickle.dump(data, output)
       output.close()

for (j, imagePath1) in enumerate(imagepaths1):
    print("please wait during processing.... {}/{}".format(i + 1,len(imagepaths1)))
    name = imagePath1.split(os.path.sep)[-2]
    image = cv2.imread(imagePath1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    boxes = face_recognition.face_locations(rgb, model= "hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
       knownEncodings.append(encoding)
       knownNames.append(name)
       print("please wait during encodings...")
       data = {"encodings": knownEncodings, "names": knownNames}
       output = open(module, "wb") 
       pickle.dump(data, output)
       output.close()
