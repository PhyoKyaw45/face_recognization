import cv2 as cv
import os
import numpy as np

# detect face
def detect_face(img):
    """
    this function detect face in given image & return face roi and face

    """
    # change to gray color to detect more efficent
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # face cascade file
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect face (don't make scale factor 1.3 , it will give u error)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # if there is no face , return Nothing
    if len(faces) == 0:
        print("lose")
        return None, None

    # get face area
    (x, y, w, h) = faces[0]

    # return face roi
    return gray[y:y+w, x:x+h], faces[0]

def prepare_taining_data(data_folder_path):
    """
    This function get images from each folder and detect and return faces and labels
    each folder = data_folder_path + subject folder path + image names

    """
    # folder from main directories
    dirs = os.listdir(data_folder_path)

    print(dirs)
    # create list of faces and labels
    faces = []

    labels = []

    # get subject folder from main directories
    for dir_names in dirs:
        # if folder name don't start with a , skip that
        if not dir_names.startswith('p'):
            continue

        label = int(dir_names.replace("p", ""))

        subject_folder_path = data_folder_path + "/" + dir_names

        print(subject_folder_path)

        subject_images_names = os.listdir(subject_folder_path)
        print(subject_images_names)
        # image names from subject folder path
        for img_names in subject_images_names:
            if img_names.startswith("."):
                continue
            # image path
            img_path = subject_folder_path+"/" + img_names


            # read images
            image = cv.imread(img_path)

            # show images
            cv.imshow("Training data", cv.resize(image, (400, 500)))
            cv.waitKey(100)

            # detect faces
            face, rect = detect_face(image)

            # save faces and labels
            if face is not None:
                # add face
                faces.append(face)
                # add label
                labels.append(label)
    cv.destroyAllWindows()
    cv.waitKey()
    cv.destroyAllWindows()
    return faces, labels

print("preparing data")
faces, labels = prepare_taining_data("training data")

print("Number of faces:", len(faces))
print("NUmber of labels", len(labels))
print(labels)
# subjects
subjects = ["", "San Lin Ko", "Lin Thu Ya", "H.P.K"]

# create lbph face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train faces and labels
face_recognizer.train(faces, np.array(labels))

def draw_Rectangle(img, rect):
    """
        this function take image and rectangle value to draw rectangle on face of given image
    """
    (x, y, w, h) = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

def PutText(img,text,x, y):
    """
        this function take image and put given text on given image with given (x,y) coordinates
    """
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

def predict(test_img):
    """
        this function take an image and predict this face is who and add labels
    """
    # get a copy image so that we avoid writing on original image
    img = test_img.copy()

    # detect face
    face, rect = detect_face(img)

    # predict
    label = face_recognizer.predict(face)

    # print label
    print(label)

    # draw Rectangle
    draw_Rectangle(img, rect)

    # put text
    label_text = subjects[label[0]]
    PutText(img, label_text, rect[0], rect[1]-5)
    return img

# feed an image to predict
test_img1 = cv.imread("test data/t1.jpg")
test_img2 = cv.imread("test data/t2.jpg")
test_img3 = cv.imread("test data//t3.jpeg")

# get predicted result
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)

# show images
cv.imshow(subjects[1], cv.resize(predicted_img1, (500, 600)))
cv.imshow(subjects[2], cv.resize(predicted_img2, (500, 600)))
cv.imshow(subjects[3], cv.resize(predicted_img3, (500, 600)))
cv.waitKey(0)
cv.destroyAllWindows()