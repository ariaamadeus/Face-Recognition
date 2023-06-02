import os
from tkinter import *
import tkinter.font as font
import tkinter.simpledialog as simpledialog
from PIL import Image, ImageTk
from tkinter import messagebox as mbox

import cv2
import face_recognition
from numpy import info, asarray
from sklearn import svm
import joblib
import datetime
import time

model = None

def get_latest_model(minus=0):
    max_number_model = 0
    for model_path in os.listdir():
        if model_path == "model.pkl":
            max_number_model = 1
        elif model_path[:5] == "model" and model_path[-4:] == ".pkl":
            if int(model_path[5]) >= max_number_model:
                max_number_model = int(model_path[5])+1
    max_number_model+=minus
    if max_number_model == 0:
        max_number_model = None
    return max_number_model

def train_svm():
    global model

    if not os.path.exists("train_data"):
        print("Please Capture Some Face First")
        return

    print("Training the model...")
    # Training the SVC classifier

    # The training data would be all the face encodings from all the known images and the labels are their names
    start = time.time() #start training stopwatch
    encodings = []
    names = []

    
    train_dir = os.listdir('train_data/')

    for person in train_dir:
        pix = os.listdir("train_data/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                "train_data/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img +" was skipped and can't be used for training")

    # Create and train the SVC classifier
    model = svm.SVC(gamma='scale')
    
    model.fit(encodings, names)

    joblib.dump(model,"model%s.pkl"%get_latest_model())
    print("Training done in %ss"%str(datetime.timedelta(seconds=int(time.time()-start)))) #stop train stopwatch

def test_svm(frame):
    global model
    if model == None: 
        train_svm()
    
    start = time.time()
    
    # Load the test image with unknown faces into a numpy array
    # test_image = face_recognition.load_image_file('test/test.jpg')

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(frame)
    num = len(face_locations)
    print("Number of faces detected: ", num)

    # Predict all the faces in the test image using the trained classifier
    list_names = []
    print("Found:")
    for i in range(num):
        test_image_enc = face_recognition.face_encodings(frame)[i]
        name = model.predict([test_image_enc])
        print(name)
    print()

def test_img_capture():

    window.destroy()

    img_counter = 0
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Recognition")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Face Recognition", frame)
        test_svm(frame)
        k = cv2.waitKey(1)
        if k % 256 == ord("q"):break
        elif k % 256 == 27:break
            # SPACE pressed
            # img_name = "test.jpg"
            # cv2.imwrite(os.path.join('test', img_name), frame)
            # print("{} written!".format(img_name))
            # print("Closing now")
            # img_counter += 1
            # break
    cam.release()
    cv2.destroyAllWindows()
    window.start_window()

def train_img_capture():

    file_name = ''
    file_name = simpledialog.askstring(title="Face Recognition",prompt="What's your Name?:")
    window = Tk()
    window.withdraw()
    
    if file_name != None and file_name != '':
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Face Training")
        img_counter = 0

        if not os.path.exists("train_data"):
            os.mkdir("train_data")

        path = "train_data/"+file_name

        if os.path.exists(path):
            for file in os.listdir(path):
                latest_file_num = int(file[5:-4])
                if img_counter < latest_file_num:
                    img_counter = latest_file_num
            print(f"Continue Adding Dataset From: {img_counter}")
        else: 
            print("Creating New Name")
            os.mkdir(path)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Face Training", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = file_name+"_{}.jpg".format(img_counter)
                cv2.imwrite(os.path.join("train_data/"+file_name, img_name), frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()
        window.destroy()
    
def display_name(list_name): 
    window=Tk()
    label = Label(window, text="Faces Recognized")
    listbox = Listbox(window, width=50)
    label.pack()
    listbox.pack(fill=BOTH, expand=1)  # adds listbox to window
    for row in list_name:
        listbox.insert(END, row)   # one line for loop
    window.mainloop()

class Window:
    def start_window(self):
        self.window = Tk()
        self.window.config(width=300, height=300,padx=20,pady=50)
        label = Label(
            self.window, text='WELCOME TO FACE RECOGNITION. SELECT AN OPTION BELOW:\n',font=font.Font(size=16))
        label.pack()
        button = Button(self.window, text="Capture",command=train_img_capture,width=20,bg="white",fg="brown",pady=10)
        button['font']=font.Font(size=16)
        button.pack()
        label = Label(self.window, text='\n')
        label.pack()
        button = Button(self.window, text="Train",command=train_svm,width=20,bg="white",fg="dark red",pady=10)
        button['font']=font.Font(size=16)
        button.pack()
        label = Label(self.window, text='\n')
        label.pack()
        button = Button(self.window, text="Test", command=test_img_capture,width=20,bg="cyan",fg="darkgreen",pady=10)
        button['font']=font.Font(size=16)
        button.pack()
        label = Label(self.window, text='\n')
        label.pack()
        button = Button(self.window, text="Quit", command=lambda:exit(),width=20,bg="Red",fg="White",pady=10)
        button['font']=font.Font(size=16)
        button.pack()
        self.window.mainloop()
    
    def destroy(self):
        self.window.destroy()

if __name__ == "__main__":

    if get_latest_model(-1) != None:
        if get_latest_model(-1) >= 0:
            model = joblib.load("model%s.pkl"%(get_latest_model(-1)))
        else: model = None
    else: model = joblib.load("model.pkl")

    window = Window()
    window.start_window()

    

    




