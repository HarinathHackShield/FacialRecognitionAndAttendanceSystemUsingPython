import sqlite3
import cv2
import os
from flask import Flask,flash,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import csv
from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# path = 'ImagesAttendance'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)


# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodes = face_recognition.face_encodings(img)
#         if encodes:  # Check if list is not empty
#             encodeList.append(encodes[0])  # Only add the first encoding if it exists
#         else:
#             print("No faces found in the image.")
#     return encodeList


# encodeListKnown = findEncodings(images)
# print('Encoding Completed')

def load_and_encode_images(path='ImagesAttendance'):
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Check if list is not empty
            encodeList.append(encodes[0])  # Only add the first encoding if it exists
        else:
            print("No faces found in the image.")
    return encodeList, classNames




#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### If these directories don't exist, create them
# Initial CSV file creation with proper headers
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Name', 'Roll', 'Time'])  # Ensure headers are explicitly defined


#### get a number of total registered users
def totalreg():
    return len(os.listdir('ImagesAttendance'))


def extract_attendance():
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    try:
        df = pd.read_csv(attendance_file)
        
        # Check if 'Time' column exists, if not create it with a default value
        if 'Time' not in df.columns:
            df['Time'] = 'No Time Recorded'  # or any default value you see fit
        
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        l = len(df)
        return names, rolls, times, l
    except FileNotFoundError:
        print("Attendance file not found.")
        return [], [], [], 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], [], 0

#### Add Attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    
    # Check if the file exists and read the current attendance records
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r', newline='') as f:
            existing_records = [row for row in csv.reader(f)]
    else:
        existing_records = []
    
    # Check if the user has already been marked present
    for record in existing_records:
        if userid in record:
            print(f"Attendance already recorded for {name}")
            return False  # Return False to indicate the attendance was not added because it's a duplicate
    
    # If the user is not already marked present, add their attendance
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, userid, current_time])
    print(f"Attendance recorded for {name}")  # Debugging log
    return True

################## ROUTING FUNCTIONS ##############################

#### Our main page
@app.route('/')
def index():
    names, rolls, times, l = extract_attendance()
    total_registered = totalreg()  # Make sure this doesn't clash with variable names
    datetoday2 = date.today().strftime("%d-%B-%Y")
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=total_registered, datetoday2=datetoday2)


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():

    encodeListKnown, classNames = load_and_encode_images()  # Reload and re-encode images
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.5,0.5)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches =  face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*2,x2*2,y2*2,x1*2
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                add_attendance(name)

        cv2.imshow('Webcam', img)

        # Check if 'q' is pressed to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))



@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        # Retrieve form data
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimage = request.files['userimage']

        # Define the single path where images will be saved
        userimagefolder = os.path.join('ImagesAttendance')
        if not os.path.exists(userimagefolder):
            os.makedirs(userimagefolder)

        # Process the uploaded image file
        if userimage:
            # Construct the new filename using username and userid
            # Assuming you want to keep the original file extension
            filename = secure_filename(f"{newusername}_{newuserid}{os.path.splitext(userimage.filename)[1]}")
            save_path = os.path.join(userimagefolder, filename)
            userimage.save(save_path)
            print(f"Saved image to: {save_path}")

        # Redirect or handle other parts of your function here
        return redirect(url_for('index'))
    else:
        # Handle GET request or show the form
        return render_template('index.html')  # Ensure you have a template named 'index.html'

    
from flask import send_file

@app.route('/download-todays-attendance')
def download_todays_attendance():
    filename = f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv'
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404




names, rolls, times, l = extract_attendance()
print(names, rolls, times, l)

if __name__ == '__main__':
    app.run(debug=True,port=1000)
