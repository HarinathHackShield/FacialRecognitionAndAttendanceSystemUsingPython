🧠 Facial Recognition Attendance System Using Python
A Python-based facial recognition system for automated employee/student attendance, utilizing real-time webcam capture and CSV exports. Built with modern computer vision libraries for high accuracy and fast detection.

🚀 Features
🎥 Real-time face detection with webcam
🧑‍💼 Attendance marking with face recognition
📂 Export attendance data to CSV files
📊 Admin panel with Pandas data insights
🧠 Powered by deep learning (FaceNet/dlib)
💾 Local database using sqlite3

🛠️ Tech Stack
Category	Libraries Used
👨‍💻 Backend	Flask
📸 Face Detection	OpenCV, face_recognition, dlib
📊 Data Handling	Pandas, NumPy, scikit-learn
💾 Storage	SQLite

📦 Installation
🔧 Prerequisites
Ensure you have Python 3.x installed.

You may also need CMake (for compiling dlib, a dependency of face_recognition).

📥 Install All Dependencies
bash
Copy
Edit
pip install Flask
pip install numpy
pip install opencv-python
pip install face_recognition
pip install pandas
pip install scikit-learn
pip install joblib

📝 Note: sqlite3 is included with Python's standard library.
🧱 CMake: Install from https://cmake.org/download/ if face_recognition fails due to dlib.

🧪 How to Run
python app.py

Then open your browser and visit:
👉 http://localhost:5000

📸 Screenshots
(Include images of real-time webcam detection, attendance CSV export, or UI if available)
