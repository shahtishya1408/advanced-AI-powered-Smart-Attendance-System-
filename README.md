# Smart Attendance System (Face Recognition + Cloud)

An advanced AI-powered Smart Attendance System that uses **Face Recognition**, **Offline-First Architecture**, and **Cloud-Ready Design** to eliminate proxy attendance, reduce manual errors, and provide real-time attendance insights for educational institutions.

This project is designed for **Google Developer Groups (GDG) Hackathons**, focusing on **Education & Skill Development**.

---

## 🚀 Key Features

### 🎯 Core Attendance Features
- Face Recognition–based attendance marking
- Automatic **Present / Late / Absent** classification
- Configurable **late threshold**
- One attendance entry per student per day (data integrity)
- Automatic daily absent record generation

### 🧠 AI & Computer Vision
- OpenCV Haar Cascade for face detection
- LBPH (Local Binary Pattern Histogram) Face Recognition
- Adjustable confidence threshold
- Offline face recognition support
- Model training & retraining from captured samples

### 🖥️ Desktop Application
- Python Tkinter GUI
- Modern UI with `ttkbootstrap` (auto-fallback to ttk)
- Dashboard with attendance statistics
- Student management (Add / View / Filter)
- Department-wise filtering

### ☁️ Cloud & Data Management
- SQLite database (local-first, cloud-ready)
- Easily extendable to Google Cloud / Firebase
- CSV, Excel & PDF attendance exports
- Offline-first architecture with future cloud sync support

### 🔊 Accessibility
- Offline **female voice text-to-speech (TTS)**
- Voice confirmation on successful recognition
- Audio feedback for errors

### 📧 Automation (Optional)
- Automatic absent email notification system
- Scheduled daily email dispatch
- SMTP support (Gmail / custom servers)

---

## 🧩 Technology Stack

| Category | Technology |
|-------|------------|
| Programming | Python 3 |
| UI | Tkinter, ttkbootstrap |
| AI / ML | OpenCV, LBPH Face Recognition |
| Database | SQLite |
| Speech | pyttsx3 (Offline TTS) |
| Reporting | CSV, Excel (pandas), PDF (reportlab) |
| Optional Cloud | Google Cloud / Firebase (extensible) |

---

## 📂 Project Structure
Smart-Attendance-System/
│
├── attendance.db # SQLite database
├── face_dataset/ # Face image samples
├── trainer.yml # Trained LBPH model
├── label_map.csv # Face label mapping
├── main.py # Main application
├── README.md # Project documentation
└── requirements.txt # Dependencies

---

## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies
```bash
pip install opencv-contrib-python pyttsx3 pandas reportlab ttkbootstrap

python main.py


📸 How It Works

Admin adds students to the system

Face samples are captured via webcam

LBPH model is trained

Face recognition marks attendance automatically

Attendance data is stored securely

Reports can be exported anytime

Optional email alerts are sent to absentees


Dashboard Insights

Total students

Present / Absent count

Attendance percentage

Department-wise filtering

Daily summaries

Privacy & Security

No cloud upload without consent

Local data storage by default

Face data stored securely

Easily adaptable to institutional policies



🌱 Future Enhancements
Google Firebase / Cloud SQL integration

Android & Web app versions

QR + Face hybrid attendance

Edge AI deployment

LMS & ERP integration

Live cloud dashboards

👨‍💻 Author

Tishya Shah
 Engineering Student
Gujarat Technological University (GTU)

