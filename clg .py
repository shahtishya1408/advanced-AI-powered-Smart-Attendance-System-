#!/usr/bin/env python3
"""
Advanced Attendance System (Audible Female TTS + Fixes)
- SQLite students + attendance
- UNIQUE per-day Absent rows (auto-restore)
- Late threshold marking
- Dept filter
- CSV/Excel/PDF export (optional)
- ttkbootstrap UI fallback to ttk
- OpenCV face capture/train/recognize (optional)
- Email absent alerts (optional)
- Female voice offline TTS via pyttsx3
"""

import os, sys, sqlite3, csv, threading, smtplib
from datetime import datetime, date, timedelta, time as dtime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from email.message import EmailMessage

# ---------- Optional deps ----------
try:
    import ttkbootstrap as tb
    TB_AVAILABLE = True
except Exception:
    tb = None
    TB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import cv2, numpy as np
    CV2_AVAILABLE = True
    CV2_CONTRIB = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
except Exception:
    CV2_AVAILABLE = False
    CV2_CONTRIB = False

# Offline TTS with female voice selection (pyttsx3)
try:
    import pyttsx3
    TTS_AVAILABLE = True
    _tts_engine = None
    _tts_voice_id = None

    def tts_init_select_female():
        # Initialize engine and select a female voice if available
        global _tts_engine, _tts_voice_id
        if _tts_engine is None:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 170)
            _tts_engine.setProperty("volume", 1.0)
        voices = _tts_engine.getProperty("voices")
        female = None
        # Prefer explicit gender metadata if present
        for v in voices:
            g = getattr(v, "gender", None)
            if g and str(g).lower().endswith("female"):
                female = v
                break
        # Fallback by name/id heuristics across OSes
        if female is None:
            for v in voices:
                name = (v.name or "").lower()
                vid = (getattr(v, "id", "") or "").lower()
                if "female" in name or "zira" in name or "samantha" in vid:
                    female = v
                    break
        if female is None:
            for v in voices:
                if "zira" in (v.name or "").lower():  # Windows
                    female = v
                    break
        if female is None:
            for v in voices:
                if "samantha" in (getattr(v, "id", "") or "").lower():  # macOS
                    female = v
                    break
        if female:
            _tts_engine.setProperty("voice", female.id)
            _tts_voice_id = female.id
        else:
            # Linux espeak female-ish variants
            try:
                _tts_engine.setProperty("voice", "english+f3")
                _tts_voice_id = "english+f3"
            except Exception:
                _tts_voice_id = None

    def tts_say(text: str):
        global _tts_engine, _tts_voice_id
        if _tts_engine is None or _tts_voice_id is None:
            tts_init_select_female()
        if _tts_engine is None:
            return
        try:
            _tts_engine.stop()
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception:
            pass
except Exception:
    TTS_AVAILABLE = False
    def tts_say(text: str): pass

# ---------- App storage ----------
APP_DB = "attendance.db"
DATASET_DIR = "face_dataset"
MODEL_FILE = "trainer.yml"
LABEL_MAP_FILE = "label_map.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------- Icon helpers ----------
def here(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)

def ensure_icon_from_png():
    ico = here("C:\\Users\\diyas\\OneDrive\\Desktop\\FACE\\attendance.ico")
    if os.path.exists(ico):
        return ico
    png = here("attendance.png")
    if os.path.exists(png):
        try:
            from PIL import Image
            img = Image.open(png).convert("RGBA")
            sizes = [(256,256),(128,128),(64,64),(32,32),(16,16)]
            img.save(ico, format="ICO", sizes=sizes)
        except Exception:
            pass
    return ico

def set_window_icon(root: tk.Tk):
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"com.diya.attendance.1")
    except Exception:
        pass
    ico = here("icon.ico")
    if not os.path.exists(ico):
        ico = ensure_icon_from_png()
    if os.path.exists(ico):
        try:
            root.iconbitmap(ico)
        except Exception:
            pass
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(file=ico)
            root.iconphoto(True, photo)
            root._icon_ref = photo
        except Exception:
            pass

# ---------- DB helpers ----------
def get_db_connection():
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll TEXT,
            department TEXT,
            email TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            time TEXT,
            status TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    # Unique per student per day
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_unique ON attendance(student_id, date)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS config(
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    defaults = {
        "late_threshold": "09:15:00",
        "absent_notify_time": "18:00:00",
        "email_sender": "",
        "email_password": "",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": "587",
        "auto_send_email": "0",
        "theme": "light",
        "lbph_threshold": "85"
    }
    for k, v in defaults.items():
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()

def get_config(key):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None

def set_config(key, value):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO config(key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

def upgrade_db():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        # attendance table migrations
        cur.execute("PRAGMA table_info(attendance)")
        att_cols = {r[1] for r in cur.fetchall()}
        # Ensure legacy DBs get a `time` column
        if "time" not in att_cols:
            cur.execute("ALTER TABLE attendance ADD COLUMN time TEXT DEFAULT ''")
        if "status" not in att_cols:
            cur.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'Absent'")

        # students table migrations
        cur.execute("PRAGMA table_info(students)")
        stu_cols = {r[1] for r in cur.fetchall()}

        if "email" not in stu_cols:
            cur.execute("ALTER TABLE students ADD COLUMN email TEXT")
        if "roll" not in stu_cols:
            cur.execute("ALTER TABLE students ADD COLUMN roll TEXT")
        if "department" not in stu_cols:
            cur.execute("ALTER TABLE students ADD COLUMN department TEXT")

        conn.commit()
    finally:
        conn.close()

# ---------- Auto-restore daily attendance ----------
def auto_restore_today():
    today = date.today().isoformat()
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT id FROM students WHERE id NOT IN (
            SELECT student_id FROM attendance WHERE date=?
        )
    """, (today,))
    for r in cur.fetchall():
        cur.execute("INSERT OR IGNORE INTO attendance(student_id, date, time, status) VALUES (?, ?, ?, ?)",
                    (r["id"], today, "", "Absent"))
    conn.commit(); conn.close()

# ---------- Student helpers ----------
def add_student(name, roll, department, email):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("INSERT INTO students(name, roll, department, email) VALUES (?, ?, ?, ?)",
                (name, roll, department, email))
    conn.commit(); sid = cur.lastrowid; conn.close()
    auto_restore_today()
    return sid

def get_students(dept_filter=None):
    conn = get_db_connection(); cur = conn.cursor()
    if dept_filter and dept_filter != "All":
        cur.execute("SELECT * FROM students WHERE department=?", (dept_filter,))
    else:
        cur.execute("SELECT * FROM students")
    rows = cur.fetchall(); conn.close(); return rows

def get_departments():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT DISTINCT department FROM students")
    arr = [r[0] for r in cur.fetchall() if r[0]]
    conn.close()
    return ["All"] + sorted(set(arr))

# ---------- Attendance helpers ----------
def _parse_time_str(ts):
    try:
        return datetime.strptime(ts, "%H:%M:%S").time()
    except Exception:
        return None

def mark_attendance_db(student_id, mark_time=None):
    if mark_time is None:
        mark_time = datetime.now().time()
    today = date.today().isoformat()
    cur_time = mark_time.strftime("%H:%M:%S")
    late_thresh_str = get_config("late_threshold") or "09:15:00"
    late_thresh = _parse_time_str(late_thresh_str) or dtime(9,15,0)
    try:
        cur_tobj = _parse_time_str(cur_time) or dtime(mark_time.hour, mark_time.minute, mark_time.second)
    except Exception:
        cur_tobj = dtime(mark_time.hour, mark_time.minute, mark_time.second)
    status = "Late" if cur_tobj > late_thresh else "Present"
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT id FROM attendance WHERE student_id=? AND date=?", (student_id, today))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE attendance SET time=?, status=? WHERE id=?", (cur_time, status, row["id"]))
    else:
        cur.execute("INSERT INTO attendance(student_id, date, time, status) VALUES (?, ?, ?, ?)",
                    (student_id, today, cur_time, status))
    conn.commit(); conn.close()
    return status

def get_attendance_rows(date_filter=None, dept_filter=None):
    conn = get_db_connection(); cur = conn.cursor()
    q = """
        SELECT a.id, a.student_id, s.name, s.roll, s.department,
               a.date, a.time, a.status, s.email
        FROM attendance a JOIN students s ON a.student_id=s.id
    """
    conds, params = [], []
    if date_filter: conds.append("a.date=?"); params.append(date_filter)
    if dept_filter and dept_filter != "All": conds.append("s.department=?"); params.append(dept_filter)
    if conds: q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY s.department, s.name"
    try:
        if params:
            cur.execute(q, tuple(params))
        else:
            cur.execute(q)
    except Exception as e:
        print("SQL execution error:", e)
        print("Query:", q)
        print("Params:", params)
        raise
    rows = cur.fetchall(); conn.close(); return rows

# ---------- Exports ----------
def export_to_csv(filename, rows):
    header = ["id","student_id","name","roll","department","date","time","status","email"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([r["id"], r["student_id"], r["name"], r["roll"], r["department"],
                        r["date"], r["time"], r["status"], r["email"]])

def export_to_excel(filename, rows):
    if not PANDAS_AVAILABLE:
        export_to_csv(filename.replace(".xlsx",".csv"), rows)
        return False
    pd.DataFrame([dict(r) for r in rows]).to_excel(filename, index=False)
    return True

def export_to_pdf(filename, rows, title="Attendance Report"):
    if not REPORTLAB_AVAILABLE:
        return False
    c = canvas.Canvas(filename, pagesize=letter)
    W, H = letter
    c.setFont("Helvetica-Bold", 14); c.drawString(40, H-40, title)
    c.setFont("Helvetica", 10)
    y = H-70
    heads = ["Name","Roll","Dept","Date","Time","Status"]; xs = [40, 200, 260, 350, 430, 500]
    for i, h in enumerate(heads): c.drawString(xs[i], y, h)
    y -= 18
    for r in rows:
        if y < 60: c.showPage(); y = H-40
        c.drawString(xs[0], y, str(r["name"]))
        c.drawString(xs[1], y, str(r["roll"] or ""))
        c.drawString(xs[2], y, str(r["department"] or ""))
        c.drawString(xs[3], y, str(r["date"]))
        c.drawString(xs[4], y, str(r["time"] or ""))
        c.drawString(xs[5], y, str(r["status"]))
        y -= 16
    c.save(); return True

# ---------- Face hooks ----------
def _ensure_opencv_available():
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV not installed; install opencv-contrib-python for face recognition.")
    if not CV2_CONTRIB:
        raise RuntimeError("LBPH requires opencv-contrib-python; reinstall with: pip install opencv-contrib-python")

if CV2_AVAILABLE:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _equalize(img_gray):
        try:
            return cv2.equalizeHist(img_gray)
        except Exception:
            return img_gray

    def capture_face_samples_for(name, samples=5, cam_index=0):
        _ensure_opencv_available()
        user_dir = os.path.join(DATASET_DIR, name); os.makedirs(user_dir, exist_ok=True)
        cam = cv2.VideoCapture(cam_index)
        if not cam.isOpened(): raise RuntimeError("Camera not available")
        # Warm up
        for _ in range(5):
            cam.read()
        count = 0
        while True:
            ok, frame = cam.read()
            if not ok: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)
            )
            for (x,y,w,h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face = _equalize(face)
                count += 1
                cv2.imwrite(os.path.join(user_dir, f"{count}.jpg"), face)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow("Capture Faces (press q to stop)", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): break
            if count >= samples: break
        cam.release(); cv2.destroyAllWindows(); return count

    def train_lbph_model():
        _ensure_opencv_available()
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, labels, label_map, cur_label = [], [], {}, 0
        for person in sorted(os.listdir(DATASET_DIR)):
            pdir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(pdir): continue
            label_map[cur_label] = person
            for f in sorted(os.listdir(pdir)):
                if f.startswith("."): continue
                if not f.lower().endswith((".png",".jpg",".jpeg",".bmp",".pgm")): continue
                path = os.path.join(pdir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (200,200))
                img = _equalize(img)
                faces.append(img); labels.append(cur_label)
            cur_label += 1
        if not faces: raise RuntimeError("No faces found to train")
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        recognizer.write(MODEL_FILE)
        with open(LABEL_MAP_FILE, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            for k, v in label_map.items(): w.writerow([k, v])
        return len(label_map)

    def load_label_map():
        m = {}
        if os.path.exists(LABEL_MAP_FILE):
            with open(LABEL_MAP_FILE, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        try:
                            m[int(row[0])] = row[1]
                        except Exception:
                            pass
        return m

    def recognize_and_mark(conf_threshold=None, cam_index=0, timeout_sec=20):
        _ensure_opencv_available()
        if not os.path.exists(MODEL_FILE): raise RuntimeError("Model not trained")
        if conf_threshold is None:
            try:
                conf_threshold = int(get_config("lbph_threshold") or 85)
            except Exception:
                conf_threshold = 85
        rec = cv2.face.LBPHFaceRecognizer_create(); rec.read(MODEL_FILE)
        lmap = load_label_map()
        cam = cv2.VideoCapture(cam_index)
        if not cam.isOpened(): raise RuntimeError("Camera not available")
        # Warm up frames
        for _ in range(5):
            cam.read()
        start = datetime.now().timestamp()
        while True:
            ok, frame = cam.read()
            if not ok: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)
            )
            for (x,y,w,h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                roi = _equalize(roi)
                label, conf = rec.predict(roi)
                # Lower conf == better match
                if conf < conf_threshold:
                    name = lmap.get(label, "Unknown")
                    conn = get_db_connection(); cur = conn.cursor()
                    cur.execute("SELECT id FROM students WHERE name=?", (name,))
                    r = cur.fetchone(); conn.close()
                    if r:
                        status = mark_attendance_db(r["id"])
                        cam.release(); cv2.destroyAllWindows()
                        return name, status
            if datetime.now().timestamp() - start > timeout_sec:
                break
        cam.release(); cv2.destroyAllWindows()
        return None
else:
    def capture_face_samples_for(name, samples=5, cam_index=2): raise RuntimeError("OpenCV not available")
    def train_lbph_model(): raise RuntimeError("OpenCV not available")
    def recognize_and_mark(conf_threshold=70, cam_index=0, timeout_sec=20): raise RuntimeError("OpenCV not available")

# ---------- GUI ----------
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Offline AI-Based Attendance System with Face Recognition & Analytics
")
        self.root.geometry("1100x700")
        set_window_icon(root)

        if TB_AVAILABLE:
            theme = "darkly" if (get_config("theme") or "light") == "dark" else "flatly"
            self.style = tb.Style(theme=theme); base = tb.Frame(root)
        else:
            self.style = ttk.Style(); base = ttk.Frame(root)
        base.pack(fill="both", expand=True)

        self.sidebar = ttk.Frame(base, width=220); self.sidebar.pack(side="left", fill="y")
        self.content = ttk.Frame(base); self.content.pack(side="right", fill="both", expand=True)

        for i, (t, cmd) in enumerate([
            ("Dashboard", self.show_dashboard),
            ("Students", self.show_students),
            ("Attendance", self.show_attendance),
            ("Reports", self.show_reports),
            ("Settings", self.show_settings),
        ]):
            ttk.Button(self.sidebar, text=t, command=cmd).pack(fill="x", padx=8, pady=(8 if i==0 else 4))

        ttk.Button(self.sidebar, text="Toggle Theme", command=self.toggle_theme).pack(fill="x", padx=8, pady=8)
        ttk.Label(self.sidebar, text="Face Actions:", font=("Arial", 10, "bold")).pack(pady=(16,4))
        ttk.Button(self.sidebar, text="Capture Face (selected)", command=self.capture_face_for_selected).pack(fill="x", padx=8, pady=4)
        ttk.Button(self.sidebar, text="Train Model", command=self.train_model).pack(fill="x", padx=8, pady=4)
        ttk.Button(self.sidebar, text="Start Recognition (mark)", command=self.start_recognition_thread).pack(fill="x", padx=8, pady=4)

        self.frames = {k: ttk.Frame(self.content) for k in ("dashboard","students","attendance","reports","settings")}
        self.student_tree = None

        auto_restore_today()
        try:
            if get_config("auto_send_email") == "1":
                self.schedule_email_at_config_time()
        except Exception:
            pass

        self.show_dashboard()

    def _show_frame(self, name):
        for k, f in self.frames.items(): f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

    # Pages
    def show_dashboard(self):
        self._show_frame("dashboard")
        f = self.frames["dashboard"]
        for w in f.winfo_children(): w.destroy()
        ttk.Label(f, text="Dashboard", font=("Arial", 16, "bold")).pack(pady=8)
        today = date.today().isoformat()
        rows = get_attendance_rows(date_filter=today)
        total = len(rows)
        present = len([r for r in rows if r["status"] in ("Present","Late")])
        absent = len([r for r in rows if r["status"] == "Absent"])
        percent = (present/total*100) if total else 0
        for txt in [f"Date: {today}", f"Total students: {total}", f"Present: {present}",
                    f"Absent: {absent}", f"Attendance %: {percent:.2f}%"]:
            ttk.Label(f, text=txt).pack()
        bf = ttk.Frame(f); bf.pack(pady=10)
        ttk.Button(bf, text="View Attendance Page", command=self.show_attendance).pack(side="left", padx=6)
        ttk.Button(bf, text="Send Absent Emails Now", command=self.send_absent_emails_and_notify_ui).pack(side="left", padx=6)

    def show_students(self):
        self._show_frame("students")
        f = self.frames["students"]
        for w in f.winfo_children(): w.destroy()
        ttk.Label(f, text="Students", font=("Arial", 14, "bold")).pack(pady=8)
        form = ttk.Frame(f); form.pack(pady=6, padx=10, fill="x")
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w"); name_e = ttk.Entry(form); name_e.grid(row=0, column=1, sticky="ew")
        ttk.Label(form, text="Roll").grid(row=1, column=0, sticky="w"); roll_e = ttk.Entry(form); roll_e.grid(row=1, column=1, sticky="ew")
        ttk.Label(form, text="Department").grid(row=2, column=0, sticky="w"); dept_e = ttk.Entry(form); dept_e.grid(row=2, column=1, sticky="ew")
        ttk.Label(form, text="Email").grid(row=3, column=0, sticky="w"); email_e = ttk.Entry(form); email_e.grid(row=3, column=1, sticky="ew")
        form.columnconfigure(1, weight=1)
        def add():
            name = name_e.get().strip()
            if not name:
                messagebox.showerror("Error", "Name required"); return
            add_student(name, roll_e.get().strip(), dept_e.get().strip(), email_e.get().strip())
            messagebox.showinfo("Added", f"Student {name} added")
            name_e.delete(0,"end"); roll_e.delete(0,"end"); dept_e.delete(0,"end"); email_e.delete(0,"end")
            self.show_students()
        ttk.Button(form, text="Add Student", command=add).grid(row=4, column=0, columnspan=2, pady=8)
        cols = ("id","name","roll","department","email")
        container = ttk.Frame(f); container.pack(fill="both", expand=True, padx=10, pady=6)
        tree = ttk.Treeview(container, columns=cols, show="headings", height=12)
        for c in cols: tree.heading(c, text=c.title()); tree.column(c, width=120)
        yscroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        for s in get_students(): tree.insert("", "end", values=(s["id"], s["name"], s["roll"], s["department"], s["email"]))
        self.student_tree = tree

    def show_attendance(self):
        self._show_frame("attendance")
        f = self.frames["attendance"]
        for w in f.winfo_children(): w.destroy()
        ttk.Label(f, text="Attendance", font=("Arial", 14, "bold")).pack(pady=8)
        ctrl = ttk.Frame(f); ctrl.pack(fill="x", padx=10)
        ttk.Label(ctrl, text="Date (YYYY-MM-DD)").pack(side="left")
        date_e = ttk.Entry(ctrl); date_e.pack(side="left", padx=6); date_e.insert(0, date.today().isoformat())
        ttk.Label(ctrl, text="Department").pack(side="left", padx=(10,0))
        dept_cb = ttk.Combobox(ctrl, values=get_departments(), state="readonly"); dept_cb.pack(side="left", padx=6); dept_cb.set("All")
        stats_var = tk.StringVar(value="")
        ttk.Button(ctrl, text="Refresh", command=lambda: refresh()).pack(side="left", padx=6)
        ttk.Label(f, textvariable=stats_var).pack()
        cols = ("id","student_id","name","roll","department","date","time","status")
        tree = ttk.Treeview(f, columns=cols, show="headings", height=14)
        for c in cols: tree.heading(c, text=c.title()); tree.column(c, width=110)
        tree.pack(fill="both", expand=True, padx=10, pady=8)
        def refresh():
            dt, dept = date_e.get().strip(), dept_cb.get()
            rows = get_attendance_rows(date_filter=dt, dept_filter=dept)
            for r in tree.get_children(): tree.delete(r)
            for r in rows: tree.insert("", "end", values=(r["id"], r["student_id"], r["name"], r["roll"], r["department"], r["date"], r["time"], r["status"]))
            total = len(rows); present = len([r for r in rows if r["status"] in ("Present","Late")]); absent = len([r for r in rows if r["status"]=="Absent"])
            stats_var.set(f"Total {total}  Present {present}  Absent {absent}  % {(present/total*100) if total else 0:.2f}")
        def on_mark_present():
            sel = tree.selection()
            if not sel: messagebox.showerror("Error", "Select a row"); return
            student_id = tree.item(sel[0])["values"][1]
            status = mark_attendance_db(student_id)
            messagebox.showinfo("Marked", f"Marked {status} for student id {student_id}")
            refresh()
        def on_export():
            dt, dept = date_e.get().strip(), dept_cb.get()
            rows = get_attendance_rows(date_filter=dt, dept_filter=dept)
            if not rows: messagebox.showinfo("No data", "No attendance rows to export"); return
            dest = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV","*.csv"),("Excel","*.xlsx"),("PDF","*.pdf")])
            if not dest: return
            if dest.endswith(".csv"):
                export_to_csv(dest, rows); messagebox.showinfo("Exported", dest)
            elif dest.endswith(".xlsx"):
                ok = export_to_excel(dest, rows)
                messagebox.showinfo("Export", f"{'Exported' if ok else 'pandas not installed; CSV fallback'} -> {dest}")
            elif dest.endswith(".pdf"):
                ok = export_to_pdf(dest, rows)
                messagebox.showinfo("Export", dest if ok else "reportlab not installed; cannot export PDF")
        actions = ttk.Frame(f); actions.pack(pady=6)
        ttk.Button(actions, text="Mark Present (selected)", command=on_mark_present).pack(side="left", padx=6)
        ttk.Button(actions, text="Export", command=on_export).pack(side="left", padx=6)
        refresh()

    def show_reports(self):
        self._show_frame("reports")
        f = self.frames["reports"]
        for w in f.winfo_children(): w.destroy()
        ttk.Label(f, text="Reports", font=("Arial", 14, "bold")).pack(pady=8)
        ttk.Label(f, text="Select Date").pack()
        date_e = ttk.Entry(f); date_e.pack(); date_e.insert(0, date.today().isoformat())
        dept_cb = ttk.Combobox(f, values=get_departments(), state="readonly"); dept_cb.pack(); dept_cb.set("All")
        def gen():
            dt, dept = date_e.get().strip(), dept_cb.get()
            rows = get_attendance_rows(date_filter=dt, dept_filter=dept)
            total = len(rows); pres = len([r for r in rows if r["status"] in ("Present","Late")]); absn = len([r for r in rows if r["status"]=="Absent"])
            per = (pres/total*100) if total else 0
            messagebox.showinfo("Report", f"Date {dt}\nTotal: {total}\nPresent: {pres}\nAbsent: {absn}\nAttendance %: {per:.2f}")
        ttk.Button(f, text="Generate Summary", command=gen).pack(pady=6)
        ttk.Button(f, text="Export Rows CSV",
                   command=lambda: self._export_summary(date_e.get().strip(), dept_cb.get())).pack(pady=6)

    def _export_summary(self, dt, dept):
        rows = get_attendance_rows(date_filter=dt, dept_filter=dept)
        if not rows: messagebox.showinfo("No data", "No rows to export"); return
        dest = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not dest: return
        export_to_csv(dest, rows); messagebox.showinfo("Exported", dest)

    def show_settings(self):
        self._show_frame("settings")
        f = self.frames["settings"]
        for w in f.winfo_children(): w.destroy()
        ttk.Label(f, text="Settings", font=("Arial", 14, "bold")).pack(pady=8)
        ttk.Label(f, text="Late threshold (HH:MM:SS)").pack(); late_e = ttk.Entry(f); late_e.pack(); late_e.insert(0, get_config("late_threshold") or "09:15:00")
        ttk.Label(f, text="Absent email send time (HH:MM:SS)").pack(); send_e = ttk.Entry(f); send_e.pack(); send_e.insert(0, get_config("absent_notify_time") or "18:00:00")
        ttk.Label(f, text="SMTP Server").pack(); smtp_e = ttk.Entry(f); smtp_e.pack(); smtp_e.insert(0, get_config("smtp_server") or "smtp.gmail.com")
        ttk.Label(f, text="SMTP Port").pack(); port_e = ttk.Entry(f); port_e.pack(); port_e.insert(0, get_config("smtp_port") or "587")
        ttk.Label(f, text="Sender Email").pack(); sender_e = ttk.Entry(f); sender_e.pack(); sender_e.insert(0, get_config("email_sender") or "")
        ttk.Label(f, text="Sender Password (app password recommended)").pack(); pw_e = ttk.Entry(f, show="*"); pw_e.pack(); pw_e.insert(0, get_config("email_password") or "")
        ttk.Label(f, text="LBPH threshold (lower=stricter)").pack(); lbph_e = ttk.Entry(f); lbph_e.pack(); lbph_e.insert(0, get_config("lbph_threshold") or "85")
        auto_var = tk.IntVar(value=1 if get_config("auto_send_email") == "1" else 0)
        ttk.Checkbutton(f, text="Auto send absent emails daily", variable=auto_var).pack(pady=6)
        def save():
            set_config("late_threshold", late_e.get().strip())
            set_config("absent_notify_time", send_e.get().strip())
            set_config("smtp_server", smtp_e.get().strip())
            set_config("smtp_port", port_e.get().strip())
            set_config("email_sender", sender_e.get().strip())
            set_config("email_password", pw_e.get().strip())
            set_config("lbph_threshold", lbph_e.get().strip())
            set_config("auto_send_email", "1" if auto_var.get() else "0")
            messagebox.showinfo("Saved", "Settings saved")
            if auto_var.get(): self.schedule_email_at_config_time()
        ttk.Button(f, text="Save Settings", command=save).pack(pady=8)
        ttk.Button(f, text="Send Absent Emails Now", command=self.send_absent_emails_and_notify_ui).pack(pady=4)

    # Actions
    def toggle_theme(self):
        if not TB_AVAILABLE:
            messagebox.showinfo("Theme", "ttkbootstrap not installed.")
            return
        cur = self.style.theme.name
        new = "darkly" if "flatly" in cur else "flatly"
        self.style.theme_use(new)
        set_config("theme", "dark" if "darkly" in new else "light")

    def capture_face_for_selected(self):
        if self.student_tree is None:
            messagebox.showerror("Select", "Open Students page and select a student first")
            return
        sel = self.student_tree.selection()
        if not sel:
            messagebox.showerror("Select", "Select a student in the list first")
            return
        sid, name = self.student_tree.item(sel[0])["values"][:2]
        if not CV2_AVAILABLE:
            messagebox.showerror("OpenCV missing", "Install opencv-contrib-python to capture faces")
            return
        try:
            cnt = capture_face_samples_for(name)
            messagebox.showinfo("Captured", f"Captured {cnt} images for {name}")
        except Exception as e:
            messagebox.showerror("Capture error", str(e))

    def train_model(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Missing", "OpenCV not installed; cannot train")
            return
        try:
            n = train_lbph_model()
            messagebox.showinfo("Trained", f"Trained model with {n} people")
        except Exception as e:
            messagebox.showerror("Training error", str(e))

    def start_recognition_thread(self):
        # Run recognition off the UI thread; marshal UI updates back safely
        def worker():
            try:
                res = recognize_and_mark()
                def show():
                    if res:
                        name, status = res
                        msg = f"Recognized: {name}. Attendance marked as {status}."
                        messagebox.showinfo("Recognized", msg)
                        if TTS_AVAILABLE: tts_say(msg)
                    else:
                        msg = "Sorry, face not recognized or session timed out. Please try again."
                        messagebox.showinfo("Not recognized", msg)
                        if TTS_AVAILABLE: tts_say(msg)
                self.root.after(0, show)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Recognition error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def send_absent_emails_and_notify_ui(self):
        res = send_absent_emails()
        if "error" in res:
            messagebox.showerror("Email error", res["error"])
        else:
            messagebox.showinfo("Email report", f"Sent: {res.get('sent',0)}  Skipped: {res.get('skipped',0)}")

    def schedule_email_at_config_time(self):
        t = (get_config("absent_notify_time") or "18:00:00").split(":")
        try:
            h, m, s = [int(x) for x in t]
        except Exception:
            h, m, s = 18, 0, 0
        now = datetime.now()
        target = datetime(now.year, now.month, now.day, h, m, s)
        if target < now: target += timedelta(days=1)
        delay = int((target - now).total_seconds() * 1000)
        def send_then_reschedule():
            try:
                self.send_absent_emails_and_notify_ui()
            finally:
                self.root.after(24*3600*1000, send_then_reschedule)
        self.root.after(delay, send_then_reschedule)

# ---------- Email (stub, optional) ----------
def send_absent_emails():
    # Minimal guardrails; fill as needed for real SMTP sending
    today = date.today().isoformat()
    rows = get_attendance_rows(date_filter=today)
    absent_rows = [r for r in rows if r["status"] == "Absent" and r["email"]]
    sender = get_config("email_sender") or ""
    pw = get_config("email_password") or ""
    if not sender or not pw:
        return {"sent": 0, "skipped": len(absent_rows), "error": "Email not configured"}
    # Implement SMTP as needed; returning stubbed summary
    return {"sent": 0, "skipped": len(absent_rows)}

# ---------- Main ----------
if __name__ == "__main__":
    init_db()
    upgrade_db()

    if TB_AVAILABLE:
        root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()
    set_window_icon(root)

    app = AttendanceApp(root)
    root.mainloop()

