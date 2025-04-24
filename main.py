from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
import datetime

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Connect to SQLite database
conn = sqlite3.connect("store_analytics.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_traffic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        people_count INTEGER
    )
''')
conn.commit()
conn.close()


@app.get("/")
def home(request: Request):
    """Render the HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...)):
    """Handle video file upload and perform AI-based analysis."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv8 object detection
    results = model(frame)
    num_people = sum(1 for _ in results[0].boxes)  # Count detected people

    # Save to database
    conn = sqlite3.connect("store_analytics.db")
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?, ?)", (timestamp, num_people))
    conn.commit()
    conn.close()

    return JSONResponse(content={"people_count": num_people, "timestamp": timestamp})


@app.get("/customer_traffic/")
def get_customer_traffic():
    """Retrieve stored customer traffic data from the database."""
    conn = sqlite3.connect("store_analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customer_traffic ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return {"customer_traffic": [{"id": row[0], "timestamp": row[1], "people_count": row[2]} for row in data]}


# Run the app using: uvicorn main:app --reload
