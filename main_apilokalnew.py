# ==================================================
# BACKEND API V10 (REAL CCTV + ESTIMASI GRAM + ANTI CRASH)
# ==================================================
from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import csv
from datetime import datetime

# --- [KONFIGURASI CCTV LOKAL] ---
RTSP_URL = "rtsp://admin:admin@192.168.30.249:554/V_ENC_000" 
CAPTURE_INTERVAL_BIG = 10 

# --- RUMUS SKRIPSI (Bisa diganti nanti sesuai data kebun) ---
# Misal: 1% kerusakan = 1.07 Gram daun hilang
GRAM_PER_PERCENT = 1.07 
# -------------------------------

MODEL_PATH_LOCAL = "unet_best_fold5.pth" 
OUTPUT_FOLDER = "hasil_deteksi_api"
DATA_FILE = "data_deteksi.csv"
CROP_SIZE = 512; STRIDE = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # BAPAK TAMBAHKAN KOLOM GRAM DISINI
        writer.writerow(["Tanggal", "Jam", "PosX", "PosY", "PersenRusak", "GramHilang", "Filename"])

# --- MODEL U-NET ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.d1 = DoubleConv(3, 64); self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256); self.d4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.uc4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.uc3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.uc2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.uc1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_class, 1)
    
    def forward(self, x):
        x1=self.d1(x); x2=self.d2(F.max_pool2d(x1, 2)); x3=self.d3(F.max_pool2d(x2, 2))
        x4=self.d4(F.max_pool2d(x3, 2)); xb=self.bottleneck(F.max_pool2d(x4, 2))
        x4=torch.cat([self.u4(xb), x4], 1); xu4=self.uc4(x4)
        x3=torch.cat([self.u3(xu4), x3], 1); xu3=self.uc3(x3)
        x2=torch.cat([self.u2(xu3), x2], 1); xu2=self.uc2(x2)
        x1=torch.cat([self.u1(xu2), x1], 1); xu1=self.uc1(x1)
        return self.final(xu1)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

is_running = False
model = None
latest_image_path = None

def load_model_unet():
    global model
    print(f"🔄 [INIT] Memuat Model U-Net ke {DEVICE}...")
    try:
        model_instance = UNet(3).to(DEVICE)
        if os.path.exists(MODEL_PATH_LOCAL):
            model_instance.load_state_dict(torch.load(MODEL_PATH_LOCAL, map_location=DEVICE))
            model_instance.eval()
            model = model_instance
            print("✅ [INIT] Model U-Net SIAP!")
        else: 
            print(f"❌ [ERROR] File model {MODEL_PATH_LOCAL} tidak ketemu! Taruh di folder yg sama ya Nok.")
    except Exception as e: 
        print(f"❌ [ERROR] Gagal load model: {e}")

def predict_crop(crop_bgr):
    if model is None: return None, None
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().squeeze().numpy()
    return img_rgb, pred_mask

def save_statistics(timestamp_obj, x, y, pred_mask, filename):
    try:
        damaged_pixels = np.sum(pred_mask == 2); total_pixels = pred_mask.size
        damage_pct = round((damaged_pixels / total_pixels) * 100, 2)
        
        # PERHITUNGAN GRAM DISINI
        gram_hilang = round(damage_pct * GRAM_PER_PERCENT, 2)
        
        tanggal = timestamp_obj.strftime('%Y-%m-%d')
        jam = timestamp_obj.strftime('%H:%M:%S')
        with open(DATA_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([tanggal, jam, x, y, damage_pct, gram_hilang, filename])
    except Exception as e: print(f"⚠️ Gagal menyimpan statistik: {e}")

@app.on_event("startup")
async def startup_event(): load_model_unet()

@app.post("/start-detection")
def start_detection(background_tasks: BackgroundTasks):
    global is_running
    if not is_running:
        is_running = True
        background_tasks.add_task(detection_loop)
    return {"status": "SUCCESS"}

@app.get("/latest-image")
def get_latest_image():
    global latest_image_path
    if latest_image_path and os.path.exists(latest_image_path):
        headers = {'Cache-Control': 'no-cache, no-store, must-revalidate'}
        return FileResponse(latest_image_path, media_type="image/jpeg", headers=headers)
    else: return Response(content="Belum ada data.", status_code=404)

@app.get("/image/{filename}")
def get_specific_image(filename: str):
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(filepath): return FileResponse(filepath, media_type="image/jpeg")
    return Response(content="Gambar tidak ditemukan.", status_code=404)

@app.get("/stats-today")
def get_stats_today():
    today_str = datetime.now().strftime('%Y-%m-%d')
    total_valid_scans = 0; total_damage_pct = 0; total_gram = 0
    high_damage_areas = []; times = []
    
    if not os.path.exists(DATA_FILE): 
        return JSONResponse(content={"date": today_str, "summary": "Belum ada pemindaian.", "average_damage_today": "0.00", "total_gram": "0.00", "hotspots": []})
    
    try:
        with open(DATA_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Tanggal"] != today_str: continue
                try:
                    pct = float(row["PersenRusak"])
                    gram = float(row.get("GramHilang", round(pct * GRAM_PER_PERCENT, 2)))
                    total_valid_scans += 1
                    total_damage_pct += pct
                    total_gram += gram
                    times.append(row["Jam"])
                    high_damage_areas.append({"pct": pct, "gram": gram, "filename": row["Filename"]})
                except ValueError: continue
        
        avg_damage = round(total_damage_pct / total_valid_scans, 2) if total_valid_scans > 0 else 0
        total_gram = round(total_gram, 2)
        high_damage_areas.sort(key=lambda item: item['pct'], reverse=True)
        
        # Format Waktu Scan
        if times:
            start_t = min(times)[:5] # Ambil HH:MM
            end_t = max(times)[:5]
            waktu_str = f"dari pukul {start_t} - {end_t}"
        else:
            waktu_str = "hari ini"

        summary = f"✅ Pemindaian selesai. Total {total_valid_scans} pemotongan gambar {waktu_str} telah diproses. Estimasi berat hilang: {total_gram} Gram."
        top_4 = high_damage_areas[:4] 

        return JSONResponse(content={
            "date": today_str, 
            "summary": summary, 
            "average_damage_today": f"{avg_damage:.2f}", 
            "total_gram": f"{total_gram:.2f}", 
            "hotspots": top_4
        })
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)

def detection_loop():
    global is_running, latest_image_path
    print(f"🚀 Menghubungkan ke CCTV...")
    try:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) 
        if not cap.isOpened(): 
            print("❌ Gagal konek CCTV! (Aman, web tetap jalan)")
            is_running = False
            return
            
        last_capture_time = time.time() - CAPTURE_INTERVAL_BIG
        while is_running:
            ret, frame = cap.read()
            if not ret: time.sleep(0.5); continue
            
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL_BIG:
                now_obj = datetime.now(); timestamp_str = now_obj.strftime('%Y%m%d_%H%M%S')
                big_frame = frame.copy(); h, w, _ = big_frame.shape
                
                for y in range(0, h, STRIDE):
                    for x in range(0, w, STRIDE):
                        if not is_running: break
                        x1=x; y1=y; x2=x1+CROP_SIZE; y2=y1+CROP_SIZE
                        if x2 > w: x2 = w; x1 = w - CROP_SIZE
                        if y2 > h: y2 = h; y1 = h - CROP_SIZE
                        
                        crop_bgr = big_frame[y1:y2, x1:x2]
                        img_rgb_512, pred_mask = predict_crop(crop_bgr)
                        
                        colors_bgr = np.array([[0,0,0], [0,255,0], [255, 255, 255]], dtype=np.uint8)
                        mask_display = colors_bgr[pred_mask]
                        img_display = cv2.cvtColor(img_rgb_512, cv2.COLOR_RGB2BGR)
                        combined_result = np.hstack((img_display, mask_display))

                        filename = f"scan_{timestamp_str}_pos_{x}_{y}.jpg"
                        filepath = os.path.join(OUTPUT_FOLDER, filename)
                        cv2.imwrite(filepath, combined_result)
                        
                        latest_image_path = filepath
                        save_statistics(now_obj, x, y, pred_mask, filename)
                last_capture_time = current_time
            else: time.sleep(0.1)
    except Exception as e: print(f"⚠️ Kesalahan sistem: {e}")
    finally: 
        if 'cap' in locals() and cap.isOpened(): cap.release()
        is_running = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)