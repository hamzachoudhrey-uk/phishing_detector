# Standalone phishing backend

Run this **on your PC** while your phone (Flutter app) is on the **same Wi‑Fi**.

## 1) Copy model files

Copy into `backend/models/`:

- `tfidf_vectorizer.pkl`
- `xgb_sms_model (1).pkl` (or your SMS model filename already supported in `main.py`)
- `url_phishing_model.pkl`

## 2) Install Python deps (once)

Open PowerShell **in this `backend` folder**:

```powershell
cd "D:\my project\my_app\backend"
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 3) Find your PC’s Wi‑Fi IP

```powershell
ipconfig
```

Use the **IPv4 Address** of your Wi‑Fi adapter (e.g. `192.168.1.50`).

## 4) Run the server (listen on all interfaces)

```powershell
cd "D:\my project\my_app\backend"
.\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Leave this window open.

## 5) Allow Windows Firewall (if the phone cannot connect)

```powershell
netsh advfirewall firewall add rule name="FastAPI SMS 8000" dir=in action=allow protocol=TCP localport=8000
```

## 6) Flutter app on the phone

1. Open **AI API settings** in the app (gear / ⋮ menu).
2. Set base URL to:

   `http://192.168.x.x:8000`

   (replace with **your** PC IPv4 from step 3)

3. Open an SMS → **Check this SMS**.

## Quick test from PC

```powershell
$body = @{ message = "Verify your bank account now" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/check_sms" -ContentType "application/json" -Body $body
```

```powershell
$urlBody = @{ url = "https://example.com/login" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/check_url" -ContentType "application/json" -Body $urlBody
```

## Endpoints

| Method | Path             | Body                 |
| ------ | ---------------- | -------------------- |
| GET    | `/`              | —                    |
| POST   | `/check_sms`     | `{"message":"..."}`  |
| POST   | `/check_message` | same as `/check_sms` |
| POST   | `/check_url`     | `{"url":"..."}`      |
