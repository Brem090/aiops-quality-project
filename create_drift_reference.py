import requests
import time
import json
import traceback

URL = "http://localhost:8000/predict"
DRIFT_URL = "http://localhost:8000/drift-stats"

NORMAL = [
    0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1,
    0.4, 0.7, -0.6, 0.3, 0.1, -0.4, 0.6, 0.2,
    -0.8, 0.5, 0.9, -0.2
]

def safe_post(payload):
    """Відправляє запит і відловлює будь-які збої"""
    start = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=5)
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ {elapsed*1000:.0f} ms | pred={data['prediction']} prob={data['probability']:.3f} drift={data['drift_detected']}")
            return True
        else:
            print(f"❌ HTTP {resp.status_code} after {elapsed:.2f}s — {resp.text[:80]}")
            return False

    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout (>5s) after {time.time()-start:.2f}s")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection error: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc()
        return False

def print_drift_stats():
    try:
        stats = requests.get(DRIFT_URL, timeout=5).json()
        print("\n📊 Drift stats:", json.dumps(stats, indent=2))
    except Exception as e:
        print("⚠️ Не вдалося отримати drift-stats:", e)

if __name__ == "__main__":
    print("🚀 Тест стабільності API з вимірюванням часу запиту\n")
    total = 0
    success = 0
    start_all = time.time()

    for i in range(1, 51):
        ok = safe_post({"features": NORMAL})
        total += 1
        success += int(ok)
        time.sleep(0.2)

    duration = time.time() - start_all
    print("\n==========================================")
    print(f"📦 Завершено: {success}/{total} успішних ({(success/total)*100:.1f}%)")
    print(f"🕐 Загальна тривалість: {duration:.2f} с")
    print("==========================================\n")

    print_drift_stats()
