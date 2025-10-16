import requests
import time
import json
import traceback
import random

URL = "http://localhost:8000/predict"
DRIFT_URL = "http://localhost:8000/drift-stats"
RESET_URL = "http://localhost:8000/reference/reset"

NORMAL = [
    0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1,
    0.4, 0.7, -0.6, 0.3, 0.1, -0.4, 0.6, 0.2,
    -0.8, 0.5, 0.9, -0.2
]

DRIFTED = [
    5.0, -6.3, 3.8, 7.2, -9.5, 4.6, -3.1, 5.9,
    -6.8, 3.3, 4.7, -5.1, 6.0, -7.2, 4.8, -5.9,
    5.1, -8.0, 3.0, 6.6
]

def jitter_vec(v, eps=0.02):
    return [x + random.uniform(-eps, eps) for x in v]

def safe_post(payload):
    start = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=5)
        elapsed = (time.time() - start) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            mark = "⚠️" if data.get("drift_detected") else "✅"
            print(f"{mark} {elapsed:.0f} ms | pred={data['prediction']} prob={data['probability']:.3f} drift={data['drift_detected']}")
            return True
        else:
            print(f"❌ HTTP {resp.status_code} after {elapsed:.0f} ms — {resp.text[:120]}")
            return False

    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout (>5s) after {(time.time()-start):.2f}s")
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
        resp = requests.get(DRIFT_URL, timeout=5)
        stats = resp.json()
        print("\n📊 Drift stats:", json.dumps(stats, indent=2))
        return stats
    except Exception as e:
        print("⚠️ Не вдалося отримати drift-stats:", e)
        return None

def reset_reference():
    try:
        r = requests.post(RESET_URL, timeout=5)
        if r.status_code == 200:
            print("🔄 reference buffer cleared")
        else:
            print("⚠️ reset failed:", r.text)
    except Exception as e:
        print("⚠️ reset exception:", e)

def wait_until_ready(max_wait_s=25.0, poll_every_s=0.5):
    """Чекає, поки детектор ініціалізується або вичерпається таймаут."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        stats = print_drift_stats()
        if stats and stats.get("detector_initialized") is True:
            return True
        time.sleep(poll_every_s)
    return False

if __name__ == "__main__":
    print("🚀 Тест стабільності API з вимірюванням часу запиту\n")

    # (опц) чистий старт
    reset_reference()

    total = 0
    success = 0
    start_all = time.time()

    # 1) Формуємо reference з невеликим джиттером
    print("📥 Відправляємо 60 нормальних запитів (з джиттером) для формування reference...\n")
    for _ in range(60):
        ok = safe_post({"features": jitter_vec(NORMAL, eps=0.02)})
        total += 1
        success += int(ok)
        time.sleep(0.12)

    # 2) Чекаємо готовності детектора
    print("\n⏳ Очікуємо ініціалізацію детектора (бекґраунд)...")
    ready = wait_until_ready(max_wait_s=25.0, poll_every_s=0.5)
    if not ready:
        print("⚠️ Детектор не ініціалізувався в межах таймауту. Продовжимо все одно — але дрейф може не спрацювати.")
    else:
        print("✅ Детектор готовий, переходимо до перевірки дрейфу.")

    # 3) Відправляємо явний дрейф
    print("\n🔥 Відправляємо 12 аномальних запитів для перевірки drift...\n")
    for i in range(12):
        # Робимо дрейф більш природним: раз сильніше, раз трохи слабше
        eps = 0.8 if i % 2 == 0 else 0.4
        drifted = [x + random.uniform(-1.0, 1.0) * (1.0 + eps) for x in DRIFTED]
        safe_post({"features": drifted})
        time.sleep(0.18)

    # 4) Підсумок
    duration = time.time() - start_all
    print("\n==========================================")
    print(f"📦 Завершено: {success}/{total} успішних ({(success/total)*100:.1f}%)")
    print(f"🕐 Загальна тривалість: {duration:.2f} с")
    print("==========================================\n")

    print("📈 Підсумковий стан детектора:")
    print_drift_stats()
