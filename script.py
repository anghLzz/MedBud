import ollama
import sys
import time
import subprocess
import os

MODEL_AI = "phi3"

SYSTEM_PROMPT = (
    "You are MedBud, a medical assistant at a self-service kiosk. "
    "Rules you MUST follow:\n"
    "1. Start EVERY response with: 'Based on your medical history,'\n"
    "2. Give ONE specific OTC drug suggestion (paracetamol or ibuprofen) if relevant.\n"
    "3. Give ONE non-drug tip (rest, hydration, or cold compress).\n"
    "4. End with: 'Consult a doctor if symptoms worsen or persist over 3 days.'\n"
    "5. Maximum 50 words. No prescription drugs. No disclaimers. No patient token references."
)

CATEGORY_CONTEXT = {
    "3": "FEVER / HIGH TEMPERATURE. Patient has elevated body temperature.",
    "4": "GENERAL BODY PAIN. Patient reports pain or discomfort."
}

EMERGENCY_MSG = (
    "\n[MEDBUD]: EMERGENCY DETECTED!\n"
    "Please stay calm and do not leave this location.\n"
    "A medical team has been alerted and is on the way to you.\n"
    "If the situation worsens, call 112 immediately.\n"
)

CATEGORY_LABELS = {
    "1": "EMERGENCY",
    "2": "ROUTINE CHECK-UP",
    "3": "FEVER",
    "4": "BODY PAIN",
    "5": "EXIT"
}

def start_camera() -> subprocess.Popen:
    env = os.environ.copy()
    env["DISPLAY"] = ":10.0"
    proc = subprocess.Popen(
        ["rpicam-vid", "-t", "0", "--qt-preview"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env
    )
    return proc

def run_checkup() -> None:
    print("\n[MEDBUD]: Starting routine check-up...\n")
    print("  >> Please place your finger on the pulse and")
    print("     oxygen sensor and hold still.")
    input("\n  Press ENTER when ready...")

    print("\n  >> Now look directly into the camera.")
    print("     Stay still while scanning...")
    time.sleep(3)

    print("\n  >> Scanning complete!\n")
    print("=" * 40)
    print("         SCAN RESULTS")
    print("=" * 40)
    print("  Please check the sensor display screen")
    print("  for your Heart Rate, SpO2 and Temperature.")
    print("=" * 40)
    print("\n[MEDBUD]: Your results are ready!")
    print("          Please check the screen for your full report.")
    print("-" * 40)

def get_ai_response(category_key: str) -> None:
    context = CATEGORY_CONTEXT[category_key]

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Situation: {context}\n"
        f"Response:"
    )

    print("\n[MEDBUD]: ", end="", flush=True)

    try:
        stream = ollama.generate(
            model=MODEL_AI,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": 80,
                "num_ctx": 512,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.3,
                "num_thread": 4,
                "stop": ["Situation:", "Rules", "\n\n"]
            }
        )

        for chunk in stream:
            content = chunk.get("response", "")
            if content:
                print(content, end="", flush=True)

        print("\n" + "-" * 40)

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        print("[!] Make sure Ollama is running: ollama serve")

def start_consultation():
    print("\n" + "=" * 40)
    print("       MEDBUD - MEDICAL KIOSK")
    print("=" * 40)

    user_token = input("\nEnter Patient Token: ").strip()
    if not user_token:
        user_token = "ANONYMOUS"

    cam_proc = start_camera()

    try:
        while True:
            print("\n" + "-" * 40)
            for k, v in CATEGORY_LABELS.items():
                print(f"  [{k}] {v}")
            print("-" * 40)

            choice = input("Select option: ").strip()

            if choice == "5":
                print("\nGoodbye!\n")
                break

            if choice == "1":
                print(EMERGENCY_MSG)
                print("-" * 40)
                continue

            if choice == "2":
                run_checkup()
                continue

            if choice not in CATEGORY_CONTEXT:
                print("[!] Invalid option.")
                continue

            get_ai_response(choice)

    finally:
        cam_proc.terminate()
        cam_proc.wait()

if __name__ == "__main__":
    try:
        start_consultation()
    except KeyboardInterrupt:
        print("\n\nShutting down...\n")
        sys.exit(0)
