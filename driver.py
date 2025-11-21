import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PUSHUP_DIR = os.path.join(ROOT, 'pushup')
PUSHUP_SCRIPT = os.path.join(PUSHUP_DIR, 'main.py')
PLANCKS_DIR = os.path.join(ROOT, 'plancks')
PLANCKS_SCRIPT = os.path.join(PLANCKS_DIR, 'main.py')


def run_pushup():
    if not os.path.isfile(PUSHUP_SCRIPT):
        print(f"Pushup script not found: {PUSHUP_SCRIPT}")
        return

    print("Launching PushUp test (will open a GUI window). Press 'S' or ESC to stop.")
    # Use the same Python interpreter that launched this script
    try:
        subprocess.run([sys.executable, PUSHUP_SCRIPT], cwd=PUSHUP_DIR)
    except Exception as e:
        print(f"Failed to launch PushUp test: {e}")


def main_menu():
    while True:
        print("\n=== Test Driver ===")
        print("1) PushUp Test")
        print("2) Plancks Test")
        print("3) Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            run_pushup()
        elif choice == '2':
            # Launch plancks
            if not os.path.isfile(PLANCKS_SCRIPT):
                print(f"Plancks script not found: {PLANCKS_SCRIPT}")
            else:
                print("Launching Plancks test (will open a GUI window). Press 'S' or ESC to stop.")
                try:
                    subprocess.run([sys.executable, PLANCKS_SCRIPT], cwd=PLANCKS_DIR)
                except Exception as e:
                    print(f"Failed to launch Plancks test: {e}")
        elif choice == '3' or choice.lower() in ('q', 'exit'):
            print("Exiting.")
            break
        else:
            print("Invalid selection. Enter the number of the choice.")


if __name__ == '__main__':
    main_menu()
