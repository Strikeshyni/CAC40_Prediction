import os
import shutil
import sys

def clean_old_models():
    models_dir = '/home/abel/personnal_projects/CAC40_stock_prediction/api_models'
    if not os.path.exists(models_dir):
        print("No api_models directory found.")
        return

    print(f"Cleaning up {models_dir}...")
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            print(f"Removing directory: {item_path}")
            shutil.rmtree(item_path)
        else:
            print(f"Removing file: {item_path}")
            os.remove(item_path)
    print("Cleanup complete. Old incompatible models removed.")

if __name__ == "__main__":
    clean_old_models()
