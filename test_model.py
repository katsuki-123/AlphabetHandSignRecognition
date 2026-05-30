# investigate_dataset.py
import os

def investigate_dataset():
    dataset_path = 'ASL_Alphabet_Dataset'
    print(f"🔍 Deep investigation of: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset directory not found!")
        return
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"\n📁 {item}/")
            
            # Check subdirectories
            subitems = os.listdir(item_path)
            print(f"   Contains {len(subitems)} items")
            
            # Show first 10 items and their types
            for subitem in subitems[:10]:
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    # It's a directory - count images inside
                    images = [f for f in os.listdir(subitem_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"   📁 {subitem}/ - {len(images)} images")
                else:
                    # It's a file
                    print(f"   📄 {subitem}")

if __name__ == "__main__":
    investigate_dataset()