import os

def rename_images():
    print("=== Image Renamer ===")

    # Ask for folder path
    folder_path = input("Enter the folder path: ").strip()

    # Validate folder
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path!")
        return

    # Ask for base name
    base_name = input("Enter the new base name (e.g., image): ").strip()
    if not base_name:
        print("❌ Base name cannot be empty!")
        return

    # Allowed image extensions
    allowed_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    # List and sort all image files
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_exts)]
    images.sort()

    if not images:
        print("⚠️ No image files found in the folder!")
        return

    # Rename files
    for i, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]  # Keep original extension
        new_name = f"{base_name}{i}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        os.rename(src, dst)
        print(f"✅ Renamed: {filename} → {new_name}")

    print("\n🎉 Renaming complete!")

if __name__ == "__main__":
    rename_images()
