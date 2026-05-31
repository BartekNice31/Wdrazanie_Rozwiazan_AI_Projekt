import zipfile
import os
import sys

def zip_single_file(file_path):
    if not os.path.exists(file_path):
        print(f"Błąd: Plik w podanej ścieżce '{file_path}' nie istnieje.")
        return
    
    zip_name = f"{file_path}.zip"
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file_path, os.path.basename(file_path))
        print(f"Sukces! Plik '{file_path}' został spakowany do '{zip_name}'.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

def zip_all_in_folder(folder_path, output_zip="archive.zip"):
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Pomijamy samo archiwum, jeśli jest w tym samym folderze
                    if file == output_zip:
                        continue
                    
                    file_path = os.path.join(root, file)
                    # Zachowujemy strukturę folderów względem folderu wejściowego
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
        print(f"Sukces! Wszystkie pliki z '{folder_path}' zostały spakowane do '{output_zip}'.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python zip_tool.py <nazwa_pliku>       - pakuje jeden plik")
        print("  python zip_tool.py .                  - pakuje wszystko w bieżącym folderze")
    else:
        path = sys.argv[1]
        if path == ".":
            zip_all_in_folder(".")
        elif os.path.isdir(path):
            zip_all_in_folder(path, f"{os.path.basename(os.path.normpath(path))}.zip")
        else:
            zip_single_file(path)
