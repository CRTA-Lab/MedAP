import subprocess
import sys
import customtkinter

def launch_script(script_path):
    # Start the requested segmentation GUI as a subprocess
    subprocess.Popen([sys.executable, script_path])

def main():
    # Create a simple popup menu using customtkinter
    root = customtkinter.CTk()
    root.title("MedAP Launcher")
    root.geometry("400x200")

    label = customtkinter.CTkLabel(root, text="Choose Medical Annotation Mode:", font=("Arial", 22))
    label.pack(pady=20)

    sam_button = customtkinter.CTkButton(root, text="Segment Anything (SAM)", font=("Arial", 20),
                                        command=lambda: (launch_script("Tkinter_annotation.py"), root.destroy()))
    sam_button.pack(pady=10)

    custom_button = customtkinter.CTkButton(root, text="Custom Model", font=("Arial", 20),
                                        command=lambda: (launch_script("Tkinter_contour_editor.py"), root.destroy()))
    custom_button.pack(pady=10)

    # Prevent closing without selection (optional)
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    root.mainloop()

if __name__ == "__main__":
    main()
