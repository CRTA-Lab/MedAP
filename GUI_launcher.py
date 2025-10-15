import customtkinter
from PIL import Image, ImageTk

def launch_script_and_wait(script_path, root):
    import subprocess, sys
    root.destroy()  # Destroy window immediately on selection
    process = subprocess.Popen([sys.executable, script_path])
    process.wait()

def main():
    root = customtkinter.CTk()
    root.title("MedAP Launcher")
    #root.attributes('-fullscreen', True)


    # Configure grid to center content and allow expansion
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)
    root.grid_columnconfigure(2, weight=1)

    # Create a frame centered in the grid
    frame = customtkinter.CTkFrame(root)
    frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)

    # Load and resize logo image
    image = Image.open("MedAP.png")
    image = image.resize((1536, 1024), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    # Image label centered in frame
    image_label = customtkinter.CTkLabel(frame, image=photo, text="")
    image_label.pack(pady=(0, 20))

    # Instruction label
    label = customtkinter.CTkLabel(frame, text="Choose Medical Annotation Mode:", font=("Arial", 20))
    label.pack(pady=10)

    # Buttons fill width of frame and spaced
    sam_button = customtkinter.CTkButton(frame, text="Segment Anything (SAM)", font=("Arial", 18),
                                        width=300,
                                        command=lambda: launch_script_and_wait("Tkinter_annotation.py", root))
    sam_button.pack(pady=10)

    custom_button = customtkinter.CTkButton(frame, text="Custom Model", font=("Arial", 18),
                                           width=300,
                                           command=lambda: launch_script_and_wait("Tkinter_contour_editor.py", root))
    custom_button.pack(pady=10)

    # Prevent closing without selection (optional)
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    root.mainloop()

if __name__ == "__main__":
    main()
