import cv2
import numpy as np
from tkinter import Tk, Canvas, filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from constants import *

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.root.configure(bg=COLOUR_ROOT_BG)
        # Initialize variables
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.zoom_value = ZOOM_VALUE
        self.zoom_factor = ZOOM_FACTOR
        self.min_zoom = ZOOM_MIN
        self.max_zoom = ZOOM_MAX
        self.image_shape = None

        # Polygon drawing state
        self.drawing_polygon = False
        self.polygon_points = []

        # Styling settings for Tkinter
        style = ttk.Style()
        style.theme_use('alt')
        style.configure('TButton', background=COLOUR_TBUTTON_BG, foreground=COLOUR_TBUTTON_FG, font=('Helvetica', 10), padding=6)
        style.map('TButton', background=[('active', COLOUR_TBUTTON_BG_ACTIVE)])

        # GUI elements
        self.canvas = Canvas(root, width=1200, height=800, bg=COLOUR_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side='left', padx=10, pady=20)

        # Button frame
        button_frame = ttk.Frame(root)
        button_frame.pack(side="right", fill="y", padx=20)

        # Buttons
        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image, style="TButton")
        self.save_button = ttk.Button(button_frame, text="Save Image", command=self.save_image, style="TButton")
        
        self.draw_polygon_button = ttk.Button(button_frame, text="Draw Polygon", command=self.start_polygon_drawing, style="TButton")
        self.reset_polygon_button = ttk.Button(button_frame, text="Reset Polygon", command=self.reset_polygon, style="TButton")
        self.exit_button = ttk.Button(button_frame, text="Exit", command=root.quit, style="TButton")
        # Arrange buttons in the grid
        self.load_button.grid(row=0, column=0, pady=10, sticky="ew")
        self.save_button.grid(row=1, column=0, pady=10, sticky="ew")
        self.draw_polygon_button.grid(row=2, column=0, pady=10, sticky="ew")
        self.reset_polygon_button.grid(row=3, column=0, pady=10, sticky="ew")
        self.exit_button.grid(row=4, column=0, pady=10, sticky="ew")

        # Zoom controls
        second_frame = ttk.Frame(button_frame)
        second_frame.grid(row=5, column=0, pady=20, sticky="ew")
        self.zoom_in_button = ttk.Button(second_frame, text="Zoom In", command=self.zoom_in, style='TButton')
        self.zoom_out_button = ttk.Button(second_frame, text="Zoom Out", command=self.zoom_out, style='TButton')
        self.zoom_in_button.grid(row=1, column=0, padx=10, pady=20, sticky="ew")
        self.zoom_out_button.grid(row=1, column=1, padx=10, pady=20, sticky="ew")

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Double-1>", self.on_double_click)  # Bind double-click to end polygon

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpeg *.jpg *.png")])
        self.file_name = file_path.split("/")[-1]
        self.root.title(self.file_name)
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_shape = [self.image.shape[1], self.image.shape[0]]  # width, height
            self.original_image = self.image.copy()
            self.zoom_value = 1.0
            self.update_canvas()

    def start_polygon_drawing(self):
        """Start polygon drawing mode."""
        self.drawing_polygon = True
        self.polygon_points.clear()
        messagebox.showinfo("Polygon Mode", "Click on the canvas to add vertices. Double-click to complete.")

    def reset_polygon(self):
        """Clear the current polygon points."""
        if self.image is not None:
            self.polygon_points.clear()
            self.update_canvas()

    def on_mouse_click(self, event):
        """Handle mouse click to add points to the polygon."""
        if self.drawing_polygon and self.image is not None:
            # Add the clicked point to the polygon points
            x, y = int((event.x - self.x) / self.zoom_value), int((event.y - self.y) / self.zoom_value)
            self.polygon_points.append((x, y))
            self.update_canvas(draw_polygon=True)

    def on_double_click(self, event):
        """Complete the polygon when double-clicked."""
        if self.drawing_polygon:
            self.complete_polygon()

    def complete_polygon(self):
        """Complete the polygon and stop polygon drawing mode."""
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Polygon Error", "At least 3 points are needed to complete a polygon.")
            return

        self.drawing_polygon = False
        # Draw the completed polygon on the image in white with thicker lines
        cv2.polylines(self.image, [np.array(self.polygon_points)], isClosed=True, color=(255, 255, 255), thickness=3)
        self.update_canvas()
        print("Polygon points:", self.polygon_points)  # Optional: print or save these points

    def update_canvas(self, draw_polygon=False):
        """Update the canvas to show the image and any temporary drawings."""
        if self.image is not None:
            # Resize the image based on zoom factor
            zoomed_width = int(self.image.shape[1] * self.zoom_value)
            zoomed_height = int(self.image.shape[0] * self.zoom_value)
            zoomed_image = cv2.resize(self.image, (zoomed_width, zoomed_height))

            # Convert image for Tkinter
            self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(zoomed_image))
            # Calculate coordinates to center the image
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.x = (canvas_width - zoomed_width) // 2
            self.y = (canvas_height - zoomed_height) // 2
            self.canvas.create_image(self.x, self.y, anchor="nw", image=self.tk_image)

            # Draw temporary polygon while adding points
            if draw_polygon and self.polygon_points:
                scaled_points = [(int(px * self.zoom_value) + self.x, int(py * self.zoom_value) + self.y) for px, py in self.polygon_points]
                for i in range(1, len(scaled_points)):
                    self.canvas.create_line(scaled_points[i - 1], scaled_points[i], fill="white", width=3)
                if len(scaled_points) > 1:
                    self.canvas.create_line(scaled_points[-1], scaled_points[0], fill="white", width=3)  # Close the loop

    def zoom_in(self):
        self.zoom_value = min(self.zoom_value + self.zoom_factor, self.max_zoom)
        self.update_canvas()

    def zoom_out(self):
        self.zoom_value = max(self.zoom_value - self.zoom_factor, self.min_zoom)
        self.update_canvas()

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    root = Tk()
    app = ImageEditor(root)
    root.mainloop()
