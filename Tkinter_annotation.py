import cv2
import numpy as np
from tkinter import Tk, Label, Canvas, filedialog, messagebox, simpledialog
from tkinter import ttk, Toplevel
from PIL import Image, ImageTk

from Segmentator import SAM_Segmentator

class ImageEditor:
     def __init__(self, root):
          self.root=root
          self.root.title("Image Editor")
          self.root.configure(bg='#2E2E2E')

          #Initialize variables
          self.image=None          #Loaded image (Image that is used to show created boxes)
          self.original_image=None #Original image (Image passed to the SAM model)
          self.tk_image=None       #Image format for canvas
          #Coordinated passed to the SAM model for segmentation purposes
          self.rect_start=None     #Coordinates of rect starting point (x,y)
          self.rect_end=None       #Coordinated of rect ending point (x,y)
          #The name specified from set of data created using SAM
          self.file_name=None
          #Zoom factors
          self.zoom_factor = 1.0
          self.min_zoom = 0.5
          self.max_zoom = 5.0
          #Original image dimensions
          self.image_shape=None
          #Prompt state
          self.prompt_state=None
          
          #Styling settings for Tkinter
          style = ttk.Style()
          style.theme_use('alt')  # Use a theme that supports customization
          style.configure('TButton', background='#444444', foreground='white', font=('Helvetica', 10), padding=6)
          style.map('TButton', background=[('active', '#555555')])

          #Create GUI elements
          self.canvas=Canvas(root, width=1200, height=800,  bg='#1E1E1E',highlightthickness=0)
          self.canvas.pack(side="left", padx=10, pady=20)  # Position the canvas on the left side

          # Create a frame for the buttons on the right side
          button_frame = ttk.Frame(root)
          button_frame.pack(side="right", fill="y", padx=20)

          #Buttons:
          # Group 1: Main actions (Load, Save, Reset, Perform Segmentation, Exit)
          self.load_button = ttk.Button(button_frame, text="Load image", command=self.load_image, style='TButton')
          self.save_button = ttk.Button(button_frame, text="Save image", command=self.save_image, style='TButton')
          self.reset_button = ttk.Button(button_frame, text="Reset Rectangle", command=self.reset_rectangle, style='TButton')
          self.perform_segmentation_button = ttk.Button(button_frame, text="Perform segmentation", command=self.perform_segmentation, style='TButton')
          self.exit_button = ttk.Button(button_frame, text="Exit", command=root.quit, style='TButton')

          # Arrange these buttons in the grid (1 column, multiple rows)
          self.load_button.grid(row=0, column=0, pady=10, sticky="ew")
          self.save_button.grid(row=1, column=0, pady=10, sticky="ew")
          self.reset_button.grid(row=2, column=0, pady=10, sticky="ew")
          self.perform_segmentation_button.grid(row=3, column=0, pady=10, sticky="ew")
          self.exit_button.grid(row=4, column=0, pady=10, sticky="ew")

          # Create a frame for other controls
          second_frame = ttk.Frame(button_frame)
          second_frame.grid(row=5, column=0, pady=20, sticky="ew")

          # Group 2: Zoom controls (Zoom In, Zoom Out)
          self.zoom_in_button = ttk.Button(second_frame, text="Zoom In", command=self.zoom_in, style='TButton')
          self.zoom_out_button = ttk.Button(second_frame, text="Zoom Out", command=self.zoom_out, style='TButton')

          # Arrange zoom buttons horizontally
          self.zoom_in_button.grid(row=0, column=0, padx=10, pady=20, sticky="ew")
          self.zoom_out_button.grid(row=0, column=1, padx=10, pady=20, sticky="ew")

          # Other controls
          self.edit_mask = ttk.Button(second_frame, text="Edit Mask",command=self.edit_mask, style="TButton")
          self.save_mask = ttk.Button(second_frame, text="Save Mask",command=self.save_mask, style="TButton")

          # Arrange input controll buttons horizontally
          self.edit_mask.grid(row=1, column=0, padx=10, sticky="ew")
          self.save_mask.grid(row=1, column=1, padx=10, sticky="ew")

          # Bind mouse events for rectangle drawing (unchanged)
          self.canvas1 = Canvas(root, width=500, height=500, bg='white')
          self.canvas.bind("<Button-1>", self.on_mouse_down)
          self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
          self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
          self.canvas.bind("<Motion>", self.on_mouse_move)  


     #Method that loads image file
     def load_image(self):
        file_path=filedialog.askopenfilename(filetypes=[("Image files"," *.jpeg *.jpg *.png")])     #Find any files to load
        self.file_name=file_path.split("/")[-1]   #Store the file name of image
        self.root.title(self.file_name)
        if file_path:
            #Load image with OpenCV
            self.image=cv2.imread(file_path)
            self.image=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            #Store the original image shape
            self.image_shape=[self.image.shape[1],self.image.shape[0]] #width, height
            #Copy the original image
            self.original_image=self.image.copy()
            self.zoom_factor=1.0
            self.update_canvas()
     
     #Zoom in method
     def zoom_in(self):
        """Zoom in by increasing the zoom factor."""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor += 0.1
        self.update_canvas()

     #Zoom out method
     def zoom_out(self):
        """Zoom out by decreasing the zoom factor."""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor -= 0.1
        self.update_canvas()

     #Define point prompt method
     def edit_mask(self):
          if self.segment is not None:
               self.segment.edit_segmentation(self.rect_start, self.rect_end)
               self.update_canvas_annotated_image()
               self.image=self.segment.image_with_contours.copy()


     #Define box prompt method
     def save_mask(self):
          pass

     #Mouse action methods:
     def on_mouse_down(self, event):
         if self.image is not None:
              self.rect_start=(event.x-self.x, event.y-self.y)

     def on_mouse_drag(self, event):
         if self.rect_start:
              self.rect_end=(event.x-self.x, event.y-self.y)
              self.update_canvas()

     def on_mouse_up(self, event):
         if self.rect_start:
              self.rect_end=(event.x-self.x, event.y-self.y)
              #self.draw_rectangle()
     
     def on_mouse_move(self, event):
         if self.image is not None:
             #Draw the cross on canvas
             self.update_canvas(crosshair=(event.x-self.x, event.y-self.y))

     #Update the canvas method
     def update_canvas(self, crosshair=None):
         if self.image is not None:
                # Resize the image based on the zoom factor
               zoomed_width = int(self.image.shape[1] * self.zoom_factor)
               zoomed_height = int(self.image.shape[0] * self.zoom_factor)
               self.zoomed_image = cv2.resize(self.image, (zoomed_width, zoomed_height))

               #Display image
               self.canvas.delete("all")
               self.tk_image=ImageTk.PhotoImage(image=Image.fromarray(self.zoomed_image))
               # Calculate coordinates to center the image
               canvas_width = self.canvas.winfo_width()
               canvas_height = self.canvas.winfo_height()
               self.x = (canvas_width - zoomed_width) // 2
               self.y = (canvas_height - zoomed_height) // 2
               #Display the image at central coordinates
               self.canvas.create_image(self.x,self.y,anchor="nw", image=self.tk_image)
                              
               #Draw rectangle on canvas
               if self.rect_start and self.rect_end:
                    x1,y1 = self.rect_start
                    x2,y2 = self.rect_end
                    self.canvas.create_rectangle(x1+self.x,y1+self.y,x2+self.x,y2+self.y, outline="red", width=2)
               #Display the cross for easier annotation
               if crosshair:
                    cx,cy=crosshair
                    canvas_width=self.canvas.winfo_width()
                    canvas_height=self.canvas.winfo_height()
                    self.canvas.create_line(0+self.x, cy+self.y, canvas_width+cx+self.x, cy+self.y, fill="white", dash=(2,2))
                    self.canvas.create_line(cx+self.x, 0+self.y, cx+self.x, canvas_height+cy+self.y, fill="white", dash=(2,2))

     #Update canvas with annotated image
     def update_canvas_annotated_image(self):
         if self.segment.annotated_image is not None:
               #Display image
               self.canvas.delete("all")
               # self.segment.annotated_image=cv2.cvtColor(self.segment.annotated_image, cv2.COLOR_BGR2RGB)
               # self.tk_image=ImageTk.PhotoImage(image=Image.fromarray(self.segment.annotated_image))
               # self.canvas.create_image(self.x,self.y,anchor="nw", image=self.tk_image)
               self.tk_image=ImageTk.PhotoImage(image=Image.fromarray(self.segment.image_with_contours))
               self.canvas.create_image(self.x,self.y,anchor="nw", image=self.tk_image)
           
     

     #Method that performs image segmentation
     def perform_segmentation(self):
          #Check if the prompt is selected
          # if self.prompt_state==None:
          #      messagebox.showinfo( "Select prompt", "Select the used prompt!")
          #Perform the segmentation if the image is uploaded
          if self.image is not None:
               #Set the string name of saved annotations
               self.file_name=simpledialog.askstring("Annotation", "Enter the filename (without extension):")
               #Check the prompt state based od starting and ending point
               if self.rect_start == self.rect_end:
                    self.prompt_state="Point"
               elif self.rect_start != self.rect_end:
                    self.prompt_state="Box"
               #Show the selected prompt
               messagebox.showinfo("Select prompt", f"Selected prompt is {self.prompt_state}!")

               self.segment=SAM_Segmentator(self.zoomed_image, self.file_name, self.rect_start, self.rect_end, self.image_shape, self.prompt_state)
               if self.segment.semgentation_successful:
                    self.update_canvas_annotated_image()
                    messagebox.showinfo( "Segmentation", "Image segmentated. Mask and txt saved successfully!")
                    self.query_user()

     #Query method to check if the user is satisfied with the annotation
     def query_user(self):
        self.image=self.segment.annotated_image_real_size

        self.query_box = Toplevel(self.root)
        self.query_box.title("Perform Segmentation")
        
        message = Label(self.query_box, text="Do you want to perform segmentation?")
        message.pack(pady=20)
        
        accept_button = ttk.Button(self.query_box, text="Accept", command=self.save_image)
        accept_button.pack(padx=20, pady=10)
        
        # Reject button
        reject_button = ttk.Button(self.query_box, text="Reject", command=self.reset_rectangle)
        reject_button.pack(padx=20)
     
        #self.update_canvas()

     #Save the image method
     def save_image(self):
          if self.image is not None:
               self.image=self.original_image
               
               self.rect_start=None
               self.rect_end=None
               annotation_save_path=f"AnnotatedDataset/txt/annotation{self.file_name}.txt"
               with open(annotation_save_path, "w") as f:
                    f.write(self.segment.yolo_annotation)

               mask_save_path=f"AnnotatedDataset/masks/{self.file_name}_mask.png"
               # Resize the mask to the original image size
               cv2.imwrite(mask_save_path, (self.segment.resized_mask * 255).astype(np.uint8))
               
               # Save the annotated image
               output_image_path = f"AnnotatedDataset/annotations/{self.file_name}_annotated.png"
               self.annotated_image_real_size= cv2.cvtColor(self.segment.annotated_image_real_size, cv2.COLOR_BGR2RGB)
               cv2.imwrite(output_image_path, self.annotated_image_real_size)
               self.image1= cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
               cv2.imwrite(output_image_path, self.image1)

               self.query_box.destroy()

     #Reset the rectangle method (in case the user is not satisfied with the bounding box)
     def reset_rectangle(self):
          if self.image is not None:
               # Reset the temporary image to the original
               self.image=self.original_image
               self.temp_image = self.image.copy()
               self.rect_start = None
               self.rect_end = None
               self.image=self.original_image.copy()
               self.update_canvas()
               self.query_box.destroy()

if __name__=="__main__":
       root=Tk()
       app=ImageEditor(root)
       root.mainloop()
