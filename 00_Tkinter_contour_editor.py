import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import splprep, splev
import math

class ContourEditor:
    def __init__(self, root):
        self.root=root
        self.root.title("Contour Editor")
        self.canvas=tk.Canvas(root, width=1200, height=800)
        self.canvas.pack()
        
        self.load_button=tk.Button(root, text="Load image", command=self.load_image)
        self.load_button.pack()
        self.save_button = tk.Button(root, text="Save", font=(12,12), command=self.save_image)
        self.save_button.pack()

        self.points=[]
        self.image=None
        self.contour=None
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)


    def load_image(self):
        file_path=filedialog.askopenfilename(filetypes=[("Image files"," *.jpeg *.jpg *.png")])
        if file_path:
            self.image=cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
            contours, _ =cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contour = max(contours, key=cv2.contourArea)
            self.resample_contour(100)
            self.draw_contour()
    def save_image(self):
        self.mask = np.zeros((self.image.shape[1], self.image.shape[0]), dtype=np.uint8)
        contour = self.points.reshape((-1, 1, 2)).astype(np.int32)
        print(self.points.shape)
        print(self.contour.shape)
        cv2.drawContours(self.mask, [contour], contourIdx=-1, color=(255, 255, ), thickness=2)

        cv2.imshow("mask",self.mask)
        cv2.waitKey(0)

    def resample_contour(self, num_points):
        contour=self.contour.reshape(-1,2)
        tck, u = splprep([contour.T[0],contour.T[1]], u=None, s=0, per=1)
        u_new=np.linspace(u.min(), u.max(), num_points)
        x_new, y_new=splev(u_new, tck, der=0)
        self.points=np.column_stack((x_new, y_new))

    def draw_contour(self):
        self.canvas.delete("all")
        for i, (x,y) in enumerate(self.points):
            x,y=int(x), int(y)
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tags=f"point_{i}")
            if i > 0:
                prev_x, prev_y=int(self.points[i-1][0]),int(self.points[i-1][1])
                self.canvas.create_line(prev_x, prev_y, x,y, fill="blue")
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def on_click(self, event):
        for i, (x,y) in enumerate(self.points):
            if abs(x - event.x) < 5 and abs(y - event.y) < 5:
                #print(f"Points: {self.points}")
                #print(f"{abs(x - event.x) < 5} , {abs(y - event.y)}")
                self.selected_point=i
                break

    def on_drag(self, event):
        if hasattr(self, "selected_point"):
            self.points[self.selected_point]=[event.x, event.y]
            self.draw_contour()
                    

if __name__ == "__main__":
    root=tk.Tk()
    app=ContourEditor(root)
    root.mainloop()