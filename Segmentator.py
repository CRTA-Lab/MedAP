import cv2
import torch
import numpy as np

from segment_anything import SamPredictor
from SAM_Med2D.segment_anything import sam_model_registry as sammed_model_registry
from argparse import Namespace

from Segmentation_helper import create_directory


#Segmentator class
class SAM_Segmentator:
    def __init__(self, image, file_name,rect_start, rect_end, real_image_shape, prompt_state):
        """
        Segmentator instance.

        Args: 
            image - an image to be segmented (zoomed)
            file_name - the image name stored from the image file
            rect_start - the coordinates (x,y) of starting point of created bounding box
            rect_end - the coordinated (x,y) of endinng points of created bouding box
            image_shape - real image shape [width, height]
            prompt_state - the prompt that is used ( Point, Box, ...)
        Output:
            No output.
        """

        self.image=image
        #self.image_name=f"{file_name.split('.')[0]}.png"
        self.image_name=file_name
        self.rect_start=rect_start
        self.rect_end=rect_end
        x1,y1=self.rect_start
        x2,y2=self.rect_end
        self.image_shape=real_image_shape
        self.prompt_state=prompt_state
        ####************ DEVELOPER STUFF ************######

        # Function to setup the directories to store the annotation results
        self.setup_directories()

        # if cuda is not available, use cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Setup model
        #self.sam_checkpoint="sam_vit_b_01ec64.pth"
        self.sam_checkpoint="sam-med2d_b.pth"
        self.model_type="vit_b"

        #SAM-Med2D
        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = self.sam_checkpoint
        self.sam = sammed_model_registry[self.model_type](args).to(self.device)

        #self.sam=sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

        #Setup the predictor
        self.predictor=SamPredictor(self.sam)

        #Process image to produce image embedding that will be used for mask prediction
        self.predictor.set_image(self.image)

        #Set the point on the object you want to detect
        if self.prompt_state=="Box":

            self.input_point = np.array([[self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1]]])
            self.input_label = np.array([1])

        if self.prompt_state=="Point":

            self.input_point = np.array([[self.rect_start[0], self.rect_start[1]]])
            self.input_label = np.array([1])
        #Perform the prediction on specified image
        self.predict()

        #Save the predictions
        self.save_prediction()
        
        self.semgentation_successful=True
        

    def setup_directories(self) -> None:
        '''
        Creates directories needed for setup, masks, annotations
        and txt files.
        '''
        create_directory('AnnotatedDataset')
        create_directory('AnnotatedDataset/masks')
        create_directory('AnnotatedDataset/annotations')
        create_directory('AnnotatedDataset/txt')
        
    def predict(self) -> None:
        '''
        Performs prediction on the specified image.
        '''
        
        if self.prompt_state=="Box":
            self.masks, self.scores, self.logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=self.input_point[None, :],
                multimask_output=False,
            )

        elif self.prompt_state=="Point":
            self.masks, self.scores, self.logits = self.predictor.predict(
                point_coords=self.input_point,
                point_labels=self.input_label,
                multimask_output=False,
            )
        
    def save_prediction(self) -> None:

        #Create YOLO-compatible annotation
        h,w =self.masks[0].shape
        y,x =np.where(self.masks[0]>0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        #YOLO format: class_id x_center  y_center width height (normalized)
        x_center=(x_min+x_max)/2.0/w
        y_center=(y_min+y_max)/2.0/h
        bbox_width=(x_max-x_min)/w
        bbox_height=(y_max-y_min)/h

        class_id=0

        #Segment annotation stored
        self.yolo_annotation=f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

        #Create a mask of annotated part in real size      
        self.resized_mask = cv2.resize(self.masks[0].astype(np.uint8), (self.image_shape[0], self.image_shape[1]), interpolation=cv2.INTER_NEAREST)

        #Find mask contours on specified mask
        self.contours, self.hierarchy = cv2.findContours(self.masks[0].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #Create an original image with mask border
        self.image_with_contours=self.image.copy()
        cv2.drawContours(self.image_with_contours, self.contours, -1, (255,255,255), 2)

        # Create a colored overlay for the mask
        colored_mask = np.zeros_like(self.image, dtype=np.uint8)
        colored_mask[self.masks[0] > 0] = [200, 200, 255]  # Green color for the mask (adjust the color as needed)

        # Apply the colored mask onto the original image
        self.annotated_image= cv2.addWeighted(self.image, 1.0, colored_mask, 0.5, 0)
        self.annotated_image= cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
        # self.annotated_image_real_size= cv2.resize(self.annotated_image,(self.image_shape[0], self.image_shape[1]))
        # self.annotated_image_real_size= cv2.cvtColor(self.annotated_image_real_size, cv2.COLOR_BGR2RGB)
        self.annotated_image_real_size= cv2.resize(self.image_with_contours,(self.image_shape[0], self.image_shape[1]))

    #Function that removes the part of the mask that is specified using bounding box
    def edit_segmentation(self, rect_start, rect_end) -> None:
        self.rect_start=rect_start
        self.rect_end=rect_end
        self.resized_mask[self.rect_start[1]:self.rect_end[1], self.rect_start[0]:self.rect_end[0]]=0

        #Find mask contours on specified mask
        self.contours, self.hierarchy = cv2.findContours(self.resized_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #Create an original image with mask border
        self.image_with_contours=self.image.copy()
        cv2.drawContours(self.image_with_contours, self.contours, -1, (255,255,255), 2)

        self.annotated_image_real_size= cv2.resize(self.image_with_contours,(self.image_shape[0], self.image_shape[1]))
