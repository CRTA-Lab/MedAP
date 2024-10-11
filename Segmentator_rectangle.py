import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os
from Segmentation_helper import show_box,show_mask,show_points
import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

#Function to segment the object using bounding box
def segment_using_rectangle(image_path,annotated_image_name):
    #Setup image:
    image=cv2.imread(image_path)
    #Convert image color to RGB:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Store image name:
    image_name=f"{annotated_image_name}.png"
    ####************ DEVELOPER STUFF ************######

    #Setup directory where annotations will be stored.
    directory = "AnnotatedDataset"
    # Check if the directory already exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    #Setup folder where mask images will be stored.
    masks_directory="AnnotatedDataset/masks"
    # Check if the directory already exists
    if not os.path.exists(masks_directory):
        # If it doesn't exist, create it
        os.makedirs(masks_directory)
        print(f"Directory '{masks_directory}' created.")
    else:
        print(f"Directory '{masks_directory}' already exists.")
        
    #Stup folder where images with annotations will be stored.
    annotations_directory="AnnotatedDataset/annotations"
    # Check if the directory already exists
    if not os.path.exists(annotations_directory):
        # If it doesn't exist, create it
        os.makedirs(annotations_directory)
        print(f"Directory '{annotations_directory}' created.")
    else:
        print(f"Directory '{annotations_directory}' already exists.")

    #Setup where txt annotations will be stored.
    txt_directory="AnnotatedDataset/txt"
    # Check if the directory already exists
    if not os.path.exists(txt_directory):
        # If it doesn't exist, create it
        os.makedirs(txt_directory)
        print(f"Directory '{txt_directory}' created.")
    else:
        print(f"Directory '{txt_directory}' already exists.")

    #Setup operating device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Setup SAM model
    sam_checkpoint="/home/istrazivac6/LukaSiktar/Ultralytics/SAM/sam_vit_b_01ec64.pth"
    model_type="vit_b"

    sam=sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    #Set the SAM Predictor
    predictor=SamPredictor(sam)

    #Process image to produce image embedding that will be used for mask prediction
    predictor.set_image(image)

    #Set the point on the object you want to detect
    input_point = np.array([[None,None]])
    input_label = np.array([1])

    #Callback function that will be triggered on mouse events
    def mouse_callback(event, x,y, flags, param):
        nonlocal input_point
        global start_point
        #Check if the event was left button
        if event==cv2.EVENT_LBUTTONDOWN:
            #Store the coordinates in the list
            start_point=(x,y)
        if event==cv2.EVENT_LBUTTONUP:
            #Store the coordinated in the list
            end_point=(x,y)
            input_point=np.array([[start_point[0], start_point[1],end_point[0], end_point[1]]])
            print(input_point)


    #Create a window and set the mouse callback function to capture the click event
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Display the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all the windows created
    cv2.destroyAllWindows()

    # The coordinates of the point can be accessed using the 'input_point' array
    print(f"Coordinates of the selected point: {input_point}")


    #Predict the object using created bounding box
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_point[None, :],
        multimask_output=False,
    )

    #Create YOLO-compatible annotation
    h,w =masks[0].shape
    y,x =np.where(masks[0]>0)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    #YOLO format: class_id x_center  y_center width height (normalized)
    x_center=(x_min+x_max)/2.0/w
    y_center=(y_min+y_max)/2.0/h
    bbox_width=(x_max-x_min)/w
    bbox_height=(y_max-y_min)/h

    class_id=0
    yolo_annotation=f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
    
    #Store the txt annotation:
    annotation_save_path=f"/home/istrazivac6/LukaSiktar/Ultralytics/SAM/AnnotatedDataset/txt/annotation{annotated_image_name}.txt"
    with open(annotation_save_path, "w") as f:
        f.write(yolo_annotation)

    #Store the mask image:
    mask_save_path=f"/home/istrazivac6/LukaSiktar/Ultralytics/SAM/AnnotatedDataset/masks/{annotated_image_name}_mask.png"
    cv2.imwrite(mask_save_path, (masks[0] * 255).astype(np.uint8))
    #Show and store annotated image:
    output_image_path=f"/home/istrazivac6/LukaSiktar/Ultralytics/SAM/AnnotatedDataset/annotations/{image_name}"

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    #Print logs
    print(f"Mask saved at: {mask_save_path}")
    print(f"Annotated image saved at: {output_image_path}")
    print(f"YOLO annotation saved at: {annotation_save_path}")