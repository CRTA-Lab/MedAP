import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from Segmentation_helper import show_mask, show_points, create_directory
import warnings

# import constants
from constants import *

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

#Function to segment the object using mouse and single point
def segment_using_mouse(image_path: str, annotated_image_name: str) -> None:
    #Setup image:
    image=cv2.imread(image_path)
    #Convert image color to RGB:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Store image name:
    image_name=f"{annotated_image_name}.png"
    ####************ DEVELOPER STUFF ************######

    # annotations
    create_directory(FOLDER_ANNOTATED)
    # masks
    create_directory(FOLDER_MASKS)
    # images with annotations
    create_directory(FOLDER_ANNOTATIONS)
    # txt annotations
    create_directory(FOLDER_TXT)

    # if cuda is available, use gpu
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup SAM model 
    sam_checkpoint="./sam_vit_b_01ec64.pth"
    model_type="vit_b"

    sam=sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Set the SAM Predictor
    predictor=SamPredictor(sam)

    # Process image to produce image embedding that will be used for mask prediction
    predictor.set_image(image)

    # Set the point and class on the object you want to detect 
    input_point = np.array([[None, None]])
    input_label = np.array([1])

    # Callback function that will be triggered on mouse events
    def mouse_callback(event, x,y, flags, param) -> None:
        nonlocal input_point
        #Check if the event was left button
        if event==cv2.EVENT_LBUTTONDOWN:
            #Store the coordinates in the list
            input_point=np.array([[x,y]])
            print(f"Point selected: {input_point}")

    # Create a window and set the mouse callback function to capture the click event
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Display the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # user can exit the program by pressing the 'BUTTON_EXIT'
        if cv2.waitKey(1) & 0xFF == ord(BUTTON_EXIT):
            break

    # Destroy all the windows created
    cv2.destroyAllWindows()

    # The coordinates of the point can be accessed using the 'input_point' array
    print(f"Coordinates of the selected point: {input_point}")


    # Predict the object using created points using mouse
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Create YOLO-compatible annotation
    h,w = masks[0].shape
    y,x = np.where(masks[0]>0)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # YOLO format: class_id x_center  y_center width height (normalized)
    x_center=(x_min+x_max)/2.0/w
    y_center=(y_min+y_max)/2.0/h
    bbox_width=(x_max-x_min)/w
    bbox_height=(y_max-y_min)/h

    class_id=0
    yolo_annotation=f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

    #Store the txt annotation:
    annotation_save_path=f"{FOLDER_TXT}/annotation{annotated_image_name}.txt"
    with open(annotation_save_path, "w") as f:
        f.write(yolo_annotation)

    #Store the mask image:
    mask_save_path=f"{FOLDER_MASKS}/{annotated_image_name}_mask.png"
    cv2.imwrite(mask_save_path, (masks[0] * 255).astype(np.uint8))

    #Show and store annotated image:
    output_image_path=f"{FOLDER_ANNOTATIONS}/{image_name}"

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Mask saved at: {mask_save_path}")
    print(f"Annotated image saved at: {output_image_path}")
    print(f"YOLO annotation saved at: {annotation_save_path}")