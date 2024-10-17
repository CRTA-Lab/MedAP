import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from Segmentation_helper import show_mask, show_points, create_directory
import warnings

from constants import *
from Shape import Shape

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

def segment_image(image_path: str, annotated_image_name: str, shape_type: Shape = Shape.RECTANGLE) -> None:
    '''
    Segments the image by giving it a shape to segment.
    
    Args:
        image_path (str): Path to image to segment.
        annotated_image_name (str): Name of the annotated image without the sufix.
        shape_type (Shape): Shape used to mark the part of the image the user wants to segment.
    '''

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color to RGB
    image_name = f"{annotated_image_name}.png"

    create_directory(FOLDER_ANNOTATED)
    create_directory(FOLDER_MASKS)
    create_directory(FOLDER_ANNOTATIONS)
    create_directory(FOLDER_TXT)

    # if cuda is not available, use cpu
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup SAM model
    sam_checkpoint: str = "./sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"

    sam=sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Set the SAM Predictor
    predictor=SamPredictor(sam)

    # Process image to produce image embedding that will be used for mask prediction
    predictor.set_image(image)

    # Set the point on the object you want to detect
    input_point = np.array([[None, None]])
    input_label = np.array([1])

    # Callback function depends on the shape used for marking
    def mouse_callback_point(event, x, y, flags, param) -> None:
        nonlocal input_point
        # Check if the event was left button
        if event==cv2.EVENT_LBUTTONDOWN:
            # Store the coordinates in the list
            input_point=np.array([[x,y]])
            print(f"Point selected: {input_point}")

    def mouse_callback_rectangle(event, x, y, flags, param) -> None:
        nonlocal input_point
        global start_point
        # Check if the event was left button
        if event==cv2.EVENT_LBUTTONDOWN:
            # Store the coordinates in the list
            start_point=(x,y)
        if event==cv2.EVENT_LBUTTONUP:
            # Store the coordinated in the list
            end_point=(x,y)
            input_point=np.array([[start_point[0], start_point[1],end_point[0], end_point[1]]])
            print(input_point)

    # Create a window and set the mouse callback function to capture the click event
    cv2.namedWindow("Image")
    if shape_type == Shape.RECTANGLE:
        cv2.setMouseCallback("Image", mouse_callback_rectangle)
    elif shape_type == Shape.POINT:
        cv2.setMouseCallback("Image", mouse_callback_point)

    while True:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # user can exit the program by pressing the 'BUTTON_EXIT'
        if cv2.waitKey(1) & 0xFF == ord(BUTTON_EXIT):
            break

    cv2.destroyAllWindows()

    print(f"Coordinates of the selected point: {input_point}")


    if shape_type == Shape.POINT:
        # Predict the object using created points using mouse
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

    elif shape_type == Shape.RECTANGLE:
        # Predict the object using created bounding box
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_point[None, :],
            multimask_output=False,
        )

    # Create YOLO-compatible annotation
    h, w = masks[0].shape
    #y,x = np.where(masks[0]>0)
    y, x = np.nonzero(masks[0]>0)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # YOLO format: class_id x_center  y_center width height (normalized)
    x_center = (x_min+x_max) / 2.0 / w
    y_center = (y_min+y_max) / 2.0 / h
    bbox_width = (x_max-x_min) / w
    bbox_height = (y_max-y_min) / h

    class_id = 0
    yolo_annotation = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

    # Store information

    annotation_save_path = f"{FOLDER_TXT}/annotation{annotated_image_name}.txt"
    with open(annotation_save_path, "w") as f:
        f.write(yolo_annotation)

    mask_save_path = f"{FOLDER_MASKS}/{annotated_image_name}_mask.png"
    cv2.imwrite(mask_save_path, (masks[0] * 255).astype(np.uint8))

    output_image_path = f"{FOLDER_ANNOTATIONS}/{image_name}"

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    
    if shape_type == Shape.POINT:
        show_points(input_point, input_label, plt.gca())

    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Mask saved at: {mask_save_path}")
    print(f"Annotated image saved at: {output_image_path}")
    print(f"YOLO annotation saved at: {annotation_save_path}")