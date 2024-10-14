import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from Segmentation_helper import show_mask, create_directory
import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

#Function to segment the object using bounding box
def segment_using_rectangle(image_path,annotated_image_name) -> None:
    #Setup image:
    image = cv2.imread(image_path)
    #Convert image color to RGB:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Store image name:
    image_name = f"{annotated_image_name}.png"
    ####************ DEVELOPER STUFF ************######

    # folder where annotations will be stored.
    create_directory('AnnotatedDataset')
    # folder where mask images will be stored.
    create_directory('AnnotatedDataset/masks')
    # folder where images with annotations will be stored.
    create_directory('AnnotatedDataset/annotations')
    # where txt annotations will be stored.
    create_directory('AnnotatedDataset/txt')

    # Setup operating device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup SAM model
    sam_checkpoint: str = "./sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    #Set the SAM Predictor
    predictor=SamPredictor(sam)

    #Process image to produce image embedding that will be used for mask prediction
    predictor.set_image(image)

    #Set the point on the object you want to detect
    input_point = np.array([[None,None]])
    input_label = np.array([1])

    #Callback function that will be triggered on mouse events
    def mouse_callback(event, x,y, flags, param) -> None:
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
    h,w = masks[0].shape
    y,x = np.where(masks[0]>0)
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
    annotation_save_path=f"./AnnotatedDataset/txt/annotation{annotated_image_name}.txt"
    with open(annotation_save_path, "w") as f:
        f.write(yolo_annotation)

    #Store the mask image:
    mask_save_path=f"./AnnotatedDataset/masks/{annotated_image_name}_mask.png"
    cv2.imwrite(mask_save_path, (masks[0] * 255).astype(np.uint8))

    #Show and store annotated image:
    output_image_path=f"./AnnotatedDataset/annotations/{image_name}"
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()
    plt.close()
    

    print(f"Mask saved at: {mask_save_path}")
    print(f"Annotated image saved at: {output_image_path}")
    print(f"YOLO annotation saved at: {annotation_save_path}")