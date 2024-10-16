from Shape import Shape
from segment import segment_image

rectangle: bool = input('Segment with rectangle (y) or point (press anything != y): ').lower() == 'y'

image: str = 'car.png'
annotation_image: str = '001'

shape = Shape.POINT

if rectangle:
    shape = Shape.RECTANGLE

segment_image(image_path=f'images/{image}', annotated_image_name=annotation_image, shape_type=shape)

#if rectangle:
#
#    from Segmentator_rectangle import segment_using_rectangle
#    segment_using_rectangle(image_path=f'images/{image}', annotated_image_name=annotation_image)
#
#else:
#    from Segmentator_point import segment_using_mouse
#    segment_using_mouse(image_path=f'images/{image}', annotated_image_name=annotation_image)