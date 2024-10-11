rectangle = input('Segment with rectangle? (y/n): ').lower() == 'y'

image = 'car.png'
annon_image = '001'

if rectangle:
    from Segmentator_point import segment_using_mouse
    segment_using_mouse(image_path=image, annotated_image_name=annon_image)

else:
    from Segmentator_rectangle import segment_using_rectangle
    segment_using_rectangle(image_path=image,annotated_image_name=annon_image)