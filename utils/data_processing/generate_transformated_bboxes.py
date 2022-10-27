from utils.data_processing.grasp import GraspRectangle, GraspRectangles


def set_rectangles_angles(rectangles):
    """Set angle==0 for each grasp rectangle"""
    new_rectangles = rectangles.copy()
    angles = []
    for idx, rectangle in enumerate(rectangles):
        angles.append(rectangle.angle)
        new_rectangles[idx].rotate(-1 * rectangle.angle, rectangle.center)

    return (new_rectangles, angles)


def build_rectangle(bboxes_transformed, format='coco'):
    """Coco format contains top left corner, width and height of rectangle"""
    if format == 'coco':
        grs = []
        for idx, rectangle in enumerate(bboxes_transformed):
            x_min = int(rectangle[0])
            y_min = int(rectangle[1])
            width = int(rectangle[2])
            height= int(rectangle[3])
            top_left = [x_min, y_min]
            bottom_left = [x_min, y_min + height]
            top_right = [x_min + width, y_min]
            bottom_right = [x_min + width, y_min + height]
            points = [top_left, bottom_left, bottom_right, top_right]
            grasp_rectangle = GraspRectangle(points)
            grasp_rectangle.rotate(bboxes_transformed[idx][4], bboxes_transformed[idx][5])
            grs.append(grasp_rectangle)

        grasp_rectangles = GraspRectangles(grs)
        return grasp_rectangles

    else:
        raise NotImplemented('Another formats are not implemented yet')


