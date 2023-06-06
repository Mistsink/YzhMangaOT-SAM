def get_center_points_from_boxes(boxes: list[list[list[int]]]) -> list[list[int]]:
    '''
    boxes: get from paddleocr
        [
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            ...
        ]
    @return: center point list
        [
            [x1, y1],
            ...
        ]
    '''
    ret = []
    for box in boxes:
        # 计算边框内最中间点的坐标
        x_center = (box[0][0] + box[2][0]) // 2
        y_center = (box[0][1] + box[2][1]) // 2
        ret.append([x_center, y_center])
    return ret