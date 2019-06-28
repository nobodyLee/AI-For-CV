def area(box):
    """
    :param box: [x1, x2, y1, y2]
    """
    return (box[1] - box[0] + 1) * (box[3] - box[2] + 1)


def iou(max_score_box, box):
    """
    :param max_score_box: [x1, x2, y1, y2]
    :param box: [x1, x2, y1, y2]
    """
    x1 = max(max_score_box[0], box[0])
    x2 = min(max_score_box[1], box[1])
    y1 = max(max_score_box[2], box[2])
    y2 = min(max_score_box[3], box[3])

    w = max(0.0, x2 - x1 + 1)
    h = max(0.0, y2 - y1 + 1)
    intersection = w * h

    ratio = intersection / (area(max_score_box) + area(box) - intersection)
    return ratio


def NMS(lists, thre):
    """
    :param lists: lists is a list. lists[0:4]: x1, x2, y1, y2; lists[4]: score
    """
    D = []
    while len(lists):
        max_score_box = max(lists, key=lambda x: x[4])
        D.append(max_score_box)
        lists.remove(max_score_box)
        for box in lists[:]:
            if iou(max_score_box[:4], box[:4]) >= thre:
                lists.remove(box)
    return D


if __name__ == '__main__':
    lists = [[690, 720, 800, 820, 0.5],
             [102, 204, 250, 358, 0.5],
             [118, 257, 250, 380, 0.8],
             [135, 280, 250, 400, 0.7],
             [118, 255, 235, 360, 0.7]]
    threshold = 0.3

    result = NMS(lists, threshold)
    print(result)
