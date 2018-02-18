def get_class(classes, category_index, boxes, scores, min_score_thresh=.5):
    # if not max_boxes_to_draw:
    #   max_boxes_to_draw = boxes.shape[0]
    annotation = []
    count = 0
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            class_name = 'N/A'
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            if class_name == 'person':
                count += 1
                annotation_frame = {}
                annotation_frame['class'] = class_name
                bounding_box = {}
                bounding_box['ymin'] = box[0]
                bounding_box['xmin'] = box[1]
                bounding_box['ymax'] = box[2]
                bounding_box['xmax'] = box[3]
                annotation_frame['bounding_box'] = bounding_box
                annotation.append(annotation_frame)
    return annotation, count
