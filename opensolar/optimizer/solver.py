import numpy as np
import cv2

ROTATION_ANGLES = {
    'N': 0,
    'NNE': -22,
    'NE': -44,
    'ENE': -67,
    'E': -90,
    'ESE': -112,
    'SE': -135,
    'SSE': -157,
    'S': 180,
    'SSW': 158,
    'SW': 135,
    'WSW': 113,
    'W': 90,
    'WNW': 68,
    'NW': 45,
    'NNW': 23
}


def contain_obstacles(orig_image, mask, panel):
    mask = mask > 0
    masked_image = orig_image * np.repeat(mask[..., None], 3, -1)

    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray_image, threshold1=80, threshold2=150)

    aux_mask = np.zeros(mask.shape)
    polygon = np.array([[panel[0], panel[1]], [panel[2], panel[3]], [panel[4], panel[5]], [panel[6], panel[7]]], np.int32)
    aux_mask = cv2.fillPoly(aux_mask, [polygon], 255)

    contain_obstacles = (aux_mask * edges).max() > 0
    return contain_obstacles

def place_solar_panels(rooftop_parts, orig_image, solar_panel_size=(10, 50)):
    for rooftop_part in rooftop_parts:
        kernel = np.ones((5, 5), np.uint8)

        rotation_angle = -ROTATION_ANGLES[rooftop_part['cls']]
        mask = cv2.dilate(rooftop_part['mask'], kernel)
        mask = cv2.erode(mask, kernel)
        
        num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

        info = list(zip(stats, centroid, list(range(num_labels))))
        info = sorted(info,key=lambda x: x[0][-1])
        center = info[-2][1]
        bbox_info = info[-2][0]
        label = info[-2][2]

        mask = ((labels == label) * 255).astype(np.uint8)

        contours,_ = cv2.findContours(mask, 1, 2)
        contour = contours[0]
        rect = cv2.minAreaRect(contour)
        rotated_angle = rect[-1]
        print(rotated_angle)
        box = cv2.boxPoints(rect)
        print(box)
        box = np.int0(box)
        image = cv2.drawContours(np.zeros((400, 400, 3)),[box],0,(0,0,255),2)
        
        cv2.imshow('label', (image).astype(np.uint8))
        cv2.waitKey(0)
        print(labels.shape)

        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle, scale=1)

        rotated_mask = cv2.warpAffine(src=rooftop_part['mask'], M=rotate_matrix, dsize=mask.shape)

        # find new centers to move masks to the origin to facilitate computation
        _, _, aux_stats, aux_centroid = cv2.connectedComponentsWithStats(rotated_mask, 8, cv2.CV_32S)

        aux_info = list(zip(aux_stats, aux_centroid))
        aux_info = sorted(aux_info,key=lambda x: x[0][-1])
        aux_center = aux_info[-2][1]
        aux_bbox_info = aux_info[-2][0]

        aux_mask = np.zeros(mask.shape)
        aux_mask[0:aux_bbox_info[3], 0:aux_bbox_info[2]] = rotated_mask[aux_bbox_info[1]:aux_bbox_info[1]+aux_bbox_info[3], aux_bbox_info[0]:aux_bbox_info[0] + aux_bbox_info[2]]

        panels = []
        for i in range(0, aux_mask.shape[0] - solar_panel_size[0], solar_panel_size[0]):
            for j in range(0, aux_mask.shape[1] - solar_panel_size[1], solar_panel_size[1]):
                if aux_mask[i, j] == 255 and aux_mask[i + solar_panel_size[0], j + solar_panel_size[1]]:
                    #aux_mask_for_draw = cv2.rectangle(aux_mask_for_draw, (j, i), (j + solar_panel_size[1], i + solar_panel_size[0]), (255, 0, 0), 3)
                    panels.append(np.array([j, i, j + solar_panel_size[1], i + solar_panel_size[0]]))
        
        panels = np.stack(panels)

        # moving back to the previous center
        panels[:, 0] += aux_bbox_info[0]
        panels[:, 2] += aux_bbox_info[0]
        panels[:, 1] += aux_bbox_info[1]
        panels[:, 3] += aux_bbox_info[1]

        # since now panels can be rotated, we need to use polygons
        panels_polygon = np.zeros((panels.shape[0], 8))
        panels_polygon[:, :2] = panels[:, :2]
        panels_polygon[:, 2] = panels[:, 0] + solar_panel_size[1]
        panels_polygon[:, 3] = panels[:, 1]
        panels_polygon[:, 4:6] = panels[:, 2:]
        panels_polygon[:, 6] = panels[:, 0]
        panels_polygon[:, 7] = panels[:, 1] + solar_panel_size[0]

        # undo rotation
        undo_rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-rotation_angle, scale=1)
        panels_polygon[:, 0:2] = (undo_rotate_matrix @ (np.concatenate([panels_polygon[:, 0:2], np.ones(panels.shape[0])[..., None]], -1).T)).T
        panels_polygon[:, 2:4] = (undo_rotate_matrix @ (np.concatenate([panels_polygon[:, 2:4], np.ones(panels.shape[0])[..., None]], -1).T)).T
        panels_polygon[:, 4:6] = (undo_rotate_matrix @ (np.concatenate([panels_polygon[:, 4:6], np.ones(panels.shape[0])[..., None]], -1).T)).T
        panels_polygon[:, 6:] = (undo_rotate_matrix @ (np.concatenate([panels_polygon[:, 6:], np.ones(panels.shape[0])[..., None]], -1).T)).T
        
        panels_polygon = panels_polygon.astype(np.int32)
        aux_mask_for_draw = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        final_panels = []
        for panel in panels_polygon:
            if mask[panel[1], panel[0]] == 255 and mask[panel[3], panel[2]] == 255 and mask[panel[5], panel[4]] == 255 and mask[panel[7], panel[6]] == 255:
                if not contain_obstacles(orig_image, mask, panel):
                    aux_mask_for_draw = cv2.polylines(aux_mask_for_draw, [panel.reshape((-1, 1, 2))], True, (255, 0, 0), 3)
                    final_panels.append([(panel[0], panel[1]), (panel[2], panel[3]), (panel[4], panel[5]), panel[6], panel[7]])
        
        cv2.imshow('mask', aux_mask_for_draw)
        cv2.waitKey(0)
    return final_panels
