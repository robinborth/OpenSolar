import numpy as np

ROTATION_ANGLES = {
    'N': 0,
    'NNE': -22,
    'NE': -44,
    'ENE': -67,
    'E': -90,
    'ESE': -112,
    'SE': -135,
    'SSE': -157,
    'S': -180,
    'SSW': -202,
    'SW': -225,
    'WSW': -247,
    'W': -270,
    'WNW': -292,
    'NW': -315,
    'NNW': -337 
}

def place_solar_panels(rooftop_parts):
    for rooftop_part in rooftop_parts:
        rotation_angle = rooftop_part['cls']
        print(rooftop_part['mask'].shape)
        height, width = rooftop_part['mask'].shape
        center = (width/2, height/2)        
        
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle, scale=1)
        rotated_mask = cv2.warpAffine(src=rooftop_part['mask'], M=rotate_matrix, dsize=(width, height))

        cv2.imshow('rotated_mask', rotated_mask)
        cv2.waitKey(0)