import cv2
import numpy as np
import torch
from opensolar.detection.utils.segment.general import scale_image
from opensolar.detection.utils.plots import colors

def masks(masks, colors, im, ori_shape, alpha=0.5):
    """Plot masks at once.
    Args:
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
        im (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
    """
    if len(masks) == 0:
        im[:] = im.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    colors = torch.tensor(colors, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(
        0
    ) * 2  # mask color summand shape(n,h,w,3)

    im = im.flip(dims=[0])  # flip channel
    im = im.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    im = im * inv_alph_masks[-1] + mcs
    im_mask = (im * 255).byte().cpu().numpy()
    im = scale_image(im.shape, im_mask, ori_shape)
    return im

def draw_instance_masks(meta_info, pred):
    img = meta_info['img_scaled']
    ori_img_shape = meta_info['orig_shape']
    pred_masks = meta_info['pred_masks']

    composed_image = masks(
        pred_masks, [colors(x['cls_id'], True) for x in pred], img, ori_img_shape
    )

    return composed_image

def draw_panels(image, roofs_panels):
    for roof in roofs_panels:
        for panel in roof['panels']:
            panel = np.array([panel[0][0], panel[0][1], panel[1][0], panel[1][1], panel[2][0], panel[2][1], panel[3][0], panel[3][1]])
            image=cv2.polylines(image, [panel.reshape((-1, 1, 2))], True, (255, 0, 0), 3)
    
    return image

def draw_edge_maps(image, edge_maps):
    for edge_map in edge_maps:
        edge_map_mask = np.where(edge_map == 255, 0, 1)
        edge_map_mask = np.repeat(edge_map_mask[:, :, None], 3, -1)
        image = image * edge_map_mask
    
    return image