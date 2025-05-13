# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image, ImageDraw
from retinaface import RetinaFace


def get_mask(image, faces, erosion=0.1):
    mask = Image.new(mode="L", size=image.size, color="black")
    width, height = image.size
    bboxes = []
    for face in faces:
        x0, y0, x1, y1 = face
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        x0 = min(x0, width)
        x1 = min(x1, width)
        y0 = min(y0, height)
        y1 = min(y1, height)
        bbox = [x0, y0, x1, y1]

        diagonal = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        bbox = [
            bbox[0] - erosion * diagonal,
            bbox[1] - erosion * diagonal,
            bbox[2] + erosion * diagonal,
            bbox[3] + erosion * diagonal,
        ]
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill="white")
        bboxes.append(bbox)
    return mask, bboxes

def blur_faces(
    img: Image.Image
) -> Image.Image:
    resp = RetinaFace.detect_faces(np.array(img))
    faces = [face["facial_area"] for face in resp.values()]
    if len(faces) == 0:
        return img
    
    mask, bboxes = get_mask(img, faces)
    image_width = img.width
    image_height = img.height
    background = Image.new('RGB', (image_width, image_height))
    for bbox in bboxes:
        img_face_crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        width, height = img_face_crop.size
        small_image = img_face_crop.resize((4, 4), Image.BILINEAR)
        pixelated_image = small_image.resize((width, height), Image.NEAREST)
        background.paste(pixelated_image, (int(bbox[0]), int(bbox[1])))
    
    
    return Image.composite(background, img, mask)


