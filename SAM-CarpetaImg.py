from PIL import Image
import os
import torch
import ultralytics
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from IPython.display import display, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
ultralytics.checks()


#definir las carpetas de entrada y salida
input_folder = "fotosExtraidas"
output_folder = "sinFondoSAM"

#----------------BORRAR CON REMBG-----------------
# loop a traves de todos los archivos en la carpeta de entrada
# for filename in os.listdir(input_folder):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         #definir input y output file paths
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename.split('.')[0]+'.png')
        
#         #cargar el input image
#         input_image = Image.open(input_path)
        
#         #remover el fondo usando rembg
#         output_image = remove(input_image)
        
#         #guardar el output en el archivo png
#         output_image.save(output_path)

        
model=YOLO('yolov8n.pt')
resultado = model.predict(source=input_folder,conf=0.25)


for resultados in resultado:
    boxes = resultados.boxes
bbox = boxes.xyxy.tolist()[0]
# print(bbox)

sam_checkpoint = "modelosEntrenados/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.cvtColor(cv2.imread(input_folder + '/' +filename), cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.split('.')[0]+'.png')
        
        input_box = np.array(bbox)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
        
        segmentation_mask = masks[0]

# Convert the segmentation mask to a binary mask
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
        white_background = np.ones_like(image) * 255

        # Apply the binary mask
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
        plt.imshow(new_image.astype(np.uint8))
        plt.axis('off')
        plt.show()

