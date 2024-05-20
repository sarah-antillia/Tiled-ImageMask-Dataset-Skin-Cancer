# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2024/05/20 to-arai
# TiledImageMaskDatasetGenerator.py

"""
From 
./Large-Skin-Cancer-master
 ├─images
 └─masks

this script creates
./Tiled-Skin-Cancer-master
 ├─images
 └─masks


1 Create tiledly splitted patches (images and masks) of 512x512 from the large image and mask files
  in ./Large-Skin-Cancer-master, and save them to ./Tiled-Skin-Cancer-master

2 Resize the large images and masks to 512x512, and save to ./Tiled-Skin-Cancer-master
3 Augment the resized images and mask by rotation operations if rotation flag=True, 
  and save them to ./Tiled-Skin-Cancer-master

In summary,   
./Tiled-Skin-Cancer-master
contains two type of dataset 
1 Tiledly splitted images and masks: microscopic segmentation dataset
2 Resized images and masks: macroscopit segmentation dataset

"""

import os
import shutil
import glob
import cv2
import numpy as np

from PIL import Image, ImageFilter

import traceback

class TiledImageMaskDatasetGenerator:

  def __init__(self, split_size=512):
    self.split_size   = split_size
    self.resize       = split_size
    self.cut_in_half  = False

    # Blur flag
    self.blur         = True
    # GausssinaBlur parameters
    # Blur parameter for the resized 
    self.blur_ksize1  = 7
    # Blur parameter for the splitted
    self.blur_ksize2  = 15

    #Augmentation parameters for the resized.
    self.rotation     = True
    self.ANGLES       = [90, 180, 270]
 
  def split(self, root_dir, output_dir):
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    input_images_dir = os.path.join(root_dir, "images")
    input_masks_dir  = os.path.join(root_dir, "masks")

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    self.resize_one(input_masks_dir,  output_masks_dir, output_images_dir, mask=True)
    self.resize_one(input_images_dir, output_masks_dir, output_images_dir, mask=False)
    
    self.split_one(input_masks_dir,  output_masks_dir, output_images_dir, mask=True)
    self.split_one(input_images_dir, output_masks_dir, output_images_dir, mask=False)


  def resize_one(self, input_images_dir, output_masks_dir, output_images_dir,  mask=False):
    image_files  = glob.glob(input_images_dir + "/*.jpg")
    image_files += glob.glob(input_images_dir + "/*.png")
    image_files  = sorted(image_files)
    print("--- image_files {}".format(image_files))
    index = 1000

    for image_file in image_files:
      index += 1
      image   = cv2.imread(image_file)
      resized = cv2.resize(image, (self.resize, self.resize))
      #resized = self.resize_to_square(image, mask=mask)

      filename = "s_" + str(index) + ".jpg"
      output_mask_filepath  = os.path.join(output_masks_dir,  filename) 
      output_image_filepath = os.path.join(output_images_dir, filename) 
      if mask:
        if self.blur:
          resized = cv2.GaussianBlur(resized, ksize=(self.blur_ksize1, self.blur_ksize1), sigmaX=0)
        cv2.imwrite(output_mask_filepath, resized)
        print("--- Saved {}".format(output_mask_filepath))
        self.augment(resized, filename, output_masks_dir, mask=True )

      else:
        if os.path.exists(output_mask_filepath):
          cv2.imwrite(output_image_filepath, resized)
          print("--- Saved {}".format(output_image_filepath))
          self.augment(resized, filename, output_images_dir, mask=False)
        else:
          pass
  
  def augment(self, image, basename, output_dir, mask=False):
    if self.rotation:
      self.rotate(image, basename, output_dir, mask=mask)

  def rotate(self, image, basename, output_dir,  mask=False):

    for angle in self.ANGLES:      
      center = (self.resize/2, self.resize/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
      color  = image[2][2].tolist()
      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, 
                                     dsize=(self.resize, self.resize), borderValue=color)
  
      filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(filepath, rotated_image)
      print("=== Saved {}".format(filepath))
    
  
  def resize_to_square(self, image, mask=False):     
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    if mask ==False:
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8) 
      (b, g, r) = image[20][20] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background += [b, g, r][::-1]
      
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (self.resize, self.resize)
    resized = cv2.resize(background, (self.resize, self.resize))

    return resized

  def split_one(self, input_images_dir, output_masks_dir, output_images_dir,  mask=False):
    image_files  = glob.glob(input_images_dir + "/*.jpg")
    image_files += glob.glob(input_images_dir + "/*.png")
    image_files  = sorted(image_files)

    # Take half of the image_files to reduce the number of splitted files. 
    if self.cut_in_half:
      hlen = int(len(image_files)/2)
      image_files = image_files[:hlen]
    print("--- image_files {}".format(image_files))
    index = 1000
    split_size = self.split_size

    for image_file in image_files:
      index += 1

      image = Image.open(image_file)

      w, h  = image.size

      vert_split_num  = h // split_size
      if h % split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // split_size

      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size
  
          cropbox = (left,  upper, right, lower )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          #line = "image file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i, left, upper, cw, ch)
          #print(line)            
          cropped_image_filename = str(index) + "_" + str(j) + "x" + str(i) + ".jpg"
          output_mask_filepath  = os.path.join(output_masks_dir,  cropped_image_filename) 
          output_image_filepath = os.path.join(output_images_dir, cropped_image_filename) 

          if mask:
            if self.is_not_empty(cropped):
              if self.blur:
                cropped = cropped.filter(ImageFilter.GaussianBlur(radius = self.blur_ksize2)) 

              cropped.save(output_mask_filepath)
              print("--- Saved {}".format(output_mask_filepath))

          else:
            if os.path.exists(output_mask_filepath):
              cropped.save(output_image_filepath)
              print("--- Saved {}".format(output_image_filepath))
            else:
              pass

  def is_not_empty(self, img):
    rc = False
    img = self.pil2cv(img)
    if img.any() > 0:    
       rc = True
    return rc
  
  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

"""
From
./Large-Skin-Cancer-master
 ├─images
 └─masks

splitting each image and mask of the dataset above to 640x640 tiled,
and save those tiledly splitted images and masks under
./Tiled-Skin-Cancer-master
 ├─images
 └─masks

"""
  


if __name__ == "__main__":
  try:
    input_dir  = "./Large-Skin-Cancer-master"
    output_dir = "./Tiled-Skin-Cancer-master"
    splitter = TiledImageMaskDatasetGenerator()

    splitter.split(input_dir, output_dir)
    
  except:
    traceback.print_exc()