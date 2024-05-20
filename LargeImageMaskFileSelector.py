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
#
# LargeImageMaskSelector.py
# 2024/05/19 to-arai

import os
import shutil

import glob
import cv2
import numpy as np
import traceback
from PIL import Image


class LargeImageMaskFileSelector:

  def __init__(self, resize=512): 
    self.RESIZE = resize
    self.SELECTABLE_IMAGE_WIDTH = 6600
    

  def generate(self, input_images_dir, input_masks_dir, images_output_dir, masks_output_dir):

    image_files = glob.glob(input_images_dir + "/*.jpg")
    
    mask_files  = glob.glob(input_masks_dir + "/*.png")
    num_images  = len(image_files)
    num_masks   = len(mask_files)
    
    print("=== generate num_image_files {} num_masks_files {}".format(num_images, num_masks))

    if num_images != num_masks:
      raise Exception("Not matched image_files and mask_files")
    
  
    for image_file in image_files:
      try:
        basename = os.path.basename(image_file)
        name     = basename.split(".")[0]
        mask_filepath  = os.path.join(input_masks_dir, name + "_segmentation.png")

        image = Image.open(image_file).convert("RGB")
        w, h  = image.size
        if w <= self.SELECTABLE_IMAGE_WIDTH:
           print("Skipping small image {} less than {}".format(image_file, w))
           continue
        
        image = self.resize_to_multiple(image)
   
        image_filepath = os.path.join(images_output_dir,  basename)
        if os.path.exists(image_filepath):
          print("Found a file of same name ")
          input("---")

        image.save(image_filepath)
        print("=== Saved {} to {}".format(image_file, image_filepath))

        mask = Image.open(mask_filepath).convert("RGB")
        #mask = self.create_mono_color_mask(mask)
        mask = self.resize_to_multiple(mask)
        out_mask_filepath = os.path.join(masks_output_dir,  basename)
        if os.path.exists(out_mask_filepath):
          print("Found a file of same name ")
          input("---")
        mask.save(out_mask_filepath)
        print("=== Saved {} to {}".format(mask_filepath, out_mask_filepath))
      except:
        traceback.print_exc()

  def resize_to_multiple(self, image):
     w, h  = image.size
     wn = w // self.RESIZE + 1
     hn = h // self.RESIZE + 1
     resized_w = wn * self.RESIZE
     resized_h = hn * self.RESIZE

     resized = image.resize( (resized_w, resized_h)) 

     return resized

"""
From:
./ISIC-2017
├─ISIC-2017_Test_v2_Data
├─ISIC-2017_Test_v2_Part1_GroundTruth
├─ISIC-2017_Training_Data
├─ISIC-2017_Training_Part1_GroundTruth
├─ISIC-2017_Validation_Data
└─ISIC-2017_Validation_Part1_GroundTruth

this script selects images and masks files of width >= 6600, 
and  generate the following images and masks datasets of width and height which are 
minimum integral multiple of 512.
./Large-Skin-Cancer-master
 ├─images
 └─masks

"""
   
if __name__ == "__main__":
  try:
    generator = LargeImageMaskFileSelector()
 
    input_images_dir  = "./ISIC-2017/ISIC-2017_Training_Data/"
    input_masks_dir   = "./ISIC-2017/ISIC-2017_Training_Part1_GroundTruth/"

    images_output_dir = "./Large-Skin-Cancer-master/images/"
    masks_output_dir  = "./Large-Skin-Cancer-master/masks/"

    if os.path.exists(images_output_dir):
      shutil.rmtree(images_output_dir)

    if not os.path.exists(images_output_dir):
      os.makedirs(images_output_dir)

    if os.path.exists(masks_output_dir):
      shutil.rmtree(masks_output_dir)

    if not os.path.exists(masks_output_dir):
      os.makedirs(masks_output_dir)

    generator.generate(input_images_dir, input_masks_dir, 
   					  images_output_dir, masks_output_dir,)
           
    input_images_dir  = "./ISIC-2017/ISIC-2017_Validation_Data/"
    input_masks_dir   = "./ISIC-2017/ISIC-2017_Validation_Part1_GroundTruth/"
 
    generator.generate(input_images_dir, input_masks_dir, 
   					  images_output_dir, masks_output_dir, )

  except:
    traceback.print_exc()
