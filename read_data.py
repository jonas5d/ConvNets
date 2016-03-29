import csv
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import os
import PIL
def makeThumb(input_img, size=(128,128)):
  image = input_img
  image.thumbnail(size, PIL.Image.ANTIALIAS)
  image_size = image.size

  thumb = image.crop( (0, 0, size[0], size[1]) )

  offset_x = max( (size[0] - image_size[0]) / 2, 0 )
  offset_y = max( (size[1] - image_size[1]) / 2, 0 )

  thumb = PIL.ImageChops.offset(thumb, offset_x, offset_y)
  return thumb
  
#parser = argparse.ArgumentParser()
#parser.add_argument("csvFile",type=str,
#                    help='Where to load the csvFile from.')
#
#args = parser.parse_args()
#loc = args.csvFile
base_path = '../../../../Desktop/image_files/'
csv_path = base_path + 'sdo_xml_features_with_url-images.csv'

directory = base_path + "processed_images_128"

if not os.path.exists(directory):
  os.makedirs(directory)

df = pd.read_csv(csv_path,sep=';',header=0)
cats = pd.unique(df.classificationCategoryCode.ravel())

images = []
targets = []
for index,row in df.iterrows():

  img_path = base_path + row['Image Relative Path']
  print img_path
  if img_path.endswith('jpeg'):
    suffix = '.jpg'
  elif img_path.endswith('png'):
    suffix = '.png'
  else:
    suffix='.tiff'
  
  if row['Image Relative Path'] == 'sdo_xml_features_with_url.csv-images/04018087010078-10000187-image-png':
    continue
  img = PIL.Image.open(img_path)
  res_img = makeThumb(img)
  images.append(res_img)
  res_img.save(directory+'/'+str(row['gtin'])+suffix)