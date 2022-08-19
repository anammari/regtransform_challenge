##################################################
## preparation script for detecting 'faulty' 
## observations in the COCO file for training data
##################################################
## Author: Ahmad Ammari
# ################################################

import json

#define the input parameters
inputfile = {}
inner = {}
annotation_path = 'IB-SEC.v2i.coco/_annotations.coco.modified.train.json'
#annotation_out = 'IB-SEC.v2i.coco/_annotations.coco.modified.valid.json'

#fetch the content of the annotations file
with open(annotation_path, 'r+') as f:
    allData = json.load(f)
    annotations_data = allData['annotations']
    images_data = allData['images']
    categories_data = allData['categories']
    licenses_data = allData['licenses']
    info_data = allData['info']
    print('annotation json have been read!')

#detect the faulty bbox in annotations
f_annotations_data = []
for i in annotations_data:
    if len(i['bbox']) > 0:
        for item in i['bbox']:
            if item < 0:
                inner = i
                f_annotations_data.append(inner)
inputfile['annotations'] = f_annotations_data

#detect the faulty image_id in annotations
image_ids = []
for i in annotations_data:
    image_ids.append(i['image_id'])
inputfile['image_ids'] = [x for x in range(0, 240) if x not in image_ids]
print('annotations with faulty bbox and/or image_ids:')
print(inputfile)

#write the filtered content to JSON file
#inputfile = json.dumps(inputfile)
# with open(annotation_out, 'a+') as f:
#     f.write(str(inputfile))
#     print('annotation json have been filtered!')