import json

#select the target classes
className = {
    6 : 'foot_note',
    9 : 'heading',
    11: 'paragraph'
}
#select the target class Ids
classNum = [6, 9, 11]

#define the input parameters
inputfile = {}
inner = {}
annotation_path = 'IB-SEC.v2i.coco/_annotations.coco.valid.json'
annotation_out = 'IB-SEC.v2i.coco/_annotations.coco.modified.valid.json'

#fetch the content of the annotations file
with open(annotation_path, 'r+') as f:
    allData = json.load(f)
    annotations_data = allData['annotations']
    images_data = allData['images']
    categories_data = allData['categories']
    licenses_data = allData['licenses']
    info_data = allData['info']
    print('annotation json have been read!')

#filter the content by the target classes
inputfile['info'] = info_data
inputfile['licenses'] = licenses_data
inputfile['images'] = images_data
f_categories_data = []
for i in categories_data:
    if (i['id'] in classNum):
        inner = i
        f_categories_data.append(inner)
inputfile['categories'] = f_categories_data
f_annotations_data = []
for i in annotations_data:
    if (i['category_id'] in classNum):
        inner = i
        f_annotations_data.append(inner)
inputfile['annotations'] = f_annotations_data

#write the filtered content to JSON file
inputfile = json.dumps(inputfile)
with open(annotation_out, 'a+') as f:
    f.write(str(inputfile))
    print('annotation json have been filtered!')