# This is a helper script that prints out the coco classes of your customzied dataset
# Make sure that the classes listed in your customized config file are in the same order as being printed by this script

import argparse
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description='COCO classcheck')
    parser.add_argument('--data', help="dataset annotation json path" , required=True, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

coco = COCO(args.data)

categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}


for category_id, category_name in category_id_to_name.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")