import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd
from tqdm import tqdm

with open('raw/tmall/user_log_format1.csv',"r") as f:              # including category_Id and brand_Id
    reader = csv.DictReader(f)
    item_category = {}
    item_brand = {}
    for row in reader:
        item = row['item_id']

        # Fill item_category dict
        if item not in item_category:
            item_category[item] = int(row['cat_id'])
        
        # Fill item_brand dict
        if not row['brand_id']: # If the brand_id is empty, fill it with 0
            brnd = 0
        else:
            brnd = int(row['brand_id'])
            
        if item not in item_brand:   # If the brand_id is empty, fill it with 0
            item_brand[item] = brnd
        elif item_brand[item] == 0:
            item_brand[item] = brnd
    
            
with open('raw/tmall/Tmall_category.csv', 'w',newline='') as csvfile:     
    writer = csv.DictWriter(csvfile,fieldnames=['item_id','category_id'])
    writer.writeheader()
    for k,v in tqdm(item_category.items()):
        writer.writerow({'item_id':k,'category_id':v})
            
with open('raw/tmall/Tmall_brand.csv', 'w',newline='') as csvfile:     
    writer = csv.DictWriter(csvfile,fieldnames=['item_id','brand_id'])
    writer.writeheader()
    for k,v in tqdm(item_brand.items()):
        writer.writerow({'item_id':k,'brand_id':v})

