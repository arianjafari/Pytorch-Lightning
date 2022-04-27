"""
Description: evaluation code for MICCAI2021 SimSurgSkill Category 1 challenge
"""

import re
import pandas as pd
import json
import os
import numpy as np
import glob

def process_box(df):
    coor = eval(df)
    bbox = [coor['x'], coor['y'], coor['w'], coor['h']]
    area = coor['w'] * coor['h']
    return bbox, area

def convert_annotation(df, type, output_file):
    # add category id
    category = {'needle driver': 1,
                'needle': 2}
    category_df = pd.DataFrame({'name': category.keys(), 'id': category.values()})
    df['category_id'] = df['obj_class'].apply(lambda x: category[x])

    # create image id and property
    df['image_id'] = df.apply(lambda x: int(f'{x.case_id:03d}{x.frame_id:05d}'), axis=1)
    df.reset_index(drop=True, inplace=True)

    # get image list
    image_list = (df.loc[:, ['image_id']]
                  .drop_duplicates().rename(columns={'image_id': 'id'}))

    # reformat bbox and create annotation id
    df['bbox'], df['area'] = zip(*df['coordinate'].apply(lambda x: process_box(x)))
    df['id'] = np.arange(1, len(df)+1)
    df['iscrowd'] = 0

    # get annotation list
    if type == 'gt':
        anno_list = df.loc[:, ['bbox', 'area', 'image_id', 'category_id', 'id', 'iscrowd']]
        anno_output = {'annotations': list(anno_list.T.to_dict().values()),
                        'images': list(image_list.T.to_dict().values()),
                        'categories': list(category_df.T.to_dict().values())}
    elif type == 'prediction':
        anno_list = df.loc[:, ['bbox', 'area',
                               'score', 'image_id', 'category_id', 'id', 'iscrowd']]
        anno_output = list(anno_list.T.to_dict().values())
    else:
        print('Invalid file type during dataset conversion')
        return

    # output annotation file
    with open(output_file, 'w') as f:
        json.dump(anno_output, f)
    return

# Generate label json for each video in one session
if __name__ == '__main__':
    # load GT files
    gt_files = sorted(glob.glob('gt/caseid*.json'))
    df_gtlist = [pd.read_json(f, orient='index') for f in gt_files]
    df_gt = pd.concat(df_gtlist)

    # load submitted files
    team_files = sorted(glob.glob('submission/caseid*.json'))
    df_list = [pd.read_json(f, orient='index') for f in team_files]
    df_team = pd.concat(df_list)

    # convert GT and submission to COCO
    convert_annotation(df_gt, type='gt', output_file='gt/gt_coco.json')
    convert_annotation(df_team,  type='prediction', output_file='submission/prediction_coco.json')