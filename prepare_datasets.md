# Prepare datasets

The training of our work is on two benchmark datasets: [COCO](https://cocodataset.org/) and [COCOCaption](https://cocodataset.org/).
Please orignize the datasets as following.
```
$CADet_ROOT/datasets/
    coco/
```

Please follow the following instruction to pre-process individual datasets.

### COCO, COCO Caption and SAM-based COCO proposals

First, download COCO data and the SAM-based COCO proposals provided via our [COCO_2017_train](https://drive.google.com/file/d/12EsjKwKxYAvo4JI1QfULNfq4mKmuyyMT/view?usp=sharing) and [COCO_2017_val](https://drive.google.com/file/d/1E0rlkotaX1ornQ7rr45QMvBbkMUG1xUR/view?usp=sharing) (alternatively, you may also generate proposals for each image using the [SAM](https://github.com/facebookresearch/segment-anything).) place them in the following way:
```
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
    proposals/
        sam_coco_2017_train_d2.pkl
        sam_coco_2017_val_d2.pkl
```

We first follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb) to create the open-vocabulary COCO split. The converted files should be like:

```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```

We further preprocess the annotation format for easier evaluation:

```
python tools/get_coco_zeroshot.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

And process the category infomation: 

```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_del.json
```

Next, we prepare the class and concept embedding following RegionCLIP. Download the embedding file from [here](https://drive.google.com/drive/folders/1_HKaLSyA9fIjTS0BXxT15NRac6uRoyIj) or generate it with tools from [RegionCLIP](https://github.com/microsoft/RegionCLIP/tree/9fd374015db384bc0548b4af85446b90e13d2ae1#extract-concept-features). The files should be like: 

```
coco/
    CADet/
        coco_65_cls_emb.pth
        coco_65_concepts.txt
        coco_nouns_4764_emb.pth
        coco_nouns_4764.txt
```

Then preprocess the COCO caption data with the predefined concepts:

```
python tools/get_tags_for_CADet_concepts.py --cc_ann datasets/coco/annotations/captions_train2017.json --allcaps --cat_path datasets/coco/CADet/coco_nouns_4764.txt --convert_caption --out_path datasets/coco/CADet/nouns_captions_train2017_4764tags_allcaps.json
``` 

This creates `datasets/coco/CADet/nouns_captions_train2017_4764tags_allcaps.json`.
