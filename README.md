# Yolo data preparation

- Install Label Studio
- go to create, you will see 3 tabs
- name your project and description in the 1st
- then upload all the images you want to label in the 2nd
- then choose labeling type -> this one is segmentation semantic something, basically the 1st option
- then define your classes, better to use XML like this
```
<View>
  <Header value="Select label and click image to segment:" />
  <Image name="image" value="$image" zoom="true" />
  
  <PolygonLabels name="label" toName="image" strokeWidth="3">
    <Label value="bhat" background="#FFD700" />
    <Label value="dal" background="#8B4513" />
    <Label value="sabji" background="#008000" />
  </PolygonLabels>
</View>
```
- then select each image, label with respective class and make sure to click "submit" to confirm
- after labeling all the images, click "export", then export in the version you want -> this one is Yolo with Images.

-  then unzip the exported folder, use spit_dataset.py to split the plain folder into train and test, as required by YOLO to train.

# Data preparation is done
