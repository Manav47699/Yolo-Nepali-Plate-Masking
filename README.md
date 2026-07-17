# Yolo-Nepali-Plate-Masking

### Creating the dataset

- Install and run label studio by:

```
pip install label-studio
label-studio
```

- Sign in, then click on create project, then name the project, select all the files/images you want to label, then selected how you want to label the data. "Semantic Segmentation with Polygons" for this project. then clean any preview or default label given and add your label OR switch to code and give your labels and assigned colors for each label in XML, like this:

```
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="Bhat" background="#FFD700"/>
    <Label value="Dal" background="#8B4513"/>
    <Label value="Sabji" background="#008000"/>
  </PolygonLabels>
</View>
```

- then you can start labeling your images.

- once done labeling, click on submit

- then come back to home dasboard, click on your project, click on export, select which type of data you wnat to export "YOLO with Images" for this project, then finally click export.

- you will have your zip file downloaded.