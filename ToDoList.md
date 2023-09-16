# To Do List:
- ~~make flow chart~~
- ~~extract text from image/bbox (OCR)~~
- ~~tick_label to x_tick or y_tick~~
- validator for nobs (check also histograms which should have an extra X value compared to y values)
- ~~create MGA scorer (compare gt to predicted outcome)~~
- Output processor: processing output to relevent data type according to chart type,
for example if a chart is line and also dots were detected, ~~categorical X values for lines,
numerical values for scatter plots~~

## classifier
- ~~upload classifier model weights~~
- ~~code to load classifier~~
- ~~create function that takes a genereted images dir and create a dataframe passable to chart data loader~~
- ~~check classifier works on generated data~~ did not work well (~ 4k added)
- ~~retrain~~ 
- check if needed, detector could also act as a classifier
- connect to general data processor (Full model)
- make sure it works with cuda


## detector
- generate data to general detector
- check line detector: Background images 
- check corrupt images
- delete miss classified background image 
- ~~Yolo wrapper, results to tensor: taking the yolo output and creating a tensor suitable for next stage prediction,
should include x,y,type,value~~
- ~~add resume option from started project~~
- detection to graph classification (just need to wrap it up)

## OCR
- choose OCR (easyoct, tesserect, trocr, pp-ocr(paddle)). currently trocr seem to work the best did not try pp-ocr
- ~~get direction of rotation, check if improves trocr~~ improves on small by eye test
- ~~connect to detector~~


## virtual machine
- ~~check if we can use a faster disk as I/O is currently the bottleneck~~
- connect notebooks to git

## corrupt images
73cfbba65962
872d1be39bae
