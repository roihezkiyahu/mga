# To Do List:
- ~~make flow chart~~
- extract text from image/bbox (OCR)
- tick_label to x_tick or y_tick
- validator for nobs (check also histograms which should have an extra X value compared to y values)
- create MGA scorer (compare gt to predicted outcome)
- Output processor: processing output to relevent data type according to chart type,
for example if a chart is line and also dots were detected, categorical X values for lines,
numerical values for dot plots

## classifier
- ~~upload classifier model weights~~
- ~~code to load classifier~~
- ~~create function that takes a genereted images dir and create a dataframe passable to chart data loader~~
- ~~check classifier works on generated data~~ did not work well (~ 4k added)
- retrain
- connect to general data processer (Full model)


## detector
- generate data to general detector
- check line detector
- Yolo wrapper, results to tensor: taking the yolo output and creating a tensor suitable for next stage prediction,
should include x,y,type,value
- ~~add resume option from started project~~


## virtual machine
- check if we can use a faster disk as I/O is currently the bottleneck
- connect notebooks to git
