# lane_segment
A pytorch implemented lane segmentation project with unet/deeplabv3plus network architecture. 
# dataset
The training dataset can be download here: https://aistudio.baidu.com/aistudio/datasetDetail/1919
# dataset spliting
Before training, split the dataset into training set and validating set  
and save their paths in a train.csv and val.csv seperately in data_list  
directory.  
Then modify the data path in train.py (line 199 and line 203) 
# training
nohup python lane_segment/train.py > lane_segment/train.log &  
# result  
|loss      |optimizer    |lr        |net        |mIOU       |
| ---      | ---         | ---      | ---       | ---       |
|CE        |SGD          |step_lr   |unet       |0.47       |
|CE+Dice   |SGD          |step_lr   |unet       |0.51       |
|CE+Dice   |SGD          |step_lr   |deeplabv3p |0.53       |
|CE+Dice   |AdamW        |step_lr   |deeplabv3p |0.56       |
|CE+Dice   |AdamW        |cosine_annealing|deeplabv3p|0.57  |