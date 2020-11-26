import os
import numpy as np


# files = os.listdir('Saved_Model/data/')
# # print(files)
# with open('data.npy','wb') as fData:
#      arr =[]
#      for f in files:
#         for a in np.load(os.path.join("Saved_Model/data", f)):
#             arr.append(a)
#      arr = np.array(arr)
#      np.save(fData, arr) 

# files = os.listdir('Saved_Model/labels/')
# with open('labels.npy','wb') as fData:
#      arr =[]
#      for f in files:
#         for a in np.load(os.path.join("Saved_Model/labels", f)):
#             arr.append(a)
#      arr = np.array(arr)
#      np.save(fData, arr)

files = os.listdir('Saved_Model/train/')
with open('train.npy','wb') as fData:
     arr =[]
     for f in files:
        for a in np.load(os.path.join("Saved_Model/train", f)):
            arr.append(a)
     arr = np.array(arr)
     np.save(fData, arr)

        
