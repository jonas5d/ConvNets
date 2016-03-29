import pandas as pd
from scipy import misc
import numpy as np
from collections import Counter
import net_helper
import argparse
from nolearn.lasagne.visualize import plot_loss
import nolearn.lasagne.visualize
import os
from sklearn.cross_validation import KFold

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
parser = argparse.ArgumentParser()

parser.add_argument("folds",type=str,
                    help='The folds that should be run')
parser.add_argument("logFolder",type=str,
                    help='The Folder that shall be used for output')
parser.add_argument("epochs",type=str,
                    help='Number of epochs.')
parser.add_argument("input_workflow",type=str,
                    help='textfile containing the workflow for the processing of input data')
parser.add_argument("net_id",type=str,
                    help='The Net ID')
parser.add_argument("--seed", nargs=1, type=int,
                    help='Seed Number random if not supplied')
parser.add_argument('--cudaLayers', action='store_true')
parser.add_argument('--sbatch', action='store_true')

args = parser.parse_args()
cudaLayers = True if args.cudaLayers else False
folds = np.array(args.folds.split(',')).astype(np.int32)

base_path = '../../../../Desktop/image_files/'
images_path = base_path + 'processed_images_128/'
csv_path = base_path + 'sdo_xml_features_with_url-images.csv'

category_to_detect = 0

df = pd.read_csv(csv_path,sep=';',header=0)
cats = pd.unique(df.classificationCategoryCode.ravel())
images = []
targets = []
for index,row in df.iterrows():
  if index%100==0:
    if index%1000==0:
      print '|',
    else:
      print '.',

  if row['Image Relative Path'] == 'sdo_xml_features_with_url.csv-images/04018087010078-10000187-image-png':
    continue
  if row['Image Relative Path'].endswith('jpeg'):
    suffix = '.jpg'
  elif row['Image Relative Path'].endswith('png'):
    suffix = '.png'
  else: 
    suffix='.tiff'

  img_path = images_path + str(row['gtin']) + suffix
  image = misc.imread(img_path)
  images.append(image[:,:,0:3])
  target = np.where(cats==row['classificationCategoryCode'])[0][0]
  if target==0:
    targets.append(1)
  else:
    targets.append(0)
print ''
x,y = np.array(images),np.array(targets)
X,Y = np.reshape(x,[x.shape[0],x.shape[3],x.shape[1],x.shape[2]]),y
theSeed = args.seed[0]
kf = KFold(X.shape[0],10,shuffle=True,random_state=theSeed)

mccs = []
accuracies = []
precisions = []
recalls = []
np.set_printoptions(threshold=np.nan)
fold=0

for train_index,test_index in kf:
  if not fold in folds:
    fold+=1
    continue
  logPath = args.logFolder+"/fold_"+str(fold)
  if not os.path.exists(logPath):
    os.makedirs(logPath)
    
  X_train,X_test = X[train_index], X[test_index]
  Y_train,Y_test = Y[train_index],Y[test_index]
  net = net_helper.create_net(X_train.astype(np.float32),
                              X_test.astype(np.float32),
                              Y_train.astype(np.int32),
                              Y_test.astype(np.int32),
                              epochs=200,
                              cuda_layers=False)
  
  net.fit(X.astype(np.float32), Y.astype(np.int32))
  preds_net = net.predict(X_test.astype(np.float32))

  wrong_in_test_net = np.where(preds_net != Y_test)
  wrong_in_data_net = test_index[wrong_in_test_net]

  mcc_net = matthews_corrcoef(Y_test,preds_net)
  acc_net = accuracy_score(Y_test,preds_net)
  precision_net = precision_score(Y_test,preds_net)
  recall_net = recall_score(Y_test,preds_net)

  print("----------------------------------------------------------------")
  print("NET: Matthews Correlation Coefficient: \t" + str(mcc_net))

  print("NET: Accuracy: \t\t\t\t" + str(acc_net))

  print("NET: Precision: \t\t\t\t" + str(precision_net))

  print("NET: Recall: \t\t\t\t" + str(recall_net))

  netFile = logPath + "/net"
  net.save_params_to(netFile)
  plt = plot_loss(net)
  plt.savefig(logPath + "/valid_train_loss.png")
  numbers_net = [mcc_net,acc_net,precision_net,recall_net]
  numbers_file_net = open(logPath + "/net_numbers.csv",'wb')
  wr_net = csv.writer(numbers_file_net)
  wr_net.writerow(numbers_net)
  numbers_file_net.close()
  f = open(logPath+'/net_info.txt','w')
  f.write("Net layers:\n")
  f.write(str(net.layers))
  f.write('\n')
  f.write("Seed: ")
  f.write(str(args.seed))
  f.close()

  fold+=1

