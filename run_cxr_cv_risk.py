"""Main testing script for the composite outcome experiment. Purpose is to determine whether using composite outcomes improves DL performance for prognosis
<image_dir> - Directory where images to run the model on are located
<model_path> - Absolute or relative file path to the .pth model file (or the prefix excluding the _0 for an ensemble model)
<output_file> - Absolute or relative file path to where the output dataframe should be written

Usage:
  run_cxr_cv_risk.py <image_dir> <model_path> <output_file> [--gpu=GPU]
  run_cxr_cv_risk.py (-h | --help)
Examples:
  run_model.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                    Show this screen.
  --gpu=GPU                    Which GPU to use? [Default:None]
"""


import warnings
warnings.simplefilter(action='ignore')
import sys


import os
from docopt import docopt
import pandas as pd

import pretrainedmodels
from sklearn.metrics import *
import math
import time


num_workers = 16
if __name__ == '__main__':

    arguments = docopt(__doc__)
  
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    #Set model path 
    mdl_path = arguments['<model_path>']
    
    #Read specs file
    specs = pd.read_csv("CXR_CV_Risk_Specs.csv",comment='#')
    
    if(arguments['--gpu'] is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments['--gpu']
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import fastai
    from fastai.vision.all import *
    import SimpleArchs

    ###set model architecture
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
    ###Results are stored in this final_df - Dummy is a dummy nonsense variable to act as the fake "target variable"
    ### valid_col is True for all samples except for an artificial sample at the end (since there needs to be a "training set" included too)
    output_df = pd.DataFrame(columns = ['File','Dummy','Prediction'])
    output_df['File'] = files
    output_df['Dummy'] = np.random.random_sample(len(files))
    col = 'Dummy'
    output_df['valid_col'] = np.repeat(True,output_df.shape[0])

    #Create an additional "fake" image to act as the training dataset
    final_df = pd.concat([output_df,output_df.iloc[[-1]]],ignore_index=True)
    #final_df.valid_col[final_df.shape[0]-1] = False
    final_df.loc[final_df.shape[0]-1,'valid_col'] = False

        
        


    block = RegressionBlock

    #Number of models = number of rows in dataframe
    ensemble = specs.shape[0]
    mbar = master_bar(range(ensemble))

    #Create empty array of num_images x 20 (20 model-ensemble)
    pred_arr = np.zeros((final_df.shape[0]-1,ensemble))
    for x in mbar:
        out_nodes = int(specs.Num_Classes[x])
        manual = False
        size = int(specs.Image_Size[x])
        bs,val_bs = 4,4
        if(int(specs.Normalize[x])==0):
            imgs = ImageDataLoaders.from_df(df=final_df,path=image_dir,label_col=col,y_block=block,bs=bs,val_bs=val_bs,valid_col="valid_col",item_tfms=Resize(size),batch_tfms=None)
        else:
            imgs = ImageDataLoaders.from_df(df=final_df,path=image_dir,label_col=col,y_block=block,bs=bs,val_bs=val_bs,valid_col="valid_col",item_tfms=Resize(size),batch_tfms=[Normalize.from_stats(*imagenet_stats)])
   
        #Set architecture of current model according to specs file
        try:
            #import pdb; pdb.set_trace()
            m = specs.Architecture[x].lower()
            if(m=="inceptionv4"):
                def get_model(pretrained=True, model_name = 'inceptionv4', **kwargs ): 
                    if pretrained:
                        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
                    else:
                        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
                    return arch

                def get_cadene_model(pretrained=True, **kwargs ): 
                    return fastai_inceptionv4
                custom_head = create_head(nf=2048*2, n_out=37) 
                fastai_inceptionv4 = nn.Sequential(*list(get_model(model_name = 'inceptionv4').children())[:-2],custom_head) 
            ###Based on the input model, create a cnn learner object
            elif(m=="resnet50"):
                mdl = fastai.vision.models.resnet50
            elif(m=="resnet34"):
                mdl = fastai.vision.models.resnet34
            elif(m=="resnet16"):
                mdl = fastai.vision.models.resnet16
            elif(m=="resnet101"):
                mdl = fastai.vision.models.resnet101
            elif(m=="resnet152"):
                mdl = fastai.vision.models.resnet152
            elif(m=="densenet121"):
                mdl = fastai.vision.models.densenet121
            elif(m=="densenet169"):
                mdl = fastai.vision.models.densenet169
            elif(m=="age"):
                mdl=fastai.vision.models.resnet34
            elif(m=="larget"):
                manual = True
                mdl = SimpleArchs.get_simple_model("LargeT",out_nodes)
            elif(m=="largew"):
                manual = True
                mdl = SimpleArchs.get_simple_model("LargeW",out_nodes)
            elif(m=="small"):
                manual = True
                mdl = SimpleArchs.get_simple_model("Small",out_nodes)
            elif(m=="tiny"):
                manual = True

                mdl = SimpleArchs.get_simple_model("Tiny",out_nodes)
            elif(m=="age"):
                mdl = fastai.vision.models.resnet34
            else:
                print("Sorry, model: " + m + " is not yet supported... coming soon!")
                quit()

            if(m=='inceptionv4'):
                learn = cnn_learner(imgs, get_cadene_model,n_out=out_nodes)
            elif(manual):
                learn = Learner(imgs,mdl)
            else:
                learn = cnn_learner(imgs, mdl,n_out=out_nodes)
            if(m=="age"):
                numFeatures = 16
                if(torch.has_cuda):
                    learn.model[1] = nn.Sequential(*learn.model[1][:-5],nn.Linear(1024,512,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(p=0.5),
nn.Linear(512,numFeatures,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(numFeatures),
                                nn.Linear(numFeatures,out_nodes,bias=True)).cuda()
                else:
                    learn.model[1] = nn.Sequential(*learn.model[1][:-5],nn.Linear(1024,512,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(p=0.5),
nn.Linear(512,numFeatures,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(numFeatures),
                                nn.Linear(numFeatures,out_nodes,bias=True))
        except:
            print("Architecture not found for model #: " + str(x))
            sys.exit(0)






            
        learn.model_dir = "."

        learn.model = nn.Sequential(learn.model,nn.Sigmoid(),nn.Flatten(start_dim=0))

        learn.load(os.path.join(os.path.abspath(os.getcwd()),mdl_path + "_" + str(x)))
        learn.remove_cb(ProgressCallback)
      
        preds,y = learn.get_preds(ds_idx=1,reorder=False)
        

        ###output predictions as column with model name
        pred_arr[:,x] = np.array(preds)
    weights = [0,0,0.835725,0,1.823355]
    predictions = np.matmul(pred_arr,np.array(weights)) - 2.694890
    output_df['CXR_CV_Risk'] = np.divide(np.exp(predictions),1+np.exp(predictions))
    output_df = output_df.drop(["valid_col","Dummy","Prediction"],axis=1)
    output_df.to_csv(arguments['<output_file>'],index=False)
