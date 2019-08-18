
# coding: utf-8
# In[ ]:
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib
from scipy import stats
matplotlib.use('agg')
print(matplotlib.__version__)
import matplotlib.pyplot as plt
import io
import os
import sys
import time
import json
#from IPython.display import display
from time import strftime, gmtime
from scipy.stats.mstats import mode
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import datetime, time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.decomposition import TruncatedSVD


def feature_imputation_main(data_df, feat_list ,impute_local_key_list=[],missing_val_list=[],impute_type = 'global',replace_value=0):
    """
    Runs imputation based "impute_type" method  
    
    """
    
    # count number of columns in data
    print("Columns in feature list : ", len(feat_list))
    print("Imputation type : ", impute_type)       

    if impute_type == 'fixed':
        temp_df = data_df.copy()
        imputed_df = pd.DataFrame(run_imputation_fixed(data_df=temp_df,feat_list=feat_list,missing_val_list=missing_val_list,replace_value=replace_value))
        print(imputed_df.head())

    if impute_type == 'global':
        temp_df = data_df.copy()
        imputed_df = pd.DataFrame(run_imputation_global(data_df=temp_df,feat_list=feat_list,missing_val_list=missing_val_list))
        print(imputed_df.head())

    if impute_type == 'local':
        temp_df = data_df.copy()
        imputed_df = pd.DataFrame(run_imputation_local(data_df=temp_df,feat_list=feat_list,impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list))
        print(imputed_df.head())

    if impute_type in ('knn'):
        temp_df = data_df.copy()
        imputed_df = pd.DataFrame(run_feature_imputation_knn(feat_list=feat_list,missing_val_list=missing_val_list,data_df=temp_df,k_neighbors=50))        
        print(imputed_df.head())
    
    print("IMPUTATION COMPLETE")
    return(imputed_df)  

##################################################################################################

def run_imputation_fixed(data_df,feat_list,missing_val_list=[],replace_value=0):
    """    
    Performs imputation on data columns indicated by parameter "feat_list"
    Performs "fixed" imputation indicated by parameter "impute_type"
    """
    #check if missing_val_list is empty
    if len(missing_val_list)==0:
        print("Please provide missing_val_list")
        return(0)    

    data_df_imputed = data_df
    data_df_imputed.loc[:,'Row_ID'] = range(1, len(data_df_imputed) + 1)
    print("Starting ... Fixed Imputation")

    for i in range(0,len(feat_list)):
        temp_df = pd.DataFrame(data_df.loc[:,[feat_list[i]]].copy())           
        temp_df.loc[:,'Row_ID'] = range(1, len(temp_df) + 1)     
        miss_rate = 100*(sum(temp_df[feat_list[i]].isin(missing_val_list)) / temp_df.shape[0]) 
        if miss_rate>0:
            # run imputation on feature            
            imputed_df = pd.DataFrame(feature_imputation_fixed(data_df=temp_df,var_name=feat_list[i],missing_val_list=missing_val_list))
            data_df_imputed = data_df_imputed.drop([feat_list[i]],axis=1)
            data_df_imputed = data_df_imputed.merge(imputed_df.loc[:,['Row_ID',feat_list[i]+'_imp']],on='Row_ID',how='left')
            data_df_imputed.rename(columns = {feat_list[i]+'_imp': feat_list[i]}, inplace=True)
        else :
            print(" ---- Feature : ",feat_list[i])
            print(" ---- No missing value")
            
    return(data_df_imputed)         


def feature_imputation_fixed(data_df,var_name,missing_val_list=[],replace_value=0):
    """    
    Performs imputation on data column indicated by parameter "var_name"
    Performs "fixed" imputation indicated by parameter "impute_type"
    """
    # Print the missing rate
    miss_rate = 100*(sum(data_df[var_name].isin(missing_val_list)) / data_df.shape[0])
    print('--- Feature to impute :',var_name)
    print('--- Miss rate before imputation :', miss_rate, '%')

    data_df.loc[:,var_name+ '_imp'] = data_df.loc[:,var_name].copy()
    data_df.loc[data_df[var_name].isin(missing_val_list),var_name+ '_imp']=replace_value            

    # Recalculate missing rate
    miss_rate = 100*(sum(data_df[var_name+'_imp'].isin(missing_val_list)) / data_df.shape[0])
    print('--- Miss rate after imputation :', miss_rate, '%')     
    return(data_df)

##################################################################################################

def run_imputation_global(data_df,feat_list,missing_val_list=[]):
    """    
    Performs imputation on data columns indicated by parameter "feat_list"
    Performs "global" imputation indicated by parameter "impute_type"
    """
    #check if missing_val_list is empty
    if len(missing_val_list)==0:
        print("Please provide missing_val_list")
        return(0)

    data_df_imputed = data_df
    data_df_imputed.loc[:,'Row_ID'] = range(1, len(data_df_imputed) + 1)
    print("Starting ... Global Imputation")

    for i in range(0,len(feat_list)):
        temp_df = pd.DataFrame(data_df.loc[:,[feat_list[i]]].copy())           
        temp_df.loc[:,'Row_ID'] = range(1, len(temp_df) + 1)     
        miss_rate = 100*(sum(temp_df[feat_list[i]].isin(missing_val_list)) / temp_df.shape[0]) 
        if miss_rate>0:
            # run imputation on feature            
            imputed_df = pd.DataFrame(feature_imputation_global(data_df=temp_df,var_name=feat_list[i],missing_val_list=missing_val_list))
            data_df_imputed = data_df_imputed.drop([feat_list[i]],axis=1)
            data_df_imputed = data_df_imputed.merge(imputed_df.loc[:,['Row_ID',feat_list[i]+'_imp']],on='Row_ID',how='left')
            data_df_imputed.rename(columns = {feat_list[i]+'_imp': feat_list[i]}, inplace=True)
        else :
            print(" ---- Feature : ",feat_list[i])
            print(" ---- No missing value")
            
    return(data_df_imputed)         

def feature_imputation_global(data_df,var_name,missing_val_list=[]):
    """    
    Performs imputation on data column indicated by parameter "var_name"
    Performs "global" imputation depending on parameter "impute_type"
    """        
    # Print the missing rate
    miss_rate = 100*(sum(data_df[var_name].isin(missing_val_list)) / data_df.shape[0])
    print('--- Feature to impute :',var_name)
    print('--- Miss rate before imputation :', miss_rate, '%')

    # If feature distribution is more or less symmetric then inpute with mean
    if data_df[var_name].dtype == int or data_df[var_name].dtype == np.int64 or data_df[var_name].dtype == float or data_df[var_name].dtype == np.float64:                
        print('--- Feature is continuous valued ')

        # remove the missing values before computing skewness
        feat_cln = pd.DataFrame(data_df.loc[~data_df[var_name].isin(missing_val_list),var_name].copy())
        feat_skew= stats.skew(feat_cln, bias=False)

        # we impute with mean as long as skewness is between -0.005 to 0.005
        if feat_skew >= -0.005 and feat_skew <= 0.005:    
            print('--- Feature is symmetric, imputation will be done with mean ')
            gbl_mean = feat_cln.mean()
            data_df.loc[:,var_name+ '_imp'] = data_df.loc[:,var_name].copy()
            data_df.loc[data_df[var_name+ '_imp'].isin(missing_val_list), var_name+ '_imp'] = gbl_mean[0]

        elif feat_skew < -0.005 or feat_skew > 0.005:
            print('--- Feature is asymmetric, imputation will be done with median ')
            gbl_median = feat_cln.median()
            data_df.loc[:,var_name+ '_imp'] =  data_df.loc[:,var_name].copy()
            data_df.loc[data_df[var_name+ '_imp'].isin(missing_val_list), var_name+ '_imp'] = gbl_median[0]

    if data_df[var_name].dtype == object:
        print('--- Feature is categorical valued ')

        # remove the missing values before computing mode
        feat_cln = pd.DataFrame(data_df.loc[~data_df[var_name].isin(missing_val_list),var_name].copy())
        print('--- Feature is categorical, imputation will be done with mode ')
        gbl_mode = feat_cln.loc[:,var_name].mode()
        print("mode: ",gbl_mode[0])
        data_df.loc[:,var_name+ '_imp'] = data_df.loc[:,var_name].copy()
        data_df.loc[data_df[var_name+ '_imp'].isin(missing_val_list), var_name+ '_imp'] = gbl_mode[0]

    # Recalculate missing rate
    miss_rate = 100*(sum(data_df[var_name+'_imp'].isin(missing_val_list)) / data_df.shape[0])
    print('--- Miss rate after imputation :', miss_rate, '%')
    return(data_df)

##################################################################################################

def run_imputation_local(data_df,feat_list,impute_local_key_list=[],missing_val_list=[]):
    """    
    Performs imputation on data columns indicated by parameter "feat_list"
    Performs "global" imputation indicated by parameter "impute_type"
    """
    #check if missing_val_list is empty
    if len(missing_val_list)==0:
        print("Please provide missing_val_list")
        return(0)

    #check if impute_type = 'local' and impute_local_key_list is empty
    if len(impute_local_key_list)==0:
        print("Please provide key list ")
        return(0)

    data_df_imputed = data_df
    data_df_imputed.loc[:,'Row_ID'] = range(1, len(data_df_imputed) + 1)
    print("Starting ... Local Imputation")

    for i in range(0,len(feat_list)):
        #print(" ---- Feature : ",feat_list[i])
        if feat_list[i] not in impute_local_key_list:
            temp_df = pd.DataFrame(data_df.loc[:,impute_local_key_list + [feat_list[i]]].copy())           
            temp_df.loc[:,'Row_ID'] = range(1, len(temp_df) + 1)     
            miss_rate = 100*(sum(temp_df[feat_list[i]].isin(missing_val_list)) / temp_df.shape[0]) 
            if miss_rate>0:
                # run imputation on feature            
                imputed_df = pd.DataFrame(feature_imputation_local(data_df=temp_df,var_name=feat_list[i],impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list))
                data_df_imputed = data_df_imputed.drop([feat_list[i]],axis=1)
                data_df_imputed = data_df_imputed.merge(imputed_df.loc[:,['Row_ID',feat_list[i]+'_imp']],on='Row_ID',how='left')
                data_df_imputed.rename(columns = {feat_list[i]+'_imp': feat_list[i]}, inplace=True)
            else :
                print(" ---- Feature : ",feat_list[i])
                print(" ---- No missing value")
            
    return(data_df_imputed)         
        
def feature_imputation_local(data_df,var_name,impute_local_key_list=[],missing_val_list=[]):
    """    
    Performs imputation on data column indicated by parameter "var_name"
    Performs "local" imputation depending on parameter "impute_type"
    """
                    
    # local imputation
    print(" ---- Feature : ",var_name)
    print("--- Columns Type :",data_df[var_name].dtype)
        
    # Print the missing rate
    miss_rate = 100*(sum(data_df[var_name].isin(missing_val_list)) / data_df.shape[0])
    print('--- Feature to impute :',var_name)
    print('--- Miss rate before imputation :', miss_rate, '%')

    if data_df[var_name].dtype == int or data_df[var_name].dtype == np.int64 or data_df[var_name].dtype == float or data_df[var_name].dtype == np.float64:
        print('--- Feature is continuous valued, imputation will be done with median ')

        # remove the missing values before computing local median
        var_list = impute_local_key_list + [var_name]
        feat_cln = pd.DataFrame(data_df.loc[~data_df[var_name].isin(missing_val_list),var_list].copy())

        # we impute with local median
        lcl_median = feat_cln.groupby(impute_local_key_list)[var_name].median().reset_index()
        #print("Imputing with local median value :", lcl_median)
        lcl_median.columns = impute_local_key_list + [var_name+'_imp']

        data_df = data_df.merge(lcl_median, on = impute_local_key_list, how='left')
        data_df.loc[data_df[var_name].isin(missing_val_list), var_name+ '_imp'] =  data_df.loc[data_df[var_name].isin(missing_val_list),var_name+ '_imp'].copy()

    if data_df[var_name].dtype == object:
        print('--- Feature is categorical valued, imputation will be done with mode ')

        # remove the missing values before computing local mode
        var_list = impute_local_key_list + [var_name]
        feat_cln = pd.DataFrame(data_df.loc[~data_df[var_name].isin(missing_val_list),var_list].copy())

        # we impute with local mode
        lcl_mode=feat_cln.groupby(impute_local_key_list).agg(lambda x:x.value_counts().index[0]).reset_index()
        lcl_mode.columns = impute_local_key_list + [var_name+'_imp']

        data_df = data_df.merge(lcl_mode, on = impute_local_key_list, how='left')
        data_df.loc[data_df[var_name].isin(missing_val_list), var_name+ '_imp'] = data_df.loc[data_df[var_name].isin(missing_val_list),var_name+ '_imp'].copy() 

    # Recalculate missing rate
    miss_rate = 100*(sum(data_df[var_name+'_imp'].isin(missing_val_list)) / data_df.shape[0])
    print('--- Miss rate after imputation :', miss_rate, '%')     

    if miss_rate>0:
        missing_rec = data_df[data_df.isnull().any(axis=1)]
        print("Records with missing values :",missing_rec)           
    return(data_df)

##################################################################################################
def run_feature_imputation_knn(feat_list,missing_val_list,data_df,k_neighbors):
    """
    
    Performs imputation on data column indicated by parameter "var_name"
    Performs "KNN" imputation depending on parameter "impute_type"

    """
    #print(data_df.shape)
    data_df.loc[:,'Row_ID'] = range(1, len(data_df) + 1)
    data_df_imputed_f=data_df
    print("Starting ... KNN Imputation")

    knn_dict=find_knn(feat_list,missing_val_list,data_df,k_neighbors)
    
    for j in knn_dict.keys():
        missing_feat_df = pd.DataFrame({'missing':data_df.iloc[j].isnull()})
        missing_feat = missing_feat_df.index[missing_feat_df['missing']== True]
        print(j,"-",missing_feat)
        print('knn dict: ',knn_dict[j])
        
        # get the knn records
        x=knn_dict[j].tolist()
        temp_df= data_df.iloc[x].copy()

        #print('Missing record row id :',data_df.loc[data_df['Row_ID']==(j+1),'Row_ID'])
        #print('Missing record  :',j)
        #print('check',temp_df.shape)
        
        # add the record with missing values
        temp_df = pd.concat([temp_df,pd.DataFrame(data_df.iloc[[j]])],axis=0)
        #print('check',temp_df.shape)

        for i in range(0,len(missing_feat)):
            print("Imputing missing value in row ", j, " feature ",missing_feat[i])
            #print("check :",temp_df.shape)
            
            #check
            #print(data_df.iloc[j])
            #print(data_df.loc[data_df['Row_ID']==(j+1),missing_feat[i]])
                        
            # check feature missrate in knn records
            print("Miss rate for feature ",missing_feat[i]," in KNN records ",(temp_df[missing_feat[i]].isna().sum())/temp_df.shape[0])
            print("Miss records for feature ",missing_feat[i]," in KNN records ",(temp_df[missing_feat[i]].isna().sum()))

            miss_rate=(temp_df[missing_feat[i]].isna().sum())/temp_df.shape[0]
            if miss_rate > 0.5:
                print("KNN record have more than 50% missrate for feature ",missing_feat[i]," other imputation methods might make more sense for this feature")
            
            data_df_imputed = temp_df
            #print('check shape:',data_df_imputed.shape)
            #print("post imputed value :",temp_df.loc[temp_df['Row_ID']==(j+1),['Row_ID',missing_feat[i]]])  
            #print('pre imp',data_df_imputed.head())
            imputed_df = pd.DataFrame(feature_imputation_global(data_df=temp_df,var_name=missing_feat[i],missing_val_list=missing_val_list))
            #print(imputed_df)
            data_df_imputed = data_df_imputed.drop([missing_feat[i],missing_feat[i]+'_imp'],axis=1)
            data_df_imputed = data_df_imputed.merge(imputed_df.loc[:,['Row_ID',missing_feat[i]+'_imp']],on='Row_ID',how='left')
            data_df_imputed.rename(columns = {missing_feat[i]+'_imp': missing_feat[i]}, inplace=True)
            #print('post imp',data_df_imputed.head())
            
            # impute missing value in original data
            print("pre imputed value :",data_df.loc[data_df['Row_ID']==(j+1),['Row_ID',missing_feat[i]]])
            print("post imputed value :",data_df_imputed.loc[data_df_imputed['Row_ID']==(j+1),['Row_ID',missing_feat[i]]])            
            
            data_df_imputed_f.loc[data_df_imputed_f['Row_ID']==(j+1),missing_feat[i]] = data_df_imputed.loc[data_df_imputed['Row_ID']==(j+1),missing_feat[i]].values[0]
            print("post imputed value :",data_df_imputed_f.loc[data_df_imputed_f['Row_ID']==(j+1),['Row_ID',missing_feat[i]]])            
    
    #check for missing values 
    print("Miss rate post imputtaion ",(data_df_imputed_f.isna().sum().sum()))
    return(data_df_imputed_f)       
        

def find_knn(feat_list,missing_val_list,data_df,k_neighbors):
    """    
    Finds K nearest neighbours for each record with missing values 
    
    """
    
    #check if missing_val_list is empty
    if len(missing_val_list)==0:
        print("Please provide missing_val_list")
        return(0)   

    # Print the missing rate
    miss_rate = 100*(data_df.isin(missing_val_list).sum().sum())/ (data_df.shape[0]*data_df.shape[1])
    print('--- Overall miss rate before imputation :', miss_rate, '%')

    missing_rec = data_df[data_df.isnull().any(axis=1)].index.tolist()
    print("Number of records with missing values :",len(missing_rec))

    if miss_rate > 0:            
        print("--- Running global imputation engine to fill up missing values  ")
        data_df_imp= feature_imputation_main(data_df=data_df,feat_list=feat_list,missing_val_list=missing_val_list,impute_type = 'global')


    # Recalculate missing rate
    miss_rate = 100*(data_df_imp.isin(missing_val_list).sum().sum())/ (data_df.shape[0]*data_df.shape[1])
    print('--- Miss rate after imputation :', miss_rate, '%')     

    # check column types
    col_types = pd.DataFrame({'col_type':data_df.dtypes})
    print(col_types.groupby(['col_type'])['col_type'].agg(['count']))

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    col_numeric = data_df.select_dtypes(include=numerics).columns
    print("Numeric Columns :",col_numeric)

    col_categ = data_df.columns[col_types['col_type'] == 'object']
    print("Categorical Columns :",col_categ)

    # check if all columns are numeric
    is_all_numeric = len(col_numeric) == data_df.shape[1]
    #print( len(col_numeric), data_df.shape[1])
    print(is_all_numeric)

    # check if all columns are categorical
    is_all_categ = len(col_categ) == data_df.shape[1]
    #print(len(col_categ), data_df.shape[1])
    print(is_all_categ)

    # check if all columns are mixed in type
    is_mixed = is_all_numeric == False and is_all_categ == False
    print(is_mixed)

    numeric_distance = "euclidean"
    categ_distance = "jaccard"
    number_of_observations = data_df.shape[0]
    number_of_variables = data_df.shape[1]
    print("number_of_observations :",number_of_observations)
    print("number_of_variables : ",number_of_variables)

    n=data_df_imp.shape[0]
    #n=2000
    data_dist = data_df_imp.loc[0:n,:].copy()
    missing_rec = data_dist[data_dist.isnull().any(axis=1)].index.tolist()
    print("Number of records with missing values :",len(missing_rec))

    data_numeric=pd.DataFrame()
    data_categ=pd.DataFrame()

    print("col_numeric :",col_numeric)
    print("col_categ :",col_categ)

    if is_all_numeric:
        print("All columns are numeric")
        data_numeric = data_dist.loc[:,col_numeric].copy()
        data_numeric = (data_numeric - data_numeric.mean())/ (data_numeric.max() - data_numeric.min())
        print("Data Numeric Dim: ",data_numeric.shape)
    else :
        if is_all_categ:
            print("All columns are categorical")
            data_categ = data_dist.loc[:,col_categ].copy()
            data_categ = pd.get_dummies(data_categ)
            print("Data Categorical Dim: ",data_categ.shape)
        else:
            if is_mixed == True:
                print("Columns are mixed")
                data_numeric = data_dist.loc[:,col_numeric].copy()
                data_numeric = (data_numeric - data_numeric.mean())/ (data_numeric.max() - data_numeric.min())
                data_categ = data_dist.loc[:,col_categ].copy()
                missing_rec = data_categ[data_categ.isnull().any(axis=1)].index.tolist()
                data_categ = pd.get_dummies(data_categ)
                print("Data Numeric Dim: ",data_numeric.shape)
                print("Data Categorical Dim: ",data_categ.shape)

    # find all records with missing data
    missing_rec = data_df[data_df.isnull().any(axis=1)].index.tolist()
    #n=2000

    chunk=200
    #number_of_observations=400
    number_of_observations=data_df.shape[0]
    #k_neighbors=10

    dist_array_full =[]
    knn_dict ={}
    # Find top k neighbours for each record with missing value
    for i in missing_rec:
        if i<=n:
            for j in range(0,number_of_observations,chunk):
                print(i, j, j+chunk)

                dist_array_part = find_dist(data_categ=data_categ,data_numeric=data_numeric,numeric_distance='euclidean',categ_distance='jaccard',col_numeric=col_numeric,col_categ=col_categ,target=i,nbd_start=j,nbd_end=min((number_of_observations-1),(j+chunk)),is_mixed=is_mixed,is_all_numeric=is_all_numeric,is_all_categ=is_all_categ,number_of_observations=chunk,number_of_variables=number_of_variables)
                #print("Neighbour Distances :",dist_array_part)
                #print("Neighbour Distances :",dist_array_part.shape)
                dist_k_neighbour_index_part = dist_array_part[0].argsort()[0:k_neighbors]
                # remove i
                dist_k_neighbour_index_part=dist_k_neighbour_index_part[dist_k_neighbour_index_part!=i]
                print("Top K Neighbour Indexes :", dist_k_neighbour_index_part)
                dist_k_neighbour_val_part = dist_array_part[0,dist_k_neighbour_index_part]
                print("Top K Neighbours :", dist_k_neighbour_val_part)

                if j==0:
                    dist_k_neighbour_index_full = dist_k_neighbour_index_part
                    dist_k_neighbour_val_full = dist_k_neighbour_val_part

                else:
                    dist_k_neighbour_index_full = np.append(dist_k_neighbour_index_full,dist_k_neighbour_index_part)
                    dist_k_neighbour_val_full = np.append(dist_k_neighbour_val_full,dist_k_neighbour_val_part)

            dist_k_neighbour_index = dist_k_neighbour_val_full.argsort()[0:k_neighbors]  
            dist_k_neighbour_val = dist_k_neighbour_val_full[dist_k_neighbour_index]
            print("Final Top K Neighbours :", dist_k_neighbour_index_full[dist_k_neighbour_index])
            print("Final Top K Neighbours Distances:", dist_k_neighbour_val)
        knn_dict[i]=dist_k_neighbour_index_full[dist_k_neighbour_index]
            
    #print(knn_dict)    
    return(knn_dict)
    

def find_dist(data_categ,data_numeric,numeric_distance,categ_distance,col_numeric,col_categ,target,nbd_start,nbd_end,is_mixed,is_all_numeric,is_all_categ,number_of_observations,number_of_variables):
    # Find euclidian distance between continuous variables
    if is_all_numeric:
        print("All columns are numeric")
        x1_numeric = data_numeric.loc[[target]].copy()
        x2_numeric = data_numeric.loc[nbd_start:nbd_end,:].copy()
        dist_mat = cdist(x1_numeric, x2_numeric, metric=numeric_distance)
        print(dist_mat.shape)
    elif is_all_categ:
        print("All columns are categorical")
        x1_categ = data_categ.loc[[target]].copy()
        x2_categ = data_categ.loc[nbd_start:nbd_end,:].copy()
        dist_mat = cdist(x1_categ, x2_categ, metric=categ_distance)
        print(dist_mat.shape)
    elif is_mixed:
        print("Columns are mixed")
        x1_numeric = data_numeric.loc[[target]].copy()
        x2_numeric = pd.DataFrame(data_numeric.loc[nbd_start:nbd_end,:].copy())
        #print("x1_numeric ",x1_numeric)
        #print("x2_numeric ",x2_numeric)

        dist_numeric = cdist(x1_numeric, x2_numeric, metric=numeric_distance)
        #print("dist_numeric ",dist_numeric)
        x1_categ = data_categ.loc[[target]].copy()
        x2_categ = data_categ.loc[nbd_start:nbd_end,:].copy()
        dist_categ = cdist(x1_categ, x2_categ, metric=categ_distance)
        #print("dist_categ ",dist_categ)
        #print(dist_categ.shape)
        #print(dist_numeric.shape)             
        #print(dist_categ)
        dist_mat = np.array([[1.0*(dist_numeric[i, j] * len(col_numeric) + dist_categ[i, j] *
                           len(col_categ)) / number_of_variables for j in range(number_of_observations)] for i in range(1)])
        print("Dist Mat shape:",dist_mat.shape)

    return(dist_mat)

    