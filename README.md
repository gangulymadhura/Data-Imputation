# Data-Imputation
# A module of python functions to impute data

This module offers 4 ways of imputing data which are as follows:
--------------------------------------------------------------------------------------------------------------------------------

1. Fixed value imputation - impute all missing values with a single value.

2. Global imputation - impute missing values with mean or median for numeric data types and mode for catgorical data types. The function accesses the skew in numeric features to decide wether mean or median is the choice of imputation.

3. Local imputation - does same as global imputation but computes the mean, median or mode based segment of the data to which it belongs. Eg, impute missing values in store sales by mean or median sales from the same state and not the entire country.

4. KNN imputation - as the name suggests, this method finds the K nearest neighbours for each record with missing value and imputes the missing values based on the K nearest neighbours. The functions checks for sparsity in K nearest neighbour and raises warning message if the feature to be used for imputation has over 50% miss rate.

How to use ?
----------------------------------------------------------------------------------------------------------------------------------

There are 6 parameters in all :
1.data_df : data to be imputed
2.feat_list : list of features that need to be imputed
3.impute_local_key_list : only needed for "local" imputation
4.missing_val_list : list of values considered to be missing values 
5.impute_type : 'fixed','global','local','KNN'
6.replace_value : imputation value, only needed for "fixed" imputation
7.k_neighbors : no. of nearest neighbours to be considered (default 50), only needed for "knn" imputation


How to call ?
----------------------------------------------------------------------------------------------------------------------------------
# Example - Fixed Imputation

from Feature_Imputation_Engine import *
import pandas as pd
import numpy as np
import matplotlib
import math

## Set Parameters
data_df = pd.read_csv('merchants.csv')
impute_type = 'fixed'  
feat_list= list(data_df.columns)
replace_value=0

data_df_imp= feature_imputation_main(data_df=data_df, feat_list=feat_list,impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list,impute_type = impute_type)

##Check for missing values
missing_val = data_df_imp.isnull().sum().sum()
print("Missing values :",missing_val)

_______________________________________________________________________________________________________________________________________
# Example - Local Imputation

from Feature_Imputation_Engine import *
import pandas as pd
import numpy as np
import matplotlib
import math

##Set Parameters
data_df = pd.read_csv('merchants.csv')
impute_type = 'local'  
impute_local_key_list = ['merchant_category_id']
feat_list= list(data_df.columns)
replace_value=0

data_df_imp= feature_imputation_main(data_df=data_df, feat_list=feat_list,impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list,impute_type = impute_type)

##Check for missing values
missing_val = data_df_imp.isnull().sum().sum()
print("Missing values :",missing_val)

_______________________________________________________________________________________________________________________________________
# Example - Global Imputation

from Feature_Imputation_Engine import *
import pandas as pd
import numpy as np
import matplotlib
import math

##Set Parameters
data_df = pd.read_csv('merchants.csv')
impute_type = 'knn'  
impute_local_key_list = ['merchant_category_id']
feat_list= list(data_df.columns)
replace_value=0

data_df_imp= feature_imputation_main(data_df=data_df, feat_list=feat_list,impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list,impute_type = 'global')

##Check for missing values
missing_val = data_df_imp.isnull().sum().sum()
print("Missing values :",missing_val)

_______________________________________________________________________________________________________________________________________
# Example - KNN Imputation

from Feature_Imputation_Engine import *
import pandas as pd
import numpy as np
import matplotlib
import math

##Set Parameters
data_df = pd.read_csv('merchants.csv')
impute_type = 'knn'  
feat_list= list(data_df.columns)
replace_value=0
k_neighbors= 50

data_df_imp= feature_imputation_main(data_df=data_df, feat_list=feat_list,impute_local_key_list=impute_local_key_list,missing_val_list=missing_val_list,impute_type = impute_type)

##Check for missing values
missing_val = data_df_imp.isnull().sum().sum()
print("Missing values :",missing_val)
