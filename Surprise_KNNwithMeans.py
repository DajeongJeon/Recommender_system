
import pandas as pd
from scipy import stats
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import KFold
from surprise import accuracy
import matplotlib.pyplot as plt


# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

sim_itembase = {'name': 'cosine',
               'user_based': False}  # compute  similarities between items
algo_itembase = KNNWithMeans(sim_options=sim_itembase)

sim_userbase = {'name': 'pearson_baseline'}  # compute  similarities between users               
algo_userbase = KNNWithMeans(sim_options=sim_userbase)


# Run 5-fold cross-validation and save results.
kf = KFold(n_splits=5)
rmse_df = pd.DataFrame(columns=['Item-based','User-based'])
for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo_itembase.fit(trainset)
    pred_itembase = algo_itembase.test(testset)
    
    algo_userbase.fit(trainset)
    pred_userbase = algo_userbase.test(testset)

    # Compute and print Root Mean Squared Error
    rmse_itembase = accuracy.rmse(pred_itembase, verbose=True)
    rmse_userbase = accuracy.rmse(pred_userbase, verbose=True)

    #rmse_row.extend([rmse_itembase, rmse_userbase])
    rmse_row = pd.DataFrame([[rmse_itembase, rmse_userbase]], columns=['Item-based','User-based'])
    rmse_df = rmse_df.append(rmse_row, ignore_index=True)    
    
    
# Box plot     
plt.title('Boxplot of the RMSE distribution across the various folds')
rmse_df.boxplot(vert=False)
plt.grid(False) # Hide grid lines
plt.show()


# t-test
'''
Function: Print_Pair_ttest
dataset1 = dataset 1 for t-test
dataset2 = dataset 2 for t-test
title = title of the pair t-test
alpha = significance level (default = 0.05)
'''

def Print_Pair_ttest(dataset1, dataset2, title = '', subtitle='',alpha = 0.05):
    t_value, p_value = stats.ttest_ind(dataset1, dataset2)
    
    if p_value <= 0.05: 
        result = 'P-value ≤ α, \nThe difference between the means is statistically significant'
    else : result = 'P-value > α, \nThe difference between the means is not statistically significant'
    
    #print the t-test results
    print('')
    print(title)
    print("T-test result ({})".format(subtitle))
    print("=============================")
    print("t-statistic  : %.4f" %t_value)
    print("P-value      : %.4f" %p_value)
    print("=============================")
    print(result)
    
Print_Pair_ttest(rmse_df['Item-based'], rmse_df['User-based'], 'RMSE','Item-based VS User-based')
