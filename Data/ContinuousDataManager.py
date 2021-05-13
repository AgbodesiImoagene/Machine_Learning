import sys
import numpy as np
import pandas as pd

if len(sys.argv) == 1: 
    print("Not enough arguments. ")
    print("Expected Format: DataManager.py \"dataset1.csv\", \"dataset2.csv\" ... ")
    exit()

df = pd.read_csv(sys.argv[1]) # Command line arg
df = df.sample(frac=1) # Shuffle rows
df = df.drop(index=np.where(pd.isnull(df))[0]) # Remove rows with blanks
features = df.iloc[:, :(df.shape[1] - 1)].copy()
output = df.iloc[:, (df.shape[1] - 1)].copy()
for i in features.columns:
    features.loc[:, i] = (features[i] - features[i].mean()) / features[i].std() # Feature Normalization
if output.mean() > 10**5:
    output = (output - output.mean()) / output.std()
features.insert(0, 't0', 1)
str1 = 'trainingInput.csv'
str2 = 'trainingOutput.csv'
features.iloc[:int(0.6 * features.shape[0]), :].to_csv(str1, index=False,
                                                       header=False, sep=' ', na_rep=0)
output.iloc[:int(0.6 * output.shape[0])].to_csv(str2, index=False,
                                                header=False, sep=' ', na_rep=0)
str1 = 'CVInput.csv'
str2 = 'CVOutput.csv'    
features.iloc[int(0.6 * features.shape[0]):int(0.8 * features.shape[0]), :].to_csv(str1, index=False,
                                                       header=False, sep=' ', na_rep=0)
output.iloc[int(0.6 * output.shape[0]):int(0.8 * output.shape[0])].to_csv(str2, index=False,
                                                header=False, sep=' ', na_rep=0)
str1 = 'testInput.csv'
str2 = 'testOutput.csv'
features.iloc[int(0.8 * features.shape[0]):, :].to_csv(str1, index=False,
                                                       header=False, sep=' ', na_rep=0)
output.iloc[int(0.8 * output.shape[0]):].to_csv(str2, index=False,
                                                header=False, sep=' ', na_rep=0)



    
