import sys
import numpy as np
import pandas as pd

if len(sys.argv) == 1: 
    print("Not enough arguments. ")
    print("Expected Format: DataManager.py \"dataset1.csv\", \"dataset2.csv\" ... ")
    exit()

for i in range(1,len(sys.argv)):
    df = pd.read_csv(sys.argv[i]) # Command line args
    df = df.select_dtypes(include='float64') # Include only floats
    df = df.sample(frac=1) # Shuffle rows
    df = df.loc[:, ('Capital Expenditures', 'Cash Ratio', 'Inventory', 'Investments',
                    'Net Cash Flow', 'Net Income',
                    'Profit Margin', 'Total Revenue', 'Gross Profit')]
    df = df.replace('', np.nan)
    df = df.replace(0, np.nan)
    df = df.drop(index=np.where(pd.isnull(df))[0])
    features = df.iloc[:, :(df.shape[1] - 1)].copy()
    output = df.iloc[:, (df.shape[1] - 1)].copy()

    for j in features.columns:
        features.loc[:, j] = (features[j] - features[j].mean()) / features[j].std() # Feature Normalization
    features.insert(0, 't0', 1)
    str1 = 'trainingInput' + str(i) + '.csv'
    str2 = 'trainingOutput' + str(i) + '.csv'
    features.iloc[:int(0.6 * features.shape[0]), :].to_csv(str1, index=False,
                                                           header=False, sep=' ', na_rep=0)
    output.iloc[:int(0.6 * output.shape[0])].to_csv(str2, index=False,
                                                    header=False, sep=' ', na_rep=0)
    str1 = 'CVInput' + str(i) + '.csv'
    str2 = 'CVOutput' + str(i) + '.csv'    
    features.iloc[int(0.6 * features.shape[0]):int(0.8 * features.shape[0]), :].to_csv(str1, index=False,
                                                           header=False, sep=' ', na_rep=0)
    output.iloc[int(0.6 * output.shape[0]):int(0.8 * output.shape[0])].to_csv(str2, index=False,
                                                    header=False, sep=' ', na_rep=0)
    str1 = 'testInput' + str(i) + '.csv'
    str2 = 'testOutput' + str(i) + '.csv'
    features.iloc[int(0.8 * features.shape[0]):, :].to_csv(str1, index=False,
                                                           header=False, sep=' ', na_rep=0)
    output.iloc[int(0.8 * output.shape[0]):].to_csv(str2, index=False,
                                                    header=False, sep=' ', na_rep=0)

