import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

# the following is used in unit testing, please ignore 
import sys
sys.path.insert(0,'..\\..\\')

def load_data():
    file = open('Absenteeism_at_work.csv', 'r')
    lines = file.readlines()
    data1 = []
    for line, i in zip(lines, range(len(lines))):
        data1.append((line.strip("\n")).split(';')) 

    df = pd.DataFrame(data1[1:], columns= data1[0] )
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def load_conditions():
    file = open('conditions.txt', 'r')
    lines = (line.strip("\n") for line in file.readlines())
    return pd.Series(lines)

def insert_conditions(df, s_conditions, reference_column, new_column_name, position):
    converted = [s_conditions[ref] for ref in df[reference_column]]
    df.insert(position, new_column_name, converted)

def remove(df, selected_col, n_min, col_list):
    sameCount = {}
    for  val in df[selected_col]:
        if val in sameCount:
            sameCount[val] +=1
        else:
            sameCount[val] = 0
    indexs = [True if v > n_min else False for v in sameCount.values() ] 
    valid = []
    for k, t in zip(sameCount.keys(),indexs):
        if t:
            valid.append(k)
    return df[col_list].loc[df[selected_col].isin(valid)] 

def extract_stats(df, sel_col):
    def count(iterable):
        return len(iterable)

    return df.groupby([sel_col]).agg([count,np.mean, np.std])

from scipy.stats import norm

def plot_stats(df, selected_col, col_list, extract_stats):
    # display each column from col_list as graph of normal function of groups
    extractedS = extract_stats(df,selected_col)
    #df = df.apply(pd.to_numeric, errors='ignore')
    
    for col in col_list:
        
        tmp = extractedS[col]
        allMeans = tmp["mean"]
        allCount = tmp["count"]
        allSTD = tmp["std"]
        names = tmp.index
        plt.title(col)
        for mu, n, sigma, name in zip(allMeans, allCount, allSTD, names):
            
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, norm.pdf(x, mu, sigma), label = name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
        plt.figure()
    plt.show()

# Extract desired numpy array from a dataframe

def select_data(df, selected_col, selected_reason):
    
    return df[df[selected_col] == selected_reason].values[:,1:].astype(float)

def make_train_test_split(X1, X2, train_size):
    import random
    X_train = [] 
    y_train = []
    X_test = []
    y_test = []
    while len(X_train) < train_size:
        if random.random() < 0.5:
            X_train.append(np.delete(X1,random.randrange(len(X1))))
            y_train.append(0)
        else:    
            X_train.append(np.delete(X2,random.randrange(len(X2))))
            y_train.append(1)


    for i in range(len(X1)+ len(X2)):
        if random.random() < 0.5:
            if len(X1) > 0:
                X_test.append(np.delete(X1,random.randrange(len(X1))))
                y_test.append(0)
            else:
                X_test.append(np.delete(X2,random.randrange(len(X2))))
                y_test.append(1)  
        else:    
            if len(X2) > 0:
                X_test.append(np.delete(X2,random.randrange(len(X2))))
                y_test.append(1) 
            else:
                X_test.append(np.delete(X1,random.randrange(len(X1))))
                y_test.append(0)

    return X_train, y_train, X_test, y_test 


def baseline_acc(targets):
    import random
    dumbClassifications = []
    for i in range(len(targets)):
        if random.random() < 0.5:
            dumbClassifications.append(0)
        else:
            dumbClassifications.append(1)
    score = 0
    for d, t in zip(dumbClassifications, targets):
        if d == t:
            score += 1
    
    return score / len(targets)

def distance(x1, x2):
    score = 0
    for i,j in zip(x1, x2):
        score += (i-j)**2
    return score**(1/2)


def pairwise_distances(X_ref, X):
    X_ref = np.array(X_ref, dtype=object)
    X = np.array(X, dtype=object)
    output = []
    for xr in X_ref:
        for x in X:
            output.append(distance(xr,x))
    return np.reshape(output, ((X_ref.shape)[0],(X.shape)[0]))


def knn_predictive_performance(X_train, y_train, X_test, y_test, k, pairwise_distances_func):
    paired = pairwise_distances_func(X_test, X_train)
    outcomes = []
    for i in range(len(X_test)):
        kNeighbours = []
        allNeighbours = paired[i]
        for ne, y in zip(allNeighbours, y_train):
            if len(kNeighbours) < k:
                kNeighbours.append([ne,y])
            else:
                for kN, i2 in zip(kNeighbours,range(k)):
                    if kN[0] > ne:
                        kNeighbours[i2] = [ne,y]
                        break
        kSum = sum(np.array(kNeighbours)[:,1])
        kSum /= k
        if kSum > 0.5:
            outcomes.append(1)
        else:
            outcomes.append(0)
    score = 0
    for d, t in zip(y_test, outcomes):
        if d == t:
           score += 1
    return score / len(outcomes)
    
# Just run the following code, do not modify it
# Just run the following code, do not modify it

df = load_data()
s_conditions = load_conditions()
insert_conditions(df, s_conditions, reference_column='Reason for absence', new_column_name='Reason', position=0)
col_list = ['Reason','Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', 
            'Education', 'Son', 'Social drinker', 'Pet', 'Body mass index']
df2 = remove(df, selected_col='Reason', n_min=10, col_list=col_list)
reasons = df2['Reason'].unique()
selected_reasons = [r for r in reasons if r != 'Physiotherapy']

r1 = reasons[2]
r2 = reasons[11]
X1 = select_data(df2, selected_col='Reason', selected_reason=r1)
X2 = select_data(df2, selected_col='Reason', selected_reason=r2)

n = len(X1)+len(X2)
m = n//2
X_train, y_train, X_test, y_test = make_train_test_split(X1, X2, train_size=m)
acc = knn_predictive_performance(X_train, y_train, X_test, y_test, k=3, pairwise_distances_func=pairwise_distances)
bacc = baseline_acc(y_test)
print(acc, bacc)