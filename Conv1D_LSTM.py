import pandas as pd

# In[]
X_data = pd.read_csv('cascading_failure.csv')
# In[]
max_chain_num = 0
for i in range(X_data.shape[0]):
    temp = int(X_data.iloc[i,2][5])
    if temp>max_chain_num:
        max_chain_num = temp
del temp
# In[]
max_chain_num += 1
# In[]
fault_list = []
for i in range(X_data.shape[0]):
    temp = X_data.iloc[i,0]
    if temp not in fault_list:
        fault_list.append(temp)
del temp
# In[]
chain_list = []
for i in range(X_data.shape[0]):
    temp = X_data.iloc[i,2]
    if temp not in chain_list:
        chain_list.append(temp)
del temp
# In[]
repeat_list = []
for i in range(X_data.shape[0]):
    temp = X_data.iloc[i,1]
    if temp not in repeat_list:
        repeat_list.append(temp)
del temp
del i
# In[]
init_data = pd.read_csv('init_case.csv')
# In[]
total_chain_num = max_chain_num * len(fault_list) * len(repeat_list)
# In[]
cursor = 0
chain_num_list = []

for fault in fault_list:
    for repeat in repeat_list:
        chain_num = 0
        while X_data.iloc[cursor,0] == fault and X_data.iloc[cursor,1] == repeat :
            if cursor<22130:
                chain_num += 1
                cursor += 1
            else:
                chain_num += 1
                break
        chain_num_list.append(chain_num)
            
# In[]
old_cursor = 0
new_cursor = 0
new_Xdata = pd.DataFrame()
recycle_num = -1
# In[]
for fault in fault_list:
    for repeat in repeat_list:
        recycle_num += 1
        add_num = len(chain_list)+1-chain_num_list[recycle_num]
        for i in range(add_num):
            new_Xdata=new_Xdata.append(init_data.iloc[0,:],ignore_index=True)
            new_Xdata.loc[new_cursor,'fault_num'] = fault
            new_Xdata.loc[new_cursor,'repeat_num'] = repeat
            new_cursor += 1
            
        for i in range(chain_num_list[recycle_num]):
            new_Xdata=new_Xdata.append(X_data.iloc[old_cursor,:],ignore_index=True)
            new_Xdata.loc[new_cursor,'chain_num'] = 'chain_%s'%(add_num+i)
            old_cursor += 1
            new_cursor += 1
            print('old_cursor',old_cursor)
            print('new_cursor',new_cursor)
# In[]
for i in range(50000):
    new_Xdata.loc[i,'chain_num'] = 'chain_%s'%(i%5+1)
# In[]
new_Xdata.to_csv('new_Xdata.csv')
# In[]
del xx,i,old_cursor,new_cursor,recycle_num,
# In[]
new_Xdata = pd.read_csv('new_Xdata.csv')
# In[]
fault_df = new_Xdata.loc[:,'fault_num']
repeat_df = new_Xdata.loc[:,'repeat_num']
chain_df = new_Xdata.loc[:,'chain_num']
# In[]
new_Xdata = new_Xdata.drop('fault_num',axis=1)
new_Xdata = new_Xdata.drop('chain_num',axis=1)
new_Xdata = new_Xdata.drop('repeat_num',axis=1)
new_Xdata = new_Xdata.drop('Unnamed: 0',axis=1)
# In[]
from sklearn import preprocessing
Normalized_Xdata = preprocessing.StandardScaler().fit_transform(new_Xdata)
# In[]
Normalized_Xdata = Normalized_Xdata.reshape(10000,5,102)
# In[]
import tensorflow as tf
model = tf.keras.Sequential()
model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64,input_shape=(5,102)),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(1)
    ])
# In[]
y_data = pd.read_csv('failure_loss.csv').loc[:,'LOSS']
for i in range(5000,6000):
    y_data = y_data.drop(i,axis=0)
del i
# In[]
import numpy as np
y_data = np.array(y_data)
# In[]
model.compile(optimizer='adam',loss='mae')
# In[]
model.fit(Normalized_Xdata,y_data,batch_size = 128,epochs=50)
# In[]
y_predict = model.predict(Normalized_Xdata)
# In[]

# In[]
# In[]
# In[]
# In[]
# In[]
# In[]
# In[]
# In[]
# In[]