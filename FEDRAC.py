import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import LSTM

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D,LeakyReLU
import os
import cv2, random
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from  keras.utils import np_utils

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	x_train, y_train = load_dataset_group('train', prefix + 'data/HAR/')
	print(x_train.shape, y_train.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'data/HAR/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	y_train = y_train - 1
	testy = testy - 1
	# one hot encode y
	# y_train = np_utils.to_categorical(y_train)
	# testy = np_utils.to_categorical(testy)
	print(x_train.shape, y_train.shape, testX.shape, testy.shape)
	return x_train, y_train, testX, testy

#load the data
'''   .
trainX = (7352, 128, 9) 7352 samples or windows, each 128 timesteps or readings, represented with 9 features where each feature data is in a separate file
testX(2947, 128, 9)
trainY (7352, 1), activity type performed (one of the six activities)
testY  (2947, 1)
'''
x_train, y_train, testX, testy = load_dataset()
n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
n_outputs=6
x_train=x_train.reshape(x_train.shape[0],n_timesteps, n_features)
testX=testX.reshape(testX.shape[0],n_timesteps, n_features)


y_train=y_train.reshape(-1)
testy=testy.reshape(-1)
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train, y_train, random_state=42)
testX,testy=shuffle(testX,testy, random_state=42)
ix=0
trainX=list()
trainy=list()
for ind11 in range(40):
  trainX.append(x_train[ix:ix+600])
  trainy.append(y_train[ix:ix+600])
  # trainX.append(x_train[ix+183:ix+366+183])
  # trainy.append(y_train[ix+183:ix+366+183])
  ix=ix+170
  print(ix)

trainX=np.array(trainX)
trainy=np.array(trainy)
# Y=np.reshape(Y, (40,360))

print(trainX.shape,trainy.shape)
# print(trainy[0])
X=trainX
Y=trainy
x_test=testX
y_test=testy
print(y_test.shape)

from tensorflow.keras.layers import BatchNormalization
class MODEL:
    @staticmethod
    def build(nl):
        model = Sequential()
        model.add(Conv1D(128,3,strides=2, padding="same",input_shape=(n_timesteps, n_features)) )
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.25))

        # model.add(Conv2D(128/nl, (3, 3), strides=(2, 2), padding="same") )
        model.add(Conv1D(64/nl, 3, strides=2, padding="same") )
        
        model.add(Conv1D(128/nl, 3, strides=2, padding="same") )
        model.add(Conv1D(256/nl, 3, strides=2, padding="same") )
        model.add(Conv1D(512/nl, 3, strides=2, padding="same") )
        model.add(Flatten())
        # model.add(Dense(256/nl,activation='relu')) #add if acc <90
        # model.add(Dense(32/nl,activation='relu'))
        model.add(Dense(6,activation='softmax'))#for output layer
        return model



from sklearn.metrics import accuracy_score
def test_model(X_test, Y_test,  model, comm_round):
    # cce = tf.keras.losses.SparseCategoricalCrossentropy()
    # #logits = model.predict(X_test, batch_size=100)
    # logits = model.predict(X_test)
    # loss = cce(Y_test, logits)
    # acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    # print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    model.compile(  optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
    loss,acc=model.evaluate(x_test,y_test)
    # acc,loss=model.evaluate(x_test,y_test)
    return acc, loss
def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
      
    return avg_grad
# print(global_model.get_weights())
def test_modelstudent(X_test, Y_test,  gmodel, comm_round):
    gmodel.compile(optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3
  )
    loss,acc=gmodel.evaluate(x_test,y_test)
    # acc,loss=gmodel.evaluate(x_test,y_test)

    return acc, loss



class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
import numpy as np
import csv

path = 'survey.csv'
with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = np.array(list(reader)).astype(float)
data=data[:40]
data=np.delete(data, 0, 1)
data[:, [2, 0]] = data[:, [0, 2]]


maxminarr=[]
sz=len(data[0])
for i in range(sz):
  tmp=[np.max(data,axis=0)[i],np.min(data,axis=0)[i]]
  maxminarr.append(tmp)

V=data
normV=[]
for i in range(len(V)):
  tmpar=[]
  for j in range(len(V[i])):
    tmp=(V[i][j]-maxminarr[j][1])/(maxminarr[j][0]-maxminarr[j][1])
    # print(tmp)
    # V[i][j]=tmp
    tmpar.append(tmp)
  # print(tmpar.shape)
  normV.append(tmpar)
    
    
import math
def euclidean_distance(V):
  dist=[]
  for i in range(len(V)):
    tmp=[]
    for j in range(len(V)):
      disc=0
      lambd=[0.4,0.4,0.2]#[1.0/3,1.0/3,1.0/3]
      for ind in range(len(V[i])):
        disc+=math.sqrt(lambd[ind]*(V[i][ind]-V[j][ind])**2)
      tmp.append(disc)
    dist.append(tmp)
  return dist

d=euclidean_distance(normV)


from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i][ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i][ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i][ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i][ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i][ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i][ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.
    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param diameter_method: see :py:function:`diameter` `method` parameter
    :param cdist_method: see :py:function:`diameter` `method` parameter
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))
    # print(min_distance,max_diameter)

    return min_distance / max_diameter



dunnarray=[]
dunnvalues=[]
rng=int(math.sqrt(len(data)))
print(rng)
for ncluster in range(2,rng+1):
      kmeans = KMeans(n_clusters=ncluster)
      # c = data[1]

      k = kmeans.fit_predict(normV)
      # d = euclidean_distances(x)
      print("No of Clusters(K)= ",ncluster)
      # print(k)

      for diameter_method in DIAMETER_METHODS:
          for cdist_method in CLUSTER_DISTANCE_METHODS:
              # dund = dunn(c, d, diameter_method, cdist_method)
              
              if diameter_method=="farthest" and cdist_method=="nearest":
                dunk = dunn(k, d, diameter_method, cdist_method)
                tmp=[ncluster,dunk]
                dunnarray.append(tmp)
                dunnvalues.append(dunk)




bestnoofcluster=dunnvalues.index(max(dunnvalues))+2
print(bestnoofcluster)
d=[]
for i in range(len(dunnvalues)):
  d.append([dunnvalues[i],i+2])
print(d)
d.sort()
d.reverse()
print(d)
aa=[b for a,b in d[:1]]
print(aa)
best3kclusters=aa

if(best3kclusters[0]>2):
    best3kclusters.append(best3kclusters[0]-1)
    best3kclusters.append(best3kclusters[0]-2)
print(best3kclusters)

nc=bestnoofcluster


lerr=np.random.uniform(low=0.02, high=0.05, size=(40))
def herr(jj,C,i):
  sumx=0
  lenx=0
  maxx=0
  minn=10
  # return len(C[jj])
  for ix in range(len(C[jj])):
    sumx+=lerr[C[jj][ix]]
    lenx+=1
    maxx=max(maxx,lerr[C[jj][ix]])
    minn=min(minn,lerr[C[jj][ix]])
  sumx+=lerr[i]
  lenx+=1
  avgerr=sumx/lenx
  tmp=max(abs(maxx-avgerr),abs(avgerr-minn))
  return avgerr,tmp



for nc in best3kclusters:
  # print("\n","Number of Clusters=",nc,"\n")
  f = open("Output.txt", "a+") 
  f.write("\n Number of Clusters=%d \n" %nc)
  f1=open("Output1.txt","a+")
  f1.close()
  kmeans = KMeans(n_clusters=nc)
  k = kmeans.fit_predict(normV)
  # print("Client with their cluster number",k,"\n")
  # f.write("Client with their cluster number %s \n"%k)
  AccomodateM=[0 for i in range(nc)]
  su=0
  lent=0
  for i in range(nc):
    su=0
    lent=0
    for j in range(len(k)):
      if(k[j]==i):
          su+=pov[j]
          lent=lent+1
    AccomodateM[i]=su/lent
  AccomodateM.sort()
  AccomodateM.reverse()  
  AccomodateM.append(0)
  # print("Cluster value for which it can accomodate any client with value around this",AccomodateM,"\n")
  f.write("Cluster value for which it can accomodate any client with value around this %s \n"%AccomodateM)




  tmp=200

  # Rf=[(tmp*10) for i in range(nc)]
  R=[]
  for i in range(nc):
    if(i==0):
      R.append(tmp)
      tmp=200
    else:
      R.append(tmp)
    # tmp=math.ceil(tmp/1.2)
  E=[]
  tmp=10
  epochlow=1
  epochmax=5
  tmp=epochmax
  divfactor=(epochmax-epochlow)/nc
  for i in range(nc):
    E.append(int(tmp))
    tmp=(tmp-divfactor)
  
  print("Global Comm Rounds",R)
  f.write("Global Comm Rounds %s \n"%R)
  print("Local Epochs",E,"\n")
  f.write("Local Epochs %s \n"%E)
  localepoch=[0 for i in range(len(pov))]
  addndata=[[] for i in range(nc)]





  qo=np.random.uniform(low=0.02, high=0.05, size=nc)
  addtmp=(0.05-0.02)/nc
  qo=[]
  tmp=0.02
  for i in range(nc):
    qo.append(round(tmp,3))
    tmp=tmp+addtmp
  tmp=0.05

  delta=[]
  tmp=0.03
  for i in range(nc):
    delta.append(tmp)
    tmp=tmp*1.00001
  print("Qo",qo)
  print("delta",delta)
  f.write("Qo %s \n" %qo)
  f.write("Delta %s \n" %delta)
  errf=[]
  tmp=0.1
  for i in range(nc):
    errf.append(tmp)
    tmp=round(tmp*1.2,3)
  print("Err",errf)
  theta=[]
  tmp=0.03
  for i in range(nc):
    theta.append(tmp)
    tmp=round(tmp*1.2,3)
  print("Theta",theta,"\n")
  f.write("Theta %s \n" %theta)


  C=[[] for i in range(nc)]
  accmthreshold=0.1
  for i in range((len(pov))):
    flag=0
    for j in range(len(C)):
      if(flag==1):
        continue
      if(i==1 and j==1):
        print("hello")
      if len(C[j])==0:
        if( (pov[i]-AccomodateM[j]>=0) or (abs(pov[i]-AccomodateM[j])-abs(pov[i]-AccomodateM[j+1])<=0) ):
          # print("(pov[i]-AccomodateM[j]>0)",(pov[i]-AccomodateM[j]))
          # print("(pov[i]-AccomodateM[j])",i,(pov[i]-AccomodateM[j]))
          # print("pov[i]-AccomodateM[j+1]",i,pov[i]-AccomodateM[j+1])
          #calculate qof
          clustrloss=lerr[i]
          if(clustrloss<=delta[j]):
            qo[j]=clustrloss
            errf[j]=0
            C[j].append(i)
            flag=1
            localepoch[i]=E[j]

        else:
          #increase data in this client
          dwkcwdkkc=0
          qofRE=E[j]/R[j]
          rat=qo[j]/qofRE
          newlocalEpoch=E[j]-1
          
          newqofRE=newlocalEpoch/R[j]
          

          clustrloss=rat*newqofRE
          # print("Clustrloss",clustrloss,"0")
          
          if(clustrloss<=delta[j] and clustrloss>=0.02):
            qo[j]=clustrloss
            errf[j]=0
            C[j].append(i)
            localepoch[i]=E[j]-1
            addndata[j].append(i)
            flag=1
            # print("Delta","Client",i,"Cluster",j,clustrloss,delta[j],0,localepoch[i])
          
          # print("LOss,",qo[j],qofRE)
          #reduce ti and ni s.t. pi run Mf
      else:

        if( (pov[i]-AccomodateM[j]>=0) or (abs(pov[i]-AccomodateM[j])-abs(pov[i]-AccomodateM[j+1])<=0) or j==len(C)-1):
          #calculate qof
          clustrlossavg,hetroloss=herr(j,C,i)
          #  print('i',i,'j',j,tmp)
          if(clustrlossavg<=delta[j] and hetroloss<=theta[j]):
            errf[j]=hetroloss
            qo[j]=clustrlossavg
            C[j].append(i)
            flag=1
            localepoch[i]=E[j]
          # else:
          #   print('i',i,'j',j,tmp,'theta',theta[j])
          #   print("CLusterlossavg",clustrlossavg,"Delta[j]",delta[j],"Hetroloss",hetroloss,"Theta[j]",theta[j])


        else:
          dwkcwdkkc=0
          qofRE=E[j]/R[j]
          rat=qo[j]/qofRE
          newlocalEpoch=E[j]-1
          newqofRE=newlocalEpoch/R[j]

          clustrloss=rat*newqofRE
          tmpvar=lerr[i]
          lerr[i]=clustrloss
          

          # clustrloss=lerr[i]
          # print("Clustrloss",clustrloss,"1")
          
          clustrlossavg,hetroloss=herr(j,C,i)
          # deltanewj=rat*clustrlossavg
          
          if(clustrlossavg<=delta[j] and hetroloss<=theta[j] and clustrloss>=0.02):

            errf[j]=hetroloss
            qo[j]=clustrlossavg
            C[j].append(i)
            flag=1
            localepoch[i]=E[j]-1
            addndata[j].append(i)
            # print("Delta","Client",i,"Cluster",j,clustrlossavg ,delta[j],1,localepoch[i])
          
          
          else:
            lerr[i]=tmpvar

          #reduce ti and ni s.t. pi run Mf

      if(flag==0 and j== len(C)-1):
        if(len(C[j])==0):
          clustrloss=lerr[i]
          qo[j]=clustrloss
          errf[j]=0
          C[j].append(i)
          flag=1
          localepoch[i]=E[j]
        else:
          clustrlossavg,hetroloss=herr(j,C,i)
          #  print('i',i,'j',j,tmp)
          errf[j]=hetroloss
          qo[j]=clustrlossavg
          C[j].append(i)
          localepoch[i]=E[j]
          flag=1


      
  for tmx in range(nc):
    C[tmx]=np.array(C[tmx])
  # print("Clustering Array",np.array(C,dtype=object))
  f.write("Clustering Array %s \n" %(np.array(C,dtype=object)))
  # print("ERRf of each cluster",errf,"\n")
  f.write("ERRf of each cluster %s \n"%errf)
  # print("Qo of each cluster",qo,"\n")


  #Model Training
  # print("Model Training","\n")
  f.write("Model Training\n")
  flag=1
  # for empclstr in range(nc):
  #   if(len(C[empclstr])==0):
  #     # flag=0
  #     f.write("Cluster no. %d is empty" %empclstr)
  #     break
  # print("Original Cluster count",nc)
  
  #shifting clients

  emptcnt=0
  # if(len(C[nc-1])==0):
    #   emptcnt=emptcnt+1
  for abd in range(0,nc):
    # print("abd",abd)
    if(len(C[abd])==0):
      cx=abd
      while(len(C[cx])==0 and cx!=nc-1):
        
        cx=cx+1
      C[abd],C[cx]=C[cx],C[abd]
      addndata[abd],addndata[cx]=addndata[cx],addndata[abd]
     # emptcnt=emptcnt+1
  for abd in range(0,nc):
    if(len(C[abd])==0):
      emptcnt=emptcnt+1

  # print("C",C)
  
  # print("emptcnt",emptcnt) 
  nc=nc-emptcnt
  emptcnt=0
  # print("NC",nc)
  for abd in range(nc-1,0,-1): 
    if(len(C[abd])<=4 and len(C[abd])>0):

      for val in C[abd]:
        C[abd-1]=np.append(C[abd-1],int(val))
        # print("Val",val,"abd",abd)
      for val in addndata[abd]:
        addndata[abd-1]=np.append(addndata[abd-1],int(val))
      C[abd]=[]
      addndata[abd]=[]
      emptcnt=emptcnt+1
    # print("ABD here",abd)
  # print("C11",C)
  var1=1
  if(len(C[0])<=4 and len(C[0])>0):
    while(1):
      if(len(C[0])+len(C[var1])>=5):
        break
      var1+=1
    for val in C[0]:
        C[var1]=np.append(C[var1],int(val))
    for val in addndata[0]:
      addndata[var1]=np.append(addndata[var1],int(val))
    C[0]=[]
    addndata[0]=[]
    emptcnt=emptcnt+1

  
  # print("emptcnt1",emptcnt) 
  # print("NC1",nc)
  for abd in range(0,nc):
    # print("abd",abd)
    if(len(C[abd])==0):
      cx=abd
      while(len(C[cx])==0 and cx!=nc-1):
        
        cx=cx+1
      C[abd],C[cx]=C[cx],C[abd]
      # print("CX",cx)
      # print("ABD",abd)
    # print("NC1",nc)
  for abd in range(0,nc):
    # print("abd",abd)
    if(len(addndata[abd])==0):
      cx=abd
      while(len(addndata[cx])==0 and cx!=nc-1):
        
        cx=cx+1
      addndata[abd],addndata[cx]=addndata[cx],addndata[abd]
      # print("CX",cx)
      # print("ABD",abd)
  # print("C2",C)
  nc=nc-emptcnt
  # print("Updated ADDitional Data",addndata)
  f.write("Final Clustering Array %s \n" %(np.array(C,dtype=object)))
    
      





  f.close()
  X=[[] for i in range(40)]
  Y=[[] for i in range(40)]
  for clstr in range(nc):
      data_size=len(x_train)
      ix=0
      for ind11 in range(len(C[clstr])):
        datarang=int(data_size/len(C[clstr]))
        X[C[clstr][ind11]]=x_train[ix:ix+2*datarang]
        Y[C[clstr][ind11]]=y_train[ix:ix+2*datarang]
        # trainX.append(x_train[ix+183:ix+366+183])
        # trainy.append(y_train[ix+183:ix+366+183])
        step=(data_size-2*datarang)/(len(C[clstr])-1)
        print("step",step)
        ix=ix+int(step)
        print("ix",ix)
  X=np.array(X)
  Y=np.array(Y)
  for i in X:

  if(flag==1):
    for clstr in range(nc):

      # clstr=4
      f = open("Output.txt", "a+") 
      f1=open("Output1.txt","a+")
      print("CLUSTER NO.",clstr,"\n")
      f.write("CLUSTER NO.%d \n" %clstr)
      f1.write("CLuster NO. %d \n" %clstr)
      f1.close()
      if(clstr==0):
        
        global1= MODEL()
        global_model = global1.build(1)
        for comm_round in range(R[clstr]):

          f1=open("Output1.txt","a+")
          global_weights = global_model.get_weights()
          scaled_local_weight_list = list()
          index=list({0,1,2,3,4})
          #random.shuffle(index)
          # print(index)
          tsf=0
          for clnt in range(len(C[clstr])):
            print("CLIENT NO: ",C[clstr][clnt])

            local = MODEL()
            local_model=local.build(1)
            
            
            local_model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
            local_model.set_weights(global_weights)
            
            #trainy[ind]=np.array(trainy[ind]).reshape(-1,1)
            if(C[clstr][clnt] in addndata[clstr]):
              print("Clnt",C[clstr][clnt],"epoch",localepoch[C[clstr][clnt]])
              history=local_model.fit(X[C[clstr][clnt]],Y[C[clstr][clnt]], epochs=localepoch[C[clstr][clnt]])#,validation_data=(x_test,y_test))
              # print("Accuracy: ",history.history["accuracy"][4])
              local_model.evaluate(x_test, y_test)
              scaling_factor=len(X[C[clstr][clnt]])/(len(C[clstr])*len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])+100*len(addndata[clstr])) #1/no.ofclients
              tsf+=scaling_factor
            else:
              # print("Clnt",clnt,"epoch",localepoch[clnt])
              history=local_model.fit(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100],Y[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100], epochs=localepoch[C[clstr][clnt]])#,validation_data=(x_test,y_test))
              # print("Accuracy: ",history.history["accuracy"][4])
              local_model.evaluate(x_test, y_test)
              scaling_factor=len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])/(len(C[clstr])*len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])+100*len(addndata[clstr])) #1/no.ofclients
              tsf+=scaling_factor

            
            # print("No of clients in cluster",len(C[clstr]))
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            K.clear_session()
          # print("Total Scaling Factor",tsf)
          average_weights = sum_scaled_weights(scaled_local_weight_list)
          #global_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
          global_model.set_weights(average_weights)
          #val_acc,val_loss=global_model.evaluate(x_train,y_train)
          #print(val_loss,val_acc)
          global_acc, global_loss = test_model(x_test,y_test, global_model, comm_round)
          print("COMM_ROUNND",comm_round," GLOBAL MASTER MODEL ACCURACY",global_acc,global_loss)
          f.write("Communication Round: %d GLOBAL MASTER MODEL ACCURACY: %f LOSS: %f \n" %(comm_round ,global_acc ,global_loss))
          f1.write("%f \t %f \n" %(global_acc,global_loss))
          f1.close()
      else:
        
        studentm=MODEL.build(2**clstr)
        student_model=Distiller(student=studentm, teacher=global_model)
        for comm_round in range(R[clstr]):
          f1=open("Output1.txt","a+")
          global_weights = student_model.get_weights()
          scaled_local_weight_list = list()
          index=list({0,1,2,3,4})
          #random.shuffle(index)
          # print(index)
          tsf=0
          for clnt in range(len(C[clstr])):
            print("CLIENT NO: ",C[clstr][clnt])
            local_model1 = Distiller(student=studentm, teacher=global_model)
            
            local_model1.compile(optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=3
            )

            local_model1.set_weights(global_weights)
            #trainy[ind]=np.array(trainy[ind]).reshape(-1,1)

            if(C[clstr][clnt] in addndata[clstr]):
              print("Clnt",C[clstr][clnt],"epoch",localepoch[C[clstr][clnt]])
              history=local_model1.fit(X[C[clstr][clnt]],Y[C[clstr][clnt]], epochs=localepoch[C[clstr][clnt]])#,validation_data=(x_test,y_test))
              # print("Accuracy: ",history.history["accuracy"][4])
              local_model1.evaluate(x_test, y_test)
              scaling_factor=len(X[C[clstr][clnt]])/(len(C[clstr])*len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])+100*len(addndata[clstr])) #1/no.ofclients
              tsf+=scaling_factor
            else:
              # print("Clnt",clnt,"epoch",localepoch[clnt])
              history=local_model1.fit(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100],Y[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100], epochs=localepoch[C[clstr][clnt]])#,validation_data=(x_test,y_test))
              # print("Accuracy: ",history.history["accuracy"][4])
              local_model1.evaluate(x_test, y_test)
              scaling_factor=len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])/(len(C[clstr])*len(X[C[clstr][clnt]][:len(X[C[clstr][clnt]])-100])+100*len(addndata[clstr])) #1/no.ofclients
              tsf+=scaling_factor


            scaled_weights = scale_model_weights(local_model1.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            K.clear_session()
          # print("Total Scaling Factor",tsf)  
          average_weights = sum_scaled_weights(scaled_local_weight_list)
          #global_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
          student_model.set_weights(average_weights)
          #val_acc,val_loss=global_model.evaluate(x_train,y_train)
          #print(val_loss,val_acc)
          global_loss1, global_acc1 = test_modelstudent(x_test,y_test, student_model, comm_round)
          print("COMM_ROUNND",comm_round,"GLOBAL SLAVE MODEL ACCURACY",global_acc1,global_loss1)
          f.write("Communication Round: %f GLOBAL SLAVE MODEL ACCURACY: %f LOSS:%f \n" %(comm_round,global_acc1 ,global_loss1))
          f1.write("%f \t %f \n" %(global_acc1,global_loss1))
          f1.close()
      f.close()
















  




