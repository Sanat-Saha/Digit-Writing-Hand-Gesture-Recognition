import socket, traceback
from xml.etree import cElementTree as ET
import re
import sys, os
from nitime import utils
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
import pywt
import scipy.stats as sp
from os import listdir
from os.path import isfile, join
import numpy as np
import cPickle
from scipy import signal

def max_val_func(a):
    b = a #Extracting the data from the 3 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for l in b:
        i = l[~np.isnan(l)]
        output[k] = np.amax(i)#saving the max's
        k=k+1
    return output[0],output[1]
def min_val(a):
    b = a #Extracting the data from the 3 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for l in b:
        i = l[~np.isnan(l)]
        output[k] = np.amin(i)#saving the max's
        k=k+1
    return output[0],output[1]
def std_val(a):
    b = a #Extracting the data from the 3 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for l in b:
        i = l[~np.isnan(l)]
        output[k] = np.std(i) #saving the stds
        k += 1
    return output[0],output[1]
def my_kurtosis(a):
    b = a # Extracting the data from the 3 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 3)
    k = 0; # For counting the current row no.
    for l in b:
        i = l[~np.isnan(l)]
        kurtosis_i = sp.kurtosis(i,fisher = True)
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return output[0],output[1]
def skewness(a):
    b = a # Extracting the data from the 3 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 3)
    k = 0; # For counting the current row no.
    for l in b:
        i = l[~np.isnan(l)]
        skewness_i = sp.skew(i)
        output[k] = skewness_i # Saving the skewness in the array created
        k +=1 # Updating the current row no.
    return output[0],output[1]
def hjorth(input):                                             # function for hjorth 
    realinput = input
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for j in realinput:
        hjorth_activity[k] = np.var(j)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
        k = k+1
    return hjorth_activity[0],hjorth_activity[1],hjorth_mobility[0],hjorth_mobility[1],hjorth_complexity[0],hjorth_complexity[1]
def first_diff_mean(arr):
    data = arr 
    diff_mean_array = np.zeros(len(data)) #Initialinling the array as all 0s
    index = 0; #current cell position in the output array
   
    for i in data:
        sum=0.0#initializing the sum at the start of each iteration
        for j in range(len(i)-1):
            sum += abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
           
        diff_mean_array[index]=sum/(len(i)-1)
        index+=1 #updating the cell position
    return diff_mean_array[0],diff_mean_array[1]
def first_diff_max(arr):
    data = arr 
    diff_max_array = np.zeros(len(data)) #Initialinling the array as all 0s
    first_diff = np.zeros(len(data[0])-1)#Initialinling the array as all 0s 
    index = 0; #current cell position in the output array
    for i in data:
        max=0.0#initializing at the start of each iteration
        for j in range(len(i)-1):
            first_diff[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
            if first_diff[j]>max: 
                max=first_diff[j] # finding the maximum of the first differences
        diff_max_array[index]=max
        index+=1 #updating the cell position
    return diff_max_array[0],diff_max_array[1]
def secDiffMean(a):
    b = a # Extracting the data of the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    for i in b:
        t = 0.0
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        for j in range(len(i)-2):
            t += abs(temp1[j+1]-temp1[j]) # Summing the 2nd Diffs
        output[k] = t/(len(i)-2) # Calculating the mean of the 2nd Diffs
        k +=1 # Updating the current row no.
    return output[0],output[1]

def secDiffMax(a):
    b = a # Extracting the data from the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    t = 0.0
    for i in b:
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        t = temp1[1] - temp1[0]
        for j in range(len(i)-2):
            if abs(temp1[j+1]-temp1[j]) > t :
                t = temp1[j+1]-temp1[j] # Comparing current Diff with the last updated Diff Max

        output[k] = t # Storing the 2nd Diff Max for channel k
        k +=1 # Updating the current row no.
    return output[0],output[1]
def wavelet_features(epoch):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    for i in range(2):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)        #calculating the coefficients of wavelet transform.
        D = [-0.268654597,-0.134594047]
        A = [-9.598793174, -14.42063872]
    for x in range(2):   
        cA_mean.append(np.mean(cA_values[x]))
        cA_std.append(np.std(cA_values[x]))
        cA_Energy.append(np.sum(np.square(cA_values[x])))
        cD_mean.append(np.mean(cD_values[x]))       # mean and standard deviation values of coefficents of each channel is stored .
        cD_std.append(np.std(cD_values[x]))
        cD_Energy.append(np.sum(np.square(cD_values[x])))
        val = np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))
        if not np.isnan(val):
            Entropy_D.append(val)
        else:
            Entropy_D.append(D[x])
        val = np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))
        if not np.isnan(val):
            Entropy_A.append(val)
        else:
            Entropy_A.append(A[x])
    return cA_mean[0],cA_mean[1],cA_std[0],cA_std[1],cD_mean[0],cD_mean[1],cD_std[0],cD_std[1],cA_Energy[0],cA_Energy[1],cD_Energy[0],cD_Energy[1],Entropy_A[0],Entropy_A[1],Entropy_D[0],Entropy_D[1]


def maxPwelch(data_win,Fs):
    
    BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([2,(len(BandF)-1)]);
    
    for j in range(2):
        f,Psd = signal.welch(data_win[j,:], Fs)
        
        for i in range(len(BandF)-1):
            fr = np.where((f>BandF[i]) & (f<=BandF[i+1]))
            PMax[j,i] = np.max(Psd[fr])
    
    return PMax[0,0],PMax[1,0],PMax[0,1],PMax[1,1],PMax[0,2],PMax[1,2],PMax[0,3],PMax[1,3]

print ("Your program is running.")

#initialization
t = 0
data = np.zeros((256,2), float)

with open('model.pkl', 'rb') as fid:
    model = cPickle.load(fid)

host = "192.168.137.1"
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

#function Defination

t = 0
while t<256:
	try:
		message, address = s.recvfrom(8192)
		_list = [x.strip() for x in message.split(',')]
		data[t,:] = _list[2:4]

	except KeyboardInterrupt:
		break
	t += 1
# data = data[30:,:]
mean_x = np.mean(data[:,0])
mean_y = np.mean(data[:,1])
data_x = (data[:,0]-mean_x)
data_y = (data[:,1]-mean_y)
max_val = max(abs(np.amax(data_x)),abs(np.amax(data_y)),abs(np.amin(data_x)),abs(np.amin(data_y)))
data_x = data_x/max_val
data_y = data_y/max_val
data_cleaned = np.empty((len(data),0), float)
data_cleaned = np.append(data_cleaned, data_x.reshape(len(data),1), axis=1)
data_cleaned = np.append(data_cleaned, data_y.reshape(len(data),1), axis=1)
data_cleaned = data_cleaned.astype(float)

data_cleaned = np.transpose(data_cleaned)
# print len(data_cleaned)
# print len(data_cleaned[0])


features = []

max1,max2 = max_val_func(data_cleaned)
features.append(max1)
features.append(max2)

min1,min2 = min_val(data_cleaned)
features.append(min1)
features.append(min2)

Std1,Std2 = std_val(data_cleaned)
features.append(Std1)
features.append(Std2)

Kurtosis1,Kurtosis2 = my_kurtosis(data_cleaned)
features.append(Kurtosis1)
features.append(Kurtosis2)

Skewness1,Skewness2 = skewness(data_cleaned)
features.append(Skewness1)
features.append(Skewness2)

feature_list = hjorth(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = first_diff_mean(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = first_diff_max(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = secDiffMean(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = secDiffMax(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = wavelet_features(data_cleaned)
for feat in feature_list:
    features.append(feat)
feature_list = maxPwelch(data_cleaned,100)
for feat in feature_list:
    features.append(feat)

prediction = model.predict(features)[0]
print prediction
