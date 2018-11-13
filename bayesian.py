import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
import time
from numpy import array
from matplotlib.pyplot import ion
from scipy.stats import norm, multivariate_normal

def unpickle(file):
#Load byte data from file
  with open(file, 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    return data

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
# Print New Line on Complete
  if iteration == total: 
    print()

def load_cifar10_data(data_dir):
# Return train_data, train_labels, test_data, test_labels
# The shape of data is 32 x 32 x3
  train_data = None
  train_labels = []

  for i in range(1, 6):
    data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
    if i == 1:
      train_data = data_dic['data']
    else:
      train_data = np.vstack((train_data, data_dic['data']))
    train_labels += data_dic['labels']

  test_data_dic = unpickle(data_dir + "/test_batch")
  test_data = test_data_dic['data']
  test_labels = test_data_dic['labels']

  train_labels = np.array(train_labels)
  test_labels = np.array(test_labels)

  return train_data, train_labels, test_data, test_labels

# Computes the classification accuracy for predicted labels _pred_ as compared to the ground truth labels _gt_
def cifar_10_evaluate(pred,gt):
  indexes = 0
  for i, val in enumerate(pred):
    if pred[i] == gt[i]:
      indexes = indexes+1
# Counting number of predicted labels, which are same as true labels (how many)
  l = indexes
  p = l/len(gt)*100
  print('The classification accuracy is '+  str(p) + "%")
  return p


def cifar_10_rand(x):
  pred = []
  for i in range(len(x)):
# Here we generate random number and append it to pred
    numb = np.int32(random.randint(0,9))
    pred.append(numb)
  x = x.tolist()
  cifar_10_evaluate(pred, x)
  print ('< -- for random')


def cifar_show_guess(img, lbl_pred, lbl_real):
# In order to check where the data shows an image correctly
  ion()
  plt.imshow(img)
  plt.title(label_names[lbl_real] + ' - real; predicted: ' + label_names[lbl_pred])
  plt.show()
  plt.pause(2)
  plt.close()

def cifar_10_features(x):
  result = []
  for item in x:
    r = item[0:1024]
    g = item[1024:2048]
    b = item[2048:3072]
    means = [np.mean(r), np.mean(g), np.mean(b)]
    result.append(means)
  return np.asarray(result)


def cifar_10_bayes_learn(f, labels):
  # для каждого айтема определяем лейбл, затем считаем усредненные параметры для каждого класса
  # для начала нужно выделить классы
  # сначала определяем лейбл изображения. затем сортируем. затем считаем параметры для каждого класса по отдельности.
  class_nums = list(range(0, 10))
  data_l = len(f)
  sorted = []
  data = []
  means = []
  variances = []
  ps = []
  labels = labels.tolist()
  for lbl in class_nums:
    list_of_same_lbl = []
    for index, item in enumerate(f):
      if (labels[index] == lbl):
        list_of_same_lbl.append(item)
    sorted.append(list_of_same_lbl)

  for item in sorted:
    item = np.asarray(item)
    data.append([np.mean(item, axis = 0, dtype=np.float64), np.std(item, axis = 0, dtype=np.float64), len(item)/data_l])
    means.append(np.mean(item, axis = 0, dtype=np.float64))
    variances.append(np.std(item, axis = 0, dtype=np.float64))
    ps.append(len(item)/data_l)
  return data, means, variances, ps

def cifar_10_multivariative_learn(f, labels):
  # для каждого айтема определяем лейбл, затем считаем усредненные параметры для каждого класса
  # для начала нужно выделить классы
  # сначала определяем лейбл изображения. затем сортируем. затем считаем параметры для каждого класса по отдельности.
  class_nums = list(range(0, 10))
  data_l = len(f)
  sorted = []
  data = []
  means = []
  covariances = []
  ps = []
  labels = labels.tolist()
  for lbl in class_nums:
    list_of_same_lbl = []
    for index, item in enumerate(f):
      if (labels[index] == lbl):
        list_of_same_lbl.append(item)
    sorted.append(list_of_same_lbl)

  for item in sorted:
    item = np.asarray(item)
    means.append(np.mean(item, axis = 0, dtype=np.float64))
    covariances.append(np.cov(item, rowvar = 0))
    ps.append(len(item)/data_l)
  return means, covariances, ps



def cifar_10_bayes_classify(f_s,mu,sigma,p,test_labels):
  pred = []
  i = 0
  l = len(f_s)
  print(l)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete')
  class_nums = list(range(0, 10))
  for f in f_s:
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete')
    probabilities = []
    for lbl in class_nums:
      prob = norm.pdf(f[0], mu[lbl][0], sigma[lbl][0]) * norm.pdf(f[1], mu[lbl][1], sigma[lbl][1]) *  norm.pdf(f[2], mu[lbl][2], sigma[lbl][2]) * p[lbl]   
      probabilities.append(prob)
    maxin,indx = max((probabilities[i],i) for i in range(len(probabilities))) 
    pred.append(indx)
    i = i+1
    
  cifar_10_evaluate(pred,test_labels)

def cifar_10_multivariative_classify(f_s,mu,cov,p,test_labels):
  pred = []
  i = 0
  l = len(f_s)
  print(l)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete')
  class_nums = list(range(0, 10))
  for f in f_s:
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete')
    probabilities = []
    for lbl in class_nums:
      prob = multivariate_normal.pdf(f, mu[lbl], cov[lbl]) * p[lbl]   
      probabilities.append(prob)
    maxin,indx = max((probabilities[i],i) for i in range(len(probabilities))) 
    pred.append(indx)
    i = i+1
    
  res = cifar_10_evaluate(pred,test_labels)
  return res

def cifar_10_features_div(x,N):
  rows = len(x)
  subimg_n = int(3*(int((32/N))**2))
  f = []
  k = 0
  printProgressBar(0, rows, prefix = 'Progress:', suffix = 'Complete')
  for item in x:
    printProgressBar(k + 1, rows, prefix = 'Progress:', suffix = 'Complete') 
    for j in range(0,subimg_n):   
        
      pieces = np.mean(item[j*3072//subimg_n:(j+1)*3072//subimg_n], axis = 0, dtype=np.float64)
      f.append(pieces)
    k = k + 1
  f = np.array(f, dtype=np.float64)
  print(f.shape)
  f = np.reshape(f, (rows,subimg_n))
  return f




# Main code  
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_dir = 'cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)



cifar_10_rand(test_labels)
f_s = cifar_10_features(train_data)
data, mu, sigma, p = cifar_10_bayes_learn(f_s, train_labels)

test_f_s = cifar_10_features(test_data)

cifar_10_bayes_classify(test_f_s,mu,sigma,p,test_labels)
print('<-- for Bayes')



mu_m, cov, p_m = cifar_10_multivariative_learn(f_s, train_labels)
cifar_10_multivariative_classify(test_f_s,mu_m,cov,p_m,test_labels)

print('<-- for Multivariative')

nums = [32,16,8]

results = []
for num in nums:
  final_div_f_s = cifar_10_features_div(train_data,num)
  div_f_s = cifar_10_features_div(test_data, num)
  mu_div, cov_div, p_div = cifar_10_multivariative_learn(final_div_f_s, train_labels)
  res = cifar_10_multivariative_classify(div_f_s, mu_div, cov_div, p_div, test_labels)
  results.append(res)

  print('<-- for Division by ' + str(num))

print(results)
plt.plot(nums,results)
plt.show()

