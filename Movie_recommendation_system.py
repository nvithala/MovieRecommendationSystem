import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from sklearn.cross_validation import train_test_split
import scipy.spatial
from scipy.stats import itemfreq

import itertools
import math
from sklearn.preprocessing import normalize

################## LOAD INPUT DATA #####################
ratings_df = pd.read_csv('C:\\Users\\nihar\\Desktop\\python_practice\\book2.csv', sep=',', names=['userId', 'movieId', 'rating'],usecols=range(3))
n_users = ratings_df.userId.unique().shape[0]
print str(n_users) + 'users'
n_items = ratings_df.movieId.unique().shape[0]
print str(n_items) + 'movies'

data = {'userId': pd.unique(ratings_df.userId.ravel())}
table1 = pd.DataFrame(data)
table1['id'] = table1.index


dataC = {'movieId':pd.unique(ratings_df.movieId.ravel())}
table2 = pd.DataFrame(dataC)
table2['id'] = table2.index


new_df = ratings_df.merge(table1,left_on='userId', right_on='userId', how='inner')
new_df_1 = new_df.merge(table2,left_on='movieId', right_on='movieId', how='inner') 


##################### UTILITY MATRIX CALLED RATINGS #######################
ratings = np.zeros((n_users, n_items))
for row in new_df_1.itertuples():
    ratings[row[4], row[5]] = row[3]
    

B=table1.as_matrix(columns=None)
joined_ratings=np.insert(ratings,10109,B[:,1],axis=1)
print 'joined rating',joined_ratings.shape



new = ratings.mean(axis=1,keepdims=True)
ar_diff1 = np.subtract(ratings,new)
ar_diff = normalize(ar_diff1, norm='l2', axis=1, copy=True)


'''IMPLEMENT K MEANS'''
def cluster_centroids(data, clusters, k=None):
    if k is None:
        k = np.max(clusters) + 1
    result = np.empty(shape=(k,) + data.shape[1:])
    for i in range(k):
        np.mean(data[clusters == i], axis=0, out=result[i])
    return result


def kmeans(data, k=None, centroids=None, steps=20):
    print 'Entered Clustering process'
    if centroids is not None and k is not None:
        assert(k == len(centroids))
    elif centroids is not None:
        k = len(centroids)
    elif k is not None:
        # choosing k points for initial clusters
        centroids = data[np.random.choice(np.arange(len(data)), k, False)]
    else:
        raise RuntimeError("Need a value for k or centroids.")

    for _ in range(max(steps, 1)):
        # Squared distances between each point and each centroid.
        sqdists = scipy.spatial.distance.cdist(centroids, data, 'sqeuclidean')

        # Index of the closest centroid to each data point.
        clusters = np.argmin(sqdists, axis=0)

        new_centroids = cluster_centroids(data, clusters, k)
        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids
       
    return centroids,clusters


'''clustered total input'''
centroids,total_data = kmeans(ar_diff,10)
print total_data.shape,ar_diff.shape


'''TeSt and DaTa SeT'''
data_train, data_test, labels_train, labels_test = train_test_split(joined_ratings, total_data, test_size=0.20, random_state=42)
data_train_copy = np.delete(data_train,10109,1)
data_test_copy= np.delete(data_test,10109,1)
#data_train_copy


'''SOFTMAX'''
def softmax_cost(theta, num_classes, input_size, lamda, data, labels):
    """
    :param theta:
    :param num_classes: the number of classes
    :param input_size: the size N of input vector
    :param lambda_: weight decay parameter
    :param data: the N x M input matrix, where each column corresponds
             a single test set
    :param labels: an M x 1 matrix containing the labels for the input data
    """
    m = data.shape[1]   
    theta = theta.reshape(num_classes, input_size)       
    theta_data = theta.dot(data)
    theta_data = theta_data - np.max(theta_data)
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lamda / 2) * np.sum(theta * theta)
    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lamda * theta

    return cost, grad.flatten()

'''SOFTMAX TRAIN'''
def softmax_train(input_size, num_classes, lamda, data, labels, options={'maxiter': 400, 'disp': True}):
    #softmaxTrain Train a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # input_size: the size of an input vector x^(i)
    # num_classes: the number of classes
    # lambda_: weight decay parameter
    # input_data: an N by M matrix containing the input data, such that
    #            inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)
    J = lambda x: softmax_cost(x, num_classes, input_size, lamda, data, labels)
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    # Return optimum theta, input size & num classes
    opt_theta = result.x
    return opt_theta, input_size, num_classes


'''SOFTMAX_TEST'''
def softmax_predict(model, data):
    # model - model trained using softmaxTrain
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    #
    # Your code should produce the prediction matrix
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).

    opt_theta, input_size, num_classes = model
    opt_theta = opt_theta.reshape(num_classes, input_size)
    prod = opt_theta.dot(data)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)

    return pred
##########ALL DECLARATIONS HERE#################
input_size = 1*10109  
input_data = np.transpose(data_train_copy)
labels_train
num_classes    = 10     # number of classes
lamda          = 0.0001 # weight decay parameter
max_iterations = 100
theta = 0.005 * np.random.randn(num_classes * input_size)

#############CALL SOFTMAX##################
(cost, grad) = softmax_cost(theta, num_classes, input_size, lamda, input_data, labels_train)
print 'done'


#############CALL TRAINING METHOD###########
options_ = {'maxiter': 100, 'disp': True}
opt_theta, input_size, num_classes = softmax_train(input_size, num_classes,
                                                           lamda, input_data, labels_train, options_)
print 'done training'


#################CALL TESTING################
data_test_copy_1 = np.transpose(data_test_copy)
predictions = softmax_predict((opt_theta, input_size, num_classes), data_test_copy_1)
print 'done predicting'
print predictions.shape,type(predictions)
#print labels_test
print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == labels_test, dtype=np.float64) / labels_test.shape[0])

########## cluster_output+user_id #############
predictions = predictions.reshape(423,1)
joined_clusters=np.insert(predictions,1,data_test[:,10109],axis=1)



#########Compare with indicator matrix ########
m=(input_data).shape[1]
indicator = scipy.sparse.csr_matrix((np.ones(m), (labels_train, np.array(range(m)))))
indicator = np.array(indicator.todense())



########## Create a list of arrays that contains all the users in a given cluster #############
lst = []
for i in range(0,10):
    k = (np.where(indicator[i,:] == 1)[0])
    lst.append(k)


######From the list obtained take the top 5 values ###########
top5 = []
for i in range(0,423):
    x = lst[joined_clusters[i,0]]
    y = joined_clusters[i,1]
    for j in range(0,3):
        bu = (np.where(ratings[x[j],:]==5)[0])
        if(len(bu)>3):
            break
    top5 = bu[:5]
    print 'movies recommended for User:',y,'are:',top5
     

############CALCULATE RMSE#######################      
k = joined_clusters[:,0]
a = centroids[k,:]
sub = np.subtract(data_test_copy,a)
sq_out = np.square(sub)
sum_out = np.sum(sq_out)
inter = (sum_out/data_test_copy.size)
sqr = math.sqrt(inter)
print 'RMSE ERROR IS:',sqr