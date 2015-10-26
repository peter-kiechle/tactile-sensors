# -*- coding: utf-8 -*-

import os, sys

# Load configuration file before pyplot
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
from scipy import stats


#from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from string import Template

import matplotlib.pyplot as plt



#---------------------------------------------------------------------------------
# Load training samples from disk using scikit's joblib (replacement of pickle)
#---------------------------------------------------------------------------------
try:
    training_samples_dict = joblib.load("dumped_training_samples.joblib.pkl")
    training_samples_raw = training_samples_dict['training_samples_raw']
    training_sample_ids = training_samples_dict['training_sample_ids']
    training_labels = training_samples_dict['training_labels']
    training_categories = training_samples_dict['training_categories']           
    print("Loading dumped_training_samples.npy!") 
except IOError:
    print("Loading dumped_training_samples.npy failed!") 

training_samples = np.copy(training_samples_raw)



#---------------------------------------------------
# Create dataset for final performance measurement
#---------------------------------------------------
#training_samples, test_samples, training_labels, test_labels = cross_validation.train_test_split(training_samples, training_labels, test_size=0.3, random_state=0)



# ----------------------
# Feature Scaling
# ----------------------

# Rescale samples (Range [0,1])
#feature_max = np.max(training_samples, axis=0, keepdims=True)
#feature_min = np.min(training_samples, axis=0, keepdims=True)
#training_samples = (training_samples-feature_min) / (feature_max-feature_min)

# Normalize samples (zero mean)
#feature_mean = np.mean(training_samples, axis=0, keepdims=True)
#feature_max = np.max(training_samples, axis=0, keepdims=True)
#feature_min = np.min(training_samples, axis=0, keepdims=True)
#training_samples = (training_samples-feature_mean) / (feature_max-feature_min)

# Standardize samples (zero mean, one standard deviation)
feature_mean = np.mean(training_samples, axis=0, keepdims=True)
feature_stddev = np.std(training_samples, axis=0, keepdims=True)
training_samples = (training_samples - feature_mean) / feature_stddev


# Just some convenient characteristics 
classes = sorted([key for key in training_categories])
n_training_samples = training_samples.shape[0]
n_features = training_samples.shape[1]
n_classes = len(classes)
n_classifiers = n_classes * (n_classes - 1) / 2 # one-vs-one

# Create categorical indices to training samples
unique_labels, unique_idx_inv = np.unique(training_labels, return_inverse=True)

# Create feature labels
pmax = 5
chebyshev_1 = np.empty(pmax*pmax, dtype='|S32')
chebyshev_5 = np.empty(pmax*pmax, dtype='|S32')
for i in range(0,pmax):
    for j in range(0,pmax):
        chebyshev_1[i*pmax+j] = Template(r"$$T_{$val1,$val2}$$ Matrix 1").substitute(val1=i, val2=j)
        chebyshev_5[i*pmax+j] = Template(r"$$T_{$val1,$val2}$$ Matrix 5").substitute(val1=i, val2=j)

feature_labels = np.array(["Diameter", "Compressibility", r"$\sigma$ Matrix 1", r"$\sigma$ Matrix 5"], dtype='|S32')
feature_labels = np.concatenate([feature_labels, chebyshev_1, chebyshev_5])


# For plotting
text_width = 6.30045 # LaTeX text width in inches
text_height = 9.25737 # LaTeX text height in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]

# Custom colormap UIBK Orange
cdict = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),

        'green': ((0.0, 1.0, 1.0),
                  (1.0, 0.5, 0.5)),

        'blue': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
                
plt.register_cmap(name='UIBK_ORANGES', data=cdict)
 

# Confusion matrix
def plot_confusion_matrix(confusion_matrix, classes, appendix):
    size_factor = 1.0
    figure_width = size_factor*text_width
    figure_size = [figure_width, figure_width]
    config.load_config_medium()
    
    # Normalize by the number of samples in each class
    confusion_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.ioff() # Disable interactive plotting
    fig = plt.figure(figsize=figure_size, dpi=100)
    ax = fig.add_subplot(111)

    n_classes = len(classes)
    ax.axis([0, n_classes, 0, n_classes])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Normalize by the number of samples in each class
    confusion_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    colormap = plt.get_cmap('UIBK_ORANGES')
    colormap.set_under([1.0, 1.0, 1.0])

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(confusion_normalized, cmap=colormap, vmin=0.001, vmax=1.0)
    #plt.colorbar(mesh)

    # Absolute number
    for i,j in ((x,y) for x in np.arange(0, len(confusion_matrix))+0.5
        for y in np.arange(0, len(confusion_matrix[0]))+0.5):
            if confusion_matrix[i][j] > 0:
                ax.annotate(str(confusion_matrix[i][j]), xy=(j,i), fontsize=4, ha='center', va='center')


    plt.tick_params(axis='x', which='minor', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='minor', left='off', right='off', labelleft='on')
    plt.tick_params(axis='x', which='major', bottom='on', top='off', labelbottom='on', direction='out')
    plt.tick_params(axis='y', which='major', left='on', right='off', labelleft='on', direction='out')

    # Set the major ticks at the centers and minor tick at the edges
    tick_marks = np.arange(n_classes)
    ax.xaxis.set_ticks(tick_marks, minor=True)
    ax.yaxis.set_ticks(tick_marks, minor=True)

    plt.xticks(tick_marks+0.5, classes, rotation=90, rotation_mode='anchor', ha='right', va='center')
    plt.yticks(tick_marks+0.5, classes)

    ax.grid(True, which='minor', linestyle='-') # Grid at minor ticks
               
    plt.title("Confusion Matrix", y=1.02)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")

    plt.tight_layout()

    plotname = "confusion_matrix_" + appendix
    fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
    #fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
    #fig.savefig(plotname+".png", pad_inches=0, dpi=300)
    plt.close(fig)




###############################################
# Simple cross validation performance metrics
###############################################

# Multiple runs
runs = 5
n_folds = 10

# Based on LibSVM
classifier = svm.SVC( C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=True, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

# SVM based on liblinear
#classifier = svm.LinearSVC(C=100, penalty="l1", loss="l2", dual=False, multi_class="ovr", class_weight=None, random_state=42, tol=0.0001)

# LDA as classifier
#classifier = LDA(n_components=14, solver='eigen', shrinkage='auto')


overall_accuracy = np.zeros([runs, 1])
for seed in range(0, runs):
    cv = cross_validation.StratifiedShuffleSplit(training_labels, n_iter=n_folds, test_size=0.2, random_state=seed)

    # Per loop metrics
    confusion_matrix = np.zeros((n_classes,n_classes), dtype='int32')
    labels_test_combined = []
    labels_prediction_combined = []

    for cv_index, (cv_train_index, cv_test_index) in enumerate(cv):
        
        # Reassemble datasets according to cross validation scheme
        samples_train = training_samples[cv_train_index]
        samples_test = training_samples[cv_test_index]
        labels_train = training_labels[cv_train_index]
        labels_test = training_labels[cv_test_index]    

        # Diameter only
        #feature_idx = np.array([0])

        # First 4 features only
        #feature_idx = np.arange(0, 4, 1)

        # Chebyshev moments only
        #feature_idx = np.arange(4, n_features, 1)

        #samples_train = samples_train[:, feature_idx]
        #samples_test = samples_test[:, feature_idx]
    
        # Transform LDA
        #lda = LDA(n_components=14, solver='eigen', shrinkage='auto')
        lda = LDA(n_components=14, solver="svd")
        lda.fit(samples_train, labels_train)    
        samples_train = lda.transform(samples_train)
        samples_test = lda.transform(samples_test)
               
        # Classify
        y_score = classifier.fit(samples_train, labels_train)
        labels_prediction = classifier.predict(samples_test)    
        
        # Store results
        labels_test_combined += list(labels_test)
        labels_prediction_combined += list(labels_prediction)

    overall_accuracy[seed] = metrics.accuracy_score(labels_test_combined, labels_prediction_combined, normalize=True, sample_weight=None)
    print("Overall accuracy: %0.3f" % overall_accuracy[seed])

print("Overall accuracy %d runs: %0.3f +- %0.3f" % (runs, overall_accuracy.mean(), overall_accuracy.std()) )





###########################################################
# SVM with crossvalidation and PCA / LDA preprocessing 
###########################################################

n_folds = 10
seed = 42

classifier = svm.SVC( C=120, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=True, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

# Split training data
cv = cross_validation.StratifiedShuffleSplit(training_labels, n_iter=n_folds, test_size=0.2, random_state=seed)

# Per loop metrics
confusion_matrix = np.zeros((n_classes,n_classes), dtype='int32')
labels_test_combined = []
labels_prediction_combined = []
for cv_index, (cv_train_index, cv_test_index) in enumerate(cv):
   
    # Reassemble datasets according to cross validation scheme
    samples_train = training_samples[cv_train_index]
    samples_test = training_samples[cv_test_index]
    labels_train = training_labels[cv_train_index]
    labels_test = training_labels[cv_test_index]    

    # Chebyshev moments only
    #feature_idx = np.hstack([np.array([0]), np.arange(3, 102, 1) ])
    #samples_train = samples_train[:, feature_idx]
    #samples_test = samples_test[:, feature_idx]
    
    # Transform LDA
    lda = LDA(n_components=14)
    lda.fit(samples_train, labels_train)    
    samples_train = lda.transform(samples_train)
    samples_test = lda.transform(samples_test)
    
    # Transform PCA   
    #pca = PCA(n_components=14)
    #pca.fit(samples_train)
    #samples_train = pca.transform(samples_train)
    #samples_test = pca.transform(samples_test)
        
    classifier.fit(samples_train, labels_train)   
    labels_prediction = classifier.predict(samples_test)

    # Store results
    labels_test_combined += list(labels_test)
    labels_prediction_combined += list(labels_prediction)


overall_accuracy = metrics.accuracy_score(labels_test_combined, labels_prediction_combined, normalize=True, sample_weight=None)
print("Overall accuracy: %0.3f" % overall_accuracy)

confusion_matrix = metrics.confusion_matrix(labels_test_combined, labels_prediction_combined)
plot_confusion_matrix(confusion_matrix, classes, "LDA_SVM") 



#################
# Custom report
#################
accuracy = np.zeros([n_classes, 1])
reliability = np.zeros([n_classes, 1])
for classID in range(n_classes):
    class_mask = np.zeros(n_classes, dtype=bool); class_mask[classID] = True
    sum_row = np.sum(confusion_matrix[classID, :])
    sum_col = np.sum(confusion_matrix[:, classID])
    accuracy[classID] = confusion_matrix[classID, classID] / (sum_row)
    reliability[classID] = confusion_matrix[classID, classID] / (sum_col)

average_accuracy = np.sum(accuracy) / n_classes
average_reliability = np.sum(reliability) / n_classes
overall_accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
#metrics.accuracy_score(labels_test_combined, labels_prediction_combined, normalize=True, sample_weight=None)


# Precision (reliability), recall (accuracy), f1-score and support
p, r, f1, s = metrics.precision_recall_fscore_support(labels_test_combined, 
                                                      labels_prediction_combined,
                                                      labels=classes,
                                                      average=None, # 'micro', 'macro', 'weighted', 'samples', None, 
                                                      sample_weight=None)

# Convert to string and build table
p_str = np.array([map("{0:.3f}".format, line) for line in p.reshape(-1,1)]) 
r_str = np.array([map("{0:.3f}".format, line) for line in r.reshape(-1,1)])
f1_str = np.array([map("{0:.3f}".format, line) for line in f1.reshape(-1,1)]) 
s_str = np.array([map("{0:d}".format, line) for line in s.reshape(-1,1)]) 

report_table = np.hstack([np.asarray(classes).reshape(n_classes,-1), 
                          p_str, r_str, f1_str, s_str])

# Latex output
print " \\\\\n".join([" & ".join(map(str,line)) for line in report_table])+" \\\\"

averages_string = "average \\/ total & {0:.3f} & {1:.3f} & {2:.3f} & {3:d} \\\\"
print(averages_string.format(np.average(p, weights=s),
                             np.average(r, weights=s),
                             np.average(f1, weights=s),
                             s.sum())) 
