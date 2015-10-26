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
 




'''
#######################################################
# Scatterplot Matrix
#######################################################

# Create discrete colors from continuous colormap
import matplotlib.colors as colors
import matplotlib.cm as cmx

colorNorm = colors.Normalize(vmin=0, vmax=n_classes-1)
scalarMap = cmx.ScalarMappable(norm=colorNorm, cmap=plt.get_cmap("jet"))
colors = scalarMap.to_rgba(np.arange(0, n_classes), alpha = 0.5)

# Markers
markers = np.array(['o', 's', '^', 'd', 'D', '*'])

# Combine colors and markers for 36 categories
#colors = np.tile(np.asarray(colors), (6,1))
#colors_category = np.repeat(np.asarray(colors), 6, axis=0)
colors_category = colors
markers_category = np.tile(np.asarray(markers), (1,6))
markers_category = markers_category.flatten('F')



size_factor = 0.9 #0.48
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()

X = training_samples

fig, axes = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=figure_size, dpi=100)
fig.subplots_adjust(hspace=0.0, wspace=0.0)

xmin = X[:,0:3].min()
xmax = X[:,0:3].max()

for row in range(4):
    for col in range(4):
        ax = axes[row][col]
        # Hide ticks
        #if row != 3:
        #    ax.set_xticks([])
        #if col != 3:
        #    ax.set_yticks([])

        ax.set_xticks([])
        ax.set_yticks([])
        
        if row != col: # Not on diagonal
            for i, label_category in enumerate(unique_labels):
                category_idx = np.where(unique_idx_inv == i)[0]
                ax.plot(X[category_idx, col], X[category_idx, row], 
                        marker=markers_category[i], markersize=1.25, label=label_category, ls='',
                        markerfacecolor=colors_category[i], markeredgecolor=[0.0, 0.0, 0.0, 1.0], markeredgewidth=0.1 )
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([xmin, xmax])
                
        else: # Diagonal
            bandwidth = 0.2
            x = X[:, col]
            #n, bins, patches = ax.hist(x, bins=25,  normed=True, histtype='bar', #bar, stepfilled
            #                           lw=0.2, color=[0.3, 0.3, 0.3, 1.0], fc=[0.7, 0.7, 0.7, 1.0] )
            #xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            xs = np.linspace(xmin, xmax, 100)
            density = stats.gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
            ys = np.concatenate([np.array([0]), density(xs), np.array([0])])
            xs = np.insert(xs, 0, xs[0]); xs = np.insert(xs, -1, xs[-1])
            #ax.fill(xs, ys, lw=0.2, color=[0.3, 0.3, 0.3, 1.0], facecolor=[1.0, 1.0, 1.0, 0.5], label='bw=0.2')
            ax.fill(xs, ys, lw=0.2, color=[0.3, 0.3, 0.3, 1.0], facecolor=[0.7, 0.7, 0.7, 0.7], label='bw=0.2') 
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([0, 1.1])
    
# Labels
axes[0][0].set_xlabel(feature_labels[0]); axes[0][0].xaxis.set_label_position('top') 
axes[0][1].set_xlabel(feature_labels[1]); axes[0][1].xaxis.set_label_position('top') 
axes[0][2].set_xlabel(feature_labels[2]); axes[0][2].xaxis.set_label_position('top') 
axes[0][3].set_xlabel(feature_labels[3]); axes[0][3].xaxis.set_label_position('top') 
axes[0][0].set_ylabel(feature_labels[0])
axes[1][0].set_ylabel(feature_labels[1])
axes[2][0].set_ylabel(feature_labels[2])
axes[3][0].set_ylabel(feature_labels[3])

# Ticks
#axes[0][3].yaxis.set_ticks_position('right')
#axes[1][3].yaxis.set_ticks_position('right')
#axes[2][3].yaxis.set_ticks_position('right')
#axes[3][3].yaxis.set_ticks_position('right')
#axes[3][0].xaxis.set_ticks_position('bottom')
#axes[3][1].xaxis.set_ticks_position('bottom')
#axes[3][2].xaxis.set_ticks_position('bottom')
#axes[3][3].xaxis.set_ticks_position('bottom')


plt.tight_layout()
#plt.show()

plotname = "scatterplotmatrix"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()

#-------------------------
# Legend in seperate plot
figure_size = [text_width, 0.31*text_width]
#fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=figure_size, dpi=100)
fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')

# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, 0.5*box.width, 0.5*box.height])

#Create custom artists
artists_category = []
for i, _ in enumerate(unique_labels):
    artists_category.append( plt.Line2D((0,1),(0,0), 
                             marker=markers_category[i], linestyle='', markersize=3,
                             markerfacecolor=colors_category[i], markeredgecolor=[0.0, 0.0, 0.0, 1.0]) )

# Create legend from custom artist/label lists
#plt.figlegend([artist for artist in artists_category], [label for label in unique_labels], 
#          fontsize='x-small', labelspacing=0.5, handletextpad=0.05,
#          loc='center right', bbox_to_anchor=(0, 0.5, 1, 1), bbox_transform=plt.gcf().transFigure )


legend = plt.legend([artist for artist in artists_category], [label for label in unique_labels], ncol=3, loc='center',
                    fontsize='small', labelspacing=0.5, handletextpad=0.05, columnspacing=0.5, borderpad=0.5,
                    fancybox=True, shadow=False, framealpha=1.0)

plotname = "scatterplotmatrix_legend"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
'''







'''
#######################################################
# Bean plot (standardized features)
#######################################################
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

config.load_config_small()
figure_size = [1*text_width, 1*text_height]


n_columns = 54
n_rows = 12 # 36
rows = np.array([[0, 12], [12, 24], [24, 36]])
#labels = np.arange(0, 54, 1)
labels = feature_labels[0:54]

for page in range(0,3):
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex=False, sharey=True, squeeze=True, figsize=figure_size, dpi=100)
    for i, label_category in enumerate(unique_labels[rows[page][0]:rows[page][1]]):
        label_idx = i+rows[page][0]
        category_idx = np.where(unique_idx_inv == label_idx)[0]
        X = training_samples[category_idx, 0:n_columns].transpose()
        ax = axes[i]
    
        sm.graphics.beanplot(X, ax=ax, labels=labels, jitter=False,
                             plot_opts={'violin_fc':[0.9, 0.9, 0.9, 0.5],
                                        'violin_ec':[0.0, 0.0, 0.0, 1.0],
                                        'violin_lw':0.2,
                                        'bean_color':[0.0, 0.0, 0.0, 0.5],
                                        'bean_size':0.1,
                                        'bean_lw':0.2,
                                        'bean_show_mean':False,
                                        'bean_show_median':False,
                                        'cutoff':False,
                                        'cutoff_val':3,
                                        'cutoff_type':'std' }) #abs
                                        #'label_fontsize':'small',
                                        #'label_rotation':0})
                      
        #sm.graphics.violinplot(X, ax=ax, labels=y, show_boxplot=False,
        #                       plot_opts={'violin_fc':[0.9, 0.9, 0.9, 0.5],
        #                                  'violin_ec':[0.0, 0.0, 0.0, 1.0],
        #                                  'violin_lw':0.2})
                               
        ax.set_ylabel(label_category, rotation=0, rotation_mode='anchor', ha='right', va='center')    
        ax.set_ylim(-10, 10)

    # Labels, ticks
    axes[0].xaxis.tick_top()
    axes[0].set_xticklabels(labels, rotation=90 );
    #axes[0].set_xlabel("Features")
    axes[0].xaxis.set_label_position('top') 
    for ax in axes[1:]:
        ax.xaxis.set_ticklabels([])  

    plt.tight_layout(pad=1.0, w_pad=0.25, h_pad=0.5)
    #plt.show()
    plotname = "beanplot_features_page_" + str(page)
    fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
    plt.close()
'''






'''
#######################################################
# Bean plot (LDA)
#######################################################

config.load_config_small()
figure_size = [1*text_width, 1*text_height]

## Transform LDA
n_components = 14
lda = LDA(n_components=n_components)
lda.fit(training_samples, training_labels)    
training_samples_transformed = lda.transform(training_samples)

n_columns = n_components
n_rows = 12 # 36
rows = np.array([[0, 12], [12, 24], [24, 36]])
labels = np.arange(0, n_columns, 1) #feature_labels[0:54] # np.arange(0, 10)

for page in range(0,3):
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex=False, sharey=True, squeeze=True, figsize=figure_size, dpi=100)

    for i, label_category in enumerate(unique_labels[rows[page][0]:rows[page][1]]):
        label_idx = i+rows[page][0]
        category_idx = np.where(unique_idx_inv == label_idx)[0]
        X = training_samples_transformed[category_idx, 0:n_columns].transpose()
        ax = axes[i]
    
        sm.graphics.beanplot(X, ax=ax, labels=labels, jitter=False,
                             plot_opts={'violin_fc':[0.9, 0.9, 0.9, 0.5],
                                        'violin_ec':[0.0, 0.0, 0.0, 1.0],
                                        'violin_lw':0.2,
                                        'bean_color':[0.0, 0.0, 0.0, 0.5],
                                        'bean_size':0.1,
                                        'bean_lw':0.2,
                                        'bean_show_mean':False,
                                        'bean_show_median':False,
                                        'cutoff':False,
                                        'cutoff_val':3,
                                        'cutoff_type':'std' }) #abs                       
                                        #'label_fontsize':'small',
                                        #'label_rotation':0})                             
                                      
        #sm.graphics.violinplot(X, ax=ax, labels=y, show_boxplot=False,
        #                       plot_opts={'violin_fc':[0.9, 0.9, 0.9, 0.5],
        #                                  'violin_ec':[0.0, 0.0, 0.0, 1.0],
        #                                  'violin_lw':0.2})
                                      
        ax.set_ylabel(label_category, rotation=0, rotation_mode='anchor', ha='right', va='center')    


    # Labels, ticks
    axes[0].xaxis.tick_top()
    axes[0].set_xticklabels(labels, rotation=90 );
    axes[0].set_xlabel("Features")
    axes[0].xaxis.set_label_position('top') 
    for ax in axes[1:]:
        ax.xaxis.set_ticklabels([])  

                   
    plt.tight_layout(pad=1.0, w_pad=0.25, h_pad=1.0)
    #plt.show()
    plotname = "beanplot_features_lda_page_" + str(page)
    fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
    plt.close()
'''











'''
#######################################################
# Recursive feature elimination
#######################################################
x = training_samples
y = training_labels

classifier = svm.SVC( C=120, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=False, random_state=None,
                      shrinking=False, tol=0.001, verbose=False)

classifier.fit(x, y)
w = classifier.coef_
svm_weights_abs = np.abs(w).sum(axis=0)
svm_weights_squared = (w ** 2).sum(axis=0)

rfe = RFE(estimator=classifier, n_features_to_select=1, step=1)
rfe.fit(x, y)

sorted_ranking = np.sort(rfe.ranking_)
sorted_ranking_idx = np.argsort(rfe.ranking_)
feature_idx = np.arange(feature_labels.shape[0])


#----------
# Plotting
#----------
size_factor = 1.0 #0.48
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
figure_height = 0.5 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()


fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

plt.bar(feature_idx, svm_weights_abs, width=0.7, label='SVM weight',
        color=config.UIBK_orange, alpha=1.0, align="center", linewidth=0.35, edgecolor=[0.0, 0.0, 0.0])
#plt.barh(feature_idx, svm_weights_abs, height=0.8, label='SVM weight',
#        color=config.UIBK_orange, alpha=0.75, align="center", linewidth=0.25)


ax.set_xlim([-1, 54])
ax.set_xticks(feature_idx)

#ax.set_xticklabels(feature_labels, rotation=270, fontsize=6, ha="center", va="top")
ax.set_xticklabels(feature_labels, rotation=90, fontsize=6, ha="center", va="top")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(which='both', direction='out')

ax.set_ylabel(r"Multiclass SVM: Feature Relevance", rotation=90)
#ax.set_xlabel("Features", rotation=0)

fig.tight_layout()
#plt.show() 
plotname = "rfe_ranking"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()

'''






'''

#######################################################
# Recursive feature elimination with cross-validation
#######################################################
# Number of features VS. cross-validation scores

x = training_samples
y = training_labels

classifier = svm.SVC( C=120, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=False, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

cv = cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=0.2, random_state=42)

rfecv = RFECV(estimator=classifier, cv=cv, step=1, scoring='accuracy')
rfecv.fit(x, y)

print("Optimal number of features : %d" % rfecv.n_features_)


#----------
# Plotting
#----------
size_factor = 0.75 #0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
#figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

ax.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color=config.UIBK_orange, alpha=0.75, lw=1.5)

ax.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linestyle="None", marker=".", markersize=6,
        markeredgewidth=0.5, markeredgecolor=[0.3, 0.3, 0.3], markerfacecolor=[1.0, 1.0, 1.0, 1.0])

ax.set_xlim([0,n_features+1])
ax.set_xlabel("Number of selected features")
ax.set_ylabel("Cross validation score (mean accuracy)")

#rfecv.support_
#rfecv.ranking_

fig.tight_layout()
#plt.show() 
plotname = "rfe_scores"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()

'''












'''
#######################################################
# 2D Gridsearch
#######################################################
# Pipelining: chaining a LDA and svm
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html

x = training_samples
y = training_labels

lda = LDA()
clf = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma=0.125, kernel='linear', max_iter=-1, probability=False, random_state=None,
              shrinking=False, tol=0.001, verbose=False)

pipe = Pipeline(steps=[('lda', lda), ('clf', clf)])

n_runs = 10
n_components = np.arange(2, 23, 1)
#n_components = np.arange(2, 54, 1)
C_range = np.concatenate([np.array([1.0]), np.linspace(10, 200, 20)])
#C_range = np.linspace(10, 200, 20) #[100]
param_grid = dict(lda__n_components=n_components, clf__C=C_range)
scores_combined = np.array([]).reshape(0, len(n_components))

# 2D Grid search with LDA
for i in range(0, n_runs):
    cv = cross_validation.StratifiedShuffleSplit(training_labels, n_iter=10, test_size=0.2, random_state=None)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=0, scoring='accuracy') #verbose=2

    grid.fit(x, y)

    #print("The best classifier is: ", grid.best_estimator_)

    # Extract scores in 2D array
    scores = np.array([sc[1] for sc in grid.grid_scores_]).reshape(len(C_range), len(n_components))
    scores_combined = np.vstack((scores_combined, scores))

# 1D Grid search without LDA
scores_without_LDA_combined = np.array([]).reshape(0, 1)
for i in range(0, n_runs):
    cv = cross_validation.StratifiedShuffleSplit(training_labels, n_iter=10, test_size=0.2, random_state=None)
    grid_without_LDA = GridSearchCV(clf, param_grid= [ {'kernel': ['linear'], 'C': C_range } ], cv=cv, verbose=0, scoring='accuracy')  #verbose=2
    grid_without_LDA.fit(x, y)
    scores_without_LDA = np.array([sc[1] for sc in grid_without_LDA.grid_scores_]).reshape(len(C_range), 1)
    scores_without_LDA_combined = np.vstack((scores_without_LDA_combined, scores_without_LDA))

# Average runs
scores_sum = scores_combined[0:len(C_range), :]
scores_without_LDA_sum = scores_without_LDA_combined[0:len(C_range)]
for i in range(1, n_runs):
    l = i*len(C_range) 
    h = (i+1) * len(C_range)
    scores_sum += scores_combined[l:h, :]
    scores_without_LDA_sum += scores_without_LDA_combined[l:h]

scores_final = np.hstack([scores_sum, scores_without_LDA_sum]) / n_runs


#----------
# Plotting
#----------
width = scores_final.shape[1]
height = scores_final.shape[0]
xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))

size_factor = 0.66
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 0.8 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

#colormap = plt.get_cmap('UIBK_ORANGES')
#colormap = plt.cm.spectral
colormap = plt.get_cmap('YlOrRd_r')

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
#plt.imshow(scores_combined, interpolation='nearest', cmap=colormap)
im = ax.pcolormesh(xs-0.5, ys-0.5, scores_final, cmap=colormap, 
                   shading="flat", linewidth=0.3, edgecolor=[0.2, 0.2, 0.2]) 

cbar = plt.colorbar(im,fraction=0.045, pad=0.05)
cbar.solids.set_edgecolor("face")

# pcolormesh settings to recreate imshow
ax.set_aspect('equal')
ax.set_xlim([-0.5, width-0.5])
ax.set_ylim([height-0.5, -0.5])
ax.invert_yaxis() # Workaround inverted y-axis

plt.xlabel('LDA components')
plt.ylabel('Regularisation parameter C')
#plt.xticks(np.arange(len(n_components)), n_components, rotation=0)
plt.yticks(np.arange(len(C_range)), C_range)


xlabels = [str(item) for item in n_components]
xlabels.append("54")
plt.xticks(np.arange(scores_final.shape[1]), xlabels, rotation='0')

ax.annotate("Original", xy=(21, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -13.0), textcoords='offset points', va='top', ha='center', fontsize=6 ) #xytext=(0, -4.0)

plt.tight_layout(pad=1.0)

#plt.show()
plotname = "grid_search_C_LDA"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi)
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi)
plt.close()
'''








'''
#######################################################
# Illustration SVM classification
#######################################################

#class1_idx = np.asarray(np.where(training_labels == "hockey ball")) # unique_labels[]
#class2_idx = np.asarray(np.where(training_labels == "foam ball")) # unique_labels[]

#class1_idx = np.asarray(np.where(training_labels == "aerosol can")) # unique_labels[]
#class2_idx = np.asarray(np.where(training_labels == "bottle plastic vinegar")) # unique_labels[]

#class1_idx = np.asarray(np.where(training_labels == "coffee mug violet")) # unique_labels[]
#class2_idx = np.asarray(np.where(training_labels == "plastic cup white")) # unique_labels[]


class1_id = np.where(np.asarray(unique_labels) == "aerosol can")[0][0]
class2_id = np.where(np.asarray(unique_labels) == "bottle plastic vinegar")[0][0]
class1_idx = np.asarray(np.where(unique_idx_inv == class1_id))
class2_idx = np.asarray(np.where(unique_idx_inv == class2_id))
classes_idx = np.concatenate([class1_idx, class2_idx], axis = 1).ravel()

X = training_samples[classes_idx, 0:2]
X[:,[0, 1]] = X[:,[1, 0]] # swap dimensions for aesthetic reasons
Y = unique_idx_inv[classes_idx]

classifier = svm.SVC( C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=True, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

classifier.fit(X, Y)


#----------
# Plotting
#----------
size_factor = 0.48 #75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
#figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()



fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

# get the separating hyperplane
w = classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:,0].min(), X[:,0].max())
yy = a * xx - (classifier.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors
margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

# plot the line, the points, and the nearest vectors to the plane
ax.plot(xx, yy, ls="-", color="k", lw=1.5, zorder=-1, label="Separating hyperplane")
ax.plot(xx, yy_down, ls="--", dashes=[3,2], color=[0.3, 0.3, 0.3], zorder=-1, label="Soft margins C=100")
ax.plot(xx, yy_up, ls="--", dashes=[3,2], color=[0.3, 0.3, 0.3], zorder=-1)

# Support vectors
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=60, facecolors=[1.0, 1.0, 1.0, 1.0], zorder=0, label="Support vectors")

#ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)


ax.plot(X[Y==class1_id, 0], X[Y==class1_id, 1], linestyle="None", label="Aerosol can", 
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_orange+[0.75])

ax.plot(X[Y==class2_id, 0], X[Y==class2_id, 1], linestyle="None", label="Plastic bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_blue+[0.75])


#ax.set_xlim([X[:,0].min(), X[:,0].max()])
#ax.set_ylim([X[:,1].min(), X[:,1].max()])
ax.set_xlim([-1.25, 2.75])
ax.set_ylim([0.04, 0.22])
ax.set_xlabel("Compressibility")
ax.set_ylabel("Diameter")

# Rarange legend entries
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_order = np.array([2, 3, 4, 0, 1])
legend_handles =  [ legend_handles[i] for i in legend_order]
legend_labels = [ legend_labels[i] for i in legend_order]
ax.legend(legend_handles, legend_labels, loc="upper right", 
          fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5})


fig.tight_layout()
#plt.show() 
plotname = "illustration_classification_SVM"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
'''







'''
#######################################################
# Illustration LDA classification
#######################################################

from scipy import linalg
import matplotlib as mpl

class1_id = np.where(np.asarray(unique_labels) == "aerosol can")[0][0]
class2_id = np.where(np.asarray(unique_labels) == "bottle plastic vinegar")[0][0]
class1_idx = np.asarray(np.where(unique_idx_inv == class1_id))
class2_idx = np.asarray(np.where(unique_idx_inv == class2_id))
classes_idx = np.concatenate([class1_idx, class2_idx], axis = 1).ravel()

X = training_samples[classes_idx, 0:2]
X[:,[0, 1]] = X[:,[1, 0]] # swap dimensions for aesthetic reasons
Y = unique_idx_inv[classes_idx]

# LDA
lda = LDA(n_components=1)
#X = lda.fit(X, Y, store_covariance=True).transform(X)
y_pred = lda.fit(X, Y, store_covariance=True).predict(X)


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    # filled gaussian at 2 standard deviation
    sigma = 1
    ell = mpl.patches.Ellipse(mean, sigma * v[0] ** 0.5, sigma * v[1] ** 0.5, 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    return ell

#----------
# Plotting
#----------
size_factor = 0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
#figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

xx, yy = np.meshgrid(np.linspace(-1.25,2.75, 50), np.linspace(0.04, 0.22, 50))
X_grid = np.c_[xx.ravel(), yy.ravel()]
zz_lda = lda.predict_proba(X_grid)[:,1].reshape(xx.shape)

#ax.contourf(xx, yy, zz_lda > 0.5, alpha=0.5)
decision_boundary = ax.contour(xx, yy, zz_lda, [0.5], linewidths=1.5, colors="k", zorder=-1)
decision_boundary.collections[0].set_label("Decision boundary") # Add label

ax.plot(X[Y==class1_id, 0], X[Y==class1_id, 1], linestyle="None", label="Aerosol can", 
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_orange+[0.75])

ax.plot(X[Y==class2_id, 0], X[Y==class2_id, 1], linestyle="None", label="Plastic bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_blue+[0.75])

ellipse1 = plot_ellipse(ax, lda.means_[0], lda.covariance_, config.alphablend(config.UIBK_orange, 0.5))
ellipse2 = plot_ellipse(ax, lda.means_[1], lda.covariance_, config.alphablend(config.UIBK_blue, 0.5))


#ax.set_xlim([X[:,0].min(), X[:,0].max()])
#ax.set_ylim([X[:,1].min(), X[:,1].max()])
ax.set_xlim([-1.25, 2.75])
ax.set_ylim([0.04, 0.22])
ax.set_xlabel("Compressibility")
ax.set_ylabel("Diameter")

#ax.legend(loc="upper right", fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5})


# Get artists and labels for legend and chose which ones to display
#handles, labels = ax.get_legend_handles_labels()

#from matplotlib.lines import Line2D
#circ1 = Line2D([0], [0], linestyle="none", marker="o", alpha=1.0, markersize=5, markerfacecolor=config.alphablend(config.UIBK_orange, 0.5))
#circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=1.0, markersize=5, markerfacecolor=config.alphablend(config.UIBK_blue, 0.5))
#Create legend from custom artist/label lists
#ax.legend([handle for i,handle in enumerate(handles)] + [circ1,circ2],
#          [label for i,label in enumerate(labels)]+["Label 1", "Label 2"],
#          loc="upper right", fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5} )


#ax.legend([handle for i,handle in enumerate(handles)] + [ellipse1, ellipse2],
#          [label for i,label in enumerate(labels)]+[r"Aerosol can PDF $1\sigma$", r"Plastic bottle PDF $1\sigma$"],
#          loc="upper right", fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5} )



# Rarange legend entries
handles, labels = ax.get_legend_handles_labels()
handles = [handle for i, handle in enumerate(handles)] + [ellipse1, ellipse2]
labels = [label for i,label in enumerate(labels)] + [r"Aerosol can PDF $1\sigma$", r"Plastic bottle PDF $1\sigma$"]

order = np.array([0, 1, 3, 4, 2])
handles =  [ handles[i] for i in order]
labels = [labels[i] for i in order]

ax.legend(handles, labels, loc="upper right", 
          fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5})


#ax.legend([handle for i,handle in enumerate(handles)] + [ellipse1, ellipse2],
#          [label for i,label in enumerate(labels)]+[r"Aerosol can PDF $1\sigma$", r"Plastic bottle PDF $1\sigma$"],
#         loc="upper right", fancybox=True, shadow=False, framealpha=1.0, numpoints=1, scatterpoints=1, prop={'size':5} )


fig.tight_layout()
#plt.show() 
plotname = "illustration_classification_LDA"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
'''












'''
#######################################################
# Dimensionality reduction: 3D dataset
#######################################################

from mpl_toolkits.mplot3d import Axes3D

class1_id = np.where(np.asarray(unique_labels) == "aerosol can")[0][0]
class2_id = np.where(np.asarray(unique_labels) == "bottle glass")[0][0]
class3_id = np.where(np.asarray(unique_labels) == "bottle plastic vinegar")[0][0]
class1_idx = np.asarray(np.where(unique_idx_inv == class1_id))
class2_idx = np.asarray(np.where(unique_idx_inv == class2_id))
class3_idx = np.asarray(np.where(unique_idx_inv == class3_id))
classes_idx = np.concatenate([class1_idx, class2_idx, class3_idx], axis = 1).ravel()

X = training_samples[classes_idx, 0:3] # First 3 features
Y = unique_idx_inv[classes_idx]

classifier = svm.SVC( C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=True, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

classifier.fit(X, Y)

x = X[:,0]
y = X[:,1]
z = X[:,2]

#----------
# Plotting
#----------
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.42 #0.31
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.gca(projection='3d')

ax.plot(x[Y==class1_id], y[Y==class1_id], z[Y==class1_id], linestyle="None", label="Aerosol can", 
        marker="o", markersize=6,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_orange+[0.75])

ax.plot(x[Y==class2_id], y[Y==class2_id], z[Y==class2_id], linestyle="None", label="Glass bottle ", 
        marker="o", markersize=6,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_blue+[0.75])

ax.plot(x[Y==class3_id], y[Y==class3_id], z[Y==class3_id], linestyle="None", label="Plastic bottle", 
        marker="o", markersize=6,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=brewer_red+[0.75])

# Set viewpoint.
ax.azim = 50 #150
ax.elev = 25

# Label axes
ax.set_xlabel("Diameter")
ax.set_ylabel("Compressibility", rotation=180)
ax.set_zlabel(r"$\sigma$ Matrix 1", rotation=-180)

#ax.axis('equal')
#ax.invert_yaxis()

#ax.set_zlim([10,100])

# Background
#ax.grid(False)
ax.xaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.yaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.zaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.legend(fancybox=True, shadow=False, framealpha=1.0, prop={'size':7})

#plt.show()

#ax.xaxis.set_major_locator(MaxNLocator(6))
#ax.zaxis.set_major_locator(MaxNLocator(7))

fig.tight_layout()

plotname = "illustration_3D"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()






#######################################################
# Illustration LDA
#######################################################
# Transform LDA
lda = LDA(n_components=2)
X_lda = lda.fit(X, Y).transform(X)

#----------
# Plotting
#----------
size_factor = 0.38
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

ax.plot(X_lda[Y==class1_id, 0], X_lda[Y==class1_id, 1], linestyle="None", label="Aerosol can", 
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_orange+[0.75])

ax.plot(X_lda[Y==class2_id, 0], X_lda[Y==class2_id, 1], linestyle="None", label="Glass bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_blue+[0.75])

ax.plot(X_lda[Y==class3_id, 0], X_lda[Y==class3_id, 1], linestyle="None", label="Plastic bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=brewer_red+[0.75])

ax.set_xlabel("LD 1")
ax.set_ylabel("LD 2")

#ax.legend(loc="upper left", fancybox=True, shadow=False)

fig.tight_layout()
#plt.show() 
plotname = "illustration_LDA"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()




#######################################################
# Illustration PCA
#######################################################
# Transform PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X, Y).transform(X)


#----------
# Plotting
#----------
size_factor = 0.38
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(1, 1, 1)

ax.plot(X_pca[Y==class1_id, 0], X_pca[Y==class1_id, 1], linestyle="None", label="Aerosol can", 
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_orange+[0.75])

ax.plot(X_pca[Y==class2_id, 0], X_pca[Y==class2_id, 1], linestyle="None", label="Glass bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=config.UIBK_blue+[0.75])

ax.plot(X_pca[Y==class3_id, 0], X_pca[Y==class3_id, 1], linestyle="None", label="Plastic bottle",
        marker="o", markersize=4,  markeredgewidth=0.5, markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=brewer_red+[0.75])

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")

#ax.legend(loc="lower right", fancybox=True, shadow=False)

fig.tight_layout()
#plt.show() 
plotname = "illustration_PCA"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
'''






'''
######################################################
# Self-organizing map (SOM) (unsupervised clustering)
######################################################
import pandas as pd
pd.__version__ # Should be '0.15.1'

sys.path.append("SOMPY")
import sompy as SOM

som = SOM.SOM('som', training_samples, mapsize = [30, 30], norm_method = 'var', initmethod='pca')
som.train(n_job = 1, shared_memory = 'no', verbose='off')

#som.view_map(what='codebook', which_dim='all', pack='Yes', text_size=2.8, save='No', 
#            save_dir='empty', grid='No', text='Yes', cmap='None', COL_SiZe=6)

# Hitmap
#som.hit_map()

# Cluster nodes
labels_sm = som.cluster(method='Kmeans', n_clusters=36)

# Hitmap with cluster 
som.hit_map_cluster_number()

'''
