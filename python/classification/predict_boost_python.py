# -*- coding: utf-8 -*-
import os, inspect
import cairo
from PIL import Image                                                                            
import numpy as np

from sklearn.externals import joblib

def show_image(imagefile, prediction_label, score_str):
    # Byte-order
    # PIL: RGBA
    # Cairo: ARGB (actually BGRA)

    # Open file with PIL
    image = Image.open(imagefile)
    
    # Replace transparent colros with grey
    data = np.array(image) # "data" is a height x width x 4 numpy array
    red = data[..., 0]
    green = data[..., 1]
    blue = data[..., 2]
    alpha = data[..., 3]

    # Replace white with red... (leaves alpha values alone...)
    transparent_areas = (red == 255) & (blue == 255) & (green == 255) & (alpha == 0)
    data[transparent_areas] = (166, 166, 166, 255)
    image = Image.fromarray(data)
    
    # Add padding if w != h
    w, h = image.size
    diff = h-w
    if diff > 0:
        background = Image.new('RGBA', size = (h,h), color = (166, 166, 166, 255))
        background.paste(image, (diff, 0))
        image = background


    img_RGBA = np.array(image)
    height, width, channels = img_RGBA.shape

    # RGBA -> ARGB
    img_ARGB = np.ascontiguousarray(img_RGBA[:, :, (2, 1, 0, 3)]) # Advanced indexing (continuous memory usage)
    
    # Create cairo surface to draw on 
    surface = cairo.ImageSurface.create_for_data(img_ARGB, cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)


    # Prediction label
    cr.select_font_face("FreeSans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(32)
    cr.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
    
    
    (x_bearing, y_bearing, text_width, text_height, x_advance, y_advance) = cr.text_extents(prediction_label) # Get text extents for alignment
    cr.move_to(10, 10+text_height) # Position: Top left
    cr.text_path(prediction_label) # Create path
    cr.set_source_rgba(1.0, 1.0, 1.0, 1.0) # Filling
    cr.fill_preserve()
    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0) # Outline
    cr.set_line_width(1.0)
    cr.stroke() 
    
    # Score
    cr.select_font_face("FreeSans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(32)
    cr.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

    (x_bearing, y_bearing, text_width, text_height, x_advance, y_advance) = cr.text_extents(score_str) # Get text extents for alignment
    cr.move_to(10, 50+text_height) # Position: Top left
    cr.text_path(score_str) # Create path
    cr.set_source_rgba(1.0, 1.0, 1.0, 1.0) # Filling
    cr.fill_preserve()
    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0) # Outline
    cr.set_line_width(1.0)
    cr.stroke() 
    
    # Back to PIL
    # ARGB -> RGBA
    image = Image.frombuffer( 'RGBA', (width, height), surface.get_data(), 'raw', 'BGRA', 0, 1) # mode, stride, orientation
    image.show()



    
def predict(feature_vector):
    #n_features = feature_vector.shape[0]
    #print("Feature Vector: {0:d}".format( n_features ))
    print feature_vector
    
    # Wrap in 2D numpy array
    feature_vector = np.array([feature_vector])
    
    # ----------------------
    # Feature preprocessing
    # ----------------------    
    feature_vector = (feature_vector - feature_mean) / feature_stddev 

    #--------------------------------------------------------
    # Transform features using Linear Discriminant Analysis
    #--------------------------------------------------------
    feature_vector = lda.transform(feature_vector)
    
    
    # --------
    # Predict
    # --------
    prediction_label = classifier.predict(feature_vector)
    classID = np.where(classifier.classes_ == prediction_label)[0][0]

    category = prediction_label[0].replace(" ", "_")
    imagefile = __realpath__ + "/images/" + category + ".png"

    # Compute probabilities of possible classes for samples in X.
    probabilities = classifier.predict_proba(feature_vector)
    probability = probabilities[0][classID]
    
    # Distance from the separating hyperplanes
    # Order of 0 to n classes in the one-vs-one classifiers: 
    # 0 vs 1, 0 vs 2, ... 0 vs n, 1 vs 2, ..., 1 vs n, ... n-1 vs n
    distances = classifier.decision_function(feature_vector)
    distance = distances[0][classID]
    
    print("\nPrediction:                   Platt scaling probability:")
    print("---------------------------------------------------------")
    print("{:30s} {:.4f}".format(prediction_label[0], probability))
    print("---------------------------------------------------------")
    for i in range(len(classifier.classes_)):
        print "{:30s} {:.4f}".format(classifier.classes_[i], probabilities[0][i])

    prediction_label_str = prediction_label[0].title() # Uppercase
    score_str = "Platt scaling probability: " + "{:.2f}".format(probability)
    show_image(imagefile, prediction_label_str, score_str)




# Get path of *this* script file
global __modfile__, __realpath__
__modfile__ = os.path.abspath(inspect.getsourcefile(lambda _: None))
__realpath__ = os.path.dirname(os.path.realpath(__modfile__))


# Load classifier and preprocessing steps from disk
global classifier, feature_mean, feature_stddev, lda

classifier_dict = joblib.load( __realpath__ + "/dumped_classifier.joblib.pkl")
classifier = classifier_dict['classifier']
feature_mean = classifier_dict['means']
feature_stddev = classifier_dict['stddevs']
lda = classifier_dict['LDA']

