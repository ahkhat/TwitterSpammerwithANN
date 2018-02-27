# Artificial Neural Network
# Part 1 - Data Preprocessing veri önişlemeden geçirilicek

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('95k-continuous.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling veri ölçeklendiriliyor 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#buraya kadar veryi işledik hazır hale getirdik




#part 4 -- evaluating the ann cross validaiton kullanıcaz her derledğinde accuracy farklı çıkıyor buyüzden daha doğru ve fazla değer etmeliyiz
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential #neural networku başlatmak için import ettik for initialize to ann
from keras.layers import Dense
# implement key for crossvalidation inside  to keras 10 kere aynı anda trainning yapıcaz
def build_classifier():# function içinde oyüzden local bir calssifier bu f
    classifier= Sequential()# parantezler boş buraya ne geleceğini zamanla yazocaz layerler oluşsun bakıcaz
    classifier.add(Dense(activation="tanh", input_dim=11, units=6, kernel_initializer="uniform")) #giriş düğümü 11 tane böylece hem input hemde 1 tane hidden layer yarattık
    classifier.add(Dense(activation="tanh", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier, X=X_train , y=y_train, cv=10,n_jobs=1)
#cv her adımda kaçfarklı acc. değeri istersen ona göre (10 folds 9 for tarin 1 for test)
#key for crossvalidation i will use scikitlearn,estimator abject to use to fit the data which is classifier
#n_jobs kaçtane cpu kullanılacak -1 tümü demek
mean=accuracies.mean() #10 tane doğruluk değerinin ortalamaması
variance=accuracies.std() #variance değeri için busa

# değer aldık varsaysak bunların ortalaması(mean) ve değerler arasındaki değişikliği(variance) görmek için


#tunning the ann accuricy artsın dye hep
# we are try to find optimal values
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense
def build_classifier(optimizer):
    classifier= Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")) #giriş düğümü 11 tane böylece hem input hemde 1 tane hidden layer yarattık
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
paramaters={'batch_size':[10,25],#kackere devir yapacak ve yığına atıcak
            'epochs':[100,250],#devir
            'optimizer':['adam','rmsprop']}#rmsprop yine sco.gra.des.'e dayalı bir func.
        #paramators bir dictionary we are trt different values for better leaning making hyper paramaters
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=paramaters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(X_train,y_train)#with this grissearch will be fit on trainning set
best_paramaters=grid_search.best_params_
best_accuracy=grid_search.best_score_










