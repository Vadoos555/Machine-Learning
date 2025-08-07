import numpy as np


class DecisionStumpClassifier:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions
    

class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.models = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # itinialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        self.models = []
        
        # iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStumpClassifier()
            min_error = float('inf')
            
            # gready search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    # error = sum of weights of missclassifeid samples
                    missclassifeid = w[y != predictions]
                    error = sum(missclassifeid)
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
            
            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            # calculate predictions and update weights
            predictions = clf.predict(X)
            
            w*= np.exp(-clf.alpha * y * predictions)
            # normalize to one
            w/= np.sum(w)
            
            # save classifier
            self.models.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.models]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        
        return y_pred
    