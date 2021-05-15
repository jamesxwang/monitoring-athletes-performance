# Packages
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBClassifier
import xgboost as xgb
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from warnings import filterwarnings
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.linear_model import BayesianRidge
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import functools
import operator

filterwarnings('ignore')

# Self-defined modules
import utility
import data_loader
from training_description.database_queries import DatabaseQueries
import data_cleaning
# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 20)


class TrainLoadModelBuilder():

    def __init__(self, dataframe, activity_features):
        TSS = 'Training Stress Score®'
        features = [feature for feature in activity_features
                    if feature != TSS
                    and not dataframe[feature].isnull().any()]
        # print('Features used for modeling: ', features)
        self.num_features = len(features)
        self.X = dataframe[features]
        self.y = dataframe[TSS]

    def _split_train_validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 25)
        # print('Shapes:  X_train: {}, y_train: {}, X_test: {}, y_test: {}'
        #     .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        return X_train, X_test, y_train, y_test

    def _validate_model_regression(self, X_test, y_test, learner):
        y_preds = learner.predict(X_test)  # predict classes for y test
        # print('Predictions Overview: ', y_preds)
        mae = mean_absolute_error(y_test, y_preds)
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))
        rsquared=r2_score(y_test, y_preds)
        return mae, rmse,rsquared

    def _display_performance_results_regression(self, model_name, mae, rmse, rsquared):
        # print('Regressor: {}'.format(model_name))
        print('Mean Absolute Error: {}, Root Mean Squared Error: {}, R-squared: {}'
              .format(round(mae, 3), round(rmse, 3), round(rsquared, 3)))

    def _display_performance_results_classification(self, model_name, accuracy, precision, recall, f1):
        print('Classifier: {}'.format(model_name))
        print('Accuracy: {}, Precision: {}, Recall: {}, F1 score: {}'
              .format(round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)))

    def _validate_model_classification(self, X_test, y_test, learner):
        y_preds = learner.predict(X_test)  # predict classes for y test
        accuracy = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds, average='macro', zero_division=0)
        recall = recall_score(y_test, y_preds, average='macro')
        f1 = f1_score(y_test, y_preds, average='macro')
        return accuracy, precision, recall, f1


class ModelLinearRegression(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        #print(list(X_train))
        #print(X_train.isnull().values.any())
        #print(np.isnan(X_train.values.any()))
        #print(X_train)
        #regressor=LinearRegression()
        ###This works
        #regressor = Ridge(alpha=0.04, normalize=True)
        ####
        #print(min(y_train),max(y_train))
        regressor=Lasso(alpha=0.04, normalize=True)
        regressor.fit(X_train, y_train)
        #scores = cross_val_score(regressor, X_train, y_train, cv=3)
        #print("Cross - validated scores:", scores)
        #################################
        # lm = LinearRegression()
        # lm.fit(X_train, y_train)
        # rfe = RFE(lm, n_features_to_select=4)
        # rfe = rfe.fit(X_train, y_train)
        ##############################
        return regressor
        #return rfe

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        ##########
        sfs = SFS(Lasso(alpha=0.04, normalize=True),
                  k_features=(3,6),
                  forward=True,
                  floating=False,
                  scoring='r2',
                  cv=3)
        sfs1 = sfs.fit(X_train, y_train)
        feat_cols = list(sfs1.k_feature_names_)
        print("The most important features are",feat_cols)
        #regressor = self._build_model(X_train[feat_cols], y_train)
        #mae, rmse, rsquared = self._validate_model_regression(X_test[feat_cols], y_test, regressor)
        ##########
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        ##########
        self._display_performance_results_regression('Linear Regression', mae, rmse,rsquared)

        return mae, regressor


class ModelSVM(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        classifier = svm.SVC(C=5, kernel='linear')
        classifier.fit(X_train, y_train)
        return classifier

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        classifier = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, classifier)
        self._display_performance_results_regression('SVM', mae, rmse, rsquared)

class ModelBayesianRidge(TrainLoadModelBuilder):
    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        clf = BayesianRidge(compute_score=True)
        clf.fit(X_train, y_train)

        return clf
    
    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('BayesianRidge', mae, rmse, rsquared)

        return mae, regressor

class ModelGaussianProcess(TrainLoadModelBuilder):
    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        k1 = 235**2 * RBF(length_scale=23.6)  # long term smooth rising trend
        k2 = 1.59**2 * RBF(length_scale=23.3) \
            * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                            periodicity_bounds="fixed")  # seasonal component
        # medium term irregularities
        k3 = 0.1**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
        k4 = 0.1**2 * RBF(length_scale=0.1) \
            + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, np.inf))  # noise terms
        # kernel_gpml = k1 + k2 + k4

        # k1 = 1**2 * RBF(length_scale=1)  # long term smooth rising trend
        # k2 = 1**2 * RBF(length_scale=1) \
        #     * ExpSineSquared(length_scale=1, periodicity=1.0)  # seasonal component
        # # medium term irregularity
        # k3 = 1**2 \
        #     * RationalQuadratic(length_scale=1, alpha=1)
        # k4 = 1**2 * RBF(length_scale=1) \
        #     + WhiteKernel(noise_level=1**2)  # noise terms
        kernel_gpml = k1 + k2 + k3 + k4

        gp_initial = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, normalize_y=True, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0)
        gp_initial.fit(X_train, y_train)
        # print("GPML kernel: %s" % gp_initial.kernel_)
        # print("Log-marginal-likelihood: %.3f"
        #     % gp_initial.log_marginal_likelihood(gp_initial.kernel_.theta))
        return gp_initial

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('GaussianProcess', mae, rmse, rsquared)

        return mae, regressor

class ModelNeuralNetwork(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, X_test, y_train, y_test):
        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        X, y = np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0)
        verbose, epochs, batch_size = 0, 30, 4
        neural_network = Sequential()
        neural_network.add(layers.LSTM(units = 32, input_shape = (X_train.shape[1], 1), return_sequences=False))
        # neural_network.add(layers.Dense(256, activation='relu'))
        # neural_network.add(layers.Dropout(0.2))
        neural_network.add(layers.BatchNormalization())
        # neural_network.add(layers.Dense(32, activation='relu'))
        neural_network.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
        opt = optimizers.Adam(learning_rate=0.01)
        neural_network.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
        neural_network.fit(X_train, y_train,  validation_data=(X_test, y_test),
                           epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
        y_preds = np.reshape(neural_network.predict(X), (X.shape[0],))
        mae = mean_absolute_error(y, y_preds)
        rmse = np.sqrt(mean_squared_error(y, y_preds))
        rsquared = r2_score(y, y_preds)
        self._display_performance_results_regression('LSTM', mae, rmse, rsquared)
        # test_mae = neural_network.evaluate(X_test, y_test, verbose=verbose)[1]
        return mae, neural_network

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        mae, neural_network = self._build_model(X_train, X_test, y_train, y_test)
        return mae, neural_network


class ModelRandomForest(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        param_grid = {
            'bootstrap': [True],
            'max_depth': [6, 8, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [2, 3, 5],
            'min_samples_split': [2, 3],
            'n_estimators': [100, 500]}
        rf = RandomForestRegressor()
        # rfc = DecisionTreeRegressor(max_depth=5, min_weight_fraction_leaf= 1e-5, min_impurity_decrease = 1, min_samples_split=2)
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                                   cv = 3, n_jobs = -1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_
        return best_grid

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('Random Forest', mae, rmse,rsquared)
#       self._display_performance_results_regression('Decision Tree', mae, rmse,rsquared)

        return mae, regressor


class TestModel(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        rfc = RandomForestRegressor(max_depth=2, random_state=0)
        rfc.fit(X_train, y_train)
        return rfc

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        return rmse, regressor


class ModelXGBoost(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        param_grid = {'objective' : ['reg:squarederror'],
                      'learning_rate' : [ 0.01,  0.05,0.07,0.09],  # so called `eta` value
                      'max_depth' : [3, 4, 5],
                      # 'min_child_weight' : [1, 5],
                      'reg_alpha' : [0.2,0.3],
                      'subsample' : [0.2,0.4],
                      # 'colsample_bytree' : [0.4,0.6],
                      'n_estimators' : [100,500]}
        xgb_reg = xgb.XGBRegressor()
        xgb_grid = GridSearchCV(estimator = xgb_reg,param_grid = param_grid,
                                cv=3,n_jobs=-1,
                                # n_jobs=5,n_jobs=5
                                verbose=False)
        xgb_grid.fit(X_train,y_train)
        best_grid = xgb_grid.best_estimator_
        return best_grid
        # xgb_reg = xgb.XGBRegressor(learning_rate=0.01,colsample_bytree = 0.4,subsample = 0.2,n_estimators=1000,reg_alpha = 0.3,
        #                              max_depth=3,min_child_weight=3,gamma=0,objective ='reg:squarederror')
        # xgb_reg.fit(X_train, y_train)
        # scores = cross_val_score(xgb_reg, X_train, y_train, cv=4)
        # print("Cross - validated scores:", scores)
        # return xgb_reg


    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('XGBoost', mae, rmse,rsquared)
        return mae, regressor


class ModelStacking(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        knn = KNeighborsClassifier(n_neighbors=1)
        rf = RandomForestClassifier(max_depth=3,max_features=6,n_estimators=50,random_state=0)
        SVM = svm.SVC(C=1.0,kernel='poly',degree=5)
        Xgb = XGBClassifier(alpha=15, colsample_bytree=0.1,learning_rate=1, max_depth=5,reg_lambda=10.0)
        gnb = GaussianNB()
        lr = LogisticRegression(C = 10.0, dual=False, max_iter=100, solver='lbfgs')
        sclf = StackingCVClassifier(classifiers=[knn, rf,lr,SVM,Xgb],
                                    meta_classifier=gnb,
                                    random_state=42)
        sclf.fit(X_train,y_train)
        return sclf

        # params = {'kneighborsclassifier__n_neighbors': [1, 5],
        #           'randomforestclassifier__n_estimators': [10, 50],
        #           'randomforestclassifier__max_depth':[3, 5, 10, 13],
        #           'randomforestclassifier__max_features': [2, 4, 6, 8, 10],
        #           'XGBClassifier__n_estimators': [400,1000],
        #           # 'XGBClassifier__max_depth': [15,20,25],
        #           # 'XGBClassifier__reg_alpha': [1.1, 1.2, 1.3],
        #           # 'XGBClassifier__reg_lambda': [1.1, 1.2, 1.3],
        #           # 'XGBClassifier__subsample': [0.7, 0.8, 0.9],
        #           'meta_classifier__C' : [0.1, 10.0]}

        # grid = GridSearchCV(estimator=sclf,
        #                     param_grid=params,
        #                     cv=5,
        #                     refit=True)
        # grid.fit(X_train, y_train)

        # cv_keys = ('mean_test_score', 'std_test_score', 'params')
        #
        # for r, _ in enumerate(grid.cv_results_['mean_test_score']) :
        #     print("%0.3f +/- %0.2f %r"
        #           % (grid.cv_results_[cv_keys[0]][r],
        #              grid.cv_results_[cv_keys[1]][r] / 2.0,
        #              grid.cv_results_[cv_keys[2]][r]))
        #
        # print('Best parameters: %s' % grid.best_params_)
        # print('Accuracy: %.2f' % grid.best_score_)
        # return grid

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('XGBoost', mae, rmse,rsquared)
        return regressor


class ModelAdaBoost(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        # the focus of parameter tuning are n_estimators and learning_rate
        param_grid = [
            {'base_estimator': [None], 'n_estimators':np.linspace(100,2000,20)},
            {'learning_rate': np.linspace(0.1,1,10), 'algorithm': ['SAMME.R'], 'random_state': [None]},
        ]

        # AB = AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.0,
        #                         algorithm='SAMME.R',  random_state=None)
        AB = AdaBoostClassifier()
        grid_search = GridSearchCV(AB, param_grid, cv=2,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train.astype('int'))
        # added.astype('int')) in case of "Unknown label type: 'continuous'" happens
        return grid_search.best_estimator_

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('AdaBoost', mae, rmse,rsquared)
        return mae, regressor


class ToyRNNModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dim=32):
        super(ToyRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        out = out.view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        new_x = []
        for item in x:
            tmp = []
            for l in item:
                tmp.append(np.array(l))
            new_x.append(np.array(tmp))

        y_predict = []
        for x in new_x:
            x = torch.tensor(x).float()
            y, h = self.forward(x)
            y = y.item()
            y_predict.append(y)
        return np.array(y_predict)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, 1, self.hidden_dim)
        return hidden

class ModelTorchRNN(TrainLoadModelBuilder):
    # PyTorch Model
    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, X_test, y_train, y_test):
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_full = np.concatenate((X_train, X_test), axis=0)
        y_full = np.concatenate((y_train, y_test), axis=0)
        print(X_train.shape, X_full.shape)
        # hyper parameter
        epochs = 100
        wd = 5e-5
        lr = 0.001 
        rnn_hidden_dim = 512
        # Build Model
        # X_train[Batch, 1, one_hots]
        # One Hot Encoding Shape: X_train[0][0][0].shape[0]
        one_hot_dim = X_train[0][0][0].shape[0]
        model = ToyRNNModel(one_hot_dim, hidden_dim=rnn_hidden_dim).cpu()
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=wd)
        t = trange(epochs, desc='Loss=0.0000', leave=True)

        # Train Model
        for epoch in t:
            model.train()
            loss_sum = 0
            for i in range(len(X_train)):
                x = np.array(X_train[i][0])
                x = torch.tensor(x).float().unsqueeze(0)
                y = torch.tensor([y_train[i]]).float()
                model.zero_grad()
                pred, _ = model(x)
                # print(pred, y)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            t.set_description("Loss=%.4f" % (loss_sum/len(X_train)))

        # Predictions
        model.eval()
        y_preds = []
        for i in range(len(X_full)):
            with torch.no_grad():
                x = np.array(X_full[i][0])
                x = torch.tensor(x).float().unsqueeze(0)
                pred, _ = model(x)
                pred = pred.item()
                y_preds.append(pred)
        y_preds = np.array(y_preds)
        mae = mean_absolute_error(y_full, y_preds)
        return mae, model

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        mae, neural_network = self._build_model(X_train, X_test, y_train, y_test)
        return mae, neural_network

class PerformanceModelBuilder():

    def __init__(self):
        pass


def process_train_load_modeling(athletes_name):
    loader = data_loader.DataLoader()
    data_set = loader.load_merged_data(athletes_name=athletes_name)
    sub_dataframe_dict = utility.split_dataframe_by_activities(data_set)
    best_model_dict = {}

    for activity, sub_dataframe in sub_dataframe_dict.items():
        utility.SystemReminder().display_activity_modeling_start(activity)
        sub_dataframe_for_modeling = sub_dataframe[sub_dataframe['Training Stress Score®'].notnull()]
        if sub_dataframe_for_modeling.shape[0] > 20:
            general_features = utility.FeatureManager().get_common_features_among_activities()
            activity_specific_features = utility.FeatureManager().get_activity_specific_features(activity)
            features = [feature for feature in general_features + activity_specific_features
                        if feature in sub_dataframe.columns
                        and not sub_dataframe[feature].isnull().any()]   # Handle columns with null

            def select_best_model():
                min_mae, best_model_type, best_regressor = float('inf'), '', None
                for model_class in [ModelBayesianRidge, ModelGaussianProcess, ModelLinearRegression, ModelNeuralNetwork, ModelRandomForest, ModelXGBoost, ModelAdaBoost]:
                    model_type = model_class.__name__[5:]
                    print('\nBuilding {}...'.format(model_type))
                    builder = model_class(sub_dataframe_for_modeling, features)
                    mae, regressor = builder.process_modeling()
                    if model_type != 'NeuralNetwork':
                        utility.save_model(athletes_name, activity, model_type, regressor)
                        if mae < min_mae: min_mae, best_model_type, best_regressor = mae, model_type, regressor
                print("\n***Best model for activity '{}' is {} with mean absolute error: {}***"
                  .format(activity, best_model_type, min_mae))
                if best_regressor is not None:
                    best_model_dict[activity] = best_model_type

            select_best_model()
            utility.SystemReminder().display_activity_modeling_end(activity, True)

        else:
            utility.SystemReminder().display_activity_modeling_end(activity, False)
    utility.update_trainload_model_types(athletes_name, best_model_dict)


def process_performance_modeling(athletes_name):
    loader = data_loader.DataLoader()
    data_set = loader.load_training_description_with_watch_data(athletes_name=athletes_name)
    sub_dataframe_dict = utility.split_dataframe_by_activities(data_set)
    best_model_dict = {}

    for activity, sub_dataframe in sub_dataframe_dict.items():
        utility.SystemReminder().display_activity_modeling_start(activity)
        sub_dataframe_for_modeling = sub_dataframe[sub_dataframe['Training Stress Score®'].notnull()]
        if sub_dataframe_for_modeling.shape[0] > 20:
            features = ['onehot']
            [feature for feature in ['onehot'] if feature in sub_dataframe.columns and not sub_dataframe[feature].isnull().any()]

            def select_best_model():
                min_mae, best_model_type, best_regressor = float('inf'), '', None
                for model_class in [ModelTorchRNN]:
                    model_type = model_class.__name__[5:]
                    print('\nBuilding {}...'.format(model_type))
                    builder = model_class(sub_dataframe_for_modeling, features)
                    mae, regressor = builder.process_modeling()
                    utility.save_model(athletes_name, activity, model_type, regressor, is_predict=True)
                    if mae < min_mae:
                        min_mae, best_model_type, best_regressor = mae, model_type, regressor
                print("\n***Best model for activity '{}' is {} with mean absolute error: {}***".format(activity, best_model_type, min_mae))
                if best_regressor is not None:
                    best_model_dict[activity] = 'Predict{}'.format(best_model_type)
            
            select_best_model()
            utility.SystemReminder().display_activity_modeling_end(activity, True)

        else:
            utility.SystemReminder().display_activity_modeling_end(activity, False)
    utility.update_performance_model_types(athletes_name, best_model_dict)

if __name__ == '__main__':
    athletes_names = [
        'eduardo oliveira',
        # 'xu chen',
        # 'carly hart'
    ]
    for athletes_name in athletes_names:
        print('\n\n\n{} {} {} '.format('='*25, athletes_name.title(), '='*25))
        process_train_load_modeling(athletes_name)
        process_performance_modeling(athletes_name)

