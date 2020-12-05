class RegressionModels:
    'Simplifying Model Creation to Use Very Efficiently.'
    
    def linear_regression (train, target):
        '''Simple Linear Regression
           Params :- 
           train - Training Set to train
           target - Target Set to predict'''
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(train, target)
        
        return model
    
    def knn_regressor (train, target, n_neighbors):
        '''KNearestNeighbors Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_neighbors - no. of nearest neighbors to take into consideration for prediction'''
        
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors = n_neighbors)
        model.fit(train, target)
        
        return model
    
    def d_tree (train, target, max_depth = 8, max_features = None, max_leaf_nodes = 31, random_state = 17):
        '''DecisionTree Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           max_depth - maximum depth that tree can grow (default set to 8)
           max_features - maximum number of features that a tree can use (default set to None)
           max_leaf_nodes - maximum number of leaf nodes that a tree can contain (default set to 31)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)'''
        
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, max_features = max_features, random_state = random_state)
        model.fit(train, target)
        
        return model
    
    def random_forest (train, target, n_estimators = 100, max_depth = 8, max_features = None, max_leaf_nodes = 31, random_state = 17):
        '''RandomForest Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_estimators - no. of trees to predict (default set to 100)
           max_depth - maximum depth that tree can grow (default set to 8)
           max_features - maximum number of features that a tree can use (default set to None)
           max_leaf_nodes - maximum number of leaf nodes that a tree can contain (default set to 31)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)'''
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, 
                                      max_features = max_features, random_state = random_state, n_jobs = -1)
        model.fit(train, target)
        
        return model
    
    def xgboost (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, colsample_bytree = 0.9, colsample_bynode = 0.9, 
                 colsample_bylevel = 0.9, importance_type = 'split', reg_alpha = 2, reg_lambda = 2):
        '''XGBoost Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_estimators - no. of trees to predict (default set to 100)
           max_depth - Maximum depth that a tree can grow (default set to 8)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)
           learning_rate - size of step to to attain towards local minima
           colsample_bytree, colsample_bynode, colsample_bylevel - part of total features to use bytree, bynode, bylevel
           importance_type - metric to split samples (default set to split)
           reg_alpha, reg_lambda - L1 regularisation and L2 regularisation respectively'''
        
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, learning_rate = learning_rate, 
                             colsample_bytree = colsample_bytree, colsample_bynode = colsample_bynode, colsample_bylevel = colsample_bylevel, 
                             importance_type = importance_type, reg_alpha = reg_alpha, reg_lambda = reg_lambda)
        model.fit(train, target)
        
        return model
    
    def xgrfboost (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, colsample_bytree = 0.9, colsample_bynode = 0.9, 
                   colsample_bylevel = 0.9, importance_type = 'split', reg_alpha = 2, reg_lambda = 2):
        '''XGRFBoost Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_estimators - no. of trees to predict (default set to 100)
           max_depth - Maximum depth that a tree can grow (default set to 8)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)
           learning_rate - size of step to to attain towards local minima
           colsample_bytree, colsample_bynode, colsample_bylevel - part of total features to use bytree, bynode, bylevel
           importance_type - metric to split samples (default set to split)
           reg_alpha, reg_lambda - L1 regularisation and L2 regularisation respectively'''
        
        from xgboost import XGBRFRegressor
        model = XGBRFRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, learning_rate = learning_rate, 
                               colsample_bytree = colsample_bytree, colsample_bynode = colsample_bynode, colsample_bylevel = colsample_bylevel, 
                               importance_type = importance_type, reg_alpha = reg_alpha, reg_lambda = reg_lambda)
        model.fit(train, target)
        
        return model
    
    def catboost (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, colsample_bylevel = 0.9, reg_lambda = 2):
        '''CatBoost Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_estimators - no. of trees to predict (default set to 100)
           max_depth - Maximum depth that a tree can grow (default set to 8)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)
           learning_rate - size of step to to attain towards local minima
           colsample_bylevel - part of total features to use bylevel
           importance_type - metric to split samples (default set to split)
           reg_lambda - L2 regularisation'''
        
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, 
                                  colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, random_state = random_state, verbose = 0)
        model.fit(train, target)
        
        return model
    
    def lightgbm (train, target, num_leaves = 32, max_depth = 8, learning_rate = 0.1, n_estimators = 100, colsample_bytree = 1.0, 
                  reg_alpha = 2, reg_lambda = 2, random_state = 17, importance_type = 'split'):
        '''LightGBM Regressor
           Params :-
           train - Training Set to train
           target - Target Set to predict
           num_leaves - maximum number of leaves that a tree can have
           max_depth - Maximum depth that a tree can grow (default set to 8)
           learning_rate - size of step to to attain towards local minima
           n_estimators - no. of trees to predict (default set to 100)
           colsample_bytree - part of total features to use bytree
           reg_alpha, reg_lambda - L1 regularisation and L2 regularisation respectively
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)
           importance_type - metric to split samples (default set to split)'''
        
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(num_leaves = num_leaves, max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators, 
                              colsample_bytree = colsample_bytree, reg_alpha = reg_alpha, reg_lambda = reg_lambda, 
                              random_state = random_state, importance_type = importance_type)
        model.fit(train, target)
        
        return model