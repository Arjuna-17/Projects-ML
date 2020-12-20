class ClassificationModels:
    'Simplifying Models Creation to Use Very Efficiently.'
    
    def logistic_regression (train, target):
        '''Simple Logistic Regression
           Params :- 
           train - Training Set to train
           target - Target Set to predict'''
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(train, target)
        print("Training Completed .....")
        
        return model
    
    def knn_classification (train, target, n_neighbors):
        '''KNearestNeighbors Classification
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_neighbors - no. of nearest neighbors to take into consideration for prediction'''
        
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors = n_neighbors)
        model.fit(train, target)
        
        return model
    
    def d_tree_classification (train, target, max_depth = 8, max_features = None, max_leaf_nodes = 31, random_state = 17):
        '''DecisionTree Classification
           Params :-
           train - Training Set to train
           target - Target Set to predict
           max_depth - maximum depth that tree can grow (default set to 8)
           max_features - maximum number of features that a tree can use (default set to None)
           max_leaf_nodes - maximum number of leaf nodes that a tree can contain (default set to 31)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)'''
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, max_features = max_features, random_state = random_state)
        model.fit(train, target)
        
        return model
    
    def random_forest_classification (train, target, n_estimators = 100, max_depth = 8, max_features = None, max_leaf_nodes = 31, random_state = 17):
        '''RandomForest Classification
           Params :-
           train - Training Set to train
           target - Target Set to predict
           n_estimators - no. of trees to predict (default set to 100)
           max_depth - maximum depth that tree can grow (default set to 8)
           max_features - maximum number of features that a tree can use (default set to None)
           max_leaf_nodes - maximum number of leaf nodes that a tree can contain (default set to 31)
           random_state - A arbitary number to get same results when run on different machine with same params (default set to 17)'''
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, 
                                       max_features = max_features, random_state = random_state, n_jobs = -1)
        model.fit(train, target)
        print("Training Completed .....")
        
        return model
    
    def xgboost_classification (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, colsample_bytree = 0.9, 
                                colsample_bynode = 0.9, colsample_bylevel = 0.9, importance_type = 'split', reg_alpha = 2, reg_lambda = 2):
        '''XGBoost Classification
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
        
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, learning_rate = learning_rate, 
                              colsample_bytree = colsample_bytree, colsample_bynode = colsample_bynode, colsample_bylevel = colsample_bylevel, 
                              importance_type = importance_type, reg_alpha = reg_alpha, reg_lambda = reg_lambda)
        model.fit(train, target)
        print("Training Completed .....")
        
        return model
    
    def xgrfboost_classification (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, colsample_bytree = 0.9, 
                                  colsample_bynode = 0.9, colsample_bylevel = 0.9, importance_type = 'split', reg_alpha = 2, reg_lambda = 2):
        '''XGRFBoost Classification
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
        
        from xgboost import XGBRFClassifier
        model = XGBRFClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, learning_rate = learning_rate, 
                                colsample_bytree = colsample_bytree, colsample_bynode = colsample_bynode, colsample_bylevel = colsample_bylevel, 
                                importance_type = importance_type, reg_alpha = reg_alpha, reg_lambda = reg_lambda)
        model.fit(train, target)
        print("Training Completed .....")
        
        return model
    
    def catboost_classification (train, target, n_estimators = 100, max_depth = 8, random_state = 17, learning_rate = 0.1, 
                                 colsample_bylevel = 0.9, reg_lambda = 2):
        '''CatBoost Classification
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
        
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, 
                                   colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, random_state = random_state)
        model.fit(train, target)
        print("Training Completed .....")
        
        return model
    
    def lightgbm_classification (train, target, num_leaves = 32, max_depth = 8, learning_rate = 0.1, n_estimators = 100, colsample_bytree = 1.0, 
                                 reg_alpha = 2, reg_lambda = 2, random_state = 17, importance_type = 'split'):
        '''LightGBM Classification
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
        
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(num_leaves = num_leaves, max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators, 
                               colsample_bytree = colsample_bytree, reg_alpha = reg_alpha, reg_lambda = reg_lambda, 
                               random_state = random_state, importance_type = importance_type)
        model.fit(train, target)
        print("Training Completed .....")
        
        return model