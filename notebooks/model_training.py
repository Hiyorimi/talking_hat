from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
from rgf.sklearn import RGFClassifier
from rgf.sklearn import FastRGFClassifier
from lightgbm import LGBMClassifier
import numpy as np

from data_loaders import load_processed_data

def report_quality(model_name = "Model 1 XGboost", y_true = [1], y_pred = [0], verbose=False):
    """
    Prints sklearn classification_report, accuracy_score, roc_auc_score
    :param model_name:
    :param y_true:
    :param y_pred:
    :param verbose: verbosity flag
    """
    print('{} Report'.format(model_name))
    if verbose:
        print(classification_report(y_true, y_pred))
    # Let's use accuracy score
    print("Accuracy for %s: %.2f" % (model_name, accuracy_score(y_true, y_pred) * 100))
    print("ROC AUC for %s: %.2f" % (model_name, roc_auc_score(y_true, y_pred) ) )
    print("\n")


def train_single_classifier_type (model, model_name, parameters,
                                  X_train, X_test, y_train, y_test, verbose=False):
    """
    Trains one model type, tuning hyperparameters with GridSearchCV
    :param model: model as the __init__ result
    :param model_name: string for verbose output
    :param parameters: params grid
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param verbose: verbose flag
    :return: model with default parameters and tuned one.
    """
    default_model = model.fit(X_train, y_train)
    # prediction and Classification Report
    pred1 = default_model.predict(X_test)

    report_quality("{} Default".format(model_name), y_test, pred1)

    model_grid = GridSearchCV(model, parameters, n_jobs=-1,
                            cv=4,
                            scoring='roc_auc',
                            verbose=0, refit=True)

    model_grid.fit(X_train, y_train)
    best_parameters, score, _ = max(model_grid.grid_scores_, key=lambda x: x[1])

    if verbose:
        print ("Best params for {} GridSearch: {}\nScore: {}".format(model_name, best_parameters, score))
        print ("\n")

    pred2 = model_grid.predict(X_test)
    report_quality('Tuned {}'.format(model_name), y_test, pred2)

    return default_model, model_grid


def train_classifiers(X_data, y):
    """
    Trains several classifiers and reporting model quality.
    :param X_data:
    :param y:
    :return: trained models
    """
    # Split the dataset into Train and Test
    seed = 42
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)

    svm = SVC()
    svm_params = {'C': [1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.001, 0.0001],
                  'kernel': ['linear', 'rbf']
                  }

    svm_model, svm_grid = train_single_classifier_type(svm,
                                                       "SVM",
                                                       svm_params, X_train, X_test, y_train, y_test)

    knn = KNeighborsClassifier()
    knn_params = {'n_neighbors':[5,6,7,8,9,10],
                  'leaf_size':[1,2,3,5],
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                  'n_jobs':[-1]
                  }
    knn_model, knn_grid = train_single_classifier_type(knn,
                                                       "KNN",
                                                       knn_params, X_train, X_test, y_train, y_test)

    # Train the XGboost Model for Classification
    xgb_model = xgb.XGBClassifier()

    # brute force scan for all parameters, here are the tricks
    # usually max_depth is 6,7,8
    # learning rate is around 0.05, but small changes may make big diff
    # tuning min_child_weight subsample colsample_bytree can have
    # much fun of fighting against overfit
    # n_estimators is how many round of boosting
    # finally, ensemble xgboost with multiple seeds may reduce variance
    xgb_parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.05, 0.1],  # so called `eta` value
                  'max_depth': [6, 7, 8],
                  'min_child_weight': [1, 11],
                  'silent': [1],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7, 0.8],
                  'n_estimators': [5, 100, 1000],  # number of trees, change it to 1000 for better results
                  'missing': [-999],
                  'seed': [1337]}

    train_model1, xgb_grid = train_single_classifier_type(xgb_model,
                                                          "XGBoost",
                                                          xgb_parameters,  X_train, X_test, y_train, y_test)


    rfc = RandomForestClassifier()

    rfc_parameters = {
        'max_depth': [4, 5, 6],
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [2, 5, 10],
    }

    rfc_model, rfc_grid = train_single_classifier_type(rfc,
                                                       "Random Forest",
                                                       rfc_parameters,  X_train, X_test, y_train, y_test)

    ext = ExtraTreesClassifier()

    ext_parameters = {
         'n_estimators': [50, 100],
         'max_features': [5, 10, 25],
         'min_samples_leaf': [2, 5, 10],
         'min_samples_split': [2, 5, 10],
     }

    ext_model, ext_grid = train_single_classifier_type(ext,
                                                       "Extra Trees",
                                                       ext_parameters,  X_train, X_test, y_train, y_test)

    lgbm = LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         n_jobs=-1,  # Updated from 'nthread'
                         silent=True)
    # Create parameters to search
    lgbm_parameters = {
        'max_depth': [5, 6, 7, 8, 9, 10, 15, 20],
        'learning_rate': [0.005],
        'n_estimators': [100, 150, 500],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [501],  # Updated from 'seed'
        'colsample_bytree': [0.65],
        'subsample': [0.7],
        'reg_alpha': [1, 10],
        'reg_lambda': [10, 100],
    }
    lgbm_model, lgbm_grid = train_single_classifier_type(lgbm,
                                                       "LGBM",
                                                       lgbm_parameters, X_train, X_test, y_train, y_test)



    rgf = RGFClassifier()
    rgf_parameters = {'max_leaf': [900],
                  'l2': [0.1, 0.05, 1.0],
                  'min_samples_leaf': [5, 4, 3],
                  'algorithm': ["RGF", "RGF_Opt", "RGF_Sib"],
                  'loss': ["Log"],
                  }

    rgf_model, rgf_grid = train_single_classifier_type(rgf,
                                                       "RGF",
                                                       rgf_parameters, X_train, X_test, y_train, y_test)


    frgf = FastRGFClassifier()
    frgf_parameters = {'max_leaf': [100, 200, 900],
                  'n_estimators': [100, 1000],
                  'max_bin': [10, 100],
                  'l2': [0.1, 100, 1000],
                  'min_samples_leaf': [5, 6],
                  'opt_algorithm': ['rgf'],
                  'loss': ["LS"],
                  }


    frgf_model, frgf_grid = train_single_classifier_type(frgf,
                                                       "FRGF",
                                                         frgf_parameters, X_train, X_test, y_train, y_test)

    return svm_model, svm_grid, \
           train_model1, xgb_grid, \
           rfc_model, rfc_grid, \
           ext_model, ext_grid, \
           lgbm_model, lgbm_grid, \
           rgf_model, rgf_grid, \
           frgf_model, frgf_grid


def prepare_data_for_training(hogwarts_df, target_column):
    """
    Cleans data of extra columns, making data usable for sinle faculty training.
    :param hogwarts_df: whole df
    :param target_column: string name of target column to leave in dataset
    :return: X_train, y
    """
    all_columns = [
        'name',
        'surname',
        'is_griffindor',
        'is_hufflpuff',
        'is_ravenclaw',
        'is_slitherin'
    ]
    columns_to_drop = []
    for c in all_columns:
        if c != target_column:
            columns_to_drop.append(c)

    data_full = hogwarts_df.copy(deep=True)
    data_full = data_full.drop(
        columns_to_drop,
        axis=1)
    X_data = data_full.drop(target_column, axis=1)
    y = data_full.loc[:, (target_column)]
    return X_data.copy(deep=True), y.copy(deep=True)


def get_faculty_models(hogwarts_df, target_column):
    """
    Searches for best model among data.
    :param hogwarts_df: pd.DataFrame
    :param target_column: string name of target column
    :return: list of models
    """
    X_data, y = prepare_data_for_training(hogwarts_df, target_column)
    faculty_models = train_classifiers(X_data, y)
    return faculty_models


def train_all_models():
    """
    Trains models for each faculty.
    :return: list of 4 models lists with trained models
    """

    hogwarts_df = load_processed_data()

    slitherin_models = get_faculty_models(hogwarts_df, 'is_slitherin')
    griffindor_models = get_faculty_models(hogwarts_df, 'is_griffindor')
    ravenclaw_models = get_faculty_models(hogwarts_df, 'is_ravenclaw')
    hufflpuff_models = get_faculty_models(hogwarts_df, 'is_hufflpuff')

    return slitherin_models, griffindor_models, ravenclaw_models, hufflpuff_models


def train_single_production_model(hogwarts_df, model, target_column):
    """
    Trains model for production use.
    :param hogwarts_df:
    :param model:
    :param target_column:
    :return: trained model
    """
    X_data, y = prepare_data_for_training(hogwarts_df, target_column)

    trained_model = model.fit(X_data, y)
    return trained_model


def train_production_models(best_models):
    """
    Prepares all the production models
    :param best_models:
    :return: list of 4 models lists with trained models
    """
    hogwarts_df = load_processed_data()

    slitherin_model = train_single_production_model(hogwarts_df, best_models[0], 'is_slitherin')
    griffindor_model = train_single_production_model(hogwarts_df, best_models[1], 'is_griffindor')
    ravenclaw_model = train_single_production_model(hogwarts_df, best_models[2], 'is_ravenclaw')
    hufflpuff_model = train_single_production_model(hogwarts_df, best_models[3], 'is_hufflpuff')

    return slitherin_model, griffindor_model, ravenclaw_model, hufflpuff_model