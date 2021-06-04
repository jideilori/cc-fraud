import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc, recall_score, precision_score,f1_score,roc_auc_score
import xgboost as xgb
import mlflow

def remove_outlier(col):
  sorted(col)
  Q1,Q3 = col.quantile([0.25,0.75])
  IQR = Q3-Q1
  lower_range = Q1 - (1.5 * IQR)
  upper_range = Q3 + (1.5 * IQR)
  return lower_range,upper_range

def strat_split(df,target,test_size,seed):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(df, df[f"{target}"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set,strat_test_set


data = pd.read_csv('./data/raw/creditcard.csv.zip')
# Take out validation data
train_data,val_df = strat_split(data,'Class',0.2,42)

# Replacing outliers using median values in all columns except amount and class
train_data = train_data.drop(['Class','Amount'],axis=1)
train_data_cols = [i for i in train_data.columns]
for i in train_data_cols:
    low, upp = remove_outlier(train_data[f'{i}'])
    train_data[f'{i}'] = np.where(train_data[f'{i}']>upp,
                                   upp,train_data[f'{i}'])
    train_data[f'{i}'] = np.where(train_data[f'{i}']<low ,
                                   low,train_data[f'{i}'])
# Add back amount and class to dataframe
train_data['Class'] = train_data['Class']
train_data['Amount'] = train_data['Amount']

# Split into train and test set
train_data = train_data.reset_index(drop=True)
train_df,test_df = strat_split(train_data,'Class',0.2,42)

x_train = train_df.drop('Class',axis=1)
y_train = train_df['Class']

x_test = test_df.drop('Class',axis=1)
y_test = test_df['Class']

x_val = val_df.drop('Class',axis=1)
y_val = val_df['Class']



xgbclf=xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    use_label_encoder=False,
    random_state=11,
    eval_metric='mlogloss'
    )
xgbclf.fit(x_train, y_train)


# # Testing
# print('---------------------------------------------------------------')
# print('----------------------Testing---------------------------------')

# y_pred = xgbclf.predict(x_test)
# conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(conf_matrix)
# print(classification_report(y_test, y_pred))
# print(recall_score(y_test,y_pred))

# # Validation

# print('-----------------------------------------------------------------')
# print('----------------------Validation---------------------------------')
# y_val_pred = xgbclf.predict(x_val)
# val_conf_matrix = pd.crosstab(y_val, y_val_pred, rownames=['Actual'], colnames=['Predicted'])
# print(val_conf_matrix)
# print(classification_report(y_val, y_val_pred))
# print(recall_score(y_val,y_val_pred))

run_name ='xgb_fraud_clf'
with mlflow.start_run(run_name=run_name) as run:
#     # get current run and experiment id
#     run_id = run.info.run_uuid
#     experiment_id = run.info.experiment_id

#     # train and predict
#     xgbclf=xgb.XGBClassifier(max_depth=8,
#                             learning_rate=0.01,
#                             use_label_encoder=False)
#     xgbclf.fit(x_train, y_train)

#     # Testing
#     y_pred = xgbclf.predict(x_test)
#     # confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#     # print(confusion_matrix)
#     # print(classification_report(y_test, y_pred))
#     # print(recall_score(y_test,y_pred))

#     # Log model and params using the MLflow sklearn APIs
#     mlflow.sklearn.log_model(xgbclf, "xgb-classifier")

#     precision = precision_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     roc = roc_auc_score(y_test, y_pred)

#     # confusion matrix values
#     tp = conf_matrix[0][0]
#     tn = conf_matrix[1][1]
#     fp = conf_matrix[0][1]
#     fn = conf_matrix[1][0]

#     # get classification metrics
#     class_report = classification_report(y_test, y_pred, output_dict=True)
#     recall_0 = class_report['0']['recall']
#     f1_score_0 = class_report['0']['f1-score']
#     recall_1 = class_report['1']['recall']
#     f1_score_1 = class_report['1']['f1-score']

#     # log metrics in mlflow
#     mlflow.log_metric("recall_0", recall_0)
#     mlflow.log_metric("f1_score_0", f1_score_0)
#     mlflow.log_metric("recall_1", recall_1)
#     mlflow.log_metric("f1_score_1", f1_score_1)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("true_positive", tp)
#     mlflow.log_metric("true_negative", tn)
#     mlflow.log_metric("false_positive", fp)
#     mlflow.log_metric("false_negative", fn)
#     mlflow.log_metric("roc", roc)

#     mlflow.log_params("max_depth",)















    # create confusion matrix plot
    # plt_cm, fig_cm, ax_cm = utils.plot_confusion_matrix(y_test, y_pred, y_test,
    #                                                     title="Classification Confusion Matrix")

    # temp_name = "confusion-matrix.png"
    # fig_cm.savefig(temp_name)
    # mlflow.log_artifact(temp_name, "confusion-matrix-plots")
    # try:
    #     os.remove(temp_name)
    # except FileNotFoundError as e:
    #     print(f"{temp_name} file is not found")

    # # create roc plot
    # plot_file = "roc-auc-plot.png"
    # probs = y_probs[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plt_roc, fig_roc, ax_roc = utils.create_roc_plot(fpr, tpr)
    # fig_roc.savefig(plot_file)
    # mlflow.log_artifact(plot_file, "roc-auc-plots")
    # try:
    #     os.remove(plot_file)
    # except FileNotFoundError as e:
    #     print(f"{temp_name} file is not found")

    # print("<->" * 40)
    # print("Inside MLflow Run with run_id {run_id} and experiment_id {experiment_id}")
    # print("max_depth of trees:", self.params["max_depth"])
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    # print("Accuracy Score =>", acc)
    # print("Precision      =>", precision)
    # print("ROC            =>", roc)

    # return experiment_id, run_id