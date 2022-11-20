print(__doc__)


import sys
import pandas as pd
from statistics import mean,pstdev
from utils.utils import flatten_images, preprocess, create_splits, train_and_save_model, get_best_model_path, model_test, print_data_frame, print_metrics


rescale_factor = 1
gamma_values = [0.01, 0.005, 0.001, 0.0005]
c_values = [0.1, 0.2, 0.5, 0.7]
hyp_comb = [{'gamma':g,"C":c}for g in gamma_values for c in c_values]
depth_value = [5,10,20,50,100]
test_size = [0.2,0.25,0.3,0.35,0.4]
validation_size_from_test_size = [0.5,0.6,0.4,0.7,0.75]
flattened_images, digits = flatten_images()
rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
svm_test_dataset_acc, decision_tree_test_dataset_acc, svm_test_dataset_f1, decision_tree_test_dataset_f1  = [], [], [], []
clf_name = sys.argv[1]


#
#random_state = int(sys.argv[2])
random_state = int(sys.argv[2])
file = open('results/{0}_{1}.txt'.format(clf_name,random_state),"w")
for i in range(5):
    print("Train set size:" ,1-(test_size[i]))
    print("Test set size :{0:.2f}".format(test_size[i]*(1 - validation_size_from_test_size[i])))
    print("validation set size : {0:.2f}".format(validation_size_from_test_size[i]*test_size[i]))
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size[i], validation_size_from_test_size[i],random_state) # 5 different splits
    #classifiers = {'SVM' : hyp_comb, 'DecisionTree' : depth_value} # two different classifiers
    if clf_name == 'SVM':
        classifiers = {clf_name : hyp_comb} 
    else:
        classifiers = {clf_name : depth_value}
    #for clf_name in classifiers:
    file.writelines("\n")
    df,path = train_and_save_model(clf_name, classifiers[clf_name], X_train, y_train, X_validation, y_validation) # training and saving the model
    file.writelines("model saved at:")
    file.writelines(str(path))
    model_path = get_best_model_path(df)  #path for the best model
    print_data_frame(clf_name, df)
    accuracy_test, f1_score_test = model_test(model_path, X_test, y_test) # metrics 
    if clf_name == 'SVM':
        #file.writelines("accuracy_test",accuracy_test)
        svm_test_dataset_acc.append(accuracy_test)
        svm_test_dataset_f1.append(f1_score_test)
    else:
        decision_tree_test_dataset_acc.append(accuracy_test)
        decision_tree_test_dataset_f1.append(f1_score_test)

    print_metrics(accuracy_test, f1_score_test)
    file.writelines("\n")
    file.writelines("test accuracy:")
    file.writelines(str(accuracy_test))
    file.writelines("\n")
    file.writelines("test macro-f1: ")
    file.writelines(str(f1_score_test))

if clf_name == 'SVM':
    svm_test_dataset_acc.append(mean(svm_test_dataset_acc))
    svm_test_dataset_f1.append(mean(svm_test_dataset_f1))
    svm_test_dataset_acc.append(pstdev(svm_test_dataset_acc[:-1]))
    svm_test_dataset_f1.append(pstdev(svm_test_dataset_f1[:-1]))
else:
    decision_tree_test_dataset_acc.append(mean(decision_tree_test_dataset_acc))
    decision_tree_test_dataset_f1.append(mean(decision_tree_test_dataset_f1))
    decision_tree_test_dataset_acc.append(pstdev(decision_tree_test_dataset_acc[:-1]))
    decision_tree_test_dataset_f1.append(pstdev(decision_tree_test_dataset_f1[:-1]))
file.close()
'''df = pd.DataFrame(list(zip(svm_test_dataset_acc, decision_tree_test_dataset_acc, svm_test_dataset_f1, decision_tree_test_dataset_f1)),columns =['SVM Accuracy', 'Decision Tree Accuracy', 'SVM f1 Score', 'Decision Tree f1 Score'], index =['1', '2', '3', '4', '5','Mean','Std'])
df.index.name = 'Run'

print(df)'''