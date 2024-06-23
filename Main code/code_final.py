 
from glob import glob  # For file path expansion
import os  # For interacting with the operating system
import numpy as np  # For numerical computing
import pandas  # For data manipulation and analysis
import matplotlib as plt  # For plotting
import mne  # For processing EEG data
from sklearn.linear_model import LogisticRegression  # For logistic regression classification
from sklearn.pipeline import Pipeline  # For constructing processing pipelines
from sklearn.preprocessing import StandardScaler  # For feature standardization
from sklearn.model_selection import GroupKFold, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.naive_bayes import GaussianNB  # For Gaussian Naive Bayes classification
from sklearn.ensemble import RandomForestClassifier  # For random forest classification
from sklearn.svm import SVC  # For support vector classification
from sklearn.ensemble import GradientBoostingClassifier  # For gradient boosting classification
from sklearn.neural_network import MLPClassifier  # For multi-layer perceptron classification
import random  # For random number generation and sampling
from scipy import stats  # For statistical functions
import matplotlib.pyplot as plt  # For plotting (duplicate import, same as above)
import seaborn as sns # For visualization
import shap # For explainability
import pandas as pd

base_dir = r"E:\Data Sorted\Music"

directories = os.listdir(base_dir)
directories = sorted(directories, key=lambda x: int(x))

# Enumerate and print the sorted directories
for count, directory in enumerate(directories):
        
    # Find all file paths matching the specified pattern
    music_path = glob(os.path.join(r"E:\Data Sorted\Music", directory, "*"))

    path = music_path
    save_path = count + 1

    # Filter file paths to separate positive and negative groups
    pos_file_path = sorted([i for i in path if 'pos' in os.path.basename(i)])
    neg_file_path = sorted([i for i in path if 'neg' in os.path.basename(i)])

    # Print the number of files in each category
    print("Number of pos files:", len(pos_file_path))
    print("Number of neg files:", len(neg_file_path))

    # Store initial sample sizes before undersampling
    pre_undersampling = [len(pos_file_path), len(neg_file_path)]

    # Determine the smaller class size between positive and negative groups
    min_class_size = min(len(pos_file_path), len(neg_file_path))

    # Undersample the larger class to balance the dataset
    random.seed(42)  # For reproducibility
    if len(pos_file_path) > min_class_size:
        pos_file_path = random.sample(pos_file_path, min_class_size)
    elif len(neg_file_path) > min_class_size:
        neg_file_path = random.sample(neg_file_path, min_class_size)

    # Print the new number of files in each category after undersampling
    print("Number of pos files after undersampling:", len(pos_file_path))
    print("Number of neg files after undersampling:", len(neg_file_path))

    # Plotting the sample size pre and post-undersampling
    post_undersampling = [len(pos_file_path), len(neg_file_path)]
    categories = ['Positive', 'Negative']

    plt.figure(figsize=(8, 6))
    plt.bar(categories, pre_undersampling, color='blue', alpha=0.5, label='Pre Undersampling')
    plt.bar(categories, post_undersampling, color='red', alpha=0.5, label='Post Undersampling')
    plt.xlabel('Category')
    plt.ylabel('Sample Size')
    plt.title('Sample Size Pre and Post Undersampling')
    plt.legend()
    for i, value in enumerate(pre_undersampling):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)
    for i, value in enumerate(post_undersampling):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)
    plt.show()

     
    # Reference Electrodes for Music Tasks
    selected_electrodes_music = [
        'Fp1', 'F7', 'Fp2', 'F8', 'F4', 'AF3', 'AF4', 'AF7', 'AF8', 
        'F3', 'F5', 'F6', 'FC1', 'FC2', 'FC5', 'FC6', 'FT7', 'FT8', 
        'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 
        'CP5', 'CP6', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 
        'PO7', 'PO8', 'O1', 'O2'
    ]

    # Reading data from EEG files
    def read_data(file_path, electrodes):
        # Load EEG data from file
        data = mne.io.read_raw_eeglab(file_path, preload=True)
        # Set EEG reference channels
        data.set_eeg_reference(ref_channels=["Fz"])
        # Apply bandpass filter to EEG data
        data.filter(l_freq=4, h_freq=45)
        # Select specific EEG channels
        data.pick_channels(electrodes)
        # Create fixed-length epochs from EEG data
        epochs = mne.make_fixed_length_epochs(data, duration=15, overlap=1)
        # Extract data from epochs
        array = epochs.get_data()
        return array


     
    #%%capture
    # Suppress output capture for this cell
    # Read EEG data from the first positive file path and output set of electrodes on a dummy head
    head_pre = mne.io.read_raw_eeglab(pos_file_path[0], preload=True)
    head_pre.pick_channels(selected_electrodes_music)

     
    # Create a standard montage for the "easycap-M1" electrode configuration
    easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    # Set the montage to the EEG data
    head_pre.set_montage(easycap_montage)
    # Plot the sensors with their names shown
    fig = head_pre.plot_sensors(show_names=True)

     
    #%%capture
    # Read EEG data for positive class from all positive file paths
    pos_epochs_array = [read_data(i, selected_electrodes_music) for i in pos_file_path]
    # Read EEG data for negative class from all negative file paths
    neg_epochs_array = [read_data(i, selected_electrodes_music) for i in neg_file_path]

     
    # Create labels for positive epochs (0 for positive class)
    pos_epoch_labels = [len(i) * [0] for i in pos_epochs_array]
    # Create labels for negative epochs (1 for negative class)
    neg_epoch_labels = [len(i) * [1] for i in neg_epochs_array]
    # Check the length of positive and negative epoch labels
    len(pos_epoch_labels), len(neg_epoch_labels)


     
    # Combine positive and negative epoch arrays into a single list
    data_list = pos_epochs_array + neg_epochs_array
    # Combine positive and negative epoch labels into a single list
    label_list = pos_epoch_labels + neg_epoch_labels

     
    # Create a list of groups based on the number of epochs in each class
    group_list = [[i] * len(j) for i, j in enumerate(data_list)]


     
    # Stack the data list vertically to create a 2D array
    data_array = np.vstack(data_list)

    # Stack the label list horizontally to create a 1D array
    label_array = np.hstack(label_list)

    # Stack the group list horizontally to create a 1D array
    group_array = np.hstack(group_list)

     
    def mean(x):
        # Calculate the mean along the last axis of the input array.
        return np.mean(x, axis=-1)

    def std(x):
        # Calculate the standard deviation along the last axis of the input array.
        return np.std(x, axis=-1)

    def ptp(x):
        # Calculate the peak-to-peak (range) along the last axis of the input array.
        return np.ptp(x, axis=-1)

    def var(x):
        # Calculate the variance along the last axis of the input array.
        return np.var(x, axis=-1)

    def minim(x):
        # Find the minimum value along the last axis of the input array.
        return np.min(x, axis=-1)

    def maxim(x):
        # Find the maximum value along the last axis of the input array.
        return np.max(x, axis=-1)

    def argminim(x):
        # Find the index of the minimum value along the last axis of the input array.
        return np.argmin(x, axis=-1)

    def argmaxim(x):
        # Find the index of the maximum value along the last axis of the input array.
        return np.argmax(x, axis=-1)

    def rms(x):
        # Calculate the root mean square along the last axis of the input array.
        return np.sqrt(np.mean(x**2, axis=-1))

    def abs_diff_signal(x):
        # Calculate the sum of absolute differences between consecutive elements along the last axis of the input array.
        return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)

    def skewness(x):
        # Calculate the skewness along the last axis of the input array.
        return stats.skew(x, axis=-1)

    def kurtosis(x):
        # Calculate the kurtosis along the last axis of the input array.
        return stats.kurtosis(x, axis=-1)

    def concatenate_features(x):
        """
        Concatenate statistical features computed from the input array along the last axis.

        Parameters:
            x (ndarray): Input array containing signals.

        Returns:
            ndarray: Concatenated array of statistical features.
        """
        return np.concatenate((mean(x), std(x), ptp(x), var(x), minim(x), maxim(x), argminim(x), argmaxim(x), rms(x), abs_diff_signal(x), skewness(x), kurtosis(x)), axis=-1)


     
    features = []

    # Compute statistical features for each epoch in the data array and append to the features list
    for d in data_array:
        features.append(concatenate_features(d))

    # Convert the features list to a numpy array
    features_array = np.array(features)

     
    gkf = GroupKFold(5)

    # Random Forest Classifier: Ensemble learning method for classification that operates by constructing a multitude of decision trees.
    # The pipeline includes standard scaling of features, and grid search for tuning hyperparameters (number of estimators and maximum depth).
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_rf = RandomForestClassifier()
    pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', clf_rf)])
    param_grid_rf = {'clf__n_estimators': [50, 100, 200],  
                    'clf__max_depth': [None, 10, 20]}
    gscv_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=gkf, n_jobs=-1)
    gscv_rf.fit(features_array, label_array, groups=group_array)
    best_score_rf = gscv_rf.best_score_

    # Logistic Regression Classifier: Linear model for binary classification with adjustable regularization.
    # The pipeline includes standard scaling of features, and grid search for tuning regularization strength (C).
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_lr = LogisticRegression(max_iter=1000)
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', clf_lr)])
    param_grid_lr = {'clf__C': [0.1, 0.5, 0.7, 1, 3, 5, 7]}
    gscv_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=gkf)
    gscv_lr.fit(features_array, label_array, groups=group_array)
    best_score_lr = gscv_lr.best_score_

    # Naive Bayes (GaussianNB) Classifier: Simple probabilistic classifier based on Bayes' theorem with strong (naive) independence assumptions.
    # The pipeline includes standard scaling of features. No hyperparameters are tuned.
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_nb = GaussianNB()
    pipe_nb = Pipeline([('scaler', StandardScaler()), ('clf', clf_nb)])
    gscv_nb = GridSearchCV(pipe_nb, param_grid={}, cv=gkf)
    gscv_nb.fit(features_array, label_array, groups=group_array)
    best_score_nb = gscv_nb.best_score_

    # Support Vector Machines (SVM) Classifier: Effective for high-dimensional spaces, especially when the number of features exceeds the number of samples.
    # The pipeline includes standard scaling of features, and grid search for tuning regularization strength (C) and kernel type.
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_svm = SVC()
    pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', clf_svm)])
    param_grid_svm = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    gscv_svm = GridSearchCV(pipe_svm, param_grid=param_grid_svm, cv=gkf, n_jobs=-1)
    gscv_svm.fit(features_array, label_array, groups=group_array)
    best_score_svm = gscv_svm.best_score_

    # Gradient Boosting Machines (GBM) Classifier: Builds an ensemble of weak learners (typically decision trees), sequentially improving performance.
    # The pipeline includes standard scaling of features, and grid search for tuning learning rate, maximum depth, and number of estimators.
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_gbm = GradientBoostingClassifier(random_state=42)
    pipe_gbm = Pipeline([('scaler', StandardScaler()), ('clf', clf_gbm)])
    param_grid_gbm = {'clf__learning_rate': [0.01, 0.1], 'clf__max_depth': [3, 5], 'clf__n_estimators': [50, 100]}
    gscv_gbm = GridSearchCV(pipe_gbm, param_grid=param_grid_gbm, cv=gkf, n_jobs=-1)
    gscv_gbm.fit(features_array, label_array, groups=group_array)
    best_score_gbm = gscv_gbm.best_score_

    # Neural Networks Classifier: Multilayer perceptron with customizable architecture and activation functions.
    # The pipeline includes standard scaling of features, and grid search for tuning hidden layer sizes and activation functions.
    # Cross-validation is performed using GroupKFold to ensure group-wise data splitting.
    clf_nn = MLPClassifier(random_state=42)
    pipe_nn = Pipeline([('scaler', StandardScaler()), ('clf', clf_nn)])
    param_grid_nn = {'clf__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'clf__activation': ['relu', 'tanh']}
    gscv_nn = GridSearchCV(pipe_nn, param_grid=param_grid_nn, cv=gkf, n_jobs=-1)
    gscv_nn.fit(features_array, label_array, groups=group_array)
    best_score_nn = gscv_nn.best_score_

     
    print("Best Score for Logistic Regression:", best_score_lr)
    print("Best Score for Random Forest:", best_score_rf)
    print("Best Score for Naive Bayes:", best_score_nb)
    print("Best Score for Support Vector Machines:", best_score_svm)
    print("Best Score for Gradient Boosting Machines:", best_score_gbm)
    print("Best Score for Neural Networks:", best_score_nn)

    # Define the model names and their corresponding best scores
    model_names = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 
                'Support Vector Machines', 'Gradient Boosting Machines', 'Neural Networks']
    best_scores = [best_score_lr, best_score_rf, best_score_nb, 
                best_score_svm, best_score_gbm, best_score_nn]

    # Sort the model names and best scores
    sorted_data = sorted(zip(model_names, best_scores), key=lambda x: x[1])

    # Unzip the sorted data
    sorted_model_names, sorted_best_scores = zip(*sorted_data)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_model_names, sorted_best_scores, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Best Model Scores')
    plt.xlim(0, 1)  # Setting limit from 0 to 1 for accuracy
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top

    # Annotate each bar with its specific score
    for bar, score in zip(bars, sorted_best_scores):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', ha='left', fontsize=10, color='black')

    plt.show()

     
    # List of statistical features
    statistical_features = ['mean', 'std', 'ptp', 'var', 'min', 'max', 'argmin', 'argmax', 'rms', 'abs_diff_signal', 'skewness', 'kurtosis']

    # Generate detailed feature names
    detailed_feature_names = []
    for electrode in selected_electrodes_music:
        for feature in statistical_features:
            detailed_feature_names.append(f"{electrode}_{feature}")

    # Check the length of detailed_feature_names
    print("Length of detailed_feature_names:", len(detailed_feature_names))

     
    # Define the feature names based on electrodes and feature functions
    feature_functions = ['mean', 'std', 'ptp', 'var', 'min', 'max', 'argmin', 'argmax', 'rms', 'abs_diff_signal', 'skewness', 'kurtosis']
    electrode_feature_names = []
    for electrode in selected_electrodes_music:
        for feature in feature_functions:
            electrode_feature_names.append(f"{electrode}_{feature}")

    # Ensure the number of features matches
    if len(electrode_feature_names) != features_array.shape[1]:
        raise ValueError("The number of electrode feature names does not match the number of features in the data")

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(gscv_rf.best_estimator_.named_steps['clf'])
    shap_values = explainer.shap_values(features_array)

    # Convert SHAP values to a DataFrame for easier manipulation
    shap_values_df = pd.DataFrame(shap_values[1], columns=electrode_feature_names)

    # Aggregate SHAP values by summing all features for each electrode
    electrode_shap_values = {}
    for electrode in selected_electrodes_music:
        electrode_shap_values[electrode] = shap_values_df.filter(like=electrode).sum(axis=1).mean()

    # Convert to DataFrame for visualization
    electrode_shap_values_df = pd.DataFrame(electrode_shap_values.items(), columns=['Electrode', 'SHAP Value'])

    # Sort values for better visualization
    electrode_shap_values_df = electrode_shap_values_df.sort_values(by='SHAP Value', ascending=False)

     
    # Sort and select the top N channels
    top_n = 20
    top_electrode_shap_values_df = electrode_shap_values_df.head(top_n)

    # Plot with Seaborn for better aesthetics
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='SHAP Value', 
        y='Electrode', 
        data=top_electrode_shap_values_df,
        palette='viridis'
    )
    plt.xlabel('Mean SHAP Value')
    plt.title(f'Top {top_n} Electrode Importance (Aggregated SHAP Values)')
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.show()

    # Get the list of top 20 electrodes
    top_20_electrodes = top_electrode_shap_values_df['Electrode'].tolist()
    print("Top 20 Electrodes are: ", top_20_electrodes)

     
    # Define the feature names based on electrodes and feature functions
    feature_functions = ['mean', 'std', 'ptp', 'var', 'min', 'max', 'argmin', 'argmax', 'rms', 'abs_diff_signal', 'skewness', 'kurtosis']
    electrode_feature_names = []

    # Iterate through each electrode and feature function to generate detailed feature names
    for electrode in selected_electrodes_music:
        for feature in feature_functions:
            electrode_feature_names.append(f"{electrode}_{feature}")

    # Ensure the number of features matches the length of the electrode feature names
    if len(electrode_feature_names) != features_array.shape[1]:
        raise ValueError("The number of electrode feature names does not match the number of features in the data")

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(gscv_rf.best_estimator_.named_steps['clf'])
    shap_values = explainer.shap_values(features_array)

    # Extract SHAP values for class 1 (assuming binary classification)
    shap_values_class1 = shap_values[1]

    # Convert SHAP values to a DataFrame for easier manipulation
    shap_values_df = pd.DataFrame(shap_values_class1, columns=electrode_feature_names)

    # Aggregate SHAP values by summing all features for each electrode
    electrode_shap_values = {}
    for electrode in selected_electrodes_music:
        electrode_shap_values[electrode] = shap_values_df.filter(like=electrode).sum(axis=1).mean()

    # Convert to DataFrame for visualization
    electrode_shap_values_df = pd.DataFrame(electrode_shap_values.items(), columns=['Electrode', 'SHAP Value'])

    # Sort values for better visualization
    electrode_shap_values_df = electrode_shap_values_df.sort_values(by='SHAP Value', ascending=False)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.barh(electrode_shap_values_df['Electrode'], electrode_shap_values_df['SHAP Value'])
    plt.xlabel('Mean SHAP Value')
    plt.title('Electrode Importance (Aggregated SHAP Values)')
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.show()

     
    #%%capture
    # Capture cell output to suppress any displayed output
    top_20_electrodes_df = electrode_shap_values_df.head(top_n)

    # Read EEG data for the top 20 electrodes from positive and negative files
    top_20_pos_epochs_array = [read_data(i, top_20_electrodes) for i in pos_file_path]
    top_20_neg_epochs_array = [read_data(i, top_20_electrodes) for i in neg_file_path]

    # Read EEG data from a positive file for the top 20 electrodes
    head_post = mne.io.read_raw_eeglab(pos_file_path[0], preload=True)
    head_post.pick_channels(top_20_electrodes)

     
    # Define the montage using the "easycap-M1" standard
    easycap_montage = mne.channels.make_standard_montage("easycap-M1")

    # Set the montage for the EEG data
    head_post.set_montage(easycap_montage)

    # Plot the sensors with names
    fig = head_post.plot_sensors(show_names=True)

     
    # Define labels for positive and negative epochs based on the top 20 electrodes
    top_20_pos_epoch_labels = [len(i) * [0] for i in top_20_pos_epochs_array]
    top_20_neg_epoch_labels = [len(i) * [1] for i in top_20_neg_epochs_array]

    # Combine the labels for positive and negative epochs
    top_20_label_list = top_20_pos_epoch_labels + top_20_neg_epoch_labels

    # Create a list to group data based on epochs
    top_20_group_list = [[i] * len(j) for i, j in enumerate(top_20_pos_epochs_array + top_20_neg_epochs_array)]

    # Convert data and labels into arrays
    top_20_data_array = np.vstack(top_20_pos_epochs_array + top_20_neg_epochs_array)
    top_20_label_array = np.hstack(top_20_label_list)
    top_20_group_array = np.hstack(top_20_group_list)

    # Extract statistical features from the data
    top_20_features = []
    for d in top_20_data_array:
        top_20_features.append(concatenate_features(d))

    # Convert the features into an array
    top_20_features_array = np.array(features)

    # Define a GroupKFold cross-validation iterator
    top_20_gkf = GroupKFold(5)

    # Define the Random Forest classifier
    top_20_clf_rf = RandomForestClassifier()
    top_20_pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_rf)])
    top_20_param_grid_rf = {'clf__n_estimators': [50, 100, 200],  
                            'clf__max_depth': [None, 10, 20]}
    top_20_gscv_rf = GridSearchCV(top_20_pipe_rf, param_grid=top_20_param_grid_rf, cv=top_20_gkf, n_jobs=-1)
    top_20_gscv_rf.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_rf = top_20_gscv_rf.best_score_

    # Define the Logistic Regression classifier
    top_20_clf_lr = LogisticRegression(max_iter=1000)  # Example max_iter value
    top_20_pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_lr)])
    top_20_param_grid_lr = {'clf__C': [0.1, 0.5, 0.7, 1, 3, 5, 7]}  # Example values for regularization strength
    top_20_gscv_lr = GridSearchCV(top_20_pipe_lr, param_grid=top_20_param_grid_lr, cv=top_20_gkf)
    top_20_gscv_lr.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_lr = top_20_gscv_lr.best_score_

    # Naive Bayes (GaussianNB)
    top_20_clf_nb = GaussianNB()
    top_20_pipe_nb = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_nb)])
    top_20_gscv_nb = GridSearchCV(top_20_pipe_nb, param_grid={}, cv=top_20_gkf)
    top_20_gscv_nb.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_nb = top_20_gscv_nb.best_score_

    # Support Vector Machines (SVM)
    top_20_clf_svm = SVC()
    top_20_pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_svm)])
    top_20_param_grid_svm = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    top_20_gscv_svm = GridSearchCV(top_20_pipe_svm, param_grid=top_20_param_grid_svm, cv=top_20_gkf, n_jobs=-1)
    top_20_gscv_svm.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_svm = top_20_gscv_svm.best_score_

    # Gradient Boosting Machines (GBM)
    top_20_clf_gbm = GradientBoostingClassifier(random_state=42)
    top_20_pipe_gbm = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_gbm)])
    top_20_param_grid_gbm = {'clf__learning_rate': [0.01, 0.1], 'clf__max_depth': [3, 5], 'clf__n_estimators': [50, 100]}
    top_20_gscv_gbm = GridSearchCV(top_20_pipe_gbm, param_grid=top_20_param_grid_gbm, cv=top_20_gkf, n_jobs=-1)
    top_20_gscv_gbm.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_gbm = top_20_gscv_gbm.best_score_

    # Neural Networks
    top_20_clf_nn = MLPClassifier(random_state=42)
    top_20_pipe_nn = Pipeline([('scaler', StandardScaler()), ('clf', top_20_clf_nn)])
    top_20_param_grid_nn = {'clf__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'clf__activation': ['relu', 'tanh']}
    top_20_gscv_nn = GridSearchCV(top_20_pipe_nn, param_grid=top_20_param_grid_nn, cv=top_20_gkf, n_jobs=-1)
    top_20_gscv_nn.fit(top_20_features_array, top_20_label_array, groups=top_20_group_array)
    top_20_best_score_nn = top_20_gscv_nn.best_score_

     
    # Print the best scores for each model
    print("Best Score for Logistic Regression:", top_20_best_score_lr)
    print("Best Score for Random Forest:", top_20_best_score_rf)
    print("Best Score for Naive Bayes:", top_20_best_score_nb)
    print("Best Score for Support Vector Machines:", top_20_best_score_svm)
    print("Best Score for Gradient Boosting Machines:", top_20_best_score_gbm)
    print("Best Score for Neural Networks:", top_20_best_score_nn)

    # Define the model names and their corresponding best scores
    top_20_model_names = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 
                        'Support Vector Machines', 'Gradient Boosting Machines', 'Neural Networks']
    top_20_best_scores = [top_20_best_score_lr, top_20_best_score_rf, top_20_best_score_nb, 
                        top_20_best_score_svm, top_20_best_score_gbm, top_20_best_score_nn]

    # Sort the model names and best scores
    top_20_sorted_data = sorted(zip(top_20_model_names, top_20_best_scores), key=lambda x: x[1])

    # Unzip the sorted data
    top_20_sorted_model_names, top_20_sorted_best_scores = zip(*top_20_sorted_data)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    top_20_bars = plt.barh(top_20_sorted_model_names, top_20_sorted_best_scores, color='green')
    plt.xlabel('Accuracy')
    plt.title('Best Model Scores based on the top scoring 20 electrodes')
    plt.xlim(0, 1)  # Setting limit from 0 to 1 for accuracy
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top

    # Annotate each bar with its specific score
    for bar, score in zip(top_20_bars, top_20_sorted_best_scores):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', ha='left', fontsize=10, color='black')

    plt.show()

     
    def save_results_to_file(results_filename, plot_directory):
            # Create directory if it doesn't exist
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        # Save results to a text file
        with open(results_filename, 'w') as file:
            file.write("Best Scores for Each Model:\n")
            file.write(f"Logistic Regression: {best_score_lr}\n")
            file.write(f"Random Forest: {best_score_rf}\n")
            file.write(f"Naive Bayes: {best_score_nb}\n")
            file.write(f"Support Vector Machines: {best_score_svm}\n")
            file.write(f"Gradient Boosting Machines: {best_score_gbm}\n")
            file.write(f"Neural Networks: {best_score_nn}\n")
            file.write("\nElectrodes:\n")
            file.write(', '.join(selected_electrodes_music))

            file.write("\nTop 20 Electrodes:\n")
            file.write(', '.join(top_20_electrodes))

            file.write("\n\nBest Scores for Each Model (Top 20 Electrodes):\n")
            file.write(f"Logistic Regression: {top_20_best_score_lr}\n")
            file.write(f"Random Forest: {top_20_best_score_rf}\n")
            file.write(f"Naive Bayes: {top_20_best_score_nb}\n")
            file.write(f"Support Vector Machines: {top_20_best_score_svm}\n")
            file.write(f"Gradient Boosting Machines: {top_20_best_score_gbm}\n")
            file.write(f"Neural Networks: {top_20_best_score_nn}\n")

        # Save plots and graphs to the directory
        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_model_names, sorted_best_scores, color='skyblue')
        plt.xlabel('Accuracy')
        plt.title('Best Model Scores')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_directory, 'best_model_scores.png'))
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.barplot(x='SHAP Value', y='Electrode', data=top_electrode_shap_values_df, palette='viridis')
        plt.xlabel('Mean SHAP Value')
        plt.title(f'Top {top_n} Electrode Importance (Aggregated SHAP Values)')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_directory, 'top_electrode_importance.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        top_20_bars = plt.barh(top_20_sorted_model_names, top_20_sorted_best_scores, color='green')
        plt.xlabel('Accuracy')
        plt.title('Best Model Scores based on the top scoring 20 electrodes')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_directory, 'best_model_scores_top_20_electrodes.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        categories = ['Positive', 'Negative']
        plt.bar(categories, pre_undersampling, color='blue', alpha=0.5, label='Pre Undersampling')
        plt.bar(categories, post_undersampling, color='red', alpha=0.5, label='Post Undersampling')
        plt.xlabel('Category')
        plt.ylabel('Sample Size')
        plt.title('Sample Size Pre and Post Undersampling')
        plt.legend()
        plt.savefig(os.path.join(plot_directory, 'sample_size_pre_post_undersampling.png'))
        plt.close()

        fig = head_pre.plot_sensors(show_names=True)
        fig.savefig(os.path.join(plot_directory, 'sensors_with_names_before.png'))
        plt.close(fig)

        fig = head_post.plot_sensors(show_names=True)
        fig.savefig(os.path.join(plot_directory, 'sensors_with_names_top_20.png'))
        plt.close(fig)

        # Plot electrode importance
        plt.figure(figsize=(10, 6))
        plt.barh(electrode_shap_values_df['Electrode'], electrode_shap_values_df['SHAP Value'])
        plt.xlabel('Mean SHAP Value')
        plt.title('Electrode Importance (Aggregated SHAP Values)')
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.savefig(os.path.join(plot_directory, 'electrode_importance.png'))
        plt.close()

        
    # Call the function to save results
    results = "results {}.txt".format(str(save_path))
    plots = "plots {}/".format(str(save_path))

    save_results_to_file(results,plots)