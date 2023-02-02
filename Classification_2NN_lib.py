import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers, losses
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg')


def lower_column_names(df):
    """ Return pandas dataframe with lowered column names

        Args:
            df (pd.DataFrame): Pandas Dataframe object

        Returns:
            df (pd.DataFrame): Pandas Dataframe object with lowercase column names
        """

    df.columns = [c.lower() for c in df.columns]

    return df


def read_labeled_data(training_set_file):
    """ Read and prepare labeled set

        Args:
            training_set_file (str): File that contains training set labels and data

        Returns:
            df_labeled_set (pd.DataFrame): Labeled set as a dataframe
            classes (list): List of phenotype classes
        """

    # Read labeled set
    print('\nReading labeled set\n')
    df_labeled_set = lower_column_names(pd.read_csv(training_set_file))
    df_labeled_set = df_labeled_set.fillna('').reset_index(drop=True)

    # Add None as a class
    classes = np.append(df_labeled_set['phenotype'].unique(), ['none'])

    return df_labeled_set, classes


def split_labeled_set(df, features, k):
    """ Split labeled set into training and test set for k-fold cross-validation

        Args:
            df (pd.DataFrame): Labeled set
            features (list): Features to be analyzed
            k (int): Number of fold in cross-validation

        Returns:
            df (pd.DataFrame): Labeled set
            X (np.array): Labeled set input data
            y (np.array): Labeled set labels
            X_train (np.array): Training set input data
            X_test (np.array): Test set input data
            y_train (np.array): Training set labels
            y_test (np.array): Test set labels
            phenotypes (list): List of phenotype classes
        """

    # Shuffle labeled set
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df[features] = df[features].fillna(0)

    # Separate labeled set data and labels
    x = np.asarray(df[features])
    f = pd.factorize(df['phenotype'], sort=True)
    y = np.zeros((f[0].shape[0], len(set(f[0]))))
    y[np.arange(f[0].shape[0]), f[0].T] = 1
    phenotypes = f[1]

    # Split training and test set for k-fold cross-validation
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    divide = x.shape[0] // k
    for cv in range(k):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (k - 1):
            end = x.shape[0]
        mask_train = np.asarray([False if x in list(range(start, end)) else True for x in list(range(0, x.shape[0]))])
        x_train.append(x[mask_train == 1].copy())
        x_test.append(x[mask_train == 0].copy())
        y_train.append(y[mask_train == 1].copy())
        y_test.append(y[mask_train == 0].copy())

    return df, x, y, x_train, x_test, y_train, y_test, phenotypes


def train_neural_network(df, param, index, output):
    """ Train neural network (NN) with cross-validation

        Args:
            df (pd.DataFrame): Training set
            param (dict): NN hyper-parameters
            index (int): Index to separate identifier and data features
            output (dict): Output filenames
        """

    print('Training neural network\n')
    data_features = df.columns.values[index:]
    id_features = df.columns.values[:index]

    # Split training and test set for cross-validation
    df, x, y, x_train, x_test, y_train, y_test, phenotypes = split_labeled_set(df, data_features, param['k_fold_cv'])

    # Initialize arrays for NN runs
    df_output = df[id_features]
    sum_prob_labeled = np.zeros([y.shape[0], y.shape[1]])
    sum_prob_test = np.zeros([y.shape[0], y.shape[1]])

    # Train NN with cross validation for evaluating performance
    performance = pd.DataFrame()
    divide = x.shape[0] // param['k_fold_cv']
    run = 1
    for cv in range(param['k_fold_cv']):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (param['k_fold_cv'] - 1):
            end = x.shape[0]
        # Train and make predictions for each fold for a number of runs
        for n in range(param['runs']):
            runn = n + cv * param['runs']
            # Train NN with training set
            model, performance = neural_network(x_train[cv], y_train[cv], param, phenotypes, performance, runn,
                                                x_test[cv], y_test[cv])
            # Predictions on test data
            probabilities_test = model.predict(x_test[cv], batch_size=param['batch_size'])
            sum_prob_test[start:end] += probabilities_test

            # Predictions on labeled data
            probabilities_labeled = model.predict(x, batch_size=param['batch_size'])
            predictions_labeled = np.argmax(probabilities_labeled, axis=1)
            sum_prob_labeled += probabilities_labeled
            df_output['Run-%d' % run] = [phenotypes[i] for i in predictions_labeled]
            run += 1

    # Save training performance of cross-validation
    num_runs = param['k_fold_cv'] * param['runs']
    plot_training_performance(performance, output['TrainingCV'], num_runs)

    # Test-set predictions
    y_pred = np.argmax(sum_prob_test, axis=1)
    y_true = np.argmax(y, axis=1)
    plot_confusion_matrix(y_true, y_pred, phenotypes, output['Confusion'])

    # Training set single cell accuracies
    cell_accuracy(df_output, sum_prob_labeled, phenotypes, num_runs, output)

    # Train NN with the complete training set once
    for n in range(param['runs']):
        model, performance = neural_network(x, y, param, phenotypes, pd.DataFrame(), 0)
        plot_training_performance(performance, '%s_%d' % (output['Training'], n), 1)
        model.save('2NN_model_%d.h5' % n)


def neural_network(x_train, y_train, param, phenotypes, performance, n, x_test=np.array([]), y_test=np.array([])):
    """ Train NN and return the model

        Args:
            x_train (np.array): Training set input data
            y_train (np.array): Training set labels
            param (dict): Neural network hyper-parameters
            phenotypes (list): List of phenotype classes
            performance (pd.DataFrame): Cross-entropy and accuracy at each training
            n (int): The specific run out of the total random initializations
            x_test (np.array): Test set input data
            y_test (np.array): Test set labels

        Returns:
            model (keras.models.Sequential): Trained neural network
        """

    # NN layer units
    input_units = x_train.shape[1]
    output_units = len(phenotypes)

    # NN architecture
    model = Sequential()
    model.add(Dense(param['hidden_units'][0],
                    input_shape=(input_units,),
                    activation='relu'))
    model.add(Dense(param['hidden_units'][1],
                    activation='relu'))
    model.add(Dense(output_units,
                    activation='softmax'))
    sgd = optimizers.SGD(lr=param['learning_rate'],
                         decay=param['decay'],
                         momentum=param['momentum'],
                         nesterov=param['nesterov'])
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])
    hist = model.fit(x_train, y_train,
                     epochs=param['num_epochs'],
                     batch_size=param['batch_size'],
                     validation_split=param['percent_to_valid'],
                     verbose=1)

    # Evaluate model
    performance['Loss_%d' % n] = hist.history['loss']
    performance['Val_Loss_%d' % n] = hist.history['val_loss']
    performance['Accuracy_%d' % n] = hist.history['accuracy']
    performance['Val_Accuracy_%d' % n] = hist.history['val_accuracy']

    if len(x_test):
        score = model.evaluate(x_test, y_test, batch_size=param['batch_size'])
        print('Test %s: %.2f' % (model.metrics_names[0], score[0]))
        print('Test %s: %.2f%%\n' % (model.metrics_names[1], score[1] * 100))
    else:
        print('Trained on all labeled samples\n')

    return model, performance


def plot_training_performance(performance, output, num_runs):
    """ Plot the cross-entropy and accuracy for training and validation set

        Args:
            performance (pd.DataFrame): Cross-entropy and accuracy at each training
            output (str): Output filenames
            num_runs (int): Total number of runs (number of random initializations times the number of folds)
        """

    fontsize = 16
    plt.figure(figsize=(10, 10))

    # Loss
    plt.subplot(211)
    train_all = []
    valid_all = []

    for i in range(num_runs):
        train = performance.iloc[:, performance.columns.get_loc('Loss_%d' % i)].values
        valid = performance.iloc[:, performance.columns.get_loc('Val_Loss_%d' % i)].values
        train_all.append(train)
        valid_all.append(valid)
        plt.plot(train, 'lightblue', alpha=0.4)
        plt.plot(valid, 'lightgreen', alpha=0.4)

    plt.plot(np.mean(train_all, axis=0), 'blue', label='Training')
    plt.plot(np.mean(valid_all, axis=0), 'green', label='Validation')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 4)
    plt.yticks(fontsize=fontsize - 4)
    fig = plt.gcf()
    plt.legend(fontsize=fontsize, loc='upper right')

    # Acc
    plt.subplot(212)
    train_all = []
    valid_all = []
    for i in range(num_runs):
        train = performance.iloc[:, performance.columns.get_loc('Accuracy_%d' % i)].values
        valid = performance.iloc[:, performance.columns.get_loc('Val_Accuracy_%d' % i)].values
        train_all.append(train)
        valid_all.append(valid)
        plt.plot(train, 'lightblue', alpha=0.4)
        plt.plot(valid, 'lightgreen', alpha=0.4)

    plt.plot(np.mean(train_all, axis=0), 'blue', label='Training')
    plt.plot(np.mean(valid_all, axis=0), 'green', label='Validation')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 4)
    plt.yticks(fontsize=fontsize - 4)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=fontsize, loc='lower right')

    fig.savefig('%s.png' % output, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def cell_accuracy(df, sum_prob, phenotypes, n, output):
    """ Calculate accuracy for labeled set samples out of n runs.
        Include average probability for each phenotype.
        Save in an output file.
        Args:
            df (pd.DataFrame): Labeled set in a dataframe
            sum_prob (np.array): Cumulative probability for each sample and label
            phenotypes (list): List of phenotype classes
            n (int): Independent neural network training runs
            output (dict): Output filenames
        """

    # Create columns for each cross-validation run
    df['accuracy'] = np.zeros(len(df))
    predictions = []

    for c in df.columns.values:
        if 'Run-' in c:
            predictions.append(df.columns.get_loc(c))

    # Calculate cell accuracy
    for i in range(len(df)):
        true_label = df.iloc[i, df.columns.get_loc('phenotype')]
        correct = 0
        for p in predictions:
            if true_label == df.iloc[i, p]:
                correct += 1
        df.iloc[i, df.columns.get_loc('accuracy')] = float(correct) / n

    # Calculate average probability for each phenotype class
    sum_prob = sum_prob / n
    for i in range(len(phenotypes)):
        df[phenotypes[i]] = sum_prob[:, i]

    # Save cell accuracy data
    df = df.sort_values('cell_id', ascending=True).reset_index(drop=True)
    df.to_csv(path_or_buf=output['CellAccuracy'], index=False)


def plot_confusion_matrix(y_true, y_pred, classes, output):
    """ Plot confusion matrix for all test set predictions.
        Args:
            y_true (np.array): Actual labels
            y_pred (np.array): Predicted labels
            classes (list): List of phenotype labels
            output (dict): Output filenames
        """

    # Normalize counts for each true-predicted label pair
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Acc %.2f%%' % (acc * 100))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    # Plot percentage of labeled samples in each true-predicted label pair
    thresh = np.max(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    # Save plot
    fig = plt.gcf()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(output, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def make_predictions(input_file, df, param, index, threshold, output):
    """ Train neural network (NN) with cross-validation
        Make predictions and save phenotype predictions

        Args:
            input_file (str): File to make predictions on
            df (pd.DataFrame): Labeled set
            param (dict): NN hyper-parameters
            index (int): Index to separate identifier and data features
            threshold (float): Probability threshold to make predictions
            output (dict): Output filenames
        """

    print('Training neural network for prediction\n')
    data_features = df.columns.values[index:]

    # Split training and test set for cross-validation
    df, x, y, _, _, _, _, phenotypes = split_labeled_set(df, data_features, param['k_fold_cv'])

    # Train NN with the complete labeled set
    df_predict = lower_column_names(pd.read_csv(input_file))
    performance = pd.DataFrame()
    sum_prob_all = np.zeros([df_predict[data_features].shape[0], y.shape[1]])
    for n in range(param['runs']):
        model, performance = neural_network(x, y, param, phenotypes, performance, n)
        # Predictions on all data
        probabilities_all = model.predict(df_predict[data_features].values, batch_size=param['batch_size'])
        sum_prob_all += probabilities_all
    plot_training_performance(performance, output['Training'], param['runs'])

    # Make predictions for the complete data
    y_all = sum_prob_all / param['runs']
    y_prob_all = (y_all >= threshold).astype('int')
    y_pred_all = np.argmax(y_all, axis=1)
    phenotype_all = []
    for i in range(len(y_pred_all)):
        pred = phenotypes[y_pred_all[i]]
        # If none of the probabilities pass the threshold, predict as None phenotype
        if sum(y_prob_all[i]) == 0:
            pred = 'none'
        phenotype_all.append(pred)

    # Save phenotype predictions for cell IDs provided
    id_features_pred = [c for c in df_predict.columns.values if c not in data_features]
    df_pred_results = df_predict[id_features_pred]
    df_pred_results['Prediction'] = np.array(phenotype_all)
    for i in range(len(phenotypes)):
        df_pred_results[phenotypes[i]] = y_all[:, i]
    df_pred_results = df_pred_results.sort_values('cell_id', ascending=True).reset_index(drop=True)
    df_pred_results.to_csv(path_or_buf=output['Predictions'], index=False)
