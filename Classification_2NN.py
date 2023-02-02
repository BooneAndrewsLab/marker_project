import argparse
from Classification_2NN_lib import *

# Parameters to specify
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='',
                    help='Input files to be classified')
parser.add_argument('-t', '--training_set_file', default='',
                    help='Training data for the classifier')
parser.add_argument('-o', '--output_folder', default='',
                    help='Folder to save results')
parser.add_argument('-p', '--probability', default=0,
                    help='Minimum probability to make predictions for a cell')
parser.add_argument('-x', '--index', default=13,
                    help='Index to separate identifier and data features')
args = parser.parse_args()

# Neural network hyper-parameters
param = {'hidden_units': [54, 18],
         'percent_to_test': 0.2,
         'percent_to_valid': 0.2,
         'batch_size': 100,
         'k_fold_cv': 5,
         'learning_rate': 0.01,
         'decay': 1e-6,
         'momentum': 0.9,
         'nesterov': True,
         'num_epochs': 50,
         'runs': 10
         }

# Output filenames
output = {'TrainingCV': 'Training_performance_CV',
          'Training': 'Training_performance',
          'Confusion': 'Training_performance_confusion_matrix.png',
          'CellAccuracy': 'Single_cell_accuracies.csv',
          'Predictions': 'Single_cell_phenotype_predictions.csv'
          }


if __name__ == '__main__':
    # Output folder
    if os.path.exists(args.output_folder):
        os.system('rm -rf %s' % args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)
    os.chdir(args.output_folder)

    # Prepare phenotype data and train NN
    df_labeled_set, phenotypes = read_labeled_data(args.training_set_file)

    train_neural_network(df_labeled_set, param, args.index, output)

    # Make predictions
    if args.input_file:
        make_predictions(args.input_file, df_labeled_set, param, args.index, args.probability, output)
