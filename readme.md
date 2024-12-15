# NLP Project

This repository contains code and data for a Natural Language Processing (NLP) project. The project includes Jupyter notebooks, python files configuration files, and various datasets for training and evaluation. The project addresses task 3 subtask 2.


### Prerequisites

- Python 3.x
- Jupyter Notebook
- VS Code (optional)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/nlp-project.git
    cd nlp-project
    ```

2. Install the required Python packages:
    ```sh
    pip install -r code/requirements.txt
    ```

### Usage 

1. Open the Jupyter notebook:
    ```sh
    jupyter notebook code/ms1_1.ipynb
    ```
    or open the code\non_dl_methods\svm_exp.ipynb notebook to run the svm
    or open the code\main.py file and run that to do all the preprocessing and BERT training

2. Follow the instructions in the notebook to run the NLP tasks.

### VS Code Workspace

You can use the provided VS Code workspace configuration for an enhanced development experience. Open `nlp_ms1.code-workspace` in VS Code.

## Structure
"code" contains:
- dl_methods -> transformer: a BERT implementation to apply to the dataset
- logs: various log files containing results of training runs during experimentation
- models: files for trained models for later reuse and the generated mapping of unique narrative-subnarrative pairs to indices (cc and ua topics combined)
- modules: various .py files to be used as a preprocessing data pipeline (loading data, text segmentation and normalization, train-test-split, conllu conversion and logging utilities)
The `training_data_16_October_release` directory contains training data for various languages. The `CoNLL` directory contains output files in CoNLL format.
- non_dl_methods 
    - -> keyword_matching: simple approach to the problem using very rudimentary keyword matching by counting the most used words in each narrative and then matching that to the test set where we count the most used words again and try to match those to the "train" wordcounts -> very bad accuracy and also not really usefull because of the many classes and the severe underrepresentation of some classes
    - -> svm-exp: JN documenting the application of one-vs-rest SVM to the problem
- main: .py file to execute BERT training
- ms1_1: JN documenting the main findings relevant for milestone 1 of the project
    - The data preprocessing pipeline in the JN was translated into the .py files in the "modules" folder
    - Contains some relevant information for milestone 2, mainly findings of data points that were labeled incorrectly
    - The label taxonomy created in this JN was not used for the implementations of milestone 2. For milestone two, mapping unique narrative-subnarrative pairs was done using the pairs contained within the dataset, which are not all that are outlined in the taxonomy. Using the whole taxonomy might be relevant later on if additional data is used that contains additional pairs not currently present in the dataset.

"CoNLL" contains:
- CoNLL files representing the data in CoNLL format (currently not used since the SVM and BERT use different input representations) ---ADD WHY 20 FILES---
"info" contains:
- The task description, label taxonomy files and possibly additional information files

"training_data_16_October_release" contains:
- The data (articles and annotations) specific to the problem with folders for different languages (BG = bulgarian, EN = English, HI = Hindi, PT = Portuguese)

"wandb" contains:
- Training run documentations generated using the python package wandb

test.conllu: is the output file of the train_test_split function wher we split the whole conllu dataset into a train and test subset
train:conllu:  is the output file of the train_test_split function wher we split the whole conllu dataset into a train and test subset

## Outlook
Status 15.12.2024: The project will be continued mainly by adjusting the existing solutions to better address the specific challenges of the problem, for example:
- Find ways to deal with the severe underrepresentation of many classes
- Use additional languages for training
- Identifying a model that deals with these challenges better than the currently used models

## License

This project is licensed under the MIT License.

##
JonasKruse and KrOnAsk are the same person, something went wrong with the user merging of GitHub accounts