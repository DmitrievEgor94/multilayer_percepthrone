import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import yaml
import os
import numpy as np
from sklearn.metrics import log_loss

from nets import MLPClassification


def load_dataset(file_name):
    if not os.path.exists(file_name):
        iris = datasets.load_iris(as_frame=True)

        data = iris.data
        data['target'] = iris.target

        data.to_excel(file_name, index=False)
    else:
        data = pd.read_excel(file_name)

    return data


def get_learning_rate_graphs(net_params, train_df, test_df):
    learning_rates = np.arange(0.001, 0.3, 0.01).tolist()
    learning_rates += [0.5, 1, 1.5, 2, 10]
    moments = [0, 0.1]

    folder_to_save_data = 'classification_data/results'

    all_losses_train = []
    all_accuracies_train = []

    all_losses_test = []
    all_accuracies_test = []

    for moment in moments:
        all_losses = []

        for lr in learning_rates:
            np.random.seed(random_state)
            net = MLPClassification(*net_params)

            print('learning_rate: ', lr)
            for i in range(90):
                for j in range(train_df.shape[0]):
                    a, b = net.forward(train_df[atrs].iloc[j])
                    net.backward(train_df['target_one_hot'].iloc[j], lrate=lr, momentum=moment)

            losses = []
            accuracies = []

            for j in range(test_df.shape[0]):
                a, b = net.forward(test_df[atrs].iloc[j])
                losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
                accuracies.append(test_df['target'].iloc[j] == np.argmax(a))

            all_losses_test.append(np.mean(losses))
            all_accuracies_test.append(np.mean(accuracies))

            print('loss:', np.mean(losses))
            print('accuracy:', np.mean(accuracies))

            losses = []
            accuracies = []

            for j in range(train_df.shape[0]):
                a, b = net.forward(train_df[atrs].iloc[j])
                losses.append(log_loss(y_true=train_df['target_one_hot'].iloc[j], y_pred=b))
                accuracies.append(train_df['target'].iloc[j] == np.argmax(a))

            all_losses_train.append(np.mean(losses))
            all_accuracies_train.append(np.mean(accuracies))

    # df_lr_losses = pd.DataFrame(zip(learning_rates, all_losses), columns=['learning rate', 'log_loss'])
    #     df_lr_losses.to_excel(f'{folder_to_save_data}/learning_rate_loss_moment_{round(moment, 2)}.xlsx', index=False)

        df_losses = pd.DataFrame(zip(learning_rates, all_losses_test, all_accuracies_test),
                                 columns=['traing data length', 'log_loss',
                                          'accuracy'])

        df_losses.to_excel(f'{folder_to_save_data}/learning_rate_loss_accuracy_test_{round(moment, 2)}.xlsx', index=False)

        df_losses = pd.DataFrame(zip(learning_rates, all_losses_train, all_accuracies_train),
                                 columns=['traing data length', 'log_loss',
                                          'accuracy'])
        df_losses.to_excel(f'{folder_to_save_data}/learning_rate_loss_accuracy_test_{round(moment, 2)}.xlsx', index=False)


def get_training_data_length_graphs(net_params, train_df, test_df):
    train_data_lengths = [30, 60, 90, 120]

    learning_rate = 0.1
    momentum = 0.1

    all_losses_train = []
    all_accuracies_train = []

    all_losses_test = []
    all_accuracies_test = []

    folder_to_save_data = 'classification_data/results'

    for length in train_data_lengths:
        np.random.seed(random_state)
        net = MLPClassification(*net_params)

        train_df = train_df.sample(frac=1, random_state=2).reset_index(drop=True)

        print(length)

        for _ in range(35):
            for j in range(length):
                a, b = net.forward(train_df[atrs].iloc[j])
                net.backward(train_df['target_one_hot'].iloc[j], lrate=learning_rate, momentum=momentum)

        losses = []
        accuracies = []

        for j in range(test_df.shape[0]):
            a, b = net.forward(test_df[atrs].iloc[j])
            losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(test_df['target'].iloc[j] == np.argmax(a))

        all_losses_test.append(np.mean(losses))
        all_accuracies_test.append(np.mean(accuracies))

        print('loss:', np.mean(losses))
        print('accuracy:', np.mean(accuracies))

        losses = []
        accuracies = []

        for j in range(train_df.shape[0]):
            a, b = net.forward(train_df[atrs].iloc[j])
            losses.append(log_loss(y_true=train_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(train_df['target'].iloc[j] == np.argmax(a))

        all_losses_train.append(np.mean(losses))
        all_accuracies_train.append(np.mean(accuracies))

    df_losses = pd.DataFrame(zip(train_data_lengths, all_losses_test, all_accuracies_test),
                             columns=['traing data length', 'log_loss',
                                      'accuracy'])

    df_losses.to_excel(f'{folder_to_save_data}/training_data_length_loss_accuracy_test.xlsx', index=False)

    df_losses = pd.DataFrame(zip(train_data_lengths, all_losses_train, all_accuracies_train),
                             columns=['traing data length', 'log_loss',
                                      'accuracy'])
    df_losses.to_excel(f'{folder_to_save_data}/training_data_length_loss_accuracy_train.xlsx', index=False)

def get_number_neurons_length_graphs(net_params, train_df, test_df):
    num_atrs, _, outputs_num = net_params

    learning_rate = 0.1
    momentum = 0.1

    all_losses_train = []
    all_accuracies_train = []

    all_losses_test = []
    all_accuracies_test = []

    folder_to_save_data = 'classification_data/results'

    hidden_layer_length_list = [2, 3, 4, 5, 6, 7, 10, 12, 15, 16, 17, 19, 20]

    for hidden_layer_length in hidden_layer_length_list:
        np.random.seed(random_state)
        net = MLPClassification(num_atrs, hidden_layer_length, outputs_num)

        print(hidden_layer_length)

        for _ in range(40):
            for j in range(train_df.shape[0]):
                a, b = net.forward(train_df[atrs].iloc[j])
                net.backward(train_df['target_one_hot'].iloc[j], lrate=learning_rate, momentum=momentum)

        losses = []
        accuracies = []

        for j in range(test_df.shape[0]):
            a, b = net.forward(test_df[atrs].iloc[j])
            losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(test_df['target'].iloc[j] == np.argmax(a))

        all_losses_test.append(np.mean(losses))
        all_accuracies_test.append(np.mean(accuracies))

        print('loss:', np.mean(losses))
        print('accuracy:', np.mean(accuracies))

        losses = []
        accuracies = []

        for j in range(train_df.shape[0]):
            a, b = net.forward(train_df[atrs].iloc[j])
            losses.append(log_loss(y_true=train_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(train_df['target'].iloc[j] == np.argmax(a))

        all_losses_train.append(np.mean(losses))
        all_accuracies_train.append(np.mean(accuracies))

    df_losses = pd.DataFrame(zip(hidden_layer_length_list, all_losses_test, all_accuracies_test),
                             columns=['hidden layer length', 'log_loss',
                                      'accuracy'])

    df_losses.to_excel(f'{folder_to_save_data}/hidden_layer_length_loss_accuracy_test.xlsx', index=False)

    df_losses = pd.DataFrame(zip(hidden_layer_length_list, all_losses_train, all_accuracies_train), columns=['hidden layer length', 'log_loss',
                                                                                            'accuracy'])
    df_losses.to_excel(f'{folder_to_save_data}/hidden_layer_length_loss_accuracy_train.xlsx', index=False)


def get_iteration_number_graphs(net_params, train_df, test_df):
    learning_rate = 0.1
    momentum = 0.1

    all_losses_test = []
    all_accuracies_test = []

    all_losses_train = []
    all_accuracies_train = []

    folder_to_save_data = 'classification_data/results'

    epochs_number = [2, 4, 6, 7, 8, 9, 10, 11, 12] + np.arange(20, 150, 10).tolist()

    for epoch_number in epochs_number:
        np.random.seed(random_state)
        net = MLPClassification(*net_params)

        print(epoch_number)

        for _ in range(epoch_number):
            for j in range(train_df.shape[0]):
                net.forward(train_df[atrs].iloc[j])
                net.backward(train_df['target_one_hot'].iloc[j], lrate=learning_rate, momentum=momentum)

        losses = []
        accuracies = []

        for j in range(test_df.shape[0]):
            a, b = net.forward(test_df[atrs].iloc[j])
            losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(test_df['target'].iloc[j] == np.argmax(a))

        all_losses_test.append(np.mean(losses))
        all_accuracies_test.append(np.mean(accuracies))

        print('loss:', np.mean(losses))
        print('accuracy:', np.mean(accuracies))

        losses = []
        accuracies = []

        for j in range(train_df.shape[0]):
            a, b = net.forward(train_df[atrs].iloc[j])
            losses.append(log_loss(y_true=train_df['target_one_hot'].iloc[j], y_pred=b))
            accuracies.append(train_df['target'].iloc[j] == np.argmax(a))

        all_losses_train.append(np.mean(losses))
        all_accuracies_train.append(np.mean(accuracies))

    df_losses = pd.DataFrame(zip(epochs_number, all_losses_test, all_accuracies_test), columns=['epoch number', 'log_loss',
                                                                                  'accuracy'])
    df_losses.to_excel(f'{folder_to_save_data}/epoch_number_loss_accuracy_test.xlsx', index=False)

    df_losses = pd.DataFrame(zip(epochs_number, all_losses_train, all_accuracies_train), columns=['epoch number', 'log_loss',
                                                                                            'accuracy'])
    df_losses.to_excel(f'{folder_to_save_data}/epoch_number_loss_accuracy_train.xlsx', index=False)


if __name__ == '__main__':
    params = yaml.load(open('params.yaml'), yaml.FullLoader)

    # установка сида для того, чтобы результаты получались теми же самыми при повторном запуске
    random_state = params['random_state']
    np.random.seed(random_state)

    classification_folder = 'classification_data'

    df = load_dataset(f'{classification_folder}/iris_dataset.xlsx')

    # Кодировка таргета one_hot: 0 - [1, 0, 0], 1 - [0, 1, 0], 2 - [0, 0, 1]
    label_encoder = LabelBinarizer()
    one_hot_target = label_encoder.fit_transform(df['target'])

    df.loc[:, 'target_one_hot'] = pd.Series(list(one_hot_target))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    print(train_df.shape, test_df.shape)

    atrs = [col for col in df.columns if col not in ['target', 'target_one_hot']]

    num_atrs = len(atrs)
    outputs_num = train_df.target.unique().shape[0]

    net = MLPClassification((num_atrs, (num_atrs + outputs_num)//2, outputs_num))

    epoch_number = 10

    for _ in range(epoch_number):
        for j in range(train_df.shape[0]):
            net.forward(train_df[atrs].iloc[j])
            net.backward(train_df['target_one_hot'].iloc[j])

    losses = []
    accuracies = []

    for j in range(test_df.shape[0]):
        a, b = net.forward(test_df[atrs].iloc[j])
        losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
        accuracies.append(test_df['target'].iloc[j] == np.argmax(a))



