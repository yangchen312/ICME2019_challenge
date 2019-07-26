import os
import tensorflow as tf
from models.model import RecommendModelHandler
import pandas as pd
import shutil


def train_and_predict(task):

    train_dataset_path_list = ['track2_data/final_track2_train_part.txt']
    val_dataset_path_list = ['track2_data/final_track2_train_part.txt']
    test_dataset_path_list = ['track2_data/final_track2_test_part.txt']
    save_model_dir = 'trained/{}'.format(task)
    num_epochs = 10
    optimizer = 'adam'
    batch_size = 256
    embedding_size = 20
    track = 2

    if task == 'like':
        lr = 0.0005
    else:
        lr = 0.0001

    model = RecommendModelHandler(
        train_dataset_path=train_dataset_path_list,
        val_dataset_path=val_dataset_path_list,
        test_dataset_path=test_dataset_path_list,
        save_model_dir=save_model_dir,
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        embedding_size=embedding_size,
        task=task,
        track=track,
        learning_rate=lr)

    model.train()
    model.predict()


def main():

    # basic logging setup for tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)

    # Remove previous checkpoints from model_dir
    if os.path.exists('trained/finish'):
        shutil.rmtree('trained/finish')
    train_and_predict(task='finish')
    if os.path.exists('trained/like'):
        shutil.rmtree('trained/like')
    train_and_predict(task='like')

    # Generate the submission
    test = pd.read_csv('track2_data/final_track2_test_part.txt', delimiter='\t', header=None)
    finish = pd.read_csv('results/finish.csv', header=None).round(2)
    like = pd.read_csv('results/like.csv', header=None).round(2)

    submission = pd.concat([test[[0, 2]], finish, like], axis=1)
    submission.columns = ['uid', 'item_id', 'finish_probability', 'like_probability']
    submission.to_csv('results/experiment_{}.csv'.format(1), index=False)


if __name__ == '__main__':
    main()
