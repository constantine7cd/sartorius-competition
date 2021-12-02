import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def _make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _build_ckpt_path(experiment_dir, experiment_name):
    return os.path.join(experiment_dir, 'checkpoints', experiment_name)


def _build_logs_path(experiment_dir, experiment_name):
    return os.path.join(experiment_dir, 'logs', experiment_name)


def train_model(cfg: dict):
    experiment_dir = cfg['experiment_dir']
    experiment_name = cfg['experiment_name']

    log_dir = _build_logs_path(experiment_dir, experiment_name)
    ckpt_dir = _build_ckpt_path(experiment_dir, experiment_name)

    _make_dir(log_dir)
    _make_dir(ckpt_dir)

    epochs = cfg['epochs']

    ds_train = cfg['ds_train'](**cfg['ds_train_params'])
    dataset_train = ds_train.dataset

    ds_val = cfg['ds_val'](**cfg['ds_val_params'])
    dataset_val = ds_val.dataset

    compile_params = {
        'optimizer': cfg['optimizer'](**cfg['optimizer_params']),
        'loss': cfg['loss'],
        'metrics': cfg['metrics']
    }

    model = cfg['model'](**cfg['model_params'])
    model.compile(**compile_params)

    tboard_callback = TensorBoard(log_dir=log_dir)

    ckpt_params = {
        'filepath': os.path.join(ckpt_dir, 'model_{epoch}'),
        'save_best_only': True,
        'save_weights_only': True,
        'monitor': 'val_loss',
        'verbose': 1
    }

    ckpt_callback = ModelCheckpoint(**ckpt_params)

    training_history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        steps_per_epoch=ds_train.n_steps,
        validation_steps=ds_val.n_steps,
        callbacks=[tboard_callback,
                   ckpt_callback]
    )

    return training_history
