import json
import warnings
from argparse import Namespace
from pathlib import Path

import mlflow
import pandas as pd
from bnai import predict
from bnai import train
from bnai import utils

from bnai import config
from bnai import data
from bnai.config import logger

# import predict
# import train
# import utils
# import config
# import data
# from config import logger

import typer
app = typer.Typer()
warnings.filterwarnings("ignore")

@app.command()
def elt_data():
    """Extract, load and transform out data assets."""

    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)
    logger.info("✅ Saved data!")

@app.command()
def train_model(args_fp):
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))


import optuna
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

@app.command()
def optimize(
    args_fp: str = "config/args.json",
    study_name: str = "optimization",
    num_trials: int = 20
) -> None:
    """Optimize hyperparameters."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


import tempfile

# Experiment tracking
import joblib

@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: bool = False,
) -> None:
    """Train a model given arguments"""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


def load_artifacts(run_id):
    """Load artifacts for a given run_id"""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }

@app.command()
def predict_tag(text: str = "", run_id: str = None) -> None:
    """Predict tag for text.
    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    # app()
    # elt_data()
    """No Optimization"""
    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # train_model(args_fp)

    """With Optimization"""
    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # optimize(args_fp, study_name="optimization", num_trials=20)

    """Train a model with experiment tracking"""
    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # train_model(args_fp, experiment_name="baselines", run_name="sgd")

    """Predict the tag for the text"""
    text = "Transfer learning with transformers for text classification."
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    predict_tag(text=text, run_id=run_id)
