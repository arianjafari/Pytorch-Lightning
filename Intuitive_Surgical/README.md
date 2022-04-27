# Setting up the environment

First, download some version of Anaconda/Miniconda, e.g., from [here](https://docs.conda.io/en/latest/miniconda.html) and install.  Next, ensure that the `conda` binary is on your path and run

```bash
$ conda create -n mlflow-launcher python=3.8 -y && conda activate mlflow-launcher && pip install -r ./requirements.txt
```

where `mlflow-launcher` is the name of the environment that will launch your entrypoints.  Note that with the entrypoint-based workflow of this repo, the `mlflow-launcher` environment is only used to launch the entrypoints.  The code executed by the entrypoint `command`s automatically use the custom `conda.yaml` environment included in the repo.


# Running the pipeline

To run entrypoints, first activate the launcher environment via

```bash
$ conda activate mlflow-launcher
```

Then you can run then entrypoints contained in  `MLproject`, e.g., 

```bash
$ mlflow run -P test_path=./Release_Test/frames/ -P checkpoint_path=./pl_models/Intuitive/checkpoint/ -P best_model=epoch=15-step=24351.ckpt -P batch_size=16 -e predict .
```

# ML Pipeline steps

1) ```mlflow run -e prepare_trainData .``` This step converts all the `mp4` files in `Release_v1/videos` to separate frames and saved them in `Release_v1/frames` as `npz` format along their labels for the training step
2) ```mlflow run -e prepare_testData .``` This step converts all the `mp4` files in `Release_Test/videos` to separate frames and saved them in `Release_Test/frames` as `npz` format along their labels (negative one by default) for the test/predict step
3) ```mlflow run -e train .``` This step loads the processed data in step 1 and feed them to the ML model for training.
4) ```mlflow run -e predict .``` This step loads the processed data in step 2 and feed them to the ML model in eval mode for prediction.
