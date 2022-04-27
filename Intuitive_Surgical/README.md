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
$ mlflow run -P test_path=./Release_Test/frames/ checkpoint_path=./pl_models/Intuitive/checkpoint/ best_model=epoch=15-step=24351.ckpt batch_size=16 -e predict .
```

# Pipeline steps

1) ```mlflow run -e prepare_trainData .```
2) ```mlflow run -e prepare_testData .```
3) ```mlflow run -e train .```
4) ```mlflow run -e predict .```
