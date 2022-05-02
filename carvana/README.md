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
$ mlflow run -P csv_path=customer_chat_sample.csv -P checkpoint_path=./pl_models/Carvana/checkpoint -P batch_size=16 -e train .
```

# ML Pipeline steps

1) ```mlflow run -e train .``` This step reads the `customer_chat_sample.csv`, split data into train, val, and test set with 80%, 10%, and 10%, respectively. At the end of the process it outputs the confusion matrix and classification report as `csv` files. 
3) ```mlflow run -e predict .``` This step reads the `customer_chat_sample.csv`, split data into train, val, and test. As the random seed is perserved and the same as the training step, the test set in this step is identical as the training step. The best model will be loaded from the checkpoint and the prediction on the test set is being done and the metrics of interest are being stored as `csv` files.
