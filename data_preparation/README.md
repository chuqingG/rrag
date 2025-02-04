# Data Preparation

This folder provides the instructions about how to load the Amazon Products Dataset for testing.

## Download dataset
```python
import kagglehub

path = kagglehub.dataset_download("asaniczka/amazon-products-dataset-2023-1-4m-products")
print(path)
```
It will show the default downloading path, we need to manually copy it to the current directory for the following steps.

## Data Processing
Before loading the data, the `DB_PARAMS` in the `*.py` need to be replaced with users own PostgreSQL settings (by default the port of postgreSQL should be 5432). 
Changing the value of 'output_dir` to set the path to store the images.
```python
# Load data, it will take a super long time because of downloading images, you can change the value of N to run in smaller batch
python data_load.py
# Drop the invalid urls 
python drop_invalid.py
```



