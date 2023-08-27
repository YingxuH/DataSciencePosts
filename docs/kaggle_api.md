## Notes on kaggle api

Notes on collecting kaggle datasets information by calling api.

install python kaggle package
```
pip install kaggle --upgrade
```

follow the [instruction](https://github.com/Kaggle/kaggle-api#api-credentials) to put the API credentials at the correct location

```python
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
```

list kaggle datasets. (refer to the original [document](https://github.com/Kaggle/kaggle-api/tree/main#list-datasets)).
The `sort_by` argument default value is 'hottest'. Valid options are 'hottest', 'votes', 'updated', and 'active'.
```python
api.datasets_list(sort_by="votes", page=1)
```

list all files from a specific dataset. The function returns a list of kaggle files object, whose attributes can be viewed by calling `dir(file)`. 
To extract its columns information, simply call `file.columns` which will return a list of dictionary with each columns metadata.
```python
files = api.dataset_list_files(dataset=<dataset_ref>)

for file in files:
    print(file.columns)
```

To download any specific file from kaggle

```python
api.dataset_download_file(<dataset_ref>, <file_name>, path=<local_destination_directory>)
```




