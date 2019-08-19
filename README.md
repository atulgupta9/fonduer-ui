# Data Extraction from Richly Formatted Documents

### Use Case Demo - Extraction of CEO names from financial documents


#### Installation

**Environment Setup**

- Unzip the **fonduer_env.zip** 

- Activate the environment
```sh
$ source fonduer_env/bin/activate
```
or 

```sh
$ pip install -r requirements.txt
```
and unzip fonduer_package.zip and place all the contents in the site packages

**DB Setup**

```sh
$ bash setup_db.sh
```

**To run the app**

```sh
$ python app.py runserver
```

### Usage

- Navigate using the App UI. 
- The documents for training and testing the application are placed in the documents folder.
- Upload the documents for training
- After training is complete for the first time, it can be disabled by modifying **app.config** in **app.py**
- To disable make the following changes in app.py
```python
app.config['FIRST_TIME_TRAIN'] = False
app.config['FIRST_TIME_TRAIN_MODEL'] = False
```
- Similarly the predict can be disabled 
```python
app.config['FIRST_TIME_PREDICT'] = False
```