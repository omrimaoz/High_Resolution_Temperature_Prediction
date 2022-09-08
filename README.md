# High_Resolution_Temperature_Prediction
Machine Learning Workshop : Project Booklet

Omri Maoz, Dan Barouch
Tel-Aviv University

## Instruction

1. Install python version 3.7 or later.
2. clone the entire code and datasets from main branch.
3. Install python packages from requierments.txt (On a virtual environment would be preferable).
4. run "python nltk_download.py" in terminal and download 'Collection->popular', 'Corpora->wordnet' and 'Models->punkt'in the opened window.
5. unzip IMDB_Dataset_v1_csv.zip and News_Dataset_v1_json.zip 
6. run "python prepare_datasets_IMDB.py" and "python prepare_datasets_NEWS.py" in the terminal from the 'Datasets' folder to produce compatible datasets  (this step might take time). Or first, unzip "IMDB_Dataset_15000_json.zip" and "News_Dataset_15000_json.zip" and then run both terminal lines to make the process shorter.
7. run "python main --dataset_name {dataset_name} --tag_feature {tag_feature} --dataset_size {dataset_size} --model {model}" in terminal from the root folder. replace each input from the bank:

- to_train -> exist for true else false. In order to train or if had a model can generate IR prediction image. For true bias need to be true and for station bias need to be false.
- dirs -> a string of list contins directories name ∈ ['Zeelim_30.5.19_0630_E', 'Mishmar_30.7.19_0640_E', 'Mishmar_30.7.19_0820_S', 'Zeelim_23.9.19_1100_E', 'Mishmar_3.3.20_1510_N', 'Zeelim_7.11.19_1550_W', 'Zeelim_29.5.19_1730_W']
- generate_dir -> a directory name ∈ ['Zeelim_30.5.19_0630_E', 'Mishmar_30.7.19_0640_E', 'Mishmar_30.7.19_0820_S', 'Zeelim_23.9.19_1100_E', 'Mishmar_3.3.20_1510_N', 'Zeelim_7.11.19_1550_W', 'Zeelim_29.5.19_1730_W']
- model_name ∈ ['ResNet18', 'ConvNet']
- sampling_method ∈ ['RFP', 'SFP'] , where 'RFP' for "relative" sampling and 'SFP' for uniform sampling.
- samples -> integer, number of samples.
- use_pretrained_weights -> exist for true else false. Relevent only for ResNet18.
- epochs -> integer, number of epochs.

# Branch Train_Colab_Large_Data
Branch to train on Colab with large training dataset
Use the notebook on colab after uploading the repository files to google drive.
