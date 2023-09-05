# Spam Detection
**In this project, using a model based on Bert transformer, SMS will be detected as spam.**

>Use **python3.7** and **requirement.txt** to implement the project.

### Pretrained model
We trained the model on two datasets whose file addresses are below

>* https://drive.google.com/file/d/1yWDPaiNXROSPGfrDI6JO1nFcumUBNkxK/view?usp=sharing
>* https://drive.google.com/file/d/1qnp4lR461oBbWRmGQUd5BaBZZvEWlx7e/view?usp=sharing

The first model is trained on the "sms.csv" dataset
The second model is trained on the "ExAIS.csv" dataset
Also, the first model has recorded an accuracy of **99.61** And the second model recorded an accuracy of **87.89**

### Train
run this command

>`python main.py -i /home/example.csv -o /home/example_dir -e 15`
>
> Options
> * -i Input CSV file absolute path
> * -o director for save weights
> * -b batch size
> * -e number of epoch for train
> * -lr learning rate