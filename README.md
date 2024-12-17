# Steps to run

- run ``` bash dgl/script/create_dev_conda_env.sh -g 12.1 -p 3.10 -t 2.1.0 ``` - according to your configuration
- run ``` conda activate dgl-dev-gpu-121 ```
- run ``` pip install torchdata==0.7.1 scikit-learn ```

- run 
``` 
python main.py --dataset cora --checkpoint ./checkpoints/best_model.pth

```
or 
- run ``` python main.py ``` - for default settings