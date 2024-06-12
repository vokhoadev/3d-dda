# Brain Tumor Segmentaion using Deep Learning.
## How to use?
### You need to clone my repo and setup it.

```bash
~ git clone https://github.com/RC-Sho0/Graduate-Thesis.git
```
Move to source code folder.

```bash
~ cd Graduate-Thesis
```
Set it up
```bash
~ python utils/setup.py <!your wandb key or empty>
```

Prepair you datalist
```bash
~ python libs/data/prepare_datalist.py --path "<Your folder contain dataset>" --output "/{path of file}/datalist.json" --stage "train" --split 'true'
```

### For training
```json
{
    "model_name": "//one in [segresnet, dynunet, vnet, swinunetr, dynunet_dda]",
    "att": "//Only use if model_name is dynunet_dda else []" 
    "project": "baseline",
    "model_trained": "//null for training stage, trained path for testing stage",
    "datalist": "//your datalist.json path",
    "config":{
        "loss": "mse",
        "max_epochs": 120,
        "name":"dda_+",
        "lr":3e-4,
        "tmax": 30,
        "results_dir":"//dir of outputs",
        "log": "//true if you want show on your wandb",
    }
}   
```
**Training:**
```bash
~ python seg_train.py --input <your exp.json file>
```
## Predict
#### 3D Dual-Domain Attention
Fill model_trained in exp.json then run
```bash
~ python libs/data/prepare_datalist.py --path "<Your folder contain dataset>" --output "/{path of file}/datalist.json" --stage "test" 

~ python 3d_dda.py --input <your exp.json file>
```
