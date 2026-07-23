# INR_for_DynamicMRI
This project serves as a continuation of the work presented in the paper "Spatiotemporal implicit neural representation for unsupervised dynamic MRI reconstruction"
 
## 1. Environmental Requirements  
### To run the reconstruction demo, the following dependencies are required:  
* Python 3.10.X  ***(Important)***
* PyTorch 2.0.0
* torchkbnufft 1.4.0
* [tiny-cuda-nn 1.7](https://github.com/NVlabs/tiny-cuda-nn)
* imageio 2.18.0
* torchvision, tensorboard, h5py, scikit-image, tqdm, numpy, scipy
* **Additional packages**:
  * [monai](https://github.com/Project-MONAI/MONAI) (e.g. `pip install monai`)
  * [mat73](https://github.com/skjns/mat73) (e.g. `pip install mat73`)
  * [sigpy](https://github.com/mikgroup/sigpy) (e.g. `pip install sigpy`)

## 2. Dataset & Preparation

* **Database Name & Source:** This project uses the **CMRxRecon** dataset: [link](https://www.synapse.org/Synapse:syn51471091/wiki/622170). 
* **Sample Data:** You can download a sample of the dataset from [here](https://drive.google.com/file/d/1DIdtHcHUDEqx-qL4930-pz9mxCI8OYMR/view?usp=sharing).
* **File Types:** The code expects and reads **.mat** files. 
* **Specifying the Data Path:** You do not need to hardcode the path in the scripts. The path to the database should be specified at runtime using the command-line arguments:
  * For standard reconstructions: Use the `-d` or `--data_dir` flag.
  * For meta-learning: Use the `--train_data_dir`, `--valid_data_dir`, and `--test_data_dir` flags.

## 3. Run the Demos

### Standard Unsupervised Reconstruction (main.py)
To run the conventional optimization-based reconstruction slice-by-slice, use the following command:  
```bash
python main.py -g 0 -s 13 -r -m -d /path/to/dataset
```
Key parameters:
* `-g`, `--gpu`: GPU index (e.g. `0`)
* `-s`, `--spokes`: Number of spokes to reconstruct (e.g. `13`)
* `-d`, `--data_dir`: Path to the test dataset folder (default: `D:\MRI_DATASETS\Test`)
* `-m`, `--mask`: Enables coarse-to-fine mask strategy
* `-r`, `--relL2`: Enables relative L2 loss constraint

To ablate relative L2 loss, use the following code:  
```bash
python main.py -g 0 -s 13 -m -d /path/to/dataset
```

To ablate the coarse-to-fine strategy, use the following code:  
```bash
python main.py -g 0 -s 13 -r -d /path/to/dataset
```

---

### Baseline Reconstructions
To evaluate the added classical and iterative baselines against the proposed method, use the following scripts:

**NUFFT (Naive) Baseline:**
```bash
python main_nufft.py -g 0 -s 13 -d /path/to/dataset
```
**GRASP Baseline:**
```bash
python main_grasp.py -g 0 -s 13 -r -d /path/to/dataset
```
---

### Meta-Learning Pretraining & Adaptation (main_meta_lr.py)
To pre-train the model parameters across different training subjects using meta-learning (Reptile) and adapt to a test subject, run:
```bash
python main_meta_lr.py -g 0 -s 13 -r -m --train_data_dir /path/to/train --valid_data_dir /path/to/valid --test_data_dir /path/to/test
```
Key meta-learning parameters:
* `--meta_epochs`: Number of meta-training iterations (default: `20`)
* `--meta_lr`: Meta learning rate / Reptile outer step size (default: `0.1`)
* `--inner_steps`: Inner adaptation steps per task (default: `5`)
* `--tasks_per_meta`: Number of tasks sampled per meta-iteration (default: `4`)
* `--task_frames`: Number of frames per task (default: `1`)
* `--train_data_dir`, `--valid_data_dir`, `--test_data_dir`: Folders containing corresponding dataset partitions.
* `--z_index`: The slice index (z_index) to extract for validation (default: `2`). Note that training is performed on random z-index slices.

---

### Interpolation Demos
To run the spatial/temporal interpolation demos:
```bash
python main_spatial_interp.py -g 0 -s 34 -r -m
```
or
```bash
python main_temporal_interp.py -g 0 -s 34 -r -m
```

The rest of the parameters can be easily changed by adding arguments to the parser. 
The detailed definitions of the arguments can be found by: 
```bash
python main.py -h
python main_meta_lr.py -h
```
