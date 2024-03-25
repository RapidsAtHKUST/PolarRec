## PolarRec: Improving Radio Interferometric Data Reconstruction with Polar Coordinates

### Abstract
In radio astronomy, visibility data, which are measurements of wave signals from radio telescopes, are transformed into images for observation of distant celestial objects. However, these resultant images usually contain both real sources and artifacts, due to signal sparsity and other factors. One way to obtain cleaner images is to reconstruct samples into dense forms before imaging. Unfortunately, existing reconstruction methods often miss some components of visibility in frequency domain, so blurred object edges and persistent artifacts remain in the images. Furthermore, the computation overhead is high on irregular visibility samples due to the data skew. To address these problems, we propose PolarRec, a transformer-encoder-conditioned reconstruction pipeline with visibility samples converted into the polar coordinate representation. This representation matches the way in which radio telescopes observe a celestial area as the Earth rotates. As a result, visibility samples distribute in the polar system more uniformly than in the Cartesian space. Therefore, we propose to use radial distance in the loss function, to help reconstruct complete visibility effectively. Also, we group visibility samples by their polar angles and propose a group-based encoding scheme to improve the efficiency. Our experiments demonstrate that PolarRec markedly improves imaging results by faithfully reconstructing all frequency components in the visibility domain while significantly reducing the computation cost in visibility data encoding. 


### Run the demo

#### Setup the conda environment
Set up the conda environment using the `requirements.txt` file.


#### Datasets
Please find the datasets at https://astronn.readthedocs.io and project of Wu. et al [1].


#### Modify the model, datapath path parameter within the bash script.

#### Train Model
Run the `train_model.sh` script from command line:
```
sh ./train_model.sh
```

#### Inference using the trained model
Modify the paths in `eval_model.sh` script.

Run the `eval_model.sh` script from command line:

```
sh ./eval_model.sh
```
The results will be saved in the `'../test_res1'` folder, including visibility reconstruction and resultant image.

Evaluate the results with SSIM, PSNR, LFD:

```
python metrics.py
```



**Reference**

[1] Wu B, Liu C, Eckart B, et al. Neural interferometry: Image reconstruction from astronomical interferometers using transformer-conditioned neural fields[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(3): 2685-2693.
