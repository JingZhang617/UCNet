# UCNet (CVPR2020)
UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders

![alt text](./training_rgbd.png)

# *** Update 2021-01-14***

Add journal submission link:

https://arxiv.org/abs/2009.03075


# *** Update 2020-09-05***

Add performance of our UC-Net on DUT RGBD saliency testing dataset(https://github.com/jiwei0921/RGBD-SOD-datasets) (400 images):

https://drive.google.com/file/d/14LpM8yB-yKqQiV5sAtDhUqTKwMBFrTGv/view?usp=sharing

# *** Update 2020-09-04***

Our journal extension will coming soon. Please find links below for our results and ablation studies.

1. Our CVAE based model results:
https://drive.google.com/file/d/12Q-MBbABFHE5DygbJf7OOSwgGL1q0vGz/view?usp=sharing
2. Our ABP based model results:
https://drive.google.com/file/d/102pSSNT1iDbohf-uWBxtQjHoz3B-IjeT/view?usp=sharing
3. Our middle-fusion ablation study:
https://drive.google.com/file/d/1hEA2slr5hW4n7XlKqN0jMXtx59NRZTpt/view?usp=sharing
4. Our late-fusion ablation study:
https://drive.google.com/file/d/1uwFSxeOfxl0KKG13V_JRhB92hZbEr_Jr/view?usp=sharing

Code for our submission: https://drive.google.com/file/d/1Bz_vy2farSXEU2v1E23NT2s7Cm4dSqv3/view?usp=sharing, which includes:

1) Our CVAE model (hybrid loss), 2) Our ABP model, 3) the middle-fusion model, 4) the late-fusion model, 5) the GSCNN model, 6) the simple CVAE model.

#

# Setup 

Install Pytorch

# Train Model

1. Download training data from: https://drive.google.com/file/d/1zslnkJaD_8h3UjxonBz0ESEZ2eguR_Zi/view?usp=sharing, and put it in folder "data"

2. Run ./train.py

# Test Model

1. Download the trained model from: https://drive.google.com/file/d/1nzGLnlmntTGbcaShfQvE6ouyfWJD-pIB/view?usp=sharing, and put it in folder "models"

2. Download the testing dataset from: https://drive.google.com/file/d/1n1bEfw3lzI6p8u1xaxEqnuEXgNqbAFTA/view?usp=sharing, and put it in folder "test_dataset"

3. Modify testing image path in "test.py" accordingly

4. Run ./test.py


# Our results:

![alt text](./competing_results_show.png)

1. Results of our model on six benchmark datasets can be found: https://drive.google.com/open?id=1NVJVU8dlf2d9h9T8ChXyNjZ5doWPYhjg or: 链接: https://pan.baidu.com/s/1M9_Bv16-tTnlgF6ayBmc6w 提取码: u8s5

2. Performance of our method can be found: https://drive.google.com/open?id=1vacU51eG7_r751lAsjKTPSGrdjzt_Z4H or: 链接: https://pan.baidu.com/s/1o6kFY8Y81_V-pftc8kTgUw 提取码: fqpd

# Performance of competing methods

![alt text](./E_F_measure.png)

Performance of competing methods can be found: https://drive.google.com/open?id=1NUMp_zKXSx8jc7u7HnPQmcYXtoiLWj6t or: 链接: https://pan.baidu.com/s/1g1dbwsGowLD_FFAx0ciSHw 提取码: sqar 

# Our Bib:

Please cite our papers if you like our work:
```
@inproceedings{Zhang2020UCNet,
  title={UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders},
  author={Zhang, Jing and Fan, Deng-Ping and Dai, Yuchao and Anwar, Saeed and Sadat Saleh, Fatemeh and Zhang, Tong and Barnes, Nick},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2020}
}
```
```
@article{zhang2021uncertainty,
  title={Uncertainty Inspired RGB-D Saliency Detection},
  author={Jing Zhang and Deng-Ping Fan and Yuchao Dai and Saeed Anwar and Fatemeh Saleh and Sadegh Aliakbarian and Nick Barnes},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2021}
}
```
# Benchmark RGB-D SOD

The complete RGB-D SOD benchmark can be found in this page:

http://dpfan.net/d3netbenchmark/


# Contact

Please contact me for further problems or discussion: zjnwpu@gmail.com


