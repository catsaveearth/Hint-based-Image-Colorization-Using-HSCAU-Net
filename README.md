# Hint-based-Image-Colorization-Using-HSCAU-Net
2022-1, Gachon University, Department of Software, Computer Vision Term Project <br><br><br><br>

## Hint-based Image Colorization
Hint-based colorization reconstructs complete colorful images by propagating the color hint given by users. For the hint-based colorization task, each solution should first convert the provided RGB image to LAB image. The inputs of the model should be an ”L” image (i.e. grayscale image or intensity image) and a color hint map. The output will be a predicted ”AB” image that will be concatenated with the input L image to produce the final colorized image. An overview is shown below.

<p align="center">
  <img alt="flow" src="https://user-images.githubusercontent.com/50789540/171644257-a9d49515-fe88-405d-803a-40ccd66f0e86.png">
  <img alt="example" src="https://user-images.githubusercontent.com/50789540/171644603-d3bde206-4c83-4d50-851a-74dd5e94ec50.png">
</p>
[example : Real-Time User-Guided Image Colorization with Learned Deep Priors](https://richzhang.github.io/InteractiveColorization/)


<br><br><br>

## HSCAU-Net Model Architecture
* Hint-and Spatial-Channel attention U-Net
![그림1](https://user-images.githubusercontent.com/50789540/171643429-96694dd4-9393-41ce-8685-252bb32ffbac.png)

* Skip-connection
<p align="center">
  <img alt="skip-connection" width="800" src="https://user-images.githubusercontent.com/50789540/171643752-d8195648-b8f5-4f63-8167-68a97d023589.png">
</p>


<br><br>
## Training dataset
* Dataset Size : 10000 / 2000
* Using Place365 and ImageNet datasets
* Image size : 256 x 256
<p align="center">
  <img alt="skip-connection" width="800" src="https://user-images.githubusercontent.com/50789540/171644914-05a0c6ea-99a8-4452-a12a-c6d136b54fab.png">
</p>

<br><br>

## Using Hyper-parameter
* Batch size : 32 <br>
* Optimizer : Adam (betas = (0.5, 0.999))<br>
* Learning rate : 5e-4 (~186 epoch) -> 5e-5 (~286 epoch) -> 5e-6 (~324 epoch) -> 5e-8 (~338 epoch)<br>
<br>
<br>


## Model Result
* Our best model => 338 epoch model<br>

|epoch|PSNR|SSIM|
|------|---|---|
|186|31.392856|0.966241|
|338|31.521526|0.966740|

<br><br>

## Test dataset Result 
<p align="center">
  <img alt="ex1" width="800" src="https://user-images.githubusercontent.com/50789540/171645292-39fb7fad-445c-4c13-96ef-d3931f4c68a2.png">
  <img alt="ex2" width="800" src="https://user-images.githubusercontent.com/50789540/171645265-95d5a5e4-6d69-4a90-8891-20f6f6f89c90.png">
</p>

<br><br>


## How to use model
You can download the optimal weight pth from the link below and run this model.
```
https://drive.google.com/file/d/14MoLIljh5RvXacBYw33ajaY3Zkd76Mu9/view?usp=sharing
```

<br><br>

## Reference paper
1. Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
2. Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European conference on computer vision. Springer, Cham, 2016.
3. Zhang, Richard, et al. "Real-time user-guided image colorization with learned deep priors." arXiv preprint arXiv:1705.02999 (2017)
4. Xiao, Yi, et al. "Interactive deep colorization using simultaneous global and local inputs." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
5. Lu, Chang, et al. "ALResNet: Attention-Driven Lightweight Residual Network for Fast and Accurate Image Recognition." 2021 The 4th International Conference on Machine Learning and Machine Intelligence. 2021.

<br><br>


## License
This project belongs to computer vision Team 6, Gachon Uni. (2022-1)
