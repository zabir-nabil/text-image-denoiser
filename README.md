# text-image-denoiser

*deep learning models trained to denoise/deblur text images (signle frame, multi-frame)* **[pytorch]**

benchmark dataset: [LP HDR dataset](http://academictorrents.com/details/8ed33d02d6b36c389dd077ea2478cc83ad117ef3) 

### Model-1: U-Net

<img src="unet.png" width="425" alt="UNet">

- [x] s&p + gaussian blur
- [x] single frame
- [ ] real noise map

#### Loss plot
<hr>
   <img src="model_loss_unet.png" width="425" alt="UNet loss">
<hr>

#### Denoised samples

<img src="unet_demo/demo0.png" width="425"/> <img src="unet_demo/demo7.png" width="425"/>