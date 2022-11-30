Single image super-resolution (SISR) is used to recover high-resolution images from a single low-resolution image. Several quantitative evaluation metrics have been designed, such as Peak Signal-to-Noise Ratio (PSNR) and perceptual loss. PSNR is focused on pixel-to-pixel intensity differences, while it tends to render over-smoothed results without high-frequency details. On the other hand, the perceptual loss is proposed to optimize the super-resolution model in a feature space rather than pixel space. Recently, a few GAN models have been proposed to generate more natural-like images based on perceptual loss. 

## ESRGAN 

ESRGAN is developed to reduce further the gap between the visual quality of rendered and ground truth high-resolution images. Its contributions are fourfold: 
- It introduces the Residual-in-Residual Dense Block (RDDB) into SRGAN's architecture. 
- It removes Batch Normalization (BN) layers and uses residual scaling and small-value initialization to facilitate training a deep network.
- It improves the discriminator using Relativistic average Gan (RaGAN).
- It adjusts the calculation of perceptual loss. 
- It does network interpolation to balance the visual quality and pixel-level loss. 
- It pre-trains the network with PSNR loss (L1 loss) before taking it as the generator and further training it using the GAN method. 

#### RDDB Block 
RDDB introduces a residual-in-residual structure, where residual learning is used both in the main path and between each block. In addition, the output of each CNN is fed to all the succeeding layers. Heuristically, more layers and connections can always boost performance. 

#### Removing BN layers
Empirically, BN layers are more likely to bring artifacts when the network is deeper and trained under the GAN framework. The authors thus remove BN layers for stable training and consistent performance. Furthermore, removing BN layers also helps to increase generalization ability and to reduce computational and memory complexity.
On the other hand, scaling factors are applied to residuals before adding them to the main path to facilitate training a deeper neural network. A smaller initialization is also found helpful for training residual architecture. 

### Relativistic average Discriminator 
Instead of directly estimating whether an input image is real or fake, a relativistic discriminator tries to predict the probability that a real image xr is relatively more realistic than a fake one xf. It is claimed that doing so will train a better generator with the benefits from both real and rendered images. 

### Perceptual loss
Perceptual loss is calculated from the output of activation layers of a pre-trained deep network. Prior arts tend to use the features after the activation layers. However, it is found in this paper that using features before activation layers can alleviate the sparse feature problem and inconsistent brightness problem. In addition, a different pre-trained VGG network which focuses more on the image texture, is adopted. 

### Network Interpolation
To balance out the noises and visual quality in the rendered images. The authors trained a PSNR-oriented network and another GAN-based network with the same architecture. The proposed network takes a weighted average of their weights as the final weights.  

### Pre-training
Pre-training the generator with PSNR loss is claimed to be helpful for avoiding undesired local optima and focusing more on texture discrimination. 

The experiments are performed with a scaling factor of x4 and 2650 4k images. The proposed model outperforms previous SOTA methods in both sharpness and detail.

## Real-ESRGAN
Although the ESRGAN model has made several enhancements to the traditional SRGAN, its performance on real-world images might still have problems due to the bias in training data. Classical image degradation models cannot simulate real-world low-quality photos. Therefore, this paper proposes a high-order degradation model to better intimate the real-world phenomenon.

The contributions of this work are as follows:
- It proposes a high-order degradation process to model practical degradations. 
- It applies pixel-shuffle on the input of the generators. It switched the discriminator from the VGG framework to U-Net and introduced spectral normalization to deal with the complex data set and alleviate the instability of U-net. 

