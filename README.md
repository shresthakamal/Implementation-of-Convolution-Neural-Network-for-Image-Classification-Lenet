## Implemenation of Convoluted Neural Network, CNN, for image classification in Mnist Fashion Dataset

### 1st COnvolution Layer

28x28x1  
10 kernel, filters  
stride -> 1  
Dimension of Filter -> 3x3  
padding valid  
o/p -> 26x26x10  
filter_shape = 4D Tensor = [fHeight, fWidth, nChannels, numFilters] = [3,3,1,10]


### Max Pool Layer

stride -> 2,2  
poolSize -> 2x2  
o/p -> 13x13x10  

### 2nd Convolution Layer

13x13x10   
20 kernel  
stride 1  
Dimension of Filter -> 3x3  
padding valid  
o/p ->  11x11x20  
filter_shape = 4D Tensor = [fHeight, fWidth, nChannels, numFilters] = [3,3,1,20]

### Max Pool Layer

stride -> 2,2  
poolSize -> 2x2  
o/p ->  6x6x20  

### Flattening Layer
input -> 6x6x20
output -> 720x1
