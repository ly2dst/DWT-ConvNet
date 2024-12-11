    This project is built for Multi-scale Wavelet Convolution for Non-destructive Defect Detection in Commercial Persimmons
    Due to the size limitation of github, the images of persimmon at wavelenth of 545nm has been extracted as regular image stored in the folder of "dataset", while train and test set have also been divided into different folders.
    The single data has reached the limitation of 25MB, which can't be uploaded to github. For the detailed hyperspectral cube data, you can conduct the author at ly2dst@163.com。
    All the functions such as loading dataset, 2-demension wavelet transaction and multiscale convolution have been generated into the python files. By running train.py model can be built in convenience and efficiency.
    The structure of hyperspectral files are as shown in the picture, which contains a description file with extension of".hdr" and data file with extension of ".cube".
<img width="651" alt="image" src="https://github.com/user-attachments/assets/71119618-2f7a-4bb3-9582-02b7d28a3a7b">
    The hdr file descripts the detailed information of hyperspectral data, including the number of rows and columes, number of bands, the data type, endianness, pixel arrangement and the involved wavelengths.
    <img width="454" alt="image" src="https://github.com/user-attachments/assets/85e08ce5-6e44-4c4a-8c27-36e1620f4180">

