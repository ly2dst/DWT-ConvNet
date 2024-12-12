    This project is built for Multi-scale Wavelet Convolution for Non-destructive Defect Detection in Commercial Persimmons
    All the functions such as loading dataset, 2-demension wavelet transaction and multiscale convolution have been generated into the python files. By running train.py model can be built in convenience and efficiency.
    The structure of hyperspectral files are as shown in the picture, which contains a description file with extension of".hdr" and data file with extension of ".cube".
    
<img width="651" alt="image" src="https://github.com/user-attachments/assets/71119618-2f7a-4bb3-9582-02b7d28a3a7b">

    The hdr file descripts the detailed information of hyperspectral data, including the number of rows and columes, number of bands, the data type, endianness, pixel arrangement and the involved wavelengths.
<img width="880" alt="image" src="https://github.com/user-attachments/assets/05c3540a-2fa2-4975-925f-84f574da0598">

    In python, the file can be read through package "spectral" with the code:
   ```python
    img=spectral.open_image('sample.hdr')
    data=img.load()
```  
    The ".cube"file and ".hdr" files must be in the same path, where open_image() function loads the description imformation while ".load()" function load the whole hyperspectral data in the same folder correspondingly according to the description. 
    Additionally, to fit the python package we suggest adjusting the extension of ".cube" file to ".img" for more compatibility. Because the files are read in binary mode, extension only effects the process of searching files.
    ".load()" function returns a 3-d numpy array
<img width="382" alt="image" src="https://github.com/user-attachments/assets/25cd7d1d-1586-4ed8-8866-802caaf6cead" />

