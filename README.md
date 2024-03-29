# Images-Blending

In this project I've implemented Images Blending and Hybrid Images. Using image pyramids, low-pass and band-pass filtering, I've created well designed algorithm that shows great results.
I've constructed Gaussian and Laplacian pyramids, and use these to implement pyramid blending, and finally compare the blending results when using different filters in the various expand and reduce operations.

See the report PDF for a complete overview of the project, in general the blending happens as follows: 

The software is built from 4 steps: 

1. Gaussian & Laplacian pyramid construction. 

2. Laplacian pyramid reconstruction (from pyramid into original image)

3. Pyramid display.

4. Pyramid Blending, in the follows steps: 

    - Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively.
  
    - Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64).
  
    - Construct the Laplacian pyramid Lout of the blended image for each level k by:
  
      Lout[k] = Gm[k] · L1[k] + (1 − Gm[k]) · L2[k]
  
      where (·) denotes pixel-wise multiplication.
  
  5. Reconstruct the resulting blended image from the Laplacian pyramid Lout (using ones for coefficients).


  
Consider the following example: 
  
Without blending, given the two following images of apple and orange, we'll get the follwing:
  
![without_mask](https://user-images.githubusercontent.com/64755588/206402595-a55a01b1-9809-4472-8b69-29deeeb864da.png)


With blending, given the upper two images and the binary mask bellow, we will get the down-right result: 

![blending_fruits](https://user-images.githubusercontent.com/64755588/206403429-69cc550b-f1cf-45fc-b6b6-1bf6c93b4818.png)




![bla](https://github.com/AvinoamNukrai/Images-Blending/assets/64755588/4512f536-6a65-4237-bed6-67408e301626)






Cool, isn't?
