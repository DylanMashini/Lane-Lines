# Lane-Lines


Lane line dectetion script 



How does it work?

Step 1:

  Greyscale the image using cv2
  
Step 2:

  Blurr the image using the [Gaussian Blur algorithm](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur).
  
  We do this to reduce noise in the photo
  
Step 3:

  Detect the edges using [canny edge detection](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)
  
  This works by looking at areas with a big difference in color between them
  
Step 4:

  Mask the edges to only show edges in the area of intrest
  
Step 5:

  [Hough transform fourmula](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
  
  This detects a line in a picture. It can detect lines, and returns a mathamtical repersentation of said line. One drawback of using this function is that it can't return curved lines. 
  
Step 6:

  Overlay the Mathamatical repersentations of the lines over the original photo
  
  You could skip this step, and use the mathamatical repersnetation of the data elsewhere (ex. nural network for steering)
  
  
  Thanks for reading this was just a project for fun, and if you have any recomendations on how to make it better than email me at dylanmashini123@gmail.com

  
  
