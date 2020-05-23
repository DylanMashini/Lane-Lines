# Lane-Lines


Lane line dectetion script 



How does it work?

Step 1:

  Greyscale the image using cv2
  
Step 2:

  Blurr the image using the Gaussian Blur fourmula.
  
  We do this to reduce noise in the photo
  
Step 3:

  Detect the edges using canny edge detection
  
  This works by looking at areas with a big difference in color between them
  
Step 4:

  Mask the edges to only show edges in the area of intrest
  
Step 5:

  [Hough transform fourmula](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
  
  This detects a shape in a picture. It can detect any shape that you can repersent mathamaticly
  
Step 6:

  Combine the Mathamatical repersentations of the lines with the original photo
  
  You could skip this step, and use the mathamatical repersnetation of teh data elsewhere (ex. nural network for steering)

  
  
