Using OpenCV for Python, I developed four adjustable instagram-style image filters making use of a variety of fundamental image processing techniques. 


The functions can be imported into a Python environment with the command

> from cmcz82 import *


Alternatively, the functions can be imported individually as follows:

> from cmcz82 import problem1
> from cmcz82 import problem2
> from cmcz82 import problem3
> from cmcz82 import problem4


All the functions can be run using default parameters as follows:

> problem1(image_path=IMAGE_PATH)

Where IMAGE_PATH is a string representing the location of an image file, for instance:

> problem1(image_path='./face1.jpg')




Every function comes with a set of adjustable parameters, which can be inputted in any order. The following is an exhaustive list of all adjustable parameters for each function.




# -------- FILTER 1: Light-leak/rainbow

 - rainbow: This must be a Bool. When set to True, the rainbow effect will be applied to the light leak. By default this is False.

 - darkening_coefficient: This must be a float between 0 and 1. Higher values indicate lighter backgrounds. By default this is set to 0.3.

 - blending_coefficient: This must be a float between 0 and 1. Higher values denote greater levels of blending. By default this is set to 0.6.

 - brightening_coefficient: This must be a float between 0 and 1. Higher values denote greater maximum brightness of the light leak. By default this is set to 0.5.

 - rainbow_strength: This must be a float between 0 and 1. Higher values denote higher weightings given to the rainbow when applied to the image. By default this is set to 0.4.

 - filter_centre: This must be a float between 0 and 1. Higher values move the filter to the right. By default this is set to 0.5 (i.e., the centre of the image).



--- EXAMPLE USAGES

> problem1(image_path='./face1.jpg', rainbow=False, darkening_coefficient=0.4, blending_coefficient=0.7)

> problem1(image_path='./face2.jpg', rainbow=True, brightening_coefficient=0.1, darkening_coefficient=0.9, blending_coefficient=0.8, rainbow_strength=0.5, filter_centre=0.6)

> problem1(image_path='./face1.jpg', rainbow=True, brightening_coefficient=0.2, darkening_coefficient=0.5, blending_coefficient=0.9, filter_centre=0.6, rainbow_strength=0.3)




# -------- FILTER 2: Sketch/charcoal 

 - colour: This must a be a string belonging to the set {'purple', 'blue', 'green', 'red'}. If a valid colour is used, the colour pencil mode is toggled. By default this is set to none (i.e., monochrome mode).

 - canvas_stroke_rate: This must be a float between 0 and 1. It determines the stroke rate of the canvas (i.e., the probability of noise during the Gaussian/salt and pepper noise construction phase). By default this is set to 0.4.

 - canvas_blending_coefficient: Must be a float between 0 and 1. Determines the relative strength of the canvas when blended with the image. By default this is 0.3.

 - canvas_stroke_width: Must be a float between 0 and 1. This determines the "width" of the strokes on the canvas (i.e., the amount of motion blur applied). By default this is 0.23 (which corellates to a motion blur kernel size of 23x23)

 - salt_and_pepper_mask: Must be a Bool. When set to True, salt and pepper noise is used to generate the mask in the monochrome filter. By default this is False (i.e., Gaussian noise is applied instead).



--- EXAMPLE USAGES

> problem2(image_path='./face2.jpg', canvas_stroke_rate=0.7, canvas_blending_coefficient=0.6, canvas_stroke_width=0.3)

> problem2(image_path='./face1.jpg', colour='blue', canvas_stroke_rate=0.6, canvas_blending_coefficient=0.5, canvas_stroke_width=0.2)

> problem2(image_path='./face2.jpg', colour='purple', canvas_blending_coefficient=0)




# -------- FILTER 3: Bilateral smoothing (beautification) and seasonal filters

 - smoothing_coefficient: Must be an int. Determines the window size for the bilateral smoothing. 0 will indicate no bilateral smoothing. By default this is set to 2 (which denotes a 5x5 window).

 - sub_filter: Can be set to either 'none' (indicating no sub filter is applied), 'late-spring', 'late-summer', or 'mid-autumn'. More details on these filters can be found on the report. 

 - contrast_intensity: Int between 0 and 64. Adjustable parameter for the 'late-summer' filter if applied. Denotes the intensity of the contrast applied to the image. By default this is 40. 

 - histogram: Bool that will toggle displaying a colour channel histogram before and after applying the filter. By default this is False (i.e., no histogram is shown).



--- EXAMPLE USAGES

> problem3(image_path='./face2.jpg', sub_filter='none', smoothing_coefficient=3)

> problem3(image_path='./face1.jpg', sub_filter='mid-autumn', smoothing_coefficient=1)

> problem3(image_path='./face2.jpg', sub_filter='late-spring', smoothing_coefficient=1)

> problem3(image_path='./face1.jpg', sub_filter='late-summer', contrast_intensity=40, smoothing_coefficient=2) 




# -------- PROBLEM 4: Face whirl!

 - radius: Must be a positive integer. Determines the radius of the swirl effect. By default this is 160.

 - strength: Positive or negative integer that determines the strength of the swirl effect. Positive values denote a clockwise swirl and negative values denote an anticlockwise swirl. For instance, 2.5 will denote a maximum swirl of 90 degrees clockwise. By default this is set to -2.5.  

 - centre: A tuple containing two ints that will represent the centre of the swirl effect. The co-ordinates must be valid in the image. By default this will be automatically set to the centre of the input image. 

 - bilinear_interpolation: This is a bool which, if set to True, will cause bilinear interpolation to be applied to the pixel mapping. If this is set to False, nearest neighbour interpolation will be used instead. This is set to True by default.

 - prefilter: This is a bool which, if set to True, will cause a Butterworth low-pass filter to be applied to the image prior to the swirl. By default this is set to False.

 - cut_off_freq: Must be a positive integer. Determines the cut-off frequency of the Butterworth low-pass filter if applied. By default this is set to 70.

 - order: Must be a small positive integer. Determines the order of the Butterworth low-pass filter if applied. By default this is set to 1.



--- EXAMPLE USAGES

> problem4(image_path='./face2.jpg', radius=160, strength=-2.5, prefilter=False, bilinear_interpolation=True)

> problem4(image_path='./face1.jpg', radius=75, strength=-2)

> problem4(image_path='./face2.jpg', radius=120, strength=3)

> problem4(image_path='./face2.jpg', radius=200, strength=-4, prefilter=True)

> problem4(image_path='./face1.jpg', radius=90, strength=1, prefilter=True, bilinear_interpolation=False, cut_off_freq=100)

> problem4(image_path='./face1.jpg', radius=90, strength=1.8, prefilter=True, bilinear_interpolation=True, cut_off_freq=300, order=3)




