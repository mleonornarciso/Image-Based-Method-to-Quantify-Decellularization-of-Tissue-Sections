# Image-Based-Method-to-Quantify-Decellularization-of-Tissue-Sections
Pipeline for image processing method described in the published manuscript "Image-Based Method to Quantify Decellularization of Tissue Sections". 
Full text: https://www.mdpi.com/1422-0067/22/16/8399
For more information or questions, please contact mnarciso@ibecbarcelona.eu

## Summary
The objective of this work is to quantify the DNA signal of native and decellularized tissue. The problem arises when the DNA signal image cannot be properly segmented due to high levels of decellularizetion (which in turn lead to low signal levels). To resolve this issue, we mask the phase contrast image and apply it to the corresponding uv image.
This algorithm takes corresponding images (phase contrast + uv image) and uses an entropy filter to mask the phase contrast image. 
This mask is improved with dilations, erosions, labeling and component selections. The final mask is applied to the uv image to obtain a value for the signal intensity of the uv image. 

## Repository content
img_processing.py is the library where the functions are stored. 
pipeline.py is where experiments should be run.
We recommend the use of jupyter notebooks for pipeline.py for a better user experience.

## Instructions 
The code uploaded is for a single image pair (pc contrast image + uv image). 
Native and decellularized median signal intensities should both be computed to reach a decellularization % based on the difference of mean signal intensity.
To analyse a full dataset of images, we advise the creation of a dictionary based on the different experimental conditions as well as the incorporation of a simple for loop to iterate between image pairs.

## Citations
If you use this work, please cite:

>*MDPI and ACS Style*
>Narciso, M.; Otero, J.; Navajas, D.; Farré, R.; Almendros, I.; Gavara, N. Image-Based Method to Quantify Decellularization of Tissue Sections. Int. J. Mol. Sci. 2021, 22, 8399. https://doi.org/10.3390/ijms22168399

>*AMA Style*
>Narciso M, Otero J, Navajas D, Farré R, Almendros I, Gavara N. Image-Based Method to Quantify Decellularization of Tissue Sections. International Journal of Molecular Sciences. 2021; 22(16):8399. https://doi.org/10.3390/ijms22168399

>*Chicago/Turabian Style*
>Narciso, Maria, Jorge Otero, Daniel Navajas, Ramon Farré, Isaac Almendros, and Núria Gavara. 2021. "Image-Based Method to Quantify Decellularization of Tissue Sections" International Journal of Molecular Sciences 22, no. 16: 8399. https://doi.org/10.3390/ijms22168399 
