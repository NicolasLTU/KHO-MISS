# Software for the Meridian Imaging Svalbard Spectrographs (KHO/UNIS)
This repository will contain all various components and the software for the new Meridian Imaging Svalbard Spectrographs (I and II for now). 
The intention is to gather all necessary tools for the smooth control of the imager from image capture to image processing (RGB composite images, keogram update, spectrogram analysis...)

MISS2 (software/hardware) is directly adapted from [MISS](https://kho.unis.no/Instruments/MISS.html), a spectrograph operational since 2019 at the [Kjell Henriksen Observatory](https://kho.unis.no/), which is operated by the [University Centre In Svalbard](https://www.unis.no/).

The software was updated in August 2024 to handle MISS1 and MISS2 using respective calibration data from July 2024. It was optimised to allow the user to switch from on imager to the other with minimal changes. All programs in capital letters shall be updated with relevant spectrograph name/parameters/paths/updates... to ensure that the imager is operated as expected (Main program: MISS-CONTROLLER.py) and that the collected data is being processed and updated in real-time on KHO's website (https://kho.unis.no/Data.html).
