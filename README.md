# Software for the Meridian Imaging Svalbard Spectrographs (KHO/UNIS)
This repository will contain all various components and the software for the new Meridian Imaging Svalbard Spectrographs (I and II for now). 
The intention is to gather all necessary tools for the smooth control of the imager from image capture to image processing (RGB composite images, keogram update, spectrogram analysis...)

MISS2 (software/hardware) is directly adapted from [MISS](https://kho.unis.no/Instruments/MISS.html), a spectrograph operational since 2019 at the [Kjell Henriksen Observatory](https://kho.unis.no/), which is operated by the [University Centre In Svalbard](https://www.unis.no/).

The software was updated in August 2024 to handle MISS1 and MISS2 using respective calibration data from July 2024. It was optimised to allow the user to switch from an imager to the other with minimal changes and with a parameter.py script gathering the main variables of interest. to ensure that the imager is operated as expected (Main-script: MISS-CONTROLLER.py) and that the collected data is being processed and updated in real-time (Main-script: REAL-TIME_MISS-DATA.py) on KHO's website (https://kho.unis.no/Data.html).


# SCRIPT TO EXECUTE FOR OPERATION OF MISS: MISS-CONTROLLER-py

# SCRIPT TO EXECUTE FOR REALTIME SPECTROGRAM AND KEOGRAM UPDATES: REAL-TIME_MISS-DATA.py

# Create a Keogram using past data from YYYY/MM/DD: past-keogram-maker.py
