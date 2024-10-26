'''
This is where main parameters (PATHS and SETTINGS) are gathered for the user to change them from one and only place, affecting the entire MISS Software.

Author: Nicolas Martinez (LTU/UNIS)

Created: August 2024

'''

import os

parameters = {
    'device_name': 'MISS2',  # Update this if needed

    # Paths
    'raw_PNG_folder': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "spectrograms_PNG"),
    'averaged_PNG_folder': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "averaged_PNG"),
    'processed_spectrogram_dir': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "Processed_spectrograms"),
    'keogram_dir': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Keograms"),
    'feed_dir': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Feed"),
    'spectro_path': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "averaged_PNG"),
    'RGB_folder': os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "RGB_columns"),

    # Imaging Settings
    'exposure_duration': 12,  # 12 seconds for each image
    'optimal_temperature': 0,  # 0°C for cooling
    'imaging_cadence': 15,  # 15 seconds between image captures
    'binX': 2,  # Horizontal binning
    'binY': 2,  # Vertical binning

    # Spectral Coefficients [a_0, a_1, a_2] - Starting from last pixel row to first pixel row - spectrogram inversion caused by Artemis Software [in Ångström/pixel row]
    'miss1_wavelength_coeffs': [4217.273360, 2.565182, 0.000170],
    'miss2_wavelength_coeffs': [4088.509, 2.673936, 1.376154e-4],

    # Sensitivity correction coefficients for MISS1 and MISS2 [a_5, a_4, a_3, a_2, a_1, a_0] - Starting from last pixel row to first pixel row - spectrogram inversion caused by Artemis Software [in R/Å]
'coeffs_sensitivity': {
    'MISS1': [-1.677600e-17, -3.125710e-13, 1.743241e-09, 2.365830e-07, -2.935140e-02, 6.662786e+01],
    'MISS2': [-1.378573e-16, 4.088257e-12, -4.806258e-08, 2.802435e-04, -8.109943e-01, 9.329611e+02]
},

    # Horizon Limits
    'miss1_horizon_limits': (280, 1140),
    'miss2_horizon_limits': (271, 1116),

    # Add num_pixels_y and num_minutes to parameters.py
    'num_pixels_y': 300,  # Number of pixels along the y-axis (for RGB with 300 rows)
    'num_minutes': 24 * 60,  # Total number of minutes in a day
}
