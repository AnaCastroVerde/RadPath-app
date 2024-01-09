# RadPath-app
Python app for 2D registration of a structural Magnetic Resonance Imaging (MRI) slice and a corresponding histopathology slide.

This repository includes the Python algorithm developed for multimodal 2D registration and a folder with test data collected at the Champalimaud Clinical Center. The purpose of this repository is to align a histopathology slide (moving image) with the corresponding T2-weighted MRI slice (fixed image). One application of increasing the correspondence between modalities is to perform biological validation of the radiomics features extracted from the MRI with the radical prostatectomy ground-truth.
 
### Repository Structure
- RadPathReg.py: Python algorithm for 2D multimodal registration.
- test-data: Folder containing the required inputs:
  - 'fixed_original.nii.gz': Fixed T2-weighted MRI image (3D image).
  - 'fixed_mask.nii.gz': Binary fixed mask (Segmentation drawn on the 3D fixed image).
  - 'moving.tif': Moving histopathology image (2D image).
- requirements.txt: Text file with the necessary Python packages for this algorithm.

### Compatibility
All files need to be provided on the same folder and need to be in NIfTI (.nii) and TIFF (.tif) formats, respectively for the MRI and histopathology images.

### Usage
This algorithm was created to be run as a streamlit application.

1) Clone this repository to your local folder;

2) Install the necessary Python packages under requirements.txt;

3) Go to your local folder and run the app using the command line:
`$ streamlit run RadPathReg.py`

4) On the app, upload the test data or your own data, define the local path to the data, modify the index slice number and rotation angle (if necessary), and press "Start registration".

### Acknowledgements
This study was approved by the Champalimaud Foundation Ethics Commitee, under the ProCAncer-I project. This work was funded by the European Union’s Horizon 2020 research and innovation programme (grant 952159) and by Fundação para a Ciência e Tecnologia UIDB/00645/2020 and https://doi.org/10.54499/UIDB/00645/2020.
This repository was created by Ana Sofia Castro Verde. For questions, please contact ana.castroverde@research.fchampalimaud.org.