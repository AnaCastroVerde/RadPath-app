## --------------------------------####### RAD-PATH REGISTRATION APP #######----------------------------------------- ##

## ------------------------------------------------ Import packages ------------------------------------------------- ##

import streamlit as st
import imageio.v2 as imageio
import itk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import corner_harris, corner_peaks
from skimage import color
import scipy.ndimage as ndi
from skimage import morphology
import cv2
import SimpleITK as sitk
import math
import time
import matplotlib.pyplot as plt

## ------------------------------------------------- Accept inputs -------------------------------------------------- ##

st.title("RadPathReg")
st.subheader("2D registration script of an MRI slice and a corresponding histopathology slide.")
st.write("Please upload the images to register:")

fixed = st.file_uploader("Fixed T2-weighted MR image (3D image):")
fixed_name = fixed.name
fixed_mask = st.file_uploader("Binary fixed mask (Segmentation drawn on the 3D fixed image):")
fixed_mask_name = fixed_mask.name
moving = st.file_uploader("Moving histopathology image (2D image):")
moving_name = moving.name
path = st.text_input("Insert file path (e.g. /Users/username/Documents/folder/):", "")
index_slice = st.slider("Index slice (number of the corresponding slice on MRI):", 0, 50, 10)
angle = st.slider("Rotation angle (angle to rotate automated points in counterclockwise orientation):", 0, 360, 0)
preprocessing = st.checkbox("Apply preprocessing")

## ------------------------------------------------ Do registration ------------------------------------------------- ##

if (fixed is not None)& (fixed_mask is not None)& (moving is not None)& (index_slice is not None)& (path is not None):

    if st.button("Start registration") == True:
        start_time = time.time()
        ## --------------------------------------- Convert 3D to 2D mask ---------------------------------------- ##
        mask_3d = itk.imread(path + fixed_mask_name, itk.F)
        Dimension = mask_3d.GetImageDimension()
        indx_slice = int(index_slice)
        extractFilter = itk.ExtractImageFilter.New(mask_3d)
        extractFilter.SetDirectionCollapseToSubmatrix()

        # set up the extraction region [one slice]
        inputRegion = mask_3d.GetBufferedRegion()
        size = inputRegion.GetSize()
        size[2] = 1  # we extract along z direction
        start = inputRegion.GetIndex()
        sliceNumber = indx_slice
        start[2] = sliceNumber

        RegionType = itk.ImageRegion[Dimension]
        desiredRegion = RegionType()
        desiredRegion.SetIndex(start)
        desiredRegion.SetSize(size)

        extractFilter.SetExtractionRegion(desiredRegion)
        extractFilter.Update()
        mask_2d = extractFilter.GetOutput()

        itk.imwrite(mask_2d, path + 'mask.nii.gz')
        mask_2d = sitk.ReadImage(path + 'mask.nii.gz', sitk.sitkFloat32)[:, :, 0]
        sitk.WriteImage(mask_2d, path + 'mask_2d.nii.gz')

        fixed_mask_2d = itk.imread(path + 'mask_2d.nii.gz', itk.UC)

        ## -------------------------------------- MRI data preprocessing ---------------------------------------- ##
        if preprocessing == True:
            with st.spinner('Bias field correction and intensity normalization in progress...'):
                #1) Bias field correction
                image_T2_0001 = sitk.ReadImage(path + fixed_name)
                image_T2_0001_float = sitk.Cast(image_T2_0001, sitk.sitkFloat32) #Convert image type from short to float
                bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
                bias_field_filter.SetNumberOfControlPoints([4,4,4])
                image_T2_0001_bias_corrected = bias_field_filter.Execute(image_T2_0001_float)
                sitk.WriteImage(image_T2_0001_float, path + 'image_T2_001_original.nii.gz')
                sitk.WriteImage(image_T2_0001_bias_corrected, path +  'image_T2_001_bias.nii.gz')

                #2) Intensity normalization, using the Z-score method
                Zscore_filter = sitk.NormalizeImageFilter()
                image_T2_0001_norm = Zscore_filter.Execute(image_T2_0001_bias_corrected)
                sitk.WriteImage(image_T2_0001_norm, path + 'image_T2_001_norm.nii.gz')
                image_T2_0001_norm = itk.imread(path + 'image_T2_001_norm.nii.gz', itk.F)
        else:
            st.write("The original fixed image is used for registration.")
            image_T2_0001_norm = itk.imread(path + fixed_name, itk.F)

        with st.spinner('Registration in progress...'):
            ## ------------------------------------ Extract 2D slice from volume ------------------------------------ ##

            Dimension = image_T2_0001_norm.GetImageDimension()
            indx_slice = int(index_slice)
            extractFilter = itk.ExtractImageFilter.New(image_T2_0001_norm)
            extractFilter.SetDirectionCollapseToSubmatrix()

            # set up the extraction region [one slice]
            inputRegion = image_T2_0001_norm.GetBufferedRegion()
            size = inputRegion.GetSize()
            size[2] = 1  # we extract along z direction
            start = inputRegion.GetIndex()
            sliceNumber = indx_slice
            start[2] = sliceNumber

            RegionType = itk.ImageRegion[Dimension]
            desiredRegion = RegionType()
            desiredRegion.SetIndex(start)
            desiredRegion.SetSize(size)

            extractFilter.SetExtractionRegion(desiredRegion)
            extractFilter.Update()
            fixed_slice = extractFilter.GetOutput()

            itk.imwrite(fixed_slice, path + 'fixed_slice.nii.gz')
            fixed_image = sitk.ReadImage(path + 'fixed_slice.nii.gz', sitk.sitkFloat32)[:, :, 0]
            sitk.WriteImage(fixed_image, path + 'fixed_2d.nii.gz')

            ## ------------------------------------- Automated control points -------------------------------------- ##

            ## Get image characteristics

            fixed = itk.imread(path + 'fixed_2d.nii.gz', itk.F)
            region_f = fixed.GetLargestPossibleRegion()
            size_f = region_f.GetSize()
            center_f = region_f.GetIndex()
            origin_f = fixed.GetOrigin()
            spacing_f = fixed.GetSpacing()

            moving = itk.imread(path + moving_name, itk.F)
            original_spacing = (0.5, 0.5)
            new_spacing = (2**5)*original_spacing[0]*(10**(-3))
            moving = itk.image_from_array(moving, is_vector=False)
            moving.SetSpacing(new_spacing)
            region_m = moving.GetLargestPossibleRegion()
            size_m = region_m.GetSize()
            center_m = region_m.GetIndex()
            origin_m = moving.GetOrigin()
            spacing_m = moving.GetSpacing()
            print("Fixed size", size_f)
            print("Fixed spacing", spacing_f)
            print("Fixed origin", origin_f)
            print("Moving size", size_m)
            print("Moving spacing", spacing_m)
            print("Moving origin", origin_m)

            ## Read moving image

            im_moving = imageio.imread(path + moving_name, format='tiff')
            im_moving = color.rgb2gray(im_moving)

            ## Segment moving image based on threshold

            thresh_moving = threshold_otsu(im_moving)
            im_thresh_moving = im_moving < thresh_moving
            im_thresh_filtered_moving = ndi.median_filter(im_thresh_moving, size = 10)
            moving_mask = morphology.binary_closing(im_thresh_filtered_moving, footprint=morphology.square(20))
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(im_thresh_filtered_moving, cmap="gray")
            axs[1].imshow(moving_mask, cmap="gray")
            axs[0].axis("off")
            axs[1].axis("off")
            st.pyplot(fig)
            moving_mask = sitk.GetImageFromArray(moving_mask.astype(int))
            sitk.WriteImage(moving_mask, path + 'moving_mask.nii.gz')

            ## Read fixed image

            im_fixed = imageio.imread(path + 'mask_2d.nii.gz', format='nii')
            im_fixed_flip = cv2.flip(im_fixed, 0)

            ## Segment fixed image based on threshold

            thresh_fixed = threshold_otsu(im_fixed)
            im_thresh_fixed = im_fixed > thresh_fixed
            thresh_fixed_flip = threshold_otsu(im_fixed_flip)
            im_thresh_fixed_flip = im_fixed_flip > thresh_fixed_flip
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(im_thresh_fixed, cmap="gray")
            axs[1].imshow(im_thresh_fixed_flip, cmap="gray")
            axs[0].axis("off")
            axs[1].axis("off")

            ## Corner peaks detection

            coords_fixed = corner_peaks(corner_harris(im_thresh_fixed), min_distance = 5, num_peaks = 10,
                                        threshold_rel = 0.01)
            coords_fixed_flip = corner_peaks(corner_harris(im_thresh_fixed_flip), min_distance = 5, num_peaks = 10,
                                             threshold_rel = 0.01)

            ## Obtain automated scaling factor for points

            moving_mask_pts = sitk.GetImageFromArray(im_thresh_filtered_moving.astype(int))
            fixed_mask_pts = sitk.GetImageFromArray(im_thresh_fixed_flip.astype(int))

            # Generate label and compute the Feret diameter - longest diameter of the mask
            filter_label = sitk.LabelShapeStatisticsImageFilter()
            filter_label.SetComputeFeretDiameter(True)
            filter_label.Execute(moving_mask_pts)
            feret_moving = filter_label.GetFeretDiameter(1)
            com_y, com_x = filter_label.GetCentroid(1)
            filter_label.Execute(fixed_mask_pts)
            feret_fixed = filter_label.GetFeretDiameter(1)
            com_y_f, com_x_f = filter_label.GetCentroid(1)

            scaling_factor = feret_moving/feret_fixed

            # Point rotation function

            def rotate(origin, point, angle):
                """
                Rotate a point counterclockwise by a given angle around a given origin.

                The angle should be given in radians.
                """
                ox, oy = origin
                px, py = point

                qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
                return qx, qy

            ## Map fixed peaks to moving image

            p = coords_fixed_flip
            moving_center = np.array([com_x, com_y], dtype=float)
            fixed_center = np.array([com_x_f, com_y_f], dtype=float)
            fixed_origin = abs(np.array([origin_f[0],origin_f[1]], dtype=float))
            moving_origin = abs(np.array([origin_m[0],origin_m[1]], dtype=float))

            q1 = p + (moving_center-fixed_center)
            q = moving_center * (1 - scaling_factor) + q1 * scaling_factor

            ## Rotate points counterclockwise by a certain angle, for better matching

            rotated_points = []
            angle = int(angle)
            for point in q:
                rotated_point = rotate(moving_center, point, math.radians(angle))
                rotated_points.append(rotated_point)
            rotated_points = np.array(rotated_points)

            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(im_fixed_flip, cmap="gray")
            axs[1].imshow(im_moving, cmap="gray")
            axs[0].plot(p[:, 1], p[:, 0], color='cyan', marker='o',
                        linestyle='None', markersize=6)
            axs[0].plot(fixed_center[1], fixed_center[0], color='red', marker='o',
                        linestyle='None', markersize=6)
            axs[1].plot(moving_center[1], moving_center[0], color='red', marker='o',
                        linestyle='None', markersize=6)
            axs[1].plot(rotated_points[:, 1], rotated_points[:, 0], color='cyan', marker='o',
                        linestyle='None', markersize=6)
            axs[0].axis("off")
            axs[1].axis("off")
            st.pyplot(fig)

            ## Flip x with y for index coordinates
            coords_fixed_flipped = np.flip(coords_fixed, axis=1)
            q_flipped = np.flip(rotated_points, axis=1)

            ## Save control points coordinates
            np.savetxt(path + 'auto_coords_moving_B1_new.txt', q_flipped)
            np.savetxt(path + 'auto_coords_fixed_B1.txt', coords_fixed_flipped)

            # Add first two lines of the text files
            mov_txt = open(path + 'auto_coords_moving_B1_new.txt', "r")
            fline = "index\n"
            sline = str(len(q_flipped))+"\n"
            oline = mov_txt.readlines()
            oline.insert(0, fline)
            oline.insert(1, sline)
            mov_txt.close()
            mov_txt = open(path+'auto_coords_moving_B1_new.txt', "w")
            mov_txt.writelines(oline)
            mov_txt.close()

            fix_txt = open(path+'auto_coords_fixed_B1.txt', "r")
            fline = "index\n"
            sline = str(len(coords_fixed_flipped))+"\n"
            oline = fix_txt.readlines()
            oline.insert(0, fline)
            oline.insert(1, sline)
            fix_txt.close()
            fix_txt = open(path+'auto_coords_fixed_B1.txt', "w")
            fix_txt.writelines(oline)
            fix_txt.close()

            ## ------------------------------------------- Registration -------------------------------------------- ##

            ## Dilate original mask to get mask with surrounding areas - improves the registration results!

            PixelType = itk.UC
            Dimension = 2
            radius = 10

            ImageType = itk.Image[PixelType, Dimension]

            reader = itk.ImageFileReader[ImageType].New()
            reader.SetFileName(path+'mask_2d.nii.gz')

            StructuringElementType = itk.FlatStructuringElement[Dimension]
            structuringElement = StructuringElementType.Ball(radius)

            grayscaleFilter = itk.GrayscaleDilateImageFilter[
                ImageType, ImageType, StructuringElementType].New()
            grayscaleFilter.SetInput(reader.GetOutput())
            grayscaleFilter.SetKernel(structuringElement)

            writer = itk.ImageFileWriter[ImageType].New()
            writer.SetFileName(path+'dilated_mask_2d.nii.gz')
            writer.SetInput(grayscaleFilter.GetOutput())

            writer.Update()

            fixed_mask_dilated = itk.imread(path+'dilated_mask_2d', itk.UC)

            moving_mask = itk.imread(path+'moving_mask', itk.UC)
            moving_mask = itk.image_from_array(moving_mask, is_vector=False)
            moving_mask.SetSpacing(new_spacing)
            spacing_mask = moving_mask.GetSpacing()
            print("Mask spacing: ", spacing_mask)

            ## Define parameter map to perform registration

            parameter_object = itk.ParameterObject.New()
            parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')

            # INITIALIZATION PARAMETERS
            parameter_map_affine['AutomaticTransformInitialization'] = ['true']
            parameter_map_affine['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']

            parameter_map_affine['Registration'] = [
                'MultiMetricMultiResolutionRegistration']
            original_metric = parameter_map_affine['Metric']
            parameter_map_affine['Metric'] = [original_metric[0],
                                              'CorrespondingPointsEuclideanDistanceMetric']

            parameter_map_affine['AutomaticScalesEstimation'] = ['true']

            # IMAGE TYPES
            parameter_map_affine['FixedInternalImagePixelType'] = ['float']
            parameter_map_affine['FixedImageDimension'] = ['3']
            parameter_map_affine['MovingInternalImagePixelType'] = ['float']
            parameter_map_affine['MovingImageDimension'] = ['3']

            # OPTIMIZER
            parameter_map_affine['Optimizer'] = ['AdaptiveStochasticGradientDescent']
            parameter_map_affine['ASGDParameterEstimationMethod'] = ['DisplacementDistribution']

            # INTERPOLATOR
            parameter_map_affine['Interpolator'] = ['LinearInterpolator']

            # GEOMETRIC TRANSFORMATION
            parameter_map_affine['Transform'] = ['SimilarityTransform']

            # IMAGE PYRAMIDS
            parameter_map_affine['NumberOfResolutions'] = ['4']
            parameter_map_affine['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']
            parameter_map_affine['MovingImagePyramid'] = ['MovingShrinkingImagePyramid']
            parameter_map_affine['FixedImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']
            parameter_map_affine['MovingImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']

            # MASKS
            parameter_map_affine['ErodeMask'] = ['false']

            # SAMPLER
            parameter_map_affine['ImageSampler'] = ['RandomCoordinate']
            parameter_map_affine['NumberOfSpatialSamples'] = ['2000']
            parameter_map_affine['NewSamplesEveryIteration'] = ['true']
            parameter_map_affine['UseRandomSampleRegion'] = ['false']
            parameter_map_affine['MaximumNumberOfSamplingAttempts'] = ['5']

            parameter_object.AddParameterMap(parameter_map_affine)

            parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
            parameter_map_bspline['Transform'] = ['BSplineTransform']
            parameter_map_bspline['Metric'] = ['AdvancedMattesMutualInformation',
                                               'TransformBendingEnergyPenalty']

            parameter_map_bspline['NumberOfResolutions'] = ['4']
            parameter_map_bspline['FinalGridSpacingInPhysicalUnits'] = ['10.0', '10.0']
            parameter_map_bspline['GridSpacingSchedule'] = ['6.0', '6.0', '4.0', '4.0', '2.5', '2.5', '1.0', '1.0']

            parameter_object.AddParameterMap(parameter_map_bspline)

            ## Call registration function

            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed, moving,
                fixed_point_set_file_name = path+'auto_coords_fixed_B1.txt',
                moving_point_set_file_name = path+'auto_coords_moving_B1_new.txt',
                fixed_mask = fixed_mask_dilated,
                moving_mask = moving_mask,
                parameter_object=parameter_object,
                log_to_console=False)

            itk.imwrite(result_image, path + 'result_image.nii.gz')

        st.success("Registration finished!")
        end_time = time.time()
        elapsed_time = end_time - start_time

        ## --------------------------------- Compute registration accuracy (Dice) ---------------------------------- ##

        # Load the Images
        fixed_mask = itk.imread(path + 'mask_2d.nii.gz', itk.F)
        result_image = itk.imread(path + 'result_image.nii.gz', itk.F)

        # Binarize the result image, according to Ostu threshold
        out = itk.OtsuMultipleThresholdsImageFilter(result_image)

        # Smooth the image, to remove annotation points
        smooth = itk.MedianImageFilter(out)

        # Compute the Dice score between the two masks (fixed T2 and histopathology registered)
        intersection = np.logical_and(fixed_mask, smooth)
        dice = np.sum(intersection) * 2.0 / (np.sum(fixed_mask) + np.sum(smooth))

        col1, col2 = st.columns(2)
        col1.metric(label="Total registration time (in seconds):", value=int(elapsed_time))
        col2.metric(label="Dice similarity coefficient (DSC):", value=round(dice, 2))