import os
import numpy as np
import SimpleITK as sitk
import scipy.stats as stats
import edt

MYO_ID = 1 
RV_ID = 5
AORTA_ID = 6

## ct image info functions 
def is_RV_contrast_image(image, whs, n_iter=100):
    # Input: image = 3D CT image, whs = whole heart segmentation, n_iter = number of iterations for t-test
    # Output: k = number of t-tests resulted in significantly equal myocardium and RV HU-values out of n_iter iterations
    # The function randomly samples 100 HU-values from the myocardium and RV and performs a t-test to check if the two samples are significantly different.
    myo_huvs = image[whs==MYO_ID]
    rv_huvs = image[whs==RV_ID]

    N_accept = 0
    for _ in range(n_iter):
        sample_myo = np.random.choice(a=myo_huvs, size=100, replace=False)
        sample_rv = np.random.choice(a=rv_huvs, size=100, replace=False)
        ttest = stats.ttest_ind(a=sample_myo, b=sample_rv, equal_var=False)
        if ttest.pvalue > 0.05:
            N_accept += 1
    
    return N_accept


def check_quality(image, whs, spacing):
    # Input: image = 3D CT image, whs = whole heart segmentation, spacing = (z_spacing, x_spacing, y_spacing)
    # Output: snr = signal-to-noise ratio, cnr = contrast-to-noise ratio, cv = coefficient of variation
    # The cnr is calculated as the difference between the mean HU-value of the aorta core and the median HU-value of the myocardium,
    # divided by the standard deviation of the aorta HU-values. Aka how big is the different between blood and muscle in "noise units". 
    # Eksempel: (400 − 100) / 30 = 10. Forskellen er 10 gange større end støjen, så blod og muskel er lette at skelne.
    # If it is a perfect picture then all the blood in aorta_huvs will have the same value and the standard deviation will be 0, so the cnr will be infinite meaning perfect contrast.

    sdf_arota = -edt.sdf(whs==AORTA_ID, anisotropy=spacing, parallel=8)
    slim_aorta = sdf_arota < -5
    aorta_huvs = image[slim_aorta]

    signal_bg = np.median(image[whs==MYO_ID])
    signal = np.mean(aorta_huvs, dtype=np.float64)
    noise = np.std(aorta_huvs, dtype=np.float64)
    
    snr = signal/noise                      # signal-to-noise ratio
    cnr = (signal - signal_bg) / noise      # contrast-to-noise ratio
    cv = noise/signal                       # Variation 

    return snr, cnr, cv

def main():
    # Get parameters for one CT-scan from file paths 
    image_path = 'data/1_200/1.img.nii.gz'
    whole_heart_segmentation_path = 'data/TotalSegmentator/1.heart.nii.gz'

    image_sitk = sitk.ReadImage(image_path)
    spc_x, spc_y, spc_z = image_sitk.GetSpacing()
    image = sitk.GetArrayFromImage(image_sitk)

    whs_sitk = sitk.ReadImage(whole_heart_segmentation_path)
    whs = sitk.GetArrayFromImage(whs_sitk)

    # k = #t-tests resulted in significantly equal myocardium and RV HU-values out of n_iter iterations 
    # So if 
    #   k = 0 : probably contrast between RV and myocardium 
    #   k > 0 : evidence of limited contrast between RV and myocardium, the larger k the larger signal towards no-contrast between myocardium and RV 
    # I found that even k = 1 shows that the contrast between RV and myocardium is non-existing (or very limited) (with n_iter = 100) in HØ1 
    # Sometimes the function does not catch all limited-contrast CT scans so the group 'k=0' might contain CT-scans with very limited contrast as well
    k = is_RV_contrast_image(image, whs, n_iter=100)
    print(f'#t-tests showing equal RV and myocardium HU-values: {k}')

    # Contrast-to-noise ratio (CNR): float number
    #   low CNR  : "bad" quality scan 
    #   high CNR : "good" quality scan
    # I would plot a histogram of the CNR-distribution for your dataset and estimate a good cut-off for CNR between low/high quality scans 
    # For HØ1 I used; CNR < 5  to indicate low quality scan 
    _,cnr, _ = check_quality(image, whs, spacing=(spc_z, spc_x, spc_y))
    print(f'Contrast-to-noice ratio = {cnr}')


if __name__ == '__main__':
    main()