from pygdsm import GlobalSkyModel16
import healpy as hp
#import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import os

frequency_mhz = 408.0

gsm = GlobalSkyModel16(freq_unit='MHz')
sky = gsm.generate(frequency_mhz)

#print("Number of pixels:", len(sky))
print("Frequency:", frequency_mhz, "MHz")
print("Minimum temperature:", sky.min(), "K")
print("Maximum temperature:", sky.max(), "K")
print("Mean temperature:", sky.mean(), "K")

hp.mollview(
    sky,
    title=f"GSM2016 at {frequency_mhz} MHz",
    unit="K",
    norm="log"
)

plt.show()

#gsm.view(logged=True)

def neg_pix(sky):
    negative = sky < 0
    idx = np.where(negative)[0]
	
    print("\nNegative values:")
    print(sky[idx])
    print("\nNegative values: min/max")
    print(sky[idx].min(), sky[idx].max())
	
    nside = hp.get_nside(sky)
    theta, phi = hp.pix2ang(nside,idx)
    gal_l = np.degrees(phi)
    gal_b = 90.0 - np.degrees(theta)
    
    print("\nGalactic coordinates of negative pixels:")
    
    for i in range(min(20, len(idx))):
        print(
        idx[i],
        f"l={gal_l[i]:.3f} deg",
        f"b={gal_b[i]:.3f} deg",
        f"T={sky[idx[i]]:.3f} K"
    )


