.. HECATE documentation master file, created by
   sphinx-quickstart on Thu Dec 18 17:36:15 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HECATE
======

.. grid:: 1 2 2 2

   .. grid-item::

      *Hecate is a goddess in ancient Greek religion and mythology, most often shown holding a pair of torches, a key, or snakes, 
      or accompanied by dogs, and in later periods depicted as three-formed or triple-bodied. Hecate is often associated with 
      illuminating what is hidden and find your way in cross-roads.*

   .. grid-item::

      .. image:: HECATE_logo_white_color.png
         :align: center
         :width: 90%

Context
-------

As the number of known exoplanets grows, the focus has shifted toward characterizing these worlds and their atmospheres. 
However, their faint observational signals are often hindered by effects arising from the host star. 
The stellar spectrum is not homogeneous across the surface of the star, due to center-to-limb variations 
(CLV), magnetic activity, etc. 
These inhomogeneities alter the spectral content of local stellar regions and can bias the absorption features observed in a 
planet's transmission spectrum, complicating atmospheric retrievals and potentially leading to misinterpretations of 
planetary properties. 
Currently, models are used to correct the transmission spectra from these issues, but we lack the certainty of their quality. 
This way, we can use the Doppler Shadow method to extract local stellar spectra and compare them to models.

The “Doppler Shadow” method allows us to probe local regions of the stellar surface. 
As a planet transits its host star, it sequentially blocks different regions of the stellar surface, causing variations in 
the observed stellar spectrum. 
The missing spectrum of the star during the transit is recovered by subtracting the flux weighted in-transit spectrum 
from a reference spectrum (usually the average out-of-transit observation). 
This allows us to recover what is usually referred to as the shadow spectra.
Accessing the local spectra of the star also allows us to better understand the behavior of the distortions in 
transmission spectra, improving the reliability of atmospheric characterization.

What is HECATE?
---------------

HECATE (HarvEsting loCAl specTra with Exoplanets) consists of a robust, modular and flexible analysis pipeline capable of 
extracting spectra occulted by transiting exoplanets -- the Doppler Shadow technique. 
This tool introduces automation and is applicable to both individual spectral lines and combinations of lines 
(using the cross-correlation technique), significantly improving reproducibility and scalability.

HECATE will support the design of future observations and expand our understanding of how the stellar signal can affect the 
extraction and interpretation of planetary atmospheric signals.

The work of Gonçalves et al. (2026) served as a blueprint for HECATE.

Under active development.

While HECATE is designed to be adaptable to various spectrographs, it has been primarily developed and tested using data 
from the ESPRESSO spectrograph.

Documentation
--------------------

.. toctree::
   :maxdepth: 2

   documentation
   api



License & Attribution
---------------------

HECATE is being developed in a
`public GitHub repository <https://github.com/telmonteiro/HECATE>`_.
