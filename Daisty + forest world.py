
# Imports
import numpy as np
import matplotlib.pyplot as plt

REVERSE = False
# Temperatures (Origional scouce code (Bennett et al., 2017))
KELVIN_OFFSET = 273.15

#(Adjusted by student, origional by (Bennett et al., 2017))
Td_conif_min = 0 + KELVIN_OFFSET
Td_conif_max = 35 + KELVIN_OFFSET

Td_decid_min = 3 + KELVIN_OFFSET
Td_decid_max = 40 + KELVIN_OFFSET

Td_trop_min = 5 + KELVIN_OFFSET
Td_trop_max = 45 + KELVIN_OFFSET

Td_ideal_trop = 23.5 + KELVIN_OFFSET
Td_ideal_conif = 21.5 + KELVIN_OFFSET
Td_ideal_decid = 22.5 + KELVIN_OFFSET


# Albedo (Adjusted by student, origional by (Bennett et al., 2017))

insul = 20
drate = 0.3

alb_trop = 0.13
area_trop = 0.01

alb_conif = 0.15
area_conif = 0.01

alb_decid = 0.20
area_decid = 0.01

alb_snice = 0.80
area_snice = 0.13

alb_sand = 0.45
area_sand = 0.08

alb_graslan = 0.26
area_graslan = 0.8

# Convergence criteria (Origional scouce code (Bennett et al., 2017))
maxconv = 1000
tol = 0.000001

# Flux terms (Origional scouce code (Bennett et al., 2017))
So = 1000
sigma = 5.67032e-8

# Flux limits and step (Origional scouce code (Bennett et al., 2017))
Sflux_min = 0.5
Sflux_max = 1.6
Sflux_step = 0.002

# If run from command line, do the whole thing 
if __name__ == '__main__':
    """Run the daisyworld model"""
    # Initialize arrays (Adjusted by student, origional by (Bennett et al., 2017))
    fluxes = np.arange(Sflux_min, Sflux_max, Sflux_step)
    if REVERSE:
        fluxes = fluxes[::-1]
    area_trop_vec = np.zeros_like(fluxes)
    area_conif_vec = np.zeros_like(fluxes)
    area_decid_vec = np.zeros_like(fluxes)
    area_snice_vec = np.zeros_like(fluxes)
    area_sand_vec = np.zeros_like(fluxes)
    area_graslan_vec = np.zeros_like(fluxes)
    Tp_vec = np.zeros_like(fluxes)

    # Loop over fluxes (Adjusted by student, origional by (Bennett et al., 2017))
    for j, flux in enumerate(fluxes):
        # Minimum daisy coverage
        if area_trop < 0.01:
            area_trop = 0.01
        if area_conif < 0.01:
            area_conif = 0.01
        if area_decid < 0.01:
            area_decid = 0.01
        area_graslan = 0.8 - (area_trop + area_conif + area_decid)

        # Reset iteration metrics (Adjusted by student, origional by (Bennett et al., 2017))
        it = 0
        dA_trop = 2*tol
        dA_conif = 2*tol
        dA_decid = 2*tol
        darea_trop_old = 0
        darea_conif_old = 0
        darea_decid_old = 0

        while it <= maxconv and dA_trop > tol and dA_conif > tol and dA_decid > tol:
            # Planetary albedo (Adjusted by student, origional by (Bennett et al., 2017))
            alb_p = (area_trop * alb_trop
                     + area_conif * alb_conif
                     + area_decid * alb_decid
                     + area_graslan * alb_graslan
                     + area_snice * alb_snice
                     + area_sand * alb_sand)
            
            # Planetary temperature (Origional scouce code (Bennett et al., 2017))
            Tp = np.power(flux*So*(1-alb_p)/sigma, 0.25)
            
            # Local temperatures (Adjusted by student, origional by (Bennett et al., 2017))
            Td_trop = insul*(alb_p-alb_trop) + Tp
            Td_conif = insul*(alb_p-alb_conif) + Tp
            Td_decid = insul*(alb_p-alb_decid) + Tp
            Td_graslan = insul*(alb_p-alb_graslan) + Tp
            Td_snice = insul*(alb_p-alb_snice) + Tp
            Td_sand = insul*(alb_p-alb_sand) + Tp

            # Determine birth rates (Adjusted by student, origional by (Bennett et al., 2017))
            if (Td_trop >= Td_trop_min
                    and Td_trop <= Td_trop_max
                    and area_trop >= 0.01):
                birth_trop = 1 - 0.003265*(Td_ideal_trop-Td_trop)**2
            else:
                birth_trop = 0.0

            if (Td_conif >= Td_conif_min
                    and Td_conif <= Td_conif_max
                    and area_conif >= 0.01):
                birth_conif = 1 - 0.003265*(Td_ideal_conif-Td_conif)**2
            else:
                birth_white = 0.0
            
            if (Td_decid >= Td_decid_min
                    and Td_decid <= Td_decid_max
                    and area_decid >= 0.01):
                birth_decid = 1 - 0.003265*(Td_ideal_decid-Td_decid)**2
            else:
                birth_decid = 0.0

            # Change in areal extents (Adjusted by student, origional by (Bennett et al., 2017))
            darea_trop = area_trop*(birth_trop*area_graslan-drate)
            darea_conif = area_conif*(birth_conif*area_graslan-drate)
            darea_decid = area_decid*(birth_decid*area_graslan-drate)

            # Change from previous iteration (Adjusted by student, origional by (Bennett et al., 2017))
            dA_trop = abs(darea_trop-darea_trop_old)
            dA_conif = abs(darea_conif-darea_conif_old)
            dA_decid = abs(darea_decid-darea_decid_old)

            # Update areas, states, and iteration count (Adjusted by student, origional by (Bennett et al., 2017))
            darea_trop_old = darea_trop
            darea_config_old = darea_conif
            darea_decid_old = darea_decid
            area_trop = area_trop+darea_trop
            area_conif = area_conif+darea_conif
            area_decid = area_decid+darea_decid
            area_graslan = 0.8-(area_trop+area_conif+area_decid)
            it += 1

        # Save states (Adjusted by student, origional by (Bennett et al., 2017))
        area_trop_vec[j] = area_trop
        area_conif_vec[j] = area_conif
        area_decid_vec[j] = area_decid
        area_snice_vec[j] = area_snice
        area_sand_vec[j] = area_sand
        area_graslan_vec[j] = area_graslan
        Tp_vec[j] = Tp
    
    # Plot data (Adjusted by student, origional by (Bennett et al., 2017))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(fluxes, 100*area_trop_vec, color='orange', label='Orange')
    ax[0].plot(fluxes, 100*area_conif_vec, color='lime', label='Lime')
    ax[0].plot(fluxes, 100*area_decid_vec, color='green', label='Green')
    ax[0].plot(fluxes, 100*area_graslan_vec, color='red', label='red')
    ax[0].plot(fluxes, 100*area_snice_vec, color='blue', label='Blue')
    ax[0].plot(fluxes, 100*area_sand_vec, color='yellow', label='Yellow')
    ax[0].set_xlabel('solar luminosity')
    ax[0].set_ylabel('area (%)')
    plt.legend()

    ax[1].plot(fluxes, Tp_vec-KELVIN_OFFSET, color='black')
    ax[1].set_xlabel('solar luminosity')
    ax[1].set_ylabel('global temperature (C)')
    plt.show()