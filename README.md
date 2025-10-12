# ColdFirnAquifer3D : A Vertically Integrated Model with Phase Change for Aquifers in Cold Firn

## Overview

This project develops a **vertically integrated model** that captures the coupled **phase change and flow dynamics** governing meltwater migration in **cold firn**, the sintered and compacted snow on glaciers and ice sheets.

The model bridges the physics of **gravity-driven saturated flow in soils** with **meltwater percolation and refreezing in firn**, offering a unified theoretical framework for hydrologic processes across temperate and polar environments. Unlike traditional approaches, this model explicitly includes **latent heat effects**, **porosity evolution**, and **residual liquid trapping**.


---

## Authors
- **Mohammad Afzal Shadab**<sup>1,2</sup> ⭐ (mashadab@princeton.edu)  
- **Howard A. Stone**<sup>3</sup>  
- **Reed M. Maxwell**<sup>1,2,4</sup>  

⭐ = corresponding author  

---

## Affiliations
1. Department of Civil and Environmental Engineering, Princeton University, Princeton, NJ 08544, USA  
2. Integrated Groundwater Modeling Center, Princeton University, Princeton, NJ 08544, USA  
3. Department of Mechanical and Aerospace Engineering, Princeton University, Princeton, NJ 08544, USA  
4. High Meadows Environmental Institute, Princeton University, Princeton, NJ 08544, USA  

---

## Citation
If you use this code or data, please cite:  
**Shadab, M.A., Stone, H.A., and Maxwell, R.M. (202X).** *A vertically integrated model with phase change for aquifers in cold firn* (under review).

---

## Getting Started

### Quick Usage
After cloning the repository and installing the required Python libraries, the main scripts can be executed directly in the `\src\Figure.py`
All figures will be saved automatically in the `\Figures` directory.

---

## Model Description

The model solves a **vertically integrated flow equation with latent heat coupling**, allowing for:
- Phase change between liquid water and ice  
- Porosity evolution and residual water trapping  
- Analytical and numerical comparisons  
- Expansion of firn aquifers in cold, heterogeneous conditions  

The code includes benchmark cases and plotting routines to reproduce figures from the manuscript.

---

## Example Results

### Analytical vs Numerical Solutions  
![Analytical vs Numerical](./Figures/Cover_photo/analyticalvsnumerical.png)

### 3D Aquifer Expansion in Cold Firn  
![3D Aquifer Expansion](./Figures/Aquifer_3D_Expansion.png)

---
