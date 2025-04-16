# E-field targeting v1.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15228154.svg)](https://doi.org/10.5281/zenodo.15228154)

This repository is used to solve optimization problems where a set of electric fields (E-fields) are superimposed to focus the E-field at a target location and in target direction. It is specifically designed for multi-coil transcranial magnetic stimulation where the E-fields are induced in the cortex usign a set of coils, but it could be used for other applications as well.

## Targeting in a spherical surface

Spherical targeting assumes E-fields are generated on a spherical surface, which may be used as an approximation of a more complex surface. The example_spherical.m script creates an example set of E-fields (akin to https://doi.org/10.1016/j.brs.2018.03.014) and solves the optimization problem for a few example targets.

![example_spherical_fields](<https://github.com/user-attachments/assets/7d29a86e-3df5-490f-8ba6-19e7e4b6e390>)
*Figure 1. A set of E-fields that need to be summed up with appropriate weightings to focus the E-field on target.*

![example_spherical_results](https://github.com/user-attachments/assets/54c452cf-d013-433e-9973-62a579234e5e)
*Figure 2. Results of the example script.*

## Targeting in a complex geometry

The example_complex_geom.m has an example of targeting in a general 3D surface, but example E-fields are not currently provided.

## Optimization constraints and objectives

Let $E = \{E¹,...,E^k\}$ be an $k$-sized set of E-fields, such that each $E^i$ is a matrix of shape (N x 3) with E-field values, and N is the number of vertices in a coordinate space $S$ with a matching shape (N x 3). The optimization parameters then consist of $k$ parameters $\{w¹,...,w^k\}$ which weigh the relative contributions of each E-field in the set to produce a total E-field as:\
$$E_{\text{norm}} = |\sum_{i=1}^k E^i w^i|$$\
We define $\hat{E}\_{\text{norm}}$ as $E_{\text{norm}}$ scaled to a maximum value of 1.

### Defining stimulation location
The optimization function has two options for defining the focus point of the E-field, i.e., the stimulation location $O_{loc}$: 'WCOG', which aims to place a weighted center of gravity to the target location, and 'Max', which aims to place the E-field maximum at the target location. The optimizers use WCOG by default.

**WCOG**\
$$O_{loc} = \frac{\sum_{n=1}^N S_n \hat{E}^q_{\text{norm, n}}}{\sum_{n=1}^N \hat{E}^q_{\text{norm, n}}}$$, where $q$ is a weighting factor (default: $q = 10$) to control how much pull E-field magnitude has on the center of gravity. 

**Max**\
$$O_{loc} = S_{\text{argmax } E_{\text{norm}}}$$

When using optimize_Efield_complex_geom.m with the WCOG setting, the coordinate space $S$ is projected onto a plane defined by the average vertex normal in the neighbourhood of the target location and mapped into a two dimensional subspace. This flattening of the coordinate space removes the depth component from the stimulation location calculation.

### Defining constraints
Given a target location $T_{loc}$ and direction $T_{dir}$ the optimization is constrained such that:\
$$|T_{loc} - O_{loc}| < \delta\_{loc}$$, and\
$$\phi < \delta\_{dir}$$, and\
$$|O_{loc} - S_{\text{argmax } E_{\text{norm}}}| < \delta\_{\text{max}}$$,\
where $\delta_{loc}$ is the maximum acceptable diffence between the target and realized stimulation locations, $\phi$ the vector angle between $T_{dir}$ and $O_{dir}$, $\delta_{dir}$ the maximum acceptable angle difference between the target and realized stimulation directions, and $\delta\_{\text{max}}$ the maximum acceptable distance between the stimulation location and the E-field maximum location. $O_{dir}$ is the direction of the E-field at the closest vertex from $O_{loc}$. The constraint thresholds $\delta\_{loc}$ and $\delta\_{dir}$ can be adjusted and default to 1 mm and 5°, respectively. $\delta\_{\text{max}}$ is 5 mm in the optimize_Efields_spherical.m and 10 mm in optimize_Efields_complex_geom.m.

### Minimizing the objective
When the optimization constraints are met, the best solution is chosen by minimizing the objective. The optimization function has two options for the objective: 'minEnergy' or 'Focality'.

**minEnergy**\
$$\text{minimize } f(w) =  \sum_{i=1}^k (\frac{w^i}{\text{max}(|w|)})^2 $$

**Focality**\
$$\text{minimize } f(w) = \sum_{n=1}^N \hat{E}^{2}_{\text{norm, n}}$$

#### Resticting E-field in specific regions
The optimization function has an option limit the E-field magnitude in specified locations $S'$ by including a penalty term to the objective function:

**minEnergy**\
$$\text{minimize } f(w) =  \sum_{i=1}^k (\frac{w^i}{\text{max}(|w|)})^2 + \text{mean}(\hat{E}\_{\text{norm, S'}})*10$$

**Focality**\
$$\text{minimize } f(w) = \sum_{n=1}^N \hat{E}^{2}\_{\text{norm, n}} + \sum_{m \in S'} \hat{E}^{2}\_{\text{norm, m}}$$
