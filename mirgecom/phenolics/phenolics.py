""":mod:`mirgecom.phenolics.phenolics` handles phenolics modeling.

Additional details are provided in
https://github.com/illinois-ceesd/phenolics-notes

Conserved Quantities Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PhenolicsConservedVars

Dependent Quantities Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PhenolicsDependentVars
.. autoclass:: PhenolicsEOS

Helper Functions
^^^^^^^^^^^^^^^^
.. autofunction:: initializer
.. autofunction:: make_conserved

"""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from meshmode.dof_array import DOFArray
from dataclasses import dataclass
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)

import sys  # noqa


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsConservedVars:
    r"""."""

    # the "epsilon_density" of each phase in the solid
    solid_species_mass: np.ndarray

    # it includes the epsilon/void fraction
    gas_density: DOFArray

    # bulk energy = solid + gas energy
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.energy)

    @property
    def nphase(self):
        """Return the number of phases in the composite material."""
        return len(self.solid_species_mass)


def initializer(eos, solid_species_mass, temperature, gas_density=None,
                pressure=None, progress=0.0):
    """Initialize state of composite material."""
    if gas_density is None and pressure is None:
        raise ValueError("Must specify one of 'gas_density' or 'pressure'")

    if isinstance(temperature, DOFArray) is False:
        raise ValueError("Temperature does not have the proper shape")

    zeros = temperature*0.0

    # progress ratio
    if isinstance(progress, DOFArray) is False:
        tau = zeros + 1.0 - progress
    else:
        tau = 1.0 - progress

    # gas constant
    Rg = 8314.46261815324/eos.gas_molar_mass(temperature)  # noqa N806

    if gas_density is None:
        eps_gas = eos.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(Rg*temperature)

    eps_rho_solid = sum(solid_species_mass)
    bulk_energy = (
        eps_rho_solid*eos.solid_enthalpy(temperature, tau)
        + eps_rho_gas*(eos.gas_enthalpy(temperature) - Rg*temperature)
    )

    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=bulk_energy, gas_density=eps_rho_gas)


def make_conserved(solid_species_mass, gas_density, energy):  # noqa D103
    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=energy, gas_density=gas_density)


@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsDependentVars:
    """State-dependent quantities.

    .. attribute:: temperature
    .. attribute:: pressure
    .. attribute:: molar_mass
    .. attribute:: viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: progress
    .. attribute:: emissivity
    .. attribute:: permeability
    .. attribute:: void_fraction
    .. attribute:: solid_density
    """

    temperature: DOFArray
    progress: DOFArray

    thermal_conductivity: DOFArray
    void_fraction: DOFArray

    pressure: DOFArray
    molar_mass: DOFArray
    viscosity: DOFArray
#    velocity: DOFArray
#    species_diffusivity: np.ndarray

#    solid_emissivity: DOFArray
    solid_permeability: DOFArray
    solid_density: DOFArray


# TODO maybe split this in two, one for "gas" and another for "solid"??
class PhenolicsEOS():
    """.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: void_fraction
    .. automethod:: thermal_conductivity
    .. automethod:: gas_molar_mass
    .. automethod:: gas_viscosity
    .. automethod:: gas_enthalpy
    .. automethod:: gas_heat_capacity
    .. automethod:: gas_dMdT
    .. automethod:: gas_pressure_diffusivity
    .. automethod:: gas_pressure
    .. automethod:: solid_density
    .. automethod:: solid_permeability
    .. automethod:: solid_emissivity
    .. automethod:: solid_enthalpy
    .. automethod:: solid_heat_capacity_cp
    """

    def __init__(self, composite, gas):
        """Initialize EOS for composite."""
        self._composite_model = composite
        self._gas_data = gas

    # ~~~~~~~~~~~~
    def eval_tau(self, wv):
        r"""Progress ratio of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists.
        """
        return 280.0/(280.0 - 220.0)*(1.0 - 220.0/self.solid_density(wv))

    def eval_temperature(self, wv, tseed, tau):
        """Temperature assumes thermal equilibrium between solid and fluid.

        Performing Newton iteration to evaluate the temperature based on the
        internal energy/enthalpy and heat capacity.

        Add equation...
        """
        niter = 3
        temp = tseed*1.0

        rho_gas = wv.gas_density
        rho_solid = self.solid_density(wv)
        rhoe = wv.energy
        for _ in range(0, niter):

            # gas constant R/M
            molar_mass = self.gas_molar_mass(temp)
            Rg = 8314.46261815324/molar_mass  # noqa N806

            eps_rho_e = (
                rho_gas*(self.gas_enthalpy(temp) - Rg*temp)
                + rho_solid*self.solid_enthalpy(temp, tau))

            bulk_cp = (
                rho_gas*(self.gas_heat_capacity(temp)
                         - Rg*(1.0 - temp/molar_mass*self.gas_dMdT(temp)))
                + rho_solid*self.solid_heat_capacity_cp(temp, tau))

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    # ~~~~~~~~~~~~ bulk gas+solid properties
    def void_fraction(self, tau):
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$."""
        return 1.0 - self._composite_model.solid_volume_fraction(tau)

    def thermal_conductivity(self, temp, tau):
        r"""Return the bulk thermal conductivity, $f(\tau, T)$."""
        return (
            self._composite_model.solid_thermal_conductivity(temp, tau)
            # + gas
        )

    # ~~~~~~~~~~~~ gas
    def gas_molar_mass(self, temp):
        """Return the gas molar mass."""
        return self._gas_data.gas_molar_mass(temp)

    def gas_viscosity(self, temp):
        """Return the gas viscosity."""
        return self._gas_data.gas_viscosity(temp)

#    # FIXME
    def gas_pressure_diffusivity(self, temp, tau):
        r"""Return the pressure diffusivity.

        It is evaluated as

        .. math::

            d_{P} = \frac{P \mathbf{K}}{\mu \epsilon}

        where $\mu$ is the gas viscosity, $\epsilon$ is the void fraction and
        $\mathbf{K}$ is the permeability matrix.
        """
        return temp*0.0

    def gas_pressure(self, wv, temp, tau):
        """Return the gas pressure."""
        Rg = 8314.46261815324/self.gas_molar_mass(temp)  # noqa N806
        eps_gas = self.void_fraction(tau)
        return (1.0/eps_gas)*wv.gas_density*Rg*temp

#    # FIXME
#    def velocity(self, wv, temperature, tau):
#        return temperature*0.0

#    # FIXME
#    def species_diffusivity(self, wv, temperature, tau):
#        return temperature*0.0

    # ~~~~~~~~~~~~ solid
    def solid_density(self, wv):
        r"""Return the solid density $\epsilon_s \rho_s$.

        The density is relative to the entire control volume and it is
        computed with the sum of all N solid phases:

        .. math::

            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        return sum(wv.solid_species_mass)

    def solid_permeability(self, tau):
        r"""Return the wall permeability based on the progress ratio $\tau$."""
        return self._composite_model.solid_permeability(tau)

    def solid_emissivity(self, tau):
        r"""Return the wall emissivity based on the progress ratio $\tau$."""
        return self._composite_model.solid_emissivity(tau)

    # ~~~~~~~~~~~~ auxiliary functions
    def gas_enthalpy(self, temp):
        """Return the gas enthalpy."""
        return self._gas_data.gas_enthalpy(temp)

    def gas_heat_capacity(self, temp):
        """Return the gas heat capacity."""
        return self._gas_data.gas_viscosity(temp)

    def gas_dMdT(self, temp):  # noqa N802
        """Return the partial derivative of molar mass wrt temperature."""
        return self._gas_data.gas_dMdT(temp)

    def solid_enthalpy(self, temp, tau):
        """Return the solid enthalpy."""
        return self._composite_model.solid_enthalpy(temp, tau)

    def solid_heat_capacity_cp(self, temp, tau):
        """Return the solid heat capacity."""
        return self._composite_model.solid_heat_capacity(temp, tau)

    def dependent_vars(self, wv: PhenolicsConservedVars,
            temperature_seed: DOFArray) -> PhenolicsDependentVars:
        """Get the dependent variables."""
        tau = self.eval_tau(wv)
        temperature = self.eval_temperature(wv, temperature_seed, tau)
        return PhenolicsDependentVars(
            progress=1.0-tau,
            temperature=temperature,
            thermal_conductivity=self.thermal_conductivity(temperature, tau),
            void_fraction=self.void_fraction(tau),
            pressure=self.gas_pressure(wv, temperature, tau),
            # velocity
            viscosity=self.gas_viscosity(temperature),
            molar_mass=self.gas_molar_mass(temperature),
            # species_diffusivity
            # enthalpy
            # heat_capacity
            solid_density=self.solid_density(wv),
            # solid_emissivity=self.emissivity(temperature, tau),
            solid_permeability=self.solid_permeability(tau)
        )
