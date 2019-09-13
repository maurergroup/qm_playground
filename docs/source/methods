.. _theory:

Theory
======


Non-adiabatic friction of adsorbate motion on metal surfaces
------------------------------------------------------------

DEAR KENNY, THIS IS JUST EXAMPLE STUFF SO YOU NOW HOW TO PUT YOUR OWN

Whereas in insulating or most semi-conducting materials the energy spectrum of vibrations and electronic excitations is clearly separated, this is not the case for metals. In metals even the smallest vibrational motion can lead to electronic excitations due to the vanishing gap between occupied and unoccupied states.

.. figure:: ./friction.png
   :align: center
   :width: 70%

   *Figure 1: Schematic view of adsorbate vibration (here shown for a CO on Cu(100) internal stretch motion) leading to changes in the electronic st    ructure that excite electron hole-pairs from below to above the Fermi level of the metal Density-of-States.*


As shown in Figure 1, adsorbate motion can lead to changes in the electronic structure that facilitate low energy electronic excitations in the metal band structure. We can treat this interaction between electrons and vibrations with perturbation theory if the following assumptions are true:
    
    * the coupling is weak compared to the individual contributions of electrons and nuclei
    * electron-hole pair excitations do not lead to a qualitative change in the nuclear dynamics
     

The Friction Tensor
-------------------

We can reformulate the matrix elements of expression :eq:`coupling` in terms of a cartesian coupling tensor:

.. math::

    |g_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{\mathbf{q}j}|^2 &= g_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{\mathbf{q}j}\cdot g_{s\mathbf{k}\nu,s\mathbf{k+q}\nu'}^{\mathbf{q}j} = \\ & \mathbf{e}^T_{\mathbf{q}j}\cdot\braket{\psi_{s\mathbf{k}\nu}|\mathbf{\nabla}_{\mathbf{R}}|\psi_{s\mathbf{k+q}\nu'}}\cdot\braket{\psi_{s\mathbf{k+q}\nu'}|\mathbf{\nabla}_{\mathbf{R}}|\psi_{s\mathbf{k}\nu}} \cdot\mathbf{e}_{\mathbf{q}j} = \\ & \mathbf{e}^T_{\mathbf{q}j}\cdot \mathbf{G}_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}\cdot \mathbf{e}_{\mathbf{q}j}.

The elements of G are defined as:

.. math::

    \mathbf{G}_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} = \braket{\psi_{s\mathbf{k}\nu}|\frac{\partial}{\partial_{a}\mathbf{R}_{n}}|\psi_{s\mathbf{k+q}\nu'}} \cdot \braket{\psi_{s\mathbf{k+q}\nu'}|\frac{\partial}{\partial_{a'}\mathbf{R}_{n'}}|\psi_{s\mathbf{k}\nu}} ,

where n and n' indicate the n-th (n'-th) atom and a and a' indicate one of the three cartesian directions x, y, and z. Inserting **G** into eq. :eq:`gamma2` we arrive at:

.. math::
   :label: friction

    \Gamma(\omega_{\mathbf{q}j}) &=  \mathbf{e}^T_{\mathbf{q}j}\cdot  \biggl( \pi\sum_{s,\mathbf{k},\nu,\nu'} \mathbf{G}_{s\mathbf{k+q}\nu',s\mathbf{k}\nu} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))  \\ \nonumber &    \cdot\delta(\epsilon_{\mathbf{k}\nu}-\epsilon_{\mathbf{k+q}\nu'}) \biggr) \cdot \mathbf{e}_{\mathbf{q}j} = \mathbf{e}^T_{\mathbf{q}j}\cdot\Lambda^{\mathbf{q}}\cdot\mathbf{e}_{\mathbf{q}j}. 
  
In eq. :eq:`friction` we have defined the friction tensor :math:`\mathbf{\Lambda}^{\mathbf{q}}` with dimensions :math:`(3N\times3N)` where N is the number of atoms. The elements of  :math:`\mathbf{\Lambda}^{\mathbf{q}}` are defined as

.. math::
    :label: friction-tensor
    
    \Lambda^{\mathbf{q}}_{n'a',na} = \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}} \sum_{s,\mathbf{k},\nu,\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot(\epsilon_{s\mathbf{k+q}\nu'}-\epsilon_{s\mathbf{k}\nu}) \cdot\delta(\epsilon_{\mathbf{k}\nu}-\epsilon_{\mathbf{k+q}\nu'}) .


Vibrational relaxation rates of adsorbates
------------------------------------------

Eq. :eq:`friction` shows that using the friction tensor one can calculate relaxation rates and vibrational lifetimes for arbitrary molecular motions.

.. figure:: ./motion1.png
   :align: center
   :width: 70%

   Figure 2: Collective motion of a periodic overlayer of CO atoms on Cu(100)

Considering periodic boundary conditions, the collective motion of an adsorbate overlayer (as shown in Fig. 2) can be described with a single adsorbate vibrational mode with wave vector :math:`q=0`. The corresponding relaxatio rate for this vibration is given by:

.. math::
    :label: rate-default

    \Gamma(\omega_j) = \mathbf{e}^T_j\cdot\Gamma^{\mathbf{q}=0}\cdot\mathbf{e}_j

Calculation of vibrational cooling using this approach is the most common [Head-Gordon1992]_ , [Persson1982]_ and is the *default* setting when calculating lifetimes in **coolvib**.


.. figure:: ./motion2.png
   :align: center
   :width: 70%

   Figure 3: Motion of only one CO atom adsorbed on Cu(100), effectively breaking periodicity.

When considering the lifetime due to motion of a single adsorbate, such as is the case for impinging adsorbates, 
low coverages or non-collective adlayer motion (see Fig. 3) one effectively needs to integrate over all possible vibrations of the same type with different wave vectors :math:`\mathbf{q}`, effectively amounting to a Fourier expansion in reciprocal space. Correspondingly, the relaxation rate is given by:

.. math::
    :label: rate-momentum

    \Gamma(\omega_{\mathbf{q}j}) = \sum_{\mathbf{q}} \mathbf{e}^T_{\mathbf{q}j}\cdot\Gamma^{\mathbf{q}}\cdot\mathbf{e}_{\mathbf{qj}} \approx  \mathbf{e}^T_j\cdot\Gamma^{\sum \mathbf{q}}\cdot\mathbf{e}_j


In the corresponding relaxation rate, we additionally include excitations that in principle violate momentum-conservation. If we assume that adsorbate vibrations interact only by simple phase modulation we can assume the atomic displacements of individual adsorbate images to be independent of wave vector yielding following expression for the friction tensor in this case:

.. math::
   :label: friction-tensor-q

   \Lambda^{\sum \mathbf{q}}_{n'a',na} = \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}} \sum_{s} \sum_{\mathbf{q}}\sum_{\mathbf{k}} \sum_{\nu,\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot(\epsilon_{s\mathbf{k+q}\nu'}-\epsilon_{s\mathbf{k}\nu}) \cdot\delta(\epsilon_{\mathbf{k}\nu}-\epsilon_{\mathbf{k+q}\nu'}) .

Numerical considerations and working equations
------------------------------------------------

Depending on if one wants to calculate vibrational lifetimes of periodic overlayers or 
single adsorbates one is left with evaluating eqs. :eq:`friction-tensor` or :eq:`friction-tensor-q`.
This is by no means a simple task due to the slow convergence of electronic structure close to the 
Fermi level and intricacies of evaluating a delta function in a non-equidistant space of excitations.

Some of these issues can be solved by transforming the equation into a continuous representation using the 
the Density-of-State of the system:

.. math::

   \sum_{\nu} 1= \sum_{\nu} \int_{-\infty}^{\infty} d\epsilon_a\cdot \delta(\epsilon_a-\epsilon_{\nu})= \int_{-\infty}^{\infty} d\epsilon_a\cdot \underbrace{\sum_{\nu} \delta(\epsilon_a-\epsilon_{\nu})}_{\rho(\epsilon)}

Introducing the Density-of-State for both sums over eigenstates in :eq:`friction-tensor` we arrive at

.. math::

    \Lambda^{\mathbf{q}}_{n'a',na} = \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  \int_{-\infty}^{\epsilon_F} d\epsilon_a\int_{\epsilon_F}^{\infty} d\epsilon_b \sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_a-\epsilon_{s\mathbf{k}\nu})\cdot\delta(\epsilon_b-\epsilon_{s\mathbf{k+q}\nu'})\cdot\epsilon \cdot\delta(\epsilon_{a}-\epsilon_{b}) .

The change in variable from the discrete space to the continuous space for the last delta function and the energy difference can be understood as a consequence of sequentially evaluating the first to delta functions (and by equivalence)
Now we change the integration variables from occupied and unoccupied states to excitation energies

.. math::

    \epsilon = \epsilon_b - \epsilon_a \Rightarrow d\epsilon_b=d\epsilon+\epsilon_a

and arrive at

.. math::

    \Lambda^{\mathbf{q}}_{n'a',na} = \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  \int_{-\infty}^{\epsilon_F} d\epsilon_a\int_{\epsilon_F}^{\infty} (d\epsilon+d\epsilon_a) \sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_a-\epsilon_{s\mathbf{k}\nu})\cdot\delta(\epsilon+\epsilon_a-\epsilon_{s\mathbf{k+q}\nu'})\cdot\epsilon \cdot\delta(-\epsilon) =
    
.. math::
    :label: friction-derivation

    &= \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  \int_{-\infty}^{\epsilon_F} d\epsilon_a\int_{\epsilon_F}^{\infty} d\epsilon_a \sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_a-\epsilon_{s\mathbf{k}\nu})\cdot\delta(\epsilon+\epsilon_a-\epsilon_{s\mathbf{k+q}\nu'})\cdot\epsilon \cdot\delta(-\epsilon) = \\ 
    &= \frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  \int_{-\infty}^{\epsilon_F} d\epsilon_a\int_{-\infty}^{\infty} d\epsilon \sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'} G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_a-\epsilon_{s\mathbf{k}\nu})\cdot\delta(\epsilon+\epsilon_a-\epsilon_{s\mathbf{k+q}\nu'})\cdot\epsilon \cdot\delta(-\epsilon)

The first term in eq. :eq:`friction-derivation` vanishes because the integration variable for occupied states :math:`d\epsilon_a` is not defined in the range between :math:`\infty` and :math:`\epsilon_F`. In the second term we can use the following properties of delta functions

.. math::

   \int \delta(\epsilon_a-\epsilon_{s\mathbf{k}\nu})\delta(\epsilon+\epsilon_a-\epsilon_{s\mathbf{k+q}\nu'})=\delta(\epsilon+\epsilon_{s\mathbf{k}\nu}-\epsilon_{s\mathbf{k+q}\nu'})

and

.. math::

   \delta(-x) = \delta(x)

to arrive at 

.. math::
   :label: friction-tensor-new


   \Lambda^{\mathbf{q}}_{n'a',na} = \int_{-\infty}^{\infty} d\epsilon \cdot\epsilon \cdot\delta(\epsilon) \cdot\underbrace{\sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'}\frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_{s\mathbf{k+q}\nu'}-\epsilon_{s\mathbf{k}\nu}-\epsilon) }_{A_{n'a',na}(\epsilon)}

   
By rewriting the friction tensor in this way we have gained several things:

    * The continuous formulation enables simple numerical discretization and accurate evaluation of the delta function :math:`\delta(\epsilon)`, thereby reducing numerical instabilities.
    * We have defined a set of **electron-phonon spectral functions** :math:`A_{n'a',na}(\epsilon)` for all combinations of cartesian components, which describe the principal spectrum of available coupling channels between electronic and nuclear degrees of freedom.
    * The second delta function transforms the set of finite transitions into smooth continuous spectral functions and therefore improves convergence with respect to Brillouin zone sampling.

The expression for the spectral functions 

.. math::
    :label: spectral-function1

    A_{n'a',na}(\epsilon) = \sum_{s,\mathbf{k}}\sum_{\nu}\sum_{\nu'}\frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_{s\mathbf{k+q}\nu'}-\epsilon_{s\mathbf{k}\nu}-\epsilon) 

and its corresponding equivalent for isolated adsorbates

.. math::
    :label: spectral-function2

    A_{n'a',na}^{\sum \mathbf{q}}(\epsilon) = \sum_{s,\mathbf{q},\mathbf{k}}\sum_{\nu}\sum_{\nu'}\frac{\pi\hbar}{\sqrt{M_{n'}}\sqrt{M_n}}  G_{s\mathbf{k+q}\nu',s\mathbf{k}\nu}^{n'a',na} \cdot(f(\epsilon_{\mathbf{k}\nu})-f(\epsilon_{\mathbf{k+q}\nu'}))\cdot\delta(\epsilon_{s\mathbf{k+q}\nu'}-\epsilon_{s\mathbf{k}\nu}-\epsilon)

.. math::


represent the main working equations of this software.


References
----------


.. [Hellsing1984] `B.` Hellsing, and M. Persson, *Physi. Scripta* **29**, 360-371 (1984)
