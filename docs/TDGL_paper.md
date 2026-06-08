# Numerical Simulation of Vortex Dynamics in Type-II Superconductors

**William D. Gropp, Hans G. Kaper, Gary K. Leaf, David M. Levine, Mario Palumbo, Valerii M. Vinokur**

Argonne National Laboratory, Argonne, IL 60439, USA

*April 4, 1995*

---

**AMS(MOS) subject classification.** Primary 81J05. Secondary 82A25, 65N05, 35J60.

**Key words and phrases.** Ginzburg-Landau model, type-II superconductors, numerical simulations, parallel computing, pattern formation, twin boundaries, vortex dynamics.

> This work was supported by the Office of Scientific Computing, U.S. Department of Energy, under Contract W-31-109-Eng-38.

---

**PROPOSED RUNNING HEAD:** Vortex Dynamics in Type-II Superconductors

**Send correspondence to:**
Hans G. Kaper
Mathematics and Computer Science Division
Argonne National Laboratory
Argonne, IL 60439, USA
Phone: (708) 252-7160
Fax: (708) 252-5986
E-mail: kaper@mcs.anl.gov

---

## Abstract

This article describes the results of several numerical simulations of vortex dynamics in type-II superconductors. The underlying mathematical model is the time-dependent Ginzburg-Landau model. The simulations concern vortex penetration in the presence of twin boundaries, interface patterns between regions of opposite vortex orientation, and magnetic-flux entry patterns in superconducting samples.

---

## 1 Ginzburg-Landau Model

In this article we report on several numerical simulations of vortex motion in type-II superconducting materials [1,2]. The simulations are based on the time-dependent Ginzburg-Landau (TDGL) equations,

$$
\frac{\hbar^2}{2m_s D} \left( \frac{\partial}{\partial t} + \frac{ie_s}{\hbar} \phi \right) \psi = -\frac{\delta \mathcal{L}}{\delta \psi^*}, \tag{1.1}
$$

$$
\frac{\sigma}{c} \left( \frac{1}{c} \frac{\partial \mathbf{A}}{\partial t} + \nabla \phi \right) = -\frac{\sigma}{c} \frac{\delta \mathcal{L}}{\delta \mathbf{A}} - \frac{1}{4\pi} \nabla \times \nabla \times \mathbf{A}. \tag{1.2}
$$

The symbol $\mathcal{L}$ stands for the density of the Helmholtz free-energy functional which, in the Ginzburg-Landau (GL) approximation, is given by an expression of the form

$$
\mathcal{L} = a|\psi|^2 + \frac{1}{2}b|\psi|^4 + \frac{1}{2m_s}\left| \frac{\hbar}{i}\nabla - \frac{e_s}{c}\mathbf{A} \right|^2. \tag{1.3}
$$

Here, $\psi$ is the (complex-valued) order parameter, $\mathbf{A}$ the vector potential, and $\phi$ the electric potential. The order parameter is identically equal to zero in a normal metal. The observable quantities are the magnetic induction, $\mathbf{B} = \nabla \times \mathbf{A}$; the density of Cooper pairs, $n_s = |\psi|^2$; and the current density, $\mathbf{J} = \sigma \mathbf{E} + \mathbf{J}_s$, where $\mathbf{E} = -(1/c)\partial\mathbf{A}/\partial t - \nabla\phi$ is the electric field (Faraday's law) and $\mathbf{J}_s$ the supercurrent density,

$$
\mathbf{J}_s = \frac{e_s \hbar}{2im_s}\left(\psi^* \nabla\psi - \psi\nabla\psi^*\right) - \frac{e_s^2}{m_s c}|\psi|^2 \mathbf{A}. \tag{1.4}
$$

The superscript $*$ denotes complex conjugation, $e_s$ is the "effective charge" of a superelectron ($e_s < 0$) and $m_s$ its "effective mass." The constant $D$ is a phenomenological diffusion coefficient. If $\mathbf{J}$ is viewed as the sum of a "normal" current, which satisfies Ohm's law, and the supercurrent, $\sigma$ may be interpreted as the "coefficient of normal conductivity."

The quantities $a$ and $b$ in (1.3) are phenomenological parameters; they are functions of external parameters, such as the temperature $T$, the concentration of impurities, etc.; $b > 0$ for all $T$, and $a$ changes sign at $T_c$ ($a < 0$ for $T < T_c$, $a > 0$ for $T > T_c$).

When the variational derivatives are written out, (1.1) and (1.2) assume the form

$$
\frac{\hbar^2}{2m_s D}\left(\frac{\partial}{\partial t} + \frac{ie_s\phi}{\hbar}\right)\psi + \frac{1}{2m_s}\left(\frac{\hbar}{i}\nabla - \frac{e_s}{c}\mathbf{A}\right)\cdot\left(\frac{\hbar}{i}\nabla - \frac{e_s}{c}\mathbf{A}\right)\psi - |a|\psi + b|\psi|^2\psi = 0, \tag{1.5}
$$

$$
\nabla \times \nabla \times \mathbf{A} = -\frac{4\sigma}{c}\left(\frac{1}{c}\frac{\partial \mathbf{A}}{\partial t} + \nabla\phi\right) + \frac{4\pi}{c}\mathbf{J}_s. \tag{1.6}
$$

The equation (1.5), first proposed by Schmid [3], was derived from the microscopic BCS theory by Gor'kov and Eliashberg [4]; in the zero-field case, it reduces to a semilinear diffusion equation with diffusion coefficient $D$. The equation (1.6) is Ampere's law, $\nabla \times \mathbf{B} = (4\pi/c)\mathbf{J}$; see [5, Chapter 5].

Our numerical approximations are set up to simulate three-dimensional rectangular configurations, where a superconducting material occupying a region $\Omega_{sc}$ is surrounded by a normal metal. The entire configuration occupies a region $\Omega$, which is in turn surrounded by vacuum. The boundary $\partial\Omega_{sc}$ of $\Omega_{sc}$ is the interface between the superconductor and the normal metal; the boundary $\partial\Omega$ of $\Omega$ is the outer boundary. Both are two-dimensional surfaces. The unit normal vector, $\mathbf{n}$, is always directed toward the exterior of the domain. An arbitrary point in space is denoted by the vector $\mathbf{x}$ or by its coordinates $(x, y, z)$.

At the interface, we impose the boundary condition

$$
\mathbf{J}_s \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega_{sc}. \tag{1.7}
$$

Strictly speaking, this condition is correct only for superconductor–insulator interfaces and needs to be generalized for superconductor–metal interfaces [6, Section 7–3]. However, we have implemented (1.7) everywhere on $\partial\Omega_{sc}$.

On the outer surface, we may prescribe a surface current $\mathbf{K}$ to account for an applied magnetic field $\mathbf{H}$ in the exterior vacuum. This surface current causes a jump discontinuity in the tangential component of $\mathbf{B}$, whose magnitude is $(4\pi/c)$ times the magnitude of $\mathbf{K}$ and whose direction is parallel to $\mathbf{K} \times \mathbf{n}$.

Although the validity of the TDGL model for high-$T_c$ superconductors is arguable — in theory, it is valid only asymptotically near the critical temperature, $T_c$ — the results are in excellent qualitative and quantitative agreement with the results of physical experiments over a wide range of temperatures. The TDGL model has been used previously to study the interaction of colliding vortices [7], nucleation in thin films [8,9], magnetization in thin films [10], and $I$–$V$ characteristics of thin films [11]. A mathematical framework for the GL functional can be found in [12]; the TDGL model is analyzed in detail in [13,14].

The computational (discrete) TDGL model is presented in Section 2, the results of the numerical simulations in Section 3. All computations were done on the IBM POWERparallel SP System at Argonne National Laboratory (128 processors, 128 megabytes per processor, theoretical peak performance 16 Gflops).


---

## 2 Computational Model

Before introducing the discrete approximations, we reduce the TDGL model to a nondimensional form, introduce the zero-electric potential gauge, and define the field variables in terms of link variables.

### 2.1 Dimensionless Form

We render the TDGL model dimensionless by measuring lengths in units of the London penetration depth $\lambda$; time in units of a characteristic relaxation time $\tau = \lambda^2/D$, where $\xi$ is coherence length; fields in units of $H_c\sqrt{2}$, where $H_c$ is the thermodynamic critical field; and energy densities in units of $H_c^2/(4\pi)$. With $\psi_0^2 = |a|/b$, we have

$$
\lambda = \left(\frac{m_s c^2}{4\pi \psi_0^2 e_s^2}\right)^{1/2}, \quad \xi = \left(\frac{\hbar^2}{2m_s|a|}\right)^{1/2}, \quad H_c = \left(\frac{4\pi|a|\psi_0^2}{1}\right)^{1/2}. \tag{2.1}
$$

The penetration depth and coherence length are characteristic lengths for the magnetic field and the order parameter, respectively. The ratio $\kappa = \lambda/\xi$ is the Ginzburg-Landau parameter, which is approximately temperature-independent; typically, $\kappa \approx 100$ in the case of a high-$T_c$ superconductor.

Distinguishing a dimensionless quantity from its dimensional counterpart by a prime, we make the following substitutions: $\mathbf{x} = (x, y, z) = (x', y', z')\lambda = \mathbf{x}'\lambda$, $t = \tau t'$, $\psi = \psi_0\psi'$, $\mathbf{A} = H_c\sqrt{2}\mathbf{A}'$, $\phi = (\lambda/c\tau)H_c\sqrt{2}\phi'$, $\mathcal{L} = (H_c^2/4\pi)\mathcal{L}'$, $\mathbf{B} = H_c\sqrt{2}\mathbf{B}'$, $\mathbf{E} = (\lambda/c\tau)H_c\sqrt{2}\mathbf{E}'$, $\mathbf{J} = (\psi_0^2 e_s\hbar/m_s\xi)\mathbf{J}'$, and $\sigma = (c^2\tau/4\pi\lambda^2)\sigma'$. Here, $\psi'$, $\mathbf{A}'$, etc. are functions of $\mathbf{x}'$ and $t'$.

Omitting all primes, we obtain the dimensionless form of the variational equations (1.1) and (1.2),

$$
\frac{\partial\psi}{\partial t} + i\phi\psi = -\frac{\delta\mathcal{L}}{\delta\psi^*}, \quad \sigma\left(\frac{\partial\mathbf{A}}{\partial t} + \nabla\phi\right) = -\frac{1}{2}\frac{\delta\mathcal{L}}{\delta\mathbf{A}} - \nabla\times\nabla\times\mathbf{A}, \tag{2.2}
$$

where

$$
\mathcal{L} = -|\psi|^2 + \frac{1}{2}|\psi|^4 + \left|\frac{1}{i}\nabla - \mathbf{A}\right|^2. \tag{2.3}
$$

The observable quantities are $\mathbf{B} = \nabla\times\mathbf{A}$; $n_s = |\psi|^2$; and $\mathbf{J} = \sigma\mathbf{E} + \mathbf{J}_s$, where $\mathbf{E} = -\partial\mathbf{A}/\partial t - \nabla\phi$ and

$$
\mathbf{J}_s = \frac{1}{2i}(\psi^*\nabla\psi - \psi\nabla\psi^*) - |\psi|^2\mathbf{A}. \tag{2.4}
$$

### 2.2 Zero Potential Gauge

The TDGL model is invariant under the gauge transformation

$$
\psi = e^{i\chi}\bar{\psi}, \quad \mathbf{A} = \bar{\mathbf{A}} + \nabla\chi, \quad \phi = \bar{\phi} - \frac{\partial\chi}{\partial t}, \tag{2.5}
$$

where the gauge $\chi$ is any function of space and time; see [2, Section 19.6]. We choose the zero-electric potential gauge, so $\phi = 0$ at all times. (Other possible gauge choices are discussed in [13].) Dropping the overbars, we thus reduce (2.2) to

$$
\frac{\partial\psi}{\partial t} = -\frac{\delta\mathcal{L}}{\delta\psi^*}, \quad \sigma\frac{\partial\mathbf{A}}{\partial t} = -\frac{1}{2}\frac{\delta\mathcal{L}}{\delta\mathbf{A}} - \nabla\times\nabla\times\mathbf{A}. \tag{2.6}
$$

### 2.3 Link Variables

We introduce the auxiliary vector $\mathbf{U} = (U_x, U_y, U_z)$,

$$
U_x(x,y,z) = \exp\left(-i\int_{x_0}^{x} A_x(\xi,y,z)\,d\xi\right),
$$

$$
U_y(x,y,z) = \exp\left(-i\int_{y_0}^{y} A_y(x,\eta,z)\,d\eta\right), \tag{2.7}
$$

$$
U_z(x,y,z) = \exp\left(-i\int_{z_0}^{z} A_z(x,y,\zeta)\,d\zeta\right).
$$

(We omit the argument $t$.) The point $\mathbf{x}_0 = (x_0, y_0, z_0)$ is an arbitrary reference point. Each $U_\alpha$ ($\alpha = x, y, z$) is complex valued and unimodular, $U_\alpha^* = U_\alpha^{-1}$. It follows from (2.7) that

$$
A_x = -\frac{1}{2i}\left(U_x^*\frac{\partial U_x}{\partial x} - U_x\frac{\partial U_x^*}{\partial x}\right),
$$

$$
A_y = -\frac{1}{2i}\left(U_y^*\frac{\partial U_y}{\partial y} - U_y\frac{\partial U_y^*}{\partial y}\right), \tag{2.8}
$$

$$
A_z = -\frac{1}{2i}\left(U_z^*\frac{\partial U_z}{\partial z} - U_z\frac{\partial U_z^*}{\partial z}\right).
$$

The energy density (2.3) can now be written in the form

$$
\mathcal{L} = -|\psi|^2 + \frac{1}{2}|\psi|^4 + \frac{1}{2}\sum_{\alpha=x,y,z}\left|\frac{\partial}{\partial\alpha}(U_\alpha\psi)\right|^2. \tag{2.9}
$$

This expression shows that the presence of a nonzero field induces anisotropic diffusion of the order parameter. The diffusion coefficient depends locally on the auxiliary vector $\mathbf{U}$.

The variables (2.7) are related to the link variables of lattice gauge theory [15,16]; see the remark following (2.32). Their introduction at this point facilitates the preservation of gauge invariance under discretization [17]. Borrowing the terminology, we refer to the vector $\mathbf{U}$ as the vector of link variables.

When the equations (2.6) are worked out, the differential equations to be solved are

$$
\frac{\partial\psi}{\partial t} - \frac{1}{2}\sum_{\alpha=x,y,z} U_\alpha^*\frac{\partial^2}{\partial\alpha^2}(U_\alpha\psi) + |\psi|^2\psi = 0 \quad \text{in } \Omega_{sc}, \tag{2.10}
$$

$$
\sigma\frac{\partial\mathbf{A}}{\partial t} + \nabla\times\nabla\times\mathbf{A} = \begin{cases} \mathbf{J}_s & \text{in } \Omega, \\ 0 & \text{in } \Omega \setminus \Omega_{sc}. \end{cases} \tag{2.11}
$$

The supercurrent density $\mathbf{J}_s$ is given in terms of $\psi$ and $\mathbf{A}$ by (2.4) or, alternatively, in terms of $\psi$ and $\mathbf{U}$ by

$$
J_{s,\alpha} = \frac{1}{2i}\left[U_\alpha^*\psi^*\frac{\partial}{\partial\alpha}(U_\alpha\psi) - U_\alpha\psi\frac{\partial}{\partial\alpha}(U_\alpha^*\psi^*)\right], \quad \alpha = x, y, z. \tag{2.12}
$$

The equation (2.10) must be solved subject to the boundary condition $\mathbf{J}_s \cdot \mathbf{n} = 0$ on $\partial\Omega_{sc}$. The TDGL model is completed by the specification of initial conditions for $|\psi|$ and $\mathbf{B}$.

We observe that, if $\mathbf{J}_s \cdot \mathbf{n} = 0$ on $\partial\Omega_{sc}$, then $\mathbf{A} \cdot \mathbf{n}$ does not vary with time on $\partial\Omega_{sc}$. (Take the divergence of both sides of (2.11), integrate the resulting identity over an infinitesimally thin volume adjacent to $\Omega_{sc}$, and apply Gauss's theorem.) Because $\mathbf{J}_s = 0$ in $\Omega \setminus \Omega_{sc}$, the same argument shows that $\mathbf{A} \cdot \mathbf{n}$ does not vary with time on $\partial\Omega$.

### 2.4 Computational Grid

We consider rectangular configurations, where a brick-shaped region $\Omega_{sc}$ of superconducting material is imbedded in a normal metal. The entire configuration has the shape of a rectangular box, which occupies a region $\Omega$ in space. The brick and the box are lined up in parallel. Outside $\Omega$, the magnetic field is given and uniform.

All computations are done on a uniform grid with mesh widths $h_x$, $h_y$, and $h_z$. The grid is asymptotically regular, in the sense that $h_\alpha/h_\beta = O(1)$ as $h = \max\{h_x, h_y, h_z\} \to 0$, for any pair $\alpha, \beta$ ($\alpha, \beta = x, y, z$). A typical grid cell is

$$
\Omega_{i,j,k} = \{\mathbf{x} = (x,y,z) : x_i < x < x_{i+1},\, y_j < y < y_{j+1},\, z_k < z < z_{k+1}\}, \tag{2.13}
$$

where

$$
x_i = x_1 + (i-1)h_x, \quad y_j = y_1 + (j-1)h_y, \quad z_k = z_1 + (k-1)h_z. \tag{2.14}
$$

The vertex $\mathbf{x}_{i,j,k} = (x_i, y_j, z_k)$ is the reference point for $\Omega_{i,j,k}$. Unless noted otherwise, the indices run through the values $i = 1,\ldots,n_x$; $j = 1,\ldots,n_y$; $k = 1,\ldots,n_z$.

Conceptually, the domain $\Omega$ lies inside the grid at a distance of one-half mesh width from the bounding faces in each coordinate direction,

$$
\Omega = \{\mathbf{x} = (x,y,z) : x_1 + \tfrac{1}{2}h_x < x < x_{n_x} + \tfrac{1}{2}h_x,\;
y_1 + \tfrac{1}{2}h_y < y < x_{n_y} + \tfrac{1}{2}h_y,\; z_1 + \tfrac{1}{2}h_z < z < x_{n_z} + \tfrac{1}{2}h_z\}. \tag{2.15}
$$

The domain $\Omega_{sc}$ is located an integer number of mesh widths inside $\Omega$, in such a way that there is always at least one layer of grid points between $\Omega_{sc}$ and $\Omega$. That is,

$$
\Omega_{sc} = \{\mathbf{x} = (x,y,z) : x_{n_{sx}} + \tfrac{1}{2}h_x < x < x_{n_{ex}-1} + \tfrac{1}{2}h_x,
$$

$$
y_{n_{sy}} + \tfrac{1}{2}h_y < y < y_{n_{ey}-1} + \tfrac{1}{2}h_y,\; z_{n_{sz}} + \tfrac{1}{2}h_z < z < z_{n_{ez}-1} + \tfrac{1}{2}h_z\}, \tag{2.16}
$$

for three pairs of integers $(n_{sx}, n_{ex})$, $(n_{sy}, n_{ey})$, and $(n_{sz}, n_{ez})$, which satisfy the inequalities $1 < n_{s\alpha} < n_{e\alpha} < n_\alpha$ ($\alpha = x, y, z$). (The subscripts $sx$, $sy$, $sz$ and $ex$, $ey$, $ez$ stand for the "starting" and "ending" values in the $x$, $y$, $z$ direction, respectively.) The somewhat unorthodox numbering is a historical accident.

We shall derive the discrete TDGL model from an approximation to the free-energy functional. The approximation is second-order accurate as the mesh width goes to zero, since all integrals are evaluated by means of the midpoint rule and derivatives approximated by central differences.


### 2.5 Discrete Variables

We denote the discrete variables by the same symbols as their continuous counterparts. The index $(i,j,k)$ is assigned to any quantity related to the grid cell $\Omega_{i,j,k}$. The primary variables are the order parameter and the vector of link variables; all other variables (vector potential, induced magnetic field, supercurrent) are expressed in terms of these primary variables. The primary variables are evaluated on staggered grids. The order parameter $\psi$,

$$
\psi = \{\psi_{i,j,k} : i = n_{sx},\ldots,n_{ex};\; j = n_{sy},\ldots,n_{ey};\; k = n_{sz},\ldots,n_{ez}\}, \tag{2.17}
$$

where

$$
\psi_{i,j,k} = \psi(x_i, y_j, z_k). \tag{2.18}
$$

The interface conditions impose a constraint on the values of $\psi_{i,j,k}$ when either of the indices is equal to its starting or ending value; see (2.45) and (2.46). The vector of link variables $\mathbf{U} = (U_x, U_y, U_z)$,

$$
U_x = \{U_{x;i,j,k} : i,j,k\},\quad U_y = \{U_{y;i,j,k} : i,j,k\},\quad U_z = \{U_{z;i,j,k} : i,j,k\}, \tag{2.19}
$$

where

$$
U_{x;i,j,k} = \exp\left(-i\int_{x_i}^{x_{i+1}} A_x(\xi,y_j,z_k)\,d\xi\right), \tag{2.20}
$$

$$
U_{y;i,j,k} = \exp\left(-i\int_{y_j}^{y_{j+1}} A_y(x_i,\eta,z_k)\,d\eta\right), \tag{2.21}
$$

$$
U_{z;i,j,k} = \exp\left(-i\int_{z_k}^{z_{k+1}} A_z(x_i,y_j,\zeta)\,d\zeta\right). \tag{2.22}
$$

The vector potential $\mathbf{A} = (A_x, A_y, A_z)$,

$$
A_x = \{A_{x;i,j,k} : i,j,k\},\quad A_y = \{A_{y;i,j,k} : i,j,k\},\quad A_z = \{A_{z;i,j,k} : i,j,k\}, \tag{2.23}
$$

where

$$
A_{x;i,j,k} = A_x(x_i + \tfrac{1}{2}h_x, y_j, z_k),\quad A_{y;i,j,k} = A_y(x_i, y_j + \tfrac{1}{2}h_y, z_k),\quad A_{z;i,j,k} = A_z(x_i, y_j, z_k + \tfrac{1}{2}h_z). \tag{2.24}
$$

The magnetic field vector $\mathbf{B} = (B_x, B_y, B_z)$,

$$
B_x = \{B_{x;i,j,k} : i,j,k\},\quad B_y = \{B_{y;i,j,k} : i,j,k\},\quad B_z = \{B_{z;i,j,k} : i,j,k\}, \tag{2.25}
$$

where

$$
B_{x;i,j,k} = B_x(x_i, y_j+\tfrac{1}{2}h_y, z_k+\tfrac{1}{2}h_z),\quad B_{y;i,j,k} = B_y(x_i+\tfrac{1}{2}h_x, y_j, z_k+\tfrac{1}{2}h_z),\quad B_{z;i,j,k} = B_z(x_i+\tfrac{1}{2}h_x, y_j+\tfrac{1}{2}h_y, z_k). \tag{2.26}
$$

The boundary conditions impose a constraint on the values of $B_{z;i,j,k}$ when either of the indices $i$ or $j$ is equal to its first or last value; see (2.47), (2.48), and (2.49). Similarly for the other directions. The supercurrent $\mathbf{J}_s = (J_{s,x}, J_{s,y}, J_{s,z})$,

$$
J_{s,x} = \{J_{s,x;i,j,k} : i = n_{sx},\ldots,n_{ex}-1;\; j = n_{sy},\ldots,n_{ey};\; k = n_{sz},\ldots,n_{ez}\}, \tag{2.27}
$$

$$
J_{s,y} = \{J_{s,y;i,j,k} : i = n_{sx},\ldots,n_{ex};\; j = n_{sy},\ldots,n_{ey}-1;\; k = n_{sz},\ldots,n_{ez}\}, \tag{2.28}
$$

$$
J_{s,z} = \{J_{s,z;i,j,k} : i = n_{sx},\ldots,n_{ex};\; j = n_{sy},\ldots,n_{ey};\; k = n_{sz},\ldots,n_{ez}-1\}, \tag{2.29}
$$

where

$$
J_{s,x;i,j,k} = J_{s,x}(x_i+\tfrac{1}{2}h_x, y_j, z_k),\quad J_{s,y;i,j,k} = J_{s,y}(x_i, y_j+\tfrac{1}{2}h_y, z_k),\quad J_{s,z;i,j,k} = J_{s,z}(x_i, y_j, z_k+\tfrac{1}{2}h_z). \tag{2.30}
$$

In Figure 1, we show a typical grid cell $\Omega_{i,j,k}$ in the interior of $\Omega_{sc}$ with the evaluation points for $\psi$ and the components of $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{J}_s$.

Within the framework of a second-order accurate approximation, the definitions (2.20), (2.21), and (2.22) are equivalent with

$$
U_{\alpha;i,j,k} = \exp(-ih_\alpha A_{\alpha;i,j,k}),\quad \alpha = x,y,z, \tag{2.31}
$$

where $A_{\alpha;i,j,k}$ is defined in (2.24). This expression is readily inverted,

$$
A_{\alpha;i,j,k} = -(ih_\alpha)^{-1}\log U_{\alpha;i,j,k},\quad \alpha = x,y,z. \tag{2.32}
$$

The relation (2.32) is the discrete analog of (2.8). It is used to compute the vector potential from the link variables.

> **Remark.** In the literature on discrete GL models, (2.31) is used to define the link variables. The link variables are introduced in an ad hoc fashion to restore gauge invariance, which is normally lost if the partial differential equations of the continuous GL model are discretized by means of finite differences [7,9,18,19]. Here, the link variables arise naturally when the GL functional is formulated in terms of $\mathbf{U}$, as in (2.9), and a consistently second-order accurate approximation is constructed by means of the midpoint rule and central differences.

The computation of the magnetic field is more complicated. We consider one component ($B_z$) in detail; the others are treated similarly.

Let $D$ be any two-dimensional domain that is orthogonal to the $z$ direction; let $\partial D$ denote the (oriented) boundary of $D$, $\mathbf{t}$ the unit tangential vector along $\partial D$. According to Stokes's identity,

$$
\exp\left(-i\iint_D B_z\,dx\,dy\right) = \exp\left(-i\oint_{\partial D} \mathbf{A}\cdot d\mathbf{t}\right). \tag{2.33}
$$

Take $D = \{\mathbf{x} = (x,y,z) \in \Omega : x_i < x < x_{i+1},\, y_j < y < y_{j+1},\, z = z_k\}$. If we approximate the area integral in (2.33) by the midpoint rule and the resulting exponential by the first two terms of its Taylor expansion, we obtain the second-order accurate identity

$$
\exp\left(-i\iint_D B_z\,dx\,dy\right) = 1 - ih_x h_y B_{z;i,j,k}, \tag{2.34}
$$

where $B_{z;i,j,k}$ is defined in (2.26). The exponential of the contour integral in (2.33) is the product of four link variables (or their complex conjugates) associated with $\Omega_{i,j,k}$,

$$
\exp\left(-i\oint_{\partial D} \mathbf{A}\cdot d\mathbf{t}\right) = U_{x;i,j+1,k}^* U_{y;i,j,k}^* U_{x;i,j,k} U_{y;i+1,j,k}. \tag{2.35}
$$

Combining (2.33), (2.34), and (2.35), we obtain $B_{z;i,j,k}$ in terms of the link variables,

$$
B_{z;i,j,k} = \frac{1 - U_{x;i,j+1,k}^* U_{y;i,j,k}^* U_{x;i,j,k} U_{y;i+1,j,k}}{ih_x h_y}. \tag{2.36}
$$

In general, we have

$$
B_{\alpha;i,j,k} = \frac{h_\alpha(1 - W_{\alpha;i,j,k})}{ih_x h_y h_z},\quad \alpha = x,y,z, \tag{2.37}
$$

where we have introduced the abbreviations

$$
W_{x;i,j,k} = U_{y;i,j,k+1}^* U_{z;i,j,k}^* U_{y;i,j,k} U_{z;i,j+1,k}, \tag{2.38}
$$

$$
W_{y;i,j,k} = U_{z;i+1,j,k}^* U_{x;i,j,k}^* U_{z;i,j,k} U_{x;i,j,k+1}, \tag{2.39}
$$

$$
W_{z;i,j,k} = U_{x;i,j+1,k}^* U_{y;i,j,k}^* U_{x;i,j,k} U_{y;i+1,j,k}. \tag{2.40}
$$

Observe that $W_{\alpha;i,j,k}$, like $U_{\alpha;i,j,k}$, is complex valued and unimodular; $W_{\alpha;i,j,k}^* = W_{\alpha;i,j,k}^{-1}$.

Lastly, we consider the supercurrent. The continuous variable is given in terms of the order parameter and link variables in (2.12). According to (2.30), we have

$$
J_{s,x;i,j,k} = \frac{1}{2i}\left[(U_x^*\psi^*)(x_i+\tfrac{1}{2}h_x, y_j, z_k)\cdot\frac{\partial(U_x\psi)}{\partial x}(x_i+\tfrac{1}{2}h_x, y_j, z_k) - (U_x\psi)(x_i+\tfrac{1}{2}h_x, y_j, z_k)\cdot\frac{\partial(U_x^*\psi^*)}{\partial x}(x_i+\tfrac{1}{2}h_x, y_j, z_k)\right]. \tag{2.41}
$$

Here, we approximate the value of $U_x$ at $(x_i+\tfrac{1}{2}h_x, y_j, z_k)$ by the average of its values at $(x_i, y_j, z_k)$ and $(x_{i+1}, y_j, z_k)$ and the value of its derivative by the difference of its values at these points (central difference approximation),

$$
(U_x\psi)(x_i+\tfrac{1}{2}h_x, y_j, z_k) = \tfrac{1}{2}(U_{x;i,j,k}\psi_{i+1,j,k} + \psi_{i,j,k}), \tag{2.42}
$$

$$
\frac{\partial(U_x\psi)}{\partial x}(x_i+\tfrac{1}{2}h_x, y_j, z_k) = \frac{U_{x;i,j,k}\psi_{i+1,j,k} - \psi_{i,j,k}}{h_x}. \tag{2.43}
$$

Hence,

$$
J_{s,x;i,j,k} = \frac{1}{2ih_x}\left(U_{x;i,j,k}\psi_{i,j,k}^*\psi_{i+1,j,k} - U_{x;i,j,k}^*\psi_{i,j,k}\psi_{i+1,j,k}^*\right), \tag{2.44}
$$

which expresses the $x$ component of the supercurrent density in terms of the order parameter and link variables. The $y$ and $z$ components are treated similarly.

### 2.6 Interface and Boundary Conditions

The interface $\partial\Omega_{sc}$ consists of six two-dimensional planar surfaces, orthogonal to the coordinate axes and located at $x = x_{n_{sx}} + \tfrac{1}{2}h_x$ ("left") and $x = x_{n_{ex}-1} + \tfrac{1}{2}h_x$ ("right"), $y = y_{n_{sy}} + \tfrac{1}{2}h_y$ ("front") and $y = y_{n_{ey}-1} + \tfrac{1}{2}h_y$ ("back"), and $z = z_{n_{sz}} + \tfrac{1}{2}h_z$ ("bottom") and $z = z_{n_{ez}-1} + \tfrac{1}{2}h_z$ ("top"); see (2.16). They are surfaces of continuity for the magnetic field. Our definition of the discrete variables accomplishes this continuity to second-order accuracy in the mesh size, because the magnetic field is evaluated at points on the surfaces.

We incorporate the boundary condition $\mathbf{J}_s \cdot \mathbf{n} = 0$ by imposing constraints on the discrete variables. For example, on the left face we impose the constraint

$$
\psi_{n_{sx},j,k} = U_{x;n_{sx},j,k}\psi_{n_{sx}+1,j,k},\quad j = n_{sy},\ldots,n_{ey};\; k = n_{sz},\ldots,n_{ez}, \tag{2.45}
$$

and on the right face,

$$
\psi_{n_{ex},j,k} = U_{x;n_{ex}-1,j,k}^*\psi_{n_{ex}-1,j,k},\quad j = n_{sy},\ldots,n_{ey};\; k = n_{sz},\ldots,n_{ez}, \tag{2.46}
$$

and similarly in the other directions.

The outer boundary $\partial\Omega$ also consists of six two-dimensional planar surfaces, orthogonal to the coordinate axes and located at $x = x_1 + \tfrac{1}{2}h_x$ ("left") and $x = x_{n_x} + \tfrac{1}{2}h_x$ ("right"), $y = y_1 + \tfrac{1}{2}h_y$ ("front") and $y = y_{n_y} + \tfrac{1}{2}h_y$ ("back"), and $z = z_1 + \tfrac{1}{2}h_z$ ("bottom") and $z = z_{n_z} + \tfrac{1}{2}h_z$ ("top"); see (2.15). Here, the induced magnetic field $\mathbf{B}$ must be matched to the applied magnetic field $\mathbf{H} = (H_x, H_y, H_z)$, which is uniform. If there is no surface current, the matching is continuous; otherwise, there is a jump discontinuity in the tangential components. We assume that the surface current density on $\partial\Omega$ is $\mathbf{K}$. We accomplish the matching by imposing constraints on the discrete variables. For example, on the left and right face we impose the constraints

$$
B_{z;1,j,k} = H_z - K_y(x_1+\tfrac{1}{2}h_x, y_j+\tfrac{1}{2}h_y, z_k),\quad B_{z;n_x,j,k} = H_z + K_y(x_{n_x}+\tfrac{1}{2}h_x, y_j+\tfrac{1}{2}h_y, z_k), \tag{2.47}
$$

on the front and back face,

$$
B_{z;i,1,k} = H_z + K_x(x_i+\tfrac{1}{2}h_x, y_1+\tfrac{1}{2}h_y, z_k),\quad B_{z;i,n_y,k} = H_z - K_x(x_i+\tfrac{1}{2}h_x, y_{n_y}+\tfrac{1}{2}h_y, z_k), \tag{2.48}
$$

and on the bottom and top face,

$$
B_{z;i,j,1} = H_z,\quad B_{z;i,j,n_z} = H_z. \tag{2.49}
$$

Similar constraints hold for the $x$ and $y$ components.

### 2.7 Discrete Energy Functional

We consider the various contributions to the free-energy functional. As before, we evaluate all integrals by means of the midpoint rule and approximate derivatives by central differences, so the resulting approximation is second-order accurate as the mesh width goes to zero. We omit the variable $t$ (time is a parameter in the discretization) and use the symbol $\sum_{\text{cyclic}}$ to indicate cyclic permutation according to the scheme

$$
(x,y,z,i,j,k) \to (y,z,x,j,k,i) \to (z,x,y,k,i,j) \to (x,y,z,i,j,k). \tag{2.50}
$$

The condensation energy is readily approximated,

$$
\mathcal{L}_{\text{cond}} \approx \int_{\Omega_{sc}} \left(-|\psi|^2 + \frac{1}{2}|\psi|^4\right) dx\,dy\,dz = \sum_{k=n_{sz}}^{n_{ez}}\sum_{j=n_{sy}}^{n_{ey}}\sum_{i=n_{sx}}^{n_{ex}} \left(-|\psi_{i,j,k}|^2 + \frac{1}{2}|\psi_{i,j,k}|^4\right)h_x h_y h_z. \tag{2.51}
$$

For the kinetic energy, we have

$$
\mathcal{L}_{\text{kin}} \approx \frac{1}{2}\sum_{\alpha=x,y,z}\int_{\Omega_{sc}}\left|\frac{\partial}{\partial\alpha}(U_\alpha\psi)\right|^2 dx\,dy\,dz = \frac{1}{2}\sum_{\text{cyclic}}\sum_{k=n_{sz}}^{n_{ez}}\sum_{j=n_{sy}}^{n_{ey}}\sum_{i=n_{sx}}^{n_{ex}-1} \left|\frac{U_{x;i,j,k}\psi_{i+1,j,k} - \psi_{i,j,k}}{h_x}\right|^2 h_x h_y h_z. \tag{2.52}
$$

Lastly, we have the field energy,

$$
\int_\Omega |\nabla\times\mathbf{A}|^2\,dx\,dy\,dz = \sum_{i,j,k}\sum_{\text{cyclic}} |B_{z;i,j,k}|^2 h_x h_y h_z = \sum_{\text{cyclic}}\sum_{i,j,k} \frac{|1 - W_{z;i,j,k}|^2}{2h_x^2 h_y^2} h_x h_y h_z. \tag{2.53}
$$

The discrete energy functional is therefore

$$
\mathcal{L}_d[\psi, \mathbf{U}] = \sum_{i,j,k}\left(\mathcal{L}_{i,j,k}[\psi,\mathbf{U}] + \sum_{\text{cyclic}}\frac{|1-W_{z;i,j,k}|^2}{2h_x^2 h_y^2}\right)h_x h_y h_z, \tag{2.54}
$$

where

$$
\mathcal{L}_{i,j,k} = -|\psi_{i,j,k}|^2 + \frac{1}{2}|\psi_{i,j,k}|^4 + \frac{1}{2}\sum_{\text{cyclic}}\left|\frac{U_{x;i,j,k}\psi_{i+1,j,k} - \psi_{i,j,k}}{h_x}\right|^2. \tag{2.55}
$$

### 2.8 Equations of Motion

It remains to take the derivatives of the discrete energy functional and derive the analog of (2.6). Because the TDGL model involves a diffusion effect that is nonlocal in terms of $\mathbf{A}$, but local in terms of $\mathbf{U}$, it is most convenient to formulate the discrete TDGL model in terms of the discrete vectors $\psi$ and $\mathbf{U}$. From (2.31) we obtain the relations

$$
\frac{\partial U_{\alpha;i,j,k}}{\partial A_{\alpha;i,j,k}} = -ih_\alpha U_{\alpha;i,j,k},\quad \alpha = x,y,z. \tag{2.56}
$$

The equation of motion for the order parameter is

$$
\frac{\partial\psi_{i,j,k}}{\partial t} = (F[\psi, \mathbf{U}])_{i,j,k}, \tag{2.57}
$$

where

$$
(F[\psi,\mathbf{U}])_{i,j,k} = \left(1 - |\psi_{i,j,k}|^2\right)\psi_{i,j,k} - \frac{1}{2}\sum_{\text{cyclic}}\frac{U_{x;i-1,j,k}^*\psi_{i-1,j,k} + 2\psi_{i,j,k} - U_{x;i,j,k}\psi_{i+1,j,k}}{h_x^2}. \tag{2.58}
$$

The cyclic sum in $F[\psi,\mathbf{U}]$ is a discretization of a weighted Laplacian; the weight at any point is determined by the values of the vector $\mathbf{U}$ at the point itself and its nearest neighbors.

The equation of motion for $U_x$ is

$$
\frac{\partial U_{x;i,j,k}}{\partial t} = -iU_{x;i,j,k}(F_U[\psi,\mathbf{U}])_{i,j,k}, \tag{2.59}
$$

where

$$
(F_U[\psi,\mathbf{U}])_{i,j,k} = \operatorname{Im}\left(\frac{W_{z;i,j,k} - W_{z;i,j-1,k}}{h_y^2} - \frac{W_{y;i,j,k} - W_{y;i,j,k-1}}{h_z^2} + U_{x;i,j,k}\psi_{i,j,k}^*\psi_{i+1,j,k}\right). \tag{2.60}
$$

The equations for $U_y$ and $U_z$ are obtained by cyclic permutation.

The equations (2.57) and (2.59) are integrated by a one-step forward-difference technique, with time step $\Delta t$,

$$
\psi_{i,j,k}(t+\Delta t) = \psi_{i,j,k}(t) + (F[\psi(t),\mathbf{U}(t)])_{i,j,k}\,\Delta t, \tag{2.61}
$$

$$
U_{x;i,j,k}(t+\Delta t) = U_{x;i,j,k}(t)\exp\left(-i(F_U[\psi(t),\mathbf{U}(t)])_{i,j,k}\,\Delta t\right), \tag{2.62}
$$

and similarly for the other components. The choice of a one-step procedure is the result of a compromise. Clearly, a multistep algorithm yields more accuracy, but requires the storage of two or more generations of computed data. This requirement poses an undesirable limitation on the size of the problems that we would like to investigate. In our simulations, we compensate for the limited accuracy by keeping the time step $\Delta t$ sufficiently small.

Because the problems of interest are diffusion-driven, rather than convection-driven, (typically, the vortex density is high, and the transport currents are relatively weak), the stability of the algorithm is determined primarily by the rate of diffusion, which in turn depends on the Ginzburg-Landau parameter: a larger value of $\kappa$ requires a smaller time step. In our simulations, we choose the time step experimentally, taking into account previous experience.

### 2.9 Generalizations

The numerical approximation of the TDGL model described in the preceding section provides the skeleton for a computational algorithm for vortex dynamics simulations. Because simulations of realistic high-$T_c$ superconductors require the incorporation of several other physically relevant effects, the actual implementation of the computational model has additional features, which we now describe briefly.

#### 2.9.1 Impurities and Material Defects

An important feature of high-$T_c$ superconducting materials is the presence of impurities and material defects. Impurities and point defects are included in the computational model by a local suppression of the condensation energy. This suppression is accomplished by replacing the term $-|\psi_{i,j,k}|^2$ in the GL functional density (2.55) by a term $(-1+\varepsilon(\mathbf{x}_{i,j,k}))|\psi_{i,j,k}|^2$. If the grid point $\mathbf{x}_{i,j,k}$ is not on a defect, $\varepsilon(\mathbf{x}_{i,j,k}) = 0$; otherwise, $\varepsilon(\mathbf{x}_{i,j,k})$ is a random variable, which is chosen from a Gaussian distribution with a specified mean value $\langle\varepsilon\rangle$ in the range $(0, \tfrac{1}{2})$ and a standard deviation $\sigma_d$. The mean and standard deviation are uniform for all defects. A twin boundary, which is a plane defect of a certain thickness, is modeled by specifying the location and number of lattice planes inside the region corresponding to the twin boundary. For each grid point $\mathbf{x}_{i,j,k}$ on the twin boundary, we use the coefficient $-1+\varepsilon(\mathbf{x}_{i,j,k})$ in the GL functional. In the actual implementation, we consider only planes whose generators are parallel to the $z$ axis, thus allowing planes parallel to the $(y,z)$- or $(z,x)$-plane, as well as diagonal planes whose orientation is determined by the mesh aspect ratio $h_y/h_x$.

#### 2.9.2 Transport Currents

Transport currents are incorporated by means of an applied surface current at the outer boundary. In the actual implementation, we allow for surface currents on four faces only (left, right, top, and bottom); moreover, if there is a surface current on any of these four faces, we assume that the domain is infinite in one of the coordinate directions spanning this face plane and that the current is in the infinite direction. Thus, we always model a section of a current path, not a current loop.


---

## 3 Numerical Simulations

An issue of interest is the pattern of flux penetration through edge barriers. Recent experiments indicate that, as magnetic flux penetrates into a sample, vortices tend to concentrate along twin boundaries [20-23]; however, it is still an open question whether twin boundaries actually enhance vortex entry and guide vortex motion, or whether they simply absorb vortices from the bulk of the sample. We explore the issue computationally in Section 3.1.

Magnetic flux vortices of opposite polarity attract and annihilate each other, so when the direction of the applied field is reversed, vortices of opposite polarity enter the sample and annihilate existing vortices. Experiments indicate an irregular interface between the regions of opposite polarity [24]. Simulations show that the TDGL model captures the roughness and irregularity of the interface; see Section 3.2.

A third numerical simulation concerned pattern formation and the spawning of vortices as a magnetic field penetrates into a superconducting sample. A specific issue is whether, in a pure sample of rectangular shape, the avoidance of the corners by the field is primarily an electromagnetic effect. Some results are presented in Section 3.3.

Finally, in Section 3.4, we present some results of long-time simulations to show that the symmetry of an intermediate metastable state can differ from the symmetry of the thermodynamical equilibrium state.

A parallel program to implement the discrete TDGL model was written in Fortran [25] and run on the IBM POWERparallel SP System at Argonne National Laboratory (128 processors, 128 megabytes per processor, theoretical peak performance 16 Gflops). It is based on the distributed-memory programming model and uses the BlockComm library (part of PETSc [26]) to handle the decomposition of the unknowns across the parallel computer and the communication between the processes. The BlockComm library provides support for the computational stencil described in Section 2. The Chameleon library [27] is used to provide both the portable message-passing interface and the scalable parallel I/O.

In all simulations, we took $\kappa = 4$ and $\sigma = 1$.

### 3.1 Twin Boundaries

The purpose of the first numerical simulation was to study vortex entry in the presence of twin boundaries. We adopted the model of a very thick sample of superconducting material surrounded by a single layer of normal metal on all sides. We assumed no variation in the longitudinal direction, so the problem was strictly two-dimensional (coordinates $x$ and $y$). To adequately resolve the vortices, we chose the mesh widths $h_x = h_y = 0.125$. With $\kappa = 4$ and the penetration length $\lambda$ as the unit of length, this choice corresponded to a mesh width of one-half of a coherence length ($\xi$) in both directions. The sample size was $32 \times 48$ units ($\lambda$), resulting in a grid of $256 \times 384$ points. A pair of twin boundaries, separated by 15 units (120 mesh widths), was placed at $45^\circ$ angles to the edges of the sample. Each twin boundary had a thickness of 2.5 coherence lengths (5 mesh widths); all defects in the twin boundaries had a mean strength $\langle\varepsilon\rangle = 0.25$, with a standard deviation of $\sigma_d = 0.125$.

After establishing the Meissner state with an applied magnetic field $H_z = 0.1$ (i.e., in dimensional units, $H_z = 0.1 \times H_c\sqrt{2} \approx 0.14 H_c$) in the $z$ direction, we increased the strength of the applied field to $H_z = 1.5$ ($\approx 2.12 H_c$). The vortex pattern (i.e., $|\psi|$) at various successive times is shown in Figure 2.

In general, we observe that the vortices tend to avoid the corners. In simulations with thin samples (not shown), the avoidance is even more pronounced, and the flux entry pattern becomes more pillow-like. This phenomenon has also been observed in magneto-optical experiments and can be explained by electrodynamic considerations.

The simulations provide insight into the effect of twin boundaries on the transport
properties of superconducting materials. Measurements of Kwok et al. taken at Argonne [20]
show a sharp drop in the angular dependence of the resistivity in the thermally assisted flux
flow (TAFF) regime when the magnetic field is aligned with the twin boundaries. These
observations suggest that vortex motion along twin boundaries is suppressed and that twin
boundaries enhance vortex pinning, even in the case where vortices move along the twins.
On the other hand, magneto-optical data on flux penetration gathered by researchers at
AT&T [21] show a higher concentration of the magnetic field near twin boundaries. These
observations suggest that twin boundaries actually enhance flux penetration. In this scenario,
twin boundaries act as channels for "easy vortex motion," and vortex pinning is reduced along
twin boundaries. More recent magneto-optical experiments at Argonne [22] hint at an even more
complicated picture including shadow effects, where the vortex concentration is enhanced on
one side of a twin boundary and reduced on the other. The experimental situation has been
clarified recently by Welp et al. [23], who showed that the two types of behavior occur in
different limits, when the direction of vortex motion is parallel or perpendicular to the
twin-boundary plane.

Our simulations reveal a rather complicated picture of vortex motion and support aspects of
all the above observations. Twin boundaries modeled as a plane containing point defects tend
to impede vortex motion in the tangential direction, so twin boundaries cannot be viewed as
mere conduits for vortices. Where the twin boundaries meet the surface of the sample, vortex
entry is enhanced. (This observation is consistent with the fact that the Bean-Livingstone
surface barrier is destroyed there.) Vortices trapped on the twin boundaries near the surface
are impeded in their motion along the twin boundary; they leave the twin boundary, pass into
the bulk within a few penetration depths from the surface, and get absorbed later by the twin
boundary. As the twin boundaries become saturated (at vortex densities that are generally
higher than in the bulk), vortices pass directly into the bulk. Therefore, one can indeed
observe higher vortex concentrations, which, at the initial stage of the penetration process,
can be interpreted as enhanced vortex penetration. But the actual vortex motion along the
twin boundaries is inhibited by the point defects on those twin boundaries. In the long term,
there is very little distortion of the flux penetration pattern.

Thermal fluctuations and bulk pinning, not included in these simulations, bring additional
features to this picture of vortex motion. As was found recently by W. Kwok (personal
communication), the motion of vortex lines trapped by the twin boundaries is favored compared
with motion in the bulk in the regime of thermally activated creep. This observation is in
qualitative agreement with collective creep theory, since the enhanced pinning within the
twin plane gives rise to smaller creep barriers for vortex motion. We are currently
investigating this feature numerically.

### 3.2 Magnetic Polarity Reversal

The purpose of the next series of simulations was to study the interface between regions of
opposite magnetic polarity. Magnetic flux vortices of opposite polarity attract and annihilate
each other, so when the direction of the applied field is reversed, vortices of opposite
polarity enter the sample and annihilate existing vortices. Experiments indicate an irregular
interface between the regions of opposite polarity [24]. The roughness is caused by a
thermodynamic instability; since the domain interface has negative surface energy, it is
unstable. As its length increases, the interface tends to break up and spawn regions of one
polarity enclosed by regions of the opposite polarity.

In our simulations, we used the configuration of Section 3.1. We first established an
equilibrium state for an applied magnetic field with $H_z = 1.5$ ($\approx 2.12 H_c$).
Subsequently, we reversed the orientation of the applied magnetic field, choosing
$H_z = -0.8$ ($\approx -1.13 H_c$).
Figure 3 gives a snapshot of $B_z$ at a particular instant. The polarity is down in the very
dark region, up in the light region. We clearly observe that the rough and inclusive nature of
the polarity interface is an intrinsic feature of the Ginzburg-Landau model. Consequently,
it is not necessary to use specially designed fluid models to account for the rough interfaces
observed in magneto-optic experiments.

### 3.3 Flux Entry Patterns

Magneto-optic experiments show a pillow-like pattern for the magnetic field penetrating a
pure superconducting sample. The avoidance of the corners by the field has been shown to be
primarily an electromagnetic effect. The goal of our numerical simulations was to show that
pillow-like patterns are indeed generated by a GL model and, furthermore, to analyze the
early penetration patterns in more detail.

We used a three-dimensional model, where a very small and very thin rectangular
superconducting sample was embedded in a block of normal metal. The dimension of the
superconducting sample was $5 \times 5 \times 0.5$ units ($\lambda$). With $\kappa = 4$ and
a mesh width of one-half of a coherence length in each direction, the superconducting sample
requires a computational grid with $40 \times 40 \times 4$ grid points. The thickness of the
layer of normal metal surrounding the superconducting sample was 2.25 units in each
direction, sufficiently large to resolve the magnetic field in the vicinity of the sample
edges. The complete computational grid had therefore $60 \times 60 \times 24$ grid points.

Starting from the Meissner state, we applied a magnetic field with $H_z = 1.8$
($\approx 2.55 H_c$) and integrated the TDGL model forward in time with a time step
$\Delta t = 0.0025$. Figure 4 shows the entry pattern of the induced magnetic field ($B_z$)
in the mid-plane of the superconducting sample at four successive moments.

**(a) $t = 3.0$.** The dark region in the center is a region where the field has not yet
penetrated. The grey regions show where the field has penetrated, and the light-colored ring
separates a region of high field density (outside the ring) from a region of low field density
(inside the ring). The field has penetrated the entire sample, including the corner regions.

**(b) $t = 8.0$.** The region where the field has penetrated is squared off, and the flux is
beginning to be excluded from the diagonals.

**(c) $t = 11.0$.** The flux is excluded from the diagonals, and a pillow-shaped pattern is
emerging. The pattern is similar to the one observed in the magneto-optic experiments.

**(d) $t = 13.75$.** The flux is further excluded from the diagonals and individual flux tubes
are created.

The pillow pattern is clearly exhibited by the GL model; however, its formation is not simply
due to the field penetrating only on the sides. At least in the case considered here, the field
first penetrates everywhere in the sample and is subsequently excluded from the diagonals.
Further simulations confirmed that, in accordance with physical arguments, the pillow-shaped
pattern is a common feature in larger samples in three dimensions as well.

We note that the original images are in color, with high densities in red, intermediate
densities in yellow, and low densities in green to blue. In the grey-tone representation, from
which the figures are drawn, yellow yields light, blue yields dark, while red and green yield
various shades of grey. The particular choice of the color map and the transition from color
to gray scale introduce some loss of information. This loss causes the interface to appear as
a surface of discontinuity, especially for $t = 8.0$ and beyond. However, the magnetic field
is continuous at the interface.

### 3.4 Symmetry

In the final configuration of Figure 4(d), taken at $t = 13.75$, the 16 flux tubes were
arranged in a pattern with $45^\circ$ rotational symmetry. We continued the computations to
study the long-time evolution of the magnetic flux configuration. Subsequent flux patterns are
shown in Figure 5.

**(a) $t = 25.0$.** The pattern of Figure 4(d) still persists.

**(b) $t = 200.0$.** The pattern is distorted, and the symmetry is broken.

**(c) $t = 325.0$.** The original pattern is destroyed, and a new pattern is emerging.

**(d) $t = 600.0$.** A new pattern is established. It has a $90^\circ$ rotational symmetry;
the 16 flux tubes are arranged in a rectangular array.

The simulations show that, in the process of penetration, a metastable state may be reached,
whose symmetry properties are different from those of the final equilibrium state. This
phenomenon of symmetry breaking during the dynamic evolution of the system toward its
equilibrium state has important practical consequences. Suppose, for example, that the geometry
of the system dictates that the equilibrium solution possess certain symmetries. One might then
wish to exploit these symmetries during the transient calculation. However, such a strategy
could (and probably would) lead to a quasi-equilibrium solution, since one cannot assume that
the symmetries in the TDGL model will not be broken in the course of the evolution toward the
equilibrium state.

---

## Acknowledgments

We acknowledge the various contributions made by our colleagues Drs. Man Kam Kwong (ANL) and
Salman Ullah (U. of Chicago) during the early stages of this work. We thank our students
Ibrahima Ba, Andrew Burdick, Nicholas Galbreath, David Gunter, and Todd Morgan for their help
with program development, visualization, and some of the calculations. The numerical
simulations were designed in collaboration with Drs. George W. Crabtree and Alexei E. Koshelev
(ANL). All computations were done at the Argonne High-Performance Computing Research Facility;
the Argonne HPCRF is funded principally by the U.S. Department of Energy, Office of Scientific
Computing.

---

## References

[1] M. Tinkham, *Introduction to Superconductivity*, R. E. Krieger Publ. Co., Malabar, Florida, 1975.

[2] A. A. Abrikosov, *Fundamentals of the Theory of Metals*, North-Holland, Amsterdam, 1988.

[3] A. Schmid, Phys. Kondens. Mater. **5** (1966), 302.

[4] L. P. Gor'kov & G. M. Eliashberg, "Generalizations of the Ginzburg-Landau equations for
non-stationary problems in the case of alloys with paramagnetic impurities," Zh. Eksp. Teor.
Fiz. **54** (1968), 612-626, Soviet Phys.-JETP **27** (1968), 328-334.

[5] J. D. Jackson, *Classical Electrodynamics*, John Wiley & Sons, New York, NY, 1975.

[6] P. DeGennes, *Superconductivity in Metals and Alloys*, Benjamin, New York, NY, 1966.

[7] K. J. M. Moriarty, E. Myers & C. Rebbi, "A vector code for the numerical simulation of
cosmic strings and flux vortices in superconductors on the ETA-10," Computer Phys.
Communications **54** (1989), 273-294.

[8] F. Liu, M. Mondello & N. Goldenfeld, Phys. Rev. Lett. **66** (1991), 3071.

[9] H. Frahm, S. Ullah & A. T. Dorsey, "Flux dynamics and the growth of the superconducting
phase," Phys. Rev. Lett. **66** (1991), 3067-3070.

[10] S. Maekawa, R. Kato & Y. Enomoto, "Computer simulations of vortex states in a
superconducting film," Research Report on Mechanism of Superconductivity, n. d., 6 pages.

[11] M. Machida & H. Kaburaki, "Direct simulation of the time-dependent Ginzburg-Landau
equation for type-II superconducting thin film: Vortex dynamics and V-I characteristics,"
Phys. Rev. Lett. **71** (1993), 3206-3209.

[12] Q. Du, M. D. Gunzburger & J. S. Peterson, "Analysis and approximation of the
Ginzburg-Landau model of superconductivity," SIAM Review **34** (1992), 54-81.

[13] Q. Du, "Global existence and uniqueness of solutions of the time-dependent
Ginzburg-Landau model for superconductivity," *Applicable Analysis*, (to appear).

[14] Q. Tang & S. Wang, "Time dependent Ginzburg-Landau equations of superconductivity,"
1995, preprint.

[15] J. B. Kogut, "An introduction to lattice gauge theory and spin systems," Rev. Mod.
Phys. **51** (1979), 659-713.

[16] J. B. Kogut, "The lattice gauge theory approach to quantum chromodynamics," Rev. Mod.
Phys. **55** (1983), 775-836.

[17] H. G. Kaper & M. K. Kwong, "Vortex configurations in type-II superconducting films,"
Journal of Computational Physics **118** (1995), 000-000.

[18] S. L. Adler & T. Piran, "Relaxation methods for gauge field equilibrium equations,"
Rev. Mod. Phys. **56** (1984), 1-40.

[19] M. M. Doria, J. E. Gubernatis & D. Rainer, "Solving the Ginzburg-Landau equations by
simulated annealing," Phys. Rev. B **41** (1990), 6335-6340.

[20] W. K. Kwok, U. Welp, G. W. Crabtree, K. G. Vandervoort, R. Hulscher & J. Z. Liu,
"Direct observation of dissipative flux motion and pinning by twin boundaries in
YBa$_2$Cu$_3$O$_{7-\delta}$ single crystals," Phys. Rev. Lett. **64** (1990), 966-969.

[21] C. Duran, P. L. Gammel, R. Wolfe, V. J. Fratello, D. J. Bishop, J. P. Rice & D. M.
Ginsberg, "Real-time imaging of the magnetic flux distribution in superconducting
YBa$_2$Cu$_3$O$_{7-\delta}$," Nature **357** (1992), 474.

[22] V. K. Vlasko-Vlasov, L. A. Dorosinskii, A. A. Polyanskii, V. I. Nikitenko, U. Welp, B. W.
Veal & G. W. Crabtree, "Study of the influence of individual twin boundaries on the magnetic
flux penetration in YBa$_2$Cu$_3$O$_{7-\delta}$," Phys. Rev. Lett. **72** (1994), 3246-3249.

[23] U. Welp, T. Gardiner, D. Gunter, J. Fendrich, G. W. Crabtree, V. K. Vlasko-Vlasov &
V. I. Nikitenko, Physica C **235-240** (1994), 241-244, Proc. IVth Int'l Conf. on Materials and
Mechanisms of Superconductivity - High-Temperature Superconductors, Grenoble (France),
July 5-9, 1994.

[24] V. K. Vlasko-Vlasov, V. I. Nikitenko, A. A. Polyanskii, G. W. Crabtree, U. Welp & B. W.
Veal, "Macroturbulence in high-$T_c$ superconductors," Physica C **222** (1994), 361-366.

[25] N. Galbreath, W. D. Gropp, D. Gunter, G. K. Leaf & D. Levine, "Parallel solution of the
three-dimensional time-dependent Ginzburg-Landau equation," in Proc. SIAM Conf. on Parallel
Processing for Scientific Computing, SIAM, Philadelphia, 1993.

[26] W. D. Gropp & B. F. Smith, "Scalable, Extensible, and Portable Numerical Libraries,"
in Proc. Conf. on Scalable Parallel Libraries, IEEE, 1994, 87-93.

[27] W. D. Gropp & B. F. Smith, "Chameleon parallel programming tools users manual,"
Argonne National Laboratory, Report ANL-93/23, 1993.

---

## Figure Captions

**Figure 1.** Evaluation points for $\psi$ (circle), $A_x$ and $J_{s,x}$ (X), $A_y$ and $J_{s,y}$ (Y),
$A_z$ and $J_{s,z}$ (Z), $B_x$ (right arrow), $B_y$ (triangle up), $B_z$ (triangle)

**Figure 2.** Successive vortex-entry patterns ($|\psi|$) in twin boundary experiment

**Figure 3.** Induced magnetic field ($B_z$) in polarity reversal simulation

**Figure 4.** Induced magnetic field ($B_z$) during the transient phase:
(a) $t = 3.0$; (b) $t = 8.0$; (c) $t = 11.0$; and (d) $t = 13.75$

**Figure 5.** Induced magnetic field ($B_z$) during the evolution toward equilibrium:
(a) $t = 25.0$; (b) $t = 200.0$; (c) $t = 325.0$; and (d) $t = 600.0$; units as in Figure 4

