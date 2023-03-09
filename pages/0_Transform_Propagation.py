import streamlit as st

# import jax.numpy as jnp
import matplotlib.pyplot as plt
from chromatix import Field, OpticalSystem, Microscope
import chromatix.functional as cx
from chromatix.ops.fft import optical_fft

from typing import Optional

st.title("Transform Propagation")


st.latex(
    r"""
    U(x,y,z) = 
        \mathcal{F}^{-1}\{ 
        \mathcal{F}\{ 
        U(\xi, \eta, 0)e^{j\frac{\pi}{\lambda z}(\xi^2+\eta^2)} 
        e^{j\frac{\pi}{\lambda z}(x\xi+y\eta)}
        \}
        \}
    """
)


def intensity_images_cropped(field_list):
    fig, axs = plt.subplots(1, len(field_list), figsize=(12, 5))
    um_length_x0 = field_list[0].shape[2] * field_list[0].dx.squeeze()
    um_length_y0 = field_list[0].shape[1] * field_list[0].dx.squeeze()
    for i, field in enumerate(field_list):
        um_length_x = field.shape[2] * field.dx.squeeze()
        um_length_y = field.shape[1] * field.dx.squeeze()
        extent = [-um_length_x / 2, um_length_x / 2, -um_length_y / 2, um_length_y / 2]
        im = axs[i].imshow(field.intensity.squeeze(), extent=extent)
        axs[i].set_xlabel("microns")
        axs[i].set_ylabel("microns")
        axs[i].set_xlim([-um_length_x0 / 2, um_length_x0 / 2])
        axs[i].set_ylim([-um_length_y0 / 2, um_length_y0 / 2])
    fig.tight_layout()
    return fig, axs


# spacing = st.selectbox("Sampling spacing (microns)", (0.001, 0.01, 0.1, 1))
spacing = st.radio("Sampling spacing (microns)", options=[0.001, 0.01, 0.1, 1], index=1)
wavelength = st.radio("Wavelength (microns)", options=[0.532])
n_medium = st.radio("Index of refraction of the medium", options=[1.0, 1.33, 1.52])
vectorial = st.checkbox("Vectorial field", value=False)
# vectorial = st.radio("Vectorial field", options=["True", "False"], index=1)
pt_source_prop_dist = st.slider(
    "Distance away from point source (microns)", min_value=1, max_value=200, value=100
)
prop_dist1 = st.slider(
    "Propagation distance 1 (microns)", min_value=1, max_value=200, value=1
)
prop_dist2 = st.slider(
    "Propagation distance 2 (microns)", min_value=1, max_value=200, value=10
)

st.subheader("Point source")
field = cx.empty_field(
    shape=(512, 512),
    dx=spacing,
    spectrum=wavelength,
    spectral_density=1.0,
    polarized=vectorial,
)
pt_source_field = cx.point_source(field, z=pt_source_prop_dist, n=n_medium)
transform_field1 = cx.transform_propagate(
    pt_source_field, z=prop_dist1, n=n_medium, N_pad=512
)
transform_field2 = cx.transform_propagate(
    pt_source_field, z=prop_dist2, n=n_medium, N_pad=512
)
fig, axs = intensity_images_cropped(
    [pt_source_field, transform_field1, transform_field2]
)
axs[0].set_title("Point source")
axs[1].set_title(f"Propogated {prop_dist1} um with Transform")
axs[2].set_title(f"Propogated {prop_dist2} um with Transform")
# plt.show()
st.pyplot(fig)

D = 40
z = 100
spectrum = 0.532
n = 1.33
Nf = (D / 2) ** 2 / (spectrum / n * z)

N = 256
dxi = D / N
Q = 5
N_pad = Q * N

st.subheader("Plane wave")
field = cx.empty_field((N, N), dxi, 0.532, 1.0)
plane_wave_field = cx.plane_wave(
    field, pupil=lambda field: cx.circular_pupil(field, dxi * N)
)
transform_field_1 = cx.transform_propagate(plane_wave_field, z=10, n=n, N_pad=N_pad)
transform_field_10 = cx.transform_propagate(plane_wave_field, z=100, n=n, N_pad=N_pad)
fig, axs = intensity_images_cropped(
    [plane_wave_field, transform_field_1, transform_field_10]
)
axs[0].set_title("Plane wave source")
axs[1].set_title("Propogated 10 um with Transform")
axs[2].set_title("Propogated 100 um with Transform")

plt.text(
    x=0.5,
    y=0.94,
    s="Plane wave propogated with Transform",
    fontsize=18,
    ha="center",
    transform=fig.transFigure,
)
plt.text(
    x=0.5,
    y=0.88,
    s="Holding micron size constant and matching test settings with the 100 um",
    fontsize=12,
    ha="center",
    transform=fig.transFigure,
)
plt.subplots_adjust(top=0.9, wspace=0.3)
st.pyplot(fig)
