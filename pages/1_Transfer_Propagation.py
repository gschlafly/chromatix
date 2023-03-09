import streamlit as st

import jax.numpy as jnp
import matplotlib.pyplot as plt

# from chromatix import Field, OpticalSystem, Microscope
import chromatix.functional as cx

# from chromatix.ops.fft import optical_fft

# from typing import Optional

st.title("Fresnel Transfer Propagation")
st.latex(
    r"""
    U(x,y,z) = \mathcal{F}^{-1}\{  
        \mathcal{F} \{
        U(x,y,0) \times e^{jkz} \exp{(-j \pi \lambda z (f_x^2 + f_y^2))}
        \}
        \}
    """
)


# Functions
def retrieve_micron_dims(field):
    um_length_x = field.shape[-2] * field.dx.squeeze()
    um_length_y = field.shape[-3] * field.dx.squeeze()
    return [-um_length_x / 2, um_length_x / 2, -um_length_y / 2, um_length_y / 2]


def add_intensity_to_axes(field, ax, extent):
    um_length_x = field.shape[-2] * field.dx.squeeze()
    um_length_y = field.shape[-3] * field.dx.squeeze()
    ax.imshow(field.intensity.squeeze(), extent=extent)
    ax.set_xlabel("microns")
    ax.set_ylabel("microns")
    ax.set_xlim([-um_length_x / 2, um_length_x / 2])
    ax.set_ylim([-um_length_y / 2, um_length_y / 2])
    return ax


# spacing = st.radio("Sampling spacing (microns)", options=[0.001, 0.01, 0.1, 1], index=1)


st.subheader("Plane Wave")

# z = 100
# spectrum = 0.532
# n = 1.33
# Nf = (D / 2) ** 2 / (spectrum / n * z)


# source_field = VectorPlaneWave(shape=(512, 512), dx = 0.0001, n = 1, spectrum=spectrum, spectral_density=1.0, k = k, Ep = Ep)


# field = cx.empty_field((N, N), dxi, 0.532, 1.0, polarized=True)
# plane_wave_field = cx.plane_wave(field, pupil=lambda field: cx.square_pupil(field, dxi * N))
col1_params, col2_params = st.columns(2)

with col1_params:
    wavelength = st.radio("Wavelength (microns)", options=[0.532])
    n_medium = st.radio("Index of refraction of the medium", options=[1.0, 1.33, 1.52])
    vectorial = st.checkbox("Vectorial field", value=False)

with col2_params:
    N = st.radio("Image size (pixel length)", options=[128, 256, 512], index=1)
    Q = st.slider(
        "Multiples of image size to pad with", min_value=0, max_value=10, value=5
    )
    N_pad = Q * N
    st.text(f"Padding with {N_pad} pixels")

D = 40
N = 256
# dxi = D / N
Q = 5
N_pad = Q * N

phi = 0  # angle between z axis and xy plane
theta = 0  # angle between x and y
k = (
    n_medium
    * 2
    * jnp.pi
    / wavelength
    * jnp.array([jnp.sin(phi) * jnp.sin(theta), jnp.sin(phi) * jnp.cos(theta)])
)  # y and x
Ep = jnp.array((1, 0, 0))

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Source field")
    spacing = st.selectbox("Sampling spacing (microns)", (D / N, 0.001, 0.01, 0.1, 1))
    # spacing = st.radio(
    #     "Sampling spacing (microns)", options=[D / N, 0.001, 0.01, 0.1, 1], index=1
    # )
    field = cx.empty_field((N, N), spacing, wavelength, n_medium, polarized=True)
    source_field = cx.vector_plane_wave(
        field, k=k, Ep=Ep, pupil=lambda field: cx.circular_pupil(field, spacing * N)
    )
    extent = retrieve_micron_dims(source_field)
    fig, ax = plt.subplots(1, 1)
    ax = add_intensity_to_axes(source_field, ax, extent)
    st.pyplot(fig)

with col2:
    st.subheader("Propogation #1")
    prop_dist1 = st.slider(
        "Propagation distance 1 (microns)", min_value=1, max_value=200, value=1
    )
    transfer_field1 = cx.transfer_propagate(
        source_field, z=prop_dist1, n=n_medium, N_pad=N_pad, mode="same"
    )
    fig, ax = plt.subplots(1, 1)
    ax = add_intensity_to_axes(transfer_field1, ax, extent)
    # fig.colorbar(ax)
    # cbar = fig.colorbar(transfer_field1, ax=ax)
    st.pyplot(fig)

with col3:
    st.subheader("Propogation #2")
    prop_dist2 = st.slider(
        "Propagation distance 2 (microns)", min_value=1, max_value=200, value=10
    )
    transfer_field2 = cx.transfer_propagate(
        source_field, z=prop_dist2, n=n_medium, N_pad=N_pad, mode="same"
    )
    fig, ax = plt.subplots(1, 1)
    ax = add_intensity_to_axes(transfer_field2, ax, extent)
    st.pyplot(fig)

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

extent = retrieve_micron_dims(source_field)
axs[0] = add_intensity_to_axes(source_field, axs[0], extent)
axs[1] = add_intensity_to_axes(transfer_field1, axs[1], extent)
axs[2] = add_intensity_to_axes(transfer_field2, axs[2], extent)
fig.tight_layout()

axs[0].set_title("Plane wave source")
axs[1].set_title("Propogated 10 um with Transfer")
axs[2].set_title("Propogated 100 um with Transfer")
# fig.title("Plane wave propogated with Transform")
plt.text(
    x=0.5,
    y=0.94,
    s="Plane wave propogated with Transfer",
    fontsize=18,
    ha="center",
    transform=fig.transFigure,
)
# plt.text(x=0.5, y=0.88, s= "Holding micron size constant and matching test settings with the 100 um", fontsize=12, ha="center", transform=fig.transFigure)
# plt.subplots_adjust(top=0.9, wspace=0.3)

plt.show()
