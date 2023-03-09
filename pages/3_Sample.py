import streamlit as st

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from chromatix.functional.samples import jones_sample

# from skimage import io
# import cv2
from einops import rearrange

# from chromatix import Field, OpticalSystem, Microscope
import chromatix.functional as cx

# from chromatix.ops.fft import optical_fft

# from typing import Optional

st.title("Propagation of polarized light")
st.markdown("Fresnel transfer propagation")
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


def add_intensity_to_axes(fig, field, ax, extent):
    um_length_x = field.shape[-2] * field.dx.squeeze()
    um_length_y = field.shape[-3] * field.dx.squeeze()
    im = ax.imshow(field.intensity.squeeze(), extent=extent)
    fig.colorbar(im, orientation="vertical")
    ax.set_xlabel("microns")
    ax.set_ylabel("microns")
    ax.set_xlim([-um_length_x / 2, um_length_x / 2])
    ax.set_ylim([-um_length_y / 2, um_length_y / 2])
    return fig


st.subheader("Optical parameters")
# source_field = VectorPlaneWave(shape=(512, 512), dx = 0.0001, n = 1, spectrum=spectrum, spectral_density=1.0, k = k, Ep = Ep)


# field = cx.empty_field((N, N), dxi, 0.532, 1.0, polarized=True)
# plane_wave_field = cx.plane_wave(field, pupil=lambda field: cx.square_pupil(field, dxi * N))
col1_params, col2_params = st.columns(2)

with col1_params:
    wavelength = st.radio("Wavelength (microns)", options=[0.532])
    N = st.radio("Image size (pixel length)", options=[128, 256, 512], index=2)
    D = 40
    spacing = st.selectbox("Sampling spacing (microns)", (D / N, 0.001, 0.01, 0.1, 1))

with col2_params:
    n_medium = st.radio(
        "Index of refraction of the medium", options=[1.0, 1.33, 1.52], horizontal=True
    )
    Q = st.slider(
        "Multiples of image size to pad with", min_value=0, max_value=10, value=5
    )
    N_pad = Q * N
    st.text(f"Padding with {N_pad} pixels")
    prop_dist1 = st.slider(
        "Propagation distance (microns)", min_value=1, max_value=200, value=100
    )


# N = 256
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
# Ep = jnp.array((1, 0, 0))
Ep = jnp.array((0, 1, 1))

st.markdown(
    "**The light source is a polarized wavefront. After "
    + "propagating, the light passing a sample and then through a linear polarizer.**"
)

sample_delay = np.load("Map_512.npy")
# st.write(sample_delay)

col1, col2 = st.columns(2)

with col1:
    # st.subheader("Polarized source field")
    st.markdown("###### Field after Generator")
    polarizer_angle = st.slider(
        "Linear polarizer angle (degrees)", min_value=0, max_value=90, step=5
    )
    polarizer_angle_frac = polarizer_angle / 90
    # spacing = st.radio(
    #     "Sampling spacing (microns)", options=[D / N, 0.001, 0.01, 0.1, 1], index=1
    # )
    field = cx.empty_field((N, N), spacing, wavelength, n_medium, polarized=True)
    source_field = cx.vector_plane_wave(
        field, k=k, Ep=Ep, pupil=lambda field: cx.circular_pupil(field, spacing * N)
    )
    source_field = cx.linear_polarizer(source_field, polarizer_angle_frac * jnp.pi / 2)
    # source_field = cx.linear_polarizer(source_field, filter_angle_frac * jnp.pi / 2)
    extent = retrieve_micron_dims(source_field)
    fig, ax = plt.subplots(1, 1)
    fig = add_intensity_to_axes(fig, source_field, ax, extent)
    # im = ax.imshow(source_field.intensity.squeeze(), extent=extent)
    # fig.colorbar(im, orientation="vertical")
    # ax.set_xlabel("microns")
    # ax.set_ylabel("microns")
    st.pyplot(fig)

with col2:
    st.markdown("###### Field after Analyzer")
    analyzer_angle = st.slider(
        "Linear analyzer angle (degrees)", min_value=0, max_value=90, step=5
    )
    analyzer_angle_frac = analyzer_angle / 90
    transfer_field1 = cx.transfer_propagate(
        source_field, z=prop_dist1, n=n_medium, N_pad=N_pad, mode="same"
    )
    # sample_delay = np.load("Map_512.npy")
    absorption = jnp.ones((1, 2, 2, 512, 512, 1))  # [1 2 2 H W 1]
    # absorption = jnp.zeros((1, 2, 2, 512, 512, 1))  # [1 2 2 H W 1]
    phase = jnp.einsum(
        "ijmn, mn -> ijmn",
        jnp.ones((2, 2, 512, 512)),
        sample_delay / jnp.max(jnp.max(sample_delay)),
    )  # [1 2 2 H W 1]
    phase = rearrange(phase, "i j m n -> 1 i j m n 1")
    # field = transfer_field1
    sample_field1 = jones_sample(transfer_field1, absorption, phase)
    # transfer_field1 = cx.Jones_sample(transfer_field1, absorption, phase)
    sample_field1 = cx.linear_polarizer(sample_field1, analyzer_angle_frac * jnp.pi / 2)
    fig1, ax1 = plt.subplots(1, 1)
    fig1 = add_intensity_to_axes(fig1, sample_field1, ax1, extent)
    st.pyplot(fig1)
