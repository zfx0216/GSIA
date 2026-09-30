# Environment requirements

The package snapshot below is the environment supplied with this project. It describes a Windows Conda environment with Python 3.11.16. The code primarily uses PyTorch, torchvision, NumPy, Pillow, scikit-image, Matplotlib, tqdm, and torchattacks. CUDA use depends on a compatible NVIDIA driver and the matching PyTorch build.

## Recreate the main Python dependencies

Create a Python 3.11 environment. Install a mutually compatible PyTorch, torchvision, and torchaudio build for CUDA 12.6 following the official PyTorch instructions, then install the remaining direct dependencies listed here. The package snapshot is the authoritative record of the user's environment; this short list is only a practical starting point.

| Package | Supplied version | Used for |
| --- | --- | --- |
| Python | 3.11.16 | Runtime |
| torch | 2.13.0+cu126 | Models, training, attacks |
| torchvision | 0.28.0+cu126 | Models, datasets, transforms |
| torchaudio | 2.11.0+cu126 | Present in the environment |
| numpy | 1.26.4 | Image and metric arrays |
| pillow | 11.1.0 | Image I/O |
| scikit-image | 0.26.0 | SSIM and PSNR |
| matplotlib | 3.11.0 | Training plots |
| tqdm | 4.70.0 | Progress display |
| torchattacks | 3.5.1 | Attack utilities |

The supplied torch, torchvision, and torchaudio versions should be installed as a compatible set from their actual distribution source. Confirm availability and compatibility before installing; the pasted package list alone does not verify that these exact wheels can be downloaded together.

## Full supplied package snapshot

This is a Conda-style package listing, including transitive Python packages and native libraries. Build strings and channels are retained as supplied. It is documentation rather than a directly installable `requirements.txt`.

```text
# Name                    Version                   Build  Channel
_openmp_mutex             2.0                     52_msvc
accelerate                1.14.0                   pypi_0    pypi
annotated-doc             0.0.5                    pypi_0    pypi
annotated-types           0.8.0                    pypi_0    pypi
anyio                     4.14.2                   pypi_0    pypi
blas                      1.0                         mkl
brotli                    1.2.0                    pypi_0    pypi
bzip2                     1.0.8                h2bbff1b_6
ca-certificates           2026.8.13            haa95532_0
cairo                     1.18.4               he9e932c_0
certifi                   2026.7.22                pypi_0    pypi
chardet                   4.0.0                    pypi_0    pypi
click                     8.5.0                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.3.3           py311h214f63a_0
cycler                    0.12.1          py311haa95532_0
decord                    0.6.0                    pypi_0    pypi
expat                     2.8.2                hd7fb8db_1
fastapi                   0.141.1                  pypi_0    pypi
filelock                  3.32.4                   pypi_0    pypi
fontconfig                2.15.0               hd211d86_0
fonttools                 4.63.0          py311h1c6eee0_0
freetype                  2.14.1               hfbffc0b_0
fsspec                    2026.7.0                 pypi_0    pypi
gradio                    6.27.0                   pypi_0    pypi
gradio-client             2.7.0                    pypi_0    pypi
graphite2                 1.3.15               h781baf4_0
groovy                    0.1.2                    pypi_0    pypi
h11                       0.16.0                   pypi_0    pypi
harfbuzz                  10.2.0               he2f9f60_1
hf-gradio                 0.4.1                    pypi_0    pypi
hf-xet                    1.6.0                    pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
huggingface-hub           1.29.0                   pypi_0    pypi
icu                       73.1                 h6c2663c_0
idna                      2.10                     pypi_0    pypi
imageio                   2.37.4                   pypi_0    pypi
intel-openmp              2025.0.0          haa95532_1165
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.5.3           py311haa95532_0
jpeg                      9f                   h89d5625_1
kiwisolver                1.5.0           py311hd7fb8db_0
lazy-loader               0.5                      pypi_0    pypi
lcms2                     2.16                 hb4a4139_0
lerc                      3.0                  hd77b12b_0
libdeflate                1.17                 h2bbff1b_1
libexpat                  2.8.2                hd7fb8db_1
libffi                    3.4.8                h42d73b9_3
libglib                   2.88.3               h71fae0a_0
libhwloc                  2.12.1          default_hfa10c62_1000
libiconv                  1.18                 hc89ec93_0
libkrb5                   1.22.2               h3d06f0e_0
libpng                    1.6.56               h2854ad3_0
libpq                     17.10                hb13705e_2
libtiff                   4.5.1                hd77b12b_0
libwebp-base              1.6.0                hbf3958f_0
libxml2                   2.13.9               h6201b9f_0
libzlib                   1.3.2                h1c6eee0_0
lz4-c                     1.9.4                h1c6eee0_5
markdown-it-py            4.2.0                    pypi_0    pypi
markupsafe                3.0.3                    pypi_0    pypi
matplotlib                3.11.0          py311haa95532_0
matplotlib-base           3.11.0          py311h330ff07_0
mdurl                     0.1.2                    pypi_0    pypi
mkl                       2025.0.0           h585ebfc_931
mkl-service               2.7.2           py311h1b35211_0
mkl_fft                   2.2.0           py311h18b77b3_0
mkl_random                1.4.1           py311hf6ff4c0_0
mpmath                    1.3.0                    pypi_0    pypi
mysql-common              9.3.0                h0b12ad4_6
mysql-libs                9.3.0                h08a8c37_6
narwhals                  2.23.0          py311haa95532_0
networkx                  3.6.1                    pypi_0    pypi
numpy                     1.26.4          py311h12f7302_1
numpy-base                1.26.4          py311he4e2855_1
openjpeg                  2.5.2                hae555c5_0
openssl                   3.5.7                hbb43b14_0
orjson                    3.12.0                   pypi_0    pypi
packaging                 26.3            py311haa95532_0
pandas                    3.0.5                    pypi_0    pypi
pcre2                     10.46                h5740b90_0
peft                      0.20.0                   pypi_0    pypi
pillow                    11.1.0          py311h096bfcc_0
pip                       26.2.1             pyhc872135_0
pixman                    0.46.4               h0701eb8_1
psutil                    7.2.2                    pypi_0    pypi
pydantic                  2.13.5                   pypi_0    pypi
pydantic-core             2.46.5                   pypi_0    pypi
pydub                     0.25.1                   pypi_0    pypi
pygments                  2.21.0                   pypi_0    pypi
pyparsing                 3.3.2           py311haa95532_0
pyqt                      6.9.1           py311h12ec796_0
pyqt6-sip                 13.10.2         py311h630b2a1_0
python                    3.11.16              hb00fc5c_0
python-dateutil           2.9.0post0      py311haa95532_2
python-multipart          0.0.32                   pypi_0    pypi
pytz                      2026.3.post1             pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
qtbase                    6.9.2                hd965823_2
qtdeclarative             6.9.2                h88b4c33_1
qtsvg                     6.9.2                h30ace32_1
qttools                   6.9.2                h7e7b719_1
qtwebchannel              6.9.2                heb02b0b_1
qtwebsockets              6.9.2                heb02b0b_1
regex                     2026.7.19                pypi_0    pypi
requests                  2.25.1                   pypi_0    pypi
rich                      15.0.0                   pypi_0    pypi
safehttpx                 0.1.7                    pypi_0    pypi
safetensors               0.8.0                    pypi_0    pypi
scikit-image              0.26.0                   pypi_0    pypi
scikit-learn              1.9.0           py311h6c0dd63_0
scipy                     1.17.1          py311h97a0a95_1
semantic-version          2.10.0                   pypi_0    pypi
setuptools                83.0.0          py311haa95532_0
shellingham               1.5.4                    pypi_0    pypi
sip                       6.12.0          py311h706e071_0
six                       1.17.0          py311haa95532_0
sqlite                    3.53.2               hee5a0db_0
starlette                 1.6.0                    pypi_0    pypi
sympy                     1.14.0                   pypi_0    pypi
tbb                       2022.3.0             h90c84d6_0
tbb-devel                 2022.3.0             h90c84d6_0
threadpoolctl             3.5.0           py311h4442805_1
tifffile                  2026.3.3                 pypi_0    pypi
tk                        8.6.15               hf199647_0
tokenizers                0.23.1                   pypi_0    pypi
tomlkit                   0.14.0                   pypi_0    pypi
torch                     2.13.0+cu126             pypi_0    pypi
torchattacks              3.5.1                    pypi_0    pypi
torchaudio                2.11.0+cu126             pypi_0    pypi
torchvision               0.28.0+cu126             pypi_0    pypi
tornado                   6.5.7           py311h1c6eee0_0
tqdm                      4.70.0                   pypi_0    pypi
transformers              5.16.1                   pypi_0    pypi
typer                     0.27.2                   pypi_0    pypi
typing-extensions         4.16.0                   pypi_0    pypi
typing-inspection         0.4.4                    pypi_0    pypi
tzdata                    2026.3                   pypi_0    pypi
ucrt                      10.0.22621.0         haa95532_0
urllib3                   1.26.20                  pypi_0    pypi
uvicorn                   0.52.4                   pypi_0    pypi
vc                        14.3                h2df5915_12
vc14_runtime              14.44.35208         h4927774_12
vcomp14                   14.44.35208         h4927774_12
vs2015_runtime            14.44.35208         ha6b5a95_12
wheel                     0.47.0          py311haa95532_0
xz                        5.8.2                h53af0af_0
zlib                      1.3.2                h1c6eee0_0
zstd                      1.5.7                h56299aa_0
```
