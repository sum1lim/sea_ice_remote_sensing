from setuptools import setup

__version__ = (0, 0, 0)

setup(
    name="sea_ice_rs",
    description="Sea Ice Remote Sensing",
    version=".".join(str(d) for d in __version__),
    author="Sangwon Lim",
    author_email="sangwonl@uvic.ca",
    packages=["sea_ice_rs"],
    include_package_data=True,
    scripts="""
        ./scripts/SOBEL
        ./scripts/threshold
        ./scripts/dist-stat
        ./scripts/extract-colour
        ./scripts/connect-lines
        ./scripts/centroids
        ./scripts/create-datasets
        ./scripts/neural-network
        ./scripts/normalize
        ./scripts/GLCM
        ./scripts/CNN
        ./scripts/test-model
    """.split(),
)
