import setuptools

setuptools.setup(
    name="pyhod",
    version="1.0.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['numpy', 'scipy', 
                      'scikit-learn', 'numba',               
                      ]
)