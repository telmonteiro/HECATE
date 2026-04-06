from setuptools import setup, find_packages

setup(name = 'HECATE',
    version = "0.2.0",
    description = 'HarvEsting loCAl specTra with Exoplanets',
    url = 'https://github.com/telmonteiro/HECATE/',
    license = 'MIT',
    author = 'Telmo Monteiro',
    author_email = 'telmo.monteiro@astro.up.pt',
    keywords = ['astronomy'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv'],},
    install_requires = ['numpy', 'dynesty', 'matplotlib', 'scipy']
)
