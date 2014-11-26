#

try:
    from setuptools import setup
except:
    from distutils.core import setup

#ext_modules = [
    #Extension('internal', [ 'interna/internal.f90'],swig_opts=['-ipo -O3 -prec-div -axW -static']#,
##        libraries = ['imf', 'svml', 'ifcore'],
        #)
    #]

config = {
    'description': 'playground code for solving simple 1D/2D QM problems',
    'author': 'Reinhard J. Maurer',
    'url': 'URL',
    'download_url': 'download it.',
    'author_email': 'reinhard.maurer@yale.edu',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['qmp'],
    'name': 'qm_playground'
}

setup(**config)

