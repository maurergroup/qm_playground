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
      name = 'qm_playground',
      version = '0.1',
      author = 'Reinhard Maurer',
      author_email = 'reinhard.maurer@yale.edu',
      description = 'playground code for solving simple 1D/2D QM problems',
      #long_description = 'python package with numerically intesive subroutines in F90',
      license = 'GNU GPL',
      platforms = 'x86_64',
      install_requires: ['nose'],
      packages = ['model'],
      }

setup(**config)

