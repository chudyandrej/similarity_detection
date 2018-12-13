from setuptools import setup, find_packages

setup(name='seq2seq',
      version='0.1',
      packages=find_packages(),
      description='Column embedding model for data column similarity',
      author='Andrej Chudy',
      author_email='achudy03@gmail.com',
      license='MIT',
      include_package_data=True,
      install_requires=[
          'keras',
          'unidecode',
          'h5py'
      ],
      zip_safe=False)
