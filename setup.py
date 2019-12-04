from setuptools import setup, find_packages
from distilkobert import __version__


setup(name='distilkobert',
      version=__version__,
      url='https://github.com/monologg/DistilKoBERT',
      license='Apache-2.0',
      author='Jangwon Park',
      author_email='adieujw@gmail.com',
      description='Distillation of KoBERT',
      packages=find_packages(exclude=['distillation', 'docker', 'nsmc']),
      long_description=open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      python_requires='>=3',
      zip_safe=False,
      include_package_data=True,
      install_requires=[
            'torch==1.1.0',
            'sentencepiece>=0.1.82',
            'transformers==2.1.1',
            'gdown'
      ]
      )
