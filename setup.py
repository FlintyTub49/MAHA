#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from distutils.core import setup
setup(
  name = 'MAHA',
  packages = ['MAHA'],
  version = '0.6',
  license='MIT',
  description = 'Performing ETL using Machine Learning',
  author = 'Mithesh R, Arth Akhouri, Heetansh Jhaveri, Ayaan Khan',
  author_email = 'arthakhouri@gmail.com',
  url = 'https://github.com/user/FlintyTub49',
  download_url = 'https://github.com/FlintyTub49/MAHA/archive/v_0.5.tar.gz',
  keywords = ['ETL', 'Machine Learning', 'Regression', 'Pandas', 'Numpy'],
  install_requires=[
          'numpy',
          'sklearn',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

