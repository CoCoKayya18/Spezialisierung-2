from setuptools import setup, find_packages

setup(
    name='ekf_slam_pkg',
    version='0.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='your_email@example.com',
    description='Description of your package',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = ekf_slam_pkg.main:main',
        ],
    },
)
