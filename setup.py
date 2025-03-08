from distutils.core import setup

setup(
    name='evc',
    packages=['evc'],  # this must be the same as the name above
    version='0.1',
    description='evc',
    author='kd',
    author_email='',
    keywords=['add', 'sub', 'tests'],
    classifiers=[],
    package_data={'static': ['index.html']},
    install_requires=['python-socketio', 'pandas', 'numpy', 'tornado', 'torch', 'torchaudio'],
)
