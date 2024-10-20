call D:/anaconda3/Scripts/activate.bat flow

cd networks\correlation_package
rmdir /S /Q *_cuda.egg-info build dist __pycache__
python setup.py install --user

cd ..\resample2d_package
rmdir /S /Q *_cuda.egg-info build dist __pycache__
python setup.py install --user

cd ..\channelnorm_package
rmdir /S /Q *_cuda.egg-info build dist __pycache__
python setup.py install --user

cd ..

cmd