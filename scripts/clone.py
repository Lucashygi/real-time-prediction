import os
import pathlib
#\models\research>SET PYTHONPATH=%cd%;%cd%\slim - caso nao importe object_detection
# Clone o repositório de modelos de tensorflow se ele ainda não existir
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  os.system('git clone --depth 1 https://github.com/jeffreire/models.git')

# os.system('protoc models/research/object_detection/protos/*.proto --python_out=.')
# os.system('python models/research/object_detection/packages/tf2/setup.py install')
print('[INFO] SUCCESS')

# # python pack_install.py