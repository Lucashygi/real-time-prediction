import os
import pathlib
#\models\research>SET PYTHONPATH=%cd%:%cd%\slim - caso nao importe object_detection
# Clone o repositório de modelos de tensorflow se ele ainda não existir
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  print('--------> [INFO] Iniciando clonagem!!')
  os.system('git clone --depth 1 https://github.com/jeffreire/models.git')

print('----> [INFO] CLONAGEM FINALIZADA!')

print('--------> Intalação do pacote Visual Studio Build C++')

os.system('start tools/vs_BuildTools.exe')

print('----> [INFO] INSTALACAO FINALIZADA!')


os.system('pip install --upgrade tensorflow==2.5')
os.system('pip install --upgrade keras==2.4.3')
os.system('pip install --upgrade keras-nightly==2.5.0.dev2021032900')
os.system('pip install tensorflow_model_optimization')
os.system('pip install apache-beam')