# 01 - python -m venv .venv
# 02 - pip3 freeze > requirements.txt
# 03 - .venv\scripts\activate
# 04 - pip install -r requirements.txt --upgrade #Updating by pip install
# ou - instalar pacote por pacote utilizado
# 05 - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# item 5 só precisa se bloquear a ativação do ambiente virtual
# 06 - python setup.py build
# 07 - .venv\scripts\deactivate
# caso tenha versão nova, executar linhas: 3 > 5 > 6 > 7

from cx_Freeze import setup, Executable

#base = 'Win32GUI'
executables = [Executable(
    'main.py', target_name='Simul_MonteCarlo.exe')]

# incluir os pacotes usados
# packages = ['numpy', 'xlwings', 'pandas', 'random', 'shutil', 'glob', 'sys', 'ctypes']
packages = ['numpy', 'xlwings', 'pandas']

# incluir as pastas do projeto
includefiles = ['output_files/', 'InputFile.xlsx']

options = {
    'build_exe': {
    'packages': packages,
    'include_files': includefiles
    },
}

setup(
    name='Prod Simulacao Monte Carlo',
    options=options,
    version='1',
    description='',
    executables=executables
)