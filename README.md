 ![plot](./yoga_demo.gif)
 
 PRIMEIRO É NECESSÁRIO TER O PYTHON INSTALADO
 
 Python >= 3.5
 
 Com o python instalado abra o visual studio code
 
 No TERMINAL DO VSCODE, execute as linhas abaixado, uma de cada vez:
 
     -Se não possuir o virtual lenv:
     
         python -m pip install --upgrade pip setuptools virtualenv
         
     python -m virtualenv kivy_venv
     
     kivy_venv\Scripts\activate
     
     python -m pip install kivy[full]
 
     git clone https://github.com/Miguel-Silva-Alves/OpenPoseKivy.git
     
     cd OpenPoseKivy
     
     pip install -r requirements.txt
     
     python webcam_pose_with_difficulty.py
