## INSTALACIÓN

1.- Clona el repositorio.

2.- Accede al directorio y ábrelo con VSCode.

3.- Crea un Entorno Virtual:
	python -m venv env
 
4.- Activa el entorno virtual env\Scripts\activate

5.- Instala las dependencias requeridas:
	pip install -r requirements.txt
 
7.- Crea una Clave de API de OpenAI y agrégala a tu archivo .txt y cambia su extensión a .env



-----Eliminar entorno virtual
# Desactivar el entorno virtual actual
deactivate

# Eliminar el entorno virtual (asegúrate de estar en el directorio correcto)
rmdir /s /q env

# Crear un nuevo entorno virtual
python -m venv env