## Installation (Linux/macOs)

```bash
# 1. Clone the repository
git clone https://github.com/CristhianCruzDelgado/26-27-IntelligentSystems-GmailSpamClassifier.git
cd 26-27-IntelligentSystems-GmailSpamClassifier

# 2. Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# 3. Check the version (the project requires 3.9 or higher)
python3 --version

# 4. (Recommended) Create and activate a virtual environment
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate

# 5. Install the dependencies
pip install -r requirements.txt
```

## Development (draft)

Hemos consultado los principales proyectos de correos y hacen uso del conjunto de datos enron,
por eso lo hemos preprocesado. (`src/preprocess_enron.py`)

Luego, hemos obtenido otro conjunto de datos en español que es
el que nos interesa ya que nuestro objetivo es clasificar bandejas de entradas en español.
Necesitamos asignar las nuevas etiquetas del proyecto, y para no hacerlo a mano se lo pedimos
a un LLM. (`Gemini`)

### Prompt

> Vas a preprocesar una base de datos, y lo harás de 3 tandas de manera que en respuesta a este prompt analizaras 333 líneas. Después de haber asumido tus responsabilidades y una vez hayas generado el resultado para yo poder copiarlo te daré la señal para que continues con las 333 líneas siguientes, y después con las otras 333. Requisitos de binario a multiclase:<br>
\- Reetiquetado multiclase: Subdivisión de los correos legítimos en categorías, como por ejemplo: Finanzas, Trabajo, Soporte y Otros, manteniendo Spam.<br>
> [spanish_spam.csv]

\- Reetiquetado multiclase: Subdivisión de los correos legítimos en categorías, como por ejemplo: Finanzas, Trabajo, Soporte y Otros, manteniendo Spam.

\- Balanceo estratificado: Muestreo aleatorio y equilibrado de correos por cada una de las categorías. (`src/preprocess_spanish.py`)


