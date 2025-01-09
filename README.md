# seguras
herramienta de machine learning que detecta el ciberacoso
!pip install gradio
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import gradio as gr #gradio para interfaz gráfico
#limpiezas

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
ls
#https://huggingface.co/datasets/somosnlp-hackathon-2022/Dataset-Acoso-Twitter-Es?library=pandas
dataset = pd.read_csv("hf://datasets/somosnlp-hackathon-2022/Dataset-Acoso-Twitter-Es/datasetfinal.csv")
dataset
dataset.describe()
#PROCESAMIENTO DE TEXTO
#usar solo dos columnas
dataset = dataset[['text', 'task1']] #variable independiente #variable
dataset
def clean_text(text):
  text= re.sub(r'http\S+|www.\S+', '', text) #URLs
  text = re.sub(r'@\w+', '', text) #elimina los usuarios de X
  text = re.sub(r'#\w+', '', text) #Elimina hashtags
  text = re.sub(r'[^\w\s]', '', text) #elimina puntuacion y numeros
  text = re.sub(r'\d+', '', text)
  text = text.lower() #convierte en minusculas
  stop_words = set(stopwords.words('spanish'))
  text = ' '.join([word for word in text.split() if word not in stop_words])
  return text
#limpio el dato/trato de limpiarlo pero no me deja #ACA ESTOY CON PROBLEMAS
dataset.loc[:, 'text_clean'] = dataset['text'].apply(clean_text)
#imprimirlas
dataset
#Convertir etiquetas categoricas
label_encoder= LabelEncoder()
dataset['task1']= label_encoder.fit_transform(dataset['task1'])
dataset
#Vectorizar
vectorizador = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizador.fit_transform(X_train)
X_test_vect = vectorizador.transform(X_test)
#voy a entrenar con logisti regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vect, y_train)

#predicciones
y_pred = model.predict(X_test_vect)

#evaluar rendimiento
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#Guardar
joblib.dump(model, 'cibeacoso_model.pkl') # Guardar el modelo para usarlo en la extensión
joblib.dump(vectorizador, 'cibeacoso_vectorizador.pkl') # Guardar el vectorizador para usarlo en la extensión
joblib.dump(label_encoder, 'cibeacoso_label_encoder.pkl') # Guardar el label encoder para usarlo en la extensión

#Cargar modelo para pruebas
model = joblib.load('cibeacoso_model.pkl')
vectorizador = joblib.load('cibeacoso_vectorizador.pkl')
label_encoder = joblib.load('cibeacoso_label_encoder.pkl')

#input
def predict_cibeacoso(text):
  try:
    text_clean = clean_text(text)
    text_vect = vectorizador.transform([text_clean])
    prediction = model.predict(text_vect)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return (
        "Al parecer el texto es ciberacoso, considera acudir a una autoridad o a una persona de confianza"
        if label == "acoso"
        else "El texto no parece ser ciberacoso, puedes sentirte SEGURA"
    )
  except Exception as e:
    return f"Error al analizar el texto: {str(e)}"
#interfaz gradio
interface = gr.Interface (
    fn = predict_cibeacoso,
    inputs = "text",
    outputs = "text",
    title = "Detección de Ciberacoso",
    description = "Ingresa un texto para analizar si es ciberacoso o no"
)
interface.launch(share=True)
