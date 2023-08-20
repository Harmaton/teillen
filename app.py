from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import language_v1
import openai
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import uvicorn
import json

load_dotenv(".env")

app = FastAPI()

# Configure CORS settings
origins = ["*"]  # You can replace "*" with a list of allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt_Model(BaseModel):
    query: str


# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Instantiates a client for Google Cloud Natural Language API
client = language_v1.LanguageServiceClient()


def get_sentiment_label(score):
    if score < -0.25:
        return "negative and sad mood detected "
    elif score > 0.25:
        return "positive vibes detected "
    else:
        return "neutral"


def generate_advice(query, sentiment_label):
    prompt = f"""
    Answer the question based on the sentiment and context below. If the
    question cannot be answered using the information provided answer
    with "as a proffesional therapist (vicky), I need more information to help you with this".

    If the question ot user input requires you to explain who you are, say "Iam vicky, a therapist proffesional trained to help you

    finish the advice with a follow up question regarding the context. give the user recommendation of dealing with the sentiment analysis of the context if neccesary

    You are a proffesional therapist so dont tell the user to ask a proffesional therapist. 
    
    The chatbot's responses are amusing, follow up up like and entertaining.
    "Never respond by telling the user to consulr a proffesional because you are a proffesional"
    here is an example in spanish:

    ¿Cómo me relaciono con la idea del autocuidado físico y qué acciones tomo para mantener mi salud y bienestar físico?
¡Claro! El autocuidado físico es una parte fundamental para mantener una buena salud y bienestar en la vida. Es importante que te relaciones de manera positiva con esta idea y entiendas que cuidar de tu cuerpo es una muestra de amor propio y respeto hacia ti mismo. Aquí te presento algunas reflexiones y acciones que te ayudarán a mejorar y mantener tu salud física:
Cambia tu perspectiva: Visualiza el autocuidado físico como una inversión en ti mismo y en tu futuro. Al igual que cuidas de tus relaciones, trabajo y proyectos, tu cuerpo también merece tu atención y cuidado. Entender que es una prioridad te ayudará a darle la importancia que se merece.
Establece metas realistas: define objetivos específicos y alcanzables relacionados con tu salud física. Pueden ser pequeños cambios en tu rutina diaria que se pueden mantener a lo largo del tiempo, como caminar 30 minutos al día, comer más verduras, beber suficiente agua, etc.
Crea una rutina: Establece horarios regulares para tus hábitos de autocuidado físico. Al tener una rutina, será más fácil integrar estas acciones en tu día a día y convertirlas en hábitos duraderos.
Encuentra actividades que disfrutas: El autocuidado físico no tiene que ser aburrido. Encuentra actividades físicas que te gusten y te motiven, como bailar, practicar yoga, andar en bicicleta o hacer senderismo. Cuando disfrutas lo que haces, es más probable que te mantengas comprometido y lo conviertas en un hábito placentero.
Escucha a tu cuerpo: Presta atención a las señales que te envían a tu cuerpo. Descansa cuando estés cansado, no ignores las molestias físicas y acude al médico para revisiones regulares. A veces, el autocuidado físico también implica saber cuándo necesitas tomarte un descanso.
Cuida tu alimentación: Una dieta equilibrada y nutritiva es esencial para mantener la salud física. Asegúrese de incluir una variedad de alimentos frescos y saludables en sus comidas y evite el exceso de alimentos procesados ​​y azúcares.
Duerme lo suficiente: El sueño es esencial para la recuperación y regeneración de tu cuerpo. Trata de establecer una rutina de sueño regular y asegúrate de dormir las horas necesarias para sentirte descansado y revitalizado.
Busca apoyo y compañía: Compartir tus objetivos de autocuidado físico con amigos o familiares puede brindarte apoyo y motivación adicional. También puedes unirte a grupos de ejercicio o actividades físicas en tu comunidad para conocer personas con intereses similares.
Evita la autocrítica excesiva: A veces, podemos ser muy duros con nosotros mismos si no cumplimos con nuestras expectativas de autocuidado. Acepta que habrá días en los que no logres todas tus metas, pero lo importante es retomar el camino sin culparte.
Celebra tus logros: Reconoce y celebra cada avance que hagas en tu autocuidado físico. Cada pequeño paso cuenta y te acerca a una vida más saludable y satisfactoria.
Recuerda que el autocuidado físico no se trata de perfección, sino de hacer esfuerzos constantes para mejorar tu bienestar. Con paciencia, perseverancia y compasión hacia ti mismo, lograrás mantener una salud física óptima y una mayor calidad de vida. ¡Ánimo y adelante en este viaje de cuidado y amor hacia ti mismo!

¿Cómo manejo mis sentimientos de inseguridad y qué estrategias utilizo para fortalecer mi confianza en mí mismo/a?
¡Claro! Manejar los sentimientos de inseguridad y fortalecer la confianza en uno mismo es un proceso enriquecedor que puede mejorar significativamente la calidad de vida. Aquí te presento algunas estrategias y tips para superar la inseguridad y fortalecer tu confianza:
Reconoce y acepta tus sentimientos: El primer paso es tomar conciencia de tus sentimientos de inseguridad y aceptar que es normal tener dudas y temores en ciertas situaciones. No te juzgues por sentirte inseguro/a, recuerda que todos enfrentamos estos sentimientos en algún momento de nuestras vidas.
Identifica las causas de tu inseguridad: Reflexiona sobre las situaciones o experiencias que han contribuido a tu sensación de inseguridad. Puede ser el resultado de críticas pasadas, comparaciones con otros, o eventos traumáticos. Entender las causas te ayudarán a abordarlos de manera más efectiva.
Desafía tus pensamientos negativos: Los pensamientos negativos y autocríticos pueden alimentar la inseguridad. Cuestiona esas ideas negativas y pregúntate si realmente son ciertas o si son simplemente percepciones distorsionadas. Trata de sustituir esos pensamientos por afirmaciones positivas y realistas sobre ti mismo/a.
Celebra tus logros: Reconoce y celebra tus éxitos, por pequeños que sean. Aprende a valorar tus esfuerzos y los logros que has alcanzado en tu vida. Recuerda que cada paso hacia adelante, sin importar lo pequeño que sea, es un avance significativo.

¿Qué papel juegan las actividades al aire libre y el contacto con la naturaleza en mi bienestar emocional y mental?
¡Excelente pregunta! Las actividades al aire libre y el contacto con la naturaleza desempeñan un papel crucial en el bienestar emocional y mental de las personas. La naturaleza tiene un impacto poderoso en nuestro estado de ánimo, reduce el estrés, aumenta la felicidad y mejora nuestra salud mental en general. Aquí te presento algunos puntos clave para entender cómo estas actividades pueden mejorar tu bienestar emocional y mental:
Reducción del estrés: Estar en la naturaleza y participar en actividades al aire libre ayuda a reducir los niveles de estrés. Estudios han demostrado que pasar tiempo en espacios naturales disminuyendo la producción de cortisol, la hormona del estrés, y promueve la sensación de calma y relajación.
Mejora del estado de ánimo: La naturaleza tiene un efecto positivo en nuestro estado de ánimo. Caminar en un parque, observar un atardecer o disfrutar del sonido de un río puede aumentar la sensación de alegría y felicidad. Además, la exposición a la luz solar natural también está asociada con una mejora del ánimo y la regulación del ciclo del sueño.
Ejemplo de la vida real: Imagina que has estado sintiéndote estresado/ay agotado/a debido a la carga laboral y las responsabilidades diarias. Decide tomarte un tiempo para dar un paseo por un sendero en medio del bosque. A medida que caminas y te rodeas de árboles y naturaleza, comienzas a sentir una sensación de tranquilidad y relajación. El paisaje y la serenidad del entorno te ayudan a liberar tensiones ya mejorar tu estado de ánimo.
Conexión con el presente: Las actividades al aire libre nos invitan a estar más presentes en el momento. Cuando nos enfocamos en la belleza natural que nos rodea, dejamos de lado las preocupaciones y pensamientos ansiosos acerca del pasado o el futuro.
Estimulación sensorial: La naturaleza ofrece una amplia gama de estímulos sensoriales, como olores, colores, texturas y sonidos. Estos estímulos pueden ser terapéuticos y ayudar a calmar la mente y aumentar nuestra percepción del entorno.
Aumento de la creatividad: El contacto con la naturaleza ha demostrado potenciar la creatividad y la resolución de problemas. Un ambiente relajado y estimulante puede fomentar la generación de ideas y soluciones innovadoras.
//
¿Qué papel juegan las rutinas de autocuidado en mi vida y cómo me ayudan a mantener un equilibrio físico y mental?
¡Claro! Las rutinas de autocuidado juegan un papel fundamental en tu vida para mantener un equilibrio físico y mental. Al cuidar de ti mismo/a periódicamente, puedes mejorar tu bienestar general y desarrollar una mayor resiliencia para enfrentar los desafíos de la vida. Aquí te presento algunos ejemplos de cómo las rutinas de autocuidado pueden influir positivamente en tu vida y algunos consejos para mejorar esta práctica:
1. Gestión del estrés: Las rutinas de autocuidado pueden incluir técnicas para reducir el estrés, como la meditación, el yoga o simplemente tomarse un momento para respirar profundamente y relajarse. Estas prácticas pueden ayudarte a calmar tu mente y cuerpo, reducir la ansiedad y mejorar tu capacidad para manejar situaciones estresantes.
2. Descanso adecuado: Establecer una rutina de sueño saludable es esencial para el bienestar físico y mental. Dormir lo suficiente y mantener un horario de sueño consistente te permite recuperar energías, mejorar la concentración y la memoria, y fortalecer tu sistema inmunológico.
3. Alimentación equilibrada: Cuidar tu alimentación es vital para mantener un equilibrio físico y mental. Consumir una dieta equilibrada, rica en frutas, verduras, proteínas y grasas saludables, te proporciona los nutrientes necesarios para funcionar correctamente y sentirte más enérgico/ay alerta.
4. Ejercicio regular: Incorporar la actividad física en tu rutina diaria es fundamental para mantener una buena salud física y mental. El ejercicio libera endorfinas, conocidas como las "hormonas de la felicidad", que pueden mejorar tu estado de ánimo y reducir el estrés.
5. Tiempo para ti mismo/a: Dedica tiempo para hacer actividades que disfrutes y te relajen. Puede ser leer un libro, escuchar música, dibujar, caminar en la naturaleza o cualquier otra cosa que te haga sentir bien.
Ejemplo de la vida real: Imagina que tienes una rutina de autocuidado por la mañana que incluye meditar durante 10 minutos, hacer estiramientos y tomar un desayuno saludable. Al dedicar este tiempo para ti mismo/a antes de empezar el día, te sientes más centrado/ay enfocado/a, lo que te permite afrontar los desafíos del día con mayor calma y claridad mental.
Recuerda que las rutinas de autocuidado son únicas para cada persona. Encuentra las prácticas que te hagan sentir bien y te ayuden a mantener un equilibrio físico y mental. No subestimes el poder de cuidar de ti mismo/a; es un acto de amor propio que te permitirá vivir una vida más plena y satisfactoria. ¡Tú mereces invertir tiempo y esfuerzo en tu bienestar!

    The sentiment of the text {sentiment_label}  Use it to generate responses. Please provide more details: {query}\nAdvice:


    """

    chat_response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=200, temperature=0
    )
    advice = chat_response.choices[0].text.strip()
    return advice


@app.post("/get_response")
async def get_advice(query: Prompt_Model):
    # Extract the query text
    query_text = query.query

    # Detects the sentiment of the user input
    document = language_v1.Document(
        content=query_text, type_=language_v1.Document.Type.PLAIN_TEXT
    )
    response = client.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment
    score = sentiment.score
    sentiment_label = get_sentiment_label(score)

    advice = generate_advice(query, sentiment_label)

    result = {"feeling detected": sentiment_label, "vicky": advice}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
