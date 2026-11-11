import os
import json
from textwrap import dedent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class ModeloGemma:
    def __init__(self, especialidad: str = "Programación", model_name: str = "gemma", temperature: float = 0.1, archivo_json: str = "historial.json"):
        self.archivo_json = archivo_json
        self.especialidad = especialidad
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(f"""\
                Eres un tutor universitario especializado en {self.especialidad} que ayuda a un estudiante.

                Reglas estrictas de identidad y rol:
                - Tú eres el tutor (la IA). NUNCA digas que tu nombre es el del estudiante.
                - El usuario es el estudiante que te hace consultas.
                - Si el estudiante te pregunta "¿cómo me llamo?" o sobre su nombre, respóndele diciéndole SU nombre a él (por ejemplo: "Te llamas Misael" o "Tu nombre es Misael").
                - Si el estudiante aún no te ha dicho su nombre en la conversación actual, indícale amablemente que todavía no lo sabes.
                - Responde siempre en español con lenguaje sencillo y claro, orientando tus explicaciones al área de {self.especialidad}.
                - Explica los conceptos paso a paso e incluye ejemplos prácticos cuando convenga.
            """)),
            MessagesPlaceholder(variable_name="historial"),
            ("human", "{pregunta}")
        ])

        self.chain = self.prompt | self.llm
        self.historial = []
        
        # Cargar historial existente si el archivo existe
        self.cargar_historial()

    def guardar_historial(self):
        """Convierte los mensajes de LangChain a JSON y los guarda en disco."""
        datos = []
        for msg in self.historial:
            if isinstance(msg, HumanMessage):
                datos.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                datos.append({"role": "ai", "content": msg.content})
                
        with open(self.archivo_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=4)

    def cargar_historial(self):
        """Carga el historial desde el archivo JSON si existe."""
        if os.path.exists(self.archivo_json):
            try:
                if os.path.getsize(self.archivo_json) == 0:
                    self.historial = []
                    return

                with open(self.archivo_json, "r", encoding="utf-8") as f:
                    datos = json.load(f)
                    self.historial = []
                    for item in datos:
                        if item.get("role") == "human":
                            self.historial.append(HumanMessage(content=item["content"]))
                        elif item.get("role") == "ai":
                            self.historial.append(AIMessage(content=item["content"]))
            except Exception as e:
                print(f"[Aviso: No se pudo cargar el historial previo: {e}]")
                self.historial = []

    def reiniciar_historial(self):
        """Limpia el historial en memoria y borra el archivo JSON de disco."""
        self.historial = []
        if os.path.exists(self.archivo_json):
            try:
                os.remove(self.archivo_json)
            except Exception as e:
                print(f"[Aviso: No se pudo eliminar el archivo de historial: {e}]")

    def obtener_historial_formateado(self) -> str:
        """Devuelve el historial formateado para mostrar en pantalla."""
        if not self.historial:
            return "El historial está vacío."
        
        lineas = ["Historial de conversación"]
        for msg in self.historial:
            if isinstance(msg, HumanMessage):
                lineas.append(f"-->Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                lineas.append(f"--->IA: {msg.content}")
        return "\n".join(lineas)

    def generar_respuesta(self, pregunta: str) -> str:
        respuesta = self.chain.invoke({
            "historial": self.historial,
            "pregunta": pregunta
        })

        # Actualiza la memoria en tiempo de ejecución
        self.historial.append(HumanMessage(content=pregunta))
        self.historial.append(AIMessage(content=str(respuesta.content)))

        # Guarda la interacción en el archivo JSON
        self.guardar_historial()

        return str(respuesta.content)