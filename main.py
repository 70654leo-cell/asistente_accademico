import sys
from modelo_gemma import ModeloGemma

def seleccionar_especialidad() -> str:
    especialidades = {
        "1": "Programación",
        "2": "Inteligencia Artificial",
        "3": "Bases de Datos"
    }
    
    print("------------------------------------------")
    print("Seleccione la especialidad:")
    print("1. Programación")
    print("2. Inteligencia Artificial")
    print("3. Bases de Datos")
    
    while True:
        opcion = input("Opción: ").strip()
        if opcion in especialidades:
            especialidad = especialidades[opcion]
            print(f"Tutor de {especialidad} iniciado.\n")
            return especialidad
        print("Opción inválida. Intente nuevamente (1, 2 o 3).")

def main():
    print("==========================================")
    print("      ASISTENTE ACADÉMICO INTELIGENTE     ")
    print("==========================================")
    
    especialidad = seleccionar_especialidad()

    try:
        asistente = ModeloGemma(especialidad=especialidad)
    except Exception as e:
        print(f"Error al inicializar el modelo: {e}")
        sys.exit(1)

    while True:
        try:
            pregunta = input("Usuario: ").strip()

            if pregunta.lower() in ["salir", "exit", "quit"]:
                print("Programa finalizado.")
                break

            if not pregunta:
                continue

            if pregunta.lower() == "historial":
                print(asistente.obtener_historial_formateado())
                print()
                continue

            if pregunta.lower() == "nuevo":
                asistente.reiniciar_historial()
                print("Nueva conversación iniciada. Historial anterior reemplazado.\n")
                continue

            # Llamada al método corregido
            respuesta = asistente.generar_respuesta(pregunta)
            print(f"--->IA: {respuesta}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nPrograma finalizado.")
            break
        except Exception as e:
            print(f"\n[Error al procesar la respuesta: {e}]\n")

if __name__ == "__main__":
    main()