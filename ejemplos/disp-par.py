from os import mkdir
from pathlib import Path
import sim
from sim import Experimento, Detector, ShieldCilindrico, BlancoCilindrico, Evento


from time import time
import numpy as np
import pickle


FLUJO = 1000
# parametros
sim.DEBUG = False
# Descomentar esta línea par forzar a que la fuente sólo
# emita a lo largo del eje z.
# sim.CONO_FUENTE = tuple((.95, 1.))
# Descomentar esta línea para no agregar el foton de 1200
# sim.HERALDO = False
# sim.EMITIR_EN_PLANO = True

DURACION = 240000000
RESET = True



cilindro1 = BlancoCilindrico(
    posicion=np.array([0, 0, 10.0]),
    alto=3,
    radio=2.54 / 2.0,
    lambda_absorcion=10000.0,  # Datos para el aluminio (https://en.wikipedia.org/wiki/Gamma_ray#/media/File:Al-gamma-xs.svg)
    lambda_dispersion=5,
)

cilindro2 = BlancoCilindrico(
    posicion=np.array([0, 0, -10.0]),
    alto=3,
    radio=2.54 / 2.0,
    lambda_absorcion=10000.0,
    lambda_dispersion=5,
)

mesa = BlancoCilindrico(
    posicion=np.array([0, 0, -19.0]),
    alto=10,
    radio=20,
    lambda_absorcion=10.0,
    lambda_dispersion=10.0,
    color=(.5,.1,.2)
)

shield1 = ShieldCilindrico(
    eje="z",
    lambda_absorcion=0.5,  # La longitud de penetracion para el plomo ~.5cm
    lambda_dispersion=0.5,  # La longitud de penetracion para el plomo ~.5cm
    radio_interior=1,
    radio_exterior=2,
    alto=10,
)


start = Detector(eficiencia=.9, posicion=np.array([12, 0, 10]), radio=2, retardo=0.5)
stop = Detector(eficiencia=.7, posicion=np.array([0, 12, -10]), radio=2, retardo=20.)

blancos = [shield1, cilindro1, cilindro2, mesa]  # [shield1, cilindro1, cilindro2]



if __name__ == "__main__":
    clean_start = RESET
    folder = f"results/{Path(__file__).name[:-3]}"
    print(folder)
    try:
        mkdir(folder)
    except:
        pass

    if not clean_start:
        try:
            with open(f"{folder}/checkpoint.pkl", "rb") as f_in:
                exp = pickle.load(f_in)
                print("cuentas almacenadas:", sum(exp.coincidencias))
        except Exception:
            print(
                f"{folder}/checkpoint.pkl no encontrado. Generando un nuevo experimento."
            )
            clean_start = True

    if clean_start:
        exp = Experimento(
            step=0.05,
            flujo_eventos=FLUJO,  # 3x10^-7 cuentas/ns en el experimento.
            start=start,
            stop=stop,
            blancos=blancos,
            folder=folder,
            canales = 256,
            checkpoint_interval=300.0,
        )

    t_start = time()
    exp.simular(DURACION)
    t_stop = time()
    exp.check_point(show_plots=True)
