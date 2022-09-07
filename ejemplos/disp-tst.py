from os import mkdir
from pathlib import Path
import sim
from sim import Experimento, Detector, ShieldCilindrico, BlancoCilindrico, Evento


from time import time
import numpy as np
import pickle


FLUJO = 50
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
    lambda_absorcion=100000.0,
    lambda_dispersion=0.003,
)

cilindro2 = BlancoCilindrico(
    posicion=np.array([0, 0, -10.0]),
    alto=3,
    radio=2.54 / 2.0,
    lambda_absorcion=100000.0,
    lambda_dispersion=0.003,
)

shield1 = ShieldCilindrico(
    eje="z",
    lambda_absorcion=0.5,  # La longitud de penetracion para el plomo ~.1cm
    lambda_dispersion=0.5,  # La longitud de penetracion para el plomo ~.1cm
    radio_interior=1,
    radio_exterior=2,
    alto=10,
)


start = Detector(eficiencia=.6, posicion=np.array([12, 0, 10]), radio=2, retardo=3.)
stop = Detector(eficiencia=.4, posicion=np.array([-12, 0, -10]), radio=2, retardo=28.)
blancos = []  # [shield1, cilindro1, cilindro2]


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
            canales = 2048,
            checkpoint_interval=100.0,
        )

    t_start = time()
    exp.simular(DURACION)
    t_stop = time()
    exp.check_point(show_plots=True)
